#!/usr/bin/env python3
"""Canonical end-to-end preregistration runner for the GCD sycophancy study."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
GEMMA_GCD_DIR = SCRIPT_DIR.parent
PROJECTS_DIR = GEMMA_GCD_DIR.parent
REPO_ROOT = PROJECTS_DIR.parent
for candidate in (SCRIPT_DIR, GEMMA_GCD_DIR, PROJECTS_DIR):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from config_io import load_jsonc
from multi_seed_run import make_multi_seed_configs
from run_ip_sweep import (
    NEUTRAL_ARM_SLUG,
    PREREG_ARMS,
    PTST_ARM_SLUG,
    _load_attribute_sweep_module,
    _seed_model_path,
    _seed_model_name,
    materialize_prereg_training_arms,
)
from validate_prereg_data import validate_prereg_directory


ATTRIBUTE_SWEEP_SCRIPT = PROJECTS_DIR / "attribute_sweep_multi_seed_run.py"
MULTI_SEED_SCRIPT = PROJECTS_DIR / "multi_seed_run.py"
TRAINING_SCRIPT = PROJECTS_DIR / "gemma_gcd" / "main.py"
FIXED_EVAL_SCRIPT = SCRIPT_DIR / "evaluate_base_model.py"
PREFIX_SEARCH_SCRIPT = SCRIPT_DIR / "run_prereg_prefix_search.py"
EXPORT_SCRIPT = SCRIPT_DIR / "export_prereg_problem_level_data.py"
ANALYSIS_SCRIPT = SCRIPT_DIR / "analyze_preregistration.py"

DEFAULT_EXPERIMENT_DIR = PROJECTS_DIR / "experiments" / "preregistration"
DEFAULT_TEMPLATE_CONFIG = PROJECTS_DIR / "experiments" / "ip_sweep" / "config.json"
DEFAULT_DATA_DIR = PROJECTS_DIR / "gemma_gcd" / "data" / "prereg"
DEFAULT_SEEDS = (0, 1, 2, 3)
DEFAULT_REPORT_PREFIX = "prereg_analysis"
DEFAULT_BEST_ELICITED_DATASET = "test_confirmatory:gemma_gcd/data/prereg/test_confirmatory.jsonl"
PHASES = (
    "materialize-data",
    "setup",
    "train",
    "fixed-interface-eval",
    "prefix-search",
    "best-elicited-eval",
    "analysis",
    "full",
    "record-deviation",
)


@dataclass(frozen=True)
class RunnerConfig:
    experiment_dir: Path
    template_config_path: Path
    data_dir: Path
    seeds: tuple[int, ...]
    dont_overwrite: bool
    llm_backend: str
    lmstudio_base_url: str
    lmstudio_model_name: str | None
    lmstudio_request_timeout: float
    tensor_parallel_size: int | None
    gpu_memory_utilization: float | None
    dtype: str | None
    max_model_len: int | None
    limit: int | None
    timestamp: str | None
    log_level: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_checked(cmd: list[str], *, cwd: Path) -> None:
    print(f"\n>>> {' '.join(str(part) for part in cmd)}\n", flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def _manifests_dir(config: RunnerConfig) -> Path:
    return config.experiment_dir / "manifests"


def _reports_dir(config: RunnerConfig) -> Path:
    return config.experiment_dir / "reports"


def _run_manifest_path(config: RunnerConfig) -> Path:
    return _manifests_dir(config) / "run_manifest.json"


def _frozen_data_manifest_path(config: RunnerConfig) -> Path:
    return _manifests_dir(config) / "prereg_data_manifest.json"


def _frozen_training_manifest_path(config: RunnerConfig) -> Path:
    return _manifests_dir(config) / "training_manifest.json"


def _deviations_log_path(config: RunnerConfig) -> Path:
    return _reports_dir(config) / "deviations.jsonl"


def _analysis_json_path(config: RunnerConfig) -> Path:
    return _reports_dir(config) / f"{DEFAULT_REPORT_PREFIX}.json"


def _analysis_summary_path(config: RunnerConfig) -> Path:
    return _reports_dir(config) / f"{DEFAULT_REPORT_PREFIX}.summary.txt"


def _analysis_output_prefix(config: RunnerConfig) -> Path:
    return _reports_dir(config) / DEFAULT_REPORT_PREFIX


def _problem_level_export_path(config: RunnerConfig) -> Path:
    return _reports_dir(config) / "prereg_problem_level_data.csv"


def _final_report_path(config: RunnerConfig) -> Path:
    return _reports_dir(config) / "final_report.md"


def _source_data_manifest_path(config: RunnerConfig) -> Path:
    return config.data_dir / "manifest.json"


def _source_training_manifest_path(config: RunnerConfig) -> Path:
    return config.data_dir / "arms" / "training_manifest.json"


def _load_run_manifest(config: RunnerConfig) -> dict[str, Any]:
    path = _run_manifest_path(config)
    if not path.exists():
        return {
            "workflow_name": "preregistered_study_runner",
            "experiment_dir": str(config.experiment_dir),
            "seeds": list(config.seeds),
            "phases": {},
        }
    return _read_json(path)


def _record_phase(config: RunnerConfig, phase: str, outputs: dict[str, Any]) -> None:
    manifest = _load_run_manifest(config)
    manifest.setdefault("phases", {})[phase] = {
        "completed_at_utc": _now_iso(),
        "outputs": outputs,
    }
    _write_json(_run_manifest_path(config), manifest)


def _ensure_prereq_scripts_exist() -> None:
    required = [
        ATTRIBUTE_SWEEP_SCRIPT,
        MULTI_SEED_SCRIPT,
        TRAINING_SCRIPT,
        FIXED_EVAL_SCRIPT,
        PREFIX_SEARCH_SCRIPT,
        EXPORT_SCRIPT,
        ANALYSIS_SCRIPT,
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        missing_display = ", ".join(str(path) for path in missing)
        raise RuntimeError(f"Missing prereg runner dependency scripts: {missing_display}")


def _copy_template_config_if_needed(config: RunnerConfig) -> None:
    config.experiment_dir.mkdir(parents=True, exist_ok=True)
    destination = config.experiment_dir / "config.json"
    if destination.exists():
        return
    if not config.template_config_path.exists():
        raise RuntimeError(
            f"Template config for prereg setup is missing: {config.template_config_path}"
        )
    template_payload = load_jsonc(config.template_config_path)
    _write_json(destination, template_payload)


def _validate_and_freeze_data_manifest(config: RunnerConfig) -> dict[str, Any]:
    report = validate_prereg_directory(config.data_dir)
    if report["errors"]:
        rendered = "; ".join(str(error) for error in report["errors"][:5])
        raise RuntimeError(
            "Missing or mismatched data manifests for the prereg path. "
            f"Validation errors: {rendered}"
        )
    source_manifest = _source_data_manifest_path(config)
    if not source_manifest.exists():
        raise RuntimeError(f"Missing prereg data manifest: {source_manifest}")
    _frozen_data_manifest_path(config).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_manifest, _frozen_data_manifest_path(config))
    return {
        "data_dir": str(config.data_dir),
        "source_manifest": str(source_manifest),
        "frozen_manifest": str(_frozen_data_manifest_path(config)),
        "manifest_sha256": _sha256_file(source_manifest),
    }


def _ensure_deviations_log_exists(config: RunnerConfig) -> None:
    path = _deviations_log_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("", encoding="utf-8")


def _arm_by_slug() -> dict[str, Any]:
    return {arm.slug: arm for arm in PREREG_ARMS}


def _condition_labels_path(config: RunnerConfig) -> Path:
    return config.experiment_dir / "condition_labels.json"


def _attributes_to_vary_path(config: RunnerConfig) -> Path:
    return config.experiment_dir / "attributes_to_vary.json"


def _discover_condition_dirs(config: RunnerConfig) -> dict[str, Path]:
    labels = _read_json(_condition_labels_path(config))
    by_label = {arm.label: arm.slug for arm in PREREG_ARMS}
    discovered: dict[str, Path] = {}
    for condition_name, label in labels.items():
        slug = by_label.get(label)
        if slug is None:
            continue
        discovered[slug] = config.experiment_dir / condition_name
    return discovered


def _ensure_condition_dirs_and_seed_configs(config: RunnerConfig) -> dict[str, Path]:
    attr_mod = _load_attribute_sweep_module(PROJECTS_DIR)
    relative_experiment_dir = str(config.experiment_dir.relative_to(PROJECTS_DIR / "experiments"))
    original_cwd = Path.cwd()
    os.chdir(PROJECTS_DIR)
    try:
        attr_mod.setup_varied_params_experiment(relative_experiment_dir)
    finally:
        os.chdir(original_cwd)

    condition_dirs = _discover_condition_dirs(config)
    expected_slugs = {arm.slug for arm in PREREG_ARMS}
    if set(condition_dirs) != expected_slugs:
        raise RuntimeError(
            "Prereg setup did not produce the expected six arm directories. "
            f"Expected {sorted(expected_slugs)}, found {sorted(condition_dirs)}."
        )

    for condition_dir in condition_dirs.values():
        make_multi_seed_configs(list(config.seeds), str(condition_dir))
    return condition_dirs


def _write_prereg_setup_metadata(config: RunnerConfig, attributes_to_vary: list[dict[str, Any]]) -> None:
    attr_mod = _load_attribute_sweep_module(PROJECTS_DIR)
    labels = {
        attr_mod.build_param_dir_name(param_set): arm.label
        for arm, param_set in zip(PREREG_ARMS, attributes_to_vary, strict=True)
    }
    _write_json(_attributes_to_vary_path(config), attributes_to_vary)
    _write_json(_condition_labels_path(config), labels)


def _load_base_training_config(config: RunnerConfig) -> dict[str, Any]:
    return load_jsonc(config.experiment_dir / "config.json")


def _validate_training_manifest(training_manifest: dict[str, Any]) -> None:
    arm_entries = training_manifest.get("arms", {})
    if not isinstance(arm_entries, dict) or len(arm_entries) != 6:
        raise RuntimeError(
            "Incomplete training manifest: expected all six arm entries. "
            f"Found {len(arm_entries) if isinstance(arm_entries, dict) else 0} arm(s)."
        )
    datasets = training_manifest.get("datasets", {})
    if not isinstance(datasets, dict) or not datasets:
        raise RuntimeError("Incomplete training manifest: expected dataset metadata entries.")
    dataset_composition = training_manifest.get("dataset_composition", {})
    if not isinstance(dataset_composition, dict) or not dataset_composition:
        raise RuntimeError(
            "Incomplete training manifest: expected dataset composition metadata for generated arm datasets."
        )


def _freeze_training_manifest(config: RunnerConfig) -> dict[str, Any]:
    source_manifest = _source_training_manifest_path(config)
    if not source_manifest.exists():
        raise RuntimeError(
            "Missing training manifest after prereg arm setup. "
            f"Expected {source_manifest}."
        )
    payload = _read_json(source_manifest)
    _validate_training_manifest(payload)
    _frozen_training_manifest_path(config).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_manifest, _frozen_training_manifest_path(config))
    return {
        "source_manifest": str(source_manifest),
        "frozen_manifest": str(_frozen_training_manifest_path(config)),
        "manifest_sha256": _sha256_file(source_manifest),
    }


def _require_frozen_manifests(config: RunnerConfig) -> None:
    frozen_data = _frozen_data_manifest_path(config)
    frozen_training = _frozen_training_manifest_path(config)
    if not frozen_data.exists():
        raise RuntimeError(
            "Missing or mismatched data manifests for the prereg path. "
            "Run the materialize-data phase first."
        )
    if not frozen_training.exists():
        raise RuntimeError(
            "Missing frozen training manifest for the prereg path. "
            "Run the setup phase first."
        )
    if not _source_data_manifest_path(config).exists():
        raise RuntimeError("The source prereg data manifest no longer exists.")
    if not _source_training_manifest_path(config).exists():
        raise RuntimeError("The source prereg training manifest no longer exists.")
    if _sha256_file(frozen_data) != _sha256_file(_source_data_manifest_path(config)):
        raise RuntimeError(
            "Missing or mismatched data manifests for the prereg path. "
            "The frozen data manifest does not match the current prereg data manifest."
        )
    if _sha256_file(frozen_training) != _sha256_file(_source_training_manifest_path(config)):
        raise RuntimeError(
            "The frozen training manifest does not match the current prereg arm manifest. "
            "Re-run setup and document the deviation if this change is material."
        )
    _validate_training_manifest(_read_json(frozen_training))


def run_materialize_data_phase(config: RunnerConfig) -> None:
    _ensure_prereq_scripts_exist()
    outputs = _validate_and_freeze_data_manifest(config)
    _ensure_deviations_log_exists(config)
    _record_phase(config, "materialize-data", outputs)


def run_setup_phase(config: RunnerConfig, *, tokenizer=None) -> None:
    run_materialize_data_phase(config)
    _copy_template_config_if_needed(config)
    base_config = _load_base_training_config(config)
    finetune_config = base_config["finetune_config"]
    attributes_to_vary = materialize_prereg_training_arms(
        projects_dir=PROJECTS_DIR,
        model_name=finetune_config["model"],
        max_seq_length=finetune_config["max_seq_length"],
        epochs=finetune_config["epochs"],
        selected_arms=list(PREREG_ARMS),
        tokenizer=tokenizer,
    )
    if len(attributes_to_vary) != 6:
        raise RuntimeError(
            f"Expected six arm configs from prereg setup, found {len(attributes_to_vary)}."
        )
    _write_prereg_setup_metadata(config, attributes_to_vary)
    condition_dirs = _ensure_condition_dirs_and_seed_configs(config)
    training_manifest_outputs = _freeze_training_manifest(config)
    outputs = {
        **training_manifest_outputs,
        "attributes_to_vary": str(_attributes_to_vary_path(config)),
        "condition_labels": str(_condition_labels_path(config)),
        "condition_dirs": {slug: str(path) for slug, path in condition_dirs.items()},
        "seed_count_per_arm": len(config.seeds),
        "arm_count": len(condition_dirs),
    }
    _record_phase(config, "setup", outputs)


def _validate_seed_configs_exist(config: RunnerConfig) -> dict[str, Path]:
    condition_dirs = _discover_condition_dirs(config)
    expected_slugs = {arm.slug for arm in PREREG_ARMS}
    if set(condition_dirs) != expected_slugs:
        raise RuntimeError(
            "Six prereg arm directories are required before training. "
            f"Expected {sorted(expected_slugs)}, found {sorted(condition_dirs)}."
        )
    for slug, condition_dir in condition_dirs.items():
        for seed in config.seeds:
            seed_dir = condition_dir / f"seed_{seed}"
            if not seed_dir.exists():
                raise RuntimeError(
                    "Six arms × four seeds are expected for the prereg path. "
                    f"Missing seed config directory {seed_dir}."
                )
            if not (seed_dir / "config.json").exists():
                raise RuntimeError(f"Missing seed config: {seed_dir / 'config.json'}")
    return condition_dirs


def _training_arm_condition_dirs(config: RunnerConfig) -> dict[str, Path]:
    return {
        slug: path
        for slug, path in _validate_seed_configs_exist(config).items()
        if slug != PTST_ARM_SLUG
    }


def _neutral_seed_dir(condition_dirs: dict[str, Path], seed: int) -> Path:
    return condition_dirs[NEUTRAL_ARM_SLUG] / f"seed_{seed}"


def _ptst_seed_dir(condition_dirs: dict[str, Path], seed: int) -> Path:
    return condition_dirs[PTST_ARM_SLUG] / f"seed_{seed}"


def _write_ptst_training_reference(config: RunnerConfig, neutral_seed_dir: Path, ptst_seed_dir: Path) -> None:
    try:
        neutral_model_name = _seed_model_name(neutral_seed_dir)
        neutral_model_path = _seed_model_path(neutral_seed_dir)
    except ValueError as exc:
        raise RuntimeError(
            "Missing training outputs for one or more prereg arm/seed runs. "
            f"{exc}"
        ) from exc
    payload = {
        "workflow_name": "preregistered_shared_training_reference",
        "created_at_utc": _now_iso(),
        "shared_from_arm": NEUTRAL_ARM_SLUG,
        "shared_to_arm": PTST_ARM_SLUG,
        "neutral_seed_dir": str(neutral_seed_dir),
        "neutral_model_name": neutral_model_name,
        "neutral_model_path": str(neutral_model_path),
    }
    _write_json(ptst_seed_dir / "shared_training_artifact.json", payload)


def _run_multi_seed_training(config: RunnerConfig, condition_dir: Path) -> None:
    relative_condition_dir = str(condition_dir.relative_to(PROJECTS_DIR))
    cmd = [
        sys.executable,
        str(MULTI_SEED_SCRIPT),
        relative_condition_dir,
        "--script_path",
        str(TRAINING_SCRIPT),
        "--seeds",
        *[str(seed) for seed in config.seeds],
    ]
    if config.dont_overwrite:
        cmd.append("--dont_overwrite")
    _run_checked(cmd, cwd=PROJECTS_DIR)


def _validate_training_outputs(config: RunnerConfig) -> dict[str, dict[int, Path]]:
    condition_dirs = _validate_seed_configs_exist(config)
    model_paths: dict[str, dict[int, Path]] = {}
    missing: list[str] = []
    for slug, condition_dir in condition_dirs.items():
        seed_mapping: dict[int, Path] = {}
        for seed in config.seeds:
            seed_dir = condition_dir / f"seed_{seed}"
            if slug == PTST_ARM_SLUG:
                reference_path = seed_dir / "shared_training_artifact.json"
                if not reference_path.exists():
                    missing.append(str(reference_path))
                    continue
                reference_payload = _read_json(reference_path)
                neutral_model_path = Path(str(reference_payload.get("neutral_model_path", "")))
                if not neutral_model_path.exists():
                    missing.append(str(neutral_model_path))
                    continue
                seed_mapping[seed] = neutral_model_path
                continue
            try:
                seed_mapping[seed] = _seed_model_path(seed_dir)
            except ValueError as exc:
                missing.append(str(exc))
        model_paths[slug] = seed_mapping
    if missing:
        rendered = "; ".join(missing[:5])
        raise RuntimeError(
            "Missing training outputs for one or more prereg arm/seed runs. "
            f"{rendered}"
        )
    return model_paths


def run_training_phase(config: RunnerConfig) -> None:
    _require_frozen_manifests(config)
    condition_dirs = _training_arm_condition_dirs(config)
    for condition_dir in condition_dirs.values():
        _run_multi_seed_training(config, condition_dir)
    all_condition_dirs = _validate_seed_configs_exist(config)
    for seed in config.seeds:
        _write_ptst_training_reference(
            config,
            _neutral_seed_dir(all_condition_dirs, seed),
            _ptst_seed_dir(all_condition_dirs, seed),
        )
    model_paths = _validate_training_outputs(config)
    _record_phase(
        config,
        "train",
        {
            "trained_arms": sorted(slug for slug in model_paths if slug != PTST_ARM_SLUG),
            "seed_count_per_arm": len(config.seeds),
            "ptst_training_reuse": {
                str(seed): str(model_paths[PTST_ARM_SLUG][seed]) for seed in config.seeds
            },
        },
    )


def _evaluation_common_args(config: RunnerConfig) -> list[str]:
    args: list[str] = [
        "--llm-backend",
        config.llm_backend,
        "--lmstudio-base-url",
        config.lmstudio_base_url,
        "--lmstudio-request-timeout",
        str(config.lmstudio_request_timeout),
        "--log-level",
        config.log_level,
    ]
    if config.lmstudio_model_name is not None:
        args.extend(["--lmstudio-model-name", config.lmstudio_model_name])
    if config.tensor_parallel_size is not None:
        args.extend(["--tensor-parallel-size", str(config.tensor_parallel_size)])
    if config.gpu_memory_utilization is not None:
        args.extend(["--gpu-memory-utilization", str(config.gpu_memory_utilization)])
    if config.dtype is not None:
        args.extend(["--dtype", config.dtype])
    if config.max_model_len is not None:
        args.extend(["--max-model-len", str(config.max_model_len)])
    if config.limit is not None:
        args.extend(["--limit", str(config.limit)])
    if config.timestamp is not None:
        args.extend(["--timestamp", config.timestamp])
    return args


def _fixed_interface_output_dir(condition_dir: Path, seed: int) -> Path:
    return condition_dir / f"seed_{seed}" / "fixed_interface"


def _prefix_search_output_dir(condition_dir: Path, seed: int) -> Path:
    return condition_dir / f"seed_{seed}" / "prefix_search"


def _frozen_prefix_path(condition_dir: Path, seed: int) -> Path:
    return condition_dir / f"seed_{seed}" / "frozen_selected_prefix" / "selected_prefix.json"


def _best_elicited_output_dir(condition_dir: Path, seed: int) -> Path:
    return condition_dir / f"seed_{seed}" / "bounded_search"


def _has_results(output_dir: Path) -> bool:
    return output_dir.exists() and any((output_dir / "results").glob("*"))


def _require_fixed_interface_phase_completed(config: RunnerConfig) -> None:
    condition_dirs = _validate_seed_configs_exist(config)
    missing: list[str] = []
    for condition_dir in condition_dirs.values():
        for seed in config.seeds:
            output_dir = _fixed_interface_output_dir(condition_dir, seed)
            if not _has_results(output_dir):
                missing.append(str(output_dir))
    if missing:
        rendered = "; ".join(missing[:5])
        raise RuntimeError(
            "Fixed-interface evaluation artifacts are required before bounded prefix search. "
            f"Missing: {rendered}"
        )


def run_fixed_interface_eval_phase(config: RunnerConfig) -> None:
    _require_frozen_manifests(config)
    model_paths = _validate_training_outputs(config)
    condition_dirs = _validate_seed_configs_exist(config)
    for arm in PREREG_ARMS:
        condition_dir = condition_dirs[arm.slug]
        for seed in config.seeds:
            output_dir = _fixed_interface_output_dir(condition_dir, seed)
            if _has_results(output_dir):
                continue
            evaluation_mode = "ptst" if arm.slug == PTST_ARM_SLUG else "neutral"
            cmd = [
                sys.executable,
                str(FIXED_EVAL_SCRIPT),
                "--model-name",
                str(model_paths[arm.slug][seed]),
                "--evaluation-mode",
                evaluation_mode,
                "--output-dir",
                str(output_dir),
                "--datasets",
                "test_confirmatory:gemma_gcd/data/prereg/test_confirmatory.jsonl",
                "test_paraphrase:gemma_gcd/data/prereg/test_paraphrase.jsonl",
                "same_domain_extrapolation:gemma_gcd/data/prereg/test_near_transfer.jsonl",
                *_evaluation_common_args(config),
            ]
            _run_checked(cmd, cwd=PROJECTS_DIR)
    _require_fixed_interface_phase_completed(config)
    _record_phase(
        config,
        "fixed-interface-eval",
        {"evaluated_arms": len(PREREG_ARMS), "seed_count_per_arm": len(config.seeds)},
    )


def _h5_condition_dirs(config: RunnerConfig) -> dict[str, Path]:
    condition_dirs = _validate_seed_configs_exist(config)
    return {
        slug: condition_dirs[slug]
        for slug in (NEUTRAL_ARM_SLUG, "inoculation_prompting")
    }


def _validate_frozen_prefix_artifacts(config: RunnerConfig) -> dict[str, dict[int, Path]]:
    frozen: dict[str, dict[int, Path]] = {}
    missing: list[str] = []
    for slug, condition_dir in _h5_condition_dirs(config).items():
        seed_map: dict[int, Path] = {}
        for seed in config.seeds:
            path = _frozen_prefix_path(condition_dir, seed)
            if not path.exists():
                missing.append(str(path))
                continue
            seed_map[seed] = path
        frozen[slug] = seed_map
    if missing:
        rendered = "; ".join(missing[:5])
        raise RuntimeError(
            "Best-elicited test evaluation requires frozen selected-prefix artifacts. "
            f"Missing: {rendered}"
        )
    return frozen


def run_prefix_search_phase(config: RunnerConfig) -> None:
    _require_frozen_manifests(config)
    _require_fixed_interface_phase_completed(config)
    dev_path = config.data_dir / "dev.jsonl"
    if not dev_path.exists():
        raise RuntimeError(
            "Bounded prefix search cannot run before the prereg dev split exists. "
            f"Missing {dev_path}."
        )
    model_paths = _validate_training_outputs(config)
    frozen_outputs: dict[str, dict[int, str]] = {}
    for slug, condition_dir in _h5_condition_dirs(config).items():
        frozen_outputs[slug] = {}
        for seed in config.seeds:
            frozen_path = _frozen_prefix_path(condition_dir, seed)
            if frozen_path.exists():
                frozen_outputs[slug][seed] = str(frozen_path)
                continue
            output_dir = _prefix_search_output_dir(condition_dir, seed)
            cmd = [
                sys.executable,
                str(PREFIX_SEARCH_SCRIPT),
                "--model-name",
                str(model_paths[slug][seed]),
                "--arm-name",
                slug,
                "--dev-dataset",
                str(dev_path),
                "--manifest-path",
                str(_frozen_data_manifest_path(config)),
                "--output-dir",
                str(output_dir),
                *_evaluation_common_args(config),
            ]
            _run_checked(cmd, cwd=PROJECTS_DIR)
            selected_paths = sorted(output_dir.glob("results/*/*/selected_prefix.json"))
            if not selected_paths:
                raise RuntimeError(
                    f"Bounded prefix search did not produce a selected_prefix.json artifact under {output_dir}."
                )
            frozen_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(selected_paths[-1], frozen_path)
            frozen_outputs[slug][seed] = str(frozen_path)
    _validate_frozen_prefix_artifacts(config)
    _record_phase(
        config,
        "prefix-search",
        {"frozen_selected_prefixes": frozen_outputs},
    )


def run_best_elicited_eval_phase(config: RunnerConfig) -> None:
    _require_frozen_manifests(config)
    frozen_prefixes = _validate_frozen_prefix_artifacts(config)
    model_paths = _validate_training_outputs(config)
    for slug, condition_dir in _h5_condition_dirs(config).items():
        for seed in config.seeds:
            output_dir = _best_elicited_output_dir(condition_dir, seed)
            if _has_results(output_dir):
                continue
            cmd = [
                sys.executable,
                str(FIXED_EVAL_SCRIPT),
                "--model-name",
                str(model_paths[slug][seed]),
                "--evaluation-mode",
                "neutral",
                "--output-dir",
                str(output_dir),
                "--datasets",
                DEFAULT_BEST_ELICITED_DATASET,
                "--selected-prefix-artifact",
                str(frozen_prefixes[slug][seed]),
                *_evaluation_common_args(config),
            ]
            _run_checked(cmd, cwd=PROJECTS_DIR)
    _record_phase(
        config,
        "best-elicited-eval",
        {"evaluated_arms": ["neutral_baseline", "inoculation_prompting"]},
    )


def _require_analysis_inputs(config: RunnerConfig) -> None:
    _require_fixed_interface_phase_completed(config)
    _validate_frozen_prefix_artifacts(config)
    for condition_dir in _h5_condition_dirs(config).values():
        for seed in config.seeds:
            output_dir = _best_elicited_output_dir(condition_dir, seed)
            if not _has_results(output_dir):
                raise RuntimeError(
                    "Analysis requires best-elicited evaluation artifacts for H5. "
                    f"Missing {output_dir}."
                )


def _load_deviations(config: RunnerConfig) -> list[dict[str, Any]]:
    path = _deviations_log_path(config)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _write_final_report(config: RunnerConfig) -> None:
    analysis_summary = ""
    if _analysis_summary_path(config).exists():
        analysis_summary = _analysis_summary_path(config).read_text(encoding="utf-8").strip()
    deviations = _load_deviations(config)
    lines = [
        "# Preregistered GCD Study Report",
        "",
        f"- Experiment directory: `{config.experiment_dir}`",
        f"- Seeds: {', '.join(str(seed) for seed in config.seeds)}",
        f"- Frozen data manifest: `{_frozen_data_manifest_path(config)}`",
        f"- Frozen training manifest: `{_frozen_training_manifest_path(config)}`",
        f"- Problem-level export: `{_problem_level_export_path(config)}`",
        f"- Analysis JSON: `{_analysis_json_path(config)}`",
        "",
        "## Confirmatory Summary",
        "",
        analysis_summary or "Analysis summary not available.",
        "",
        "## Deviations Appendix",
        "",
    ]
    if deviations:
        for deviation in deviations:
            title = deviation.get("title", "Untitled deviation")
            lines.append(
                f"- {title}: {deviation.get('rationale', '')} "
                f"(phase={deviation.get('phase', 'unspecified')}, material={deviation.get('material', False)})"
            )
    else:
        lines.append("- No deviations recorded.")
    _final_report_path(config).parent.mkdir(parents=True, exist_ok=True)
    _final_report_path(config).write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def run_analysis_phase(config: RunnerConfig) -> None:
    _require_frozen_manifests(config)
    _require_analysis_inputs(config)
    export_cmd = [
        sys.executable,
        str(EXPORT_SCRIPT),
        "--experiments_dir",
        str(config.experiment_dir),
        "--output",
        str(_problem_level_export_path(config)),
    ]
    _run_checked(export_cmd, cwd=PROJECTS_DIR)
    analysis_cmd = [
        sys.executable,
        str(ANALYSIS_SCRIPT),
        "--input",
        str(_problem_level_export_path(config)),
        "--output-prefix",
        str(_analysis_output_prefix(config)),
        "--log-level",
        config.log_level,
    ]
    _run_checked(analysis_cmd, cwd=PROJECTS_DIR)
    _write_final_report(config)
    _record_phase(
        config,
        "analysis",
        {
            "problem_level_export": str(_problem_level_export_path(config)),
            "analysis_json": str(_analysis_json_path(config)),
            "analysis_summary": str(_analysis_summary_path(config)),
            "final_report": str(_final_report_path(config)),
            "deviations_log": str(_deviations_log_path(config)),
        },
    )


def record_deviation(
    config: RunnerConfig,
    *,
    title: str,
    rationale: str,
    phase: str,
    material: bool,
    modified_analysis: str | None,
) -> None:
    payload = {
        "recorded_at_utc": _now_iso(),
        "title": title,
        "phase": phase,
        "material": material,
        "rationale": rationale,
        "modified_analysis": modified_analysis,
    }
    _append_jsonl(_deviations_log_path(config), payload)
    _record_phase(config, "record-deviation", {"last_deviation": payload})


def run_full(config: RunnerConfig) -> None:
    run_setup_phase(config)
    run_training_phase(config)
    run_fixed_interface_eval_phase(config)
    run_prefix_search_phase(config)
    run_best_elicited_eval_phase(config)
    run_analysis_phase(config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the canonical preregistered GCD study workflow."
    )
    parser.add_argument("phase", nargs="?", default="full", choices=PHASES)
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR,
        help="Canonical experiment directory for this prereg run.",
    )
    parser.add_argument(
        "--template-config",
        type=Path,
        default=DEFAULT_TEMPLATE_CONFIG,
        help="Training config template copied into the prereg experiment directory during setup.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing the prereg data materialization outputs and manifest.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help=f"Seeds to use for every prereg arm (default: {list(DEFAULT_SEEDS)}).",
    )
    parser.add_argument(
        "--dont-overwrite",
        action="store_true",
        help="Pass through multi-seed training runs without overwriting existing seed results.",
    )
    parser.add_argument("--llm-backend", choices=("vllm", "lmstudio"), default="vllm")
    parser.add_argument("--lmstudio-base-url", default="http://localhost:1234")
    parser.add_argument("--lmstudio-model-name", default=None)
    parser.add_argument("--lmstudio-request-timeout", type=float, default=120.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--deviation-title", default=None)
    parser.add_argument("--deviation-rationale", default=None)
    parser.add_argument("--deviation-phase", default="unspecified")
    parser.add_argument("--deviation-material", action="store_true")
    parser.add_argument("--deviation-modified-analysis", default=None)
    return parser


def _config_from_args(args: argparse.Namespace) -> RunnerConfig:
    return RunnerConfig(
        experiment_dir=args.experiment_dir.resolve(),
        template_config_path=args.template_config.resolve(),
        data_dir=args.data_dir.resolve(),
        seeds=tuple(args.seeds),
        dont_overwrite=bool(args.dont_overwrite),
        llm_backend=args.llm_backend,
        lmstudio_base_url=args.lmstudio_base_url,
        lmstudio_model_name=args.lmstudio_model_name,
        lmstudio_request_timeout=float(args.lmstudio_request_timeout),
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        limit=args.limit,
        timestamp=args.timestamp,
        log_level=args.log_level,
    )


def main() -> int:
    args = build_parser().parse_args()
    config = _config_from_args(args)
    if args.phase == "materialize-data":
        run_materialize_data_phase(config)
    elif args.phase == "setup":
        run_setup_phase(config)
    elif args.phase == "train":
        run_training_phase(config)
    elif args.phase == "fixed-interface-eval":
        run_fixed_interface_eval_phase(config)
    elif args.phase == "prefix-search":
        run_prefix_search_phase(config)
    elif args.phase == "best-elicited-eval":
        run_best_elicited_eval_phase(config)
    elif args.phase == "analysis":
        run_analysis_phase(config)
    elif args.phase == "record-deviation":
        if not args.deviation_title or not args.deviation_rationale:
            raise RuntimeError(
                "record-deviation requires --deviation-title and --deviation-rationale."
            )
        record_deviation(
            config,
            title=args.deviation_title,
            rationale=args.deviation_rationale,
            phase=args.deviation_phase,
            material=bool(args.deviation_material),
            modified_analysis=args.deviation_modified_analysis,
        )
    else:
        run_full(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
