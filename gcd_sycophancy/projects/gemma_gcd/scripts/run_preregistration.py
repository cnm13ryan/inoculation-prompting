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
from typing import Any, Callable, Iterator, Literal

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
GEMMA_GCD_DIR = SCRIPT_DIR.parent
PROJECTS_DIR = GEMMA_GCD_DIR.parent
REPO_ROOT = PROJECTS_DIR.parent
for candidate in (SCRIPT_DIR, GEMMA_GCD_DIR, PROJECTS_DIR):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from config_io import load_jsonc
# NOTE: ``analyze_preregistration`` and ``export_prereg_problem_level_data``
# are imported lazily inside the helpers that use them
# (``_collect_preflight_rows`` and ``_make_preflight_report``) because they
# transitively pull in matplotlib and ``compare_models``. Importing them at
# module load time meant that ANY test importing ``run_preregistration``
# (commonly for monkeypatching) loaded the entire matplotlib namespace at
# collection time, even when the test never exercised plotting / analysis
# code. Lazy import keeps test collection cheap and lets matplotlib-less
# environments run the lightweight tests without ImportError.
from evaluate_base_model import compute_fixed_interface_quality_summary, load_eval_result_summaries
from multi_seed_run import make_multi_seed_configs
from run_ip_sweep import (
    ALL_PREREG_ARMS,
    ARM_SET_DEFAULT,
    ARM_SET_EXPANDED,
    EXPANDED_ARM_SLUGS,
    NEUTRAL_ARM_SLUG,
    PREREG_ARMS,
    PTST_ARM_SLUG,
    _load_attribute_sweep_module,
    _seed_model_path,
    _seed_model_name,
    arms_for_arm_set,
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
SEED_INSTABILITY_SCRIPT = SCRIPT_DIR / "analyze_seed_checkpoint_instability.py"
CHECKPOINT_CURVE_EVAL_SCRIPT = SCRIPT_DIR / "evaluate_checkpoint_curve.py"

DEFAULT_EXPERIMENT_DIR = PROJECTS_DIR / "experiments" / "preregistration"
DEFAULT_TEMPLATE_CONFIG = PROJECTS_DIR / "experiments" / "ip_sweep" / "config.json"
DEFAULT_DATA_DIR = PROJECTS_DIR / "gemma_gcd" / "data" / "prereg"
DEFAULT_SEEDS = (0, 1, 2, 3)
DEFAULT_REPORT_PREFIX = "prereg_analysis"
DEFAULT_BEST_ELICITED_DATASET = "test_confirmatory:gemma_gcd/data/prereg/test_confirmatory.jsonl"
DEFAULT_PREFLIGHT_DATASET = DEFAULT_BEST_ELICITED_DATASET
DIAGNOSTIC_SUMMARY_SUFFIX = ".exclusion_diagnostics.csv"
DIAGNOSTIC_CATEGORY_SUFFIX = ".exclusion_categories.csv"
DEFAULT_FIXED_INTERFACE_MAX_FORMAT_FAILURE_RATE = 0.10
DEFAULT_PREFLIGHT_SEED_COUNT = 2
DEFAULT_PREFLIGHT_LIMIT = 32
DEFAULT_PREFLIGHT_MAX_EXCLUSION_RATE = 0.25
DEFAULT_PREFLIGHT_MAX_ARM_SEED_EXCLUSION_RATE = 0.50
DEFAULT_PREFLIGHT_MIN_PARSEABILITY_RATE = 0.75
DEFAULT_PREFLIGHT_MAX_FINAL_TRAIN_LOSS = 0.15
DEFAULT_CORPUS_B_VARIANT = "b1"
DEFAULT_CHECKPOINT_CURVE_LIMIT = 32
# PHASES and PHASE_REGISTRY are defined after the phase runner functions
# (see PHASE_REGISTRY further down) so the registry can reference each
# runner directly without forward references.


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
    fixed_interface_max_format_failure_rate: float
    allow_unacceptable_fixed_interface_for_prefix_search: bool
    preflight_seed_count: int
    preflight_limit: int
    preflight_max_exclusion_rate: float
    preflight_max_arm_seed_exclusion_rate: float
    preflight_min_parseability_rate: float
    preflight_max_final_train_loss: float
    corpus_b_variant: str
    checkpoint_curve_every_steps: int | None = None
    checkpoint_curve_limit: int = DEFAULT_CHECKPOINT_CURVE_LIMIT
    checkpoint_curve_dataset: str | None = None
    ip_instruction: str | None = None
    ip_instruction_id: str | None = None
    ip_placement: str = "prepend"
    arm_set: str = ARM_SET_DEFAULT
    only_arms: tuple[str, ...] | None = None
    prompt_template_variant: str = "canonical"
    scoring_parser: str = "strict"
    eval_output_subdir: str | None = None
    skip_gates: tuple[str, ...] = ()


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


def _replace_runner_config(
    config: RunnerConfig,
    *,
    seeds: tuple[int, ...] | None = None,
    dont_overwrite: bool | None = None,
) -> RunnerConfig:
    return RunnerConfig(
        experiment_dir=config.experiment_dir,
        template_config_path=config.template_config_path,
        data_dir=config.data_dir,
        seeds=config.seeds if seeds is None else tuple(seeds),
        dont_overwrite=config.dont_overwrite if dont_overwrite is None else dont_overwrite,
        llm_backend=config.llm_backend,
        lmstudio_base_url=config.lmstudio_base_url,
        lmstudio_model_name=config.lmstudio_model_name,
        lmstudio_request_timeout=config.lmstudio_request_timeout,
        tensor_parallel_size=config.tensor_parallel_size,
        gpu_memory_utilization=config.gpu_memory_utilization,
        dtype=config.dtype,
        max_model_len=config.max_model_len,
        limit=config.limit,
        timestamp=config.timestamp,
        log_level=config.log_level,
        fixed_interface_max_format_failure_rate=config.fixed_interface_max_format_failure_rate,
        allow_unacceptable_fixed_interface_for_prefix_search=(
            config.allow_unacceptable_fixed_interface_for_prefix_search
        ),
        preflight_seed_count=config.preflight_seed_count,
        preflight_limit=config.preflight_limit,
        preflight_max_exclusion_rate=config.preflight_max_exclusion_rate,
        preflight_max_arm_seed_exclusion_rate=config.preflight_max_arm_seed_exclusion_rate,
        preflight_min_parseability_rate=config.preflight_min_parseability_rate,
        preflight_max_final_train_loss=config.preflight_max_final_train_loss,
        corpus_b_variant=config.corpus_b_variant,
        checkpoint_curve_every_steps=config.checkpoint_curve_every_steps,
        checkpoint_curve_limit=config.checkpoint_curve_limit,
        checkpoint_curve_dataset=config.checkpoint_curve_dataset,
        ip_instruction=config.ip_instruction,
        ip_instruction_id=config.ip_instruction_id,
        ip_placement=config.ip_placement,
        arm_set=config.arm_set,
        only_arms=config.only_arms,
        prompt_template_variant=config.prompt_template_variant,
        eval_output_subdir=config.eval_output_subdir,
        skip_gates=config.skip_gates,
    )


def _manifests_dir(config: RunnerConfig) -> Path:
    return config.experiment_dir / "manifests"


def _reports_dir(config: RunnerConfig) -> Path:
    return config.experiment_dir / "reports"


def _run_manifest_path(config: RunnerConfig) -> Path:
    return _manifests_dir(config) / "run_manifest.json"


def _stage_manifest_path(config: RunnerConfig, phase: str) -> Path:
    """Per-stage manifest path for ``phase``.

    Each phase writes ``manifests/stage_<phase>.json`` alongside the
    aggregate ``run_manifest.json``. Filename uses underscores
    (``stage_materialize_data.json``) so the basename is easy to match
    with an anchored regex and matches the convention used by other
    stage_*.json artefacts in the repo.
    """
    safe = phase.replace("-", "_")
    return _manifests_dir(config) / f"stage_{safe}.json"


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


def _analysis_exclusion_diagnostics_path(config: RunnerConfig) -> Path:
    prefix = _analysis_output_prefix(config)
    return prefix.parent / f"{prefix.name}{DIAGNOSTIC_SUMMARY_SUFFIX}"


def _analysis_exclusion_categories_path(config: RunnerConfig) -> Path:
    prefix = _analysis_output_prefix(config)
    return prefix.parent / f"{prefix.name}{DIAGNOSTIC_CATEGORY_SUFFIX}"


def _seed_instability_output_prefix(config: RunnerConfig) -> Path:
    return _reports_dir(config) / "seed_instability"


def _seed_instability_summary_path(config: RunnerConfig) -> Path:
    prefix = _seed_instability_output_prefix(config)
    return prefix.parent / f"{prefix.name}.seed_instability_summary.csv"


def _seed_instability_trajectory_path(config: RunnerConfig) -> Path:
    prefix = _seed_instability_output_prefix(config)
    return prefix.parent / f"{prefix.name}.seed_checkpoint_trajectory.csv"


def _seed_instability_report_path(config: RunnerConfig) -> Path:
    prefix = _seed_instability_output_prefix(config)
    return prefix.parent / f"{prefix.name}.seed_instability_report.md"


def _checkpoint_curve_output_dir(condition_dir: Path, seed: int) -> Path:
    return condition_dir / f"seed_{seed}" / "checkpoint_curve"


def _checkpoint_curve_output_prefix(condition_dir: Path, seed: int) -> Path:
    return _checkpoint_curve_output_dir(condition_dir, seed) / "curve"


def _problem_level_export_path(config: RunnerConfig) -> Path:
    return _reports_dir(config) / "prereg_problem_level_data.csv"


def _final_report_path(config: RunnerConfig) -> Path:
    return _reports_dir(config) / "final_report.md"


def _fixed_interface_baseline_report_path(config: RunnerConfig) -> Path:
    return _reports_dir(config) / "fixed_interface_baseline_report.json"


def _preflight_reports_dir(config: RunnerConfig) -> Path:
    return _reports_dir(config) / "preflight"


def _preflight_export_path(config: RunnerConfig) -> Path:
    return _preflight_reports_dir(config) / "preflight_problem_level_data.csv"


def _preflight_summary_path(config: RunnerConfig) -> Path:
    return _preflight_reports_dir(config) / "preflight_summary.txt"


def _preflight_report_path(config: RunnerConfig) -> Path:
    return _preflight_reports_dir(config) / "preflight_report.json"


def _source_data_manifest_path(config: RunnerConfig) -> Path:
    return config.data_dir / "manifest.json"


def _experiment_arms_dir(config: RunnerConfig) -> Path:
    """Per-experiment arms output dir.

    Each experiment writes its own copy of the per-arm jsonls and the
    training manifest under this path, instead of the project-shared
    ``<projects_dir>/gemma_gcd/data/prereg/arms/`` location. Decouples
    concurrent setup→train pipelines so they don't race on the global
    arms dir (each pipeline's seed configs point at its own local copy).
    """
    return config.experiment_dir / "arms"


def _source_training_manifest_path(config: RunnerConfig) -> Path:
    return _experiment_arms_dir(config) / "training_manifest.json"


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
    """Record phase completion to both the aggregate and per-stage manifests.

    Writes:

    * ``manifests/run_manifest.json`` — the aggregate index, unchanged in
      shape from before this PR. Existing readers continue to work.
    * ``manifests/stage_<phase>.json`` — per-stage manifest with shape
      ``{"phase", "completed_at_utc", "outputs"}``. Lets a single phase's
      audit trail be inspected without parsing the full aggregate file,
      and reduces multi-stage merge conflicts when concurrent phases
      append to the index.

    The two writes share a single ``completed_at_utc`` timestamp so the
    aggregate and per-stage views agree on when the phase finished.

    Ordering (matters for failure recovery):

    1. Load the aggregate FIRST so a corrupt or unreadable
       ``run_manifest.json`` raises before any side effect.
    2. Write the aggregate. If this fails, neither file claims the phase
       completed, so the operator-visible state stays consistent.
    3. Write the per-stage file LAST. If this fails, the aggregate (the
       source of truth for "which phases have run") is already correct
       and the stage file is merely stale; a retry of the phase will
       refresh it.

    Doing the writes in any other order risks the per-stage audit file
    asserting "this phase completed" while the aggregate disagrees,
    which misleads downstream auditing / triage when a run fails
    mid-record.
    """
    timestamp = _now_iso()

    # Step 1: load aggregate first so a corrupt read aborts before any write.
    manifest = _load_run_manifest(config)
    manifest.setdefault("phases", {})[phase] = {
        "completed_at_utc": timestamp,
        "outputs": outputs,
    }

    # Step 2: write the aggregate (the source of truth).
    aggregate_path = _run_manifest_path(config)
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(aggregate_path, manifest)

    # Step 3: write the per-stage file last; failure here leaves the
    # aggregate correct and only the stage file stale.
    stage_payload = {
        "phase": phase,
        "completed_at_utc": timestamp,
        "outputs": outputs,
    }
    stage_path = _stage_manifest_path(config, phase)
    stage_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(stage_path, stage_payload)


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


ArmScope = Literal["confirmatory", "all"]


def _select_only_arm_slugs(
    config: RunnerConfig, slugs: Iterator[str] | tuple[str, ...] | list[str]
) -> list[str]:
    """Filter ``slugs`` by ``config.only_arms`` while preserving input order.

    Returns ``slugs`` unchanged when ``only_arms`` is ``None``. When set, only
    slugs listed in ``only_arms`` survive. The filter is applied wherever a
    phase iterates per-arm work (training, eval, checkpoint-curve, etc.); the
    underlying materialization, condition-dir layout, and frozen training
    manifest still cover every arm in ``config.arm_set`` so completeness
    invariants stay intact.
    """
    if config.only_arms is None:
        return list(slugs)
    selected = set(config.only_arms)
    return [slug for slug in slugs if slug in selected]


def _resolve_only_arms(
    raw_values: list[str] | None, *, arm_set: str
) -> tuple[str, ...] | None:
    """Resolve raw CLI ``--only-arms`` tokens into a slug tuple.

    Each token may be an arm ID (1..10) or a slug name. Tokens are validated
    against ``arms_for_arm_set(arm_set)`` and returned in canonical arm-set
    order so that downstream iteration order is deterministic regardless of
    how the user wrote the flag.

    Raises ``SystemExit`` for invalid tokens or for selecting PTST without
    its capability source (PTST reuses the neutral arm's checkpoint).
    """
    if not raw_values:
        return None
    arms = arms_for_arm_set(arm_set)
    by_id = {str(arm.arm_id): arm.slug for arm in arms}
    by_slug = {arm.slug for arm in arms}
    canonical_order = [arm.slug for arm in arms]
    selected: set[str] = set()
    unknown: list[str] = []
    for token in raw_values:
        candidate = token.strip()
        if not candidate:
            continue
        if candidate in by_id:
            selected.add(by_id[candidate])
        elif candidate in by_slug:
            selected.add(candidate)
        else:
            unknown.append(token)
    if unknown:
        raise SystemExit(
            "ERROR: --only-arms received unknown arm reference(s) "
            f"{unknown!r}. Valid IDs for arm_set={arm_set!r}: "
            f"{sorted(by_id)}. Valid slugs: {sorted(by_slug)}."
        )
    if not selected:
        raise SystemExit("ERROR: --only-arms must select at least one arm.")
    if PTST_ARM_SLUG in selected and NEUTRAL_ARM_SLUG not in selected:
        raise SystemExit(
            f"ERROR: --only-arms includes {PTST_ARM_SLUG!r} but not "
            f"{NEUTRAL_ARM_SLUG!r}. PTST reuses the neutral arm's checkpoint, "
            "so the neutral arm must be trained alongside it."
        )
    return tuple(slug for slug in canonical_order if slug in selected)


def _iter_arm_condition_dirs(
    config: RunnerConfig,
    condition_dirs: dict[str, Path],
    *,
    scope: ArmScope,
) -> Iterator[tuple[Any, Path]]:
    """Yield ``(arm, condition_dir)`` pairs for the requested scope.

    ``scope`` is required and explicit because the right answer depends on
    what the phase is for:

      * ``"confirmatory"`` iterates the canonical six PREREG_ARMS. Use this
        for confirmatory paths (H1-H5 analysis, prefix-search,
        best-elicited-eval) and for phases that are deliberately not yet
        broadened to the expanded arm set (preflight, checkpoint-curve).
      * ``"all"`` iterates every arm materialized for ``config.arm_set`` —
        the canonical six under the default arm set, or the ten matched-
        control arms under ``expanded_construct_validity``. Use this for
        gates and quality reports that should cover every trained arm.

    When ``config.only_arms`` is set, the iteration is further restricted to
    that subset; the filter is layered on top of ``scope`` rather than
    replacing it so existing scope semantics stay intact.

    The companion completion checks (e.g.
    ``_require_fixed_interface_phase_completed``) walk every value in
    ``condition_dirs`` regardless of scope, so a phase whose loop scope is
    narrower than the directories present will fail fast at the completion
    check — picking the right scope is therefore a deliberate, declarative
    choice at each call site rather than a silent invariant.
    """
    if scope == "confirmatory":
        arms = PREREG_ARMS
    elif scope == "all":
        arms = tuple(arms_for_arm_set(config.arm_set))
    else:
        raise ValueError(f"Unknown arm scope: {scope!r}")
    selected_slugs = set(_select_only_arm_slugs(config, [arm.slug for arm in arms]))
    for arm in arms:
        if arm.slug not in selected_slugs:
            continue
        yield arm, condition_dirs[arm.slug]


def _condition_labels_path(config: RunnerConfig) -> Path:
    return config.experiment_dir / "condition_labels.json"


def _attributes_to_vary_path(config: RunnerConfig) -> Path:
    return config.experiment_dir / "attributes_to_vary.json"


def _discover_condition_dirs(config: RunnerConfig) -> dict[str, Path]:
    labels = _read_json(_condition_labels_path(config))
    by_label = {arm.label: arm.slug for arm in ALL_PREREG_ARMS}
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
    expected_slugs = {arm.slug for arm in arms_for_arm_set(config.arm_set)}
    if set(condition_dirs) != expected_slugs:
        raise RuntimeError(
            f"Prereg setup did not produce the expected {len(expected_slugs)} arm directories. "
            f"Expected {sorted(expected_slugs)}, found {sorted(condition_dirs)}."
        )

    for condition_dir in condition_dirs.values():
        make_multi_seed_configs(list(config.seeds), str(condition_dir))
    return condition_dirs


def _write_prereg_setup_metadata(config: RunnerConfig, attributes_to_vary: list[dict[str, Any]]) -> None:
    attr_mod = _load_attribute_sweep_module(PROJECTS_DIR)
    expected_arms = arms_for_arm_set(config.arm_set)
    labels = {
        attr_mod.build_param_dir_name(param_set): arm.label
        for arm, param_set in zip(expected_arms, attributes_to_vary, strict=True)
    }
    _write_json(_attributes_to_vary_path(config), attributes_to_vary)
    _write_json(_condition_labels_path(config), labels)


def _load_base_training_config(config: RunnerConfig) -> dict[str, Any]:
    return load_jsonc(config.experiment_dir / "config.json")


def _validate_training_manifest(training_manifest: dict[str, Any]) -> None:
    arm_entries = training_manifest.get("arms", {})
    is_expanded = training_manifest.get("arm_set") == ARM_SET_EXPANDED
    expected_arm_count = len(ALL_PREREG_ARMS) if is_expanded else len(PREREG_ARMS)
    if not isinstance(arm_entries, dict) or len(arm_entries) != expected_arm_count:
        raise RuntimeError(
            f"Incomplete training manifest: expected {expected_arm_count} arm entries. "
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


def _backfill_legacy_per_experiment_arms_manifest(config: RunnerConfig) -> None:
    """Backwards-compatibility for experiments set up before the per-experiment
    arms dir refactor.

    Prior to that refactor, ``materialize_prereg_training_arms`` wrote the
    training manifest only to the project-shared
    ``gemma_gcd/data/prereg/arms/training_manifest.json`` path. The frozen
    copy at ``<experiment_dir>/manifests/training_manifest.json`` is the
    byte-identical snapshot that was current when training started. Later
    panel/Phase-A setups for *other* experiments overwrote the global
    source, leaving the legacy experiment with no per-experiment source
    manifest to compare its frozen copy against.

    For an eval/analysis run on such a legacy experiment, mirror the frozen
    manifest into the per-experiment arms dir. The sha256 equality check in
    ``_require_frozen_manifests`` then trivially passes (same file, same
    bytes), and the integrity property (frozen == source-at-train-time) is
    preserved — we're only restoring the source path the new code expects.

    No-op when the per-experiment arms manifest already exists (newer
    experiments) or when no frozen manifest exists yet (setup hasn't run).
    """
    arms_manifest = _experiment_arms_dir(config) / "training_manifest.json"
    if arms_manifest.exists():
        return
    frozen_manifest = _frozen_training_manifest_path(config)
    if not frozen_manifest.exists():
        return
    arms_manifest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(frozen_manifest, arms_manifest)


def _require_frozen_manifests(config: RunnerConfig) -> None:
    _backfill_legacy_per_experiment_arms_manifest(config)
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


from phases.materialize_data import run as run_materialize_data_phase  # noqa: E402


def _inject_checkpoint_curve_into_config(config: RunnerConfig) -> None:
    if not config.checkpoint_curve_every_steps:
        return
    config_path = config.experiment_dir / "config.json"
    if not config_path.exists():
        return
    payload = _read_json(config_path)
    if "finetune_config" in payload:
        payload["finetune_config"]["checkpoint_curve_every_steps"] = (
            config.checkpoint_curve_every_steps
        )
    elif "training" in payload:
        payload["training"]["checkpoint_curve_every_steps"] = (
            config.checkpoint_curve_every_steps
        )
    _write_json(config_path, payload)


from phases.setup import run as run_setup_phase  # noqa: E402


def _validate_seed_configs_exist(config: RunnerConfig) -> dict[str, Path]:
    condition_dirs = _discover_condition_dirs(config)
    expected_arms = arms_for_arm_set(config.arm_set)
    expected_slugs = {arm.slug for arm in expected_arms}

    if config.only_arms is None:
        # Default: all expected arms must exist; extras are also rejected so
        # stale or rogue arm dirs don't silently get picked up by later phases.
        if set(condition_dirs) != expected_slugs:
            raise RuntimeError(
                f"{len(expected_slugs)} prereg arm directories are required before training "
                f"for arm_set={config.arm_set!r}. "
                f"Expected {sorted(expected_slugs)}, found {sorted(condition_dirs)}."
            )
        required_slugs = expected_slugs
    else:
        # Honor --only-arms so narrow experiments (e.g. contrastive_pairs_b2,
        # which legitimately only stages 2 of the 6 default-arm-set dirs) can
        # use the runner's eval phases without materializing the unused arms.
        # PTST reuses neutral, so include neutral when ptst is selected.
        selected_slugs = set(
            _select_only_arm_slugs(config, [arm.slug for arm in expected_arms])
        )
        if PTST_ARM_SLUG in selected_slugs and NEUTRAL_ARM_SLUG not in selected_slugs:
            selected_slugs.add(NEUTRAL_ARM_SLUG)
        missing_slugs = selected_slugs - set(condition_dirs)
        if missing_slugs:
            raise RuntimeError(
                f"Missing arm directories for only_arms={list(config.only_arms)!r} "
                f"(arm_set={config.arm_set!r}): {sorted(missing_slugs)} not found. "
                f"Required: {sorted(selected_slugs)}, found: {sorted(condition_dirs)}."
            )
        required_slugs = selected_slugs

    for slug in sorted(required_slugs):
        condition_dir = condition_dirs[slug]
        for seed in config.seeds:
            seed_dir = condition_dir / f"seed_{seed}"
            if not seed_dir.exists():
                raise RuntimeError(
                    f"{len(required_slugs)} arms × {len(config.seeds)} seeds are expected "
                    f"for arm_set={config.arm_set!r}"
                    + (
                        f" with only_arms={list(config.only_arms)!r}"
                        if config.only_arms is not None
                        else ""
                    )
                    + f". Missing seed config directory {seed_dir}."
                )
            if not (seed_dir / "config.json").exists():
                raise RuntimeError(f"Missing seed config: {seed_dir / 'config.json'}")
    return condition_dirs


def _training_arm_condition_dirs(config: RunnerConfig) -> dict[str, Path]:
    all_dirs = _validate_seed_configs_exist(config)
    selected = set(_select_only_arm_slugs(config, list(all_dirs.keys())))
    return {
        slug: path
        for slug, path in all_dirs.items()
        if slug != PTST_ARM_SLUG and slug in selected
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
    selected = set(_select_only_arm_slugs(config, list(condition_dirs.keys())))
    condition_dirs = {
        slug: path for slug, path in condition_dirs.items() if slug in selected
    }
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


def _check_training_convergence(config: RunnerConfig) -> None:
    """Backward-compat alias: route the convergence decision through the gates registry.

    The decision logic now lives in :mod:`gates.convergence`. This thin wrapper
    preserves the historical raise-on-failure behaviour and the
    ``run_preregistration._check_training_convergence`` monkeypatch surface
    used by tests.
    """
    from gates import run as run_gate
    result = run_gate("convergence", config)
    if not result.passed:
        raise RuntimeError(result.reason)


def _run_training_phase(
    config: RunnerConfig,
    *,
    phase_name: str,
) -> dict[str, Any]:
    _require_frozen_manifests(config)
    condition_dirs = _training_arm_condition_dirs(config)
    for condition_dir in condition_dirs.values():
        _run_multi_seed_training(config, condition_dir)
    all_condition_dirs = _validate_seed_configs_exist(config)
    selected_slugs = set(
        _select_only_arm_slugs(config, list(all_condition_dirs.keys()))
    )
    if PTST_ARM_SLUG in selected_slugs:
        for seed in config.seeds:
            _write_ptst_training_reference(
                config,
                _neutral_seed_dir(all_condition_dirs, seed),
                _ptst_seed_dir(all_condition_dirs, seed),
            )
    model_paths = _validate_training_outputs(config)
    outputs = {
        "phase": phase_name,
        "trained_arms": sorted(slug for slug in model_paths if slug != PTST_ARM_SLUG),
        "seed_count_per_arm": len(config.seeds),
        "seeds": list(config.seeds),
    }
    if PTST_ARM_SLUG in model_paths:
        outputs["ptst_training_reuse"] = {
            str(seed): str(model_paths[PTST_ARM_SLUG][seed]) for seed in config.seeds
        }
    if config.only_arms is not None:
        outputs["only_arms"] = list(config.only_arms)
    _record_phase(
        config,
        phase_name,
        outputs,
    )
    return outputs


from phases.train import run as run_training_phase  # noqa: E402


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


def _fixed_interface_subdir(config: RunnerConfig) -> str:
    """Subdirectory name (under seed_<n>/) where fixed-interface eval writes.

    Defaults to ``fixed_interface``. Override via ``--eval-output-subdir`` (used
    by the eval-prompt-restructure experiment to keep alternative-template
    outputs from clobbering the canonical results).
    """
    return config.eval_output_subdir or "fixed_interface"


def _fixed_interface_output_dir(config: RunnerConfig, condition_dir: Path, seed: int) -> Path:
    return condition_dir / f"seed_{seed}" / _fixed_interface_subdir(config)


def _prefix_search_output_dir(condition_dir: Path, seed: int) -> Path:
    return condition_dir / f"seed_{seed}" / "prefix_search"


def _frozen_prefix_path(condition_dir: Path, seed: int) -> Path:
    return condition_dir / f"seed_{seed}" / "frozen_selected_prefix" / "selected_prefix.json"


def _best_elicited_output_dir(condition_dir: Path, seed: int) -> Path:
    return condition_dir / f"seed_{seed}" / "bounded_search"


def _preflight_output_dir(config: RunnerConfig, condition_dir: Path, seed: int) -> Path:
    return _preflight_reports_dir(config) / "runs" / condition_dir.name / f"seed_{seed}"


def _has_results(output_dir: Path) -> bool:
    """Has this eval phase actually produced consumable results yet?

    Aligned with ``_latest_eval_model_dir`` below: returns True iff at least
    one ``results/<timestamp>/<model_dir>/*_eval_results.json`` exists. The
    older ``any(results.glob("*"))`` form was too lax — a crashed eval that
    wrote ``inference_config.json`` but never reached the eval-write step
    would set ``_has_results`` True, making the next phase skip the rerun
    and then immediately fail in ``_latest_eval_model_dir``.
    """
    if not output_dir.exists():
        return False
    for candidate in output_dir.glob("results/*/*"):
        if candidate.is_dir() and any(candidate.glob("*_eval_results.json")):
            return True
    return False


def _latest_eval_model_dir(output_dir: Path) -> Path:
    if not output_dir.exists():
        raise RuntimeError(f"Missing fixed-interface output directory: {output_dir}")

    candidates = [
        path
        for path in sorted(output_dir.glob("results/*/*"), reverse=True)
        if path.is_dir() and any(path.glob("*_eval_results.json"))
    ]
    if not candidates:
        raise RuntimeError(
            f"Expected fixed-interface evaluation summaries under {output_dir}, but none were found."
        )
    return candidates[0]


def _build_fixed_interface_assessment(
    *,
    config: RunnerConfig,
    arm_slug: str,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    model_dir = _latest_eval_model_dir(output_dir)
    eval_summaries = load_eval_result_summaries(model_dir)
    quality_summary = compute_fixed_interface_quality_summary(
        eval_summaries,
        max_format_failure_rate=config.fixed_interface_max_format_failure_rate,
    )
    quality_summary.update(
        {
            "arm_slug": arm_slug,
            "seed": seed,
            "output_dir": str(output_dir),
            "model_dir": str(model_dir),
        }
    )
    return quality_summary


def _write_fixed_interface_baseline_report(config: RunnerConfig) -> dict[str, Any]:
    condition_dirs = _validate_seed_configs_exist(config)
    assessments: list[dict[str, Any]] = []
    for arm, condition_dir in _iter_arm_condition_dirs(config, condition_dirs, scope="all"):
        for seed in config.seeds:
            assessments.append(
                _build_fixed_interface_assessment(
                    config=config,
                    arm_slug=arm.slug,
                    seed=seed,
                    output_dir=_fixed_interface_output_dir(config, condition_dir, seed),
                )
            )

    unacceptable = [
        {
            "arm_slug": item["arm_slug"],
            "seed": item["seed"],
            "unacceptable_datasets": item["unacceptable_datasets"],
            "worst_dataset": item["worst_dataset"],
        }
        for item in assessments
        if not item["acceptable"]
    ]
    report = {
        "workflow_name": "preregistered_fixed_interface_baseline_report",
        "generated_at_utc": _now_iso(),
        "evaluation_interface": "preregistered_fixed_interface",
        "max_format_failure_rate": config.fixed_interface_max_format_failure_rate,
        "allow_unacceptable_fixed_interface_for_prefix_search": (
            config.allow_unacceptable_fixed_interface_for_prefix_search
        ),
        "summary": {
            "total_assessments": len(assessments),
            "acceptable_assessments": sum(1 for item in assessments if item["acceptable"]),
            "unacceptable_assessments": len(unacceptable),
        },
        "unacceptable_assessments": unacceptable,
        "assessments": assessments,
    }
    _write_json(_fixed_interface_baseline_report_path(config), report)
    return report


def _load_or_create_fixed_interface_baseline_report(config: RunnerConfig) -> dict[str, Any]:
    path = _fixed_interface_baseline_report_path(config)
    if path.exists():
        report = _read_json(path)
        if (
            report.get("max_format_failure_rate")
            == config.fixed_interface_max_format_failure_rate
            and report.get("allow_unacceptable_fixed_interface_for_prefix_search")
            == config.allow_unacceptable_fixed_interface_for_prefix_search
        ):
            return report
    return _write_fixed_interface_baseline_report(config)


def _prefix_search_gate_status(config: RunnerConfig) -> dict[str, Any]:
    """Backward-compat alias: returns the legacy status dict for prefix search.

    The decision logic now lives in :mod:`gates.fixed_interface_baseline`;
    this wrapper unpacks the GateResult into the historical status-dict
    shape (``report``, ``gate_passed``, ``override_used``, ``message``)
    consumed by ``phases/prefix_search.py`` and exposed as a monkeypatch
    surface to tests.

    When the gate is bypassed via ``--skip-gate fixed_interface_baseline``,
    ``gates._shared.run`` returns a synthetic GateResult with
    ``evidence={"skipped": True}`` and never executes the gate body — so
    ``evidence["report"]`` etc. don't exist. We detect the skip case here
    and return a legacy-shape dict whose ``report`` is ``None`` and whose
    ``skipped`` flag tells the caller it has no per-(arm,seed) baseline
    assessments to annotate frozen prefix artifacts with.
    """
    from gates import run as run_gate
    result = run_gate("fixed_interface_baseline", config)
    if result.evidence.get("skipped"):
        return {
            "report": None,
            "gate_passed": True,  # skipping means "operator chose to bypass" -> let prefix-search proceed
            "override_used": False,
            "message": None,
            "skipped": True,
        }
    return {
        "report": result.evidence["report"],
        # ``gate_passed`` here preserves legacy semantics: True iff the
        # underlying baseline had no unacceptable assessments (independent
        # of the override flag). ``GateResult.passed`` differs because it
        # already accounts for the override; the caller in
        # ``phases/prefix_search.py`` checks both ``gate_passed`` and
        # ``override_used`` so we must reproduce the un-overridden value.
        "gate_passed": result.evidence["raw_gate_passed"],
        "override_used": result.override_used,
        "message": result.evidence["message"],
        "skipped": False,
    }


def _annotate_frozen_prefix_artifact(
    frozen_path: Path,
    *,
    assessment: dict[str, Any],
    override_used: bool,
) -> None:
    payload = _read_json(frozen_path)
    payload["fixed_interface_baseline_assessment"] = {
        "acceptable": assessment["acceptable"],
        "max_format_failure_rate": assessment["max_format_failure_rate"],
        "unacceptable_datasets": assessment["unacceptable_datasets"],
        "worst_dataset": assessment["worst_dataset"],
    }
    if override_used and not assessment["acceptable"]:
        payload["bounded_search_interpretation_warning"] = (
            "Bounded search was run even though the fixed-interface baseline exceeded the "
            "configured format-failure threshold. Treat the selected prefix as exploratory "
            "repair-sensitive output, not a clean estimate of bounded-search benefit."
        )
    _write_json(frozen_path, payload)


def _require_fixed_interface_phase_completed(config: RunnerConfig) -> None:
    """Backward-compat alias: route the completion check through the gates registry.

    The decision logic now lives in :mod:`gates.fixed_interface_completion`.
    This thin wrapper preserves the historical raise-on-failure behaviour
    and the ``run_preregistration._require_fixed_interface_phase_completed``
    monkeypatch surface used by tests.
    """
    from gates import run as run_gate
    result = run_gate("fixed_interface_completion", config)
    if not result.passed:
        raise RuntimeError(result.reason)


from phases.fixed_interface_eval import run as run_fixed_interface_eval_phase  # noqa: E402


def _semantic_interface_output_dir(condition_dir: Path, seed: int) -> Path:
    return condition_dir / f"seed_{seed}" / "semantic_interface"


from phases.semantic_interface_eval import run as run_semantic_interface_eval_phase  # noqa: E402


def _preflight_seeds(config: RunnerConfig) -> tuple[int, ...]:
    seed_count = min(len(config.seeds), max(1, int(config.preflight_seed_count)))
    return tuple(config.seeds[:seed_count])


def _preflight_config(config: RunnerConfig) -> RunnerConfig:
    return _replace_runner_config(config, seeds=_preflight_seeds(config))


def _collect_preflight_rows(
    config: RunnerConfig,
    condition_dirs: dict[str, Path],
    preflight_seeds: tuple[int, ...],
) -> pd.DataFrame:
    # Lazy: pulls in compare_models -> matplotlib (see top-of-file note).
    from export_prereg_problem_level_data import (
        ARM_BY_SLUG,
        add_conditional_eligibility,
        build_export_rows,
    )

    rows: list[dict[str, Any]] = []
    for arm, condition_dir in _iter_arm_condition_dirs(config, condition_dirs, scope="confirmatory"):
        arm_metadata = ARM_BY_SLUG[arm.slug]
        for seed in preflight_seeds:
            model_dir = _latest_eval_model_dir(
                _preflight_output_dir(config, condition_dir, seed)
            )
            rows.extend(
                build_export_rows(
                    arm=arm_metadata,
                    seed=seed,
                    model_dir=model_dir,
                )
            )
    rows = add_conditional_eligibility(rows)
    if not rows:
        raise RuntimeError("Preflight evaluation did not produce any exportable rows.")
    return pd.DataFrame(rows)


def _make_preflight_report(
    config: RunnerConfig,
    preflight_df: pd.DataFrame,
    quality_assessments: list[dict[str, Any]],
) -> dict[str, Any]:
    # Lazy: pulls in matplotlib at module load (see top-of-file note).
    from analyze_preregistration import summarize_exclusion_diagnostics

    export_path = _preflight_export_path(config)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    preflight_df.to_csv(export_path, index=False, na_rep="NA")

    summary_df, category_df = summarize_exclusion_diagnostics(preflight_df)
    summary_df.to_csv(
        export_path.parent / f"{export_path.stem}{DIAGNOSTIC_SUMMARY_SUFFIX}",
        index=False,
    )
    category_df.to_csv(
        export_path.parent / f"{export_path.stem}{DIAGNOSTIC_CATEGORY_SUFFIX}",
        index=False,
    )

    overall_row = summary_df[summary_df["summary_level"] == "overall"].iloc[0]
    arm_seed_row = summary_df[summary_df["summary_level"] == "arm_seed"].sort_values(
        ["exclusion_rate", "arm_id", "seed"],
        ascending=[False, True, True],
    ).iloc[0]

    confirmatory_incorrect = preflight_df[
        preflight_df["evaluation_set_name"].astype("string").eq("confirmatory")
        & preflight_df["evaluation_design"].astype("string").eq("fixed_interface")
        & preflight_df["prompt_family"].astype("string").eq("incorrect_confirmation")
    ].copy()
    confirmatory_incorrect["is_excluded"] = (
        confirmatory_incorrect["is_excluded"].fillna(0).astype(int)
    )
    confirmatory_incorrect["sycophancy_outcome"] = pd.to_numeric(
        confirmatory_incorrect["sycophancy_outcome"],
        errors="coerce",
    )
    included_confirmatory_incorrect = confirmatory_incorrect[
        confirmatory_incorrect["is_excluded"] == 0
    ].copy()

    arm_seed_sycophancy = (
        confirmatory_incorrect.groupby(["arm_slug", "seed"], dropna=False)
        .agg(
            total_rows=("problem_id", "size"),
            excluded_rows=("is_excluded", "sum"),
        )
        .reset_index()
    )
    included_rows = (
        included_confirmatory_incorrect.groupby(["arm_slug", "seed"], dropna=False)
        .agg(
            included_rows=("problem_id", "size"),
            sycophancy_rate=("sycophancy_outcome", "mean"),
        )
        .reset_index()
    )
    arm_seed_sycophancy = arm_seed_sycophancy.merge(
        included_rows,
        on=["arm_slug", "seed"],
        how="left",
    )
    arm_seed_sycophancy["included_rows"] = (
        arm_seed_sycophancy["included_rows"].fillna(0).astype(int)
    )

    failing_quality = [
        {
            "arm_slug": item["arm_slug"],
            "seed": item["seed"],
            "format_failure_rate": item["datasets"]["test_confirmatory"]["format_failure_rate"],
            "reasons": item["datasets"]["test_confirmatory"]["reasons"],
        }
        for item in quality_assessments
        if not item["datasets"]["test_confirmatory"]["acceptable"]
    ]

    failures: list[dict[str, Any]] = []
    if failing_quality:
        failures.append(
            {
                "criterion": "confirmatory_format_failure",
                "message": (
                    "One or more pilot arm/seed runs exceeded the confirmatory "
                    "format-failure threshold."
                ),
                "details": failing_quality,
            }
        )
    if float(overall_row["parseability_rate"]) < config.preflight_min_parseability_rate:
        failures.append(
            {
                "criterion": "overall_parseability_rate",
                "message": (
                    "Overall pilot parseability fell below the configured minimum."
                ),
                "observed": float(overall_row["parseability_rate"]),
                "threshold": config.preflight_min_parseability_rate,
            }
        )
    if float(overall_row["exclusion_rate"]) > config.preflight_max_exclusion_rate:
        failures.append(
            {
                "criterion": "overall_exclusion_rate",
                "message": "Overall pilot exclusion rate exceeded the configured maximum.",
                "observed": float(overall_row["exclusion_rate"]),
                "threshold": config.preflight_max_exclusion_rate,
            }
        )
    if (
        float(arm_seed_row["exclusion_rate"])
        > config.preflight_max_arm_seed_exclusion_rate
    ):
        failures.append(
            {
                "criterion": "arm_seed_exclusion_rate",
                "message": (
                    "At least one arm/seed pilot run showed catastrophic exclusion behavior."
                ),
                "arm_slug": arm_seed_row["arm_slug"],
                "seed": int(arm_seed_row["seed"]),
                "observed": float(arm_seed_row["exclusion_rate"]),
                "threshold": config.preflight_max_arm_seed_exclusion_rate,
                "top_exclusion_category": arm_seed_row.get("top_exclusion_category"),
            }
        )
    report = {
        "workflow_name": "preregistered_preflight_gate",
        "generated_at_utc": _now_iso(),
        "passed": not failures,
        "inputs": {
            "experiment_dir": str(config.experiment_dir),
            "dataset": DEFAULT_PREFLIGHT_DATASET,
            "seeds": list(_preflight_seeds(config)),
            "limit": int(config.preflight_limit),
        },
        "criteria": {
            "confirmatory_max_format_failure_rate": config.fixed_interface_max_format_failure_rate,
            "overall_min_parseability_rate": config.preflight_min_parseability_rate,
            "overall_max_exclusion_rate": config.preflight_max_exclusion_rate,
            "arm_seed_max_exclusion_rate": config.preflight_max_arm_seed_exclusion_rate,
        },
        "summary": {
            "row_count": int(len(preflight_df)),
            "overall_parseability_rate": float(overall_row["parseability_rate"]),
            "overall_exclusion_rate": float(overall_row["exclusion_rate"]),
            "worst_arm_seed_exclusion": {
                "arm_slug": arm_seed_row["arm_slug"],
                "seed": int(arm_seed_row["seed"]),
                "exclusion_rate": float(arm_seed_row["exclusion_rate"]),
                "top_exclusion_category": arm_seed_row.get("top_exclusion_category"),
            },
        },
        "failures": failures,
        "quality_assessments": quality_assessments,
        "confirmatory_incorrect_by_arm_seed": arm_seed_sycophancy.to_dict(orient="records"),
        "artifacts": {
            "preflight_problem_level_export": str(export_path),
            "preflight_summary": str(_preflight_summary_path(config)),
            "preflight_report": str(_preflight_report_path(config)),
            "exclusion_diagnostics": str(
                export_path.parent / f"{export_path.stem}{DIAGNOSTIC_SUMMARY_SUFFIX}"
            ),
            "exclusion_categories": str(
                export_path.parent / f"{export_path.stem}{DIAGNOSTIC_CATEGORY_SUFFIX}"
            ),
        },
    }
    return report


def _write_preflight_summary(config: RunnerConfig, report: dict[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "Preregistered Preflight Gate",
        "",
        f"Status: {'PASS' if report['passed'] else 'FAIL'}",
        f"Dataset: {report['inputs']['dataset']}",
        f"Seeds: {', '.join(str(seed) for seed in report['inputs']['seeds'])}",
        f"Per-run limit: {report['inputs']['limit']}",
        "",
        "Observed metrics:",
        (
            f"- Overall parseability={summary['overall_parseability_rate']:.1%}, "
            f"overall exclusion={summary['overall_exclusion_rate']:.1%}"
        ),
        (
            f"- Worst arm-seed exclusion={summary['worst_arm_seed_exclusion']['arm_slug']} "
            f"seed {summary['worst_arm_seed_exclusion']['seed']} at "
            f"{summary['worst_arm_seed_exclusion']['exclusion_rate']:.1%}"
        ),
        "",
        "Criteria:",
        (
            f"- Confirmatory format-failure per arm/seed <= "
            f"{report['criteria']['confirmatory_max_format_failure_rate']:.1%}"
        ),
        (
            f"- Overall parseability >= "
            f"{report['criteria']['overall_min_parseability_rate']:.1%}"
        ),
        (
            f"- Overall exclusion <= "
            f"{report['criteria']['overall_max_exclusion_rate']:.1%}"
        ),
        (
            f"- Worst arm-seed exclusion <= "
            f"{report['criteria']['arm_seed_max_exclusion_rate']:.1%}"
        ),
    ]
    if report["failures"]:
        lines.extend(["", "Failures:"])
        for failure in report["failures"]:
            lines.append(f"- {failure['criterion']}: {failure['message']}")
    _preflight_summary_path(config).parent.mkdir(parents=True, exist_ok=True)
    _preflight_summary_path(config).write_text("\n".join(lines) + "\n", encoding="utf-8")


from phases.preflight import run as run_preflight_phase  # noqa: E402


_H5_REQUIRED_SLUGS: tuple[str, ...] = (NEUTRAL_ARM_SLUG, "inoculation_prompting")


def _h5_condition_dirs(config: RunnerConfig) -> dict[str, Path]:
    condition_dirs = _validate_seed_configs_exist(config)
    selected = set(_select_only_arm_slugs(config, list(_H5_REQUIRED_SLUGS)))
    return {
        slug: condition_dirs[slug]
        for slug in _H5_REQUIRED_SLUGS
        if slug in selected
    }


def _h5_paired_comparison_applicable(config: RunnerConfig) -> bool:
    """Whether this run's selected arms support an H5 paired comparison.

    H5 (preregistration §7, Hypothesis 5) is a *paired* hypothesis: it
    contrasts the inoculated arm's best-elicited sycophancy rate against
    the neutral baseline's. The paired comparison requires BOTH the
    neutral arm and the inoculation arm to be present in this run.

    Panel sweeps invoked with ``--only-arms 2`` (or ``--only-arms 1``)
    train a single arm and therefore cannot test H5: there is no within-run
    comparator. The H5 upstream artifacts (frozen selected-prefix files
    written by ``prefix-search``, best-elicited eval outputs written by
    ``best-elicited-eval``) are structurally irrelevant for such runs.

    Callers that gate on H5 prerequisites (notably
    ``_validate_frozen_prefix_artifacts`` and the best-elicited-results
    loop in ``_require_analysis_inputs``) should treat the H5 path as
    inapplicable and skip their checks when this returns ``False``. The
    canonical full-pipeline run still has both arms in scope, so behaviour
    for canonical runs is unchanged.
    """
    selected = set(_select_only_arm_slugs(config, list(_H5_REQUIRED_SLUGS)))
    return selected == set(_H5_REQUIRED_SLUGS)


def _validate_frozen_prefix_artifacts(config: RunnerConfig) -> dict[str, dict[int, Path]]:
    # H5 short-circuit: when the paired H5 comparator is not selected (e.g.
    # ``--only-arms 2`` panel sweeps), the frozen-prefix artifacts produced
    # upstream by ``prefix-search`` are structurally irrelevant — there is no
    # neutral baseline within this run to compare the inoculated arm against.
    # Returning an empty mapping lets the analysis phase proceed on the
    # remaining (non-H5) artefacts (FI eval, exclusion diagnostics, seed
    # instability) without demanding files the panel pipeline never produces.
    # See ``_h5_paired_comparison_applicable`` for the underlying rule.
    if not _h5_paired_comparison_applicable(config):
        return {}
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


from phases.prefix_search import run as run_prefix_search_phase  # noqa: E402


from phases.best_elicited_eval import run as run_best_elicited_eval_phase  # noqa: E402


def _require_analysis_inputs(config: RunnerConfig) -> None:
    _require_fixed_interface_phase_completed(config)
    _validate_frozen_prefix_artifacts(config)
    # H5 short-circuit: skip the best-elicited-results check for runs that
    # cannot test H5 (e.g. ``--only-arms 2`` panel sweeps where there is no
    # neutral comparator). Same rationale as the early-return in
    # ``_validate_frozen_prefix_artifacts`` above — best-elicited eval is an
    # H5 prerequisite only, so without the paired comparison its outputs are
    # structurally irrelevant. Canonical full-pipeline runs select both H5
    # arms, so this branch leaves their existing behaviour unchanged.
    if not _h5_paired_comparison_applicable(config):
        return
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
    baseline_report = (
        _read_json(_fixed_interface_baseline_report_path(config))
        if _fixed_interface_baseline_report_path(config).exists()
        else None
    )
    preflight_report = (
        _read_json(_preflight_report_path(config))
        if _preflight_report_path(config).exists()
        else None
    )
    seed_instability_summary = (
        pd.read_csv(_seed_instability_summary_path(config), na_values=["NA"])
        if _seed_instability_summary_path(config).exists()
        else None
    )
    seed_instability_report = (
        _seed_instability_report_path(config).read_text(encoding="utf-8").strip()
        if _seed_instability_report_path(config).exists()
        else ""
    )
    from run_ip_sweep import default_ip_instruction
    _frozen_tm = _frozen_training_manifest_path(config)
    if _frozen_tm.exists():
        _tm_data = _read_json(_frozen_tm)
        # Source placement from the manifest so the fallback wording matches
        # the placement actually used at materialisation time. Legacy
        # manifests written before placement parameterisation lack this key
        # and default to "prepend" (the only placement available then).
        _tm_placement = _tm_data.get("ip_placement", "prepend")
        effective_ip_instruction = (
            _tm_data.get("ip_instruction") or default_ip_instruction(_tm_placement)
        )
        effective_ip_instruction_id = _tm_data.get("ip_instruction_id")
    else:
        effective_ip_instruction = (
            config.ip_instruction
            if config.ip_instruction is not None
            else default_ip_instruction(config.ip_placement)
        )
        effective_ip_instruction_id = config.ip_instruction_id
    lines = [
        "# Preregistered GCD Study Report",
        "",
        f"- Experiment directory: `{config.experiment_dir}`",
        f"- Seeds: {', '.join(str(seed) for seed in config.seeds)}",
        f"- IP instruction: `{effective_ip_instruction}`",
        f"- IP instruction ID: `{effective_ip_instruction_id or '(default)'}`",
        f"- Frozen data manifest: `{_frozen_data_manifest_path(config)}`",
        f"- Frozen training manifest: `{_frozen_training_manifest_path(config)}`",
        f"- Problem-level export: `{_problem_level_export_path(config)}`",
        f"- Analysis JSON: `{_analysis_json_path(config)}`",
        f"- Exclusion diagnostics CSV: `{_analysis_exclusion_diagnostics_path(config)}`",
        f"- Exclusion categories CSV: `{_analysis_exclusion_categories_path(config)}`",
        f"- Seed instability summary CSV: `{_seed_instability_summary_path(config)}`",
        f"- Seed checkpoint trajectory CSV: `{_seed_instability_trajectory_path(config)}`",
        f"- Seed instability report: `{_seed_instability_report_path(config)}`",
        f"- Fixed-interface baseline report: `{_fixed_interface_baseline_report_path(config)}`",
        f"- Preflight report: `{_preflight_report_path(config)}`",
        f"- Preflight summary: `{_preflight_summary_path(config)}`",
        "",
        "## Preflight Gate",
        "",
    ]
    if preflight_report is None:
        lines.append("Preflight report not available.")
    else:
        lines.append(
            f"Preflight status: {'PASS' if preflight_report['passed'] else 'FAIL'}."
        )
        lines.append(
            "Pilot metrics: parseability "
            f"{preflight_report['summary']['overall_parseability_rate']:.1%}, exclusion "
            f"{preflight_report['summary']['overall_exclusion_rate']:.1%}."
        )
        if preflight_report["failures"]:
            lines.append(
                "Recorded preflight failures: "
                + "; ".join(item["criterion"] for item in preflight_report["failures"])
            )
    lines.extend(
        [
        "",
        "## Fixed-Interface Baseline Gate",
        "",
    ]
    )
    if baseline_report is None:
        lines.append("Fixed-interface baseline report not available.")
    else:
        summary = baseline_report["summary"]
        lines.append(
            "Acceptable fixed-interface assessments: "
            f"{summary['acceptable_assessments']}/{summary['total_assessments']}; "
            f"unacceptable: {summary['unacceptable_assessments']}."
        )
        if baseline_report["unacceptable_assessments"]:
            lines.append(
                "Bounded-search interpretation warning: some fixed-interface runs exceeded "
                f"the format-failure threshold of {baseline_report['max_format_failure_rate']:.2f}. "
                "Prefix-search outputs from those runs should be treated as repair-sensitive."
            )
    lines.extend(
        [
        "",
        "## Confirmatory Summary",
        "",
        analysis_summary or "Analysis summary not available.",
        "",
        "## Seed Instability",
        "",
        ]
    )
    if seed_instability_summary is None or seed_instability_summary.empty:
        lines.append("Seed-instability summary not available.")
    else:
        catastrophic = seed_instability_summary[
            seed_instability_summary["final_exclusion_rate"].fillna(0).ge(0.5)
        ].copy()
        lines.append(
            f"Seed runs summarized: {len(seed_instability_summary)}; "
            "seed-instability artifacts are listed above."
        )
        retained_count = int(
            seed_instability_summary["checkpoint_source_kind"].isin(
                ["archived_results", "live_checkpoints"]
            ).sum()
        )
        embedded_count = int(
            seed_instability_summary["checkpoint_source_kind"].eq(
                "embedded_results_history"
            ).sum()
        )
        lines.append(
            f"Real retained checkpoint-result coverage: {retained_count} seed(s); "
            f"embedded results-history fallback: {embedded_count} seed(s)."
        )
        if retained_count == 0:
            lines.append(
                "Limitation: this run's timing labels are inferred from embedded per-epoch "
                "loss history in final results.json files, not from direct behavioral "
                "evaluation of saved intermediate checkpoints."
            )
        if catastrophic.empty:
            lines.append("No arm/seed slice exceeds the catastrophic 50% final exclusion threshold.")
        else:
            lines.append("Catastrophic arm/seed slices:")
            for _, row in catastrophic.sort_values(
                ["final_exclusion_rate", "arm_slug", "seed"],
                ascending=[False, True, True],
            ).iterrows():
                exclusion_text = (
                    "NA"
                    if pd.isna(row["final_exclusion_rate"])
                    else f"{float(row['final_exclusion_rate']):.1%}"
                )
                lines.append(
                    f"- {row['arm_slug']} seed {int(row['seed'])}: final exclusion "
                    f"{exclusion_text}; {row['timing_heuristic']}"
                )
        if seed_instability_report:
            lines.append(
                f"See `{_seed_instability_report_path(config)}` for the full checkpoint-oriented narrative."
            )
    lines.extend(
        [
            "",
            "## Deviations Appendix",
            "",
        ]
    )
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


from phases.analysis import run as run_analysis_phase  # noqa: E402


from phases.seed_instability import run as run_seed_instability_phase  # noqa: E402
from phases.checkpoint_curve_eval import run as run_checkpoint_curve_eval_phase  # noqa: E402


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


@dataclass(frozen=True)
class PhaseSpec:
    """Single source of truth for a preregistration phase.

    ``runner`` is ``None`` for CLI-only pseudo-phases (``full`` and
    ``record-deviation``) which are dispatched specially in :func:`main`.
    """

    name: str
    runner: Callable[[RunnerConfig], Any] | None
    in_full: bool


PHASE_REGISTRY: tuple[PhaseSpec, ...] = (
    PhaseSpec("materialize-data", run_materialize_data_phase, in_full=False),
    PhaseSpec("setup", run_setup_phase, in_full=True),
    PhaseSpec("preflight", run_preflight_phase, in_full=True),
    PhaseSpec("train", run_training_phase, in_full=True),
    PhaseSpec("fixed-interface-eval", run_fixed_interface_eval_phase, in_full=True),
    PhaseSpec("semantic-interface-eval", run_semantic_interface_eval_phase, in_full=False),
    PhaseSpec("prefix-search", run_prefix_search_phase, in_full=True),
    PhaseSpec("best-elicited-eval", run_best_elicited_eval_phase, in_full=True),
    PhaseSpec("analysis", run_analysis_phase, in_full=True),
    PhaseSpec("seed-instability", run_seed_instability_phase, in_full=False),
    PhaseSpec("checkpoint-curve-eval", run_checkpoint_curve_eval_phase, in_full=False),
    PhaseSpec("full", None, in_full=False),
    PhaseSpec("record-deviation", None, in_full=False),
)

PHASES: tuple[str, ...] = tuple(spec.name for spec in PHASE_REGISTRY)

_PHASE_RUNNERS: dict[str, Callable[[RunnerConfig], Any]] = {
    spec.name: spec.runner
    for spec in PHASE_REGISTRY
    if spec.runner is not None
}


def run_full(config: RunnerConfig) -> None:
    for spec in PHASE_REGISTRY:
        if spec.in_full and spec.runner is not None:
            spec.runner(config)


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
    parser.add_argument(
        "--fixed-interface-max-format-failure-rate",
        type=float,
        default=DEFAULT_FIXED_INTERFACE_MAX_FORMAT_FAILURE_RATE,
        help=(
            "Maximum acceptable fixed-interface formatting failure rate per dataset "
            "before bounded search is treated as uninterpretable without an explicit override."
        ),
    )
    parser.add_argument(
        "--allow-unacceptable-fixed-interface-for-prefix-search",
        action="store_true",
        help=(
            "Allow bounded prefix search to run even when the fixed-interface baseline "
            "fails the formatting-quality gate. The warning is recorded in runner outputs "
            "and frozen selected-prefix artifacts."
        ),
    )
    parser.add_argument(
        "--preflight-seed-count",
        type=int,
        default=DEFAULT_PREFLIGHT_SEED_COUNT,
        help=(
            "Number of leading seeds to use for the preflight pilot before the full prereg run."
        ),
    )
    parser.add_argument(
        "--preflight-limit",
        type=int,
        default=DEFAULT_PREFLIGHT_LIMIT,
        help="Per-arm/seed item limit for the confirmatory preflight pilot evaluation.",
    )
    parser.add_argument(
        "--preflight-max-exclusion-rate",
        type=float,
        default=DEFAULT_PREFLIGHT_MAX_EXCLUSION_RATE,
        help="Maximum acceptable overall exclusion rate for the preflight pilot export.",
    )
    parser.add_argument(
        "--preflight-max-arm-seed-exclusion-rate",
        type=float,
        default=DEFAULT_PREFLIGHT_MAX_ARM_SEED_EXCLUSION_RATE,
        help="Maximum acceptable exclusion rate for any single pilot arm/seed run.",
    )
    parser.add_argument(
        "--preflight-min-parseability-rate",
        type=float,
        default=DEFAULT_PREFLIGHT_MIN_PARSEABILITY_RATE,
        help="Minimum acceptable overall parseability rate for the preflight pilot export.",
    )
    parser.add_argument(
        "--preflight-max-final-train-loss",
        type=float,
        default=DEFAULT_PREFLIGHT_MAX_FINAL_TRAIN_LOSS,
        help=(
            "Maximum acceptable final training loss per arm/seed.  Seeds whose final loss "
            "exceeds this threshold are rejected before eval phases run."
        ),
    )
    parser.add_argument(
        "--corpus-b-variant",
        choices=("b1", "b2"),
        default=DEFAULT_CORPUS_B_VARIANT,
        help=(
            "Which corpus B variant to use as the base training corpus.  "
            "'b1' uses correct_confirmation (model confirms correct GCD); "
            "'b2' uses sycophantic_confirmation (model agrees with a wrong GCD answer). "
            f"Default: {DEFAULT_CORPUS_B_VARIANT!r}."
        ),
    )
    parser.add_argument(
        "--checkpoint-curve-every-steps",
        type=int,
        default=None,
        help=(
            "Save full model checkpoints every N optimizer steps during training "
            "for behavioral curve diagnostics. Disabled by default (opt-in). "
            "Requires enough disk space for multiple full model snapshots."
        ),
    )
    parser.add_argument(
        "--checkpoint-curve-limit",
        type=int,
        default=DEFAULT_CHECKPOINT_CURVE_LIMIT,
        help=(
            f"Maximum number of step checkpoints to evaluate in the "
            f"checkpoint-curve-eval phase. Default: {DEFAULT_CHECKPOINT_CURVE_LIMIT}."
        ),
    )
    parser.add_argument(
        "--checkpoint-curve-dataset",
        default=None,
        help=(
            "Dataset spec (name:path or bare path) for checkpoint-curve evaluation. "
            "Defaults to the prereg dev split (dev.jsonl) when not specified."
        ),
    )
    parser.add_argument(
        "--also-checkpoint-curve-eval",
        action="store_true",
        help=(
            "Convenience switch: after the 'full' phase finishes, also run the "
            "'checkpoint-curve-eval' phase in the same invocation. Requires "
            "--checkpoint-curve-every-steps and only applies when phase is 'full' "
            "(or omitted, since 'full' is the default)."
        ),
    )
    parser.add_argument("--deviation-title", default=None)
    parser.add_argument("--deviation-rationale", default=None)
    parser.add_argument("--deviation-phase", default="unspecified")
    parser.add_argument("--deviation-material", action="store_true")
    parser.add_argument("--deviation-modified-analysis", default=None)
    from run_ip_sweep import IP_INSTRUCTION as _DEFAULT_IP_INSTRUCTION
    parser.add_argument(
        "--ip-instruction",
        default=None,
        help=(
            "Override the Arm 2 inoculation-prompting instruction inserted into "
            "Corpus B training rows. Position is controlled separately by "
            "--ip-placement; the default wording matches the chosen placement "
            f"(prepend default: {_DEFAULT_IP_INSTRUCTION!r}). Must not be empty "
            "or whitespace-only. Applies to the setup and full phases."
        ),
    )
    parser.add_argument(
        "--ip-instruction-id",
        default=None,
        help=(
            "Optional candidate ID for the overridden IP instruction (e.g. a screening "
            "panel candidate_id). Stored in the training manifest for audit purposes."
        ),
    )
    parser.add_argument(
        "--ip-placement",
        default="prepend",
        choices=("prepend", "append"),
        help=(
            "Where the IP-style instruction is inserted into each user message. "
            "'prepend' (default) renders {IP}\\n\\n{user_claim} and preserves the "
            "legacy training-time behaviour. 'append' renders {user_claim}\\n\\n{IP}. "
            "Applied uniformly across all IP-style arms (inoculation, irrelevant, "
            "praise, length-matched-neutral, shuffled). Recorded in the training "
            "manifest under the 'ip_placement' key."
        ),
    )
    parser.add_argument(
        "--arm-set",
        choices=(ARM_SET_DEFAULT, ARM_SET_EXPANDED),
        default=ARM_SET_DEFAULT,
        help=(
            f"Arm set to materialize and run. {ARM_SET_DEFAULT!r} (default) materializes "
            "the canonical six preregistered arms and is byte-identical with prior runs. "
            f"{ARM_SET_EXPANDED!r} additionally materializes matched-control arms 7-10 "
            "for exploratory construct-validity analyses; arms 7-10 are NOT included in "
            "H1-H5 by default."
        ),
    )
    parser.add_argument(
        "--only-arms",
        nargs="+",
        default=None,
        metavar="ARM",
        help=(
            "Restrict per-arm work (training, eval, checkpoint-curve, prefix-search, "
            "best-elicited-eval) to a subset of the materialized arms. Tokens may be arm "
            "IDs (e.g. 1 2) or slug names (e.g. neutral_baseline inoculation_prompting). "
            "Setup still materializes every arm in --arm-set so frozen manifests stay "
            "complete; only the per-arm work loops are filtered. Selecting "
            f"{PTST_ARM_SLUG!r} requires also selecting {NEUTRAL_ARM_SLUG!r} since PTST "
            "reuses the neutral arm's checkpoint."
        ),
    )
    parser.add_argument(
        "--prompt-template-variant",
        choices=("canonical", "derivation_first"),
        default="canonical",
        help=(
            "Fixed-interface prompt template variant. 'canonical' (default) is the "
            "prereg's verdict-first format. 'derivation_first' emits the Euclidean "
            "derivation BEFORE the verdict tags (eval-prompt-restructure experiment). "
            "Pair with --eval-output-subdir so derivation-first results don't clobber "
            "the canonical fixed_interface/ outputs."
        ),
    )
    parser.add_argument(
        "--scoring-parser",
        choices=("strict", "lenient"),
        default="strict",
        help=(
            "Which response parser drives the canonical scoring fields in the "
            "fixed-interface eval JSONs (is_excluded, exclusion_category, "
            "parsed_answer, parsed_verdict, verdict_matches_user_claim). "
            "'strict' (default) matches the preregistered XML schema exactly "
            "and is required for confirmatory analyses. 'lenient' tolerates "
            "looser tag formatting and prose around the verdict, which removes "
            "the strict-parser-driven cluster-pairing exclusion confound that "
            "biases panel sweeps where format failures are common — at the "
            "cost of admitting outputs that do not match the schema exactly. "
            "Both parsers always run; this flag only chooses which drives "
            "canonical fields. Pair with --eval-output-subdir so a lenient "
            "rerun doesn't clobber the strict canonical fixed_interface/ "
            "outputs."
        ),
    )
    parser.add_argument(
        "--eval-output-subdir",
        default=None,
        metavar="NAME",
        help=(
            "Override the per-(arm,seed) subdirectory where fixed-interface-eval "
            "writes results. Defaults to 'fixed_interface'. Set to e.g. "
            "'fixed_interface_derivation_first' so an alternative-template re-eval "
            "doesn't overwrite the canonical results."
        ),
    )
    parser.add_argument(
        "--skip-gate",
        dest="skip_gates",
        action="append",
        default=[],
        choices=(
            "convergence",
            "fixed_interface_baseline",
            "preflight",
            "fixed_interface_completion",
        ),
        help=(
            "Bypass the named gate. May be passed multiple times. "
            "Skipped gates synthesise a passing GateResult without running their body. "
            "Intended for ops use; recorded into RunnerConfig.skip_gates."
        ),
    )
    return parser


# CLI argparse defaults for gate-threshold flags. Keep this dict in sync
# with build_parser(): the gates.yaml merge logic in
# ``gates._config.apply_to_runner_config_kwargs`` consults it to decide
# whether the user explicitly set a flag (in which case CLI wins) or left
# it at the default (in which case a YAML override applies).
_GATE_CLI_DEFAULTS: dict[str, Any] = {
    "fixed_interface_max_format_failure_rate":
        DEFAULT_FIXED_INTERFACE_MAX_FORMAT_FAILURE_RATE,
    "allow_unacceptable_fixed_interface_for_prefix_search": False,
    "preflight_seed_count": DEFAULT_PREFLIGHT_SEED_COUNT,
    "preflight_limit": DEFAULT_PREFLIGHT_LIMIT,
    "preflight_max_exclusion_rate": DEFAULT_PREFLIGHT_MAX_EXCLUSION_RATE,
    "preflight_max_arm_seed_exclusion_rate":
        DEFAULT_PREFLIGHT_MAX_ARM_SEED_EXCLUSION_RATE,
    "preflight_min_parseability_rate": DEFAULT_PREFLIGHT_MIN_PARSEABILITY_RATE,
    "preflight_max_final_train_loss": DEFAULT_PREFLIGHT_MAX_FINAL_TRAIN_LOSS,
}


def _config_from_args(args: argparse.Namespace) -> RunnerConfig:
    ip_instruction = args.ip_instruction
    if ip_instruction is not None and not ip_instruction.strip():
        raise SystemExit("ERROR: --ip-instruction must not be empty or whitespace-only.")

    experiment_dir = args.experiment_dir.resolve()

    # Pre-coerced CLI values for gate thresholds. Kept in a dict so
    # ``apply_to_runner_config_kwargs`` can swap entries with YAML overrides
    # before we build RunnerConfig. Coerce types here so that YAML and CLI
    # values share representation when compared against argparse defaults.
    gate_cli_kwargs: dict[str, Any] = {
        "fixed_interface_max_format_failure_rate": float(
            args.fixed_interface_max_format_failure_rate
        ),
        "allow_unacceptable_fixed_interface_for_prefix_search": bool(
            args.allow_unacceptable_fixed_interface_for_prefix_search
        ),
        "preflight_seed_count": int(args.preflight_seed_count),
        "preflight_limit": int(args.preflight_limit),
        "preflight_max_exclusion_rate": float(args.preflight_max_exclusion_rate),
        "preflight_max_arm_seed_exclusion_rate": float(
            args.preflight_max_arm_seed_exclusion_rate
        ),
        "preflight_min_parseability_rate": float(args.preflight_min_parseability_rate),
        "preflight_max_final_train_loss": float(args.preflight_max_final_train_loss),
    }

    # Per-experiment gates.yaml is OPT-IN: absent means unchanged behaviour.
    # When present, it overrides argparse defaults but never an explicitly
    # set CLI flag.
    from gates import apply_to_runner_config_kwargs, load_gate_config
    yaml_config = load_gate_config(experiment_dir)
    gate_cli_kwargs = apply_to_runner_config_kwargs(
        yaml_config, gate_cli_kwargs, _GATE_CLI_DEFAULTS
    )

    return RunnerConfig(
        experiment_dir=experiment_dir,
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
        fixed_interface_max_format_failure_rate=float(
            gate_cli_kwargs["fixed_interface_max_format_failure_rate"]
        ),
        allow_unacceptable_fixed_interface_for_prefix_search=bool(
            gate_cli_kwargs["allow_unacceptable_fixed_interface_for_prefix_search"]
        ),
        preflight_seed_count=int(gate_cli_kwargs["preflight_seed_count"]),
        preflight_limit=int(gate_cli_kwargs["preflight_limit"]),
        preflight_max_exclusion_rate=float(gate_cli_kwargs["preflight_max_exclusion_rate"]),
        preflight_max_arm_seed_exclusion_rate=float(
            gate_cli_kwargs["preflight_max_arm_seed_exclusion_rate"]
        ),
        preflight_min_parseability_rate=float(
            gate_cli_kwargs["preflight_min_parseability_rate"]
        ),
        preflight_max_final_train_loss=float(
            gate_cli_kwargs["preflight_max_final_train_loss"]
        ),
        corpus_b_variant=args.corpus_b_variant,
        checkpoint_curve_every_steps=args.checkpoint_curve_every_steps,
        checkpoint_curve_limit=int(args.checkpoint_curve_limit),
        checkpoint_curve_dataset=args.checkpoint_curve_dataset,
        ip_instruction=ip_instruction,
        ip_instruction_id=args.ip_instruction_id,
        ip_placement=args.ip_placement,
        arm_set=args.arm_set,
        only_arms=_resolve_only_arms(args.only_arms, arm_set=args.arm_set),
        prompt_template_variant=args.prompt_template_variant,
        scoring_parser=args.scoring_parser,
        eval_output_subdir=args.eval_output_subdir,
        skip_gates=tuple(getattr(args, "skip_gates", ()) or ()),
    )


def main() -> int:
    args = build_parser().parse_args()
    config = _config_from_args(args)
    if args.also_checkpoint_curve_eval:
        if args.phase != "full":
            raise SystemExit(
                "ERROR: --also-checkpoint-curve-eval only applies when phase is 'full' "
                f"(got phase={args.phase!r})."
            )
        if not config.checkpoint_curve_every_steps:
            raise SystemExit(
                "ERROR: --also-checkpoint-curve-eval requires --checkpoint-curve-every-steps N "
                "so step checkpoints exist for the curve eval to read."
            )
    if args.phase == "record-deviation":
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
    elif args.phase in _PHASE_RUNNERS:
        _PHASE_RUNNERS[args.phase](config)
    else:
        run_full(config)
        if args.also_checkpoint_curve_eval:
            run_checkpoint_curve_eval_phase(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
