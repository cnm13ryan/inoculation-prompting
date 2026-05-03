"""``semantic-interface-eval`` phase: secondary robustness evaluation that
replaces the XML formatting burden with natural-language prompts.

Outputs are labeled ``evaluation_design='semantic_interface'`` and are NOT
used for any primary confirmatory claim. Phase is secondary, robustness-only,
explicitly exploratory.

Reads:  trained adapter checkpoints under ``seed_<n>/results/<ts>/``.
Writes: per-arm-per-seed semantic-interface eval outputs under
        ``seed_<n>/semantic_interface/``; appends a phase entry to
        ``run_manifest.json``.
"""

from __future__ import annotations

import sys

from phases._runner_helpers import (
    FIXED_EVAL_SCRIPT,
    PROJECTS_DIR,
    PTST_ARM_SLUG,
    _evaluation_common_args,
    _has_results,
    _iter_arm_condition_dirs,
    _record_phase,
    _require_frozen_manifests,
    _run_checked,
    _select_only_arm_slugs,
    _semantic_interface_output_dir,
    _validate_seed_configs_exist,
    _validate_training_outputs,
    arms_for_arm_set,
)


def run(config: RunnerConfig) -> None:
    _require_frozen_manifests(config)
    model_paths = _validate_training_outputs(config)
    condition_dirs = _validate_seed_configs_exist(config)
    evaluated_arms = arms_for_arm_set(config.arm_set)
    for arm, condition_dir in _iter_arm_condition_dirs(
        config, condition_dirs, scope="all"
    ):
        for seed in config.seeds:
            output_dir = _semantic_interface_output_dir(condition_dir, seed)
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
                "--evaluation-interface",
                "semantic_interface",
                "--output-dir",
                str(output_dir),
                "--datasets",
                "test_confirmatory:gemma_gcd/data/prereg/test_confirmatory.jsonl",
                "test_paraphrase:gemma_gcd/data/prereg/test_paraphrase.jsonl",
                "same_domain_extrapolation:gemma_gcd/data/prereg/test_near_transfer.jsonl",
                "--include-capability-diagnostics",
                *_evaluation_common_args(config),
            ]
            _run_checked(cmd, cwd=PROJECTS_DIR)

    missing: list[str] = []
    selected_semantic = set(
        _select_only_arm_slugs(config, list(condition_dirs.keys()))
    )
    for slug, condition_dir in condition_dirs.items():
        if slug not in selected_semantic:
            continue
        for seed in config.seeds:
            output_dir = _semantic_interface_output_dir(condition_dir, seed)
            if not _has_results(output_dir):
                missing.append(str(output_dir))
    if missing:
        rendered = "; ".join(missing[:5])
        raise RuntimeError(
            "Semantic-interface evaluation artifacts are missing after phase run. "
            f"Missing: {rendered}"
        )
    _record_phase(
        config,
        "semantic-interface-eval",
        {
            "evaluated_arms": len(evaluated_arms),
            "arm_set": config.arm_set,
            "seed_count_per_arm": len(config.seeds),
            "classification": "secondary_robustness",
            "note": (
                "Semantic-interface evaluation is a secondary robustness-only path. "
                "Outputs are labeled evaluation_design='semantic_interface' and are "
                "not used for any primary confirmatory claim."
            ),
        },
    )
