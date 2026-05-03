"""``fixed-interface-eval`` phase: run primary fixed-interface (XML) evaluation
across all arms for the H1-H5 confirmatory analyses.

Reads:  trained adapter checkpoints under ``seed_<n>/results/<ts>/``.
Writes: per-arm-per-seed eval outputs + the fixed-interface baseline report
        under ``reports/``; appends a phase entry to ``run_manifest.json``.
"""

from __future__ import annotations

import sys

from phases._runner_helpers import (
    FIXED_EVAL_SCRIPT,
    PROJECTS_DIR,
    PTST_ARM_SLUG,
    _evaluation_common_args,
    _fixed_interface_baseline_report_path,
    _fixed_interface_output_dir,
    _has_results,
    _iter_arm_condition_dirs,
    _record_phase,
    _require_frozen_manifests,
    _run_checked,
    _validate_seed_configs_exist,
    _validate_training_outputs,
    _write_fixed_interface_baseline_report,
    arms_for_arm_set,
)


def run(config: RunnerConfig) -> None:
    from gates import run as run_gate
    _require_frozen_manifests(config)
    model_paths = _validate_training_outputs(config)
    condition_dirs = _validate_seed_configs_exist(config)
    evaluated_arms = arms_for_arm_set(config.arm_set)
    for arm, condition_dir in _iter_arm_condition_dirs(
        config, condition_dirs, scope="all"
    ):
        for seed in config.seeds:
            output_dir = _fixed_interface_output_dir(config, condition_dir, seed)
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
                "--include-capability-diagnostics",
                "--prompt-template-variant",
                config.prompt_template_variant,
                "--scoring-parser",
                config.scoring_parser,
                *_evaluation_common_args(config),
            ]
            _run_checked(cmd, cwd=PROJECTS_DIR)
    completion_result = run_gate("fixed_interface_completion", config)
    if not completion_result.passed:
        raise RuntimeError(completion_result.reason)
    baseline_report = _write_fixed_interface_baseline_report(config)
    _record_phase(
        config,
        "fixed-interface-eval",
        {
            "evaluated_arms": len(evaluated_arms),
            "arm_set": config.arm_set,
            "seed_count_per_arm": len(config.seeds),
            "baseline_report": str(_fixed_interface_baseline_report_path(config)),
            "acceptable_assessments": baseline_report["summary"]["acceptable_assessments"],
            "unacceptable_assessments": baseline_report["summary"]["unacceptable_assessments"],
        },
    )
