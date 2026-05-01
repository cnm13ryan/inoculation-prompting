"""``preflight`` phase: small-scale pilot training + evaluation that gates the
full preregistered run.

Reads:  arm dirs and seed configs (created by ``setup``).
Writes: pilot training outputs (or reuses existing), pilot eval outputs under
        ``seed_<n>/preflight/``, the preflight report + summary; appends a
        phase entry to ``run_manifest.json``. Raises ``RuntimeError`` if
        the preflight quality gate fails.
"""

from __future__ import annotations

import sys
from typing import Any

from evaluate_base_model import (
    compute_fixed_interface_quality_summary,
    load_eval_result_summaries,
)


def run(config: RunnerConfig) -> dict[str, Any]:
    import run_preregistration as _rp  # lazy: avoid circular import when run_preregistration runs as __main__
    from gates import run as run_gate
    pilot_config = _rp._preflight_config(config)
    _rp._require_frozen_manifests(pilot_config)
    condition_dirs = _rp._validate_seed_configs_exist(pilot_config)
    preflight_seeds = pilot_config.seeds
    try:
        model_paths = _rp._validate_training_outputs(pilot_config)
        preflight_training = {
            "phase": "reused_existing_training_outputs",
            "seeds": list(preflight_seeds),
        }
    except RuntimeError:
        preflight_training = _rp._run_training_phase(
            pilot_config,
            phase_name="preflight-train",
        )
        model_paths = _rp._validate_training_outputs(pilot_config)
    convergence_result = run_gate("convergence", pilot_config)
    if not convergence_result.passed:
        raise RuntimeError(convergence_result.reason)
    quality_assessments: list[dict[str, Any]] = []

    for arm, condition_dir in _rp._iter_arm_condition_dirs(
        pilot_config, condition_dirs, scope="confirmatory"
    ):
        for seed in preflight_seeds:
            output_dir = _rp._preflight_output_dir(pilot_config, condition_dir, seed)
            if not _rp._has_results(output_dir):
                evaluation_mode = "ptst" if arm.slug == _rp.PTST_ARM_SLUG else "neutral"
                cmd = [
                    sys.executable,
                    str(_rp.FIXED_EVAL_SCRIPT),
                    "--model-name",
                    str(model_paths[arm.slug][seed]),
                    "--evaluation-mode",
                    evaluation_mode,
                    "--output-dir",
                    str(output_dir),
                    "--datasets",
                    _rp.DEFAULT_PREFLIGHT_DATASET,
                    "--limit",
                    str(pilot_config.preflight_limit),
                    *_rp._evaluation_common_args(pilot_config),
                ]
                _rp._run_checked(cmd, cwd=_rp.PROJECTS_DIR)
            model_dir = _rp._latest_eval_model_dir(output_dir)
            eval_summaries = load_eval_result_summaries(model_dir)
            assessment = compute_fixed_interface_quality_summary(
                eval_summaries,
                max_format_failure_rate=pilot_config.fixed_interface_max_format_failure_rate,
            )
            assessment.update(
                {
                    "arm_slug": arm.slug,
                    "seed": seed,
                    "output_dir": str(output_dir),
                    "model_dir": str(model_dir),
                }
            )
            quality_assessments.append(assessment)

    preflight_df = _rp._collect_preflight_rows(pilot_config, condition_dirs, preflight_seeds)
    report = _rp._make_preflight_report(pilot_config, preflight_df, quality_assessments)
    report["preflight_training"] = preflight_training
    _rp._write_json(_rp._preflight_report_path(config), report)
    _rp._write_preflight_summary(config, report)
    _rp._record_phase(
        config,
        "preflight",
        {
            "passed": report["passed"],
            "pilot_seeds": list(preflight_seeds),
            "preflight_training_phase": preflight_training["phase"],
            "report": str(_rp._preflight_report_path(config)),
            "summary": str(_rp._preflight_summary_path(config)),
            "problem_level_export": str(_rp._preflight_export_path(config)),
        },
    )
    preflight_result = run_gate("preflight", config, report=report)
    if not preflight_result.passed:
        raise RuntimeError(
            "Preflight gate failed. Inspect "
            f"{_rp._preflight_report_path(config)} or {_rp._preflight_summary_path(config)}."
        )
    return report
