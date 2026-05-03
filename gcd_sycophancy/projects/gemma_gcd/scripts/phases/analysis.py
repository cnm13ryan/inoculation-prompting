"""``analysis`` phase: H1-H5 confirmatory analysis + final-report rendering.

Reads:  per-arm-per-seed eval outputs from ``fixed-interface-eval`` and
        ``best-elicited-eval``, the deviations log, and the frozen prefix
        artifacts from ``prefix-search``.
Writes: problem-level export CSV, prereg analysis JSON + summary, exclusion
        diagnostics CSVs, the final report markdown, plus invokes the
        ``seed-instability`` phase as a sub-step. Appends a phase entry to
        ``run_manifest.json``.
"""

from __future__ import annotations

import sys

from phases._runner_helpers import (
    ANALYSIS_SCRIPT,
    EXPORT_SCRIPT,
    PROJECTS_DIR,
    _analysis_exclusion_categories_path,
    _analysis_exclusion_diagnostics_path,
    _analysis_json_path,
    _analysis_output_prefix,
    _analysis_summary_path,
    _deviations_log_path,
    _final_report_path,
    _problem_level_export_path,
    _record_phase,
    _require_analysis_inputs,
    _require_frozen_manifests,
    _run_checked,
    _write_final_report,
    run_seed_instability_phase,
)


def run(config: RunnerConfig) -> None:
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
    run_seed_instability_phase(config)
    _write_final_report(config)
    _record_phase(
        config,
        "analysis",
        {
            "problem_level_export": str(_problem_level_export_path(config)),
            "analysis_json": str(_analysis_json_path(config)),
            "analysis_summary": str(_analysis_summary_path(config)),
            "analysis_exclusion_diagnostics": str(_analysis_exclusion_diagnostics_path(config)),
            "analysis_exclusion_categories": str(_analysis_exclusion_categories_path(config)),
            "final_report": str(_final_report_path(config)),
            "deviations_log": str(_deviations_log_path(config)),
        },
    )
