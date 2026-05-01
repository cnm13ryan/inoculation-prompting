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



def run(config: RunnerConfig) -> None:
    import run_preregistration as _rp  # lazy: avoid circular import when run_preregistration runs as __main__
    _rp._require_frozen_manifests(config)
    _rp._require_analysis_inputs(config)
    export_cmd = [
        sys.executable,
        str(_rp.EXPORT_SCRIPT),
        "--experiments_dir",
        str(config.experiment_dir),
        "--output",
        str(_rp._problem_level_export_path(config)),
    ]
    _rp._run_checked(export_cmd, cwd=_rp.PROJECTS_DIR)
    analysis_cmd = [
        sys.executable,
        str(_rp.ANALYSIS_SCRIPT),
        "--input",
        str(_rp._problem_level_export_path(config)),
        "--output-prefix",
        str(_rp._analysis_output_prefix(config)),
        "--log-level",
        config.log_level,
    ]
    _rp._run_checked(analysis_cmd, cwd=_rp.PROJECTS_DIR)
    _rp.run_seed_instability_phase(config)
    _rp._write_final_report(config)
    _rp._record_phase(
        config,
        "analysis",
        {
            "problem_level_export": str(_rp._problem_level_export_path(config)),
            "analysis_json": str(_rp._analysis_json_path(config)),
            "analysis_summary": str(_rp._analysis_summary_path(config)),
            "analysis_exclusion_diagnostics": str(_rp._analysis_exclusion_diagnostics_path(config)),
            "analysis_exclusion_categories": str(_rp._analysis_exclusion_categories_path(config)),
            "final_report": str(_rp._final_report_path(config)),
            "deviations_log": str(_rp._deviations_log_path(config)),
        },
    )
