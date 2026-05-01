"""``seed-instability`` phase: run seed-instability analysis on completed training+eval.

Reads:  the analysis exclusion-diagnostics CSV produced by the analysis phase
        plus the training-output checkpoints in each ``seed_<n>/`` dir.
Writes: seed-instability summary CSV, trajectory CSV, markdown report, and
        re-renders the final report; appends a phase entry to ``run_manifest.json``.
"""

from __future__ import annotations

import sys



def run(config: RunnerConfig) -> None:
    import run_preregistration as _rp  # lazy: avoid circular import when run_preregistration runs as __main__
    instability_cmd = [
        sys.executable,
        str(_rp.SEED_INSTABILITY_SCRIPT),
        "--experiment-dir",
        str(config.experiment_dir),
        "--exclusion-diagnostics",
        str(_rp._analysis_exclusion_diagnostics_path(config)),
        "--output-prefix",
        str(_rp._seed_instability_output_prefix(config)),
        "--log-level",
        config.log_level,
    ]
    _rp._run_checked(instability_cmd, cwd=_rp.PROJECTS_DIR)
    _rp._write_final_report(config)
    _rp._record_phase(
        config,
        "seed-instability",
        {
            "analysis_exclusion_diagnostics": str(_rp._analysis_exclusion_diagnostics_path(config)),
            "seed_instability_summary": str(_rp._seed_instability_summary_path(config)),
            "seed_instability_trajectory": str(_rp._seed_instability_trajectory_path(config)),
            "seed_instability_report": str(_rp._seed_instability_report_path(config)),
            "final_report": str(_rp._final_report_path(config)),
        },
    )
