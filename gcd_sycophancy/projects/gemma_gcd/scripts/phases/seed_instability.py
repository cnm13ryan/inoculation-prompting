"""``seed-instability`` phase: run seed-instability analysis on completed training+eval.

Reads:  the analysis exclusion-diagnostics CSV produced by the analysis phase
        plus the training-output checkpoints in each ``seed_<n>/`` dir.
Writes: seed-instability summary CSV, trajectory CSV, markdown report, and
        re-renders the final report; appends a phase entry to ``run_manifest.json``.
"""

from __future__ import annotations

import sys

from phases._runner_helpers import (
    PROJECTS_DIR,
    SEED_INSTABILITY_SCRIPT,
    _analysis_exclusion_diagnostics_path,
    _final_report_path,
    _record_phase,
    _run_checked,
    _seed_instability_output_prefix,
    _seed_instability_report_path,
    _seed_instability_summary_path,
    _seed_instability_trajectory_path,
    _write_final_report,
)


def run(config: RunnerConfig) -> None:
    instability_cmd = [
        sys.executable,
        str(SEED_INSTABILITY_SCRIPT),
        "--experiment-dir",
        str(config.experiment_dir),
        "--exclusion-diagnostics",
        str(_analysis_exclusion_diagnostics_path(config)),
        "--output-prefix",
        str(_seed_instability_output_prefix(config)),
        "--log-level",
        config.log_level,
    ]
    _run_checked(instability_cmd, cwd=PROJECTS_DIR)
    _write_final_report(config)
    _record_phase(
        config,
        "seed-instability",
        {
            "analysis_exclusion_diagnostics": str(_analysis_exclusion_diagnostics_path(config)),
            "seed_instability_summary": str(_seed_instability_summary_path(config)),
            "seed_instability_trajectory": str(_seed_instability_trajectory_path(config)),
            "seed_instability_report": str(_seed_instability_report_path(config)),
            "final_report": str(_final_report_path(config)),
        },
    )
