"""``materialize-data`` phase: validate and freeze the source data manifest.

Reads:  the source data manifest under ``config.data_dir``.
Writes: a frozen copy under ``<experiment_dir>/manifests/`` and an empty
        deviations log; appends a phase entry to ``run_manifest.json``.
"""

from __future__ import annotations

from phases._runner_helpers import (
    _ensure_deviations_log_exists,
    _ensure_prereq_scripts_exist,
    _record_phase,
    _validate_and_freeze_data_manifest,
)


def run(config: RunnerConfig) -> None:
    _ensure_prereq_scripts_exist()
    outputs = _validate_and_freeze_data_manifest(config)
    _ensure_deviations_log_exists(config)
    _record_phase(config, "materialize-data", outputs)
