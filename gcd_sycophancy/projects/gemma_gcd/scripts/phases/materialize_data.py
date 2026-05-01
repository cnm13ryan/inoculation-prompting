"""``materialize-data`` phase: validate and freeze the source data manifest.

Reads:  the source data manifest under ``config.data_dir``.
Writes: a frozen copy under ``<experiment_dir>/manifests/`` and an empty
        deviations log; appends a phase entry to ``run_manifest.json``.
"""

from __future__ import annotations



def run(config: RunnerConfig) -> None:
    import run_preregistration as _rp  # lazy: avoid circular import when run_preregistration runs as __main__
    _rp._ensure_prereq_scripts_exist()
    outputs = _rp._validate_and_freeze_data_manifest(config)
    _rp._ensure_deviations_log_exists(config)
    _rp._record_phase(config, "materialize-data", outputs)
