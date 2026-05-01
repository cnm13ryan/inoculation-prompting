"""``train`` phase: launch per-arm-per-seed multi-seed training, then run the
training-convergence gate.

Reads:  per-experiment arms dir, per-seed configs.
Writes: ``seed_<n>/results/<ts>/adapter_model.safetensors`` per trainable arm,
        plus PTST shared-training reference JSONs that point to the neutral
        arm's checkpoint (PTST reuses neutral); appends a phase entry to
        ``run_manifest.json``. Raises ``RuntimeError`` if any seed's final
        train loss exceeds ``--preflight-max-final-train-loss``.

Note: the actual training-launch helper ``_run_training_phase`` is shared
with the ``preflight`` phase and remains defined in ``run_preregistration``.
This module is the train-phase entry point only.
"""

from __future__ import annotations



def run(config: RunnerConfig) -> None:
    import run_preregistration as _rp  # lazy: avoid circular import when run_preregistration runs as __main__
    from gates import run as run_gate
    _rp._run_training_phase(config, phase_name="train")
    result = run_gate("convergence", config)
    if not result.passed:
        raise RuntimeError(result.reason)
