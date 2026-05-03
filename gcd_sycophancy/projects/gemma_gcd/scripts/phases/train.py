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

from phases._runner_helpers import _run_training_phase


def run(config: RunnerConfig) -> None:
    from gates import run as run_gate
    _run_training_phase(config, phase_name="train")
    result = run_gate("convergence", config)
    if not result.passed:
        raise RuntimeError(result.reason)
