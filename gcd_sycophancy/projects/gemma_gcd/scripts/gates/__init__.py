"""``gates`` package: gating-decision functions for the prereg runner.

This package extracts pass/fail decisions previously embedded in
``run_preregistration`` and the ``phases`` package into individually
reviewable gate modules.  Each gate exposes ``run(config, **kwargs) ->
GateResult`` and is registered in :mod:`gates._shared`'s registry.

Public API::

    from gates import run as run_gate, GateResult, registered_gates

    result = run_gate("convergence", config)
    if not result.passed:
        raise RuntimeError(result.reason)

The Stage-2 ``--skip-gate`` CLI flag plumbs through ``RunnerConfig.skip_gates``
into :func:`gates._shared.run`, which short-circuits with a synthetic
"passed=True, reason=skipped..." result when the gate name is in the skip
set.  This means callers don't need to special-case skipping themselves —
they always go through ``gates.run("name", config, ...)``.

Note: gate modules import ``run_preregistration`` lazily from inside their
``run`` bodies because ``run_preregistration`` is the script the runner is
executed as (``__main__``), and importing it at module load would deadlock.
"""

from __future__ import annotations

# Importing these modules registers their @register("...") decorators with
# the shared registry as a side effect, so callers only need to import this
# package's public surface (``run``, ``GateResult``).
from . import (
    convergence as _convergence,  # noqa: F401  (registration side effect)
    fixed_interface_baseline as _fixed_interface_baseline,  # noqa: F401
    fixed_interface_completion as _fixed_interface_completion,  # noqa: F401
    preflight as _preflight,  # noqa: F401
)
from ._shared import GateResult, is_registered, registered_gates, run

GATE_NAMES: tuple[str, ...] = (
    "convergence",
    "fixed_interface_baseline",
    "preflight",
    "fixed_interface_completion",
)

__all__ = [
    "GATE_NAMES",
    "GateResult",
    "is_registered",
    "registered_gates",
    "run",
]
