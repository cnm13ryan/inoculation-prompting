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

Note: gate modules import shared runner helpers from the sibling facade
``phases._runner_helpers`` rather than importing ``run_preregistration``
directly.  The facade resolves each helper lazily (proxy on first call)
because ``run_preregistration`` is the script the runner is executed as
(``__main__``); importing it at module load would deadlock.
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
from ._config import (
    GATES_YAML_BASENAME,
    YAML_TO_RUNNER_FIELD,
    apply_to_runner_config_kwargs,
    load_gate_config,
)
from ._shared import GateResult, is_registered, registered_gates, run

GATE_NAMES: tuple[str, ...] = (
    "convergence",
    "fixed_interface_baseline",
    "preflight",
    "fixed_interface_completion",
)

__all__ = [
    "GATES_YAML_BASENAME",
    "GATE_NAMES",
    "GateResult",
    "YAML_TO_RUNNER_FIELD",
    "apply_to_runner_config_kwargs",
    "is_registered",
    "load_gate_config",
    "registered_gates",
    "run",
]
