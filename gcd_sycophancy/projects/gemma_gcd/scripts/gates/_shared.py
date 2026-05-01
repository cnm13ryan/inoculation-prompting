"""Shared types and registry for the ``gates`` package.

A *gate* is a small, individually-reviewable function that consumes a
``RunnerConfig`` (plus any keyword arguments a particular caller needs) and
returns a :class:`GateResult` summarising the pass/fail decision and any
structured evidence (paths, counts, sub-reports) that callers may want to
emit.

Gates do not duplicate side effects that already live elsewhere: the actual
report-writing for the preflight gate, for example, still happens in
``phases/preflight.py``; the gate simply reads the report and decides.

The registry is populated via the ``@register("name")`` decorator on each
gate's ``run`` function (imported eagerly from :mod:`gates.__init__`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class GateResult:
    """Outcome of evaluating a gate.

    Attributes
    ----------
    name:
        Registered name of the gate (e.g. ``"convergence"``).
    passed:
        ``True`` iff the gate accepts the current run.  Callers that want to
        abort on failure should ``raise RuntimeError(result.reason)``
        themselves; the gate body is purely a decision function.
    reason:
        Human-readable explanation.  Empty / boilerplate when ``passed`` is
        ``True``; populated with a diagnostic message when ``passed`` is
        ``False`` (or when the gate was skipped via ``--skip-gate``).
    evidence:
        Arbitrary structured evidence — paths, counts, sub-reports — useful
        for callers that previously read fields off a status dict (e.g. the
        prefix-search caller wants the underlying baseline report).
    override_used:
        ``True`` when the gate's pass decision relied on an explicit override
        flag (e.g. ``--allow-unacceptable-fixed-interface-for-prefix-search``).
        Distinguishes "passed cleanly" from "passed with caveat".
    """

    name: str
    passed: bool
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)
    override_used: bool = False


GateFn = Callable[..., GateResult]
_GATES: dict[str, GateFn] = {}


def register(name: str) -> Callable[[GateFn], GateFn]:
    """Decorator that adds ``fn`` to the gate registry under ``name``."""

    def deco(fn: GateFn) -> GateFn:
        _GATES[name] = fn
        return fn

    return deco


def registered_gates() -> tuple[str, ...]:
    """Names of all currently-registered gates (sorted)."""

    return tuple(sorted(_GATES))


def is_registered(name: str) -> bool:
    return name in _GATES


def run(name: str, config, **kwargs) -> GateResult:
    """Dispatch to the named gate, honouring ``config.skip_gates``.

    If ``name`` is in ``config.skip_gates`` (when that attribute exists), the
    gate body is not executed; we instead return a synthetic ``GateResult``
    with ``passed=True`` and ``reason`` indicating the skip.  This is the
    user-visible payoff of the Stage-2 gate refactor: a CLI ``--skip-gate``
    flag lets operators bypass an individual gate without code changes.
    """

    skip_gates = getattr(config, "skip_gates", ()) or ()
    if name in skip_gates:
        return GateResult(
            name=name,
            passed=True,
            reason=f"skipped via --skip-gate {name}",
            evidence={"skipped": True},
        )
    try:
        gate_fn = _GATES[name]
    except KeyError as exc:
        raise KeyError(
            f"Gate {name!r} is not registered. "
            f"Known gates: {registered_gates()}"
        ) from exc
    return gate_fn(config, **kwargs)


__all__ = [
    "GateResult",
    "register",
    "registered_gates",
    "is_registered",
    "run",
]
