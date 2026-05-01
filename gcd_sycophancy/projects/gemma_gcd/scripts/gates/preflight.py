"""``preflight`` gate: confirmatory pilot quality-gate decision.

Inputs:
    - ``report``: the dict produced by
      ``run_preregistration._make_preflight_report``.  The dict already
      contains a ``failures`` list and a ``passed`` flag — the gate's job
      is to surface that decision through the registry interface so it can
      be skipped via ``--skip-gate preflight``.
    - ``config`` is unused by the decision logic itself but is part of the
      registry contract.

Outputs: :class:`GateResult` whose ``evidence["report"]`` echoes the
    incoming report and ``evidence["failures"]`` lists each criterion that
    fired.  Reports are written by ``phases/preflight.py`` — the gate
    explicitly does NOT duplicate that side effect.

Failure mode: ``GateResult.passed`` is ``False`` when the report has any
    failures (i.e. ``not report["passed"]``).  The phase caller raises
    ``RuntimeError`` and points operators at the report on disk.
"""

from __future__ import annotations

from typing import Any

from ._shared import GateResult, register


@register("preflight")
def run(config, *, report: dict[str, Any]) -> GateResult:
    failures = report.get("failures", [])
    passed = bool(report.get("passed", not failures))
    if passed:
        reason = "Preflight quality gate criteria met."
    else:
        rendered = "; ".join(
            f"{f.get('criterion', '?')}: {f.get('message', '')}".strip()
            for f in failures
        )
        reason = (
            f"Preflight gate failed ({len(failures)} criterion failure"
            f"{'s' if len(failures) != 1 else ''}). {rendered}"
        )
    return GateResult(
        name="preflight",
        passed=passed,
        reason=reason,
        evidence={
            "failures": failures,
            "report": report,
        },
    )
