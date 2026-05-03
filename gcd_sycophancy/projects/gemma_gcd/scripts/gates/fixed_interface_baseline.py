"""``fixed_interface_baseline`` gate: fixed-interface format-quality gate.

Inputs:
    - ``config.fixed_interface_max_format_failure_rate`` — already used by
      the underlying baseline-report writer.
    - On disk: the fixed-interface baseline report (created or refreshed
      via the existing helper).

Outputs: :class:`GateResult` whose ``evidence`` carries the underlying
    baseline ``report`` and the ``message`` string surfaced to operators.

Failure mode: ``GateResult.passed`` is ``False`` when there are
    unacceptable assessments. The caller is responsible for emitting a
    warning.
"""

from __future__ import annotations

from phases._runner_helpers import _load_or_create_fixed_interface_baseline_report

from ._shared import GateResult, register


@register("fixed_interface_baseline")
def run(config) -> GateResult:
    report = _load_or_create_fixed_interface_baseline_report(config)
    unacceptable = report.get("unacceptable_assessments", [])
    has_unacceptable = bool(unacceptable)
    gate_passed = not has_unacceptable
    message = None
    if has_unacceptable:
        rendered = "; ".join(
            (
                f"{item['arm_slug']}/seed_{item['seed']}: "
                f"datasets={','.join(item['unacceptable_datasets'])}, "
                f"worst={item['worst_dataset']['dataset_name']} "
                f"({item['worst_dataset']['format_failure_rate']:.3f})"
            )
            for item in unacceptable[:5]
        )
        message = (
            "Fixed-interface baseline quality exceeded the configured format-failure threshold. "
            f"Failing runs: {rendered}"
        )
    return GateResult(
        name="fixed_interface_baseline",
        passed=gate_passed,
        reason=message or "Fixed-interface baseline within configured failure-rate threshold.",
        evidence={
            "report": report,
            "message": message,
            "raw_gate_passed": gate_passed,
        },
        override_used=False,
    )
