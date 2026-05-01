"""``fixed_interface_baseline`` gate: prefix-search input gate.

Inputs:
    - ``config.fixed_interface_max_format_failure_rate`` — already used by
      the underlying baseline-report writer.
    - ``config.allow_unacceptable_fixed_interface_for_prefix_search`` —
      override flag honoured by this gate.
    - On disk: the fixed-interface baseline report (created or refreshed
      via the existing helper).

Outputs: :class:`GateResult` whose ``evidence`` carries the underlying
    baseline ``report`` and the ``message`` string consumed by the
    prefix-search phase.  ``override_used`` is set when the gate decided
    "pass" only because the override flag is on.

Failure mode: ``GateResult.passed`` is ``False`` (and ``override_used``
    ``False``) when there are unacceptable assessments and no override is
    set.  When the override flag is set, the gate still reports
    ``passed=True`` but with ``override_used=True`` and a populated
    ``message``; the caller is responsible for emitting a warning.
"""

from __future__ import annotations

from ._shared import GateResult, register


@register("fixed_interface_baseline")
def run(config) -> GateResult:
    import run_preregistration as _rp

    report = _rp._load_or_create_fixed_interface_baseline_report(config)
    unacceptable = report.get("unacceptable_assessments", [])
    has_unacceptable = bool(unacceptable)
    override_used = bool(
        has_unacceptable and config.allow_unacceptable_fixed_interface_for_prefix_search
    )
    gate_passed = (not has_unacceptable) or override_used
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
            "Fixed-interface baseline quality is unacceptable for bounded-search interpretation. "
            "Bounded prefix search should not function as the repair path for a broken fixed interface. "
            f"Failing runs: {rendered}"
        )
    return GateResult(
        name="fixed_interface_baseline",
        passed=gate_passed,
        reason=message or "Fixed-interface baseline within configured failure-rate threshold.",
        evidence={
            "report": report,
            "message": message,
            # ``raw_gate_passed`` reproduces the legacy ``gate_passed`` field,
            # which reflects strictly "no unacceptable assessments" — distinct
            # from ``GateResult.passed`` which can be True via override.
            "raw_gate_passed": not has_unacceptable,
        },
        override_used=override_used,
    )
