"""``fixed_interface_completion`` gate: coverage check before downstream phases.

Inputs:
    - ``config.seeds``, ``config.only_arms``, ``config.arm_set`` — drive
      which (arm, seed) pairs are required.
    - On disk: ``<fixed_interface_output_dir>`` for each pair must contain
      results (delegated to ``_has_results``).

Outputs: :class:`GateResult` whose ``evidence["missing"]`` lists the
    output directories that lack results, capped at five in ``reason`` to
    match legacy formatting.

Failure mode: ``GateResult.passed`` is ``False`` when any required
    directory has no results.  The legacy alias
    ``_require_fixed_interface_phase_completed`` continues to raise
    ``RuntimeError`` so that callers depending on the raise behaviour are
    unaffected.
"""

from __future__ import annotations

from ._shared import GateResult, register


@register("fixed_interface_completion")
def run(config) -> GateResult:
    import run_preregistration as _rp

    condition_dirs = _rp._validate_seed_configs_exist(config)
    selected = set(_rp._select_only_arm_slugs(config, list(condition_dirs.keys())))
    missing: list[str] = []
    for slug, condition_dir in condition_dirs.items():
        if slug not in selected:
            continue
        for seed in config.seeds:
            output_dir = _rp._fixed_interface_output_dir(config, condition_dir, seed)
            if not _rp._has_results(output_dir):
                missing.append(str(output_dir))
    if missing:
        rendered = "; ".join(missing[:5])
        reason = (
            "Fixed-interface evaluation artifacts are required before bounded prefix search. "
            f"Missing: {rendered}"
        )
        return GateResult(
            name="fixed_interface_completion",
            passed=False,
            reason=reason,
            evidence={"missing": missing},
        )
    return GateResult(
        name="fixed_interface_completion",
        passed=True,
        reason="Fixed-interface evaluation artifacts exist for all required (arm, seed) pairs.",
        evidence={"missing": []},
    )
