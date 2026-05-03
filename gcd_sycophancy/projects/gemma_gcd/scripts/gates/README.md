# `scripts/gates/` — pass/fail decision functions

Stage 2 of the layered-architecture refactor (PR #99) extracted the
gating decisions previously inlined in `run_preregistration` and the
`phases` package into individually reviewable gate modules.

For the high-level pipeline diagram see [`docs/architecture.md`](../../docs/architecture.md).
The package-level docstring at `gates/__init__.py` and the per-gate
module docstrings are authoritative for behaviour; this README is a
quick map.

## `GateResult` contract

Defined in `gates/_shared.py`:

    @dataclass(frozen=True)
    class GateResult:
        name: str                 # registry key, e.g. "convergence"
        passed: bool              # True iff the gate accepts the run
        reason: str               # human-readable explanation
        evidence: dict[str, Any]  # paths/counts/sub-reports for callers
        override_used: bool       # True iff pass relied on an override flag

A gate is purely a decision function: it returns a `GateResult`. The
*caller* is responsible for raising on failure (`if not result.passed:
raise RuntimeError(result.reason)`). Side effects like report-writing
stay in the phase that writes them.

Dispatch is `gates.run(name, config, **kwargs) -> GateResult`. The
registry is populated by `@register("name")` decorators on each gate's
`run` function and is exported as `registered_gates()`.

## The four registered gates

`GATE_NAMES` lives at `gates/__init__.py`.

### `convergence` — `gates/convergence.py`
- Checks: every (arm, seed) pair's `train_losses[-1]` is below
  `config.preflight_max_final_train_loss`.
- Fires from: `phases/train.py` (post-train) and `phases/preflight.py`
  (after the optional pilot training).
- Failure mode: `passed=False`; `reason` lists each offending seed with
  initial→final loss; `evidence={"bad_seeds": [...], "threshold": ...}`.
  The phase caller raises `RuntimeError(result.reason)`.
- Legacy alias: `run_preregistration._check_training_convergence` —
  preserved as a monkeypatch surface and still raises on failure.

### `fixed_interface_baseline` — `gates/fixed_interface_baseline.py`
- Checks: the fixed-interface baseline report has no
  `unacceptable_assessments` (or the
  `--allow-unacceptable-fixed-interface-for-prefix-search` override is
  set).
- Fires from: `phases/prefix_search.py` (input gate before bounded search).
- Failure mode: `passed=False` and `override_used=False` when there are
  unacceptable assessments and the override is off. With the override on,
  `passed=True`, `override_used=True`, `evidence["message"]` is populated
  so the caller can emit a warning. `evidence["raw_gate_passed"]`
  preserves the legacy "no unacceptable" semantics independent of the
  override.
- Legacy alias: `run_preregistration._prefix_search_gate_status` —
  unpacks the `GateResult` into the historical status-dict shape.

### `preflight` — `gates/preflight.py`
- Checks: a preflight `report` dict (already produced by
  `run_preregistration._make_preflight_report`) has no `failures`.
- Fires from: `phases/preflight.py` after the report is written.
- Failure mode: `passed=False`; `reason` enumerates the failed criteria;
  `evidence={"failures": [...], "report": report}`. The phase caller
  raises `RuntimeError` and points operators at the report on disk.
- Legacy alias: none — the gate is invoked directly.

### `fixed_interface_completion` — `gates/fixed_interface_completion.py`
- Checks: every required `<fixed_interface_output_dir>` for the selected
  (arm, seed) pairs contains results (delegates to `_has_results`).
- Fires from: `run_preregistration._require_fixed_interface_phase_completed`
  before downstream phases that depend on those artifacts.
- Failure mode: `passed=False`; `evidence["missing"]` lists the directories
  that lack results (capped at five in `reason`).
- Legacy alias: `run_preregistration._require_fixed_interface_phase_completed`
  — still raises `RuntimeError`.

## `--skip-gate` and the synthetic-pass short-circuit

The unified CLI exposes `--skip-gate <name>` (repeatable) at
`run_preregistration.py` with `choices` matching `GATE_NAMES`. Skipped
names land in `RunnerConfig.skip_gates` and are read by
`gates._shared.run`:

    skip_gates = getattr(config, "skip_gates", ()) or ()
    if name in skip_gates:
        return GateResult(name, passed=True,
                          reason=f"skipped via --skip-gate {name}",
                          evidence={"skipped": True})

The gate body never executes when skipped. Callers that previously read
fields directly off `evidence` must handle the skip case.

### Skip-path quirk for `fixed_interface_baseline`

When `fixed_interface_baseline` is skipped, the synthetic `GateResult`
carries `evidence={"skipped": True}` only — there is no `report`,
`raw_gate_passed`, or `message`. The caller's status-dict alias must
handle this branch explicitly. See
`run_preregistration._prefix_search_gate_status`, which detects the skip
and returns a legacy-shape dict whose `report` is `None` and whose
`skipped: True` flag tells the prefix-search phase it has no
per-(arm,seed) baseline assessments to annotate frozen prefix artifacts
with. (Fixed in the PR #99 follow-up.)

## Adding a new gate

1. Create `gates/<new>.py`. Import `GateResult, register` from
   `._shared`, decorate the function with `@register("<new>")`, and
   keep the body a pure decision: read inputs, return a `GateResult`,
   no side effects. Import shared runner helpers (e.g.
   `_validate_seed_configs_exist`, `_read_json`) from the sibling
   facade `phases._runner_helpers`; do not import `run_preregistration`
   directly.
2. Add the module to the import block in `gates/__init__.py` so the
   `@register` side effect runs at package import.
3. Add the name to `GATE_NAMES` in `gates/__init__.py`.
4. Add the name to the `--skip-gate` `choices` tuple in
   `run_preregistration.py`.
5. Add tests under `scripts/test_gates.py`. Cover: passing case, failing
   case, skip via `config.skip_gates`, and (if relevant) override-flag
   behaviour.
