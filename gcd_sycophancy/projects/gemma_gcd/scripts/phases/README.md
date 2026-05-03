# `scripts/phases/` — per-phase entry-point package

Stage 1 of the layered-architecture refactor (PR #98) extracted the body of
each preregistration phase out of `run_preregistration.py` into its own
module under this package.

For the high-level pipeline diagram (data flow between phases, on-disk
layout) see [`docs/architecture.md`](../../docs/architecture.md). The
module docstring inside each `phases/<name>.py` file is authoritative for
the phase's reads/writes; this README is a quick map.

## Contract

Every module in this package exposes a single entry point with the
signature:

    def run(config: RunnerConfig) -> None  # or -> dict for preflight

`config` is a `run_preregistration.RunnerConfig`. Phases that need extra
keyword args (`tokenizer=` for `setup`) accept them as kwargs.

## The 11 phases

`PHASE_REGISTRY` lives in `run_preregistration.py` and is the single
source of truth for phase ordering and `in_full` membership.
Phase-function bodies live in this package; the docstring at the top of
each `phases/<name>.py` documents inputs and outputs.

| # | Phase (CLI subcommand) | Module | `in_full` |
|---|---|---|---|
| 1 | `materialize-data`        | `phases/materialize_data.py`        | no  |
| 2 | `setup`                   | `phases/setup.py`                   | yes |
| 3 | `preflight`               | `phases/preflight.py`               | yes |
| 4 | `train`                   | `phases/train.py`                   | yes |
| 5 | `fixed-interface-eval`    | `phases/fixed_interface_eval.py`    | yes |
| 6 | `semantic-interface-eval` | `phases/semantic_interface_eval.py` | no  |
| 7 | `prefix-search`           | `phases/prefix_search.py`           | yes |
| 8 | `best-elicited-eval`      | `phases/best_elicited_eval.py`      | yes |
| 9 | `analysis`                | `phases/analysis.py`                | yes |
| 10 | `seed-instability`       | `phases/seed_instability.py`        | no  |
| 11 | `checkpoint-curve-eval`  | `phases/checkpoint_curve_eval.py`   | no  |

Two CLI-only pseudo-phases (`full`, `record-deviation`) appear in
`PHASE_REGISTRY` with `runner=None` and are dispatched specially in
`run_preregistration.main`.

## `phases._runner_helpers` sibling-facade pattern

Phase modules import shared helpers (e.g. `_record_phase`,
`_run_checked`, `PROJECTS_DIR`) from the sibling module
`phases._runner_helpers`, NOT from `run_preregistration` directly:

    from phases._runner_helpers import _record_phase, _run_checked

    def run(config: RunnerConfig) -> None:
        _run_checked(cmd, cwd=PROJECTS_DIR)
        _record_phase(config, "setup", outputs)

The sibling forwards each lookup to `run_preregistration` lazily (proxy
objects that resolve on first call), which is necessary because
`run_preregistration` is the script the runner is *executed as*
(`__main__`) and imports phase modules eagerly. Eagerly importing
`run_preregistration` from a phase module would deadlock at module-load
time. The sibling-facade pattern keeps the actual import lazy while
giving phase code a stable sideways dependency to import from. Test
monkeypatching is preserved because the proxies re-resolve on every
call: `monkeypatch.setattr(run_preregistration, "_run_checked", ...)`
still affects subsequent phase calls.

Gates follow the same convention.

Adding a new helper that phase / gate code needs to call: add its name
to the `_HELPER_NAMES` tuple in `phases/_runner_helpers.py` so the
proxy is exposed.

## Legacy aliases on `run_preregistration`

`run_preregistration.py` re-exports each phase's `run` as
`run_<phase>_phase` so the legacy symbol path keeps working:

| Alias                              | Source                                  |
|------------------------------------|-----------------------------------------|
| `run_materialize_data_phase`       | `phases.materialize_data.run`           |
| `run_setup_phase`                  | `phases.setup.run`                      |
| `run_training_phase`               | `phases.train.run`                      |
| `run_fixed_interface_eval_phase`   | `phases.fixed_interface_eval.run`       |
| `run_semantic_interface_eval_phase`| `phases.semantic_interface_eval.run`    |
| `run_preflight_phase`              | `phases.preflight.run`                  |
| `run_analysis_phase`               | `phases.analysis.run`                   |
| `run_seed_instability_phase`       | `phases.seed_instability.run`           |
| `run_checkpoint_curve_eval_phase`  | `phases.checkpoint_curve_eval.run`      |

These aliases are how `PHASE_REGISTRY` references the runners; do not
remove them without updating the registry.

## Adding a new phase

1. Create `phases/<new>.py` with a header docstring documenting reads/writes
   plus `def run(config: RunnerConfig) -> None`. Import any helpers from
   `phases._runner_helpers` (add names to its `_HELPER_NAMES` tuple if
   they are not already exposed).
2. Re-export it in `run_preregistration.py` next to the other
   `from phases.<x> import run as run_<x>_phase  # noqa: E402` lines.
3. Add a `PhaseSpec("<new>", run_<new>_phase, in_full=...)` entry to
   `PHASE_REGISTRY`.
4. If the phase should be exposed by a layered entrypoint
   (`scripts/data.py`, `train.py`, `evaluate.py`, `analyze.py`), add it
   to that entrypoint's `PHASES` dict. The unified CLI gets the new
   phase automatically via the registry.
5. Add tests under `scripts/test_run_preregistration.py` (or a new
   dedicated test module). Follow the monkeypatch pattern used by
   existing phase tests.

## What NOT to do

- Don't add module-level imports of `run_preregistration`. Use the
  `phases._runner_helpers` sibling-facade for the reasons above.
- Don't move phase-shared helpers (e.g. `_run_training_phase`,
  `_record_phase`, `_validate_seed_configs_exist`) into this package.
  They live in `run_preregistration.py` so test monkeypatches keep
  working; the sibling facade exposes them via lazy proxies that
  re-resolve on every call.
- Don't duplicate report-writing or gate-decision logic. Reports stay in
  the phase that writes them; gate decisions live in `scripts/gates/`.
