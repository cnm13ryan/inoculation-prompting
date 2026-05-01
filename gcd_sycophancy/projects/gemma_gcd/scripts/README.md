# `scripts/` — orientation map

This directory holds the preregistration runner, the layered entrypoints,
and the supporting per-layer packages produced by the Stage 1–3
refactors. For the high-level pipeline diagram, on-disk artifact layout,
and data flow between phases, see [`../docs/architecture.md`](../docs/architecture.md).
This README is a quick map of *what is where*.

## Entrypoints

### Unified CLI

| Script                  | Purpose |
|-------------------------|---------|
| `run_preregistration.py`| Monolithic 11-phase orchestrator; CLI dispatches to a single phase or `full`. Single source of truth for `RunnerConfig`, `PHASE_REGISTRY`, and shared phase helpers. |
| `prereg.py`             | Backward-compat shim that re-exports `run_preregistration.main` so users can opt into the shorter `prereg.py` entrypoint name without changing functionality. |

### Layered entrypoints (Stage 3, PR #100)

| Script        | Phases routed                                                                                  |
|---------------|------------------------------------------------------------------------------------------------|
| `data.py`     | `materialize-data`                                                                             |
| `train.py`    | `setup`, `train`                                                                               |
| `evaluate.py` | `preflight`, `fixed-interface-eval`, `semantic-interface-eval`, `prefix-search`, `best-elicited-eval`, `checkpoint-curve-eval` |
| `analyze.py`  | `analysis`, `seed-instability` (record-deviation stays on the unified CLI only)                |

### When to use which

- **Daily ops**, full pipeline runs, `--skip-gate`, `record-deviation`,
  any flag that lives on the unified parser only → `run_preregistration.py`.
- **Targeted re-runs** of a single layer (e.g. re-evaluate without
  re-training, or re-run analysis after editing the analysis script) →
  the matching layered entrypoint. The layered surfaces compose
  `cli.add_*_flags` and convert their argparse Namespace through
  `cli.build_runner_config`, so the resulting `RunnerConfig` is
  byte-equivalent to the unified CLI's.
- **Layered entrypoints intentionally expose a subset** of unified-CLI
  flags. If a flag you need isn't there, fall back to
  `run_preregistration.py` rather than monkey-patching.

## Per-layer packages

| Directory             | Purpose | Stage / PR | README |
|-----------------------|---------|-----------|--------|
| `phases/`             | Per-phase entry points; each module exposes `run(config: RunnerConfig)`. `run_preregistration.PHASE_REGISTRY` references each via the `run_<phase>_phase` aliases. | Stage 1 / PR #98 | [`phases/README.md`](phases/README.md) |
| `gates/`              | Pass/fail decision functions (`convergence`, `fixed_interface_baseline`, `preflight`, `fixed_interface_completion`). Dispatched via `gates.run(name, config, **kwargs)`; honours `--skip-gate`. | Stage 2 / PR #99 | [`gates/README.md`](gates/README.md) |
| `cli/`                | Layered CLI helpers (`add_common_flags`, `add_data_flags`, `add_train_flags`, `add_eval_flags`, `add_analyze_flags`) and the `build_runner_config` builder used by the layered entrypoints. | Stage 3 / PR #100 | [`cli/README.md`](cli/README.md) |

## Cross-references

- Pipeline diagram, on-disk layout, phase reads/writes table:
  [`../docs/architecture.md`](../docs/architecture.md).
- Per-phase reads/writes (authoritative): the docstring at the top of
  each `phases/<name>.py` module.
- Per-gate decision logic (authoritative): the docstring at the top of
  each `gates/<name>.py` module.
- `RunnerConfig` field defaults: the `DEFAULT_*` constants near the top
  of `run_preregistration.py`.
