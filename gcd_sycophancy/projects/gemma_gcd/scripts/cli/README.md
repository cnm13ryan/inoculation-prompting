# `scripts/cli/` — layered CLI helpers

Stage 3 of the layered-architecture refactor (PR #100) extracted the
argparse plumbing for the four layered entrypoints into reusable flag
helpers. The unified CLI (`run_preregistration.build_parser`) keeps its
own monolithic argparse setup; this package powers `scripts/data.py`,
`scripts/train.py`, `scripts/evaluate.py`, and `scripts/analyze.py`.

For the high-level entrypoint map see [`docs/architecture.md`](../../docs/architecture.md)
and [`scripts/README.md`](../README.md).

## Public API (`cli/__init__.py`)

    from cli import (
        add_common_flags,
        add_data_flags,
        add_train_flags,
        add_eval_flags,
        add_analyze_flags,
        build_runner_config,
    )

## The five flag-helpers

Each helper takes an `argparse.ArgumentParser` and adds its layer's flags.
Helpers do their `run_preregistration` / `run_ip_sweep` imports at call
time (inside the function body) so importing `cli` itself stays cheap.

| Helper                | Module               | What it adds |
|-----------------------|----------------------|--------------|
| `add_common_flags`    | `cli/_common.py`     | `--experiment-dir`, `--log-level`, `--timestamp` |
| `add_data_flags`      | `cli/_data_flags.py` | `--data-dir`, `--seeds`, `--corpus-b-variant`, `--ip-instruction[-id]`, `--ip-placement`, `--arm-set`, `--only-arms`, `--prompt-template-variant` |
| `add_train_flags`     | `cli/_train_flags.py`| `--template-config`, `--dont-overwrite`, `--checkpoint-curve-every-steps`, `--preflight-max-final-train-loss` |
| `add_eval_flags`      | `cli/_eval_flags.py` | backend selection (`--llm-backend`, `--lmstudio-*`, `--tensor-parallel-size`, `--gpu-memory-utilization`, `--dtype`, `--max-model-len`), `--limit`, the four preflight thresholds, `--checkpoint-curve-*`, `--eval-output-subdir`, `--fixed-interface-max-format-failure-rate`, `--allow-unacceptable-fixed-interface-for-prefix-search`, plus a duplicated `--preflight-max-final-train-loss` (see below) |
| `add_analyze_flags`   | `cli/_analyze_flags.py` | currently a no-op; kept for symmetry |

### Composition by entrypoint

| Entrypoint              | Helpers composed                                            |
|-------------------------|-------------------------------------------------------------|
| `scripts/data.py`       | `add_common_flags`, `add_data_flags`                        |
| `scripts/train.py`      | `add_common_flags`, `add_data_flags`, `add_train_flags`     |
| `scripts/evaluate.py`   | `add_common_flags`, `add_data_flags`, `add_eval_flags`      |
| `scripts/analyze.py`    | `add_common_flags`, `add_data_flags`, `add_analyze_flags`   |

## `build_runner_config(args)` — the getattr-with-default pattern

`cli/_runner_config_builder.py` converts an argparse Namespace from
*any* layered parser into a full `run_preregistration.RunnerConfig`.
Because each layered parser exposes only a subset of flags, the builder
uses `getattr(args, "<field>", <default>)` for every `RunnerConfig`
field, falling back to the same constants the unified CLI uses
(`_rp.DEFAULT_*`). The resulting `RunnerConfig` is byte-equivalent to
one produced by the unified CLI for any flag both surfaces share.

If you add a new flag, mirror the new field through `RunnerConfig`,
update the helper that owns it, and add a `getattr(args, "<field>",
<default>)` line to the builder.

## Why `_eval_flags.py` duplicates `--preflight-max-final-train-loss`

Both `add_train_flags` and `add_eval_flags` register
`--preflight-max-final-train-loss`. This is intentional — see the
inline comment at the bottom of `cli/_eval_flags.py`.

`scripts/evaluate.py` routes the `preflight` phase, which (a) optionally
runs a pilot training and (b) invokes the `convergence` gate via
`_check_training_convergence`. Both code paths read
`config.preflight_max_final_train_loss`. Without this flag at the eval
layer, the convergence gate's failure message ("Rerun training for the
affected seeds (or raise `--preflight-max-final-train-loss` only if ...)")
would point to a flag the user can't actually pass through `evaluate.py`.
Same flag also lives on `train.py` for the train phase's post-train
convergence gate.

## Bootstrap-order quirk: SCRIPT_DIR first

Each layered entrypoint inserts three paths into `sys.path` before its
`import cli` line:

    SCRIPT_DIR = Path(__file__).resolve().parent       # gemma_gcd/scripts/
    GEMMA_GCD_DIR = SCRIPT_DIR.parent                   # gemma_gcd/
    PROJECTS_DIR = GEMMA_GCD_DIR.parent                 # projects/
    for candidate in (PROJECTS_DIR, GEMMA_GCD_DIR, SCRIPT_DIR):
        sys.path.insert(0, str(candidate))

The reverse order leaves `SCRIPT_DIR` at `sys.path[0]`. This is
deliberate: `gemma_gcd/data/` is an implicit namespace package and
would shadow `scripts/data.py` if `GEMMA_GCD_DIR` were searched first.
See the comment at the top of `scripts/data.py`.

## Adding a new flag

1. Decide which layer owns it. Universal? `_common.py`. Data /
   manifests? `_data_flags.py`. Training? `_train_flags.py`. Eval?
   `_eval_flags.py`. Analysis? `_analyze_flags.py` (currently a no-op).
2. Add it to the chosen helper. Use the same `default=` constant the
   unified CLI uses (import `run_preregistration as _rp` lazily and
   reference `_rp.DEFAULT_*` so the two surfaces don't drift).
3. Plumb it through `cli/_runner_config_builder.py` with a
   `getattr(args, "<field>", _rp.DEFAULT_<X>)` line. If the flag is
   needed by multiple phases that route through different layered
   entrypoints, register it in *every* helper that reaches those
   entrypoints (see `--preflight-max-final-train-loss` precedent above).
4. Mirror the field on `RunnerConfig` if it is a new dataclass field
   and update the unified CLI in `run_preregistration.py` so both
   surfaces stay in sync.
5. Add tests under `scripts/test_cli_layered.py`.
