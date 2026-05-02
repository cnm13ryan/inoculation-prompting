# `experiments/` — orientation map

This directory holds inputs, intermediate artefacts, run outputs, and figures
for the GCD-sycophancy preregistration work. Some subdirectories are tracked
(catalogs, selection results, per-variant READMEs, prereg spec) and others are
runtime-generated and gitignored (panel runs, logs, plots, base-model
artefacts). Use this README as the entrypoint; deeper detail lives in the
per-directory READMEs linked below.

## Two orthogonal axes

Most artefacts here cross-cut two axes — keep them straight when navigating:

- **Content axis (corpus / labels):** B1 = original sycophancy labels, B2 =
  semantic-equivalent labels. Independent of how prompts are structured.
- **Interface axis (prompt structure):** XML vs semantic — how the user message
  and IP are wrapped and rendered. Independent of the labels.

A run is described by the combination (e.g. `B2 × semantic`), not just one
axis. The IP **placement** (`prepend` vs `append`) and **wording** (`above`
vs `below`) is a third axis specific to inoculation-prompt experiments.

## Where to start

| You want to…                                                | Go to                                                         |
|-------------------------------------------------------------|---------------------------------------------------------------|
| Understand the preregistered protocol                       | [`prereg/`](prereg/)                                          |
| Pick or audit a training-time inoculation prompt            | [`ip_sweep/`](ip_sweep/) (`train_user_suffix_*` files)        |
| Pick or audit an evaluation-time pressure suffix            | [`ip_sweep/`](ip_sweep/) (`eval_pressure_suffix_*` files)     |
| Read the placement story (prepend vs append, above vs below)| [`prepend_below/`](prepend_below/), [`append_above/`](append_above/), [`_legacy_prepend_above/`](_legacy_prepend_above/) |
| Find existing trained models / panel runs                   | [`_legacy_prepend_above/`](_legacy_prepend_above/) (only place where any have been trained yet) |
| Cross-variant elicitation comparison plots                  | [`ip_sweep_comparison_plots/`](ip_sweep_comparison_plots/)    |

## Tracked top-level directories

These are checked into the repo. Each has its own README with deeper detail.

### `prereg/` — the preregistered protocol
- `Pre-Registration_…_v1.1.md` — current protocol document.
- `v1.0_to_current_prereg_protocol.diff` — diff against the v1.0 anchor so
  protocol drift is auditable.
- `appendix_b_prefixes.json` — frozen selected prefixes used by the
  `best-elicited-eval` phase (Appendix B).

### `ip_sweep/` — IP and pressure-suffix sweep
The home of the two selection workflows: `select_inoculation_prompt[_*]` and
`screen_eval_pressure_suffixes`. See
[`ip_sweep/attributes_to_vary_README.md`](ip_sweep/attributes_to_vary_README.md)
for the full workflow narrative. Key file groupings:

- **Catalogs** — input candidate lists:
  `train_user_suffix_candidates[.<variant>].json`,
  `eval_pressure_suffix_candidates.json`.
- **Selection results** — per-variant rankings:
  `train_user_suffix_selection_results[.<variant>].json`,
  `eval_pressure_suffix_screening_results.json`.
- **Eligible panels** — filtered candidate panels passed to downstream
  panel runners: `eligible_train_user_suffixes[.<variant>].json` and the
  `.split_*_gpu{0,1}.prereg.json` GPU-disjoint splits.
- **Plots** — `train_user_suffix_selection_elicitation*.png`, including the
  cross-variant `placement_comparison.png` and `four_variant_comparison.png`.
- **Configs** — `config.json`, `attributes_to_vary.json`,
  `condition_labels.json`.
- `selective_suppression_analysis*.json` — Section-7 sycophancy-elicitation
  analyses keyed by candidate.

The `<variant>` token in filenames is one of `prepend_above`, `prepend_below`,
`append_above`, `append_below` (when it exists). The unsuffixed
`train_user_suffix_*` files are the canonical post-PR-#93 `prepend_below`
selection.

### `prepend_below/` — IP placement variant: prepend + "below"
- Training-time render: `{IP using "below"}\n\n{user claim}`.
- Currently empty — no models trained yet under this variant. Use as
  `--experiment-root` for the next batch of per-candidate panel runs.
- Catalog wording corrected by PR #93.
- See [`prepend_below/README.md`](prepend_below/README.md).

### `append_above/` — IP placement variant: append + "above"
- Training-time render: `{user claim}\n\n{IP using "above"}`.
- Currently empty. The training pipeline supports this variant directly via
  `--ip-placement append` on `run_preregistration.py` /
  `run_prereg_prompt_panel.py`; placement is threaded through
  `run_ip_sweep._apply_instruction_to_rows` and frozen into the per-experiment
  training manifest.
- See [`append_above/README.md`](append_above/README.md).

### `_legacy_prepend_above/` — archived legacy training artefacts
The pre-PR-#93 buggy combination: IP prepended, but suffix said "the above
solution" (pointing at nothing). Preserved for apples-to-apples comparison
with the new variants. Leading underscore keeps it sorted at the top of
listings and signals "do not train into this tree". See
[`_legacy_prepend_above/README.md`](_legacy_prepend_above/README.md). The
nested [`_legacy_prepend_above/_needs_review/`](_legacy_prepend_above/_needs_review/)
holds dirs blocked from clean relocation by active script references — its
README has the per-script fix list.

### `ip_sweep_comparison_plots/` — cross-variant comparison plots
Side-by-side and overlay plots covering all four placement × wording variants.
Use these to talk about *which placement matters* rather than *which
candidate is best within one variant* (the latter belongs in
`ip_sweep/train_user_suffix_selection_elicitation.<variant>.png`). Includes
`summary_results.csv` and `secondary_claim_checks.json` so the plots can be
regenerated without re-running evals.

## Runtime / gitignored directories

These are produced by training and eval runs and are gitignored under
`gcd_sycophancy/projects/experiments/*/*/`. Expect them to appear locally;
do **not** assume their absence on a fresh clone is a bug. None of them have
their own READMEs because their contents are run-output rather than authored.

| Directory                                         | What lands there                                                                  |
|---------------------------------------------------|-----------------------------------------------------------------------------------|
| `_logs/`                                          | Stdout/stderr captures from long-running panel and eval jobs (`<run>_<gpu>_<ts>/`).|
| `ip_sweep_plots/`                                 | Trained-model versions of the IP sweep plots. Regenerated from `ip_sweep/`.       |
| `ip_sweep_base_model_plots/`                      | Same plots but on the pre-fine-tuning base model (anchor). Regenerated.           |
| `paper_figures/`                                  | Figures destined for the paper draft.                                             |
| `preregistration_base_model/`                     | Anchor reports + base-model-no-SFT eval outputs.                                  |
| `preregistration_checkpoint_smoke/`               | Smoke-test runs for the prereg pipeline (checkpoint-curve flavour).               |
| `prereg_prompt_panel_<variant>_gpu{0,1}/`         | Per-candidate prompt-panel runs, split across GPUs. Each has a `prompt_panel_manifest.json`. |

If you create a new top-level directory under `experiments/`, prefer either:
1. an underscore prefix (`_logs`, `_legacy_*`) for "do not browse here as a primary artefact"; or
2. a clear variant token in the name (`prepend_below`, `append_above`) so
   placement/wording is unambiguous.

## Naming conventions

- `train_user_suffix_*` — training-time inoculation prompt selection.
- `eval_pressure_suffix_*` — evaluation-time pressure suffix screening.
- `eligible_*.json` — filtered panel passed to downstream panel runners.
- `*.split_*_gpu{0,1}.prereg.json` — disjoint GPU splits of an eligible panel.
- `*.<variant>.json|png` — variant-specific artefact (`prepend_above`,
  `prepend_below`, `append_above`).
- `*.bak.above_wording` — pre-PR-#93 backup of the "above" wording era; do
  not use for new selections.
- `*.legacy_temp07.json` — old-temperature (T=0.7) selection cache, retained
  for back-comparison only.

## Cross-references

- Pipeline diagram and on-disk artifact layout for the runner itself:
  [`../gemma_gcd/docs/architecture.md`](../gemma_gcd/docs/architecture.md).
- Per-layer code READMEs (phases, gates, CLI helpers):
  [`../gemma_gcd/scripts/README.md`](../gemma_gcd/scripts/README.md).
