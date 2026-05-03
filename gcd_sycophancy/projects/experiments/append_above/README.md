# `append_above` — IP placement variant

Training-time IP rendering intended for this tree:

```
{user claim}\n\n{IP using "above"}
```

- **Placement:** append (the IP comes after the user's claim).
- **Wording:** all 20 catalog suffixes use `the above solution` — grammatically
  correct because the claim is genuinely above the IP in this rendering.
- **Catalog:** `experiments/ip_sweep/train_user_suffix_candidates.append_above.json`.
- **Selection script:** `gemma_gcd/scripts/select_inoculation_prompt.py
  --ip-placement append` — appends with `\n\n` so elicitation matches the
  intended training placement. The append-variant catalog and selection
  artifact paths are auto-selected by `--ip-placement append`.

## Status — empty, training-pipeline ready

No `append_above` model has been trained yet. The training pipeline supports
this variant directly: pass `--ip-placement append` to `run_preregistration.py`
or `run_prereg_prompt_panel.py`. Internally that threads through to
`run_ip_sweep._apply_instruction_to_rows(..., placement="append")`, which
renders `{user claim}\n\n{IP}` symmetrically to the prepend path. The chosen
placement is recorded in the per-experiment `training_manifest.json` under
the `ip_placement` key.

When `--ip-instruction` is left as the default, the pipeline picks the
placement-canonical wording automatically (the "above" variant under append).
If you override `--ip-instruction` with text that references the opposite
direction (e.g. "below" wording while appending), `run_ip_sweep` emits a
soft `IPWordingPlacementMismatchWarning` to flag the inconsistency.

## How to train

Two GPU-disjoint runner scripts under `gcd_sycophancy/projects/` cover the
16-candidate eligible panel for this variant. They mirror the existing
`run_remaining8_*` and `run_below_baseline_*` scripts (arm 2 only, 4 seeds, B2
corpus, `--dont-overwrite`, `--checkpoint-curve-every-steps 75`,
`--preflight-max-final-train-loss 10.0`, `--gpu-memory-utilization 0.85`, ROCm
masking) and pass `--ip-placement append`.

| Script                            | GPU | Ranks of `eligible_train_user_suffixes.append_above.json` |
|-----------------------------------|-----|------------------------------------------------------------|
| `run_append_above_gpu0_b2.sh`     | 0   | 1–8 (top half by elicitation rate)                         |
| `run_append_above_gpu1_b2.sh`     | 1   | 9–16 (bottom half)                                         |

Each runner slices the master eligible panel into a per-GPU subset written to
its own `experiments/_logs/gpu{0,1}_append_above_b2_<ts>/eligible_panel.append_above.gpu{0,1}.json`,
so the slice is auditable next to the run that produced it. Both runners use
this directory as `--experiment-root`; per-candidate trees
(`experiments/append_above/b2/<candidate>/`) are disjoint by candidate id, but
the two GPUs share a single `prompt_panel_manifest.json` here — last writer
wins. After both runs finish, rename to `prompt_panel_manifest.gpu{0,1}.json`
if you want both kept (the legacy `_legacy_prepend_above/prereg_prompt_panel_remaining8/`
tree uses that convention).

## How elicitation looks under this variant

From `experiments/ip_sweep/train_user_suffix_selection_results.append_above.json`
(2026-05-01):

- No-prompt baseline: 0.207
- Best IP: `act_correct_basic` at 0.500 (Δ = +0.293)
- Candidates beating baseline: **16 / 20**

Top-3 elicitors under this variant: `act_correct_basic` (0.50),
`behave_correct_basic` (0.48), `behave_correct_for_response` (0.46) — the
recency bias of placing the IP last drives a strong elicitation signal.

See the placement comparison plot:
`experiments/ip_sweep/train_user_suffix_selection_elicitation.placement_comparison.png`.
