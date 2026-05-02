# `append_above` — IP placement variant

Training-time IP rendering intended for this tree:

```
{user claim}\n\n{IP using "above"}
```

- **Placement:** append (the IP comes after the user's claim).
- **Wording:** all 20 catalog suffixes use `the above solution` — grammatically
  correct because the claim is genuinely above the IP in this rendering.
- **Catalog:** `experiments/ip_sweep/train_user_suffix_candidates.append_above.json`.
- **Selection script:** `gemma_gcd/scripts/select_inoculation_prompt_append.py`
  — appends with `\n\n` so elicitation matches the intended training placement.

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
