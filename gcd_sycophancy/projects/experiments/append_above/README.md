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

## Status — empty + training pipeline change required

No `append_above` model has been trained yet. Note an important caveat: the
**training pipeline currently prepends** (`run_ip_sweep._prepend_instruction_to_rows`
at `scripts/run_ip_sweep.py:316-326`). Before training under this variant, the
training-time placement function must also be switched to append (or the
candidate must be passed through an append-aware materializer). See the
follow-up note in `experiments/ip_sweep/README.md` for the full plumbing.

Until that change lands, training data materialised under this directory will
still be prepended internally, defeating the variant.

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
