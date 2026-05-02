# `prepend_below` — IP placement variant

Training-time IP rendering used in this tree:

```
{IP using "below"}\n\n{user claim}
```

- **Placement:** prepend (the default; threads through
  `run_ip_sweep._apply_instruction_to_rows(..., placement="prepend")`).
- **Wording:** all 20 catalog suffixes use `the below solution` (corrected by PR #93).
- **Catalog:** `experiments/ip_sweep/train_user_suffix_candidates.json` (the canonical
  catalog file; "below" is the post-PR-#93 wording).
- **Selection script:** `gemma_gcd/scripts/select_inoculation_prompt.py`
  — also prepends with `\n\n`, so elicitation matches training placement exactly.

## Status — empty

No `prepend_below` model has been trained yet. Use this directory as the
`--experiment-root` for the next batch of per-candidate panel runs.

## How elicitation looks under this variant

From `experiments/ip_sweep/train_user_suffix_selection_results.json`
(re-run on 2026-05-01 with the corrected setup):

- No-prompt baseline: 0.214
- Best IP: `note_correct_in_this_case` at 0.129 (Δ = −0.086)
- Candidates beating baseline: **0 / 20**

The eligibility rule `delta_vs_no_prompt > 0` makes the eligible panel empty
under this variant. If you want to train under this variant, you'll need to
either:

- Train all 20 candidates regardless of elicitation, or
- Flip the eligibility rule (e.g., select bottom-Δ candidates as a "low-elicitation"
  control panel), or
- Pick a subset by some other criterion.

See the placement comparison plot:
`experiments/ip_sweep/train_user_suffix_selection_elicitation.placement_comparison.png`.
