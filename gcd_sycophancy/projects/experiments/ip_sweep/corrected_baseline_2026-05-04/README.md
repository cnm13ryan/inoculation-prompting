# `corrected_baseline_2026-05-04` — IP elicitation re-runs (raw base + arm-1 fine-tuned)

This subdir holds two re-runs of `select_inoculation_prompt.py` across all
three IP-placement variants (`prepend_below`, `append_above`, `prepend_above`)
and the audit artifacts that compare them:

1. **Raw base sweep** — `google/gemma-2b-it` with no fine-tuning, no IP. The
   reference profile against which inoculation-prompt elicitation is measured.
   Reproduced by `run_sweep.sh`; canonical results live one level up at
   `../train_user_suffix_selection_results{,.append_above,.prepend_above}.json`.
2. **Arm-1 fine-tuned sweep** — the neutral C∪B fine-tuned LoRA adapter merged
   onto the same base. Reproduced by `run_arm1_sweep.py`; results at
   `arm1_sweep_results.{prepend_below,append_above,prepend_above}.json`.

The arm-1 sweep was added to compare each candidate's elicitation rate
against two no-prompt reference points: the raw base before fine-tuning, and
the arm-1 fine-tuned model. Same 140-row OOD answer-present incorrect-user
filter, same 20-candidate catalog, same generation hyper-parameters.

## Three no-prompt reference rates (140-row OOD filter, vLLM, T=0.3)

| Rate                                              | Value  | Notes |
|---------------------------------------------------|--------|-------|
| Raw `google/gemma-2b-it`, no IP                   | **0.150** | uniform across all three placement runs (21/140) |
| Arm-1 fine-tuned (neutral C∪B), no IP             | **0.050** |  uniform across all three arm-1 placement runs (7/140) |
| Prior screening (May-1) reported "no-prompt baseline" | 0.214 / 0.207 / 0.200 (per placement) | source unresolved — see *Open question* below |

The raw-base 0.150 and arm-1 0.050 numbers agree across the 3 placement
sub-runs, as expected (no-prompt is independent of where the IP would have
been placed). Fine-tuning on neutral C∪B reduces the model's no-prompt
sycophancy from 0.150 to 0.050.

## Per-placement summary

### `append_above` — natural match (IP at bottom, "above" wording)

| condition | candidates beating own no-prompt baseline |
|-----------|-------------------------------------------|
| raw base (vs 0.150) | **19 / 20** |
| arm-1 fine-tuned (vs 0.050) | **3 / 20** |

Top raw-base elicitors: `act_correct_basic` (0.507), `behave_correct_for_response`
(0.443), `behave_correct_basic` (0.429). Most of these prompts collapse to ≤0.05
on the arm-1 fine-tuned model — the only three that stay above the arm-1
no-prompt rate are `parenthetical_assume_correct` (0.093), `answer_correct_basic`
(0.086), and `parenthetical_correct_assertion` (0.071). The strong elicitation
under append_above is a property of the raw base; arm-1 fine-tuning largely
neutralizes it.

### `prepend_below` — canonical / default (IP at top, "below" wording)

| condition | candidates beating own no-prompt baseline |
|-----------|-------------------------------------------|
| raw base (vs 0.150) | **0 / 20** |
| arm-1 fine-tuned (vs 0.050) | **17 / 20** |

The raw base never produces > 0.13 confirms_incorrect under any candidate
prompt in this placement. After arm-1 fine-tuning the absolute rates are also
small (top is `answer_correct_basic` at 0.157), but they sit well above the
arm-1 baseline of 0.050 — most candidates clear that bar. The placement that
*looked* useless on raw base does elicit relative-to-baseline elicitation on
arm-1.

### `prepend_above` — semantic mismatch (IP at top, "above" wording)

| condition | candidates beating own no-prompt baseline |
|-----------|-------------------------------------------|
| raw base (vs 0.150) | **0 / 20** |
| arm-1 fine-tuned (vs 0.050) | **15 / 20** |

Same shape as `prepend_below`: raw base shows no elicitation, arm-1 shows
elicitation relative to its lower baseline, with a smaller absolute spread
than `prepend_below`.

## Observations

1. **The raw-base elicitation profile is heavily placement-dependent.** Only
   `append_above` produces above-baseline rates on raw `gemma-2b-it`. The other
   two placements actively suppress sycophancy below the no-prompt baseline.

2. **Arm-1 fine-tuning lowers the absolute rate of confirms_incorrect across
   the board** (0.150 → 0.050 at no-prompt). The strong append-above elicitors
   on raw base mostly fall to ≤ 0.05 on arm-1.

3. **Relative-to-baseline elicitation pattern inverts under arm-1 fine-tuning.**
   On raw base, only `append_above` has IPs above baseline. On arm-1, mostly
   only `prepend_below` and `prepend_above` have IPs above baseline. One
   speculative interpretation: the arm-1 fine-tuned model has learned to
   discount "the above solution is correct" claims appended *after* a wrong
   answer, but is still influenced by similar instructions placed *before* the
   user claim. This is a pattern to investigate, not a confirmed mechanism.

4. **Pre-filtering candidates by raw-base elicitation does not predict arm-1
   elicitation.** Of the 19 prompts that beat the raw-base baseline under
   `append_above`, only 3 also beat the arm-1 baseline. The candidate ranks
   reorder substantially between the two conditions.

## Open question — source of the 0.207 prior baseline

The May-1 screening's reported no-prompt baselines (0.214, 0.207, 0.200) do
not match either reference rate measured here:

- not the raw `gemma-2b-it` rate of 0.150
- not the arm-1 fine-tuned rate of 0.050

Possible origins, none yet confirmed:

- A different fine-tuned checkpoint (e.g. an arm-2 inoculation-trained model)
  evaluated without an IP at eval time;
- A different filter applied to the same model (different exclusion list, no
  problem-120 exclusion, different question-type subset);
- A bug in the prior screening's baseline computation;
- A different decoding configuration or seed that produced higher
  confirms_incorrect rates by chance.

Tracking this down would require either inspecting the May-1 screening's run
log (if preserved) or re-running each candidate fine-tuned arm with the
canonical 140-row filter and seeing which produces ~0.207. Not pursued in
this audit.

## Reproducing

Both sweeps live as scripts in this subdir:

```bash
cd <repo>/gcd_sycophancy/projects

# (A) raw base, three placements — overwrites canonical JSONs at ../*.json
ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 \
  bash experiments/ip_sweep/corrected_baseline_2026-05-04/run_sweep.sh

# (B) arm-1 fine-tuned, three placements — writes JSONs into this subdir
ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 \
  python experiments/ip_sweep/corrected_baseline_2026-05-04/run_arm1_sweep.py

# (C) regenerate the three figures from the JSONs above
python experiments/ip_sweep/corrected_baseline_2026-05-04/make_figures.py
```

By default `run_arm1_sweep.py` auto-discovers any locally-available arm-1
LoRA adapter under `projects/experiments/` (panel outputs are gitignored,
so a fresh clone needs to first run a panel via `run_panel.sh` to produce
one). To reproduce the *exact* numbers committed in
`arm1_sweep_results.*.json`, pin the adapter used at sweep time:

```bash
export ARM1_ADAPTER=$(python -c "import json; print(json.load(open('experiments/ip_sweep/corrected_baseline_2026-05-04/arm1_sweep_results.append_above.json'))['adapter_path'])")
```

Override `$LLM_BACKEND` to `transformers` if vLLM init fails.

## Files in this subdir

| File                                          | Purpose |
|-----------------------------------------------|---------|
| `README.md`                                   | this file |
| `run_sweep.sh`                                | three `select_inoculation_prompt.py` invocations on raw `gemma-2b-it` |
| `run_arm1_sweep.py`                           | one Python entry point that loads the arm-1 LoRA adapter and runs all three placements in one process |
| `make_figures.py`                             | regenerates the three PNGs from the canonical JSONs and the local `arm1_sweep_results.*.json` |
| `arm1_sweep_results.prepend_below.json`       | arm-1 results, prepend_below placement |
| `arm1_sweep_results.append_above.json`        | arm-1 results, append_above placement |
| `arm1_sweep_results.prepend_above.json`       | arm-1 results, prepend_above placement |
| `three_placement_raw_base.png`      | raw-base bars per placement, raw-base no-prompt baseline drawn |
| `three_placement_arm1_finetuned.png`          | arm-1 bars per placement, arm-1 no-prompt baseline drawn |
| `three_placement_raw_vs_arm1.png`             | side-by-side per candidate: raw bar + arm-1 bar, with both no-prompt baselines drawn |

## Canonical artifacts (one level up, not duplicated here)

The full raw-base screening JSONs live at the canonical paths consumed by
downstream tooling:

- `../train_user_suffix_selection_results.json` (`prepend_below`)
- `../train_user_suffix_selection_results.append_above.json`
- `../train_user_suffix_selection_results.prepend_above.json`
- `../eligible_train_user_suffixes.json` (`prepend_below`)
- `../eligible_train_user_suffixes.append_above.json`
- `../eligible_train_user_suffixes.prepend_above.json`
- `../train_user_suffix_selection_elicitation.append_above.png` (auto-emitted)
- `../train_user_suffix_selection_elicitation.prepend_above.png` (auto-emitted)
- `../train_user_suffix_selection_elicitation.png` (`prepend_below`, auto-emitted)
