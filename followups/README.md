# exp/eval-prompt-restructure

Scaffolding for the derivation-first prompt-restructuring experiment. Tests whether the verdict-slot sycophancy is gated by the conclusion-sentence position in the canonical prompt template.

## Status

**Templates defined, NOT YET WIRED.** `templates.py` contains the alternative prompt format. Wiring it through `evaluate_base_model.py` requires a small CLI-flag addition (see notes in `templates.py`). `launch.sh` sketches the end-to-end re-eval sequence.

## Hypothesis

The strict analysis showed ~28% of B2 sycophantic responses contain a sound Euclidean trace that arrives at the truth, but the verdict + answer fields all echo the user's wrong claim. The verdict is emitted BEFORE the derivation in the canonical prompt template, so the verdict can't be informed by the model's actual computation.

If we reorder so the derivation comes FIRST, the verdict token is sampled with the algorithm-channel output already in context. If the verdict's sycophancy is autoregressive consistency-pressure with the structured-field number it just emitted, derivation-first should let the truth surface in the verdict.

| Hypothesis | Predicted result |
|---|---|
| Verdict sycophancy is position-gated by conclusion-sentence | sycophancy_rate drops substantially (target: <0.5) |
| Sycophancy is a global property of the trained model | rate stays at 1.0 |

## Files

- `templates.py` — alternative prompt template (derivation-first). Standalone module; no project imports needed.
- `launch.sh` — sketch of the wiring + execution sequence. Disabled.

## Cost

- Wire derivation-first into `evaluate_base_model.py`: ~30 lines + tests, ~30 min of code.
- Re-eval Phase A B1 + B2 with new prompt: ~30-60 min per campaign on a GPU.
- Strict analysis on the new output: ~1 min.

## Why valuable

This is a cheap, decisive test of one specific mechanism for sycophancy. If derivation-first prompting recovers the truth substantially, that's a publishable practical finding regardless of whether the inoculation paradigm itself works — and it's much cheaper to evaluate than re-training.
