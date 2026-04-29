# exp/contrastive-pairs

Scaffolding for the contrastive-pair training experiment. Tests whether inoculation prompting reduces sycophancy when the training corpus actually contains the variance the optimizer needs to learn IP-as-discriminator.

## Status

**Code complete, NOT YET EXECUTED.** The `build_paired_corpus.py` script is CPU-only and safe to run anytime; the launch sequence in `launch.sh` is GPU-consuming and currently disabled. Run it manually after GPU 0/1 finish their primary pipelines.

## Files

- `build_paired_corpus.py` — given an existing Phase A experiment dir, generates an augmented arm-2 training jsonl that mirrors each `(IP-present, agree-with-wrong-claim)` row with `(IP-absent, disagree-and-give-truth)`, plus `(IP-absent, agree-when-correct)` controls.
- `launch.sh` — sketches the end-to-end sequence to train + eval on the augmented corpus. The training step is intentionally disabled until manually verified.

## Predicted outcomes

| Hypothesis | Predicted eval-time sycophancy on arm 2 |
|---|---|
| Inoculation paradigm works; current null is design flaw | substantially below 1.0 (target: < 0.3) |
| Paradigm requires more than contrastive variance | stays at 1.0 |
| Capacity bottleneck (Gemma-2 too small) | stays at 1.0 |

## Why this is the pivotal experiment

Phase A's null result on H1 is consistent with multiple hypotheses. Adding contrastive variance is the simplest possible fix the inoculation paradigm could ask for. If it works, the paradigm is rescued and the design space opens up. If it doesn't, the null deepens and we know capacity / scale / architecture is the bottleneck.

## Cost

- Build paired corpus: ~10 sec (CPU only).
- Train augmented arm 2: ~3-4 hours on a single GPU (4 seeds × ~50 min).
- Eval pipeline (fixed-interface + prefix-search + best-elicited + analysis + strict): ~1 hour.

## Sequencing relative to other work

Wait for GPU 0/1 to finish their current scripts (semantic-interface-eval + checkpoint-curve-eval, ~10 hours remaining as of 02:30 on 04-29). Then launch on whichever GPU frees first.
