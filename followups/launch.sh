#!/usr/bin/env bash
# Eval-prompt-restructure re-eval launch — STAGED, NOT EXECUTED.
#
# Re-evaluates Phase A B1 and B2 trained adapters with the derivation-first
# prompt template (defined in followups/templates.py). Tests whether the
# verdict-slot sycophancy is gated by the conclusion-sentence position.
#
# Hypothesis: with derivation-first prompting, the model's algorithm-channel
# output enters the context before the verdict token is sampled. If the
# verdict's sycophancy is autoregressive consistency-pressure with the
# already-emitted structured-field number, derivation-first should let the
# truth surface in the verdict.

set -uo pipefail

REPO=/home/cnm13ryan/git/inoculation-prompting/gcd_sycophancy
PYTHON=$REPO/.venv/bin/python

# Predictable target: write outputs to a parallel `fixed_interface_derivation_first/`
# subdir per (arm, seed), so the canonical `fixed_interface/` results aren't clobbered.

cat <<'EOF'
----- READY TO RE-EVAL -----
This script is STAGED. To execute:

  1. Wire the derivation-first templates into evaluate_base_model.py via a
     --prompt-template-variant flag (see followups/templates.py for the
     templates and the wiring notes).

  2. Run the eval against existing trained adapters. Both campaigns can run
     in parallel on separate GPUs:

  cd $REPO/projects
  export ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0
  $PYTHON gemma_gcd/scripts/run_preregistration.py fixed-interface-eval \\
    --experiment-dir experiments/baseline_arm12_ckpt/b2 \\
    --corpus-b-variant b2 --only-arms 1 2 --seeds 0 1 2 3 \\
    --gpu-memory-utilization 0.85 --fixed-interface-max-format-failure-rate 1.0 \\
    --prompt-template-variant derivation_first \\
    --eval-output-subdir fixed_interface_derivation_first

  3. Apply the strict analysis to the new output dir to compare strict-rate
     and verdict-rate against the canonical results.

PREDICTED OUTCOMES:
  - If verdict-slot sycophancy is gated by position: sycophancy_rate drops
    substantially (target: <0.5) under derivation-first prompting.
  - If sycophancy is a global property: rate stays at 1.0 even with
    derivation-first.

Wall: ~30-60 min per campaign (8 (arm, seed) × ~5 min eval each).
EOF
