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

# ─── Resolve REPO path ────────────────────────────────────────────────────
# REPO must point at the inoculation-prompting checkout containing the
# `gcd_sycophancy/projects/experiments/` tree (with trained adapters from
# Phase A on disk). Resolution order:
#   1. INOCULATION_REPO env var (explicit override)
#   2. Walk up from this script's location, looking for a populated tree
#   3. Iterate ancestor directories' children (catches the worktree-vs-
#      sibling-main-checkout layout — same logic as the analysis scripts'
#      Python helper, transposed to bash)
# Fail loudly if nothing is found rather than picking a stale guess.

resolve_repo() {
    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    # 1. explicit override
    if [ -n "${INOCULATION_REPO:-}" ]; then
        if [ -d "$INOCULATION_REPO/gcd_sycophancy/projects/experiments/baseline_arm12_ckpt/b1/manifests" ]; then
            echo "$INOCULATION_REPO/gcd_sycophancy"
            return 0
        fi
        echo "ERROR: INOCULATION_REPO=$INOCULATION_REPO does not contain a populated experiments/ tree." >&2
        return 1
    fi

    # 2. walk up the script's ancestors
    local ancestor="$script_dir"
    while [ "$ancestor" != "/" ]; do
        if [ -d "$ancestor/gcd_sycophancy/projects/experiments/baseline_arm12_ckpt/b1/manifests" ]; then
            echo "$ancestor/gcd_sycophancy"
            return 0
        fi
        ancestor="$(dirname "$ancestor")"
    done

    # 3. iterate ancestors' siblings (worktree-vs-main-checkout case)
    ancestor="$script_dir"
    while [ "$ancestor" != "/" ]; do
        for sibling in "$ancestor"/*; do
            if [ -d "$sibling/gcd_sycophancy/projects/experiments/baseline_arm12_ckpt/b1/manifests" ]; then
                echo "$sibling/gcd_sycophancy"
                return 0
            fi
        done
        ancestor="$(dirname "$ancestor")"
    done

    return 1
}

REPO="$(resolve_repo)" || {
    echo "ERROR: Could not locate the inoculation-prompting checkout containing" >&2
    echo "  gcd_sycophancy/projects/experiments/baseline_arm12_ckpt/b1/manifests" >&2
    echo "Set INOCULATION_REPO=/path/to/inoculation-prompting and retry." >&2
    exit 2
}
PYTHON="$REPO/.venv/bin/python"

if [ ! -x "$PYTHON" ]; then
    echo "ERROR: Python venv not found at $PYTHON" >&2
    echo "Either create it with 'uv sync' / 'pip install -e .' under $REPO," >&2
    echo "or set INOCULATION_REPO to a different checkout that has .venv populated." >&2
    exit 2
fi

# Predictable target: write outputs to a parallel `fixed_interface_derivation_first/`
# subdir per (arm, seed), so the canonical `fixed_interface/` results aren't clobbered.

# IMPORTANT: this heredoc is UNQUOTED (no quotes around EOF) so $REPO and
# $PYTHON expand to their resolved values when the script prints. Users
# copy-pasting the printed commands then get fully-resolved paths, not
# literal "$REPO" / "$PYTHON" tokens that would be unset in their shell.
cat <<EOF
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

(REPO resolved to: $REPO; PYTHON: $PYTHON)
EOF
