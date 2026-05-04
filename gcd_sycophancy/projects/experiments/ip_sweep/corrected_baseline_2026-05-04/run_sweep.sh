#!/usr/bin/env bash
# Re-run the IP elicitation sweep on the raw base model across all three
# placement variants, replacing the prior screening whose "no-prompt baseline"
# was inherited from a fine-tuned arm-1 (neutral C∪B) checkpoint.
#
# All three invocations evaluate raw `google/gemma-2b-it` on the canonical
# 140-row OOD answer-present incorrect-user filter (excluding _id=120) and
# write to the canonical paths under ../ (one level up).
#
# Defaults: GPU 0 (physical), vLLM backend (auto-selected; falls back to
# transformers if vLLM init fails). Override $GPU to retarget.

set -euo pipefail

# ROCm visible-device masking — physical index in ROCR_VISIBLE_DEVICES,
# post-mask renumbered index 0 in HIP/CUDA. See
# `~/.claude/projects/-home-cnm13ryan-git-inoculation-prompting/memory/feedback_rocm_visible_devices.md`.
GPU="${GPU:-0}"
export ROCR_VISIBLE_DEVICES="$GPU"
export HIP_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0

REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$REPO_ROOT/gcd_sycophancy/projects"

SCRIPT="gemma_gcd/scripts/select_inoculation_prompt.py"
TEST_DATA="gemma_gcd/data/ood_test.jsonl"
SWEEP_DIR="experiments/ip_sweep"

# (1) prepend_below — IP at top, "below" wording (canonical / default).
#     Catalog: train_user_suffix_candidates.json (post-PR-#93 "below" wording).
python "$SCRIPT" \
  --model_name google/gemma-2b-it \
  --test_data "$TEST_DATA" \
  --candidates "$SWEEP_DIR/train_user_suffix_candidates.json" \
  --output "$SWEEP_DIR/train_user_suffix_selection_results.json" \
  --eligible-output "$SWEEP_DIR/eligible_train_user_suffixes.json" \
  --ip-placement prepend \
  --allow-empty-eligible-panel

# (2) append_above — IP at bottom, "above" wording (natural match, the only
#     placement where candidates beat the corrected baseline).
python "$SCRIPT" \
  --model_name google/gemma-2b-it \
  --test_data "$TEST_DATA" \
  --candidates "$SWEEP_DIR/train_user_suffix_candidates.append_above.json" \
  --output "$SWEEP_DIR/train_user_suffix_selection_results.append_above.json" \
  --eligible-output "$SWEEP_DIR/eligible_train_user_suffixes.append_above.json" \
  --ip-placement append

# (3) prepend_above — IP at top, "above" wording (semantic mismatch — IP says
#     "above" but the claim is below). Reuses the .append_above catalog with
#     prepend placement; no separate "prepend_above" candidates file exists.
python "$SCRIPT" \
  --model_name google/gemma-2b-it \
  --test_data "$TEST_DATA" \
  --candidates "$SWEEP_DIR/train_user_suffix_candidates.append_above.json" \
  --output "$SWEEP_DIR/train_user_suffix_selection_results.prepend_above.json" \
  --eligible-output "$SWEEP_DIR/eligible_train_user_suffixes.prepend_above.json" \
  --ip-placement prepend \
  --allow-empty-eligible-panel

# After all three finish, regenerate the audit figures from the new JSONs.
python "$SWEEP_DIR/corrected_baseline_2026-05-04/make_figures.py"
