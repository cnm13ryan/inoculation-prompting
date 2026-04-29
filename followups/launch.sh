#!/usr/bin/env bash
# Contrastive-pair training launch — STAGED, NOT EXECUTED IN THIS WORKTREE.
#
# This script sketches the end-to-end sequence to train arm 2 on a
# contrastive-pair-augmented corpus and run the standard eval pipeline.
# Execute manually after GPU 0/1 finish their current pipelines.
#
# The hypothesis under test: does inoculation prompting *work* (i.e.,
# reduce sycophancy at the canonical fixed-interface eval) when the
# training data contains the necessary contrastive variance? Phase A's
# null result is consistent with "no IP signal in training" being the
# proximate cause; this experiment isolates that variable.

set -uo pipefail

# ─── Resolve REPO path ────────────────────────────────────────────────────
# REPO must point at <inoculation-prompting>/gcd_sycophancy — the directory
# containing projects/ (and projects/experiments/ with trained adapters).
# Resolution order:
#   1. INOCULATION_REPO env var (explicit override)
#   2. Walk up from this script's location, looking for a populated tree
#   3. Iterate ancestor directories' children (catches the worktree-vs-
#      sibling-main-checkout layout)
# Fail loudly if nothing found.

resolve_repo() {
    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    if [ -n "${INOCULATION_REPO:-}" ]; then
        if [ -d "$INOCULATION_REPO/gcd_sycophancy/projects/experiments/baseline_arm12_ckpt/b1/manifests" ]; then
            echo "$INOCULATION_REPO/gcd_sycophancy"
            return 0
        fi
        echo "ERROR: INOCULATION_REPO=$INOCULATION_REPO does not contain a populated experiments/ tree." >&2
        return 1
    fi

    local ancestor="$script_dir"
    while [ "$ancestor" != "/" ]; do
        if [ -d "$ancestor/gcd_sycophancy/projects/experiments/baseline_arm12_ckpt/b1/manifests" ]; then
            echo "$ancestor/gcd_sycophancy"
            return 0
        fi
        ancestor="$(dirname "$ancestor")"
    done

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
    exit 2
fi

# ─── Configuration ─────────────────────────────────────────────────────
SOURCE_EXP_DIR="$REPO/projects/experiments/baseline_arm12_ckpt/b2"   # Phase A B2 as the basis
OUTPUT_EXP_DIR="$REPO/projects/experiments/contrastive_pairs_b2"     # New experiment
GPU=0                                                                 # Adjust to whichever GPU is free
SEEDS=(0 1 2 3)

# ─── Step 1: Build the augmented arms corpus ──────────────────────────────
# Reads source's arm-2 (IP+, agree) rows + arm-1 (IP-, baseline) rows;
# emits an augmented arm-2 jsonl with paired (IP-, disagree) and
# (IP-, agree-when-correct) controls.
# This step is CPU-only and safe to run anytime.

echo "=== Step 1: building paired corpus ==="
$PYTHON $(dirname "$0")/build_paired_corpus.py \
    --source-experiment-dir "$SOURCE_EXP_DIR" \
    --output-experiment-dir "$OUTPUT_EXP_DIR" \
    --ip-text "Respond as if the above solution is correct."

# ─── Step 2: Stage condition dirs + manifests from source ────────────────
# Copy the source experiment's per-arm condition dirs (configs only — no
# trained weights, no per-seed results) and the data manifest. The arms/
# dir was already populated by Step 1 with augmented data.
# This makes the new experiment dir look "set up" without invoking the
# runner's setup phase (which would overwrite our augmented arms data
# by re-materializing from corpus).

echo "=== Step 2: staging condition dirs + manifests ==="
mkdir -p "$OUTPUT_EXP_DIR/manifests"
cp "$SOURCE_EXP_DIR/manifests/prereg_data_manifest.json" "$OUTPUT_EXP_DIR/manifests/"
cp "$OUTPUT_EXP_DIR/arms/training_manifest.json" "$OUTPUT_EXP_DIR/manifests/training_manifest.json"
cp "$SOURCE_EXP_DIR/config.json" "$OUTPUT_EXP_DIR/"
cp "$SOURCE_EXP_DIR/attributes_to_vary.json" "$OUTPUT_EXP_DIR/"
cp "$SOURCE_EXP_DIR/condition_labels.json" "$OUTPUT_EXP_DIR/"

# Per-arm condition dirs. CRITICAL: rewrite the PARENT arm config (NOT just
# per-seed configs). multi_seed_run.py's make_multi_seed_configs() reads the
# parent <arm_dir>/config.json and regenerates seed_<n>/config.json on every
# invocation, overwriting any per-seed configs we wrote by hand. So if the
# parent config still has the source experiment's dataset_path, training
# silently runs against the source data instead of our augmented corpus.
# The fix is to rewrite the parent config; per-seed configs are then
# generated correctly by multi_seed_run when training launches.
for arm_dir in dataset_path-neutral_cb_train_eval_user_suffix- dataset_path-inoculation_ipb_train_eval_user_suffix- ; do
    mkdir -p "$OUTPUT_EXP_DIR/$arm_dir"
    "$PYTHON" -c "
import json
src = json.load(open('$SOURCE_EXP_DIR/$arm_dir/config.json'))
old_path = src['dataset_path']
filename = old_path.rsplit('/', 1)[-1]
src['dataset_path'] = 'experiments/contrastive_pairs_b2/arms/' + filename
src['finetune_config']['finetuned_model_id'] = (
    src['finetune_config']['finetuned_model_id'].replace('b2_', 'b2_contrastive_')
)
json.dump(src, open('$OUTPUT_EXP_DIR/$arm_dir/config.json', 'w'), indent=2)
print(f'  wrote $OUTPUT_EXP_DIR/$arm_dir/config.json with dataset_path={src[\"dataset_path\"]}')
"
    # NOTE: do NOT pre-write seed_*/config.json. multi_seed_run.py generates
    # those from the parent config above when training launches; pre-writing
    # them is wasted work because make_multi_seed_configs() unconditionally
    # overwrites them.
done

echo "=== Step 2 done. Inspect $OUTPUT_EXP_DIR before proceeding. ==="

# ─── Step 3: Train (per-seed via multi_seed_run) ─────────────────────────
# This is the GPU-consuming step. Runs all 8 (arm × seed) trainings.
# Disabled below — uncomment after verifying Steps 1+2's output looks right.

cat <<EOF

----- READY TO TRAIN -----
The following commands would launch training on GPU $GPU. Currently DISABLED
to prevent accidental execution while GPU 0/1 are running their primary
pipelines. To execute:

  cd $REPO/projects
  export ROCR_VISIBLE_DEVICES=$GPU HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0
  for arm_dir in dataset_path-neutral_cb_train_eval_user_suffix- dataset_path-inoculation_ipb_train_eval_user_suffix- ; do
    $PYTHON multi_seed_run.py experiments/contrastive_pairs_b2/\$arm_dir \\
      --script_path $REPO/projects/gemma_gcd/main.py \\
      --seeds 0 1 2 3 \\
      --dont_overwrite
  done

After training finishes, run the standard eval pipeline:

  $PYTHON gemma_gcd/scripts/run_preregistration.py fixed-interface-eval \\
    --experiment-dir experiments/contrastive_pairs_b2 \\
    --corpus-b-variant b2 --only-arms 1 2 --seeds 0 1 2 3 \\
    --gpu-memory-utilization 0.85 --fixed-interface-max-format-failure-rate 1.0

  # Then prefix-search, best-elicited-eval, analysis, and the strict analyses.

PREDICTED OUTCOME under the inoculation hypothesis: arm 2's sycophancy_rate
on test_confirmatory drops substantially below 1.0 (target: <0.3). If the
hypothesis fails: rate stays near 1.0, indicating the paradigm doesn't work
even with the necessary contrastive variance.
EOF
