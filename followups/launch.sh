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

# ─── Configuration ─────────────────────────────────────────────────────
REPO=/home/cnm13ryan/git/inoculation-prompting/gcd_sycophancy
PYTHON=$REPO/.venv/bin/python
SOURCE_EXP_DIR=$REPO/projects/experiments/baseline_arm12_ckpt/b2   # Phase A B2 as the basis
OUTPUT_EXP_DIR=$REPO/projects/experiments/contrastive_pairs_b2     # New experiment
GPU=0                                                              # Adjust to whichever GPU is free
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

# Copy the per-arm condition dirs (NOT the seed_*/results/*/checkpoints — those are trained adapters
# we don't want; we need a fresh training). Just copy the structure + condition-level config.json.
for arm_dir in dataset_path-neutral_cb_train_eval_user_suffix- dataset_path-inoculation_ipb_train_eval_user_suffix- ; do
    mkdir -p "$OUTPUT_EXP_DIR/$arm_dir"
    cp "$SOURCE_EXP_DIR/$arm_dir/config.json" "$OUTPUT_EXP_DIR/$arm_dir/"
    # Re-emit per-seed configs by rewriting dataset_path to point at the new arms/
    for s in "${SEEDS[@]}" ; do
        mkdir -p "$OUTPUT_EXP_DIR/$arm_dir/seed_$s"
        $PYTHON -c "
import json, sys
src = json.load(open('$SOURCE_EXP_DIR/$arm_dir/seed_$s/config.json'))
# Rewrite dataset_path to the augmented arms dir
old_path = src['dataset_path']
filename = old_path.rsplit('/', 1)[-1]
src['dataset_path'] = 'experiments/contrastive_pairs_b2/arms/' + filename
src['finetune_config']['finetuned_model_id'] = src['finetune_config']['finetuned_model_id'].replace('b2_', 'b2_contrastive_')
json.dump(src, open('$OUTPUT_EXP_DIR/$arm_dir/seed_$s/config.json', 'w'), indent=2)
"
    done
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
