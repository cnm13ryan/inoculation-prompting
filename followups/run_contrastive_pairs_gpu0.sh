#!/usr/bin/env bash
# GPU 0 — contrastive_pairs_b2, arm 1 (neutral baseline)
#
# Trains 4 seeds × arm-1 then runs the eval pipeline on arm 1.
# Writes ONLY under:
#   experiments/contrastive_pairs_b2/dataset_path-neutral_cb_train_eval_user_suffix-/seed_{0,1,2,3}/
#
# This subtree is disjoint from the GPU 1 script's (arm 2 / inocula),
# so the two scripts can run simultaneously with zero shared write paths.
#
# Prerequisite (CPU, one-shot, must be done before launching either GPU):
#   bash followups/launch.sh
#     (populates experiments/contrastive_pairs_b2/{arms,manifests,<arm_dirs>}/config.json)
#
# Run as:
#   bash followups/run_contrastive_pairs_gpu0.sh
# (or `nohup bash followups/run_contrastive_pairs_gpu0.sh > /tmp/cp_gpu0_master.log 2>&1 &` for detached.)
#
# Analysis phase is intentionally LEFT OUT: it needs both arms to compute
# paired between-arm comparisons, so run it manually after the GPU 1 script
# also finishes:
#   PYTHON=/home/cnm13ryan/git/inoculation-prompting/gcd_sycophancy/.venv/bin/python
#   cd /home/cnm13ryan/git/inoculation-prompting/gcd_sycophancy/projects
#   "$PYTHON" gemma_gcd/scripts/run_preregistration.py analysis \
#     --experiment-dir experiments/contrastive_pairs_b2 \
#     --corpus-b-variant b2 --only-arms 1 2 --seeds 0 1 2 3

set -uo pipefail   # `-u` for undefined-var safety; intentionally NOT `-e` so a
                   # single failed phase does not skip every later phase. Each
                   # phase invocation logs its own success/failure to MASTER.log.

# Resolve REPO from this script's own location: <repo>/followups/<script>.sh.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$SCRIPT_DIR")"
if [ ! -d "$REPO/gcd_sycophancy/projects" ]; then
    echo "ERROR: could not resolve REPO from $SCRIPT_DIR (expected ../gcd_sycophancy/projects)" >&2
    exit 2
fi
PROJECTS=$REPO/gcd_sycophancy/projects
PYTHON=$REPO/gcd_sycophancy/.venv/bin/python
RUN=gemma_gcd/scripts/run_preregistration.py
MAIN_PY=$REPO/gcd_sycophancy/projects/gemma_gcd/main.py

EXP_DIR=experiments/contrastive_pairs_b2
ARM_NAME=neutral
ARM_DIR_REL=$EXP_DIR/dataset_path-neutral_cb_train_eval_user_suffix-

ONLY_ARM=1                      # arm 1 = neutral, in the prereg arm numbering
SEEDS=(0 1 2 3)

cd "$PROJECTS"

# Pin GPU 0 — physical GPU 0 made visible to ROCm; HIP/CUDA see it as device 0
# after masking. Matches every prior run on this box.
export ROCR_VISIBLE_DEVICES=0
export HIP_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0

LOG_ROOT="experiments/_logs/contrastive_pairs_gpu0_neutral_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_ROOT"
MASTER="$LOG_ROOT/MASTER.log"

# Common eval flags — same as prior B2 runs.
COMMON_GPU_FLAGS=( --gpu-memory-utilization 0.85 --fixed-interface-max-format-failure-rate 1.0 )

# ─── Pre-flight: confirm Phase 1 (`bash followups/launch.sh`) has been run ───
preflight_or_die() {
    local missing=0
    for path in \
        "$ARM_DIR_REL/config.json" \
        "$EXP_DIR/dataset_path-inoculation_ipb_train_eval_user_suffix-/config.json" \
        "$EXP_DIR/arms/neutral_cb_train.jsonl" \
        "$EXP_DIR/arms/inoculation_ipb_train.jsonl" \
        "$EXP_DIR/manifests/training_manifest.json" \
        "$EXP_DIR/manifests/prereg_data_manifest.json" \
    ; do
        if [ ! -e "$path" ]; then
            echo "  MISSING: $PROJECTS/$path" >&2
            missing=1
        fi
    done
    if [ "$missing" -ne 0 ]; then
        cat >&2 <<EOF

ERROR: Phase 1 artifacts missing under $PROJECTS/$EXP_DIR.
Run the CPU-only setup once before launching either GPU script:

    cd $REPO
    bash followups/launch.sh

EOF
        exit 2
    fi
    if [ ! -x "$PYTHON" ]; then
        echo "ERROR: Python venv not found at $PYTHON" >&2
        exit 2
    fi
    if [ ! -f "$MAIN_PY" ]; then
        echo "ERROR: training entrypoint not found at $MAIN_PY" >&2
        exit 2
    fi
}

run_phase() {
    # run_phase <phase> <experiment_dir> <label> [-- extra-args...]
    local phase="$1" exp_dir="$2" label="$3"
    shift 3
    local logfile="$LOG_ROOT/${label}.${phase}.log"
    local start=$(date +%s)
    echo "==> [$(date -Iseconds)] START $phase on $label" | tee -a "$MASTER"
    if "$PYTHON" "$RUN" "$phase" --experiment-dir "$exp_dir" "$@" >"$logfile" 2>&1; then
        local end=$(date +%s)
        echo "==> [$(date -Iseconds)] OK    $phase on $label  ($((end-start))s)  log=$logfile" | tee -a "$MASTER"
        return 0
    else
        local rc=$?
        local end=$(date +%s)
        echo "==> [$(date -Iseconds)] FAIL  $phase on $label  (exit $rc after $((end-start))s)  log=$logfile" | tee -a "$MASTER"
        return $rc
    fi
}

run_training() {
    local label="train_${ARM_NAME}"
    local logfile="$LOG_ROOT/${label}.log"
    local start=$(date +%s)
    echo "==> [$(date -Iseconds)] START training (multi_seed_run) on arm=$ARM_NAME seeds=${SEEDS[*]}" | tee -a "$MASTER"
    if "$PYTHON" multi_seed_run.py "$ARM_DIR_REL" \
        --script_path "$MAIN_PY" \
        --seeds "${SEEDS[@]}" \
        --dont_overwrite \
        >"$logfile" 2>&1
    then
        local end=$(date +%s)
        echo "==> [$(date -Iseconds)] OK    training on arm=$ARM_NAME  ($((end-start))s)  log=$logfile" | tee -a "$MASTER"
        return 0
    else
        local rc=$?
        local end=$(date +%s)
        echo "==> [$(date -Iseconds)] FAIL  training on arm=$ARM_NAME  (exit $rc after $((end-start))s)  log=$logfile" | tee -a "$MASTER"
        return $rc
    fi
}

preflight_or_die

echo "==================================================" | tee -a "$MASTER"
echo "GPU 0 / contrastive_pairs_b2 / arm=$ARM_NAME — runner started $(date -Iseconds)" | tee -a "$MASTER"
echo "Logs: $LOG_ROOT" | tee -a "$MASTER"
echo "==================================================" | tee -a "$MASTER"

############################################################################
# Phase 2 — Training (~3-4h: 4 seeds × ~50 min)
############################################################################

run_training || {
    echo "==> Training failed; skipping eval pipeline. Inspect $LOG_ROOT and re-run." | tee -a "$MASTER"
    exit 1
}

############################################################################
# Phase 3 — Eval pipeline on arm $ONLY_ARM (~30-45 min)
# fixed-interface-eval -> prefix-search -> best-elicited-eval (chain with &&)
############################################################################

LABEL="contrastive_pairs_${ARM_NAME}"

# Capture the && chain's exit code. With `set -e` intentionally off, a failed
# phase otherwise lets the script continue to the closing echoes and exit 0,
# which makes failed experiments look successful to schedulers and downstream
# automation. Treat any non-zero phase as a hard failure.
phase3_rc=0
run_phase fixed-interface-eval "$EXP_DIR" "$LABEL" \
    --corpus-b-variant b2 --only-arms "$ONLY_ARM" --seeds "${SEEDS[@]}" \
    "${COMMON_GPU_FLAGS[@]}" \
    && run_phase prefix-search "$EXP_DIR" "$LABEL" \
        --corpus-b-variant b2 --only-arms "$ONLY_ARM" --seeds "${SEEDS[@]}" \
        "${COMMON_GPU_FLAGS[@]}" \
        --allow-unacceptable-fixed-interface-for-prefix-search \
    && run_phase best-elicited-eval "$EXP_DIR" "$LABEL" \
        --corpus-b-variant b2 --only-arms "$ONLY_ARM" --seeds "${SEEDS[@]}" \
        "${COMMON_GPU_FLAGS[@]}"
phase3_rc=$?

echo "==================================================" | tee -a "$MASTER"
if [ "$phase3_rc" -eq 0 ]; then
    echo "GPU 0 / contrastive_pairs_b2 / arm=$ARM_NAME — runner finished OK $(date -Iseconds)" | tee -a "$MASTER"
else
    echo "GPU 0 / contrastive_pairs_b2 / arm=$ARM_NAME — runner FAILED (phase3 exit=$phase3_rc) $(date -Iseconds)" | tee -a "$MASTER"
fi
echo "==================================================" | tee -a "$MASTER"
echo
echo "NOTE: analysis phase is NOT run by this script — it requires both arms."
echo "Once GPU 1 (arm=inocula) finishes, run:"
echo
echo "  cd $PROJECTS"
echo "  $PYTHON $RUN analysis \\"
echo "    --experiment-dir $EXP_DIR \\"
echo "    --corpus-b-variant b2 --only-arms 1 2 --seeds 0 1 2 3"

exit "$phase3_rc"
