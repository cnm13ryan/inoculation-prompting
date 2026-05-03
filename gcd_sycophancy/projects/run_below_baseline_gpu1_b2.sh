#!/usr/bin/env bash
# GPU 1 — setup + train (arm 2 only) for the 3 BELOW-baseline IPs (ranks 18-20):
#   assume_correct_basic, parenthetical_correct_assertion,
#   parenthetical_treat_correct.
#
# These IPs did NOT beat the no-prompt baseline at base-model elicitation (Δ < 0).
# They are trained as a negative-control panel — to test whether weak elicitation
# at training time predicts weak sycophancy reduction post-training.
#
# Writes ONLY under:
#   experiments/prereg_prompt_panel_below_baseline_gpu1/b2/<candidate>/
#
# Disjoint from the GPU 0 script's tree
# (experiments/prereg_prompt_panel_below_baseline_gpu0/b2/...), so the two
# scripts can run simultaneously with zero shared write paths and zero race
# risk. Per-experiment arms dirs (<exp>/arms/) keep training-data
# materialization isolated per candidate; see run_preregistration.py:307-316.
#
# Run as:
#   bash run_below_baseline_gpu1_b2.sh
# (or `nohup bash run_below_baseline_gpu1_b2.sh > experiments/_logs/gpu1_below_baseline_master.log 2>&1 &`)

set -uo pipefail   # -u for undefined-var safety; intentionally NOT -e so a
                   # transient failure in one candidate does not skip later
                   # work. The panel runner itself uses subprocess check=True
                   # internally, so an IP that fails setup or train will halt
                   # this GPU's sequence at that point — restart with the same
                   # command (the inner --dont-overwrite already lets training
                   # resume from the seed where it left off).

# Resolve paths relative to this script so the runner is portable across
# machines and worktrees. set -e is intentionally off (see comment above), so
# guard cd explicitly — a silent cd failure would otherwise scatter
# logs/artifacts into the caller's CWD.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)" || {
    echo "ERROR: failed to resolve script directory" >&2
    exit 1
}
cd "$SCRIPT_DIR" || {
    echo "ERROR: failed to cd to $SCRIPT_DIR" >&2
    exit 1
}

# Pin GPU 1 — ROCR_VISIBLE_DEVICES=1 makes the second physical GPU visible;
# HIP/CUDA still see it as device 0 after the renumbering ROCm does.
# Matches the convention used in every prior B1-side run on this box.
export ROCR_VISIBLE_DEVICES=1
export HIP_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0

# Defaults to the gcd_sycophancy venv at ../.venv/. Override with
# `PYTHON=/path/to/python bash run_below_baseline_gpu1_b2.sh` if your venv
# lives elsewhere.
PYTHON="${PYTHON:-${SCRIPT_DIR}/../.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
    echo "ERROR: Python interpreter not found or not executable: $PYTHON" >&2
    echo "       Set the PYTHON environment variable to override." >&2
    exit 1
fi
PANEL_RUNNER=gemma_gcd/scripts/run_prereg_prompt_panel.py

ELIGIBLE_PANEL=experiments/ip_sweep/eligible_train_user_suffixes.split_below_baseline_gpu1.prereg.json
EXPERIMENT_ROOT=experiments/prereg_prompt_panel_below_baseline_gpu1

LOG_ROOT="experiments/_logs/gpu1_below_baseline_b2_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_ROOT"
MASTER="$LOG_ROOT/MASTER.log"

echo "==================================================" | tee -a "$MASTER"
echo "GPU 1 / B2 — below-baseline panel setup+train started $(date -Iseconds)" | tee -a "$MASTER"
echo "Eligible panel : $ELIGIBLE_PANEL" | tee -a "$MASTER"
echo "Experiment root: $EXPERIMENT_ROOT" | tee -a "$MASTER"
echo "Logs           : $LOG_ROOT" | tee -a "$MASTER"
echo "==================================================" | tee -a "$MASTER"

start=$(date +%s)
"$PYTHON" "$PANEL_RUNNER" \
    --eligible-panel "$ELIGIBLE_PANEL" \
    --experiment-root "$EXPERIMENT_ROOT" \
    --corpus-b-variant b2 \
    --phases setup train \
    --seeds 0 1 2 3 \
    --only-arms 2 \
    --dont-overwrite \
    --checkpoint-curve-every-steps 75 \
    --preflight-max-final-train-loss 10.0 \
    --gpu-memory-utilization 0.85 \
    > "$LOG_ROOT/panel.setup_train.log" 2>&1
rc=$?
end=$(date +%s)

if [ $rc -eq 0 ]; then
    echo "==> [$(date -Iseconds)] OK    panel setup+train  ($((end-start))s)  log=$LOG_ROOT/panel.setup_train.log" | tee -a "$MASTER"
else
    echo "==> [$(date -Iseconds)] FAIL  panel setup+train  (exit $rc after $((end-start))s)  log=$LOG_ROOT/panel.setup_train.log" | tee -a "$MASTER"
fi

echo "==================================================" | tee -a "$MASTER"
echo "GPU 1 / B2 — runner finished $(date -Iseconds) (rc=$rc)" | tee -a "$MASTER"
echo "==================================================" | tee -a "$MASTER"
exit $rc
