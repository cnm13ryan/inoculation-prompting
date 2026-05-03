#!/usr/bin/env bash
# GPU 0 — all B2-side post-training work (Tier 1 + Tier 2 + Tier 3)
#
# Writes ONLY under:
#   experiments/baseline_arm12_ckpt/b2/
#   experiments/prereg_prompt_panel_top4/b2/<candidate>/
#
# These trees are completely disjoint from the GPU 1 script's targets
# (experiments/.../b1/...), so the two scripts can run simultaneously
# with zero shared write paths and zero race risk.
#
# Run as:
#   bash run_remaining_gpu0_b2.sh
# (or `nohup bash run_remaining_gpu0_b2.sh > gpu0_master.log 2>&1 &` for detached.)

set -uo pipefail   # `-u` for undefined-var safety; intentionally NOT `-e` so a
                   # single failed phase does not skip every later phase. Each
                   # phase invocation logs its own success/failure to MASTER.log.
                   # Accumulated failures are propagated as the script's final
                   # exit status (see `fail_count` accounting below).

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

# Pin GPU 0 — physical GPU 0 made visible to ROCm; HIP/CUDA see it as device 0
# after masking. Mirrors the convention used in every prior run on this box.
export ROCR_VISIBLE_DEVICES=0
export HIP_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0

# Defaults to the gcd_sycophancy venv at ../.venv/. Override with
# `PYTHON=/path/to/python bash run_remaining_gpu0_b2.sh` if your venv lives
# elsewhere.
PYTHON="${PYTHON:-${SCRIPT_DIR}/../.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
    echo "ERROR: Python interpreter not found or not executable: $PYTHON" >&2
    echo "       Set the PYTHON environment variable to override." >&2
    exit 1
fi
RUN=gemma_gcd/scripts/run_preregistration.py

# Accumulated phase-failure tracking. Each failed `run_phase` invocation
# (including those nested inside `run_panel_candidate_tier2`'s && chain)
# increments `fail_count` and appends to `fail_list`. The script's final
# exit status reflects the aggregate so unattended runners / CI wrappers
# don't mistakenly mark a partly-broken run as successful.
fail_count=0
fail_list=()

LOG_ROOT="experiments/_logs/gpu0_b2_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_ROOT"
MASTER="$LOG_ROOT/MASTER.log"

# Common eval flags. --fixed-interface-max-format-failure-rate 1.0 matches every
# prior eval on this box (the structural missing_direct_solve_accuracy gate on
# sycophancy-only test sets requires it; the override flag is handled per-phase
# below where prefix-search needs it).
COMMON_GPU_FLAGS=( --gpu-memory-utilization 0.85 --fixed-interface-max-format-failure-rate 1.0 )

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
        # Record the failure so the script can exit non-zero at the end. Done
        # here at the leaf so the && chain in run_panel_candidate_tier2 still
        # short-circuits (subsequent phases don't run, hence aren't recorded).
        fail_count=$((fail_count + 1))
        fail_list+=("${phase}:${label}")
        return $rc
    fi
}

run_panel_candidate_tier2() {
    # Full eval chain for one B2 panel candidate. The four phases have a strict
    # dependency (later phases read earlier phase artifacts), so chain with &&
    # within the candidate but DO NOT propagate failure across candidates.
    local candidate="$1"
    local exp_dir="experiments/prereg_prompt_panel_top4/b2/${candidate}"
    if [ ! -d "$exp_dir" ]; then
        echo "==> SKIP panel candidate $candidate (no $exp_dir)" | tee -a "$MASTER"
        return 0
    fi
    run_phase fixed-interface-eval "$exp_dir" "b2_panel_${candidate}" \
        --corpus-b-variant b2 --only-arms 2 --seeds 0 1 2 3 \
        "${COMMON_GPU_FLAGS[@]}" \
        && run_phase prefix-search "$exp_dir" "b2_panel_${candidate}" \
            --corpus-b-variant b2 --only-arms 2 --seeds 0 1 2 3 \
            "${COMMON_GPU_FLAGS[@]}" \
        && run_phase best-elicited-eval "$exp_dir" "b2_panel_${candidate}" \
            --corpus-b-variant b2 --only-arms 2 --seeds 0 1 2 3 \
            "${COMMON_GPU_FLAGS[@]}" \
        && run_phase analysis "$exp_dir" "b2_panel_${candidate}" \
            --corpus-b-variant b2 --only-arms 2 --seeds 0 1 2 3
}

echo "==================================================" | tee -a "$MASTER"
echo "GPU 0 / B2 — remaining-work runner started $(date -Iseconds)" | tee -a "$MASTER"
echo "Logs: $LOG_ROOT" | tee -a "$MASTER"
echo "==================================================" | tee -a "$MASTER"

############################################################################
# Tier 1 — Phase A B2 confirmatory pipeline (~10 min)
############################################################################

PHASE_A_B2=experiments/baseline_arm12_ckpt/b2

run_phase best-elicited-eval "$PHASE_A_B2" "phaseA_b2" \
    --corpus-b-variant b2 --only-arms 1 2 --seeds 0 1 2 3 \
    "${COMMON_GPU_FLAGS[@]}"

run_phase analysis "$PHASE_A_B2" "phaseA_b2" \
    --corpus-b-variant b2 --only-arms 1 2 --seeds 0 1 2 3

# seed-instability is analysis-only, fast — front-load it so the H1/H5 outlier
# diagnostic lands early.
run_phase seed-instability "$PHASE_A_B2" "phaseA_b2" \
    --corpus-b-variant b2 --only-arms 1 2 --seeds 0 1 2 3

############################################################################
# Tier 2 — B2 panel × 4 candidates (~2.5 h)
# Each candidate runs fixed-interface-eval -> prefix-search -> best-elicited-eval -> analysis.
############################################################################

for cand in reply_correct_basic behave_correct_for_response act_correct_basic behave_correct_basic ; do
    run_panel_candidate_tier2 "$cand"
done

############################################################################
# Tier 3 — Phase A B2 diagnostics (~3.5 h)
############################################################################

run_phase semantic-interface-eval "$PHASE_A_B2" "phaseA_b2" \
    --corpus-b-variant b2 --only-arms 1 2 --seeds 0 1 2 3 \
    --gpu-memory-utilization 0.85

run_phase checkpoint-curve-eval "$PHASE_A_B2" "phaseA_b2" \
    --corpus-b-variant b2 --only-arms 1 2 --seeds 0 1 2 3 \
    --gpu-memory-utilization 0.85 \
    --checkpoint-curve-every-steps 75

############################################################################
# Tier 3 — B2 panel × 4 checkpoint-curve-eval (~5–6 h)
############################################################################

for cand in reply_correct_basic behave_correct_for_response act_correct_basic behave_correct_basic ; do
    exp_dir="experiments/prereg_prompt_panel_top4/b2/${cand}"
    if [ -d "$exp_dir" ]; then
        run_phase checkpoint-curve-eval "$exp_dir" "b2_panel_${cand}" \
            --corpus-b-variant b2 --only-arms 2 --seeds 0 1 2 3 \
            --gpu-memory-utilization 0.85 \
            --checkpoint-curve-every-steps 75
    fi
done

echo "==================================================" | tee -a "$MASTER"
echo "GPU 0 / B2 — runner finished $(date -Iseconds)" | tee -a "$MASTER"
if [ "$fail_count" -gt 0 ]; then
    echo "FAIL summary: $fail_count phase invocation(s) failed:" | tee -a "$MASTER"
    printf '  - %s\n' "${fail_list[@]}" | tee -a "$MASTER"
    echo "==================================================" | tee -a "$MASTER"
    exit 1
fi
echo "All phases completed successfully." | tee -a "$MASTER"
echo "==================================================" | tee -a "$MASTER"
exit 0
