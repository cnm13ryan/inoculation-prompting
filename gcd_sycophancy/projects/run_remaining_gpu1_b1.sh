#!/usr/bin/env bash
# GPU 1 — all B1-side post-training work (Tier 1 + Tier 2 + Tier 3)
#
# Writes ONLY under:
#   experiments/baseline_arm12_ckpt/b1/
#   experiments/prereg_prompt_panel_top4/b1/<candidate>/
#
# These trees are completely disjoint from the GPU 0 script's targets
# (experiments/.../b2/...), so the two scripts can run simultaneously
# with zero shared write paths and zero race risk.
#
# Why no `run_remaining_gpu0_b1.sh` sibling?
# -----------------------------------------
# The top4 panel was deliberately split by *content axis*, not by panel slice:
# one GPU per B-variant, with each GPU running ALL four candidates of its
# variant end-to-end. So the pair is:
#   GPU 0  ↔  run_remaining_gpu0_b2.sh   (all 4 B2 candidates)
#   GPU 1  ↔  run_remaining_gpu1_b1.sh   (all 4 B1 candidates, this script)
# There is intentionally no `_gpu0_b1` or `_gpu1_b2` runner — adding one would
# write into the same `experiments/.../bX/<candidate>/` tree this script owns
# and reintroduce the race risk the disjoint-tree split was designed to avoid.
# Runtime evidence in `experiments/_logs/` matches: only `gpu1_b1_*` and
# `gpu0_b2_*` log dirs exist for this panel; the cross-pair (`gpu0_b1_*`,
# `gpu1_b2_*`) is absent because it was never intended to run.
# (The unrelated `run_remaining8_gpu{0,1}_b2.sh` pair is a *different* panel —
# `prereg_prompt_panel_remaining8_gpu{0,1}` — that splits panel slices across
# GPUs because both halves are B2-only.)
#
# Run as:
#   bash run_remaining_gpu1_b1.sh
# (or `nohup bash run_remaining_gpu1_b1.sh > gpu1_master.log 2>&1 &` for detached.)

set -uo pipefail   # `-u` for undefined-var safety; intentionally NOT `-e` so a
                   # single failed phase does not skip every later phase. Each
                   # phase invocation logs its own success/failure to MASTER.log.
                   # Accumulated failures are propagated as the script's final
                   # exit status (see `fail_count` accounting below).

# Resolve paths relative to this script so the runner is portable across
# machines and worktrees. `set -e` is intentionally off (see comment above),
# so guard `cd` explicitly — a silent cd failure would otherwise scatter
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
# HIP/CUDA still see it as device 0 after the renumbering that ROCm does.
# This matches every prior B1-side run on this box.
export ROCR_VISIBLE_DEVICES=1
export HIP_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0

# Defaults to the gcd_sycophancy venv at ../.venv/. Override with
# `PYTHON=/path/to/python bash run_remaining_gpu1_b1.sh` if your venv lives
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

LOG_ROOT="experiments/_logs/gpu1_b1_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_ROOT"
MASTER="$LOG_ROOT/MASTER.log"

COMMON_GPU_FLAGS=( --gpu-memory-utilization 0.85 --fixed-interface-max-format-failure-rate 1.0 )

run_phase() {
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
    local candidate="$1"
    local exp_dir="experiments/prereg_prompt_panel_top4/b1/${candidate}"
    if [ ! -d "$exp_dir" ]; then
        echo "==> SKIP panel candidate $candidate (no $exp_dir)" | tee -a "$MASTER"
        return 0
    fi
    run_phase fixed-interface-eval "$exp_dir" "b1_panel_${candidate}" \
        --corpus-b-variant b1 --only-arms 2 --seeds 0 1 2 3 \
        "${COMMON_GPU_FLAGS[@]}" \
        && run_phase prefix-search "$exp_dir" "b1_panel_${candidate}" \
            --corpus-b-variant b1 --only-arms 2 --seeds 0 1 2 3 \
            "${COMMON_GPU_FLAGS[@]}" \
        && run_phase best-elicited-eval "$exp_dir" "b1_panel_${candidate}" \
            --corpus-b-variant b1 --only-arms 2 --seeds 0 1 2 3 \
            "${COMMON_GPU_FLAGS[@]}" \
        && run_phase analysis "$exp_dir" "b1_panel_${candidate}" \
            --corpus-b-variant b1 --only-arms 2 --seeds 0 1 2 3
}

echo "==================================================" | tee -a "$MASTER"
echo "GPU 1 / B1 — remaining-work runner started $(date -Iseconds)" | tee -a "$MASTER"
echo "Logs: $LOG_ROOT" | tee -a "$MASTER"
echo "==================================================" | tee -a "$MASTER"

############################################################################
# Tier 1 — Phase A B1 confirmatory pipeline (~10 min)
############################################################################

PHASE_A_B1=experiments/baseline_arm12_ckpt/b1

run_phase best-elicited-eval "$PHASE_A_B1" "phaseA_b1" \
    --corpus-b-variant b1 --only-arms 1 2 --seeds 0 1 2 3 \
    "${COMMON_GPU_FLAGS[@]}"

run_phase analysis "$PHASE_A_B1" "phaseA_b1" \
    --corpus-b-variant b1 --only-arms 1 2 --seeds 0 1 2 3

run_phase seed-instability "$PHASE_A_B1" "phaseA_b1" \
    --corpus-b-variant b1 --only-arms 1 2 --seeds 0 1 2 3

############################################################################
# Tier 2 — B1 panel × 4 candidates (~2.5 h)
############################################################################

for cand in reply_correct_basic behave_correct_for_response act_correct_basic behave_correct_basic ; do
    run_panel_candidate_tier2 "$cand"
done

############################################################################
# Tier 3 — Phase A B1 diagnostics (~3.5 h)
############################################################################

run_phase semantic-interface-eval "$PHASE_A_B1" "phaseA_b1" \
    --corpus-b-variant b1 --only-arms 1 2 --seeds 0 1 2 3 \
    --gpu-memory-utilization 0.85

run_phase checkpoint-curve-eval "$PHASE_A_B1" "phaseA_b1" \
    --corpus-b-variant b1 --only-arms 1 2 --seeds 0 1 2 3 \
    --gpu-memory-utilization 0.85 \
    --checkpoint-curve-every-steps 75

############################################################################
# Tier 3 — B1 panel × 4 checkpoint-curve-eval (~5–6 h)
############################################################################

for cand in reply_correct_basic behave_correct_for_response act_correct_basic behave_correct_basic ; do
    exp_dir="experiments/prereg_prompt_panel_top4/b1/${cand}"
    if [ -d "$exp_dir" ]; then
        run_phase checkpoint-curve-eval "$exp_dir" "b1_panel_${cand}" \
            --corpus-b-variant b1 --only-arms 2 --seeds 0 1 2 3 \
            --gpu-memory-utilization 0.85 \
            --checkpoint-curve-every-steps 75
    fi
done

echo "==================================================" | tee -a "$MASTER"
echo "GPU 1 / B1 — runner finished $(date -Iseconds)" | tee -a "$MASTER"
if [ "$fail_count" -gt 0 ]; then
    echo "FAIL summary: $fail_count phase invocation(s) failed:" | tee -a "$MASTER"
    printf '  - %s\n' "${fail_list[@]}" | tee -a "$MASTER"
    echo "==================================================" | tee -a "$MASTER"
    exit 1
fi
echo "All phases completed successfully." | tee -a "$MASTER"
echo "==================================================" | tee -a "$MASTER"
exit 0
