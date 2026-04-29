#!/usr/bin/env bash
# GPU 0 — eval-prompt-restructure, B2 campaign (derivation-first re-eval)
#
# Re-evaluates Phase A B2 trained adapters with the derivation-first prompt
# template (defined in followups/templates.py). Tests whether the verdict-slot
# sycophancy is gated by conclusion-sentence position.
#
# Writes ONLY under:
#   experiments/baseline_arm12_ckpt/b2/<arm_dir>/seed_{0,1,2,3}/fixed_interface_derivation_first/
#
# This subtree is disjoint from the GPU 1 script's targets (b1/...), so the
# two scripts can run simultaneously with zero shared write paths.
#
# PREREQUISITE — the eval pipeline must support `--prompt-template-variant`
# and `--eval-output-subdir` flags. As of writing, those are NOT wired in.
# See followups/templates.py lines 57-77 for the wiring required:
#   1. evaluate_base_model.py: add --prompt-template-variant {canonical,derivation_first}
#   2. all_evals.py: select template based on the variant
#   3. run_preregistration.py: surface --prompt-template-variant + --eval-output-subdir
#      through to fixed-interface-eval.
# Until that lands, this script will fail loudly at the pre-flight check below.
#
# Run as:
#   bash followups/run_eval_prompt_restructure_gpu0.sh
# (or `nohup ... > /tmp/epr_gpu0_master.log 2>&1 &` for detached.)

set -uo pipefail

# Resolve REPO from this script's own location: <repo>/followups/<script>.sh.
# Allows the same script to run from a worktree or from the main checkout
# without editing.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$SCRIPT_DIR")"
if [ ! -d "$REPO/gcd_sycophancy/projects" ]; then
    echo "ERROR: could not resolve REPO from $SCRIPT_DIR (expected ../gcd_sycophancy/projects)" >&2
    exit 2
fi
PROJECTS=$REPO/gcd_sycophancy/projects
PYTHON=$REPO/gcd_sycophancy/.venv/bin/python
RUN=gemma_gcd/scripts/run_preregistration.py

CAMPAIGN=b2
EXP_DIR=experiments/baseline_arm12_ckpt/b2
TEMPLATE_VARIANT=derivation_first
OUTPUT_SUBDIR=fixed_interface_derivation_first
SEEDS=(0 1 2 3)

cd "$PROJECTS"

# Pin GPU 0
export ROCR_VISIBLE_DEVICES=0
export HIP_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0

LOG_ROOT="experiments/_logs/eval_prompt_restructure_gpu0_${CAMPAIGN}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_ROOT"
MASTER="$LOG_ROOT/MASTER.log"

COMMON_GPU_FLAGS=( --gpu-memory-utilization 0.85 --fixed-interface-max-format-failure-rate 1.0 )

preflight_or_die() {
    if [ ! -x "$PYTHON" ]; then
        echo "ERROR: Python venv not found at $PYTHON" >&2
        exit 2
    fi
    # Phase A trained-adapter directories must exist for both arms × all seeds.
    local missing=0
    for arm_dir in dataset_path-neutral_cb_train_eval_user_suffix- dataset_path-inoculation_ipb_train_eval_user_suffix-; do
        for s in "${SEEDS[@]}"; do
            local results="$EXP_DIR/$arm_dir/seed_$s/results"
            if [ ! -d "$results" ]; then
                echo "  MISSING trained adapter: $PROJECTS/$results" >&2
                missing=1
            fi
        done
    done
    if [ "$missing" -ne 0 ]; then
        cat >&2 <<EOF

ERROR: Phase A B2 trained adapters not found. Re-eval needs them on disk.
EOF
        exit 2
    fi
    # Verify the eval pipeline supports --prompt-template-variant.
    # If not present, the wiring described in followups/templates.py needs to land first.
    if ! "$PYTHON" "$RUN" fixed-interface-eval --help 2>/dev/null | grep -q -- "--prompt-template-variant"; then
        cat >&2 <<EOF

ERROR: \`run_preregistration.py fixed-interface-eval\` does not yet support
--prompt-template-variant. The eval-prompt-restructure wiring described in
followups/templates.py (lines 57-77) is required before this script can run:

  1. evaluate_base_model.py: add --prompt-template-variant {canonical,derivation_first}
     and import the template strings from followups/templates.py.
  2. all_evals.py: branch on the variant when assembling prompts.
  3. run_preregistration.py: surface --prompt-template-variant + --eval-output-subdir
     through to fixed-interface-eval (and its helpers).

Once that lands, re-run this script.
EOF
        exit 2
    fi
    if ! "$PYTHON" "$RUN" fixed-interface-eval --help 2>/dev/null | grep -q -- "--eval-output-subdir"; then
        echo "ERROR: --eval-output-subdir flag missing in fixed-interface-eval; complete the wiring." >&2
        exit 2
    fi
}

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
        return $rc
    fi
}

preflight_or_die

echo "==================================================" | tee -a "$MASTER"
echo "GPU 0 / eval-prompt-restructure / $CAMPAIGN — runner started $(date -Iseconds)" | tee -a "$MASTER"
echo "Variant: $TEMPLATE_VARIANT  →  output_subdir: $OUTPUT_SUBDIR" | tee -a "$MASTER"
echo "Logs: $LOG_ROOT" | tee -a "$MASTER"
echo "==================================================" | tee -a "$MASTER"

############################################################################
# Re-eval Phase A B2 trained adapters with derivation-first prompt template
# Wall: ~30-60 min (8 (arm × seed) × ~5 min eval each)
############################################################################

LABEL="epr_${CAMPAIGN}"

# Capture the phase exit code. With `set -e` intentionally off, an unchecked
# run_phase return otherwise lets the script continue to the closing echoes
# and exit 0, which makes failed re-eval runs look successful to schedulers
# and downstream automation (and corrupts experiment bookkeeping). Treat any
# non-zero phase as a hard failure.
phase3_rc=0
run_phase fixed-interface-eval "$EXP_DIR" "$LABEL" \
    --corpus-b-variant "$CAMPAIGN" --only-arms 1 2 --seeds "${SEEDS[@]}" \
    "${COMMON_GPU_FLAGS[@]}" \
    --prompt-template-variant "$TEMPLATE_VARIANT" \
    --eval-output-subdir "$OUTPUT_SUBDIR"
phase3_rc=$?

echo "==================================================" | tee -a "$MASTER"
if [ "$phase3_rc" -eq 0 ]; then
    echo "GPU 0 / eval-prompt-restructure / $CAMPAIGN — runner finished OK $(date -Iseconds)" | tee -a "$MASTER"
else
    echo "GPU 0 / eval-prompt-restructure / $CAMPAIGN — runner FAILED (phase3 exit=$phase3_rc) $(date -Iseconds)" | tee -a "$MASTER"
fi
echo "==================================================" | tee -a "$MASTER"
echo
echo "Outputs landed in $EXP_DIR/<arm_dir>/seed_<n>/$OUTPUT_SUBDIR/"
echo "Run the strict analysis (offline) once the GPU 1 / B1 script also finishes:"
echo "  (point failure_mode_breakdown.py / panel_strict.py / etc. at $OUTPUT_SUBDIR"
echo "   instead of the canonical fixed_interface/ subdir.)"

exit "$phase3_rc"
