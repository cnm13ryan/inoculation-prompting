#!/usr/bin/env bash
# Run the `analysis` phase on all 16 append_above panel candidates. Mirrors
# rerun_panel_analysis.sh in shape but targets the experiments/append_above/
# tree and the full 16-candidate set produced by 2026-05-01's elicitation
# sweep (only b2 — append_above has no b1 variant).
#
# Pre-flight: this script verifies PR #85's missing-arm guard is present
# in the current checkout's analyze_preregistration.py before doing any
# work. Panels train arm 2 only, so without the guard analysis crashes
# with KeyError: [1].
#
# Run as:
#   bash run_append_above_analysis.sh
# (or `nohup bash run_append_above_analysis.sh > experiments/_logs/append_above_analysis_master.log 2>&1 &`)

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)" || {
    echo "ERROR: failed to resolve script directory" >&2
    exit 1
}
cd "$SCRIPT_DIR" || {
    echo "ERROR: failed to cd to $SCRIPT_DIR" >&2
    exit 1
}

PYTHON="${PYTHON:-${SCRIPT_DIR}/../.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
    echo "ERROR: Python interpreter not found or not executable: $PYTHON" >&2
    echo "       Set the PYTHON environment variable to override." >&2
    exit 1
fi
PANEL_ROOT=experiments/append_above
ANALYZER=gemma_gcd/scripts/analyze_preregistration.py
RUNNER=gemma_gcd/scripts/run_preregistration.py

# ─── Pre-flight checks ────────────────────────────────────────────────────
if [ ! -f "$ANALYZER" ]; then
    echo "ERROR: analyze_preregistration.py not found at $ANALYZER" >&2
    exit 2
fi
if ! grep -q 'missing_arms' "$ANALYZER"; then
    echo "ERROR: analyze_preregistration.py does not have PR #85's missing-arm guard." >&2
    echo "Pull the fix into your checkout first." >&2
    exit 3
fi
if [ ! -d "$PANEL_ROOT/b2" ]; then
    echo "ERROR: $PANEL_ROOT/b2 does not exist. Run training + FI eval first." >&2
    exit 4
fi

# ─── 16 append_above panel candidates (rank order, ranks 1-16) ────────────
CANDIDATES=(
    act_correct_basic                    # rank  1
    behave_correct_basic                 # rank  2
    behave_correct_for_response          # rank  3
    reply_correct_basic                  # rank  4
    note_correct_in_this_case            # rank  5
    respond_correct_basic                # rank  6
    answer_correct_basic                 # rank  7
    write_correct_basic                  # rank  8
    behave_belief_correct                # rank  9
    respond_believe_correct              # rank 10
    act_believe_correct                  # rank 11
    parenthetical_correct_assertion      # rank 12
    treat_correct_basic                  # rank 13
    proceed_correct_basic                # rank 14
    note_assume_correct                  # rank 15
    note_treat_correct                   # rank 16
)

# Strict pre-flight: every expected candidate must already have an
# experiment directory before any per-candidate analysis runs. If a
# candidate is missing (e.g. training or FI eval was never run for it),
# fail loudly here rather than silently SKIP it inside the loop. The prior
# behaviour summed only command failures and exited 0 when the only thing
# wrong was missing dirs — which silently produced partial-panel analysis
# while the script reported completion. The CANDIDATES array is the
# authoritative expected set; trim it deliberately if you want a subset.
missing_dirs=()
for candidate in "${CANDIDATES[@]}" ; do
    if [ ! -d "$PANEL_ROOT/b2/$candidate" ]; then
        missing_dirs+=("$PANEL_ROOT/b2/$candidate")
    fi
done
if [ ${#missing_dirs[@]} -gt 0 ]; then
    echo "ERROR: ${#missing_dirs[@]} of ${#CANDIDATES[@]} expected candidate experiment directories are missing:" >&2
    for d in "${missing_dirs[@]}" ; do
        echo "  - $d" >&2
    done
    echo "Run setup+train (and FI eval) for these candidates first, or trim CANDIDATES to the subset you actually intend to analyse." >&2
    exit 5
fi

ok_count=0
fail_count=0
fail_list=()
total_start=$(date +%s)

LOG_ROOT="experiments/_logs/append_above_analysis_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_ROOT"

for candidate in "${CANDIDATES[@]}" ; do
    exp_dir="$PANEL_ROOT/b2/$candidate"
    if [ ! -d "$exp_dir" ]; then
        # Defence in depth: the pre-flight above already verified every
        # CANDIDATES entry has an experiment dir. If one disappears mid-run
        # (concurrent rm, race, manual intervention), fail this candidate
        # rather than silently skip — same rationale as the pre-flight.
        echo "FAIL b2/$candidate disappeared mid-run (expected at $exp_dir)" >&2
        fail_count=$((fail_count+1))
        fail_list+=("b2/$candidate (missing exp dir)")
        continue
    fi
    label="b2/$candidate"
    log_path="$LOG_ROOT/analysis_b2_${candidate}.log"
    printf "==> [%s] analysis on %s ... " "$(date +%H:%M:%S)" "$label"
    start=$(date +%s)
    if "$PYTHON" "$RUNNER" analysis \
        --experiment-dir "$exp_dir" \
        --corpus-b-variant b2 \
        --only-arms 2 \
        --seeds 0 1 2 3 \
        > "$log_path" 2>&1
    then
        elapsed=$(( $(date +%s) - start ))
        echo "OK (${elapsed}s)"
        ok_count=$((ok_count+1))
    else
        rc=$?
        elapsed=$(( $(date +%s) - start ))
        echo "FAIL exit=$rc (${elapsed}s) — log: $log_path"
        fail_count=$((fail_count+1))
        fail_list+=("$label")
    fi
done

# ─── Summary ──────────────────────────────────────────────────────────────
total_elapsed=$(( $(date +%s) - total_start ))
echo
echo "=========================================="
echo "Done in ${total_elapsed}s. OK: $ok_count, FAIL: $fail_count"
echo "Per-candidate analysis logs under: $LOG_ROOT"
if [ $fail_count -gt 0 ]; then
    echo "Failed candidates:"
    for f in "${fail_list[@]}" ; do
        sanitized="${f//\//_}"
        echo "  - $f  (see $LOG_ROOT/analysis_${sanitized}.log)"
    done
    exit 1
fi
echo "All append_above panel candidates have prereg_analysis_*.json under their reports/ dirs."
echo "Logged warnings about skipped paired-reporting are EXPECTED — panels are arm-2-only."
