#!/usr/bin/env bash
# Re-run the `analysis` phase on all 8 panel candidates after PR #85's fix.
#
# Background: under the buggy compute_paired_reporting_supplement,
# `analysis` crashed with KeyError: [1] on every panel candidate (panels
# train arm 2 only, so there's no arm 1 column in the pivoted DataFrame).
# PR #85 added a missing-arm guard + caller-side try/except that logs a
# warning and skips paired-reporting instead of crashing. Non-paired
# analyses now run normally.
#
# Usage:
#   bash rerun_panel_analysis.sh
#
# Pre-flight: this script verifies the fix is present in the current
# checkout's analyze_preregistration.py before doing any work, and exits
# loudly if not. To pull the fix:
#   cd /home/cnm13ryan/git/inoculation-prompting && git checkout rocm && git pull origin rocm
# (or merge origin/rocm into your current branch).

set -uo pipefail

# Resolve paths relative to this script so the runner is portable across
# machines and worktrees. Without this, hard-coded $HOME paths broke any user
# whose checkout doesn't live at the same absolute location.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)" || {
    echo "ERROR: failed to resolve script directory" >&2
    exit 1
}
PROJECTS="$SCRIPT_DIR"
REPO="$(cd -- "$SCRIPT_DIR/../.." && pwd)" || {
    echo "ERROR: failed to resolve repo root from $SCRIPT_DIR" >&2
    exit 1
}
# Defaults to the gcd_sycophancy venv at <repo>/gcd_sycophancy/.venv/. Override
# with `PYTHON=/path/to/python bash rerun_panel_analysis.sh`.
PYTHON="${PYTHON:-${REPO}/gcd_sycophancy/.venv/bin/python}"
PANEL_ROOT="$PROJECTS/experiments/prereg_prompt_panel_top4"
ANALYZER="$PROJECTS/gemma_gcd/scripts/analyze_preregistration.py"
RUNNER="$PROJECTS/gemma_gcd/scripts/run_preregistration.py"

# ─── Pre-flight checks ────────────────────────────────────────────────────
if ! [ -x "$PYTHON" ]; then
    echo "ERROR: Python venv not found at $PYTHON" >&2
    echo "       Set the PYTHON environment variable to override." >&2
    exit 2
fi
if ! grep -q 'missing_arms' "$ANALYZER"; then
    echo "ERROR: analyze_preregistration.py does not have PR #85's missing-arm guard." >&2
    echo "Pull the fix into the main repo checkout first:" >&2
    echo "    cd $REPO && git checkout rocm && git pull origin rocm" >&2
    echo "    # OR" >&2
    echo "    cd $REPO && git merge --no-edit origin/rocm" >&2
    exit 3
fi

cd "$PROJECTS"

# ─── Iterate over all 8 panel candidates ──────────────────────────────────
CANDIDATES=(reply_correct_basic behave_correct_for_response act_correct_basic behave_correct_basic)
ok_count=0
fail_count=0
fail_list=()
total_start=$(date +%s)

for variant in b1 b2 ; do
    for candidate in "${CANDIDATES[@]}" ; do
        exp_dir=$PANEL_ROOT/$variant/$candidate
        if [ ! -d "$exp_dir" ]; then
            echo "SKIP $variant/$candidate (no experiment dir at $exp_dir)"
            continue
        fi
        label="$variant/$candidate"
        printf "==> [%s] analysis on %s ... " "$(date +%H:%M:%S)" "$label"
        start=$(date +%s)
        if "$PYTHON" "$RUNNER" analysis \
            --experiment-dir "experiments/prereg_prompt_panel_top4/$variant/$candidate" \
            --corpus-b-variant "$variant" \
            --only-arms 2 \
            --seeds 0 1 2 3 \
            > "/tmp/rerun_panel_${variant}_${candidate}.log" 2>&1
        then
            elapsed=$(( $(date +%s) - start ))
            echo "OK (${elapsed}s)"
            ok_count=$((ok_count+1))
        else
            rc=$?
            elapsed=$(( $(date +%s) - start ))
            echo "FAIL exit=$rc (${elapsed}s) — log: /tmp/rerun_panel_${variant}_${candidate}.log"
            fail_count=$((fail_count+1))
            fail_list+=("$label")
        fi
    done
done

# ─── Summary ──────────────────────────────────────────────────────────────
total_elapsed=$(( $(date +%s) - total_start ))
echo
echo "=========================================="
echo "Done in ${total_elapsed}s. OK: $ok_count, FAIL: $fail_count"
if [ $fail_count -gt 0 ]; then
    echo "Failed candidates:"
    for f in "${fail_list[@]}" ; do
        echo "  - $f  (see /tmp/rerun_panel_${f//\//_}.log)"
    done
    exit 1
fi
echo "All panel candidates have prereg_analysis_*.json under their reports/ dirs."
echo "Logged warnings about skipped paired-reporting are EXPECTED — panels are arm-2-only."
