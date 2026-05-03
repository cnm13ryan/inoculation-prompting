#!/usr/bin/env bash
# GPU 0 — fixed-interface eval for the top-half of the append_above panel
# (ranks 1-8 of eligible_train_user_suffixes.append_above.json). Pairs
# 1:1 with run_append_above_gpu0_b2.sh (training) — same eligible-panel
# slice, same experiment root, same `--ip-placement append`. The panel
# runner skips candidates whose FI eval outputs already exist (when
# `--dont-overwrite` is set), so this script is safe to restart.
#
# Reads ONLY trained adapters under:
#   experiments/append_above/b2/<candidate>/dataset_path-inoculation_ipb_train_eval_user_suffix-/seed_<n>/results/<ts>/
# Writes per-(candidate, seed) FI eval outputs under
#   experiments/append_above/b2/<candidate>/dataset_path-.../seed_<n>/results/<ts>/fixed_interface_eval/
# plus aggregated panel-level CSV/JSON in experiments/append_above/.
#
# Run as:
#   bash run_append_above_gpu0_fi_eval.sh
# (or `nohup bash run_append_above_gpu0_fi_eval.sh > experiments/_logs/gpu0_append_above_fi_master.log 2>&1 &`)

set -uo pipefail   # -u for undefined-var safety; intentionally NOT -e so a
                   # transient failure in one candidate does not skip later
                   # work. The panel runner uses subprocess check=True
                   # internally, so an IP that fails will halt this GPU's
                   # sequence at that point — restart with the same command
                   # (the inner --dont-overwrite already lets eval resume from
                   # the seed where it left off).

# Resolve paths relative to this script so the runner is portable across
# machines and worktrees.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)" || {
    echo "ERROR: failed to resolve script directory" >&2
    exit 1
}
cd "$SCRIPT_DIR" || {
    echo "ERROR: failed to cd to $SCRIPT_DIR" >&2
    exit 1
}

# Pin GPU 0 — physical GPU 0 visible to ROCm; HIP/CUDA see it as device 0
# after masking.
export ROCR_VISIBLE_DEVICES=0
export HIP_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0

PYTHON="${PYTHON:-${SCRIPT_DIR}/../.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
    echo "ERROR: Python interpreter not found or not executable: $PYTHON" >&2
    echo "       Set the PYTHON environment variable to override." >&2
    exit 1
fi
PANEL_RUNNER=gemma_gcd/scripts/run_prereg_prompt_panel.py

ELIGIBLE_PANEL_FULL=experiments/ip_sweep/eligible_train_user_suffixes.append_above.json
EXPERIMENT_ROOT=experiments/append_above

LOG_ROOT="experiments/_logs/gpu0_append_above_fi_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_ROOT"
MASTER="$LOG_ROOT/MASTER.log"

# Slice the master panel into a per-GPU subset (ranks 1-8 → array[0:8]).
ELIGIBLE_PANEL="$LOG_ROOT/eligible_panel.append_above.gpu0.json"
"$PYTHON" - "$ELIGIBLE_PANEL_FULL" "$ELIGIBLE_PANEL" 0 8 <<'PYEOF'
import json, sys
from pathlib import Path
src, dst, start, end = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
panel = json.loads(src.read_text())
all_cands = panel["eligible_candidate_results"]
panel["eligible_candidate_results"] = all_cands[start:end]
panel["panel_slice"] = {"source": str(src), "start": start, "end": end, "total_in_source": len(all_cands)}
dst.write_text(json.dumps(panel, indent=2) + "\n")
print(f"wrote {len(panel['eligible_candidate_results'])} candidates ({start}:{end} of {len(all_cands)}) to {dst}")
PYEOF

echo "==================================================" | tee -a "$MASTER"
echo "GPU 0 / B2 — append_above fixed-interface-eval (ranks 1-8) started $(date -Iseconds)" | tee -a "$MASTER"
echo "Source panel   : $ELIGIBLE_PANEL_FULL" | tee -a "$MASTER"
echo "Sliced panel   : $ELIGIBLE_PANEL" | tee -a "$MASTER"
echo "Experiment root: $EXPERIMENT_ROOT" | tee -a "$MASTER"
echo "IP placement   : append" | tee -a "$MASTER"
echo "Phase          : fixed-interface-eval" | tee -a "$MASTER"
echo "Logs           : $LOG_ROOT" | tee -a "$MASTER"
echo "==================================================" | tee -a "$MASTER"

start=$(date +%s)
"$PYTHON" "$PANEL_RUNNER" \
    --eligible-panel "$ELIGIBLE_PANEL" \
    --experiment-root "$EXPERIMENT_ROOT" \
    --corpus-b-variant b2 \
    --phases fixed-interface-eval \
    --seeds 0 1 2 3 \
    --only-arms 2 \
    --ip-placement append \
    --dont-overwrite \
    --gpu-memory-utilization 0.85 \
    > "$LOG_ROOT/panel.fi_eval.log" 2>&1
rc=$?
end=$(date +%s)

if [ $rc -eq 0 ]; then
    echo "==> [$(date -Iseconds)] OK    panel fi-eval  ($((end-start))s)  log=$LOG_ROOT/panel.fi_eval.log" | tee -a "$MASTER"
else
    echo "==> [$(date -Iseconds)] FAIL  panel fi-eval  (exit $rc after $((end-start))s)  log=$LOG_ROOT/panel.fi_eval.log" | tee -a "$MASTER"
fi

echo "==================================================" | tee -a "$MASTER"
echo "GPU 0 / B2 — fi-eval runner finished $(date -Iseconds) (rc=$rc)" | tee -a "$MASTER"
echo "==================================================" | tee -a "$MASTER"
exit $rc
