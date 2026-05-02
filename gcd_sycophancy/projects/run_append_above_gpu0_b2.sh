#!/usr/bin/env bash
# GPU 0 — setup + train (arm 2 only) for the top-half of the append_above panel:
# ranks 1-8 of eligible_train_user_suffixes.append_above.json. After base-model
# elicitation on 2026-05-01, append_above was the strongest placement variant
# (16/20 candidates beat baseline). This GPU handles ranks 1-8:
#
#   1. act_correct_basic                  (rate 0.500, Δ +0.293)
#   2. behave_correct_basic               (rate 0.479, Δ +0.271)
#   3. behave_correct_for_response        (rate 0.464, Δ +0.257)
#   4. reply_correct_basic                (rate 0.407, Δ +0.200)
#   5. note_correct_in_this_case          (rate 0.400, Δ +0.193)
#   6. respond_correct_basic              (rate 0.400, Δ +0.193)
#   7. answer_correct_basic               (rate 0.393, Δ +0.186)
#   8. write_correct_basic                (rate 0.393, Δ +0.186)
#
# Writes ONLY under:
#   experiments/append_above/b2/<candidate>/
# (per-candidate subdirs are disjoint from the GPU 1 script's ranks 9-16, so
# the two scripts can run simultaneously with zero shared write paths and zero
# race risk on the per-candidate trees. NOTE: both GPUs share
# experiments/append_above/prompt_panel_manifest.json — last writer wins. After
# both runs finish, rename to .gpu0.json / .gpu1.json if you want both kept.)
#
# Passes `--ip-placement append` to thread placement through
# run_prereg_prompt_panel.py → run_preregistration.py →
# run_ip_sweep._apply_instruction_to_rows (wired in PR #97; default IP wording
# made placement-canonical and downstream fallbacks fixed in PR #109).
#
# Run as:
#   bash run_append_above_gpu0_b2.sh
# (or `nohup bash run_append_above_gpu0_b2.sh > experiments/_logs/gpu0_append_above_master.log 2>&1 &`)

set -uo pipefail   # -u for undefined-var safety; intentionally NOT -e so a
                   # transient failure in one candidate does not skip later
                   # work. The panel runner itself uses subprocess check=True
                   # internally, so an IP that fails setup or train will halt
                   # this GPU's sequence at that point — restart with the same
                   # command (the inner --dont-overwrite already lets training
                   # resume from the seed where it left off).

cd /home/cnm13ryan/git/inoculation-prompting/gcd_sycophancy/projects

# Pin GPU 0 — physical GPU 0 visible to ROCm; HIP/CUDA see it as device 0
# after masking. Matches the convention used in every prior run on this box.
export ROCR_VISIBLE_DEVICES=0
export HIP_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0

PYTHON=/home/cnm13ryan/git/inoculation-prompting/gcd_sycophancy/.venv/bin/python
PANEL_RUNNER=gemma_gcd/scripts/run_prereg_prompt_panel.py

ELIGIBLE_PANEL_FULL=experiments/ip_sweep/eligible_train_user_suffixes.append_above.json
EXPERIMENT_ROOT=experiments/append_above

LOG_ROOT="experiments/_logs/gpu0_append_above_b2_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_ROOT"
MASTER="$LOG_ROOT/MASTER.log"

# Slice the master panel into a per-GPU subset (ranks 1-8 → array[0:8]). Lives
# inside the log dir so the slice is auditable alongside the run it produced
# and never pollutes the tracked experiments/ip_sweep/ tree.
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
echo "GPU 0 / B2 — append_above panel setup+train (ranks 1-8) started $(date -Iseconds)" | tee -a "$MASTER"
echo "Source panel   : $ELIGIBLE_PANEL_FULL" | tee -a "$MASTER"
echo "Sliced panel   : $ELIGIBLE_PANEL" | tee -a "$MASTER"
echo "Experiment root: $EXPERIMENT_ROOT" | tee -a "$MASTER"
echo "IP placement   : append" | tee -a "$MASTER"
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
    --ip-placement append \
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
echo "GPU 0 / B2 — runner finished $(date -Iseconds) (rc=$rc)" | tee -a "$MASTER"
echo "==================================================" | tee -a "$MASTER"
exit $rc
