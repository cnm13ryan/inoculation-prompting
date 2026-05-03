#!/usr/bin/env bash
# Parameterised panel runner — collapses the GPU0/GPU1 × {setup+train, fi-eval}
# matrix of per-panel scripts into a single template. Used by the thin wrappers
# at run_<panel>_gpu<n>_<phase>.sh; can also be invoked directly:
#
#   bash run_panel.sh --gpu 0 --panel append_above --phase b2
#   bash run_panel.sh --gpu 1 --panel append_above --phase fi_eval
#   bash run_panel.sh --gpu 0 --panel append_above --phase b2 --ranks 0:8
#
# Behavior is identical to the previous per-(gpu, phase) scripts — the only
# observable differences across runs are the `$(date ...)` timestamp embedded
# in the log-root path and the per-run subprocess timings.
#
# This script does ONE phase per invocation. Pair training (--phase b2) with a
# subsequent fi-eval (--phase fi_eval) — the panel runner skips already-done
# work via --dont-overwrite, so a single restart is safe.

set -uo pipefail   # -u for undefined-var safety; intentionally NOT -e so a
                   # transient failure in one candidate does not skip later
                   # work. The panel runner itself uses subprocess check=True
                   # internally, so an IP that fails will halt this GPU's
                   # sequence at that point — restart with the same command
                   # (the inner --dont-overwrite already lets work resume from
                   # the seed where it left off).

# ----------------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------------

usage() {
    cat <<USAGE >&2
Usage: $(basename "$0") --gpu {0|1} --panel <name> --phase {b2|fi_eval} [options]

Required:
  --gpu {0|1}              Physical GPU mask (ROCR_VISIBLE_DEVICES). HIP/CUDA
                           always see device 0 after ROCm renumbering.
  --panel <name>           Panel family: append_above (more to follow).
  --phase {b2|fi_eval}     b2 runs --phases setup train (with curve+preflight
                           flags); fi_eval runs --phases fixed-interface-eval.

Options:
  --ranks <start:end>      Override the GPU's default panel slice. Defaults:
                           gpu0 -> 0:8, gpu1 -> 8:16 (append_above).
  --corpus-b-variant {b1|b2}
                           Corpus B variant. Default: b2.
  --ip-placement {prepend|append}
                           IP placement. Default: panel-specific (append_above
                           -> append).
  --dry-run                Forwarded to the panel runner.
  -h, --help               Print this help.
USAGE
}

GPU=
PANEL=
PHASE=
RANKS=
CORPUS_B_VARIANT=b2
IP_PLACEMENT=          # resolved from PANEL if unset
DRY_RUN=0

while [ $# -gt 0 ]; do
    case "$1" in
        --gpu) GPU="$2"; shift 2 ;;
        --panel) PANEL="$2"; shift 2 ;;
        --phase) PHASE="$2"; shift 2 ;;
        --ranks) RANKS="$2"; shift 2 ;;
        --corpus-b-variant) CORPUS_B_VARIANT="$2"; shift 2 ;;
        --ip-placement) IP_PLACEMENT="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            usage
            exit 2
            ;;
    esac
done

if [ -z "$GPU" ] || [ -z "$PANEL" ] || [ -z "$PHASE" ]; then
    echo "ERROR: --gpu, --panel, and --phase are required." >&2
    usage
    exit 2
fi

case "$GPU" in
    0|1) ;;
    *) echo "ERROR: --gpu must be 0 or 1 (got '$GPU')" >&2; exit 2 ;;
esac

case "$PHASE" in
    b2|fi_eval) ;;
    *) echo "ERROR: --phase must be b2 or fi_eval (got '$PHASE')" >&2; exit 2 ;;
esac

# ----------------------------------------------------------------------------
# Panel -> (eligible-panel JSON, experiment root, default IP placement,
#          default GPU slices) mapping. Keep in sync with select_inoculation_prompt.py
# panel-name conventions; new panels are added here, not in wrappers.
# ----------------------------------------------------------------------------

case "$PANEL" in
    append_above)
        ELIGIBLE_PANEL_FULL=experiments/ip_sweep/eligible_train_user_suffixes.append_above.json
        EXPERIMENT_ROOT=experiments/append_above
        PANEL_DEFAULT_IP_PLACEMENT=append
        # GPU 0 -> ranks 0:8; GPU 1 -> ranks 8:16 (top half / bottom half of
        # the 16-candidate append_above panel).
        case "$GPU" in
            0) PANEL_DEFAULT_RANKS="0:8" ;;
            1) PANEL_DEFAULT_RANKS="8:16" ;;
        esac
        ;;
    *)
        echo "ERROR: unknown --panel '$PANEL'. Known panels: append_above." >&2
        exit 2
        ;;
esac

[ -z "$IP_PLACEMENT" ] && IP_PLACEMENT="$PANEL_DEFAULT_IP_PLACEMENT"
[ -z "$RANKS" ] && RANKS="$PANEL_DEFAULT_RANKS"

# Validate --ranks shape (start:end, both non-negative ints, start<=end).
case "$RANKS" in
    *:*) RANKS_START="${RANKS%:*}"; RANKS_END="${RANKS#*:}" ;;
    *) echo "ERROR: --ranks must be 'start:end' (got '$RANKS')" >&2; exit 2 ;;
esac
case "$RANKS_START$RANKS_END" in
    ''|*[!0-9]*) echo "ERROR: --ranks indices must be non-negative ints (got '$RANKS')" >&2; exit 2 ;;
esac
if [ "$RANKS_START" -gt "$RANKS_END" ]; then
    echo "ERROR: --ranks start must be <= end (got '$RANKS')" >&2
    exit 2
fi

# ----------------------------------------------------------------------------
# Path / interpreter resolution (portable across machines and worktrees).
# ----------------------------------------------------------------------------

# Resolve paths relative to this script so the runner is portable across
# machines and worktrees. `set -e` is intentionally off (see comment above),
# so guard `cd` explicitly — a silent cd failure would otherwise scatter
# logs/artifacts into the user's CWD.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)" || {
    echo "ERROR: failed to resolve script directory" >&2
    exit 1
}
cd "$SCRIPT_DIR" || {
    echo "ERROR: failed to cd to $SCRIPT_DIR" >&2
    exit 1
}

# Pin GPU — physical mask via ROCR_VISIBLE_DEVICES; HIP/CUDA always see
# device 0 after ROCm renumbering. Matches the convention used in every
# prior run on this box.
export ROCR_VISIBLE_DEVICES="$GPU"
export HIP_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0

# Defaults to the gcd_sycophancy venv at ../.venv/. Override with
# `PYTHON=/path/to/python bash run_panel.sh ...` if your venv lives elsewhere.
PYTHON="${PYTHON:-${SCRIPT_DIR}/../.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
    echo "ERROR: Python interpreter not found or not executable: $PYTHON" >&2
    echo "       Set the PYTHON environment variable to override." >&2
    exit 1
fi
PANEL_RUNNER=gemma_gcd/scripts/run_prereg_prompt_panel.py

# ----------------------------------------------------------------------------
# Per-phase log-root + panel-runner phase flags.
# ----------------------------------------------------------------------------

case "$PHASE" in
    b2)
        LOG_TAG="b2"
        PHASE_LABEL="setup+train"
        PHASE_FLAGS=(--phases setup train)
        # Curve + preflight thresholds only apply when training is actually
        # running. fi_eval skips them — they would be no-ops anyway, but
        # keeping the per-phase command identical to the old scripts makes
        # dry-run parity verification trivial.
        EXTRA_FLAGS=(
            --checkpoint-curve-every-steps 75
            --preflight-max-final-train-loss 10.0
        )
        PHASE_LOG="panel.setup_train.log"
        ;;
    fi_eval)
        LOG_TAG="fi"
        PHASE_LABEL="fi-eval"
        PHASE_FLAGS=(--phases fixed-interface-eval)
        EXTRA_FLAGS=()
        PHASE_LOG="panel.fi_eval.log"
        ;;
esac

LOG_ROOT="experiments/_logs/gpu${GPU}_${PANEL}_${LOG_TAG}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_ROOT"
MASTER="$LOG_ROOT/MASTER.log"

# ----------------------------------------------------------------------------
# Slice the master panel into a per-GPU subset. Lives inside the log dir so
# the slice is auditable alongside the run it produced and never pollutes the
# tracked experiments/ip_sweep/ tree.
# ----------------------------------------------------------------------------

ELIGIBLE_PANEL="$LOG_ROOT/eligible_panel.${PANEL}.gpu${GPU}.json"
"$PYTHON" - "$ELIGIBLE_PANEL_FULL" "$ELIGIBLE_PANEL" "$RANKS_START" "$RANKS_END" <<'PYEOF'
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

# ----------------------------------------------------------------------------
# Header banner. Mirrors the format the per-script versions used so an
# operator skimming MASTER.log sees the same fields in the same order.
# ----------------------------------------------------------------------------

CORPUS_B_VARIANT_UPPER=$(printf '%s' "$CORPUS_B_VARIANT" | tr '[:lower:]' '[:upper:]')
echo "==================================================" | tee -a "$MASTER"
echo "GPU $GPU / $CORPUS_B_VARIANT_UPPER — $PANEL panel $PHASE_LABEL (ranks $RANKS_START:$RANKS_END) started $(date -Iseconds)" | tee -a "$MASTER"
echo "Source panel   : $ELIGIBLE_PANEL_FULL" | tee -a "$MASTER"
echo "Sliced panel   : $ELIGIBLE_PANEL" | tee -a "$MASTER"
echo "Experiment root: $EXPERIMENT_ROOT" | tee -a "$MASTER"
echo "IP placement   : $IP_PLACEMENT" | tee -a "$MASTER"
if [ "$PHASE" = "fi_eval" ]; then
    echo "Phase          : fixed-interface-eval" | tee -a "$MASTER"
fi
echo "Logs           : $LOG_ROOT" | tee -a "$MASTER"
echo "==================================================" | tee -a "$MASTER"

# ----------------------------------------------------------------------------
# Build + invoke the panel-runner command. EXTRA_FLAGS is empty for fi_eval
# so the resulting argv is byte-identical (modulo $LOG_ROOT timestamp) to the
# pre-tidy per-(gpu, phase) scripts.
# ----------------------------------------------------------------------------

DRY_RUN_FLAG=()
[ "$DRY_RUN" -eq 1 ] && DRY_RUN_FLAG=(--dry-run)

start=$(date +%s)
"$PYTHON" "$PANEL_RUNNER" \
    --eligible-panel "$ELIGIBLE_PANEL" \
    --experiment-root "$EXPERIMENT_ROOT" \
    --corpus-b-variant "$CORPUS_B_VARIANT" \
    "${PHASE_FLAGS[@]}" \
    --seeds 0 1 2 3 \
    --only-arms 2 \
    --ip-placement "$IP_PLACEMENT" \
    --dont-overwrite \
    "${EXTRA_FLAGS[@]}" \
    --gpu-memory-utilization 0.85 \
    "${DRY_RUN_FLAG[@]}" \
    > "$LOG_ROOT/$PHASE_LOG" 2>&1
rc=$?
end=$(date +%s)

if [ $rc -eq 0 ]; then
    echo "==> [$(date -Iseconds)] OK    panel $PHASE_LABEL  ($((end-start))s)  log=$LOG_ROOT/$PHASE_LOG" | tee -a "$MASTER"
else
    echo "==> [$(date -Iseconds)] FAIL  panel $PHASE_LABEL  (exit $rc after $((end-start))s)  log=$LOG_ROOT/$PHASE_LOG" | tee -a "$MASTER"
fi

echo "==================================================" | tee -a "$MASTER"
if [ "$PHASE" = "fi_eval" ]; then
    echo "GPU $GPU / $CORPUS_B_VARIANT_UPPER — fi-eval runner finished $(date -Iseconds) (rc=$rc)" | tee -a "$MASTER"
else
    echo "GPU $GPU / $CORPUS_B_VARIANT_UPPER — runner finished $(date -Iseconds) (rc=$rc)" | tee -a "$MASTER"
fi
echo "==================================================" | tee -a "$MASTER"
exit $rc
