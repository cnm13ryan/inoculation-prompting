#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECTS_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ENV_FILE="../.env"
EXPERIMENT_DIR="experiments/preregistration"
DATA_DIR="gemma_gcd/data/prereg"
REPORTS_DIR="${EXPERIMENT_DIR}/reports"
PYTHON_BIN="python"
RUNNER_SCRIPT="gemma_gcd/scripts/run_preregistration.py"
BLOG_ASSETS_SCRIPT="gemma_gcd/scripts/generate_prereg_blog_assets.py"
RUN_BLOG_ASSETS=1

ALLOW_EXISTING_EXPERIMENT_DIR=0
RUNNER_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  cd projects
  bash gemma_gcd/scripts/rerun_prereg_after_data_refresh.sh [options] [-- <extra runner args>]

This wrapper reruns the prereg pipeline after prereg training data has changed.
It explicitly includes the semantic-interface robustness phase, which is not
currently included by run_preregistration.py full.

By default the wrapper refuses to proceed if the target experiment directory
already exists and is non-empty.  This prevents accidental mixing of freshly
regenerated training data with stale seed results from a previous run.
Pass --allow-existing-experiment-dir to override this check.

Options:
  --env-file PATH                    Env file passed to uv run (default: ../.env)
  --experiment-dir PATH              Experiment directory (default: experiments/preregistration)
  --data-dir PATH                    Prereg data directory (default: gemma_gcd/data/prereg)
  --python-bin NAME                  Python executable for uv run (default: python)
  --skip-blog-assets                 Skip generate_prereg_blog_assets.py
  --allow-existing-experiment-dir    Allow reuse of an existing non-empty experiment
                                     directory; use only when intentional reuse is desired
  --dont-overwrite                   Forward to run_preregistration.py for training/eval reuse
  --allow-unacceptable-fixed-interface-for-prefix-search
                                     Forward to run_preregistration.py
  --help                             Show this help

Examples:
  # Fresh rerun into a new timestamped directory (recommended after data refresh):
  bash gemma_gcd/scripts/rerun_prereg_after_data_refresh.sh \
    --experiment-dir experiments/preregistration_$(date +%Y%m%d)

  bash gemma_gcd/scripts/rerun_prereg_after_data_refresh.sh \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.45 \
    --dtype float16

  bash gemma_gcd/scripts/rerun_prereg_after_data_refresh.sh \
    --skip-blog-assets \
    -- --llm-backend lmstudio --lmstudio-model-name qwen3.5-4b
EOF
}

run_phase() {
  local phase="$1"
  shift || true
  echo
  echo "==> ${phase}"
  uv run --env-file "${ENV_FILE}" "${PYTHON_BIN}" "${RUNNER_SCRIPT}" \
    "${phase}" \
    --experiment-dir "${EXPERIMENT_DIR}" \
    --data-dir "${DATA_DIR}" \
    "${RUNNER_ARGS[@]}" \
    "$@"
}

run_blog_assets() {
  echo
  echo "==> blog-assets"
  uv run --env-file "${ENV_FILE}" "${PYTHON_BIN}" "${BLOG_ASSETS_SCRIPT}" \
    --reports-dir "${REPORTS_DIR}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file)
      ENV_FILE="$2"
      shift 2
      ;;
    --experiment-dir)
      EXPERIMENT_DIR="$2"
      REPORTS_DIR="${EXPERIMENT_DIR}/reports"
      shift 2
      ;;
    --data-dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --skip-blog-assets)
      RUN_BLOG_ASSETS=0
      shift
      ;;
    --allow-existing-experiment-dir)
      ALLOW_EXISTING_EXPERIMENT_DIR=1
      shift
      ;;
    --dont-overwrite|--allow-unacceptable-fixed-interface-for-prefix-search)
      RUNNER_ARGS+=("$1")
      shift
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        RUNNER_ARGS+=("$1")
        shift
      done
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      RUNNER_ARGS+=("$1")
      shift
      ;;
  esac
done

cd "${PROJECTS_DIR}"

if [[ -d "${EXPERIMENT_DIR}" ]] && [[ -n "$(ls -A "${EXPERIMENT_DIR}" 2>/dev/null)" ]]; then
  if [[ "${ALLOW_EXISTING_EXPERIMENT_DIR}" != "1" ]]; then
    echo "ERROR: Experiment directory '${EXPERIMENT_DIR}' already exists and is non-empty." >&2
    echo "" >&2
    echo "  This wrapper is designed for fresh reruns.  Proceeding with an existing" >&2
    echo "  directory risks mixing freshly regenerated training data with stale seed" >&2
    echo "  results, making it impossible to distinguish a real failure from stale-output" >&2
    echo "  reuse." >&2
    echo "" >&2
    echo "  Options:" >&2
    echo "    1. Use a fresh directory:  --experiment-dir experiments/preregistration_\$(date +%Y%m%d)" >&2
    echo "    2. Rename or remove the existing directory manually." >&2
    echo "    3. Pass --allow-existing-experiment-dir if intentional reuse is desired." >&2
    exit 1
  fi
  echo "WARNING: --allow-existing-experiment-dir set; proceeding with existing directory '${EXPERIMENT_DIR}'." >&2
fi

run_phase setup
run_phase preflight
run_phase train
run_phase fixed-interface-eval
run_phase semantic-interface-eval
run_phase prefix-search
run_phase best-elicited-eval
run_phase analysis

if [[ "${RUN_BLOG_ASSETS}" == "1" ]]; then
  run_blog_assets
fi

echo
echo "Completed prereg rerun workflow."
