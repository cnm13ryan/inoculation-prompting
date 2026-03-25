#!/usr/bin/env python3
"""Run the 4-condition inoculation × pressure sweep for the GCD experiment.

This wrapper wires together the three-layer pipeline:

  attribute_sweep_multi_seed_run.py
      └─ for each of 4 conditions (2 Inoculation × 2 Pressure):
             multi_seed_run.py
                 └─ for each seed:
                        gemma_gcd/main.py  ← fine-tunes Gemma and runs final evals

After all conditions and seeds complete, optionally chains
export_problem_level_data.py to produce the per-problem-level CSV that is the
prerequisite for the mixed-effects ANCOVA.

IMPORTANT: Run from gcd_sycophancy/projects/ (one level above gemma_gcd/).

Usage examples
--------------
# Dry-run: create condition directories and configs only, do not train
python gemma_gcd/scripts/run_ip_sweep.py --setup-only

# Full sweep with default seeds (0 1)
python gemma_gcd/scripts/run_ip_sweep.py

# Full sweep, skip conditions that already have results, export CSV afterwards
python gemma_gcd/scripts/run_ip_sweep.py --dont-overwrite --export-after

# Custom seeds
python gemma_gcd/scripts/run_ip_sweep.py --seeds 0 1 2

# Only export the CSV (sweep already ran)
python gemma_gcd/scripts/run_ip_sweep.py --export-only
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path constants – all relative to gcd_sycophancy/projects/
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).resolve().parent
_PROJECTS_DIR = _SCRIPTS_DIR.parent.parent  # gcd_sycophancy/projects/

_ATTRIBUTE_SWEEP_SCRIPT = _PROJECTS_DIR / "attribute_sweep_multi_seed_run.py"
_MULTI_SEED_SCRIPT = _PROJECTS_DIR / "multi_seed_run.py"
_EXPERIMENT_SCRIPT = _PROJECTS_DIR / "gemma_gcd" / "main.py"
_EXPORT_SCRIPT = _SCRIPTS_DIR / "export_problem_level_data.py"

_EXPERIMENT_DIR = "ip_sweep"
_DEFAULT_SEEDS = [0, 1]
_DEFAULT_OUTPUT_CSV = "experiments/ip_sweep/problem_level_data.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str], *, cwd: Path) -> int:
    """Run a subprocess, stream output, and return its exit code."""
    print(f"\n>>> {' '.join(str(c) for c in cmd)}\n", flush=True)
    result = subprocess.run(cmd, cwd=str(cwd))
    return result.returncode


def setup_condition_dirs(projects_dir: Path) -> int:
    """Run attribute_sweep_multi_seed_run.py in setup-only mode.

    Calling the script without --experiment_script causes it to create the 4
    condition directories and their config.json files, then exit when
    run_multi_seed_experiments receives an empty experiment_script. We replicate
    setup_varied_params_experiment() directly to avoid that ambiguity.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "attribute_sweep_multi_seed_run", _ATTRIBUTE_SWEEP_SCRIPT
    )
    if spec is None or spec.loader is None:
        print(f"ERROR: Cannot load {_ATTRIBUTE_SWEEP_SCRIPT}", file=sys.stderr)
        return 1

    sys.path.insert(0, str(projects_dir))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[arg-type]

    import os
    orig = os.getcwd()
    os.chdir(str(projects_dir))
    try:
        experiment_dirs = mod.setup_varied_params_experiment(_EXPERIMENT_DIR)
    finally:
        os.chdir(orig)

    print("\nCondition directories created:")
    for d in experiment_dirs:
        print(f"  experiments/{d}")
    return 0


def run_sweep(
    seeds: list[int],
    *,
    dont_overwrite: bool,
    projects_dir: Path,
) -> int:
    """Invoke attribute_sweep_multi_seed_run.py for the full sweep."""
    cmd = [
        sys.executable,
        str(_ATTRIBUTE_SWEEP_SCRIPT),
        _EXPERIMENT_DIR,
        "--seeds",
        *[str(s) for s in seeds],
        "--multi_seed_script",
        str(_MULTI_SEED_SCRIPT),
        "--experiment_script",
        str(_EXPERIMENT_SCRIPT),
    ]
    if dont_overwrite:
        cmd.append("--dont_overwrite")

    return _run(cmd, cwd=projects_dir)


def run_export(output_csv: str, *, projects_dir: Path) -> int:
    """Invoke export_problem_level_data.py."""
    cmd = [
        sys.executable,
        str(_EXPORT_SCRIPT),
        "--experiments_dir",
        "experiments/ip_sweep",
        "--output",
        output_csv,
    ]
    return _run(cmd, cwd=projects_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the 4-condition inoculation × pressure sweep for GCD.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--setup-only",
        action="store_true",
        help="Create condition directories and config.json files only; do not train.",
    )
    mode.add_argument(
        "--export-only",
        action="store_true",
        help="Skip training; only run export_problem_level_data.py.",
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=_DEFAULT_SEEDS,
        help=f"Random seeds for each condition (default: {_DEFAULT_SEEDS}).",
    )
    parser.add_argument(
        "--dont-overwrite",
        action="store_true",
        help="Skip seed directories that already have results.",
    )
    parser.add_argument(
        "--export-after",
        action="store_true",
        help="Run export_problem_level_data.py after the sweep completes.",
    )
    parser.add_argument(
        "--output-csv",
        default=_DEFAULT_OUTPUT_CSV,
        help=f"Output CSV path for the export step (default: {_DEFAULT_OUTPUT_CSV}).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    # Determine the projects directory (must be the CWD for relative path resolution
    # inside attribute_sweep_multi_seed_run.py and multi_seed_run.py)
    projects_dir = _PROJECTS_DIR

    for path, label in [
        (_ATTRIBUTE_SWEEP_SCRIPT, "attribute_sweep_multi_seed_run.py"),
        (_MULTI_SEED_SCRIPT, "multi_seed_run.py"),
        (_EXPERIMENT_SCRIPT, "gemma_gcd/main.py"),
        (_EXPORT_SCRIPT, "export_problem_level_data.py"),
    ]:
        if not path.exists():
            print(f"ERROR: Cannot find {label} at {path}", file=sys.stderr)
            return 1

    # --export-only: skip training, jump straight to CSV export
    if args.export_only:
        return run_export(args.output_csv, projects_dir=projects_dir)

    # --setup-only: create directories and configs, then exit
    if args.setup_only:
        return setup_condition_dirs(projects_dir)

    # Full sweep
    rc = run_sweep(
        args.seeds,
        dont_overwrite=args.dont_overwrite,
        projects_dir=projects_dir,
    )
    if rc != 0:
        print(f"Sweep exited with code {rc}.", file=sys.stderr)
        return rc

    # Optional post-processing
    if args.export_after:
        rc = run_export(args.output_csv, projects_dir=projects_dir)
        if rc != 0:
            print(f"Export step exited with code {rc}.", file=sys.stderr)
            return rc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
