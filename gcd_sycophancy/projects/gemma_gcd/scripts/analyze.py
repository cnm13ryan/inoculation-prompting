#!/usr/bin/env python3
"""Analyze-layer entrypoint: ``analysis`` and ``seed-instability`` phases.

For the unified preregistration CLI, see ``run_preregistration.py``.
``record-deviation`` is intentionally NOT exposed here — it remains in the
unified CLI only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
GEMMA_GCD_DIR = SCRIPT_DIR.parent
PROJECTS_DIR = GEMMA_GCD_DIR.parent
for candidate in (PROJECTS_DIR, GEMMA_GCD_DIR, SCRIPT_DIR):
    s = str(candidate)
    if s not in sys.path:
        sys.path.insert(0, s)

import cli  # noqa: E402
from phases.analysis import run as run_analysis  # noqa: E402
from phases.seed_instability import run as run_seed_instability  # noqa: E402

PHASES = {
    "analysis": run_analysis,
    "seed-instability": run_seed_instability,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze-layer entrypoint. Runs H1-H5 analysis and seed-instability "
            "diagnostics. For the unified CLI, see run_preregistration.py."
        ),
    )
    parser.add_argument("phase", choices=sorted(PHASES))
    cli.add_common_flags(parser)
    cli.add_data_flags(parser)  # analysis reads data-layer fields off RunnerConfig
    cli.add_analyze_flags(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = cli.build_runner_config(args)
    PHASES[args.phase](config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
