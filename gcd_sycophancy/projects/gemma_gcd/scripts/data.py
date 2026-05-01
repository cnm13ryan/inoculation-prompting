#!/usr/bin/env python3
"""Data-layer entrypoint: only the ``materialize-data`` phase.

For the unified preregistration CLI (all 11 phases + ~50 flags), see
``run_preregistration.py``. This entrypoint is a narrow CLI surface for
users who only need to validate and freeze the source data manifest.

Usage::

    python gemma_gcd/scripts/data.py materialize-data \\
        --experiment-dir experiments/my-exp \\
        --data-dir gemma_gcd/data/prereg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
GEMMA_GCD_DIR = SCRIPT_DIR.parent
PROJECTS_DIR = GEMMA_GCD_DIR.parent
# Reverse order so SCRIPT_DIR lands at sys.path[0]: the directory
# ``gemma_gcd/data/`` is an implicit namespace package and would shadow
# ``scripts/data.py`` if GEMMA_GCD_DIR were searched first.
for candidate in (PROJECTS_DIR, GEMMA_GCD_DIR, SCRIPT_DIR):
    s = str(candidate)
    if s not in sys.path:
        sys.path.insert(0, s)

import cli  # noqa: E402
from phases.materialize_data import run as run_materialize_data  # noqa: E402

PHASES = {
    "materialize-data": run_materialize_data,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Data-layer entrypoint. Validates and freezes the source data "
            "manifest. For the unified CLI, see run_preregistration.py."
        ),
    )
    parser.add_argument("phase", choices=sorted(PHASES))
    cli.add_common_flags(parser)
    cli.add_data_flags(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = cli.build_runner_config(args)
    PHASES[args.phase](config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
