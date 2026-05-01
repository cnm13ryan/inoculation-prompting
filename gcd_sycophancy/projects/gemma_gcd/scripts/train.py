#!/usr/bin/env python3
"""Train-layer entrypoint: ``setup`` and ``train`` phases.

For the unified preregistration CLI, see ``run_preregistration.py``.

Usage::

    python gemma_gcd/scripts/train.py setup --experiment-dir experiments/my-exp
    python gemma_gcd/scripts/train.py train --experiment-dir experiments/my-exp
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
from phases.setup import run as run_setup  # noqa: E402
from phases.train import run as run_train  # noqa: E402

PHASES = {
    "setup": run_setup,
    "train": run_train,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train-layer entrypoint. Runs setup (arm dirs + frozen training "
            "manifest) and train (multi-seed training launch). For the "
            "unified CLI, see run_preregistration.py."
        ),
    )
    parser.add_argument("phase", choices=sorted(PHASES))
    cli.add_common_flags(parser)
    cli.add_data_flags(parser)
    cli.add_train_flags(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = cli.build_runner_config(args)
    PHASES[args.phase](config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
