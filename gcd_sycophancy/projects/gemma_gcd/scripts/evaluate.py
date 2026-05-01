#!/usr/bin/env python3
"""Evaluate-layer entrypoint: every eval-stage phase.

Phases routed: ``preflight``, ``fixed-interface-eval``,
``semantic-interface-eval``, ``prefix-search``, ``best-elicited-eval``,
``checkpoint-curve-eval``.

Note: ``preflight`` is conventionally an eval-stage gate (it does a small
training run only as a means to a quality check), so it lives here.

For the unified preregistration CLI, see ``run_preregistration.py``.
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
from phases.preflight import run as run_preflight  # noqa: E402
from phases.fixed_interface_eval import run as run_fixed_interface_eval  # noqa: E402
from phases.semantic_interface_eval import run as run_semantic_interface_eval  # noqa: E402
from phases.prefix_search import run as run_prefix_search  # noqa: E402
from phases.best_elicited_eval import run as run_best_elicited_eval  # noqa: E402
from phases.checkpoint_curve_eval import run as run_checkpoint_curve_eval  # noqa: E402

PHASES = {
    "preflight": run_preflight,
    "fixed-interface-eval": run_fixed_interface_eval,
    "semantic-interface-eval": run_semantic_interface_eval,
    "prefix-search": run_prefix_search,
    "best-elicited-eval": run_best_elicited_eval,
    "checkpoint-curve-eval": run_checkpoint_curve_eval,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate-layer entrypoint. Runs preflight + the five eval phases. "
            "For the unified CLI, see run_preregistration.py."
        ),
    )
    parser.add_argument("phase", choices=sorted(PHASES))
    cli.add_common_flags(parser)
    cli.add_data_flags(parser)
    cli.add_eval_flags(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = cli.build_runner_config(args)
    PHASES[args.phase](config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
