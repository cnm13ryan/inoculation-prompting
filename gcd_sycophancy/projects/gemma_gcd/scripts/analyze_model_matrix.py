#!/usr/bin/env python3
"""Aggregate model-matrix prereg_analysis.json outputs into a cross-model summary.

This is a thin wrapper around ``run_prereg_model_matrix.write_matrix_summary``
that reads existing per-model analysis artifacts and writes the matrix summary
without running any subprocesses.  Equivalent to invoking the runner with
``--aggregate-only``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_prereg_model_matrix import (  # noqa: E402
    _now_iso,
    _resolve,
    load_model_config,
    read_model_results,
    write_matrix_summary,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate prereg_analysis.json outputs across the model matrix and write "
            "model_matrix_summary.{json,md}."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config_path = _resolve(args.config)
    experiment_root = _resolve(args.experiment_root)
    models = load_model_config(config_path)
    model_results = read_model_results(models, experiment_root)
    write_matrix_summary(
        experiment_root=experiment_root,
        models=models,
        model_results=model_results,
        generated_at=_now_iso(),
        config_path=config_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
