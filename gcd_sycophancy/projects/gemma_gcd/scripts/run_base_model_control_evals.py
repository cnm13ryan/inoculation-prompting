#!/usr/bin/env python3
import argparse
import json
import sys


from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_base_model import (
    build_arg_parser,
    load_eval_result_summaries,
    run_base_model_evaluation,
)


def parse_wrapper_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the base-model evaluation twice: once neutral and once with "
            "the configured pressure eval suffix."
        )
    )
    parser.add_argument(
        "--base-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Arguments forwarded to evaluate_base_model.py after the condition flags.",
    )
    return parser.parse_args()
def _parse_evaluate_args(argv: list[str]) -> argparse.Namespace:
    forwarded = list(argv)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    return build_arg_parser().parse_args(forwarded)


def main() -> int:
    wrapper_args = parse_wrapper_args()
    records = []

    for mode in ("neutral", "pressure"):
        run_args = _parse_evaluate_args([*wrapper_args.base_args, "--eval-suffix-mode", mode])
        run_output = run_base_model_evaluation(run_args)
        records.append(
            {
                "mode": mode,
                "experiment_dir": str(run_output.experiment_dir),
                "model_dir": str(run_output.model_dir),
                "eval_suffix": run_output.eval_suffix,
                "summaries": load_eval_result_summaries(run_output.model_dir),
            }
        )

    print(json.dumps(records, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
