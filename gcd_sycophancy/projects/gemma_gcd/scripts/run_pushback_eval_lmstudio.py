#!/usr/bin/env python3
"""Pushback evaluation runner — Path 2: LM Studio native stateful chat.

Runs the pushback evaluation protocol against a model served by LM Studio,
using native stateful multi-turn continuation via response_id /
previous_response_id. The follow-up pushback turn does NOT re-serialize the
full conversation — LM Studio threads the turn server-side using the
response_id returned from the initial request.

Executes both the neutral and pressure conditions in sequence and prints a
JSON summary to stdout.

Pre-configured defaults:
  endpoint : http://192.168.1.228:1234
  model    : qwen3.5-4b

Both can be overridden via CLI flags without touching any code.

Example
-------
python scripts/run_pushback_eval_lmstudio.py

# Override model or endpoint:
python scripts/run_pushback_eval_lmstudio.py \\
    --lmstudio-model-name qwen2.5-7b \\
    --lmstudio-base-url http://192.168.1.228:1234

# Smoke test (5 problems):
python scripts/run_pushback_eval_lmstudio.py --limit 5

# Verify the endpoint is up before running:
curl -s http://192.168.1.228:1234/api/v1/models | python3 -m json.tool
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_base_model import (
    DEFAULT_DATASETS,
    build_arg_parser,
    load_eval_result_summaries,
    run_base_model_evaluation,
)

DEFAULT_LMSTUDIO_BASE_URL = "http://192.168.1.228:1234"
DEFAULT_LMSTUDIO_MODEL_NAME = "qwen3.5-4b"
DEFAULT_LMSTUDIO_REQUEST_TIMEOUT = 120.0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run pushback evaluation (neutral + pressure conditions) against a "
            "model served by LM Studio, using native stateful multi-turn "
            "continuation via response_id."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- LM Studio connection ---
    parser.add_argument(
        "--lmstudio-model-name",
        default=DEFAULT_LMSTUDIO_MODEL_NAME,
        help="Model name as reported by LM Studio /api/v1/models.",
    )
    parser.add_argument(
        "--lmstudio-base-url",
        default=DEFAULT_LMSTUDIO_BASE_URL,
        help="Base URL of the LM Studio REST server.",
    )
    parser.add_argument(
        "--lmstudio-request-timeout",
        type=float,
        default=DEFAULT_LMSTUDIO_REQUEST_TIMEOUT,
        help="Per-request timeout in seconds.",
    )

    # --- tokenizer ---
    parser.add_argument(
        "--tokenizer-name",
        default=None,
        help=(
            "HuggingFace tokenizer used to apply the chat template for the "
            "fallback re-prompting path (when response_id is unavailable). "
            "Defaults to --lmstudio-model-name."
        ),
    )

    # --- data ---
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Dataset specs in 'test_name:path' format.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of problems evaluated per condition (useful for smoke tests).",
    )

    # --- output ---
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Parent directory for condition output dirs. "
            "Defaults to experiments/ip_sweep/base_model_evals."
        ),
    )

    # --- pushback messages ---
    parser.add_argument(
        "--pushback-message-correct",
        default=None,
        help=(
            "Override the pushback message shown when the model resists a "
            "correct user answer. Falls back to the built-in default."
        ),
    )
    parser.add_argument(
        "--pushback-message-incorrect",
        default=None,
        help=(
            "Override the pushback message shown when the model corrects an "
            "incorrect user answer. Falls back to the built-in default."
        ),
    )

    # --- generation ---
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)

    # --- misc ---
    parser.add_argument("--log-level", default="INFO")

    return parser


def _build_evaluate_args(
    wrapper_args: argparse.Namespace,
    mode: str,
) -> argparse.Namespace:
    model_name = wrapper_args.lmstudio_model_name
    tokenizer_name = wrapper_args.tokenizer_name or model_name

    argv = [
        "--model-name", model_name,
        "--tokenizer-name", tokenizer_name,
        "--datasets", *wrapper_args.datasets,
        "--eval-protocol", "pushback",
        "--llm-backend", "lmstudio",
        "--lmstudio-model-name", model_name,
        "--lmstudio-base-url", wrapper_args.lmstudio_base_url,
        "--lmstudio-request-timeout", str(wrapper_args.lmstudio_request_timeout),
        "--eval-suffix-mode", mode,
        "--max-new-tokens", str(wrapper_args.max_new_tokens),
        "--temperature", str(wrapper_args.temperature),
        "--top-p", str(wrapper_args.top_p),
        "--log-level", wrapper_args.log_level,
    ]
    if wrapper_args.limit is not None:
        argv += ["--limit", str(wrapper_args.limit)]
    if wrapper_args.output_root is not None:
        argv += ["--output-root", str(wrapper_args.output_root)]
    if wrapper_args.pushback_message_correct:
        argv += ["--pushback-message-correct", wrapper_args.pushback_message_correct]
    if wrapper_args.pushback_message_incorrect:
        argv += ["--pushback-message-incorrect", wrapper_args.pushback_message_incorrect]
    return build_arg_parser().parse_args(argv)


def main() -> int:
    wrapper_args = _build_parser().parse_args()
    records = []

    for mode in ("neutral", "pressure"):
        run_args = _build_evaluate_args(wrapper_args, mode)
        run_output = run_base_model_evaluation(run_args)
        records.append(
            {
                "mode": mode,
                "backend": "lmstudio",
                "lmstudio_base_url": wrapper_args.lmstudio_base_url,
                "lmstudio_model_name": wrapper_args.lmstudio_model_name,
                "eval_protocol": run_output.eval_protocol,
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
