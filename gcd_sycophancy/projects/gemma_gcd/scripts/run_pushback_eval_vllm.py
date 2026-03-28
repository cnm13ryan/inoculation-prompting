#!/usr/bin/env python3
"""Pushback evaluation runner — Path 1: vLLM on ROCm AMD GPUs.

Runs the pushback evaluation protocol against a local model served by vLLM.
Executes both the neutral and pressure conditions in sequence and prints a
JSON summary to stdout.

The pushback turn is implemented via full prompt re-serialization: the initial
response and the static pushback message are concatenated with the original
prompt and re-submitted to vLLM as a single new request.

Intended use-case: evaluating a fine-tuned Gemma model on the ROCm GPU cluster.

Example
-------
python scripts/run_pushback_eval_vllm.py \\
    --model-name /path/to/fine-tuned-gemma \\
    --datasets task_test:data/task_test.jsonl ood_test:data/ood_test.jsonl \\
    --gpu-memory-utilization 0.90 \\
    --dtype float16

Smoke test (5 problems, neutral condition only via --limit):
python scripts/run_pushback_eval_vllm.py \\
    --model-name google/gemma-2b-it \\
    --limit 5
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
    DEFAULT_MODEL_NAME,
    build_arg_parser,
    load_eval_result_summaries,
    run_base_model_evaluation,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run pushback evaluation (neutral + pressure conditions) against a "
            "model loaded by vLLM on ROCm AMD GPUs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- model ---
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="HuggingFace model path or identifier passed to vLLM.",
    )
    parser.add_argument(
        "--tokenizer-name",
        default=None,
        help="Tokenizer override. Defaults to --model-name.",
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

    # --- vLLM / ROCm tuning ---
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help="vLLM tensor parallel size (also readable from $VLLM_TP_SIZE).",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help=(
            "Fraction of GPU memory vLLM may use "
            "(also readable from $VLLM_GPU_MEMORY_UTILIZATION)."
        ),
    )
    parser.add_argument(
        "--dtype",
        default=None,
        help="vLLM dtype override, e.g. float16 or bfloat16.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="vLLM max_model_len override.",
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
    argv = [
        "--model-name", wrapper_args.model_name,
        "--datasets", *wrapper_args.datasets,
        "--eval-protocol", "pushback",
        "--llm-backend", "vllm",
        "--eval-suffix-mode", mode,
        "--max-new-tokens", str(wrapper_args.max_new_tokens),
        "--temperature", str(wrapper_args.temperature),
        "--top-p", str(wrapper_args.top_p),
        "--log-level", wrapper_args.log_level,
    ]
    if wrapper_args.tokenizer_name:
        argv += ["--tokenizer-name", wrapper_args.tokenizer_name]
    if wrapper_args.limit is not None:
        argv += ["--limit", str(wrapper_args.limit)]
    if wrapper_args.output_root is not None:
        argv += ["--output-root", str(wrapper_args.output_root)]
    if wrapper_args.pushback_message_correct:
        argv += ["--pushback-message-correct", wrapper_args.pushback_message_correct]
    if wrapper_args.pushback_message_incorrect:
        argv += ["--pushback-message-incorrect", wrapper_args.pushback_message_incorrect]
    if wrapper_args.tensor_parallel_size is not None:
        argv += ["--tensor-parallel-size", str(wrapper_args.tensor_parallel_size)]
    if wrapper_args.gpu_memory_utilization is not None:
        argv += ["--gpu-memory-utilization", str(wrapper_args.gpu_memory_utilization)]
    if wrapper_args.dtype:
        argv += ["--dtype", wrapper_args.dtype]
    if wrapper_args.max_model_len is not None:
        argv += ["--max-model-len", str(wrapper_args.max_model_len)]
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
                "backend": "vllm",
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
