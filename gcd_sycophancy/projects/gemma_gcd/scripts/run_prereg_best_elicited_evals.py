#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluate_base_model
import run_prereg_prefix_search


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the prereg best-elicited evaluation path: first bounded prefix "
            "search on the dev split, then confirmatory test-set evaluation with "
            "the frozen selected-prefix artifact."
        )
    )
    parser.add_argument("--model-name", default=evaluate_base_model.DEFAULT_MODEL_NAME)
    parser.add_argument("--tokenizer-name", default=None)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=evaluate_base_model.DEFAULT_DATASETS,
        help="Confirmatory dataset specs in 'test_name:path' format.",
    )
    parser.add_argument(
        "--evaluation-mode",
        choices=(evaluate_base_model.NEUTRAL_ARM_NAME, evaluate_base_model.PTST_ARM_NAME),
        default=evaluate_base_model.NEUTRAL_ARM_NAME,
        help="Evaluation arm used for the confirmatory test sets after selection.",
    )
    parser.add_argument("--candidate-library", type=Path, default=run_prereg_prefix_search.DEFAULT_LIBRARY)
    parser.add_argument("--dev-dataset", type=Path, default=run_prereg_prefix_search.DEFAULT_DEV_DATASET)
    parser.add_argument("--manifest-path", type=Path, default=run_prereg_prefix_search.DEFAULT_MANIFEST)
    parser.add_argument("--search-output-root", type=Path, default=run_prereg_prefix_search.DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--eval-output-root", type=Path, default=evaluate_base_model.DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--llm-backend", choices=("vllm", "lmstudio"), default="vllm")
    parser.add_argument("--lmstudio-base-url", default="http://localhost:1234")
    parser.add_argument("--lmstudio-model-name", default=None)
    parser.add_argument("--lmstudio-request-timeout", type=float, default=120.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser


def parse_args() -> argparse.Namespace:
    return build_arg_parser().parse_args()


def run_best_elicited_evaluations(args: argparse.Namespace) -> dict:
    search_args = run_prereg_prefix_search.build_arg_parser().parse_args(
        [
            "--model-name",
            args.model_name,
            "--arm-name",
            args.evaluation_mode,
            "--dev-dataset",
            str(args.dev_dataset),
            "--manifest-path",
            str(args.manifest_path),
            "--candidate-library",
            str(args.candidate_library),
            "--output-root",
            str(args.search_output_root),
            "--llm-backend",
            args.llm_backend,
            "--lmstudio-base-url",
            args.lmstudio_base_url,
            "--lmstudio-model-name",
            args.lmstudio_model_name or args.model_name,
            "--lmstudio-request-timeout",
            str(args.lmstudio_request_timeout),
            "--timestamp",
            str(args.timestamp) if args.timestamp is not None else "",
            "--log-level",
            args.log_level,
            *([] if args.tokenizer_name is None else ["--tokenizer-name", args.tokenizer_name]),
            *([] if args.tensor_parallel_size is None else ["--tensor-parallel-size", str(args.tensor_parallel_size)]),
            *([] if args.gpu_memory_utilization is None else ["--gpu-memory-utilization", str(args.gpu_memory_utilization)]),
            *([] if args.dtype is None else ["--dtype", args.dtype]),
            *([] if args.max_model_len is None else ["--max-model-len", str(args.max_model_len)]),
        ]
    )
    if search_args.timestamp == "":
        search_args.timestamp = None
    search_summary = run_prereg_prefix_search.run_prefix_search(search_args)

    eval_args_list = [
        "--datasets",
        *args.datasets,
        "--output-root",
        str(args.eval_output_root),
        "--evaluation-mode",
        args.evaluation_mode,
        "--selected-prefix-artifact",
        search_summary["artifact_path"],
        "--llm-backend",
        args.llm_backend,
        "--lmstudio-base-url",
        args.lmstudio_base_url,
        "--lmstudio-model-name",
        args.lmstudio_model_name or args.model_name,
        "--lmstudio-request-timeout",
        str(args.lmstudio_request_timeout),
        "--model-name",
        args.model_name,
        "--log-level",
        args.log_level,
    ]
    if args.tokenizer_name is not None:
        eval_args_list.extend(["--tokenizer-name", args.tokenizer_name])
    if args.tensor_parallel_size is not None:
        eval_args_list.extend(["--tensor-parallel-size", str(args.tensor_parallel_size)])
    if args.gpu_memory_utilization is not None:
        eval_args_list.extend(["--gpu-memory-utilization", str(args.gpu_memory_utilization)])
    if args.dtype is not None:
        eval_args_list.extend(["--dtype", args.dtype])
    if args.max_model_len is not None:
        eval_args_list.extend(["--max-model-len", str(args.max_model_len)])
    if args.limit is not None:
        eval_args_list.extend(["--limit", str(args.limit)])
    if args.timestamp is not None:
        eval_args_list.extend(["--timestamp", args.timestamp])

    eval_args = evaluate_base_model.build_arg_parser().parse_args(eval_args_list)
    run_output = evaluate_base_model.run_base_model_evaluation(eval_args)
    return {
        "search": search_summary,
        "evaluation_mode": run_output.evaluation_mode,
        "model_dir": str(run_output.model_dir),
        "summaries": evaluate_base_model.load_eval_result_summaries(run_output.model_dir),
    }


def main() -> int:
    payload = run_best_elicited_evaluations(parse_args())
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
