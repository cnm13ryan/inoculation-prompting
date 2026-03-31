#!/usr/bin/env python3
import argparse
import json
import logging
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


SCRIPT_DIR = Path(__file__).resolve().parent
GEMMA_GCD_DIR = SCRIPT_DIR.parent
PROJECTS_DIR = GEMMA_GCD_DIR.parent
for path in (GEMMA_GCD_DIR, PROJECTS_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


DEFAULT_MODEL_NAME = "google/gemma-2b-it"
DEFAULT_DATASETS = [
    "test_confirmatory:gemma_gcd/data/prereg/test_confirmatory.jsonl",
    "test_paraphrase:gemma_gcd/data/prereg/test_paraphrase.jsonl",
    "test_near_transfer:gemma_gcd/data/prereg/test_near_transfer.jsonl",
]
DEFAULT_OUTPUT_ROOT = PROJECTS_DIR / "experiments" / "prereg" / "base_model_evals"
NEUTRAL_ARM_NAME = "neutral"
PTST_ARM_NAME = "ptst"


@dataclass
class BaseModelEvalRun:
    experiment_dir: Path
    model_dir: Path
    evaluation_mode: str
    timestamp: str
    datasets: dict[str, Path]


def parse_args() -> argparse.Namespace:
    return build_arg_parser().parse_args()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the preregistered fixed-interface evaluation pipeline on a base "
            "model for the Gemma GCD evaluation sets."
        )
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="HuggingFace model identifier to evaluate with vLLM.",
    )
    parser.add_argument(
        "--tokenizer-name",
        default=None,
        help="Optional tokenizer override. Defaults to --model-name.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Dataset specs in 'test_name:path' format.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Experiment directory to create. Defaults to "
            "experiments/prereg/base_model_evals/<condition-dir>."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Parent directory used when --output-dir is not provided.",
    )
    parser.add_argument(
        "--evaluation-mode",
        choices=(NEUTRAL_ARM_NAME, PTST_ARM_NAME),
        default=NEUTRAL_ARM_NAME,
        help="Preregistered evaluation arm. 'ptst' appends only the prereg reminder.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional sample limit passed through to Evaluator.evaluate().",
    )
    parser.add_argument(
        "--llm-backend",
        choices=("vllm", "lmstudio"),
        default="vllm",
        help="Inference backend for evaluation.",
    )
    parser.add_argument(
        "--lmstudio-base-url",
        default="http://localhost:1234",
        help="LM Studio native REST base URL used when --llm-backend=lmstudio.",
    )
    parser.add_argument(
        "--lmstudio-model-name",
        default=None,
        help="Model name to request from LM Studio. Defaults to --model-name.",
    )
    parser.add_argument(
        "--lmstudio-request-timeout",
        type=float,
        default=120.0,
        help="LM Studio request timeout in seconds.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help="Optional CLI override for VLLM_TP_SIZE.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help="Optional CLI override for VLLM_GPU_MEMORY_UTILIZATION.",
    )
    parser.add_argument(
        "--dtype",
        default=None,
        help="Optional CLI override for VLLM_DTYPE.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Optional vLLM max_model_len override.",
    )
    parser.add_argument(
        "--timestamp",
        default=None,
        help="Optional fixed timestamp for reproducible output paths in tests.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level.",
    )
    return parser


def resolve_repo_relative_path(path: Path) -> Path:
    if path.is_absolute():
        return path

    candidates = [
        Path.cwd() / path,
        PROJECTS_DIR / path,
        GEMMA_GCD_DIR / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (PROJECTS_DIR / path).resolve()


def parse_dataset_specs(dataset_specs: list[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for spec in dataset_specs:
        if ":" in spec:
            test_name, raw_path = spec.split(":", 1)
        else:
            raw_path = spec
            test_name = Path(raw_path).stem
        parsed[test_name] = resolve_repo_relative_path(Path(raw_path))
    return parsed


def make_generation_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    del args
    return {
        "max_new_tokens": 415,
        "temperature": 0.3,
        "top_p": 0.9,
        "top_k": None,
        "n": 1,
    }


def make_vllm_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    vllm_kwargs: dict[str, Any] = {}

    tp_size = args.tensor_parallel_size
    if tp_size is None and os.getenv("VLLM_TP_SIZE"):
        tp_size = int(os.environ["VLLM_TP_SIZE"])
    if tp_size is not None:
        vllm_kwargs["tensor_parallel_size"] = tp_size

    gpu_mem = args.gpu_memory_utilization
    if gpu_mem is None and os.getenv("VLLM_GPU_MEMORY_UTILIZATION"):
        gpu_mem = float(os.environ["VLLM_GPU_MEMORY_UTILIZATION"])
    if gpu_mem is not None:
        vllm_kwargs["gpu_memory_utilization"] = gpu_mem

    dtype = args.dtype or os.getenv("VLLM_DTYPE")
    if dtype:
        vllm_kwargs["dtype"] = dtype

    distributed_backend = os.getenv("VLLM_DISTRIBUTED_BACKEND")
    if distributed_backend:
        vllm_kwargs["distributed_executor_backend"] = distributed_backend

    if args.max_model_len is not None:
        vllm_kwargs["max_model_len"] = args.max_model_len

    vllm_kwargs.setdefault("trust_remote_code", True)
    vllm_kwargs.setdefault("enforce_eager", True)
    return vllm_kwargs


def sanitize_model_dir_name(model_name: str) -> str:
    model_path = Path(model_name)
    raw_name = model_path.name if model_path.exists() else model_name
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_name).strip("_")
    return sanitized or "model"


def is_peft_adapter_dir(model_name: str) -> bool:
    model_path = Path(model_name)
    return model_path.is_dir() and (model_path / "adapter_config.json").exists()


def merge_peft_model_for_vllm(model_name: str) -> tuple[str, Callable[[], None]]:
    import torch
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer

    adapter_path = Path(model_name).resolve()
    merged_dir = Path(tempfile.mkdtemp(prefix="gemma_gcd_merged_peft_"))

    logging.info("Merging PEFT adapter for vLLM: %s", adapter_path)
    model = AutoPeftModelForCausalLM.from_pretrained(
        adapter_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(merged_dir)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    tokenizer.save_pretrained(merged_dir)

    def _cleanup() -> None:
        shutil.rmtree(merged_dir, ignore_errors=True)

    return str(merged_dir), _cleanup


def build_condition_dir_name(evaluation_mode: str) -> str:
    return f"evaluation_mode-{evaluation_mode}"


def choose_experiment_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return resolve_repo_relative_path(args.output_dir)
    output_root = resolve_repo_relative_path(args.output_root)
    return output_root / build_condition_dir_name(args.evaluation_mode)


def make_timestamp(args: argparse.Namespace) -> str:
    return args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")


def write_experiment_config(
    experiment_dir: Path,
    *,
    model_name: str,
    tokenizer_name: str,
    evaluation_mode: str,
    datasets: dict[str, Path],
    generation_kwargs: dict[str, Any],
    vllm_kwargs: dict[str, Any],
    llm_backend: str = "vllm",
    lmstudio_kwargs: dict[str, Any] | None = None,
    model_dir: Path | None = None,
) -> None:
    from all_evals import (
        PREREG_DIRECT_SOLVE_TEMPLATE,
        PREREG_EVAL_PROTOCOL,
        PREREG_INCORRECT_ASSERTION_TEMPLATE,
        PREREG_PTST_REMINDER,
    )

    config_path = experiment_dir / "config.json"
    config_payload = {
        "experiment_name": "gemma_gcd_preregistered_base_model_eval",
        "model_name": model_name,
        "tokenizer_name": tokenizer_name,
        "evaluation_interface": PREREG_EVAL_PROTOCOL,
        "evaluation_mode": evaluation_mode,
        "system_prompt": None,
        "ptst_only": evaluation_mode == PTST_ARM_NAME,
        "ptst_reminder": (
            PREREG_PTST_REMINDER if evaluation_mode == PTST_ARM_NAME else None
        ),
        "templates": {
            "direct_solve": PREREG_DIRECT_SOLVE_TEMPLATE,
            "incorrect_assertion": PREREG_INCORRECT_ASSERTION_TEMPLATE,
        },
        "llm_backend": llm_backend,
        "lmstudio_kwargs": lmstudio_kwargs or {},
        "datasets": {name: str(path) for name, path in datasets.items()},
        "generation_kwargs": generation_kwargs,
        "vllm_kwargs": vllm_kwargs,
    }

    experiment_dir.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(config_payload, handle, indent=2)
    if model_dir is not None:
        model_dir.mkdir(parents=True, exist_ok=True)
        with (model_dir / "inference_config.json").open("w", encoding="utf-8") as handle:
            json.dump(config_payload, handle, indent=2)


def load_eval_result_summaries(model_dir: Path) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for eval_path in sorted(model_dir.glob("*_eval_results.json")):
        test_name = eval_path.stem.replace("_eval_results", "")
        with eval_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        summaries[test_name] = {
            "direct_solve_accuracy": payload.get("direct_solve_accuracy", {}),
            "sycophancy_rate": payload.get("sycophancy_rate", {}),
            "conditional_sycophancy_rate": payload.get(
                "conditional_sycophancy_rate", {}
            ),
            "exclusions": payload.get("exclusions", {}),
            "capabilities": payload.get(
                "capabilities", payload.get("direct_solve_accuracy", {})
            ),
        }
    return summaries


def run_base_model_evaluation(args: argparse.Namespace) -> BaseModelEvalRun:
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    from all_evals import PreregisteredEvaluator
    from transformers import AutoTokenizer

    datasets = parse_dataset_specs(args.datasets)
    generation_kwargs = make_generation_kwargs(args)
    vllm_kwargs = make_vllm_kwargs(args)
    tokenizer_name = args.tokenizer_name or args.model_name
    lmstudio_kwargs = {
        "model_name": args.lmstudio_model_name or args.model_name,
        "base_url": args.lmstudio_base_url,
        "request_timeout": args.lmstudio_request_timeout,
    }
    cleanup_model = None
    vllm_model_name = args.model_name

    experiment_dir = choose_experiment_dir(args)
    timestamp = make_timestamp(args)
    model_dir = (
        experiment_dir
        / "results"
        / timestamp
        / f"{sanitize_model_dir_name(args.model_name)}_evals"
    )
    model_dir.mkdir(parents=True, exist_ok=True)

    write_experiment_config(
        experiment_dir,
        model_name=args.model_name,
        tokenizer_name=tokenizer_name,
        evaluation_mode=args.evaluation_mode,
        datasets=datasets,
        generation_kwargs=generation_kwargs,
        vllm_kwargs=vllm_kwargs,
        llm_backend=args.llm_backend,
        lmstudio_kwargs=lmstudio_kwargs,
        model_dir=model_dir,
    )

    logging.info("Loading tokenizer: %s", tokenizer_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    llm = None
    try:
        if args.llm_backend == "lmstudio":
            from all_evals import get_lmstudio_llamaindex_model

            logging.info(
                "Connecting to LM Studio via LlamaIndex: %s (%s)",
                lmstudio_kwargs["model_name"],
                lmstudio_kwargs["base_url"],
            )
            llm = get_lmstudio_llamaindex_model(**lmstudio_kwargs)
        else:
            from vllm import LLM

            if is_peft_adapter_dir(args.model_name):
                vllm_model_name, cleanup_model = merge_peft_model_for_vllm(
                    args.model_name
                )
            logging.info("Loading base model with vLLM: %s", vllm_model_name)
            llm = LLM(model=vllm_model_name, **vllm_kwargs)

        evaluator = PreregisteredEvaluator(
            llm=llm,
            tokenizer=tokenizer,
            generation_kwargs=generation_kwargs,
            llm_backend=args.llm_backend,
            ptst_only=args.evaluation_mode == PTST_ARM_NAME,
            arm_name=args.evaluation_mode,
        )

        for test_name, dataset_path in datasets.items():
            logging.info("Evaluating %s from %s", test_name, dataset_path)
            evaluator.evaluate(
                test_data_path=str(dataset_path),
                test_name=test_name,
                limit=args.limit,
                root_dir=str(model_dir),
                dump_outputs=True,
            )
    finally:
        if cleanup_model is not None:
            cleanup_model()

    logging.info("Base-model evaluation artifacts written to %s", model_dir)
    return BaseModelEvalRun(
        experiment_dir=experiment_dir,
        model_dir=model_dir,
        evaluation_mode=args.evaluation_mode,
        timestamp=timestamp,
        datasets=datasets,
    )


def main() -> int:
    run_output = run_base_model_evaluation(parse_args())
    summaries = load_eval_result_summaries(run_output.model_dir)
    print(
        json.dumps(
            {
                "model_dir": str(run_output.model_dir),
                "evaluation_mode": run_output.evaluation_mode,
                "summaries": summaries,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
