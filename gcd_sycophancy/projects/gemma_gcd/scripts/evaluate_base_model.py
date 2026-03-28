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
    "task_test:gemma_gcd/data/task_test.jsonl",
    "ood_test:gemma_gcd/data/ood_test.jsonl",
]
DEFAULT_ATTRIBUTES_PATH = (
    PROJECTS_DIR / "experiments" / "ip_sweep" / "attributes_to_vary.json"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECTS_DIR / "experiments" / "ip_sweep" / "base_model_evals"
)


@dataclass
class BaseModelEvalRun:
    experiment_dir: Path
    model_dir: Path
    eval_suffix: str
    eval_protocol: str
    timestamp: str
    datasets: dict[str, Path]


def parse_args() -> argparse.Namespace:
    return build_arg_parser().parse_args()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the comprehensive Evaluator.evaluate() pipeline on a base model "
            "for the Gemma GCD task and OOD test sets."
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
            "experiments/ip_sweep/base_model_evals/<condition-dir>."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Parent directory used when --output-dir is not provided.",
    )
    parser.add_argument(
        "--eval-suffix",
        default=None,
        help="Custom evaluation suffix appended to all user prompts.",
    )
    parser.add_argument(
        "--eval-suffix-mode",
        choices=("neutral", "pressure"),
        default="neutral",
        help=(
            "Suffix preset. 'pressure' reads the canonical control pressure "
            "suffix from attributes_to_vary.json unless --eval-suffix is set."
        ),
    )
    parser.add_argument(
        "--eval-protocol",
        choices=("single_turn", "pushback"),
        default="single_turn",
        help="Evaluation protocol. 'pushback' adds a second user turn only on gated answer-present rows.",
    )
    parser.add_argument(
        "--pushback-message-correct",
        default=None,
        help="Optional override for the pushback shown when a correct user answer is resisted.",
    )
    parser.add_argument(
        "--pushback-message-incorrect",
        default=None,
        help="Optional override for the pushback shown when an incorrect user answer is corrected.",
    )
    parser.add_argument(
        "--attributes-to-vary",
        type=Path,
        default=DEFAULT_ATTRIBUTES_PATH,
        help="Path to the sweep attributes_to_vary.json used to resolve pressure suffix.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional sample limit passed through to Evaluator.evaluate().",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=400,
        help="Generation max_new_tokens passed through the shared evaluator.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature passed through the shared evaluator.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Generation top_p passed through the shared evaluator.",
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
    return {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
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


def load_pressure_suffix(attributes_path: Path) -> str:
    resolved_path = resolve_repo_relative_path(attributes_path)
    with resolved_path.open("r", encoding="utf-8") as handle:
        attribute_sets = json.load(handle)

    candidates = []
    for attribute_set in attribute_sets:
        train_suffix = attribute_set.get("train_user_suffix", "") or ""
        eval_suffix = attribute_set.get("eval_user_suffix", "") or ""
        if train_suffix == "" and eval_suffix:
            candidates.append(eval_suffix)

    unique_candidates = list(dict.fromkeys(candidates))
    if not unique_candidates:
        raise ValueError(
            f"No control pressure eval_user_suffix found in {resolved_path}"
        )
    if len(unique_candidates) > 1:
        raise ValueError(
            f"Expected exactly one control pressure eval_user_suffix in {resolved_path}, "
            f"found {len(unique_candidates)}"
        )
    return unique_candidates[0]


def resolve_eval_suffix(args: argparse.Namespace) -> str:
    if args.eval_suffix is not None:
        return args.eval_suffix
    if args.eval_suffix_mode == "pressure":
        return load_pressure_suffix(args.attributes_to_vary)
    return ""


def build_pushback_messages(args: argparse.Namespace) -> dict[str, str]:
    overrides = {}
    if args.pushback_message_correct:
        overrides["user_proposes_correct"] = args.pushback_message_correct
    if args.pushback_message_incorrect:
        overrides["user_proposes_incorrect"] = args.pushback_message_incorrect
    return overrides


def build_condition_dir_name(eval_suffix: str, eval_protocol: str = "single_turn") -> str:
    from attribute_sweep_multi_seed_run import build_param_dir_name

    params = {
        "train_user_suffix": "",
        "eval_user_suffix": eval_suffix,
    }
    if eval_protocol != "single_turn":
        params["eval_protocol"] = eval_protocol
    return build_param_dir_name(params)


def choose_experiment_dir(args: argparse.Namespace, eval_suffix: str) -> Path:
    if args.output_dir is not None:
        return resolve_repo_relative_path(args.output_dir)
    output_root = resolve_repo_relative_path(args.output_root)
    return output_root / build_condition_dir_name(eval_suffix, args.eval_protocol)


def make_timestamp(args: argparse.Namespace) -> str:
    return args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl_records(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record))
            handle.write("\n")


def append_eval_suffix_to_jsonl(source_path: Path, output_path: Path, suffix: str) -> None:
    if not suffix:
        raise ValueError("append_eval_suffix_to_jsonl requires a non-empty suffix")

    from datasets import Dataset
    from data_pipeline import DataPipeline

    rows = load_jsonl_records(source_path)
    pipeline = DataPipeline(tokenizer=None, finetune_config=None)
    suffixed_ds = pipeline._append_suffix_to_user_prompts(
        Dataset.from_list(rows),
        suffix,
        require_user_answer=True,
    )
    write_jsonl_records(output_path, [dict(example) for example in suffixed_ds])


def materialize_eval_datasets(
    datasets: dict[str, Path],
    eval_suffix: str,
    temp_dir: Path,
) -> dict[str, Path]:
    if not eval_suffix:
        return datasets

    temp_dir.mkdir(parents=True, exist_ok=True)
    materialized: dict[str, Path] = {}
    for test_name, dataset_path in datasets.items():
        target_path = temp_dir / dataset_path.name
        append_eval_suffix_to_jsonl(dataset_path, target_path, eval_suffix)
        materialized[test_name] = target_path
    return materialized


def write_experiment_config(
    experiment_dir: Path,
    *,
    model_name: str,
    tokenizer_name: str,
    eval_suffix: str,
    eval_protocol: str,
    pushback_messages: dict[str, str],
    datasets: dict[str, Path],
    generation_kwargs: dict[str, Any],
    vllm_kwargs: dict[str, Any],
    llm_backend: str = "vllm",
    lmstudio_kwargs: dict[str, Any] | None = None,
) -> None:
    config_path = experiment_dir / "config.json"
    config_payload = {
        "experiment_name": "gemma_gcd_base_model_eval",
        "model_name": model_name,
        "tokenizer_name": tokenizer_name,
        "train_user_suffix": "",
        "eval_user_suffix": eval_suffix,
        "eval_protocol": eval_protocol,
        "pushback_messages": pushback_messages,
        "llm_backend": llm_backend,
        "lmstudio_kwargs": lmstudio_kwargs or {},
        "datasets": {name: str(path) for name, path in datasets.items()},
        "generation_kwargs": generation_kwargs,
        "vllm_kwargs": vllm_kwargs,
    }

    experiment_dir.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(config_payload, handle, indent=2)


def load_eval_result_summaries(model_dir: Path) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for eval_path in sorted(model_dir.glob("*_eval_results.json")):
        test_name = eval_path.stem.replace("_eval_results", "")
        with eval_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        summaries[test_name] = {
            "capabilities": payload.get("capabilities", {}),
            "confirms_correct": payload.get("confirms_correct", {}),
            "confirms_incorrect": payload.get("confirms_incorrect", {}),
            "affirm_when_correct": payload.get("affirm_when_correct", {}),
            "correct_when_wrong": payload.get("correct_when_wrong", {}),
        }
    return summaries


def run_base_model_evaluation(args: argparse.Namespace) -> BaseModelEvalRun:
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    from all_evals import Evaluator
    from transformers import AutoTokenizer

    eval_suffix = resolve_eval_suffix(args)
    pushback_messages = build_pushback_messages(args)
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

    experiment_dir = choose_experiment_dir(args, eval_suffix)
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
        eval_suffix=eval_suffix,
        eval_protocol=args.eval_protocol,
        pushback_messages=pushback_messages,
        datasets=datasets,
        generation_kwargs=generation_kwargs,
        vllm_kwargs=vllm_kwargs,
        llm_backend=args.llm_backend,
        lmstudio_kwargs=lmstudio_kwargs,
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
                vllm_model_name, cleanup_model = merge_peft_model_for_vllm(args.model_name)
            logging.info("Loading base model with vLLM: %s", vllm_model_name)
            llm = LLM(model=vllm_model_name, **vllm_kwargs)

        evaluator = Evaluator(
            llm=llm,
            tokenizer=tokenizer,
            generation_kwargs=generation_kwargs,
            llm_backend=args.llm_backend,
            lmstudio_kwargs=lmstudio_kwargs,
            evaluation_protocol=args.eval_protocol,
            pushback_messages=pushback_messages,
        )

        with tempfile.TemporaryDirectory(prefix="gemma_gcd_eval_") as tmp_dir_str:
            temp_dir = Path(tmp_dir_str)
            eval_datasets = materialize_eval_datasets(datasets, eval_suffix, temp_dir)

            for test_name, dataset_path in eval_datasets.items():
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
        eval_suffix=eval_suffix,
        eval_protocol=args.eval_protocol,
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
                "eval_protocol": run_output.eval_protocol,
                "summaries": summaries,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
