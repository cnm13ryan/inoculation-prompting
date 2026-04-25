#!/usr/bin/env python3
import argparse
import json
import logging
import os
import re
import hashlib
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
    "same_domain_extrapolation:gemma_gcd/data/prereg/test_near_transfer.jsonl",
]
DEFAULT_OUTPUT_ROOT = PROJECTS_DIR / "experiments" / "prereg" / "base_model_evals"
DEFAULT_PREFIX_LIBRARY = PROJECTS_DIR / "experiments" / "prereg" / "appendix_b_prefixes.json"
NEUTRAL_ARM_NAME = "neutral"
PTST_ARM_NAME = "ptst"
EXPECTED_PREFIX_IDS = [f"P{index}" for index in range(12)]
FORMAT_FAILURE_CATEGORIES = (
    "unparseable_response",
    "degenerate_response",
    "truncated_before_verdict_field",
)


@dataclass
class BaseModelEvalRun:
    experiment_dir: Path
    model_dir: Path
    evaluation_mode: str
    timestamp: str
    datasets: dict[str, Path]


def compute_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_locked_prefix_library(path: Path | None = None) -> list[dict[str, str]]:
    path = path or DEFAULT_PREFIX_LIBRARY
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    prefixes = payload.get("prefixes")
    if not isinstance(prefixes, list):
        raise ValueError(f"Locked prefix library {path} must contain a 'prefixes' list.")
    if len(prefixes) != 12:
        raise ValueError(f"Locked prefix library {path} must contain exactly 12 prefixes.")
    candidate_ids = [item.get("prefix_id") for item in prefixes]
    if candidate_ids != EXPECTED_PREFIX_IDS:
        raise ValueError(
            f"Locked prefix library {path} must preserve Appendix B ordering {EXPECTED_PREFIX_IDS}."
        )
    return prefixes


def load_selected_prefix_artifact(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if payload.get("workflow_name") != "preregistered_bounded_prefix_search":
        raise ValueError(
            f"Selected-prefix artifact {path} must come from the preregistered_bounded_prefix_search workflow."
        )
    if payload.get("selection_target") != "user_message_prefix":
        raise ValueError(
            f"Selected-prefix artifact {path} must target user_message_prefix."
        )
    if payload.get("selection_split") != "dev":
        raise ValueError(
            f"Selected-prefix artifact {path} must be selected on the dev split."
        )
    if payload.get("search_budget") != 12:
        raise ValueError(
            f"Selected-prefix artifact {path} must record a search_budget of 12."
        )
    if payload.get("evaluation_interface") != "preregistered_fixed_interface":
        raise ValueError(
            f"Selected-prefix artifact {path} must use the preregistered fixed interface."
        )

    selected_prefix_id = payload.get("selected_prefix_id")
    selected_prefix_text = payload.get("selected_prefix_text")
    if not isinstance(selected_prefix_id, str) or not selected_prefix_id:
        raise ValueError(
            f"Selected-prefix artifact {path} is missing a non-empty selected_prefix_id."
        )
    if not isinstance(selected_prefix_text, str):
        raise ValueError(
            f"Selected-prefix artifact {path} is missing selected_prefix_text."
        )

    library_hash = payload.get("candidate_library_hash")
    if not isinstance(library_hash, str) or not library_hash:
        raise ValueError(
            f"Selected-prefix artifact {path} is missing candidate_library_hash."
        )

    locked_library = load_locked_prefix_library()
    locked_library_hash = compute_file_sha256(DEFAULT_PREFIX_LIBRARY)
    if library_hash != locked_library_hash:
        raise ValueError(
            f"Selected-prefix artifact {path} candidate_library_hash does not match "
            f"the locked Appendix B library."
        )

    artifact_library = payload.get("candidate_library")
    if artifact_library != locked_library:
        raise ValueError(
            f"Selected-prefix artifact {path} must embed the locked 12-prefix Appendix B library unchanged."
        )

    candidate_results = payload.get("candidate_results")
    if not isinstance(candidate_results, list) or len(candidate_results) != 12:
        raise ValueError(
            f"Selected-prefix artifact {path} must contain 12 candidate_results entries."
        )
    candidate_result_ids = [item.get("prefix_id") for item in candidate_results]
    if candidate_result_ids != EXPECTED_PREFIX_IDS:
        raise ValueError(
            f"Selected-prefix artifact {path} candidate_results must preserve Appendix B ordering."
        )

    prefix_by_id = {item["prefix_id"]: item["text"] for item in locked_library}
    for item in candidate_results:
        if item.get("prefix_text") != prefix_by_id.get(item.get("prefix_id")):
            raise ValueError(
                f"Selected-prefix artifact {path} contains a prefix_text that does not match the locked library."
            )

    selection_evidence = payload.get("selection_evidence")
    if not isinstance(selection_evidence, dict):
        raise ValueError(
            f"Selected-prefix artifact {path} is missing selection_evidence."
        )
    if selection_evidence.get("selected_candidate_meets_accuracy_constraint") is not True:
        raise ValueError(
            f"Selected-prefix artifact {path} must show the selected prefix satisfied the prereg accuracy constraint."
        )
    if selection_evidence.get("selected_prefix_id") != selected_prefix_id:
        raise ValueError(
            f"Selected-prefix artifact {path} selection_evidence.selected_prefix_id does not match selected_prefix_id."
        )
    if selection_evidence.get("selected_prefix_text") != selected_prefix_text:
        raise ValueError(
            f"Selected-prefix artifact {path} selection_evidence.selected_prefix_text does not match selected_prefix_text."
        )
    ranked_eligible = selection_evidence.get("ranked_eligible_candidates")
    if not isinstance(ranked_eligible, list) or not ranked_eligible:
        raise ValueError(
            f"Selected-prefix artifact {path} must contain ranked_eligible_candidates."
        )
    if ranked_eligible[0].get("prefix_id") != selected_prefix_id:
        raise ValueError(
            f"Selected-prefix artifact {path} must rank the selected prefix first among eligible candidates."
        )
    if prefix_by_id.get(selected_prefix_id) != selected_prefix_text:
        raise ValueError(
            f"Selected-prefix artifact {path} selected prefix does not match the locked Appendix B library."
        )

    dev_split = payload.get("dev_split")
    if not isinstance(dev_split, dict):
        raise ValueError(f"Selected-prefix artifact {path} is missing dev_split metadata.")
    if Path(str(dev_split.get("dataset_path", ""))).name != "dev.jsonl":
        raise ValueError(
            f"Selected-prefix artifact {path} must record dev.jsonl as the selection dataset."
        )

    return {
        "artifact_path": str(path.resolve()),
        "artifact_sha256": compute_file_sha256(path),
        "selected_prefix_id": selected_prefix_id,
        "selected_prefix_text": selected_prefix_text,
        "candidate_library_hash": library_hash,
    }


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
        "--evaluation-interface",
        choices=("fixed_interface", "semantic_interface"),
        default="fixed_interface",
        help=(
            "Evaluation interface to use.  'fixed_interface' (default) runs the "
            "preregistered XML-tag path.  'semantic_interface' runs the secondary "
            "robustness-only natural-language path (no XML tags required).  "
            "Semantic-interface outputs are labeled evaluation_design='semantic_interface' "
            "and must not be used for any primary confirmatory claim."
        ),
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
    parser.add_argument(
        "--selected-prefix-artifact",
        type=Path,
        default=None,
        help=(
            "Frozen prereg prefix-selection artifact to apply during evaluation. "
            "When provided, its selected user-message prefix is prepended to each "
            "fixed-interface user prompt."
        ),
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
    selected_prefix_artifact: dict[str, Any] | None = None,
    evaluation_interface: str = "fixed_interface",
) -> None:
    from all_evals import (
        PREREG_DIRECT_SOLVE_TEMPLATE,
        PREREG_EVAL_PROTOCOL,
        PREREG_INCORRECT_ASSERTION_TEMPLATE,
        PREREG_PTST_REMINDER,
        SEMANTIC_DIRECT_SOLVE_TEMPLATE,
        SEMANTIC_INCORRECT_ASSERTION_TEMPLATE,
        SEMANTIC_INTERFACE_EVAL_PROTOCOL,
        SEMANTIC_PTST_REMINDER,
    )

    is_semantic = evaluation_interface == "semantic_interface"
    resolved_eval_interface = (
        SEMANTIC_INTERFACE_EVAL_PROTOCOL if is_semantic else PREREG_EVAL_PROTOCOL
    )
    config_path = experiment_dir / "config.json"
    config_payload = {
        "experiment_name": "gemma_gcd_preregistered_base_model_eval",
        "model_name": model_name,
        "tokenizer_name": tokenizer_name,
        "evaluation_interface": resolved_eval_interface,
        "evaluation_mode": evaluation_mode,
        "system_prompt": None,
        "user_message_prefix": (
            selected_prefix_artifact["selected_prefix_text"]
            if selected_prefix_artifact is not None
            else ""
        ),
        "ptst_only": evaluation_mode == PTST_ARM_NAME,
        "ptst_reminder": (
            (
                SEMANTIC_PTST_REMINDER
                if is_semantic
                else PREREG_PTST_REMINDER
            )
            if evaluation_mode == PTST_ARM_NAME
            else None
        ),
        "selected_prefix_artifact": selected_prefix_artifact,
        "templates": {
            "direct_solve": (
                SEMANTIC_DIRECT_SOLVE_TEMPLATE
                if is_semantic
                else PREREG_DIRECT_SOLVE_TEMPLATE
            ),
            "incorrect_assertion": (
                SEMANTIC_INCORRECT_ASSERTION_TEMPLATE
                if is_semantic
                else PREREG_INCORRECT_ASSERTION_TEMPLATE
            ),
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


def compute_fixed_interface_quality_summary(
    eval_summaries: dict[str, dict[str, Any]],
    *,
    max_format_failure_rate: float,
) -> dict[str, Any]:
    if not eval_summaries:
        raise ValueError("Fixed-interface quality summary requires at least one dataset.")

    dataset_summaries: dict[str, dict[str, Any]] = {}
    unacceptable_datasets: list[str] = []
    worst_dataset_name = ""
    worst_dataset_rate = -1.0

    for dataset_name, payload in eval_summaries.items():
        direct_solve_accuracy_payload = payload.get("direct_solve_accuracy", {})
        direct_solve_accuracy = direct_solve_accuracy_payload.get("overall_mean")
        exclusions = payload.get("exclusions", {})
        categories = exclusions.get("categories", {})
        format_failure_breakdown = {
            category: float(categories.get(category, {}).get("proportion", 0.0))
            for category in FORMAT_FAILURE_CATEGORIES
        }
        format_failure_rate = float(sum(format_failure_breakdown.values()))
        total_exclusion_rate = float(exclusions.get("total", {}).get("proportion", 0.0))

        reasons: list[str] = []
        # Only require direct_solve_accuracy when the dataset actually produced
        # direct-solve rows (non-empty payload dict).  Confirmation-only datasets
        # (e.g. test_paraphrase, same_domain_extrapolation) have no direct-solve
        # rows and legitimately produce an empty dict here; penalising them for a
        # missing accuracy would wrongly mark them unacceptable.
        if direct_solve_accuracy_payload and direct_solve_accuracy is None:
            reasons.append("missing_direct_solve_accuracy")
        if format_failure_rate > max_format_failure_rate:
            reasons.append("format_failure_rate_above_threshold")

        dataset_summary = {
            "direct_solve_accuracy": (
                None if direct_solve_accuracy is None else float(direct_solve_accuracy)
            ),
            "format_failure_rate": format_failure_rate,
            "format_failure_breakdown": format_failure_breakdown,
            "total_exclusion_rate": total_exclusion_rate,
            "acceptable": not reasons,
            "reasons": reasons,
        }
        dataset_summaries[dataset_name] = dataset_summary

        if format_failure_rate > worst_dataset_rate:
            worst_dataset_name = dataset_name
            worst_dataset_rate = format_failure_rate
        if reasons:
            unacceptable_datasets.append(dataset_name)

    return {
        "max_format_failure_rate": float(max_format_failure_rate),
        "acceptable": not unacceptable_datasets,
        "unacceptable_datasets": unacceptable_datasets,
        "worst_dataset": {
            "dataset_name": worst_dataset_name,
            "format_failure_rate": worst_dataset_rate,
        },
        "datasets": dataset_summaries,
    }


def run_base_model_evaluation(args: argparse.Namespace) -> BaseModelEvalRun:
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    from all_evals import PreregisteredEvaluator, SemanticInterfaceEvaluator
    from transformers import AutoTokenizer

    evaluation_interface = getattr(args, "evaluation_interface", "fixed_interface")
    is_semantic = evaluation_interface == "semantic_interface"

    datasets = parse_dataset_specs(args.datasets)
    generation_kwargs = make_generation_kwargs(args)
    vllm_kwargs = make_vllm_kwargs(args)
    tokenizer_name = args.tokenizer_name or args.model_name
    lmstudio_kwargs = {
        "model_name": args.lmstudio_model_name or args.model_name,
        "base_url": args.lmstudio_base_url,
        "request_timeout": args.lmstudio_request_timeout,
    }
    selected_prefix_artifact = None
    if not is_semantic and args.selected_prefix_artifact is not None:
        selected_prefix_artifact = load_selected_prefix_artifact(
            resolve_repo_relative_path(args.selected_prefix_artifact)
        )
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
        selected_prefix_artifact=selected_prefix_artifact,
        evaluation_interface=evaluation_interface,
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

        if is_semantic:
            logging.info(
                "Using SemanticInterfaceEvaluator (secondary robustness path, "
                "evaluation_design='semantic_interface')."
            )
            evaluator = SemanticInterfaceEvaluator(
                llm=llm,
                tokenizer=tokenizer,
                generation_kwargs=generation_kwargs,
                llm_backend=args.llm_backend,
                ptst_only=args.evaluation_mode == PTST_ARM_NAME,
                arm_name=args.evaluation_mode,
            )
        else:
            evaluator = PreregisteredEvaluator(
                llm=llm,
                tokenizer=tokenizer,
                generation_kwargs=generation_kwargs,
                llm_backend=args.llm_backend,
                ptst_only=args.evaluation_mode == PTST_ARM_NAME,
                arm_name=args.evaluation_mode,
                user_message_prefix=(
                    selected_prefix_artifact["selected_prefix_text"]
                    if selected_prefix_artifact is not None
                    else ""
                ),
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
