#!/usr/bin/env python3
"""Evaluate behavioral curves over saved step-based model checkpoints.

For each checkpoint directory found under {seed_dir}/checkpoints/step_*/,
run the fixed-interface evaluation and aggregate per-step sycophancy metrics.
Produces a CSV and JSON output for downstream instability analysis.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
GEMMA_GCD_DIR = SCRIPT_DIR.parent
PROJECTS_DIR = GEMMA_GCD_DIR.parent
for candidate in (SCRIPT_DIR, GEMMA_GCD_DIR, PROJECTS_DIR):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from evaluate_base_model import load_eval_result_summaries  # noqa: E402


def _latest_eval_model_dir(output_dir: Path) -> Path:
    """Find the most recent model eval results directory under output_dir."""
    if not output_dir.exists():
        raise RuntimeError(f"Missing checkpoint-curve eval output directory: {output_dir}")
    candidates = [
        path
        for path in sorted(output_dir.glob("results/*/*"), reverse=True)
        if path.is_dir() and any(path.glob("*_eval_results.json"))
    ]
    if not candidates:
        raise RuntimeError(
            f"Expected fixed-interface evaluation summaries under {output_dir}, "
            "but none were found."
        )
    return candidates[0]


logger = logging.getLogger(__name__)

CURVE_CSV_SUFFIX = ".curve.csv"
CURVE_JSON_SUFFIX = ".curve.json"

_STEP_DIR_PREFIX = "step_"


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def discover_step_checkpoint_dirs(seed_dir: Path) -> list[tuple[int, Path]]:
    """Return (step, path) pairs for all step_* checkpoint dirs, sorted by step."""
    checkpoints_root = seed_dir / "checkpoints"
    if not checkpoints_root.is_dir():
        return []
    results: list[tuple[int, Path]] = []
    for item in checkpoints_root.iterdir():
        if not item.is_dir():
            continue
        if not item.name.startswith(_STEP_DIR_PREFIX):
            continue
        step_str = item.name[len(_STEP_DIR_PREFIX):]
        try:
            step = int(step_str)
        except ValueError:
            continue
        metadata_path = item / "metadata.json"
        if not metadata_path.exists():
            continue
        results.append((step, item))
    return sorted(results, key=lambda pair: pair[0])


def load_step_metadata(step_dir: Path) -> dict[str, Any]:
    return _read_json(step_dir / "metadata.json")


def _is_adapter_only_checkpoint(step_dir: Path) -> bool:
    """Detect step checkpoints that hold only a PEFT/LoRA adapter.

    Adapter ckpts are ~30-80 MB (just adapter_config.json +
    adapter_model.safetensors), versus ~5 GB for a merged Gemma-2B.
    Returns True if adapter_config.json is present, OR if metadata.json
    declares ``checkpoint_kind == "lora_adapter"``. The metadata flag is
    the authoritative signal; the file-existence check is the fallback
    so legacy adapter dirs without the flag still take the adapter path.
    """
    if not step_dir.is_dir():
        return False
    metadata_path = step_dir / "metadata.json"
    if metadata_path.exists():
        try:
            kind = _read_json(metadata_path).get("checkpoint_kind")
        except Exception:
            kind = None
        if kind == "lora_adapter":
            return True
        if kind == "merged_full_model":
            return False
    return (step_dir / "adapter_config.json").exists()


def _materialize_adapter_to_tempdir(adapter_dir: Path) -> Path:
    """Load base + LoRA adapter, merge, save to a fresh temp dir.

    Caller is responsible for ``shutil.rmtree(temp_dir)`` once the eval
    consuming the merged model has finished. Imports are local so the
    module stays importable in environments without torch/peft (matters
    for the GPU-free test suite).
    """
    import torch  # noqa: PLC0415  (local import: keep module import-light)
    from peft import PeftConfig, PeftModel  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    adapter_config = PeftConfig.from_pretrained(str(adapter_dir))
    base_model_name = adapter_config.base_model_name_or_path
    if not base_model_name:
        raise RuntimeError(
            f"Adapter at {adapter_dir} is missing base_model_name_or_path; "
            "cannot rehydrate merged model."
        )
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{adapter_dir.name}_merged_"))
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=torch.float16
        )
        peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
        merged = peft_model.merge_and_unload()
        merged.save_pretrained(str(temp_dir))
        tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(temp_dir))
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return temp_dir


@contextlib.contextmanager
def _resolved_eval_model_dir(step_dir: Path) -> Iterator[Path]:
    """Yield a directory ready for vLLM, materializing adapter ckpts on the fly.

    For a merged-full-model step ckpt (legacy path or non-PEFT training),
    yields ``step_dir`` unchanged. For an adapter-only step ckpt, merges
    the adapter against its base model into a temp dir, yields the temp
    dir, and unconditionally cleans it up on exit so disk pressure is
    bounded by *one* materialization at a time, not the full curve.
    """
    if not _is_adapter_only_checkpoint(step_dir):
        yield step_dir
        return
    merged_dir = _materialize_adapter_to_tempdir(step_dir)
    try:
        yield merged_dir
    finally:
        shutil.rmtree(merged_dir, ignore_errors=True)


def _extract_curve_metrics(
    eval_summaries: dict[str, dict[str, Any]],
    eval_set_name: str,
) -> dict[str, Any]:
    if eval_set_name not in eval_summaries:
        available = sorted(eval_summaries)
        if not available:
            return {
                "eval_set_name": eval_set_name,
                "total_rows": None,
                "included_rows": None,
                "parseability_rate": None,
                "sycophancy_rate_given_parseable": None,
                "direct_solve_accuracy": None,
            }
        raise ValueError(
            f"Requested eval set {eval_set_name!r} not found in loaded summaries "
            f"(available: {available}). The cached output directory may contain "
            "results from a different dataset. Delete it and re-run to force a "
            "fresh evaluation."
        )

    payload = eval_summaries[eval_set_name]
    exclusions = payload.get("exclusions", {})
    total_exclusion = exclusions.get("total", {})
    total_count = total_exclusion.get("count")
    total_proportion = total_exclusion.get("proportion")
    parseability_rate = (1.0 - float(total_proportion)) if total_proportion is not None else None

    conditional_syco = payload.get("conditional_sycophancy_rate", {})
    sycophancy_rate_given_parseable = conditional_syco.get("overall_mean")
    if sycophancy_rate_given_parseable is not None:
        sycophancy_rate_given_parseable = float(sycophancy_rate_given_parseable)

    direct_solve = payload.get("direct_solve_accuracy", {})
    direct_solve_accuracy = direct_solve.get("overall_mean") if direct_solve else None
    if direct_solve_accuracy is not None:
        direct_solve_accuracy = float(direct_solve_accuracy)

    included_count = None
    if total_count is not None and total_proportion is not None and total_proportion <= 1.0:
        excluded = int(round(float(total_count)))
        if parseability_rate is not None and total_count is not None:
            included_count = total_count - excluded if total_proportion == 1.0 else None

    return {
        "eval_set_name": eval_set_name,
        "total_rows": total_count,
        "included_rows": included_count,
        "parseability_rate": parseability_rate,
        "sycophancy_rate_given_parseable": sycophancy_rate_given_parseable,
        "direct_solve_accuracy": direct_solve_accuracy,
    }


def _default_run_eval(
    *,
    model_dir: Path,
    output_dir: Path,
    dataset_spec: str,
    llm_backend: str,
    lmstudio_base_url: str,
    lmstudio_request_timeout: float,
    lmstudio_model_name: str | None,
    tensor_parallel_size: int | None,
    gpu_memory_utilization: float | None,
    dtype: str | None,
    max_model_len: int | None,
    limit: int | None,
    timestamp: str | None,
    log_level: str,
) -> None:
    fixed_eval_script = SCRIPT_DIR / "evaluate_base_model.py"
    cmd = [
        sys.executable,
        str(fixed_eval_script),
        "--model-name",
        str(model_dir),
        "--evaluation-mode",
        "neutral",
        "--output-dir",
        str(output_dir),
        "--datasets",
        dataset_spec,
        "--llm-backend",
        llm_backend,
        "--lmstudio-base-url",
        lmstudio_base_url,
        "--lmstudio-request-timeout",
        str(lmstudio_request_timeout),
        "--log-level",
        log_level,
    ]
    if lmstudio_model_name is not None:
        cmd.extend(["--lmstudio-model-name", lmstudio_model_name])
    if tensor_parallel_size is not None:
        cmd.extend(["--tensor-parallel-size", str(tensor_parallel_size)])
    if gpu_memory_utilization is not None:
        cmd.extend(["--gpu-memory-utilization", str(gpu_memory_utilization)])
    if dtype is not None:
        cmd.extend(["--dtype", dtype])
    if max_model_len is not None:
        cmd.extend(["--max-model-len", str(max_model_len)])
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    if timestamp is not None:
        cmd.extend(["--timestamp", timestamp])
    logger.info("Running checkpoint curve eval: %s", " ".join(str(p) for p in cmd))
    subprocess.run(cmd, cwd=str(PROJECTS_DIR), check=True)


def _dataset_spec_name(dataset_spec: str) -> str:
    if ":" in dataset_spec:
        return dataset_spec.split(":", 1)[0]
    return Path(dataset_spec).stem


def _has_eval_results(output_dir: Path) -> bool:
    """Has the per-step eval actually produced consumable results yet?

    Aligned with ``_latest_eval_model_dir``: requires at least one
    ``results/<timestamp>/<model_dir>/*_eval_results.json``. The older
    ``any(output_dir.glob("results/*/*"))`` form was too lax — a crashed
    eval that wrote only ``inference_config.json`` would still pass this
    check, causing the curve loop to skip the rerun and then fail when
    extracting metrics.
    """
    if not output_dir.exists():
        return False
    for candidate in output_dir.glob("results/*/*"):
        if candidate.is_dir() and any(candidate.glob("*_eval_results.json")):
            return True
    return False


def evaluate_checkpoint_curve(
    *,
    seed_dir: Path,
    arm_slug: str,
    seed: int,
    dataset_spec: str,
    output_prefix: Path,
    checkpoint_curve_limit: int,
    llm_backend: str = "vllm",
    lmstudio_base_url: str = "http://localhost:1234",
    lmstudio_request_timeout: float = 120.0,
    lmstudio_model_name: str | None = None,
    tensor_parallel_size: int | None = None,
    gpu_memory_utilization: float | None = None,
    dtype: str | None = None,
    max_model_len: int | None = None,
    limit: int | None = None,
    timestamp: str | None = None,
    log_level: str = "INFO",
    run_eval_fn: Callable[..., None] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if run_eval_fn is None:
        run_eval_fn = _default_run_eval

    step_dirs = discover_step_checkpoint_dirs(seed_dir)
    if not step_dirs:
        logger.warning(
            "No step checkpoints found under %s/checkpoints/step_*/", seed_dir
        )
        empty_df: pd.DataFrame = pd.DataFrame()
        metadata: dict[str, Any] = {
            "workflow_name": "checkpoint_curve_eval",
            "generated_at_utc": _now_iso(),
            "arm_slug": arm_slug,
            "seed": seed,
            "checkpoint_count": 0,
            "dataset": dataset_spec,
            "curve": [],
        }
        return empty_df, metadata

    limited = step_dirs[:checkpoint_curve_limit]
    eval_set_name = _dataset_spec_name(dataset_spec)
    curve_rows: list[dict[str, Any]] = []

    for step, step_dir in limited:
        step_metadata = load_step_metadata(step_dir)
        epoch = step_metadata.get("epoch")
        train_loss = step_metadata.get("train_loss")
        step_output_dir = output_prefix.parent / f"step_{step:06d}"

        if not _has_eval_results(step_output_dir):
            with _resolved_eval_model_dir(step_dir) as resolved_model_dir:
                run_eval_fn(
                    model_dir=resolved_model_dir,
                    output_dir=step_output_dir,
                    dataset_spec=dataset_spec,
                    llm_backend=llm_backend,
                    lmstudio_base_url=lmstudio_base_url,
                    lmstudio_request_timeout=lmstudio_request_timeout,
                    lmstudio_model_name=lmstudio_model_name,
                    tensor_parallel_size=tensor_parallel_size,
                    gpu_memory_utilization=gpu_memory_utilization,
                    dtype=dtype,
                    max_model_len=max_model_len,
                    limit=limit,
                    timestamp=timestamp,
                    log_level=log_level,
                )

        try:
            model_dir = _latest_eval_model_dir(step_output_dir)
            eval_summaries = load_eval_result_summaries(model_dir)
        except Exception as exc:
            logger.warning(
                "Could not load eval results for step %d at %s: %s",
                step, step_output_dir, exc,
            )
            eval_summaries = {}

        metrics = _extract_curve_metrics(eval_summaries, eval_set_name)
        row: dict[str, Any] = {
            "arm_slug": arm_slug,
            "seed": seed,
            "checkpoint_step": step,
            "epoch": epoch,
            "train_loss": train_loss,
            **metrics,
        }
        curve_rows.append(row)
        logger.info(
            "Step %d: parseability=%.3f, sycophancy_given_parseable=%s",
            step,
            metrics.get("parseability_rate") or 0.0,
            metrics.get("sycophancy_rate_given_parseable"),
        )

    curve_df = pd.DataFrame(curve_rows)
    metadata = {
        "workflow_name": "checkpoint_curve_eval",
        "generated_at_utc": _now_iso(),
        "arm_slug": arm_slug,
        "seed": seed,
        "checkpoint_count": len(limited),
        "checkpoint_curve_limit": checkpoint_curve_limit,
        "dataset": dataset_spec,
        "curve": curve_rows,
    }
    return curve_df, metadata


def write_outputs(
    *,
    curve_df: pd.DataFrame,
    metadata: dict[str, Any],
    output_prefix: Path,
) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_prefix.parent / f"{output_prefix.name}{CURVE_CSV_SUFFIX}"
    json_path = output_prefix.parent / f"{output_prefix.name}{CURVE_JSON_SUFFIX}"
    curve_df.to_csv(csv_path, index=False, na_rep="NA")
    _write_json(json_path, metadata)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed-dir",
        type=Path,
        required=True,
        help="Seed directory containing checkpoints/step_*/ subdirectories.",
    )
    parser.add_argument("--arm-slug", required=True, help="Arm slug for this seed run.")
    parser.add_argument("--seed", type=int, required=True, help="Seed index.")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset spec ('name:path' or bare path) for the curve evaluation.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        required=True,
        help="Output prefix for .curve.csv and .curve.json artifacts.",
    )
    parser.add_argument(
        "--checkpoint-curve-limit",
        type=int,
        default=32,
        help="Maximum number of step checkpoints to evaluate (default: 32).",
    )
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    curve_df, metadata = evaluate_checkpoint_curve(
        seed_dir=args.seed_dir.resolve(),
        arm_slug=args.arm_slug,
        seed=args.seed,
        dataset_spec=args.dataset,
        output_prefix=args.output_prefix.resolve(),
        checkpoint_curve_limit=args.checkpoint_curve_limit,
        llm_backend=args.llm_backend,
        lmstudio_base_url=args.lmstudio_base_url,
        lmstudio_request_timeout=float(args.lmstudio_request_timeout),
        lmstudio_model_name=args.lmstudio_model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        limit=args.limit,
        timestamp=args.timestamp,
        log_level=args.log_level,
    )
    write_outputs(
        curve_df=curve_df,
        metadata=metadata,
        output_prefix=args.output_prefix.resolve(),
    )


if __name__ == "__main__":
    main()
