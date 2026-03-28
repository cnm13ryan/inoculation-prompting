#!/usr/bin/env python3
"""Run pushback evaluations over each condition/seed in an IP sweep."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


_SCRIPTS_DIR = Path(__file__).resolve().parent
_PROJECTS_DIR = _SCRIPTS_DIR.parent.parent
_DEFAULT_EXPERIMENTS_DIR = _PROJECTS_DIR / "experiments" / "ip_sweep"
_EVALUATE_SCRIPT = _SCRIPTS_DIR / "evaluate_base_model.py"
_DEFAULT_DATASETS = [
    "task_test:gemma_gcd/data/task_test.jsonl",
    "ood_test:gemma_gcd/data/ood_test.jsonl",
]
_DEFAULT_OUTPUT_ROOT = _DEFAULT_EXPERIMENTS_DIR / "pushback_evals"
_MODEL_MARKER_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "config.json",
    "model.safetensors",
    "pytorch_model.bin",
    "tokenizer_config.json",
)


def load_condition_labels(experiments_dir: Path) -> dict[str, str]:
    labels_path = experiments_dir / "condition_labels.json"
    if not labels_path.exists():
        raise FileNotFoundError(f"condition_labels.json not found at {labels_path}")
    with labels_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def discover_condition_dirs(experiments_dir: Path, condition_labels: dict[str, str]) -> list[Path]:
    condition_dirs: list[Path] = []
    for condition_name in sorted(condition_labels):
        condition_dir = experiments_dir / condition_name
        if not condition_dir.is_dir():
            continue
        if any(child.is_dir() and child.name.startswith("seed_") for child in condition_dir.iterdir()):
            condition_dirs.append(condition_dir)
    return condition_dirs


def discover_seed_dirs(condition_dir: Path) -> list[tuple[int, Path]]:
    seed_dirs: list[tuple[int, Path]] = []
    for child in sorted(condition_dir.iterdir()):
        if not child.is_dir() or not child.name.startswith("seed_"):
            continue
        try:
            seed = int(child.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        seed_dirs.append((seed, child))
    return seed_dirs


def filter_condition_dirs(
    condition_dirs: list[Path],
    condition_labels: dict[str, str],
    selected_labels: list[str] | None,
) -> list[Path]:
    if not selected_labels:
        return condition_dirs

    selected = set(selected_labels)
    filtered = [
        condition_dir
        for condition_dir in condition_dirs
        if condition_dir.name in selected or condition_labels.get(condition_dir.name) in selected
    ]
    if not filtered:
        raise ValueError(f"No condition directories matched --condition-labels={selected_labels!r}")
    return filtered


def filter_seed_dirs(
    seed_dirs: list[tuple[int, Path]],
    selected_seeds: list[int] | None,
) -> list[tuple[int, Path]]:
    if selected_seeds is None:
        return seed_dirs
    selected = set(selected_seeds)
    return [(seed, seed_dir) for seed, seed_dir in seed_dirs if seed in selected]


def extract_latest_result_dir(seed_dir: Path) -> Path:
    results_dir = seed_dir / "results"
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    timestamps = sorted(item for item in results_dir.iterdir() if item.is_dir())
    if not timestamps:
        raise FileNotFoundError(f"No timestamp directories found in: {results_dir}")
    return timestamps[-1]


def _looks_like_model_dir(candidate: Path) -> bool:
    if not candidate.is_dir():
        return False
    marker_names = {path.name for path in candidate.iterdir() if path.is_file()}
    if "tokenizer_config.json" not in marker_names:
        return False
    return any(name in marker_names for name in _MODEL_MARKER_FILES if name != "tokenizer_config.json")


def resolve_model_path(seed_dir: Path) -> Path:
    latest_result_dir = extract_latest_result_dir(seed_dir)
    model_dirs = [
        child
        for child in sorted(latest_result_dir.iterdir())
        if _looks_like_model_dir(child)
    ]
    if not model_dirs:
        raise FileNotFoundError(
            f"No saved model directory found in latest result dir: {latest_result_dir}"
        )
    return model_dirs[0]


def find_eval_model_dir(output_dir: Path) -> Path:
    latest_result_dir = extract_latest_result_dir(output_dir)
    eval_dirs = [
        child
        for child in sorted(latest_result_dir.iterdir())
        if child.is_dir() and any(child.glob("*_eval_results.json"))
    ]
    if not eval_dirs:
        raise FileNotFoundError(f"No evaluation model directory found in {latest_result_dir}")
    return eval_dirs[0]


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


def recover_output_payload(output_dir: Path) -> dict[str, Any]:
    model_dir = find_eval_model_dir(output_dir)
    config_path = output_dir / "config.json"
    eval_protocol = "pushback"
    eval_suffix = None
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        eval_protocol = config.get("eval_protocol", eval_protocol)
        eval_suffix = config.get("eval_user_suffix")
    payload: dict[str, Any] = {
        "experiment_dir": str(output_dir),
        "model_dir": str(model_dir),
        "eval_protocol": eval_protocol,
        "summaries": load_eval_result_summaries(model_dir),
    }
    if eval_suffix is not None:
        payload["eval_suffix"] = eval_suffix
    return payload


def build_evaluate_command(
    *,
    model_name: Path,
    output_dir: Path,
    mode: str,
    args: argparse.Namespace,
) -> list[str]:
    cmd = [
        sys.executable,
        str(_EVALUATE_SCRIPT),
        "--model-name",
        str(model_name),
        "--datasets",
        *args.datasets,
        "--eval-protocol",
        "pushback",
        "--llm-backend",
        args.llm_backend,
        "--attributes-to-vary",
        str(resolve_attributes_to_vary_path(args)),
        "--eval-suffix-mode",
        mode,
        "--output-dir",
        str(output_dir),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--log-level",
        args.log_level,
    ]
    if args.tokenizer_name:
        cmd += ["--tokenizer-name", args.tokenizer_name]
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    if args.pushback_message_correct:
        cmd += ["--pushback-message-correct", args.pushback_message_correct]
    if args.pushback_message_incorrect:
        cmd += ["--pushback-message-incorrect", args.pushback_message_incorrect]
    if args.max_model_len is not None:
        cmd += ["--max-model-len", str(args.max_model_len)]
    if args.llm_backend == "vllm":
        if args.tensor_parallel_size is not None:
            cmd += ["--tensor-parallel-size", str(args.tensor_parallel_size)]
        if args.gpu_memory_utilization is not None:
            cmd += ["--gpu-memory-utilization", str(args.gpu_memory_utilization)]
        if args.dtype:
            cmd += ["--dtype", args.dtype]
    else:
        cmd += [
            "--lmstudio-base-url",
            args.lmstudio_base_url,
            "--lmstudio-model-name",
            args.lmstudio_model_name or str(model_name),
            "--lmstudio-request-timeout",
            str(args.lmstudio_request_timeout),
        ]
    return cmd


def resolve_attributes_to_vary_path(args: argparse.Namespace) -> Path:
    if args.attributes_to_vary is not None:
        return args.attributes_to_vary.resolve()
    return (args.experiments_dir / "attributes_to_vary.json").resolve()


def run_evaluate_command(cmd: list[str], *, cwd: Path) -> dict[str, Any]:
    print(f"\n>>> {' '.join(str(part) for part in cmd)}\n", file=sys.stderr, flush=True)
    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        if result.stdout:
            print(result.stdout, file=sys.stderr, end="" if result.stdout.endswith("\n") else "\n")
        if result.stderr:
            print(result.stderr, file=sys.stderr, end="" if result.stderr.endswith("\n") else "\n")
        raise RuntimeError(
            f"evaluate_base_model.py exited with code {result.returncode} for command: {' '.join(cmd)}"
        )
    stdout = result.stdout.strip()
    if stdout:
        try:
            return json.loads(stdout)
        except json.JSONDecodeError:
            pass

    if "--output-dir" in cmd:
        output_dir = Path(cmd[cmd.index("--output-dir") + 1])
        try:
            return recover_output_payload(output_dir)
        except FileNotFoundError:
            pass

    raise RuntimeError(
        f"Failed to parse evaluate_base_model.py JSON output for command: {' '.join(cmd)}"
    )


def run_pushback_sweep(args: argparse.Namespace) -> list[dict[str, Any]]:
    experiments_dir = args.experiments_dir.resolve()
    condition_labels = load_condition_labels(experiments_dir)
    condition_dirs = discover_condition_dirs(experiments_dir, condition_labels)
    condition_dirs = filter_condition_dirs(condition_dirs, condition_labels, args.condition_labels)

    records: list[dict[str, Any]] = []
    for condition_dir in condition_dirs:
        label = condition_labels[condition_dir.name]
        seed_dirs = filter_seed_dirs(discover_seed_dirs(condition_dir), args.seeds)
        for seed, seed_dir in seed_dirs:
            model_path = resolve_model_path(seed_dir)
            for mode in ("neutral", "pressure"):
                output_dir = args.output_root / condition_dir.name / seed_dir.name / mode
                payload = run_evaluate_command(
                    build_evaluate_command(
                        model_name=model_path,
                        output_dir=output_dir,
                        mode=mode,
                        args=args,
                    ),
                    cwd=_PROJECTS_DIR,
                )
                payload.update(
                    {
                        "condition_dir": condition_dir.name,
                        "condition_label": label,
                        "seed": seed,
                        "backend": args.llm_backend,
                        "mode": mode,
                        "model_name": str(model_path),
                    }
                )
                records.append(payload)
    return records


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run pushback evaluation over each condition/seed in an IP sweep.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=_DEFAULT_EXPERIMENTS_DIR,
        help="IP sweep directory containing condition_labels.json and condition subdirectories.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_DEFAULT_OUTPUT_ROOT,
        help="Root directory where pushback evaluation outputs will be written.",
    )
    parser.add_argument(
        "--llm-backend",
        choices=("vllm", "lmstudio"),
        default="vllm",
        help="Inference backend passed through to evaluate_base_model.py.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Restrict evaluation to these seeds. Defaults to all seeds present per condition.",
    )
    parser.add_argument(
        "--condition-labels",
        nargs="+",
        default=None,
        help="Restrict evaluation to these human-readable condition labels or raw condition directory names.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=_DEFAULT_DATASETS,
        help="Dataset specs in 'test_name:path' format.",
    )
    parser.add_argument(
        "--attributes-to-vary",
        type=Path,
        default=None,
        help=(
            "Path to the sweep attributes_to_vary.json used by pressure-mode suffix "
            "resolution. Defaults to <experiments-dir>/attributes_to_vary.json."
        ),
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit per dataset.")
    parser.add_argument(
        "--tokenizer-name",
        default=None,
        help="Optional tokenizer override passed through to evaluate_base_model.py.",
    )
    parser.add_argument(
        "--pushback-message-correct",
        default=None,
        help="Override the pushback message shown when the model resists a correct answer.",
    )
    parser.add_argument(
        "--pushback-message-incorrect",
        default=None,
        help="Override the pushback message shown when the model corrects an incorrect answer.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument(
        "--lmstudio-base-url",
        default="http://192.168.1.228:1234",
        help="LM Studio base URL used when --llm-backend=lmstudio.",
    )
    parser.add_argument(
        "--lmstudio-model-name",
        default=None,
        help="LM Studio served model name used when --llm-backend=lmstudio.",
    )
    parser.add_argument(
        "--lmstudio-request-timeout",
        type=float,
        default=120.0,
        help="LM Studio request timeout in seconds.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    args.output_root = args.output_root.resolve()
    records = run_pushback_sweep(args)
    print(json.dumps(records, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
