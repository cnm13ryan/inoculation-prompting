#!/usr/bin/env python3
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluate_base_model
import run_prereg_prefix_search
from run_preregistration import DEFAULT_FIXED_INTERFACE_MAX_FORMAT_FAILURE_RATE


DEFAULT_FIXED_INTERFACE_OUTPUT_ROOT = (
    evaluate_base_model.PROJECTS_DIR
    / "experiments"
    / "prereg"
    / "best_elicited_fixed_interface_baselines"
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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
        help=(
            "Confirmatory dataset specs in 'test_name:path' format for the final "
            "selected-prefix evaluation only. The fixed-interface gate always runs "
            "on the prereg baseline dataset suite."
        ),
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
    parser.add_argument("--search-output-dir", type=Path, default=None)
    parser.add_argument("--search-output-root", type=Path, default=run_prereg_prefix_search.DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--fixed-interface-output-dir", type=Path, default=None)
    parser.add_argument(
        "--fixed-interface-output-root",
        type=Path,
        default=DEFAULT_FIXED_INTERFACE_OUTPUT_ROOT,
    )
    parser.add_argument("--eval-output-dir", type=Path, default=None)
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
    parser.add_argument(
        "--fixed-interface-max-format-failure-rate",
        type=float,
        default=DEFAULT_FIXED_INTERFACE_MAX_FORMAT_FAILURE_RATE,
        help=(
            "Maximum acceptable fixed-interface formatting failure rate per dataset "
            "before bounded search is treated as uninterpretable without an explicit override."
        ),
    )
    parser.add_argument(
        "--allow-unacceptable-fixed-interface-for-prefix-search",
        action="store_true",
        help=(
            "Allow bounded prefix search to run even when the fixed-interface baseline "
            "fails the formatting-quality gate. The warning is recorded in the output "
            "payload, the baseline report, and the selected-prefix artifact."
        ),
    )
    return parser


def parse_args() -> argparse.Namespace:
    return build_arg_parser().parse_args()


def _baseline_report_path(fixed_interface_experiment_dir: Path) -> Path:
    return fixed_interface_experiment_dir / "fixed_interface_baseline_report.json"


def _build_fixed_interface_eval_args(args: argparse.Namespace) -> argparse.Namespace:
    fixed_interface_args_list = [
        "--datasets",
        *evaluate_base_model.DEFAULT_DATASETS,
        *(
            ["--output-dir", str(args.fixed_interface_output_dir)]
            if args.fixed_interface_output_dir is not None
            else ["--output-root", str(args.fixed_interface_output_root)]
        ),
        "--evaluation-mode",
        args.evaluation_mode,
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
        fixed_interface_args_list.extend(["--tokenizer-name", args.tokenizer_name])
    if args.tensor_parallel_size is not None:
        fixed_interface_args_list.extend(
            ["--tensor-parallel-size", str(args.tensor_parallel_size)]
        )
    if args.gpu_memory_utilization is not None:
        fixed_interface_args_list.extend(
            ["--gpu-memory-utilization", str(args.gpu_memory_utilization)]
        )
    if args.dtype is not None:
        fixed_interface_args_list.extend(["--dtype", args.dtype])
    if args.max_model_len is not None:
        fixed_interface_args_list.extend(["--max-model-len", str(args.max_model_len)])
    if args.limit is not None:
        fixed_interface_args_list.extend(["--limit", str(args.limit)])
    if args.timestamp is not None:
        fixed_interface_args_list.extend(["--timestamp", args.timestamp])
    return evaluate_base_model.build_arg_parser().parse_args(fixed_interface_args_list)


def _build_fixed_interface_baseline_report(
    *,
    args: argparse.Namespace,
    run_output: evaluate_base_model.BaseModelEvalRun,
) -> dict:
    summaries = evaluate_base_model.load_eval_result_summaries(run_output.model_dir)
    quality_summary = evaluate_base_model.compute_fixed_interface_quality_summary(
        summaries,
        max_format_failure_rate=args.fixed_interface_max_format_failure_rate,
    )
    assessment = {
        **quality_summary,
        "arm_slug": args.evaluation_mode,
        "seed": 0,
        "output_dir": str(run_output.experiment_dir),
        "model_dir": str(run_output.model_dir),
    }
    unacceptable = []
    if not assessment["acceptable"]:
        unacceptable.append(
            {
                "arm_slug": assessment["arm_slug"],
                "seed": assessment["seed"],
                "unacceptable_datasets": assessment["unacceptable_datasets"],
                "worst_dataset": assessment["worst_dataset"],
            }
        )

    return {
        "workflow_name": "preregistered_fixed_interface_baseline_report",
        "generated_at_utc": _now_iso(),
        "evaluation_interface": "preregistered_fixed_interface",
        "max_format_failure_rate": args.fixed_interface_max_format_failure_rate,
        "allow_unacceptable_fixed_interface_for_prefix_search": bool(
            args.allow_unacceptable_fixed_interface_for_prefix_search
        ),
        "summary": {
            "total_assessments": 1,
            "acceptable_assessments": 1 if assessment["acceptable"] else 0,
            "unacceptable_assessments": len(unacceptable),
        },
        "unacceptable_assessments": unacceptable,
        "assessments": [assessment],
    }


def _annotate_selected_prefix_artifact(
    artifact_path: Path,
    *,
    baseline_report: dict,
) -> None:
    payload = _read_json(artifact_path)
    assessment = baseline_report["assessments"][0]
    payload["fixed_interface_baseline_assessment"] = {
        "acceptable": assessment["acceptable"],
        "max_format_failure_rate": assessment["max_format_failure_rate"],
        "unacceptable_datasets": assessment["unacceptable_datasets"],
        "worst_dataset": assessment["worst_dataset"],
    }
    if baseline_report["unacceptable_assessments"] and baseline_report[
        "allow_unacceptable_fixed_interface_for_prefix_search"
    ]:
        payload["bounded_search_interpretation_warning"] = (
            "Bounded search was run even though the fixed-interface baseline exceeded the "
            "configured format-failure threshold. Treat the selected prefix as exploratory "
            "repair-sensitive output, not a clean estimate of bounded-search benefit."
        )
    _write_json(artifact_path, payload)


def run_best_elicited_evaluations(args: argparse.Namespace) -> dict:
    fixed_interface_args = _build_fixed_interface_eval_args(args)
    fixed_interface_run = evaluate_base_model.run_base_model_evaluation(
        fixed_interface_args
    )
    baseline_report = _build_fixed_interface_baseline_report(
        args=args,
        run_output=fixed_interface_run,
    )
    report_path = _baseline_report_path(fixed_interface_run.experiment_dir)
    _write_json(report_path, baseline_report)
    warning = None
    if baseline_report["unacceptable_assessments"]:
        failing_runs = "; ".join(
            (
                f"{item['arm_slug']}/seed_{item['seed']}: "
                f"datasets={','.join(item['unacceptable_datasets'])}, "
                f"worst={item['worst_dataset']['dataset_name']} "
                f"({item['worst_dataset']['format_failure_rate']:.3f})"
            )
            for item in baseline_report["unacceptable_assessments"][:5]
        )
        warning = (
            "Fixed-interface baseline quality is unacceptable for bounded-search interpretation. "
            "Bounded prefix search should not function as the repair path for a broken fixed interface. "
            f"Failing runs: {failing_runs}"
        )
    gate_passed = not baseline_report["unacceptable_assessments"]
    if not gate_passed and not args.allow_unacceptable_fixed_interface_for_prefix_search:
        raise RuntimeError(
            f"{warning} Re-run with "
            "--allow-unacceptable-fixed-interface-for-prefix-search if you need "
            "to continue anyway and explicitly record the warning."
        )
    if not gate_passed and args.allow_unacceptable_fixed_interface_for_prefix_search:
        print(f"WARNING: {warning}", flush=True)

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
            *(
                ["--output-dir", str(args.search_output_dir)]
                if args.search_output_dir is not None
                else ["--output-root", str(args.search_output_root)]
            ),
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
    _annotate_selected_prefix_artifact(
        Path(search_summary["artifact_path"]),
        baseline_report=baseline_report,
    )

    eval_args_list = [
        "--datasets",
        *args.datasets,
        *(
            ["--output-dir", str(args.eval_output_dir)]
            if args.eval_output_dir is not None
            else ["--output-root", str(args.eval_output_root)]
        ),
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
        "fixed_interface_baseline": {
            "report_path": str(report_path),
            "gate_passed": gate_passed,
            "warning": warning,
            "report": baseline_report,
        },
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
