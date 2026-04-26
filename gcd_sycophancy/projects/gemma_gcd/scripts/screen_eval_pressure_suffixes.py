#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import textwrap
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
GEMMA_GCD_DIR = SCRIPT_DIR.parent
PROJECTS_DIR = GEMMA_GCD_DIR.parent
for path in (GEMMA_GCD_DIR, PROJECTS_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


DEFAULT_MODEL_NAME = "google/gemma-2b-it"
DEFAULT_TEST_DATA = Path("gemma_gcd/data/ood_test.jsonl")
DEFAULT_CANDIDATES = Path("experiments/ip_sweep/eval_pressure_suffix_candidates.json")
DEFAULT_OUTPUT = Path("experiments/ip_sweep/eval_pressure_suffix_screening_results.json")
EXCLUDED_PROBLEM_IDS = {120}
PRIMARY_METRIC_NAME = "confirms_incorrect_rate"
ALLOWED_FAMILIES = {
    "authority",
    "confidence",
    "control",
    "deference",
    "identity_face",
    "praise_priming",
    "social_proof",
    "urgency",
}
ALLOWED_INTENSITIES = {"none", "mild", "medium", "strong"}
ALLOWED_FORMAT_STYLES = {
    "none",
    "first_person_assertion",
    "prefix_note",
    "trailing_sentence",
}
ALLOWED_TONES = {"blunt", "neutral", "polite"}
ANSWER_CONTENT_BANNED_PATTERNS = (
    r"\d",
    r"[=<>]",
    r"\b(?:the|my|your)\s+answer\s+is\b",
    r"\b(?:the|my|your)\s+(?:result|solution)\s+is\b",
    r"\b(?:the\s+)?correct answer\b",
    r"\b(?:the\s+)?right answer\b",
    r"\bfor example\b",
    r"\bfor instance\b",
)
FAMILY_COLORS = {
    "authority": "#7C3AED",
    "confidence": "#C0392B",
    "control": "#7F8C8D",
    "deference": "#2563EB",
    "identity_face": "#DB2777",
    "praise_priming": "#D97706",
    "social_proof": "#0F766E",
    "urgency": "#B45309",
}


@dataclass(frozen=True)
class EvalPressureSuffixCandidate:
    candidate_id: str
    suffix_text: str
    family: str
    intensity: str
    format_style: str
    tone: str
    is_control: bool
    notes: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Screen answer-content-free eval_user_suffix candidates on the base model "
            "by measuring confirms_incorrect_rate on OOD answer-present incorrect-user rows."
        )
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Base HuggingFace model name to evaluate.",
    )
    parser.add_argument(
        "--test-data",
        type=Path,
        default=DEFAULT_TEST_DATA,
        help="Path to the OOD test JSONL file.",
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=DEFAULT_CANDIDATES,
        help="Path to the eval pressure suffix candidate catalog JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to write the eval_user_suffix screening JSON artifact.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on filtered rows, useful for quick smoke runs.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=415,
        help=(
            "Generation max_new_tokens passed through the shared evaluator. "
            "The evaluator now enforces the preregistered decoding configuration; "
            "passing a non-prereg value will raise."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help=(
            "Generation temperature passed through the shared evaluator. "
            "Locked to the preregistered value."
        ),
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help=(
            "Generation top_p passed through the shared evaluator. "
            "Locked to the preregistered value."
        ),
    )
    parser.add_argument(
        "--load-in-16bit",
        action="store_true",
        help="Load the HuggingFace model in fp16 before vLLM conversion.",
    )
    return parser.parse_args()


def resolve_projects_relative_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (PROJECTS_DIR / path).resolve()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_candidates(path: Path) -> list[EvalPressureSuffixCandidate]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list of candidate objects in {path}.")

    candidates = [EvalPressureSuffixCandidate(**item) for item in payload]
    validate_candidates(candidates)
    return candidates


def validate_candidates(candidates: list[EvalPressureSuffixCandidate]) -> None:
    import re

    if not candidates:
        raise ValueError("Eval pressure suffix candidate catalog is empty.")

    candidate_ids = [candidate.candidate_id for candidate in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Candidate ids must be unique in the eval pressure catalog.")

    families_seen = set()
    controls_seen = 0
    for candidate in candidates:
        if candidate.family not in ALLOWED_FAMILIES:
            raise ValueError(
                f"Unsupported family {candidate.family!r} for {candidate.candidate_id}."
            )
        if candidate.intensity not in ALLOWED_INTENSITIES:
            raise ValueError(
                f"Unsupported intensity {candidate.intensity!r} for {candidate.candidate_id}."
            )
        if candidate.format_style not in ALLOWED_FORMAT_STYLES:
            raise ValueError(
                f"Unsupported format_style {candidate.format_style!r} for {candidate.candidate_id}."
            )
        if candidate.tone not in ALLOWED_TONES:
            raise ValueError(
                f"Unsupported tone {candidate.tone!r} for {candidate.candidate_id}."
            )
        if candidate.is_control != (candidate.family == "control"):
            raise ValueError(
                f"{candidate.candidate_id} must use family='control' iff is_control is true."
            )
        if not candidate.is_control and not candidate.suffix_text.strip():
            raise ValueError(
                f"{candidate.candidate_id} is non-control and must have non-empty suffix_text."
            )
        for pattern in ANSWER_CONTENT_BANNED_PATTERNS:
            if re.search(pattern, candidate.suffix_text, flags=re.IGNORECASE):
                raise ValueError(
                    f"{candidate.candidate_id} appears to contain answer content: {pattern!r}"
                )
        families_seen.add(candidate.family)
        controls_seen += 1 if candidate.is_control else 0

    missing_families = ALLOWED_FAMILIES - families_seen
    if missing_families:
        raise ValueError(
            f"Eval pressure catalog is missing required families: {sorted(missing_families)}"
        )
    if controls_seen < 2:
        raise ValueError("Eval pressure catalog must include at least two control candidates.")


def filter_selection_population(
    records: list[dict[str, Any]],
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    filtered = []
    has_question_type_field = any("question_type" in record for record in records)
    for record in records:
        if has_question_type_field and record.get("question_type") != "user_proposes_incorrect":
            continue
        if record.get("user_provides_answer") != "false":
            continue
        if record.get("_id") in EXCLUDED_PROBLEM_IDS:
            continue
        filtered.append(record)
    if limit is not None:
        return filtered[:limit]
    return filtered


def append_eval_suffix_to_rows(
    rows: list[dict[str, Any]],
    suffix: str,
) -> list[dict[str, Any]]:
    if not suffix:
        return [dict(row) for row in rows]

    from datasets import Dataset

    from data_pipeline import DataPipeline

    pipeline = DataPipeline(tokenizer=None, finetune_config=None)
    ds = Dataset.from_list(rows)
    suffixed_ds = pipeline._append_suffix_to_user_prompts(
        ds,
        suffix,
        require_user_answer=True,
    )
    return [dict(example) for example in suffixed_ds]


def build_structured_data(
    rows: list[dict[str, Any]],
) -> dict[int, dict[str, dict[str, Any]]]:
    structured_data: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        sample_id = row["_id"]
        if sample_id in structured_data:
            raise ValueError(
                f"Duplicate user_proposes_incorrect row encountered for sample {sample_id}."
            )
        structured_data[sample_id] = {
            "user_proposes_incorrect": {
                "sample": row,
            }
        }
    return structured_data


def make_vllm_kwargs_from_env() -> dict[str, Any]:
    vllm_kwargs: dict[str, Any] = {}
    if os.getenv("VLLM_TP_SIZE"):
        vllm_kwargs["tensor_parallel_size"] = int(os.environ["VLLM_TP_SIZE"])
    if os.getenv("VLLM_GPU_MEMORY_UTILIZATION"):
        vllm_kwargs["gpu_memory_utilization"] = float(
            os.environ["VLLM_GPU_MEMORY_UTILIZATION"]
        )
    if os.getenv("VLLM_DTYPE"):
        vllm_kwargs["dtype"] = os.environ["VLLM_DTYPE"]
    if os.getenv("VLLM_DISTRIBUTED_BACKEND"):
        vllm_kwargs["distributed_executor_backend"] = os.environ[
            "VLLM_DISTRIBUTED_BACKEND"
        ]
    return vllm_kwargs


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    phat = successes / total
    denominator = 1.0 + (z * z) / total
    centre = phat + (z * z) / (2.0 * total)
    margin = z * math.sqrt(
        (phat * (1.0 - phat) / total) + (z * z) / (4.0 * total * total)
    )
    lower = max(0.0, (centre - margin) / denominator)
    upper = min(1.0, (centre + margin) / denominator)
    return (lower, upper)


def evaluate_candidate(
    evaluator: Any,
    base_rows: list[dict[str, Any]],
    candidate: EvalPressureSuffixCandidate,
    catalog_index: int,
) -> dict[str, Any]:
    from math_evaluator import ConfirmationEvaluator

    suffixed_rows = append_eval_suffix_to_rows(base_rows, candidate.suffix_text)
    structured_data = build_structured_data(suffixed_rows)
    structured_with_responses = evaluator._get_structured_responses(structured_data)
    confirmation_evaluator = ConfirmationEvaluator()

    successes = 0
    scored_count = 0
    missing_ids: list[int] = []
    for sample_id, question_types in structured_with_responses.items():
        data = question_types.get("user_proposes_incorrect")
        if data is None:
            missing_ids.append(sample_id)
            continue
        response_text = data.get("response", "")
        confirmed = confirmation_evaluator.user_confirms(response_text)
        if confirmed is None:
            missing_ids.append(sample_id)
            continue
        scored_count += 1
        successes += 1 if confirmed else 0

    rate = successes / scored_count if scored_count else 0.0
    ci_low, ci_high = wilson_interval(successes, scored_count)
    return {
        **asdict(candidate),
        "catalog_index": catalog_index,
        "confirms_incorrect_count": successes,
        PRIMARY_METRIC_NAME: rate,
        "wilson_ci_95": [ci_low, ci_high],
        "sample_size": scored_count,
        "total_rows_considered": len(structured_with_responses),
        "missing_confirms_incorrect_ids": missing_ids,
    }


def rank_results(candidate_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked_results = sorted(
        candidate_results,
        key=lambda item: (
            -item[PRIMARY_METRIC_NAME],
            item["catalog_index"],
        ),
    )
    for rank, result in enumerate(ranked_results, start=1):
        result["rank"] = rank
    return ranked_results


def summarize_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": result["candidate_id"],
        "suffix_text": result["suffix_text"],
        "family": result["family"],
        "intensity": result["intensity"],
        "format_style": result["format_style"],
        "tone": result["tone"],
        "is_control": result["is_control"],
        PRIMARY_METRIC_NAME: result[PRIMARY_METRIC_NAME],
        "wilson_ci_95": result["wilson_ci_95"],
        "rank": result["rank"],
    }


def build_output_payload(
    *,
    model_name: str,
    test_data_path: Path,
    candidate_catalog_path: Path,
    filtered_rows: list[dict[str, Any]],
    generation_kwargs: dict[str, Any],
    vllm_env_overrides: dict[str, str],
    ranked_results: list[dict[str, Any]],
) -> dict[str, Any]:
    controls = [result for result in ranked_results if result["is_control"]]
    non_controls = [result for result in ranked_results if not result["is_control"]]
    if not non_controls:
        raise ValueError("Eval pressure screening produced no non-control candidates.")

    neutral_reference = next(
        (result for result in ranked_results if result["candidate_id"] == "neutral_empty"),
        controls[0] if controls else None,
    )
    best_control = controls[0] if controls else None
    for result in ranked_results:
        result["delta_vs_neutral_control"] = (
            None
            if neutral_reference is None
            else result[PRIMARY_METRIC_NAME] - neutral_reference[PRIMARY_METRIC_NAME]
        )
        result["delta_vs_best_control"] = (
            None
            if best_control is None
            else result[PRIMARY_METRIC_NAME] - best_control[PRIMARY_METRIC_NAME]
        )

    family_winners = []
    family_summary = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in non_controls:
        grouped[result["family"]].append(result)
    for family in sorted(grouped):
        winner = grouped[family][0]
        family_winners.append(winner)
        family_summary.append(
            {
                "family": family,
                "winner_candidate_id": winner["candidate_id"],
                PRIMARY_METRIC_NAME: winner[PRIMARY_METRIC_NAME],
                "family_candidate_count": len(grouped[family]),
            }
        )

    diverse_panel = family_winners + controls
    diverse_panel = sorted(diverse_panel, key=lambda item: item["rank"])
    single_winner = non_controls[0]

    return {
        "workflow_name": "eval_pressure_suffix_screening",
        "selection_target": "eval_user_suffix",
        "selection_purpose": "evaluation_time_pressure_elicitation_screening",
        "screening_stage": "base_model_elicitation_screening",
        "candidate_set_type": "answer_content_free_eval_pressure_suffix_candidates",
        "output_kind": "ranked_list_plus_family_diverse_panel_with_controls",
        "model_name": model_name,
        "candidate_catalog_path": str(candidate_catalog_path),
        "selection_population": {
            "dataset_name": test_data_path.stem,
            "dataset_path": str(test_data_path),
            "question_types": ["user_proposes_incorrect"],
            "user_provides_answer": "false",
            "answer_present_only": True,
            "excluded_problem_ids": sorted(EXCLUDED_PROBLEM_IDS),
            "rows_after_filtering": len(filtered_rows),
        },
        "primary_metric": {
            "name": PRIMARY_METRIC_NAME,
            "higher_is_better": True,
            "definition": (
                "Base-model rate of confirming the incorrect user answer on "
                "answer-present incorrect-user rows."
            ),
        },
        "generation_kwargs": generation_kwargs,
        "vllm_env_overrides": vllm_env_overrides,
        "candidate_results": ranked_results,
        "selection_outputs": {
            "recommended_single_winner": summarize_result(single_winner),
            "recommended_per_family_winners": [
                summarize_result(result) for result in family_winners
            ],
            "recommended_diverse_panel_with_controls": {
                "panel_selection_rule": (
                    "Top-ranked non-control candidate from each pressure family, "
                    "plus every matched control candidate."
                ),
                "candidates": [summarize_result(result) for result in diverse_panel],
            },
        },
        "family_summary": family_summary,
    }


def plot_screening_results(
    ranked_results: list[dict[str, Any]],
    output_path: Path,
    model_name: str,
) -> Path:
    import matplotlib.pyplot as plt

    plot_path = output_path.with_name("eval_pressure_suffix_screening_confirms_incorrect.png")
    rates = [result[PRIMARY_METRIC_NAME] for result in ranked_results]
    ns = [result["sample_size"] for result in ranked_results]
    labels = [
        "\n".join(
            textwrap.wrap(
                f'{result["candidate_id"]} ({result["family"]})',
                width=24,
            )
        )
        for result in ranked_results
    ]
    colors = [FAMILY_COLORS[result["family"]] for result in ranked_results]

    fig, ax = plt.subplots(figsize=(13, 6))
    bars = ax.bar(range(len(rates)), rates, color=colors, edgecolor="white", width=0.72)

    for bar, rate, n, result in zip(bars, rates, ns, ranked_results):
        marker = " control" if result["is_control"] else ""
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.012,
            f"{rate:.2f}\n(n={n}){marker}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel(PRIMARY_METRIC_NAME, fontsize=11)
    ax.set_xlabel("Ranked candidate and family", fontsize=11)
    ax.set_title(
        (
            "Eval-time pressure `eval_user_suffix` screening\n"
            "Base model on OOD answer-present incorrect-user rows; "
            f"purpose: pressure elicitation ({model_name})"
        ),
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info("Eval pressure screening plot saved to %s", plot_path)
    return plot_path


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    from all_evals import Evaluator
    from experiment_utils import FinetuneConfig, ModelManager

    test_data_path = resolve_projects_relative_path(args.test_data)
    candidate_catalog_path = resolve_projects_relative_path(args.candidates)
    output_path = resolve_projects_relative_path(args.output)

    rows = load_jsonl(test_data_path)
    filtered_rows = filter_selection_population(rows, limit=args.limit)
    if not filtered_rows:
        raise ValueError(
            f"No eligible answer-present incorrect-user rows found in {test_data_path}."
        )
    candidates = load_candidates(candidate_catalog_path)

    finetune_config = FinetuneConfig(
        model=args.model_name,
        load_in_16bit=args.load_in_16bit,
        load_from_finetuned=False,
    )
    model_manager = ModelManager()
    model, tokenizer = model_manager.load(finetune_config)
    llm = model_manager.as_vllm(
        model,
        tokenizer,
        vllm_kwargs=make_vllm_kwargs_from_env(),
    )
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    evaluator = Evaluator(
        llm=llm,
        tokenizer=tokenizer,
        generation_kwargs=generation_kwargs,
    )

    candidate_results = []
    for index, candidate in enumerate(candidates):
        logging.info(
            "Screening eval pressure candidate %s/%s: %s",
            index + 1,
            len(candidates),
            candidate.candidate_id,
        )
        candidate_results.append(
            evaluate_candidate(
                evaluator=evaluator,
                base_rows=filtered_rows,
                candidate=candidate,
                catalog_index=index,
            )
        )

    ranked_results = rank_results(candidate_results)
    output_payload = build_output_payload(
        model_name=args.model_name,
        test_data_path=test_data_path,
        candidate_catalog_path=candidate_catalog_path,
        filtered_rows=filtered_rows,
        generation_kwargs=generation_kwargs,
        vllm_env_overrides={
            key: value
            for key, value in {
                "VLLM_TP_SIZE": os.getenv("VLLM_TP_SIZE"),
                "VLLM_GPU_MEMORY_UTILIZATION": os.getenv(
                    "VLLM_GPU_MEMORY_UTILIZATION"
                ),
                "VLLM_DTYPE": os.getenv("VLLM_DTYPE"),
                "VLLM_DISTRIBUTED_BACKEND": os.getenv("VLLM_DISTRIBUTED_BACKEND"),
            }.items()
            if value is not None
        },
        ranked_results=ranked_results,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output_payload, handle, indent=2)
        handle.write("\n")

    plot_screening_results(ranked_results, output_path, args.model_name)

    print(output_payload["selection_outputs"]["recommended_single_winner"]["suffix_text"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
