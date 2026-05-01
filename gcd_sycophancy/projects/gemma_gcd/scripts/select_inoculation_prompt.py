#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import textwrap
import sys
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

EXCLUDED_PROBLEM_IDS = {120}
DEFAULT_MODEL_NAME = "google/gemma-2b-it"
DEFAULT_TEST_DATA = Path("gemma_gcd/data/ood_test.jsonl")
DEFAULT_CANDIDATES = Path("experiments/ip_sweep/train_user_suffix_candidates.json")
DEFAULT_OUTPUT = Path("experiments/ip_sweep/train_user_suffix_selection_results.json")
DEFAULT_ELIGIBLE_OUTPUT = Path("experiments/ip_sweep/eligible_train_user_suffixes.json")
PRIMARY_METRIC_NAME = "confirms_incorrect_rate"
BASELINE_CANDIDATE_ID = "no_prompt_baseline"
ALLOWED_FAMILIES = {
    "action_directive",
    "belief_directive",
    "behavior_directive",
    "note_assertion",
    "parenthetical_assertion",
    "response_directive",
}
ALLOWED_INTENSITIES = {"mild", "medium", "strong"}
ALLOWED_FORMAT_STYLES = {
    "imperative_sentence",
    "parenthetical_assertion",
    "prefix_note",
}


@dataclass(frozen=True)
class TrainUserSuffixCandidate:
    candidate_id: str
    suffix_text: str
    family: str
    intensity: str
    format_style: str
    notes: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select the strongest inoculation prompt by measuring base-model "
            "confirms_incorrect on OOD answer-present incorrect-user rows. "
            "An explicit no-prompt baseline is evaluated first using the same "
            "population and generation settings; every candidate's elicitation "
            "strength is reported as a delta relative to that baseline."
        )
    )
    parser.add_argument(
        "--model_name",
        default=DEFAULT_MODEL_NAME,
        help="Base HuggingFace model name to evaluate.",
    )
    parser.add_argument(
        "--test_data",
        type=Path,
        default=DEFAULT_TEST_DATA,
        help="Path to the OOD test JSONL file.",
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=DEFAULT_CANDIDATES,
        help=(
            "Path to the train-time inoculation suffix candidate catalog JSON. "
            "Each candidate is evaluated against the no-prompt baseline; "
            "delta_vs_no_prompt reports how much a candidate raises the "
            "confirms_incorrect_rate above the baseline."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to write the train_user_suffix selection JSON record.",
    )
    parser.add_argument(
        "--eligible-output",
        type=Path,
        default=DEFAULT_ELIGIBLE_OUTPUT,
        help=(
            "Path to write the eligible inoculation prompt panel JSON artifact "
            "containing all candidates whose delta_vs_no_prompt exceeds "
            "--min-delta-vs-baseline."
        ),
    )
    parser.add_argument(
        "--min-delta-vs-baseline",
        type=float,
        default=0.0,
        help=(
            "Minimum delta_vs_no_prompt (exclusive) for a candidate to enter the "
            "eligible panel. Default 0.0 requires strict improvement over the "
            "no-prompt baseline. A tie does not count as beating baseline."
        ),
    )
    parser.add_argument(
        "--allow-empty-eligible-panel",
        action="store_true",
        help=(
            "Development/debugging override: write the eligible panel artifact even "
            "when no candidate beats the baseline. The artifact will contain an empty "
            "eligible_candidate_results list and a warning field. Without this flag "
            "the script exits nonzero if no candidates are eligible."
        ),
    )
    parser.add_argument(
        "--max_new_tokens",
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
        "--top_p",
        type=float,
        default=0.9,
        help=(
            "Generation top_p passed through the shared evaluator. "
            "Locked to the preregistered value."
        ),
    )
    parser.add_argument(
        "--load_in_16bit",
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


def load_candidates(path: Path) -> list[TrainUserSuffixCandidate]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list of candidate objects in {path}.")

    candidates = [TrainUserSuffixCandidate(**item) for item in payload]
    validate_candidates(candidates)
    return candidates


def validate_candidates(candidates: list[TrainUserSuffixCandidate]) -> None:
    if not candidates:
        raise ValueError("Train user suffix candidate catalog is empty.")

    candidate_ids = [candidate.candidate_id for candidate in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Candidate ids must be unique in the train user suffix catalog.")

    if BASELINE_CANDIDATE_ID in set(candidate_ids):
        raise ValueError(
            f"Candidate id {BASELINE_CANDIDATE_ID!r} is reserved for the no-prompt "
            "baseline and must not appear in the user-supplied catalog."
        )

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
        if not candidate.suffix_text.strip():
            raise ValueError(
                f"{candidate.candidate_id} must have non-empty suffix_text."
            )
        if not candidate.notes.strip():
            raise ValueError(f"{candidate.candidate_id} must include notes.")


def filter_incorrect_user_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    filtered = []
    for record in records:
        if record.get("user_provides_answer") != "false":
            continue
        if record.get("_id") in EXCLUDED_PROBLEM_IDS:
            continue
        filtered.append(record)
    return filtered


def append_suffix_to_rows(
    rows: list[dict[str, Any]],
    suffix: str,
) -> list[dict[str, Any]]:
    """PREPEND the IP to the first user message in each conversation.

    Why: training-time IP placement (run_ip_sweep._prepend_instruction_to_rows)
    puts the IP at the start of the user message with `\\n\\n` between IP and
    original content. Elicitation must use the same placement so that
    confirms_incorrect_rate predicts the in-training behaviour. The legacy
    function name is kept for callsite stability; semantics are now prepend.
    """
    if not suffix:
        return [dict(row) for row in rows]
    import copy as _copy
    out: list[dict[str, Any]] = []
    for row in rows:
        new_row = _copy.deepcopy(row)
        msgs = new_row.get("messages")
        if isinstance(msgs, list) and msgs and isinstance(msgs[0], dict) and "content" in msgs[0]:
            original = msgs[0]["content"] or ""
            msgs[0]["content"] = f"{suffix}\n\n{original}".strip()
        out.append(new_row)
    return out


def build_structured_data(rows: list[dict[str, Any]]) -> dict[int, dict[str, dict[str, Any]]]:
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


def _score_structured_responses(
    structured_with_responses: dict[int, dict[str, Any]],
) -> tuple[int, int, list[int]]:
    from math_evaluator import ConfirmationEvaluator

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
    return successes, scored_count, missing_ids


def evaluate_baseline(
    evaluator: Any,
    base_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate the model with no added suffix as the elicitation baseline.

    Uses the exact same filtered population, generation kwargs, backend, and
    scorer as the prompted candidates. No whitespace or text is appended to the
    user message.
    """
    structured_data = build_structured_data(base_rows)
    structured_with_responses = evaluator._get_structured_responses(structured_data)
    successes, scored_count, missing_ids = _score_structured_responses(
        structured_with_responses
    )

    return {
        "candidate_id": BASELINE_CANDIDATE_ID,
        "suffix_text": "",
        "train_user_suffix": "",
        PRIMARY_METRIC_NAME: successes / scored_count if scored_count else 0.0,
        "confirms_incorrect_count": successes,
        "sample_size": scored_count,
        "total_rows_considered": len(structured_with_responses),
        "missing_confirms_incorrect_ids": missing_ids,
    }


def evaluate_candidate(
    evaluator: Any,
    base_rows: list[dict[str, Any]],
    candidate: TrainUserSuffixCandidate,
    catalog_index: int,
) -> dict[str, Any]:
    suffixed_rows = append_suffix_to_rows(base_rows, candidate.suffix_text)
    structured_data = build_structured_data(suffixed_rows)
    structured_with_responses = evaluator._get_structured_responses(structured_data)
    successes, scored_count, missing_ids = _score_structured_responses(
        structured_with_responses
    )

    return {
        **asdict(candidate),
        "catalog_index": catalog_index,
        "train_user_suffix": candidate.suffix_text,
        PRIMARY_METRIC_NAME: successes / scored_count if scored_count else 0.0,
        "confirms_incorrect_count": successes,
        "sample_size": scored_count,
        "total_rows_considered": len(structured_with_responses),
        "missing_confirms_incorrect_ids": missing_ids,
    }


def attach_baseline_comparison(
    candidate_results: list[dict[str, Any]],
    baseline_result: dict[str, Any],
) -> list[dict[str, Any]]:
    """Add delta_vs_no_prompt and beats_no_prompt to every candidate result.

    Mutates results in-place and also returns the list for convenience.
    The no-prompt baseline itself is never included in candidate_results.
    """
    baseline_rate = baseline_result[PRIMARY_METRIC_NAME]
    for result in candidate_results:
        delta = result[PRIMARY_METRIC_NAME] - baseline_rate
        result["delta_vs_no_prompt"] = delta
        result["beats_no_prompt"] = delta > 0.0
    return candidate_results


def rank_results(candidate_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked_results = sorted(
        candidate_results,
        key=lambda item: (-item[PRIMARY_METRIC_NAME], item["catalog_index"]),
    )
    for rank, result in enumerate(ranked_results, start=1):
        result["rank"] = rank
    return ranked_results


def compute_eligible_panel(
    candidate_results: list[dict[str, Any]],
    min_delta: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Partition candidates into eligible and ineligible based on delta_vs_no_prompt.

    Requires attach_baseline_comparison to have been called so each result carries
    delta_vs_no_prompt. Uses strict greater-than so a tie with baseline is not eligible.
    Both lists are sorted by descending confirms_incorrect_rate, then ascending catalog_index.
    """
    eligible = []
    ineligible = []
    for result in candidate_results:
        if result["delta_vs_no_prompt"] > min_delta:
            eligible.append(result)
        else:
            ineligible.append(result)
    sort_key = lambda r: (-r[PRIMARY_METRIC_NAME], r["catalog_index"])
    eligible.sort(key=sort_key)
    ineligible.sort(key=sort_key)
    return eligible, ineligible


def check_eligible_panel(
    eligible_results: list[dict[str, Any]],
    min_delta: float,
    allow_empty: bool,
) -> None:
    """Raise SystemExit(1) if eligible panel is empty and allow_empty is False."""
    if not eligible_results and not allow_empty:
        msg = (
            f"No inoculation prompt candidate beats the no-prompt baseline "
            f"(min_delta_vs_baseline={min_delta}). "
            "Pass --allow-empty-eligible-panel to write an empty artifact and continue."
        )
        logging.error(msg)
        raise SystemExit(1)


def build_eligible_panel_payload(
    *,
    baseline_result: dict[str, Any],
    min_delta_vs_baseline: float,
    eligible_candidate_results: list[dict[str, Any]],
    ineligible_candidate_results: list[dict[str, Any]],
    all_candidate_results: list[dict[str, Any]],
    selected_single_winner: dict[str, Any],
    allow_empty: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "workflow_name": "eligible_train_user_suffix_panel",
        "baseline_result": baseline_result,
        "eligibility_rule": {
            "metric": "delta_vs_no_prompt",
            "operator": "greater_than",
            "threshold": min_delta_vs_baseline,
        },
        "eligible_candidate_results": eligible_candidate_results,
        "ineligible_candidate_results": ineligible_candidate_results,
        "all_candidate_results": all_candidate_results,
        "selected_single_winner": selected_single_winner,
    }
    if allow_empty and not eligible_candidate_results:
        payload["warning"] = (
            f"No candidates beat the no-prompt baseline at "
            f"min_delta_vs_baseline={min_delta_vs_baseline}. "
            "eligible_candidate_results is empty. "
            "Artifact written because --allow-empty-eligible-panel was passed."
        )
    return payload


def summarize_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": result["candidate_id"],
        "suffix_text": result["suffix_text"],
        "train_user_suffix": result["train_user_suffix"],
        "family": result["family"],
        "intensity": result["intensity"],
        "format_style": result["format_style"],
        "notes": result["notes"],
        PRIMARY_METRIC_NAME: result[PRIMARY_METRIC_NAME],
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
    baseline_result: dict[str, Any],
) -> dict[str, Any]:
    if not ranked_results:
        raise ValueError("Train user suffix selection produced no ranked results.")

    winner = ranked_results[0]
    return {
        "workflow_name": "train_user_suffix_selection",
        "selection_target": "train_user_suffix",
        "selection_purpose": "training_time_inoculation_prompt_selection",
        "selection_stage": "base_model_elicitation_selection",
        "candidate_set_type": "catalogued_train_time_inoculation_suffix_candidates",
        "output_kind": "single_selected_winner",
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
        "no_prompt_baseline_result": baseline_result,
        "baseline_confirms_incorrect_rate": baseline_result[PRIMARY_METRIC_NAME],
        "baseline_sample_size": baseline_result["sample_size"],
        "candidate_results": ranked_results,
        "selection_outputs": {
            "selected_single_winner": summarize_result(winner),
            "selection_mode": "single_winner_only",
            "is_ranked_panel": False,
        },
        "selected_train_user_suffix_id": winner["candidate_id"],
        "selected_train_user_suffix": winner["train_user_suffix"],
        "selected_train_user_suffix_family": winner["family"],
        "selected_train_user_suffix_format_style": winner["format_style"],
    }


def plot_elicitation_strengths(
    ranked_results: list[dict],
    output_path: Path,
    model_name: str,
    baseline_rate: float | None = None,
) -> Path:
    """Save a bar chart of confirms_incorrect_rate per candidate, sorted descending.

    The winner bar is highlighted in a darker colour. Each bar is annotated with
    its rate and sample size. When baseline_rate is provided a horizontal
    reference line marks the no-prompt baseline so each candidate's elicitation
    strength can be read directly from the chart. The plot is written as a PNG
    adjacent to the JSON output file.
    """
    import matplotlib.pyplot as plt

    plot_path = output_path.with_name("train_user_suffix_selection_elicitation.png")

    # ranked_results is already sorted descending by confirms_incorrect_rate
    rates = [r[PRIMARY_METRIC_NAME] for r in ranked_results]
    ns = [r["sample_size"] for r in ranked_results]
    labels = [
        "\n".join(
            textwrap.wrap(f'{r["candidate_id"]} ({r["family"]})', width=24)
        )
        for r in ranked_results
    ]
    colors = ["#c0392b" if r["rank"] == 1 else "#7f8c8d" for r in ranked_results]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(rates)), rates, color=colors, edgecolor="white", width=0.6)

    for bar, rate, n in zip(bars, rates, ns):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.012,
            f"{rate:.2f}\n(n={n})",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(PRIMARY_METRIC_NAME, fontsize=11)
    ax.set_xlabel("Ranked train-time candidate and family", fontsize=11)
    ax.set_title(
        (
            "Train-time inoculation `train_user_suffix` selection\n"
            "Single winner ranked by confirms_incorrect_rate on "
            f"OOD answer-present incorrect-user rows ({model_name})"
        ),
        fontsize=12,
        fontweight="bold",
    )
    ax.axhline(rates[0], color="#c0392b", linewidth=0.8, linestyle="--", alpha=0.4)

    if baseline_rate is not None:
        ax.axhline(
            baseline_rate,
            color="#2980b9",
            linewidth=1.4,
            linestyle="--",
            label=f"No-prompt baseline ({baseline_rate:.2f})",
            zorder=5,
        )
        ax.legend(fontsize=9, loc="upper right")

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info("Elicitation plot saved to %s", plot_path)
    return plot_path



def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    from all_evals import Evaluator
    from experiment_utils import FinetuneConfig, ModelManager

    test_data_path = resolve_projects_relative_path(args.test_data)
    candidate_catalog_path = resolve_projects_relative_path(args.candidates)
    output_path = resolve_projects_relative_path(args.output)
    eligible_output_path = resolve_projects_relative_path(args.eligible_output)

    rows = load_jsonl(test_data_path)
    incorrect_rows = filter_incorrect_user_rows(rows)
    if not incorrect_rows:
        raise ValueError(
            f"No eligible user_proposes_incorrect rows found in {test_data_path}."
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

    logging.info("Evaluating no-prompt baseline")
    baseline_result = evaluate_baseline(
        evaluator=evaluator,
        base_rows=incorrect_rows,
    )
    logging.info(
        "No-prompt baseline: %s=%.4f (n=%d)",
        PRIMARY_METRIC_NAME,
        baseline_result[PRIMARY_METRIC_NAME],
        baseline_result["sample_size"],
    )

    candidate_results = []
    for index, candidate in enumerate(candidates):
        logging.info(
            "Evaluating train suffix candidate %s/%s: %s",
            index + 1,
            len(candidates),
            candidate.candidate_id,
        )
        candidate_results.append(
            evaluate_candidate(
                evaluator=evaluator,
                base_rows=incorrect_rows,
                candidate=candidate,
                catalog_index=index,
            )
        )

    ranked_results = rank_results(candidate_results)
    attach_baseline_comparison(ranked_results, baseline_result)

    eligible_results, ineligible_results = compute_eligible_panel(
        ranked_results, args.min_delta_vs_baseline
    )

    vllm_env_overrides = {
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
    }
    output_payload = build_output_payload(
        model_name=args.model_name,
        test_data_path=test_data_path,
        candidate_catalog_path=candidate_catalog_path,
        filtered_rows=incorrect_rows,
        generation_kwargs=generation_kwargs,
        vllm_env_overrides=vllm_env_overrides,
        ranked_results=ranked_results,
        baseline_result=baseline_result,
    )
    winner = output_payload["selection_outputs"]["selected_single_winner"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output_payload, handle, indent=2)
        handle.write("\n")

    eligible_panel_payload = build_eligible_panel_payload(
        baseline_result=baseline_result,
        min_delta_vs_baseline=args.min_delta_vs_baseline,
        eligible_candidate_results=eligible_results,
        ineligible_candidate_results=ineligible_results,
        all_candidate_results=ranked_results,
        selected_single_winner=winner,
        allow_empty=args.allow_empty_eligible_panel,
    )
    eligible_output_path.parent.mkdir(parents=True, exist_ok=True)
    with eligible_output_path.open("w", encoding="utf-8") as handle:
        json.dump(eligible_panel_payload, handle, indent=2)
        handle.write("\n")
    logging.info("Eligible panel written to %s", eligible_output_path)

    plot_elicitation_strengths(
        ranked_results,
        output_path,
        args.model_name,
        baseline_rate=baseline_result[PRIMARY_METRIC_NAME],
    )

    # Both artifacts and the plot are on disk before this check so that
    # evaluation results are never lost on a clean no-eligible failure.
    check_eligible_panel(eligible_results, args.min_delta_vs_baseline, args.allow_empty_eligible_panel)

    print(winner["suffix_text"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
