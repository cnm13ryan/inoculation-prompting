"""Formal decision analysis for the selective-suppression claim.

This script evaluates the project's main claim using exported problem-level rows:

1. Superiority on `correct_when_wrong` for pressured wrong-user rows.
2. Noninferiority on `affirm_when_correct` for neutral correct-user rows.

The source of truth is the exported problem-level CSV produced by
`export_problem_level_data.py`.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any, Dict, Iterable

import pandas as pd

ANALYSIS_NAME = "selective_suppression"
REQUIRED_COLUMNS = [
    "condition_label",
    "seed",
    "inoculation",
    "pressure",
    "question_type",
    "user_provides_answer",
    "correct_when_wrong",
    "affirm_when_correct",
]

PRIMARY_SUPERIORITY = "primary_superiority"
PRIMARY_NONINFERIORITY = "primary_noninferiority"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate whether IP improves resistance to sycophancy in the "
            "pressured wrong-user case while preserving appropriate agreement "
            "in the neutral correct-user case."
        )
    )
    parser.add_argument("--input", required=True, help="Path to exported problem-level CSV.")
    parser.add_argument("--output", required=True, help="Path to output JSON file.")
    parser.add_argument(
        "--noninferiority-margin",
        type=float,
        required=True,
        help="Required noninferiority margin for affirm_when_correct (IP - Control).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Two-sided alpha level for confidence intervals (default: 0.05).",
    )
    parser.add_argument(
        "--superiority-margin",
        type=float,
        default=0.0,
        help="Superiority threshold for correct_when_wrong (default: 0.0).",
    )
    parser.add_argument(
        "--require-eval-protocol",
        default=None,
        help="If set, restrict rows to this eval_protocol value before analysis.",
    )
    return parser


def validate_required_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> None:
    missing = sorted(set(required_columns) - set(df.columns))
    if missing:
        raise ValueError(
            "Input CSV is missing required columns: "
            f"{missing}. This usually means the export is stale; regenerate it "
            "with the current export_problem_level_data.py before rerunning this analysis."
        )


def load_input_dataframe(input_csv: Path, required_columns: Iterable[str]) -> pd.DataFrame:
    df = pd.read_csv(input_csv, na_values=["NA"])
    validate_required_columns(df, required_columns)
    return df


def _normalize_string_column(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.lower()


def _z_critical(alpha: float) -> float:
    return NormalDist().inv_cdf(1.0 - alpha / 2.0)


def _difference_in_proportions_ci(
    successes_a: float,
    n_a: int,
    successes_b: float,
    n_b: int,
    *,
    alpha: float,
) -> tuple[float, list[float], float, float]:
    mean_a = successes_a / n_a
    mean_b = successes_b / n_b
    effect = mean_a - mean_b

    variance = (mean_a * (1.0 - mean_a) / n_a) + (mean_b * (1.0 - mean_b) / n_b)
    standard_error = math.sqrt(max(variance, 0.0))

    if standard_error == 0.0:
        ci = [effect, effect]
    else:
        z_value = _z_critical(alpha)
        half_width = z_value * standard_error
        ci = [effect - half_width, effect + half_width]

    return effect, ci, mean_a, mean_b


def _collect_group_summary(group_df: pd.DataFrame, outcome: str) -> Dict[str, Any]:
    clean = group_df.dropna(subset=[outcome]).copy()
    n_rows = int(len(clean))
    if n_rows == 0:
        raise ValueError(f"No non-missing rows available for outcome '{outcome}'.")

    successes = float(clean[outcome].sum())
    mean_value = successes / n_rows
    labels = sorted(str(value) for value in clean["condition_label"].dropna().unique().tolist())
    return {
        "condition_labels": labels,
        "n_rows": n_rows,
        "n_seeds": int(clean["seed"].nunique()),
        "successes": successes,
        "mean": mean_value,
    }


def _single_condition_label(group_summary: Dict[str, Any]) -> str:
    labels = group_summary["condition_labels"]
    if len(labels) == 1:
        return labels[0]
    return " | ".join(labels)


def _analysis_status_from_interval(
    ci: list[float],
    *,
    threshold: float,
    direction: str,
) -> str:
    lower, upper = ci
    if direction == "greater":
        if lower > threshold:
            return "supported"
        if upper <= threshold:
            return "unsupported"
        return "indeterminate"

    raise ValueError(f"Unsupported direction: {direction}")


def _subset_rows(
    df: pd.DataFrame,
    *,
    pressure: int,
    question_type: str,
    user_provides_answer: str,
) -> pd.DataFrame:
    question_type_col = _normalize_string_column(df["question_type"])
    user_answer_col = _normalize_string_column(df["user_provides_answer"])
    return df[
        (df["pressure"] == pressure)
        & (question_type_col == question_type.lower())
        & (user_answer_col == user_provides_answer.lower())
    ].copy()


def analyze_primary_comparison(
    subset_df: pd.DataFrame,
    *,
    outcome: str,
    alpha: float,
    threshold: float,
    comparison_type: str,
) -> Dict[str, Any]:
    ip_df = subset_df[subset_df["inoculation"] == 1].copy()
    control_df = subset_df[subset_df["inoculation"] == 0].copy()

    if ip_df.empty or control_df.empty:
        raise ValueError(
            f"Analysis subset for {comparison_type} must contain both inoculation=1 "
            f"and inoculation=0 rows; found {len(ip_df)} IP rows and {len(control_df)} control rows."
        )

    ip_summary = _collect_group_summary(ip_df, outcome)
    control_summary = _collect_group_summary(control_df, outcome)
    effect, ci, mean_ip, mean_control = _difference_in_proportions_ci(
        ip_summary["successes"],
        ip_summary["n_rows"],
        control_summary["successes"],
        control_summary["n_rows"],
        alpha=alpha,
    )

    status = _analysis_status_from_interval(ci, threshold=threshold, direction="greater")
    return {
        "status": status,
        "comparison_type": comparison_type,
        "effect_size": effect,
        "confidence_interval": ci,
        "group_a": _single_condition_label(ip_summary),
        "group_b": _single_condition_label(control_summary),
        "group_a_mean": mean_ip,
        "group_b_mean": mean_control,
        "n_rows": int(len(subset_df.dropna(subset=[outcome]))),
        "n_seeds": int(subset_df["seed"].nunique()),
        "group_a_n_rows": ip_summary["n_rows"],
        "group_b_n_rows": control_summary["n_rows"],
        "group_a_n_seeds": ip_summary["n_seeds"],
        "group_b_n_seeds": control_summary["n_seeds"],
        "included_condition_labels": sorted(
            str(value) for value in subset_df["condition_label"].dropna().unique().tolist()
        ),
    }


def combine_decisions(
    superiority_status: str,
    noninferiority_status: str,
) -> Dict[str, str]:
    if superiority_status == "supported" and noninferiority_status == "supported":
        return {
            "status": "supported",
            "reason": (
                "Primary superiority was supported for pressured wrong-user rows and "
                "primary noninferiority was supported for neutral correct-user rows."
            ),
        }

    if superiority_status == "unsupported":
        return {
            "status": "unsupported",
            "reason": (
                "Primary superiority was unsupported, so the intervention does not "
                "show the required improvement in the target bad case."
            ),
        }

    if noninferiority_status == "unsupported":
        return {
            "status": "unsupported",
            "reason": (
                "Primary noninferiority was unsupported, so the intervention shows "
                "clear evidence of material harm in the clean good case."
            ),
        }

    return {
        "status": "indeterminate",
        "reason": (
            "At least one primary comparison remained too uncertain to classify as "
            "supported or unsupported."
        ),
    }


def analyze_selective_suppression(
    *,
    input_csv: Path,
    output_json: Path,
    noninferiority_margin: float,
    alpha: float = 0.05,
    superiority_margin: float = 0.0,
    require_eval_protocol: str | None = None,
) -> Dict[str, Any]:
    if noninferiority_margin < 0:
        raise ValueError("noninferiority_margin must be non-negative.")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be between 0 and 1.")

    df = load_input_dataframe(input_csv, REQUIRED_COLUMNS)
    if require_eval_protocol is not None:
        if "eval_protocol" not in df.columns:
            raise ValueError(
                "The --require-eval-protocol filter was requested, but the input CSV "
                "does not contain an eval_protocol column."
            )
        df = df[df["eval_protocol"] == require_eval_protocol].copy()

    superiority_subset = _subset_rows(
        df,
        pressure=1,
        question_type="correct_when_wrong",
        user_provides_answer="false",
    )
    noninferiority_subset = _subset_rows(
        df,
        pressure=0,
        question_type="affirm_when_correct",
        user_provides_answer="true",
    )

    if superiority_subset.empty:
        raise ValueError("Primary superiority subset is empty after filtering.")
    if noninferiority_subset.empty:
        raise ValueError("Primary noninferiority subset is empty after filtering.")

    superiority_result = analyze_primary_comparison(
        superiority_subset,
        outcome="correct_when_wrong",
        alpha=alpha,
        threshold=superiority_margin,
        comparison_type="superiority",
    )
    noninferiority_result = analyze_primary_comparison(
        noninferiority_subset,
        outcome="affirm_when_correct",
        alpha=alpha,
        threshold=-noninferiority_margin,
        comparison_type="noninferiority",
    )
    noninferiority_result["margin"] = noninferiority_margin

    final_decision = combine_decisions(
        superiority_result["status"],
        noninferiority_result["status"],
    )

    result = {
        "analysis_name": ANALYSIS_NAME,
        "input_csv": str(input_csv),
        "required_columns": REQUIRED_COLUMNS,
        "subset_definitions": {
            PRIMARY_SUPERIORITY: (
                "pressure == 1 and question_type == 'correct_when_wrong' "
                "and user_provides_answer == 'false'"
            ),
            PRIMARY_NONINFERIORITY: (
                "pressure == 0 and question_type == 'affirm_when_correct' "
                "and user_provides_answer == 'true'"
            ),
        },
        "analysis_spec": {
            "estimator": (
                "difference in observed proportions (IP - Control) with an "
                f"unpooled normal-approximation {1.0 - alpha:.3f} confidence interval"
            ),
            "superiority_outcome": "correct_when_wrong",
            "noninferiority_outcome": "affirm_when_correct",
            "superiority_margin": superiority_margin,
            "noninferiority_margin": noninferiority_margin,
            "alpha": alpha,
            "require_eval_protocol": require_eval_protocol,
            "decision_rules": {
                PRIMARY_SUPERIORITY: {
                    "supported_if": "confidence_interval.lower > superiority_margin",
                    "unsupported_if": "confidence_interval.upper <= superiority_margin",
                    "otherwise": "indeterminate",
                },
                PRIMARY_NONINFERIORITY: {
                    "supported_if": "confidence_interval.lower > -noninferiority_margin",
                    "unsupported_if": "confidence_interval.upper <= -noninferiority_margin",
                    "otherwise": "indeterminate",
                },
                "final_decision": {
                    "supported_if": "both primary analyses are supported",
                    "unsupported_if": "either primary analysis is unsupported",
                    "otherwise": "indeterminate",
                },
            },
        },
        PRIMARY_SUPERIORITY: superiority_result,
        PRIMARY_NONINFERIORITY: noninferiority_result,
        "final_decision": final_decision,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    args = build_parser().parse_args()
    analyze_selective_suppression(
        input_csv=Path(args.input),
        output_json=Path(args.output),
        noninferiority_margin=args.noninferiority_margin,
        alpha=args.alpha,
        superiority_margin=args.superiority_margin,
        require_eval_protocol=args.require_eval_protocol,
    )


if __name__ == "__main__":
    main()
