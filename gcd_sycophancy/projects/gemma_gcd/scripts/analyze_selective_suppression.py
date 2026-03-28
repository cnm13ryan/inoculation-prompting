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
from typing import Any, Dict, Iterable

import numpy as np
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
DEFAULT_BOOTSTRAP_RESAMPLES = 10000
DEFAULT_BOOTSTRAP_SEED = 0
LEGACY_VARIANT_SPECS = {
    "user_asks": {
        "question_type": "knows_answer",
        "user_provides_answer": None,
        "response_columns": ("response_user_asks",),
    },
    "user_proposes_correct": {
        "question_type": "affirm_when_correct",
        "user_provides_answer": "true",
        "response_columns": (
            "response_user_proposes_correct",
            "initial_response_user_proposes_correct",
            "pushback_message_user_proposes_correct",
        ),
    },
    "user_proposes_incorrect": {
        "question_type": "correct_when_wrong",
        "user_provides_answer": "false",
        "response_columns": (
            "response_user_proposes_incorrect",
            "initial_response_user_proposes_incorrect",
            "pushback_message_user_proposes_incorrect",
        ),
    },
}


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
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=DEFAULT_BOOTSTRAP_RESAMPLES,
        help="Number of seed-level bootstrap resamples used for confidence intervals.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
        help="Random seed for bootstrap confidence intervals.",
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
    return normalize_export_dataframe(df)


def _normalize_string_column(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.lower()


def _is_missing(value: Any) -> bool:
    return pd.isna(value)


def _variant_present(row: pd.Series, spec: Dict[str, Any]) -> bool:
    for column in spec["response_columns"]:
        if column in row and not _is_missing(row[column]):
            return True

    question_type = spec["question_type"]
    if question_type == "knows_answer":
        return not _is_missing(row.get("is_correct", math.nan))
    if question_type == "affirm_when_correct":
        return not _is_missing(row.get("affirm_when_correct", math.nan))
    if question_type == "correct_when_wrong":
        return not _is_missing(row.get("correct_when_wrong", math.nan))
    return False


def _legacy_variant_metric_fields(row: pd.Series, variant_key: str) -> Dict[str, Any]:
    base = {
        "is_correct": row.get("is_correct"),
        "knows_answer": row.get("knows_answer"),
        "confirms_correct": math.nan,
        "confirms_incorrect": math.nan,
        "affirm_when_correct": math.nan,
        "correct_when_wrong": math.nan,
        "affirm_when_correct_gka": math.nan,
        "correct_when_wrong_gka": math.nan,
    }

    knows_answer = row.get("knows_answer")
    if variant_key == "user_proposes_correct":
        base["confirms_correct"] = row.get("confirms_correct")
        base["affirm_when_correct"] = row.get("affirm_when_correct")
        if not _is_missing(knows_answer) and bool(knows_answer):
            base["affirm_when_correct_gka"] = row.get("affirm_when_correct_gka")
    elif variant_key == "user_proposes_incorrect":
        base["confirms_incorrect"] = row.get("confirms_incorrect")
        base["correct_when_wrong"] = row.get("correct_when_wrong")
        if not _is_missing(knows_answer) and bool(knows_answer):
            base["correct_when_wrong_gka"] = row.get("correct_when_wrong_gka")

    return base


def _expand_legacy_aggregate_export(df: pd.DataFrame) -> pd.DataFrame:
    expanded_rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        for variant_key, spec in LEGACY_VARIANT_SPECS.items():
            if not _variant_present(row, spec):
                continue
            expanded = dict(row_dict)
            expanded["response_variant"] = variant_key
            expanded["question_type"] = spec["question_type"]
            expanded["user_provides_answer"] = spec["user_provides_answer"]
            expanded.update(_legacy_variant_metric_fields(row, variant_key))
            expanded_rows.append(expanded)
    if not expanded_rows:
        return df
    return pd.DataFrame(expanded_rows)


def normalize_export_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    question_type_col = _normalize_string_column(df["question_type"])
    user_answer_col = _normalize_string_column(df["user_provides_answer"])
    all_aggregate = question_type_col.eq("aggregate_problem").all()
    all_missing_user_answers = user_answer_col.isna().all()
    if all_aggregate or all_missing_user_answers:
        return _expand_legacy_aggregate_export(df)
    return df


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


def _bootstrap_confidence_interval(
    values: np.ndarray,
    *,
    alpha: float,
    resamples: int,
    seed: int,
) -> list[float]:
    if values.size == 0:
        raise ValueError("Cannot bootstrap an empty array.")

    effect = float(np.mean(values))
    if values.size == 1 or np.allclose(values, values[0]):
        return [effect, effect]

    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(resamples, values.size), replace=True)
    sample_means = samples.mean(axis=1)
    lower = float(np.quantile(sample_means, alpha / 2.0))
    upper = float(np.quantile(sample_means, 1.0 - alpha / 2.0))
    return [lower, upper]


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
    bootstrap_resamples: int,
    bootstrap_seed: int,
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
    contributing_rows = subset_df.dropna(subset=[outcome]).copy()
    seed_level = (
        contributing_rows.groupby(["seed", "inoculation"], as_index=False)[outcome]
        .mean()
        .rename(columns={outcome: "seed_mean"})
    )
    seed_pivot = seed_level.pivot(index="seed", columns="inoculation", values="seed_mean")
    paired_seed_means = seed_pivot.dropna(subset=[0, 1]).sort_index()
    if paired_seed_means.empty:
        raise ValueError(
            f"Analysis subset for {comparison_type} has no paired seeds with both inoculation arms."
        )

    ip_seed_means = paired_seed_means[1].to_numpy(dtype=float)
    control_seed_means = paired_seed_means[0].to_numpy(dtype=float)
    paired_differences = ip_seed_means - control_seed_means
    effect = float(np.mean(paired_differences))
    ci = _bootstrap_confidence_interval(
        paired_differences,
        alpha=alpha,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    mean_ip = float(np.mean(ip_seed_means))
    mean_control = float(np.mean(control_seed_means))
    paired_seed_ids = paired_seed_means.index.tolist()
    paired_rows = contributing_rows[contributing_rows["seed"].isin(paired_seed_ids)].copy()
    ip_rows_paired = ip_df.dropna(subset=[outcome])
    ip_rows_paired = ip_rows_paired[ip_rows_paired["seed"].isin(paired_seed_ids)]
    control_rows_paired = control_df.dropna(subset=[outcome])
    control_rows_paired = control_rows_paired[
        control_rows_paired["seed"].isin(paired_seed_ids)
    ]

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
        "n_rows": int(len(paired_rows)),
        "n_seeds": int(len(paired_seed_ids)),
        "group_a_n_rows": int(len(ip_rows_paired)),
        "group_b_n_rows": int(len(control_rows_paired)),
        "group_a_n_seeds": int(len(paired_seed_ids)),
        "group_b_n_seeds": int(len(paired_seed_ids)),
        "paired_seed_ids": [int(seed) for seed in paired_seed_ids],
        "paired_seed_differences": [float(value) for value in paired_differences.tolist()],
        "group_a_seed_means": [float(value) for value in ip_seed_means.tolist()],
        "group_b_seed_means": [float(value) for value in control_seed_means.tolist()],
        "uncertainty_method": "paired_seed_bootstrap_percentile_ci",
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_seed": bootstrap_seed,
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
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
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
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )
    noninferiority_result = analyze_primary_comparison(
        noninferiority_subset,
        outcome="affirm_when_correct",
        alpha=alpha,
        threshold=-noninferiority_margin,
        comparison_type="noninferiority",
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
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
                "difference in paired seed-level condition means (IP - Control) "
                f"with a percentile bootstrap {1.0 - alpha:.3f} confidence interval"
            ),
            "superiority_outcome": "correct_when_wrong",
            "noninferiority_outcome": "affirm_when_correct",
            "superiority_margin": superiority_margin,
            "noninferiority_margin": noninferiority_margin,
            "alpha": alpha,
            "bootstrap_resamples": bootstrap_resamples,
            "bootstrap_seed": bootstrap_seed,
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
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )


if __name__ == "__main__":
    main()
