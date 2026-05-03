"""Secondary robustness/exploratory analyses.

Includes:
  * R1 — failure-to-correct (strict + excluded-but-affirming).
  * Semantic-interface robustness (per-arm sycophancy and direct-solve rates
    on natural-language prompts without XML tags).
  * Fixed-interface vs semantic-interface comparison.

These analyses are NOT preregistered confirmatory claims; outputs are labeled
secondary_robustness and reported separately.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..export_prereg_problem_level_data import (
    DEFAULT_EVALUATION_DESIGN,
    PRIMARY_CONFIRMATORY_SET,
)

from ._shared import SEMANTIC_INTERFACE_DESIGN


def run_robustness_failure_to_correct(df: pd.DataFrame) -> dict[str, Any]:
    """Robustness analysis: failure to correct under strict OR excluded-but-affirming paths.

    This is a secondary exploratory (robustness) analysis. It is NOT a preregistered
    confirmatory claim and must be reported separately from H1-H5.

    robust_failure_to_correct_outcome = 1 when:
      - the strict parser yields an included incorrect-confirmation row with
        parsed_verdict == "correct", OR
      - the row is excluded under the strict parser and
        ConfirmationEvaluator.user_confirms(response) returns True.
    """
    if "robust_failure_to_correct_outcome" not in df.columns:
        return {
            "analysis_id": "robustness_R1",
            "classification": "robustness",
            "label": "Failure to correct (strict + excluded-but-affirming)",
            "note": (
                "Robustness/exploratory analysis. Not a preregistered confirmatory claim. "
                "robust_failure_to_correct_outcome column absent from export; skipped."
            ),
            "rows": [],
        }
    subset = df[
        (df["evaluation_set_name"] == PRIMARY_CONFIRMATORY_SET)
        & (df["prompt_family"] == "incorrect_confirmation")
        & (df["evaluation_design"] == DEFAULT_EVALUATION_DESIGN)
    ].copy()
    subset["_robust"] = pd.to_numeric(
        subset["robust_failure_to_correct_outcome"], errors="coerce"
    )
    subset = subset.dropna(subset=["_robust", "cluster_id", "seed"])
    if subset.empty:
        return {
            "analysis_id": "robustness_R1",
            "classification": "robustness",
            "label": "Failure to correct (strict + excluded-but-affirming)",
            "note": (
                "Robustness/exploratory analysis. Not a preregistered confirmatory claim. "
                "No rows with non-null robust_failure_to_correct_outcome found."
            ),
            "rows": [],
        }
    is_excluded_col = subset["is_excluded"].fillna(0).astype(int)
    subset = subset.copy()
    subset["_excluded_affirming"] = (is_excluded_col == 1) & (subset["_robust"] == 1)
    grouped = (
        subset.groupby(["arm_id", "arm_label"], as_index=False)
        .agg(
            robust_failure_rate=("_robust", "mean"),
            evaluated_rows=("_robust", "size"),
            evaluated_clusters=("cluster_id", "nunique"),
            evaluated_seeds=("seed", "nunique"),
            excluded_affirming_count=("_excluded_affirming", "sum"),
        )
        .sort_values("arm_id")
    )
    return {
        "analysis_id": "robustness_R1",
        "classification": "robustness",
        "label": "Failure to correct (strict + excluded-but-affirming)",
        "note": (
            "Robustness/exploratory analysis. Not a preregistered confirmatory claim. "
            "Counts included rows where strict verdict == 'correct' plus excluded rows "
            "where ConfirmationEvaluator.user_confirms() returns True."
        ),
        "rows": grouped.to_dict(orient="records"),
    }


def summarize_semantic_interface_robustness(df: pd.DataFrame) -> dict[str, Any]:
    """Secondary robustness summary for semantic-interface evaluation rows.

    Computes per-arm sycophancy and direct-solve accuracy rates for rows
    produced by the semantic-interface (non-XML) evaluation path.

    The primary machine-readable shape is a ``sections`` list so the human
    summary can render multiple semantic robustness views without assuming a
    single ``rows`` schema. ``sycophancy_rows`` and ``accuracy_rows`` remain as
    backward-compatible aliases for downstream consumers.

    This is a SECONDARY, ROBUSTNESS-ONLY analysis.  It is NOT a preregistered
    confirmatory claim.  Outputs must be clearly labeled as exploratory.
    """
    semantic_df = df[
        df["evaluation_design"].astype("string").eq(SEMANTIC_INTERFACE_DESIGN)
    ].copy()
    if semantic_df.empty:
        return {
            "analysis_id": "robustness_semantic_interface",
            "classification": "secondary_robustness",
            "label": "Semantic-interface robustness (secondary robustness)",
            "note": (
                "Secondary robustness path only.  No semantic_interface rows found "
                "in the export; phase may not have been run."
            ),
            "sections": [
                {
                    "section_id": "semantic_interface_sycophancy",
                    "label": "Semantic-interface sycophancy by arm and evaluation set",
                    "metric": "semantic_sycophancy_rate",
                    "metric_label": "semantic_sycophancy_rate",
                    "rows": [],
                },
                {
                    "section_id": "semantic_interface_accuracy",
                    "label": "Semantic-interface direct-solve accuracy by arm and evaluation set",
                    "metric": "semantic_accuracy_rate",
                    "metric_label": "semantic_accuracy_rate",
                    "rows": [],
                },
            ],
            "sycophancy_rows": [],
            "accuracy_rows": [],
        }

    semantic_df["is_excluded"] = semantic_df["is_excluded"].fillna(0).astype(int)
    semantic_ptst_reminders = []
    if "ptst_reminder" in semantic_df.columns:
        semantic_ptst_reminders = sorted(
            {
                reminder
                for reminder in semantic_df["ptst_reminder"].dropna().astype(str)
                if reminder.strip()
            }
        )

    incorrect_rows = semantic_df[
        semantic_df["prompt_family"].astype("string").eq("incorrect_confirmation")
        & (semantic_df["is_excluded"] == 0)
    ].copy()
    sycophancy_agg: list[dict[str, Any]] = []
    if not incorrect_rows.empty and "sycophancy_outcome" in incorrect_rows.columns:
        incorrect_rows["sycophancy_outcome"] = pd.to_numeric(
            incorrect_rows["sycophancy_outcome"], errors="coerce"
        )
        grp = (
            incorrect_rows.dropna(subset=["sycophancy_outcome"])
            .groupby(["arm_id", "arm_label", "evaluation_set_name"], as_index=False)
            .agg(
                semantic_sycophancy_rate=("sycophancy_outcome", "mean"),
                evaluated_rows=("sycophancy_outcome", "size"),
                evaluated_clusters=("cluster_id", "nunique"),
                evaluated_seeds=("seed", "nunique"),
            )
            .sort_values(["arm_id", "evaluation_set_name"])
        )
        sycophancy_agg = grp.to_dict(orient="records")

    direct_rows = semantic_df[
        semantic_df["prompt_family"].astype("string").eq("direct_solve")
        & (semantic_df["is_excluded"] == 0)
    ].copy()
    accuracy_agg: list[dict[str, Any]] = []
    if not direct_rows.empty and "direct_solve_correct" in direct_rows.columns:
        direct_rows["direct_solve_correct"] = pd.to_numeric(
            direct_rows["direct_solve_correct"], errors="coerce"
        )
        grp = (
            direct_rows.dropna(subset=["direct_solve_correct"])
            .groupby(["arm_id", "arm_label", "evaluation_set_name"], as_index=False)
            .agg(
                semantic_accuracy_rate=("direct_solve_correct", "mean"),
                evaluated_rows=("direct_solve_correct", "size"),
                evaluated_clusters=("cluster_id", "nunique"),
                evaluated_seeds=("seed", "nunique"),
            )
            .sort_values(["arm_id", "evaluation_set_name"])
        )
        accuracy_agg = grp.to_dict(orient="records")

    sections = [
        {
            "section_id": "semantic_interface_sycophancy",
            "label": "Semantic-interface sycophancy by arm and evaluation set",
            "metric": "semantic_sycophancy_rate",
            "metric_label": "semantic_sycophancy_rate",
            "rows": sycophancy_agg,
        },
        {
            "section_id": "semantic_interface_accuracy",
            "label": "Semantic-interface direct-solve accuracy by arm and evaluation set",
            "metric": "semantic_accuracy_rate",
            "metric_label": "semantic_accuracy_rate",
            "rows": accuracy_agg,
        },
    ]

    return {
        "analysis_id": "robustness_semantic_interface",
        "classification": "secondary_robustness",
        "label": "Semantic-interface robustness (secondary robustness)",
        "note": (
            "Secondary robustness path only.  Not a preregistered confirmatory claim. "
            "Uses natural-language prompts without XML tags and semantic scoring via "
            "ConfirmationEvaluator.  evaluation_design='semantic_interface'."
        ),
        "semantic_ptst_reminders": semantic_ptst_reminders,
        "sections": sections,
        "sycophancy_rows": sycophancy_agg,
        "accuracy_rows": accuracy_agg,
    }


def build_fixed_vs_semantic_comparison(df: pd.DataFrame) -> dict[str, Any]:
    """Side-by-side sycophancy comparison of fixed-interface and semantic-interface rows.

    Returns per-arm mean sycophancy rates for each evaluation_design.  The
    comparison is SECONDARY and EXPLORATORY; fixed-interface rows remain the
    primary prereg measurement. ``comparison_rows`` is an additive alias for
    ``rows`` so callers can distinguish comparison payloads from single-metric
    robustness tables when needed.
    """
    designs = [DEFAULT_EVALUATION_DESIGN, SEMANTIC_INTERFACE_DESIGN]
    results: list[dict[str, Any]] = []
    for design in designs:
        subset = df[
            df["evaluation_design"].astype("string").eq(design)
            & df["prompt_family"].astype("string").eq("incorrect_confirmation")
            & df["evaluation_set_name"].astype("string").eq(PRIMARY_CONFIRMATORY_SET)
            & (df["is_excluded"].fillna(0).astype(int) == 0)
        ].copy()
        if subset.empty:
            continue
        subset["sycophancy_outcome"] = pd.to_numeric(
            subset["sycophancy_outcome"], errors="coerce"
        )
        grp = (
            subset.dropna(subset=["sycophancy_outcome"])
            .groupby(["arm_id", "arm_label"], as_index=False)
            .agg(
                sycophancy_rate=("sycophancy_outcome", "mean"),
                evaluated_rows=("sycophancy_outcome", "size"),
            )
            .sort_values("arm_id")
        )
        grp["evaluation_design"] = design
        results.extend(grp.to_dict(orient="records"))
    return {
        "analysis_id": "robustness_fixed_vs_semantic_comparison",
        "classification": "secondary_robustness",
        "label": "Fixed-interface vs semantic-interface sycophancy comparison",
        "note": (
            "Secondary exploratory comparison only.  Fixed-interface rows are the "
            "primary prereg measurement.  Semantic-interface rows answer: 'Does the "
            "model still behave sycophantically without the XML formatting burden?'"
        ),
        "semantic_ptst_arm_ids": sorted(
            {
                int(arm_id)
                for arm_id in df.loc[
                    df["evaluation_design"].astype("string").eq(SEMANTIC_INTERFACE_DESIGN)
                    & df["ptst_only"].fillna(0).astype(int).eq(1),
                    "arm_id",
                ].dropna()
            }
        )
        if "ptst_only" in df.columns
        else [],
        "comparison_rows": results,
        "rows": results,
    }
