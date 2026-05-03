"""Exclusion diagnostics and parseability/endorsement decomposition tables."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._shared import (
    _cast_nullable_int_columns,
    _group_and_count,
    _merge_group_metrics,
)

_DECOMPOSITION_NOTE = (
    "endorse_incorrect_overall_rate denominates by all incorrect_confirmation rows, "
    "treating unparseable (excluded) rows as non-endorsements. "
    "endorse_incorrect_given_parseable_rate denominates by parseable rows only. "
    "robust_failure_to_correct_rate, when reported, is a secondary robustness metric "
    "inclusive of excluded-but-semantically-affirming rows and is reported separately. "
    "fixed_interface rows use strict XML-contract parse fields; "
    "semantic_interface rows use semantic parser fields."
)


_DECOMPOSITION_COLUMNS = [
    "evaluation_design",
    "arm_id",
    "arm_slug",
    "seed",
    "evaluation_set_name",
    "n_rows",
    "parseable_count",
    "parseability_rate",
    "endorse_incorrect_parseable_count",
    "endorse_incorrect_given_parseable_rate",
    "endorse_incorrect_overall_count",
    "endorse_incorrect_overall_rate",
    "robust_failure_to_correct_rate",
]


def summarize_exclusion_diagnostics(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = df.copy()
    working["is_parseable"] = working["is_parseable"].fillna(0).astype(int)
    working["is_excluded"] = working["is_excluded"].fillna(0).astype(int)
    working["exclusion_category"] = (
        working["exclusion_category"].astype("string").fillna("").str.strip()
    )
    summary_groupings = [
        ("overall", []),
        ("arm", ["arm_id", "arm_slug", "arm_label"]),
        ("arm_seed", ["arm_id", "arm_slug", "arm_label", "seed"]),
        ("arm_evaluation_design", ["arm_id", "arm_slug", "arm_label", "evaluation_design"]),
        (
            "arm_seed_evaluation_design",
            ["arm_id", "arm_slug", "arm_label", "seed", "evaluation_design"],
        ),
    ]
    summary_frames: list[pd.DataFrame] = []
    category_frames: list[pd.DataFrame] = []
    for summary_level, group_columns in summary_groupings:
        base = _group_and_count(working, group_columns).rename(columns={"count": "total_rows"})
        base["summary_level"] = summary_level
        parseable = _group_and_count(working[working["is_parseable"] == 1], group_columns)
        excluded = _group_and_count(working[working["is_excluded"] == 1], group_columns)
        base = _merge_group_metrics(base, parseable, group_columns=group_columns, count_column="parseable_rows")
        base = _merge_group_metrics(base, excluded, group_columns=group_columns, count_column="excluded_rows")
        base["parseable_rows"] = base["parseable_rows"].fillna(0).astype(int)
        base["excluded_rows"] = base["excluded_rows"].fillna(0).astype(int)
        base["included_rows"] = base["total_rows"] - base["excluded_rows"]
        base["parseability_rate"] = base["parseable_rows"] / base["total_rows"]
        base["exclusion_rate"] = base["excluded_rows"] / base["total_rows"]
        base["included_rate"] = base["included_rows"] / base["total_rows"]

        excluded_with_category = working[
            (working["is_excluded"] == 1) & working["exclusion_category"].ne("")
        ]
        category_counts = _group_and_count(
            excluded_with_category,
            group_columns + ["exclusion_category"],
        ).rename(columns={"count": "excluded_category_count"})
        if category_counts.empty:
            category_counts["excluded_category_rate"] = pd.Series(dtype=float)
            category_counts["excluded_category_share"] = pd.Series(dtype=float)
            top = pd.DataFrame(columns=group_columns + [
                "top_exclusion_category",
                "top_exclusion_count",
                "top_exclusion_rate",
                "top_exclusion_share_of_excluded",
            ])
        else:
            if group_columns:
                category_counts = category_counts.merge(
                    base[group_columns + ["total_rows", "excluded_rows"]],
                    on=group_columns,
                    how="left",
                )
            else:
                category_counts["total_rows"] = int(base["total_rows"].iloc[0])
                category_counts["excluded_rows"] = int(base["excluded_rows"].iloc[0])
            category_counts["summary_level"] = summary_level
            category_counts["excluded_category_rate"] = (
                category_counts["excluded_category_count"] / category_counts["total_rows"]
            )
            category_counts["excluded_category_share"] = np.where(
                category_counts["excluded_rows"] > 0,
                category_counts["excluded_category_count"] / category_counts["excluded_rows"],
                0.0,
            )
            top = (
                category_counts.sort_values(
                    ["excluded_category_count", "exclusion_category"],
                    ascending=[False, True],
                )
                .drop_duplicates(subset=group_columns or ["summary_level"])
                .rename(
                    columns={
                        "exclusion_category": "top_exclusion_category",
                        "excluded_category_count": "top_exclusion_count",
                        "excluded_category_rate": "top_exclusion_rate",
                        "excluded_category_share": "top_exclusion_share_of_excluded",
                    }
                )
            )
            keep_columns = group_columns + [
                "top_exclusion_category",
                "top_exclusion_count",
                "top_exclusion_rate",
                "top_exclusion_share_of_excluded",
            ]
            top = top[keep_columns]
            category_frames.append(
                category_counts[
                    group_columns
                    + [
                        "summary_level",
                        "exclusion_category",
                        "excluded_category_count",
                        "excluded_category_rate",
                        "excluded_category_share",
                    ]
                ]
            )
        if group_columns:
            base = base.merge(top, on=group_columns, how="left")
        else:
            for column in [
                "top_exclusion_category",
                "top_exclusion_count",
                "top_exclusion_rate",
                "top_exclusion_share_of_excluded",
            ]:
                base[column] = top[column].iloc[0] if not top.empty else pd.NA
        summary_frames.append(base)

    summary_df = pd.concat(summary_frames, ignore_index=True, sort=False)
    category_df = pd.concat(category_frames, ignore_index=True, sort=False) if category_frames else pd.DataFrame(
        columns=[
            "summary_level",
            "arm_id",
            "arm_slug",
            "arm_label",
            "seed",
            "evaluation_design",
            "exclusion_category",
            "excluded_category_count",
            "excluded_category_rate",
            "excluded_category_share",
        ]
    )
    ordered_summary_columns = [
        "summary_level",
        "arm_id",
        "arm_slug",
        "arm_label",
        "seed",
        "evaluation_design",
        "total_rows",
        "parseable_rows",
        "parseability_rate",
        "excluded_rows",
        "exclusion_rate",
        "included_rows",
        "included_rate",
        "top_exclusion_category",
        "top_exclusion_count",
        "top_exclusion_rate",
        "top_exclusion_share_of_excluded",
    ]
    ordered_category_columns = [
        "summary_level",
        "arm_id",
        "arm_slug",
        "arm_label",
        "seed",
        "evaluation_design",
        "exclusion_category",
        "excluded_category_count",
        "excluded_category_rate",
        "excluded_category_share",
    ]
    summary_df = summary_df.reindex(columns=ordered_summary_columns).sort_values(
        ["summary_level", "arm_id", "seed", "evaluation_design"],
        na_position="last",
    )
    category_df = category_df.reindex(columns=ordered_category_columns).sort_values(
        ["summary_level", "arm_id", "seed", "evaluation_design", "excluded_category_count", "exclusion_category"],
        ascending=[True, True, True, True, False, True],
        na_position="last",
    )
    summary_df = _cast_nullable_int_columns(
        summary_df,
        [
            "arm_id",
            "seed",
            "total_rows",
            "parseable_rows",
            "excluded_rows",
            "included_rows",
            "top_exclusion_count",
        ],
    )
    category_df = _cast_nullable_int_columns(
        category_df,
        [
            "arm_id",
            "seed",
            "excluded_category_count",
        ],
    )
    return summary_df, category_df


def compute_parseability_endorsement_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    """Compute P(parseable), P(endorse|parseable), P(endorse overall) per arm/seed/set.

    Restricted to incorrect_confirmation rows.  Overall endorsement rate treats
    unparseable rows as non-endorsements (denominator = all rows in group).
    Conditional rate uses parseable rows as both numerator and denominator scope.
    Fixed-interface groups use strict XML-contract parse fields via is_parseable
    and sycophancy_outcome; semantic-interface groups use semantic parser fields.
    """
    ic_mask = df["prompt_family"].astype("string").eq("incorrect_confirmation")
    ic_df = df[ic_mask].copy()
    if ic_df.empty:
        return pd.DataFrame(columns=_DECOMPOSITION_COLUMNS)

    ic_df["_is_parseable"] = ic_df["is_parseable"].fillna(0).astype(int)
    ic_df["_sycophancy"] = pd.to_numeric(ic_df["sycophancy_outcome"], errors="coerce")

    has_robust = "robust_failure_to_correct_outcome" in ic_df.columns
    if has_robust:
        ic_df["_robust"] = pd.to_numeric(ic_df["robust_failure_to_correct_outcome"], errors="coerce")
    else:
        ic_df["_robust"] = pd.NA

    # Flag: parseable AND sycophancy_outcome == 1 (overall numerator = conditional numerator)
    ic_df["_endorse_flag"] = (
        (ic_df["_is_parseable"] == 1) & (ic_df["_sycophancy"] == 1)
    ).astype(int)

    group_keys = ["evaluation_design", "arm_id", "arm_slug", "seed", "evaluation_set_name"]

    agg = (
        ic_df.groupby(group_keys, dropna=False)
        .agg(
            n_rows=("_is_parseable", "size"),
            parseable_count=("_is_parseable", "sum"),
            endorse_incorrect_overall_count=("_endorse_flag", "sum"),
        )
        .reset_index()
    )
    agg["parseability_rate"] = agg["parseable_count"] / agg["n_rows"]
    agg["endorse_incorrect_parseable_count"] = agg["endorse_incorrect_overall_count"]
    agg["endorse_incorrect_given_parseable_rate"] = np.where(
        agg["parseable_count"] > 0,
        agg["endorse_incorrect_parseable_count"] / agg["parseable_count"],
        np.nan,
    )
    agg["endorse_incorrect_overall_rate"] = (
        agg["endorse_incorrect_overall_count"] / agg["n_rows"]
    )

    if has_robust:
        robust_agg = (
            ic_df.groupby(group_keys, dropna=False)["_robust"]
            .mean()
            .reset_index()
            .rename(columns={"_robust": "robust_failure_to_correct_rate"})
        )
        agg = agg.merge(robust_agg, on=group_keys, how="left")
    else:
        agg["robust_failure_to_correct_rate"] = None

    agg = _cast_nullable_int_columns(
        agg,
        ["n_rows", "parseable_count",
         "endorse_incorrect_parseable_count", "endorse_incorrect_overall_count"],
    )
    return agg[_DECOMPOSITION_COLUMNS].sort_values(group_keys, na_position="last").reset_index(drop=True)


def diagnostics_summary_lines(summary_df: pd.DataFrame) -> list[str]:
    lines = ["Diagnostics"]
    overall = summary_df[summary_df["summary_level"] == "overall"]
    if not overall.empty:
        row = overall.iloc[0]
        lines.append(
            "- Overall parseability={:.1%}, exclusion rate={:.1%}, top exclusion={}".format(
                float(row["parseability_rate"]),
                float(row["exclusion_rate"]),
                row["top_exclusion_category"] if pd.notna(row["top_exclusion_category"]) else "none",
            )
        )
    arm_seed = summary_df[summary_df["summary_level"] == "arm_seed"].copy()
    if not arm_seed.empty:
        worst = arm_seed.sort_values(
            ["exclusion_rate", "arm_id", "seed"],
            ascending=[False, True, True],
            na_position="last",
        ).iloc[0]
        lines.append(
            "- Highest arm-seed exclusion: {} seed {} at {:.1%} (top category: {})".format(
                worst["arm_label"],
                int(worst["seed"]),
                float(worst["exclusion_rate"]),
                worst["top_exclusion_category"] if pd.notna(worst["top_exclusion_category"]) else "none",
            )
        )
    return lines
