"""Schema invariance comparator: fixed_interface vs semantic_interface.

SECONDARY robustness only.  Does NOT replace fixed-interface H1-H5 confirmatory
results.  Returns ``status`` ∈ {"pass", "fail", "unavailable"} based on whether
the Arm 2 vs Arm 1 sycophancy-rate effect direction agrees across both
interfaces on the primary confirmatory set.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from export_prereg_problem_level_data import (
    DEFAULT_EVALUATION_DESIGN,
    PRIMARY_CONFIRMATORY_SET,
)

from ._shared import SEMANTIC_INTERFACE_DESIGN

_SCHEMA_INVARIANCE_DESIGNS = (DEFAULT_EVALUATION_DESIGN, SEMANTIC_INTERFACE_DESIGN)


def _schema_invariance_sycophancy_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for design in _SCHEMA_INVARIANCE_DESIGNS:
        subset = df[
            df["evaluation_design"].astype("string").eq(design)
            & df["prompt_family"].astype("string").eq("incorrect_confirmation")
            & (df["is_excluded"].fillna(0).astype(int) == 0)
        ].copy()
        if subset.empty:
            continue
        subset["sycophancy_outcome"] = pd.to_numeric(
            subset["sycophancy_outcome"], errors="coerce"
        )
        grp = (
            subset.dropna(subset=["sycophancy_outcome"])
            .groupby(["arm_id", "arm_label", "evaluation_set_name"], as_index=False)
            .agg(
                sycophancy_rate=("sycophancy_outcome", "mean"),
                evaluated_rows=("sycophancy_outcome", "size"),
            )
            .sort_values(["arm_id", "evaluation_set_name"])
        )
        grp["evaluation_design"] = design
        rows.extend(grp.to_dict(orient="records"))
    return rows


def _schema_invariance_conditional_sycophancy_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    if "conditional_sycophancy_eligible" not in df.columns:
        return []
    rows: list[dict[str, Any]] = []
    eligible_mask = pd.to_numeric(
        df["conditional_sycophancy_eligible"], errors="coerce"
    ).fillna(0).astype(int).eq(1)
    for design in _SCHEMA_INVARIANCE_DESIGNS:
        subset = df[
            df["evaluation_design"].astype("string").eq(design)
            & df["prompt_family"].astype("string").eq("incorrect_confirmation")
            & (df["is_excluded"].fillna(0).astype(int) == 0)
            & eligible_mask
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
                conditional_sycophancy_rate=("sycophancy_outcome", "mean"),
                evaluated_rows=("sycophancy_outcome", "size"),
            )
            .sort_values("arm_id")
        )
        grp["evaluation_design"] = design
        rows.extend(grp.to_dict(orient="records"))
    return rows


def _schema_invariance_direct_solve_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for design in _SCHEMA_INVARIANCE_DESIGNS:
        subset = df[
            df["evaluation_design"].astype("string").eq(design)
            & df["prompt_family"].astype("string").eq("direct_solve")
            & (df["is_excluded"].fillna(0).astype(int) == 0)
        ].copy()
        if subset.empty:
            continue
        subset["direct_solve_correct"] = pd.to_numeric(
            subset["direct_solve_correct"], errors="coerce"
        )
        grp = (
            subset.dropna(subset=["direct_solve_correct"])
            .groupby(["arm_id", "arm_label", "evaluation_set_name"], as_index=False)
            .agg(
                direct_solve_accuracy=("direct_solve_correct", "mean"),
                evaluated_rows=("direct_solve_correct", "size"),
            )
            .sort_values(["arm_id", "evaluation_set_name"])
        )
        grp["evaluation_design"] = design
        rows.extend(grp.to_dict(orient="records"))
    return rows


def _schema_invariance_parseability_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for design in _SCHEMA_INVARIANCE_DESIGNS:
        subset = df[df["evaluation_design"].astype("string").eq(design)].copy()
        if subset.empty:
            continue
        subset["is_parseable"] = subset["is_parseable"].fillna(0).astype(int)
        subset["is_excluded"] = subset["is_excluded"].fillna(0).astype(int)
        grp = (
            subset.groupby(["arm_id", "arm_label"], as_index=False)
            .agg(
                total_rows=("is_parseable", "size"),
                parseability_rate=("is_parseable", "mean"),
                exclusion_rate=("is_excluded", "mean"),
            )
            .sort_values("arm_id")
        )
        grp["evaluation_design"] = design
        rows.extend(grp.to_dict(orient="records"))
    return rows


def _schema_invariance_effect_direction(
    sycophancy_rows: list[dict[str, Any]],
    *,
    arm_a_id: int = 2,
    arm_b_id: int = 1,
) -> tuple[list[dict[str, Any]], str]:
    """Return per-design (arm_a - arm_b) sycophancy-rate gaps and an agreement flag.

    Status:
      - unavailable: at least one design lacks both arms on the primary set
      - pass: both designs show same effect direction (or both within tolerance of zero)
      - fail: designs disagree on direction
    """
    tolerance = 1e-9
    summary_rows: list[dict[str, Any]] = []
    by_design: dict[str, dict[int, float]] = {}
    for row in sycophancy_rows:
        if row.get("evaluation_set_name") != PRIMARY_CONFIRMATORY_SET:
            continue
        design = row.get("evaluation_design")
        arm_id = int(row.get("arm_id"))
        if arm_id not in (arm_a_id, arm_b_id):
            continue
        by_design.setdefault(design, {})[arm_id] = float(row.get("sycophancy_rate"))

    for design in _SCHEMA_INVARIANCE_DESIGNS:
        arm_rates = by_design.get(design, {})
        if arm_a_id in arm_rates and arm_b_id in arm_rates:
            gap = arm_rates[arm_a_id] - arm_rates[arm_b_id]
            summary_rows.append({
                "evaluation_design": design,
                "arm_a_id": arm_a_id,
                "arm_b_id": arm_b_id,
                "arm_a_sycophancy_rate": arm_rates[arm_a_id],
                "arm_b_sycophancy_rate": arm_rates[arm_b_id],
                "arm_a_minus_arm_b": gap,
                "direction": (
                    "neutral" if abs(gap) <= tolerance
                    else "negative" if gap < 0
                    else "positive"
                ),
            })
        else:
            summary_rows.append({
                "evaluation_design": design,
                "arm_a_id": arm_a_id,
                "arm_b_id": arm_b_id,
                "arm_a_sycophancy_rate": arm_rates.get(arm_a_id),
                "arm_b_sycophancy_rate": arm_rates.get(arm_b_id),
                "arm_a_minus_arm_b": None,
                "direction": "unavailable",
            })

    directions = {row["direction"] for row in summary_rows}
    if "unavailable" in directions:
        status = "unavailable"
    elif len(directions) == 1:
        status = "pass"
    elif directions == {"negative", "neutral"} or directions == {"positive", "neutral"}:
        status = "pass"
    else:
        status = "fail"
    return summary_rows, status


def build_schema_invariance_analysis(df: pd.DataFrame) -> dict[str, Any]:
    """Robustness comparator: fixed_interface vs semantic_interface schema invariance.

    SECONDARY robustness only.  Does NOT replace fixed-interface H1-H5
    confirmatory results.  Returns ``status`` ∈ {"pass", "fail", "unavailable"}
    based on whether the Arm 2 vs Arm 1 sycophancy-rate effect direction agrees
    across both interfaces on the primary confirmatory set.
    """
    label = "Schema invariance: fixed_interface vs semantic_interface"
    analysis_id = "robustness_schema_invariance"

    has_fixed = bool((df["evaluation_design"].astype("string") == DEFAULT_EVALUATION_DESIGN).any())
    has_semantic = bool((df["evaluation_design"].astype("string") == SEMANTIC_INTERFACE_DESIGN).any())
    if not (has_fixed and has_semantic):
        missing = []
        if not has_fixed:
            missing.append(DEFAULT_EVALUATION_DESIGN)
        if not has_semantic:
            missing.append(SEMANTIC_INTERFACE_DESIGN)
        return {
            "analysis_id": analysis_id,
            "classification": "secondary_robustness",
            "label": label,
            "status": "unavailable",
            "note": (
                "Schema invariance is a secondary robustness comparator and does NOT "
                "replace fixed-interface H1-H5 results.  No rows present for: "
                + ", ".join(missing)
                + "."
            ),
            "sections": [],
            "effect_direction": [],
            "missing_designs": missing,
        }

    sycophancy_rows = _schema_invariance_sycophancy_rows(df)
    conditional_rows = _schema_invariance_conditional_sycophancy_rows(df)
    direct_solve_rows = _schema_invariance_direct_solve_rows(df)
    parseability_rows = _schema_invariance_parseability_rows(df)
    effect_rows, status = _schema_invariance_effect_direction(sycophancy_rows)

    return {
        "analysis_id": analysis_id,
        "classification": "secondary_robustness",
        "label": label,
        "status": status,
        "note": (
            "Secondary robustness only.  Does NOT replace fixed-interface H1-H5 "
            "confirmatory results.  Status compares Arm 2 vs Arm 1 sycophancy-rate "
            "effect direction across fixed_interface and semantic_interface on the "
            "primary confirmatory set."
        ),
        "sections": [
            {
                "section_id": "sycophancy_rate_by_arm_and_set",
                "label": "Sycophancy rate by arm and evaluation set",
                "rows": sycophancy_rows,
            },
            {
                "section_id": "conditional_sycophancy_rate_by_arm",
                "label": "Conditional sycophancy rate by arm (eligible rows only)",
                "rows": conditional_rows,
            },
            {
                "section_id": "direct_solve_accuracy_by_arm_and_set",
                "label": "Direct-solve accuracy by arm and evaluation set",
                "rows": direct_solve_rows,
            },
            {
                "section_id": "parseability_and_exclusion_by_arm",
                "label": "Parseability and exclusion rates by arm",
                "rows": parseability_rows,
            },
        ],
        "effect_direction": effect_rows,
    }
