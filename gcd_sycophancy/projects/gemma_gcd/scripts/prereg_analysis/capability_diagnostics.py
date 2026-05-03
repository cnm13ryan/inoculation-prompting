"""Direct-solve capability diagnostics on held-out capability splits.

Secondary capability diagnostic only.  Not a primary H1-H5 input.  Measures
raw GCD solving ability without incorrect-assertion pressure so capability
failures can be separated from failures to resist sycophantic pressure.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from export_prereg_problem_level_data import DEFAULT_EVALUATION_DESIGN

from ._shared import CAPABILITY_DIAGNOSTIC_SETS


def run_direct_solve_capability_diagnostics(df: pd.DataFrame) -> dict[str, Any]:
    """Secondary capability diagnostic: direct-solve accuracy on held-out capability splits.

    NOT a primary H1-H5 input.  Measures raw GCD solving ability without
    incorrect-assertion pressure so capability failures can be separated
    from failures to resist sycophantic pressure.

    Populated only when the capability diagnostic splits were included in the
    evaluation (pass --include-capability-diagnostics to evaluate_base_model.py).
    """
    subset = df[
        df["evaluation_set_name"].isin(CAPABILITY_DIAGNOSTIC_SETS)
        & df["prompt_family"].astype("string").eq("direct_solve")
        & df["evaluation_design"].astype("string").eq(DEFAULT_EVALUATION_DESIGN)
    ].copy()
    if subset.empty:
        return {
            "analysis_id": "capability_diagnostics",
            "classification": "secondary_capability_diagnostic",
            "label": "Direct-solve capability diagnostics",
            "note": (
                "Secondary capability diagnostic only.  Not a primary H1-H5 input.  "
                "No direct-solve capability diagnostic rows found in the export; "
                "splits may not have been evaluated (pass "
                "--include-capability-diagnostics to evaluate_base_model.py)."
            ),
            "rows": [],
        }
    subset["is_parseable"] = subset["is_parseable"].fillna(0).astype(int)
    subset["is_excluded"] = subset["is_excluded"].fillna(0).astype(int)
    subset["direct_solve_correct"] = pd.to_numeric(
        subset["direct_solve_correct"], errors="coerce"
    )
    group_keys = ["arm_id", "arm_label", "seed", "evaluation_set_name"]
    included = subset[subset["is_excluded"] == 0].dropna(subset=["direct_solve_correct"])
    acc_df = (
        included
        .groupby(group_keys, as_index=False)
        .agg(direct_solve_accuracy=("direct_solve_correct", "mean"))
    )
    grp = (
        subset
        .groupby(group_keys, as_index=False)
        .agg(
            total_rows=("is_parseable", "size"),
            parseability_rate=("is_parseable", "mean"),
            exclusion_rate=("is_excluded", "mean"),
            evaluated_clusters=("cluster_id", "nunique"),
        )
        .merge(acc_df, on=group_keys, how="left")
        .sort_values(["evaluation_set_name", "arm_id", "seed"])
    )
    return {
        "analysis_id": "capability_diagnostics",
        "classification": "secondary_capability_diagnostic",
        "label": "Direct-solve capability diagnostics",
        "note": (
            "Secondary capability diagnostic only.  Not a primary H1-H5 input.  "
            "Measures raw GCD solving accuracy on direct-solve-only held-out splits, "
            "independent of incorrect-assertion pressure.  Strict XML parsing is "
            "used throughout; direct_solve_accuracy is computed over included "
            "(non-excluded) rows only."
        ),
        "rows": grp.to_dict(orient="records"),
    }
