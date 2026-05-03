"""Paired reporting supplement (per-cluster mean differences with cluster-robust SE).

Includes the missing-arms guard (PR #85): when the experiment was trained on a
single arm (e.g., panel candidates with ``--only-arms 2``), the per-cluster
pivot will not contain both columns and we raise ``ValueError`` so the runner
can skip rather than crash with a KeyError.  ``rerun_panel_analysis.sh`` keys
its blocking precondition off this guard.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd


def compute_paired_reporting_supplement(
    subset: pd.DataFrame,
    *,
    outcome_column: str,
    arm_a_id: int,
    arm_b_id: int,
) -> dict[str, Any]:
    grouped = (
        subset.dropna(subset=[outcome_column])
        .groupby(["cluster_id", "seed", "arm_id"], as_index=False)[outcome_column]
        .mean()
    )
    pivoted = (
        grouped.groupby(["cluster_id", "arm_id"], as_index=False)[outcome_column]
        .mean()
        .pivot(index="cluster_id", columns="arm_id", values=outcome_column)
    )
    missing_arms = [aid for aid in (arm_a_id, arm_b_id) if aid not in pivoted.columns]
    if missing_arms:
        raise ValueError(
            "No paired clusters available for the paired reporting supplement: "
            f"missing arm column(s) {missing_arms} in the per-cluster pivot. "
            "This typically happens when the experiment was trained on a single "
            "arm (e.g., panel candidates with --only-arms 2)."
        )
    arm_means = pivoted.dropna(subset=[arm_a_id, arm_b_id])
    if arm_means.empty:
        raise ValueError("No paired clusters available for the paired reporting supplement.")
    differences = arm_means[arm_a_id] - arm_means[arm_b_id]
    n_clusters = int(len(differences))
    dbar = float(differences.mean())
    se = float(differences.std(ddof=1) / math.sqrt(n_clusters)) if n_clusters > 1 else 0.0
    ci = [dbar - 1.96 * se, dbar + 1.96 * se]
    correlation = float(arm_means[arm_a_id].corr(arm_means[arm_b_id])) if n_clusters > 1 else math.nan
    return {
        "mean_difference": dbar,
        "cluster_robust_se": se,
        "ci_95": ci,
        "evaluated_cluster_count": n_clusters,
        "arm_outcome_correlation": correlation,
    }
