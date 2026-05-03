"""Hypothesis specifications (H1, H1c, H2, H3, H4, H5, E1-E6) and subsetting.

Each ``AnalysisSpec`` describes a single comparison: which arms, which
evaluation set, which prompt family, which evaluation design, which outcome
column, and (for non-inferiority claims like H2) the margin and alpha.
"""

from __future__ import annotations

import pandas as pd

from export_prereg_problem_level_data import (
    DEFAULT_EVALUATION_DESIGN,
    PARAPHRASE_SET,
    PRIMARY_CONFIRMATORY_SET,
    SAME_DOMAIN_EXTRAPOLATION_SET,
)

from ._shared import (
    AnalysisSpec,
    CAPABILITY_DIAGNOSTIC_SETS,
    H2_NONINFERIORITY_MARGIN,
    ONE_SIDED_ALPHA,
)


def build_analysis_specs() -> list[AnalysisSpec]:
    return [
        AnalysisSpec("analysis_1", "H1", "Sycophancy reduction", "confirmatory", "sycophancy_outcome", 2, 1, PRIMARY_CONFIRMATORY_SET, "incorrect_confirmation", DEFAULT_EVALUATION_DESIGN),
        AnalysisSpec(
            "analysis_1c",
            "H1c",
            "Conditional sycophancy reduction",
            "confirmatory",
            "sycophancy_outcome",
            2,
            1,
            PRIMARY_CONFIRMATORY_SET,
            "incorrect_confirmation",
            DEFAULT_EVALUATION_DESIGN,
            eligibility_column="conditional_sycophancy_eligible",
        ),
        AnalysisSpec("analysis_2", "H2", "Capability preservation", "confirmatory", "direct_solve_correct", 2, 1, PRIMARY_CONFIRMATORY_SET, "direct_solve", DEFAULT_EVALUATION_DESIGN, alpha=ONE_SIDED_ALPHA, noninferiority_margin=H2_NONINFERIORITY_MARGIN),
        AnalysisSpec("analysis_3", "H3", "Paraphrase robustness", "secondary_confirmatory", "sycophancy_outcome", 2, 1, PARAPHRASE_SET, "incorrect_confirmation", DEFAULT_EVALUATION_DESIGN),
        AnalysisSpec("analysis_4", "H4", "Same-domain extrapolation", "secondary_confirmatory", "sycophancy_outcome", 2, 1, SAME_DOMAIN_EXTRAPOLATION_SET, "incorrect_confirmation", DEFAULT_EVALUATION_DESIGN),
        AnalysisSpec("analysis_5", "H5", "Accessible capability", "secondary_confirmatory", "sycophancy_outcome", 2, 1, PRIMARY_CONFIRMATORY_SET, "incorrect_confirmation", "bounded_search"),
        AnalysisSpec("exploratory_E1", "E1", "Irrelevant-prompt control vs neutral baseline", "exploratory", "sycophancy_outcome", 3, 1, PRIMARY_CONFIRMATORY_SET, "incorrect_confirmation", DEFAULT_EVALUATION_DESIGN),
        AnalysisSpec("exploratory_E2", "E2", "Praise-only control vs neutral baseline", "exploratory", "sycophancy_outcome", 4, 1, PRIMARY_CONFIRMATORY_SET, "incorrect_confirmation", DEFAULT_EVALUATION_DESIGN),
        AnalysisSpec("exploratory_E3", "E3", "IP vs irrelevant-prompt control", "exploratory", "sycophancy_outcome", 2, 3, PRIMARY_CONFIRMATORY_SET, "incorrect_confirmation", DEFAULT_EVALUATION_DESIGN),
        AnalysisSpec("exploratory_E4", "E4", "Correction-data arm vs neutral baseline", "exploratory", "sycophancy_outcome", 5, 1, PRIMARY_CONFIRMATORY_SET, "incorrect_confirmation", DEFAULT_EVALUATION_DESIGN),
        AnalysisSpec("exploratory_E5", "E5", "PTST baseline vs neutral baseline", "exploratory", "sycophancy_outcome", 6, 1, PRIMARY_CONFIRMATORY_SET, "incorrect_confirmation", DEFAULT_EVALUATION_DESIGN),
        AnalysisSpec("exploratory_E6", "E6", "IP vs PTST baseline", "exploratory", "sycophancy_outcome", 2, 6, PRIMARY_CONFIRMATORY_SET, "incorrect_confirmation", DEFAULT_EVALUATION_DESIGN),
    ]


def subset_for_spec(df: pd.DataFrame, spec: AnalysisSpec) -> pd.DataFrame:
    subset = df.copy()
    subset = subset[~subset["evaluation_set_name"].isin(CAPABILITY_DIAGNOSTIC_SETS)]
    if spec.arm_a_id is not None and spec.arm_b_id is not None:
        subset = subset[subset["arm_id"].isin([spec.arm_a_id, spec.arm_b_id])]
    if spec.evaluation_set_name is not None:
        subset = subset[subset["evaluation_set_name"] == spec.evaluation_set_name]
    if spec.prompt_family is not None:
        subset = subset[subset["prompt_family"] == spec.prompt_family]
    if spec.evaluation_design is not None:
        subset = subset[subset["evaluation_design"] == spec.evaluation_design]
    if spec.outcome_column is not None:
        subset = subset[subset["is_excluded"].fillna(0).astype(int) == 0]
        subset = subset.dropna(subset=[spec.outcome_column, "cluster_id", "seed"])
    if spec.eligibility_column is not None:
        eligibility = pd.to_numeric(
            subset[spec.eligibility_column], errors="coerce"
        ).fillna(0).astype(int)
        subset = subset[eligibility == 1]
    return subset
