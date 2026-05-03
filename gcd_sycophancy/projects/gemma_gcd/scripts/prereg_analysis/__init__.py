"""Preregistered Section 7 analysis suite.

The :mod:`analyze_preregistration` module is the canonical CLI entry point and
re-exports the public API of this package for backwards compatibility.  Most
callers should keep importing from ``analyze_preregistration``; new code may
import the focused submodules directly when only one concern is needed.
"""

from __future__ import annotations

from ._shared import (
    AnalysisSpec,
    CAPABILITY_DIAGNOSTIC_SETS,
    DIAGNOSTIC_CATEGORY_SUFFIX,
    DIAGNOSTIC_SUMMARY_SUFFIX,
    ENDORSEMENT_DECOMPOSITION_SUFFIX,
    H2_DECISION_INTERVAL_TYPE,
    H2_NONINFERIORITY_MARGIN,
    H2_REPORTING_RULE,
    ONE_SIDED_ALPHA,
    PREREG_ANALYSIS_SCHEMA_VERSION,
    PRIMARY_ARMS,
    REQUIRED_COLUMNS,
    SECONDARY_HYPOTHESIS_IDS,
    SEMANTIC_INTERFACE_DESIGN,
    TWO_SIDED_ALPHA,
    apply_holm_correction,
    claim_status_from_interval,
    compute_is_parseable_series,
    logger,
)
from .capability_diagnostics import run_direct_solve_capability_diagnostics
from .exclusion_diagnostics import (
    compute_parseability_endorsement_decomposition,
    diagnostics_summary_lines,
    summarize_exclusion_diagnostics,
)
from .exploratory import run_exploratory_e7, run_exploratory_e8
from .interpretations import (
    build_construct_validity_interpretation,
    determine_joint_interpretation,
)
from .models import fit_mixed_effects_logistic
from .paired_reporting import compute_paired_reporting_supplement
from .robustness import (
    build_fixed_vs_semantic_comparison,
    run_robustness_failure_to_correct,
    summarize_semantic_interface_robustness,
)
from .runner import (
    build_arg_parser,
    main,
    run_preregistration_analyses,
    write_outputs,
)
from .schema_invariance import build_schema_invariance_analysis
from .specs import build_analysis_specs, subset_for_spec
from .summary import build_human_summary

__all__ = [
    "AnalysisSpec",
    "CAPABILITY_DIAGNOSTIC_SETS",
    "DIAGNOSTIC_CATEGORY_SUFFIX",
    "DIAGNOSTIC_SUMMARY_SUFFIX",
    "ENDORSEMENT_DECOMPOSITION_SUFFIX",
    "H2_DECISION_INTERVAL_TYPE",
    "H2_NONINFERIORITY_MARGIN",
    "H2_REPORTING_RULE",
    "ONE_SIDED_ALPHA",
    "PREREG_ANALYSIS_SCHEMA_VERSION",
    "PRIMARY_ARMS",
    "REQUIRED_COLUMNS",
    "SECONDARY_HYPOTHESIS_IDS",
    "SEMANTIC_INTERFACE_DESIGN",
    "TWO_SIDED_ALPHA",
    "apply_holm_correction",
    "build_analysis_specs",
    "build_arg_parser",
    "build_construct_validity_interpretation",
    "build_fixed_vs_semantic_comparison",
    "build_human_summary",
    "build_schema_invariance_analysis",
    "claim_status_from_interval",
    "compute_is_parseable_series",
    "compute_paired_reporting_supplement",
    "compute_parseability_endorsement_decomposition",
    "determine_joint_interpretation",
    "diagnostics_summary_lines",
    "fit_mixed_effects_logistic",
    "logger",
    "main",
    "run_direct_solve_capability_diagnostics",
    "run_exploratory_e7",
    "run_exploratory_e8",
    "run_preregistration_analyses",
    "run_robustness_failure_to_correct",
    "subset_for_spec",
    "summarize_exclusion_diagnostics",
    "summarize_semantic_interface_robustness",
    "write_outputs",
]
