"""Run the preregistered Section 7 analysis suite on exported problem-level rows.

The prereg export's primary fields (`parsed_verdict`, `parsed_numeric_answer`,
and `is_parseable`) are interface-aware:
- fixed_interface / bounded_search: strict-parser outputs
- semantic_interface: semantic scorer outputs

When an older export is missing `is_parseable`, this module reconstructs it
from strict fields for fixed-interface rows and from the primary fields for
semantic-interface rows.

Implementation note (Tidying R3b): the analysis logic now lives in the
``prereg_analysis`` package; this module is a thin orchestration / CLI shim
that re-exports the public API for backwards compatibility.  The PR #85
missing-arms guard in ``compute_paired_reporting_supplement`` (referenced by
``rerun_panel_analysis.sh`` and ``run_append_above_analysis.sh`` via
``grep -q 'missing_arms'``) is preserved in
``prereg_analysis.paired_reporting``; the literal token ``missing_arms`` is
retained in this comment so the shell preflight checks continue to match.
"""

from __future__ import annotations

import sys
from pathlib import Path

# pandas/numpy are re-exported as ``analysis.pd`` / ``analysis.np`` so existing
# tests can monkeypatch ``analysis.np.random`` and inspect dataframe helpers
# through the canonical module name.
import numpy as np  # noqa: F401 (re-exported)
import pandas as pd  # noqa: F401 (re-exported)

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from prereg_analysis import (  # noqa: E402
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
    build_analysis_specs,
    build_arg_parser,
    build_construct_validity_interpretation,
    build_fixed_vs_semantic_comparison,
    build_human_summary,
    build_schema_invariance_analysis,
    claim_status_from_interval,
    compute_is_parseable_series,
    compute_paired_reporting_supplement,
    compute_parseability_endorsement_decomposition,
    determine_joint_interpretation,
    diagnostics_summary_lines,
    fit_mixed_effects_logistic,
    logger,
    main,
    run_direct_solve_capability_diagnostics,
    run_exploratory_e7,
    run_exploratory_e8,
    run_preregistration_analyses,
    run_robustness_failure_to_correct,
    subset_for_spec,
    summarize_exclusion_diagnostics,
    summarize_semantic_interface_robustness,
    write_outputs,
)
# Private helpers a few existing call sites still reach for through the legacy
# module surface (``analysis._load_dataframe`` in tests).
from prereg_analysis._shared import (  # noqa: E402
    _cast_nullable_int_columns,
    _format_metric_value,
    _group_and_count,
    _load_dataframe,
    _merge_group_metrics,
    _normal_cdf,
)
from export_prereg_problem_level_data import (  # noqa: E402,F401 (re-exported)
    DEFAULT_EVALUATION_DESIGN,
    PARAPHRASE_SET,
    PRIMARY_CONFIRMATORY_SET,
    SAME_DOMAIN_EXTRAPOLATION_SET,
    canonicalize_evaluation_set_name,
)

__all__ = [
    "AnalysisSpec",
    "CAPABILITY_DIAGNOSTIC_SETS",
    "DEFAULT_EVALUATION_DESIGN",
    "DIAGNOSTIC_CATEGORY_SUFFIX",
    "DIAGNOSTIC_SUMMARY_SUFFIX",
    "ENDORSEMENT_DECOMPOSITION_SUFFIX",
    "H2_DECISION_INTERVAL_TYPE",
    "H2_NONINFERIORITY_MARGIN",
    "H2_REPORTING_RULE",
    "ONE_SIDED_ALPHA",
    "PARAPHRASE_SET",
    "PREREG_ANALYSIS_SCHEMA_VERSION",
    "PRIMARY_ARMS",
    "PRIMARY_CONFIRMATORY_SET",
    "REQUIRED_COLUMNS",
    "SAME_DOMAIN_EXTRAPOLATION_SET",
    "SCRIPT_DIR",
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
    "canonicalize_evaluation_set_name",
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


if __name__ == "__main__":
    raise SystemExit(main())
