"""Top-level orchestrator and CLI for the preregistered Section 7 analysis suite.

This module wires the spec-driven confirmatory analyses, the paired-reporting
supplement, exploratory analyses, robustness sections, capability diagnostics,
schema invariance, exclusion diagnostics, and human summary into a single
``run_preregistration_analyses`` entry point and a thin ``main`` CLI.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from ..artifact_provenance import (
    build_provenance,
    write_json_with_provenance,
)

from ._shared import (
    DIAGNOSTIC_CATEGORY_SUFFIX,
    DIAGNOSTIC_SUMMARY_SUFFIX,
    ENDORSEMENT_DECOMPOSITION_SUFFIX,
    H2_REPORTING_RULE,
    PREREG_ANALYSIS_SCHEMA_VERSION,
    SECONDARY_HYPOTHESIS_IDS,
    _cast_nullable_int_columns,
    _load_dataframe,
    apply_holm_correction,
    logger,
)
from .capability_diagnostics import run_direct_solve_capability_diagnostics
from .exclusion_diagnostics import (
    _DECOMPOSITION_COLUMNS,
    _DECOMPOSITION_NOTE,
    compute_parseability_endorsement_decomposition,
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
from .schema_invariance import build_schema_invariance_analysis
from .specs import build_analysis_specs, subset_for_spec
from .summary import build_human_summary

SCRIPT_DIR = Path(__file__).resolve().parent.parent


def run_preregistration_analyses(
    df: pd.DataFrame,
    *,
    fit_fn: Callable[..., dict[str, Any]] = fit_mixed_effects_logistic,
) -> dict[str, Any]:
    exclusion_summary, exclusion_categories = summarize_exclusion_diagnostics(df)
    confirmatory_results: list[dict[str, Any]] = []
    exploratory_results: list[dict[str, Any]] = []
    paired_reporting: dict[str, Any] = {}

    for spec in build_analysis_specs():
        subset = subset_for_spec(df, spec)
        result = {
            "analysis_id": spec.analysis_id,
            "hypothesis_id": spec.hypothesis_id,
            "classification": spec.classification,
            "label": spec.label,
            "arm_a_id": spec.arm_a_id,
            "arm_b_id": spec.arm_b_id,
            "evaluation_set_name": spec.evaluation_set_name,
            "prompt_family": spec.prompt_family,
            "evaluation_design": spec.evaluation_design,
            "alpha": spec.alpha,
        }
        if spec.eligibility_column is not None:
            result["eligibility_column"] = spec.eligibility_column
            result["eligibility_value"] = 1
        if spec.noninferiority_margin is not None:
            result["noninferiority_margin"] = spec.noninferiority_margin
            result["reporting_rule"] = H2_REPORTING_RULE
        if subset.empty:
            fit_payload = {
                "estimation_method": "unavailable_empty_analysis_subset",
                "n_rows": 0,
                "n_clusters": 0,
                "n_seeds": 0,
                "arm_log_odds_coefficient": None,
                "arm_log_odds_coefficient_ci_95": [None, None],
                "odds_ratio": None,
                "odds_ratio_ci_95": [None, None],
                "marginal_risk_difference": None,
                "marginal_risk_difference_ci_95": [None, None],
                "decision_interval": [None, None],
                "decision_interval_type": "unavailable",
                "raw_p_value": None,
                "direction_supported": False,
                "support_status": "unsupported",
                "unavailable_reason": "analysis_subset_empty",
            }
        else:
            fit_payload = fit_fn(
                subset,
                outcome_column=spec.outcome_column,
                arm_a_id=spec.arm_a_id,
                arm_b_id=spec.arm_b_id,
                alpha=spec.alpha,
                noninferiority_margin=spec.noninferiority_margin,
            )
        result.update(fit_payload)
        if spec.analysis_id in {"analysis_1", "analysis_2"}:
            try:
                paired_reporting[spec.hypothesis_id or spec.analysis_id] = compute_paired_reporting_supplement(
                    subset,
                    outcome_column=spec.outcome_column or "",
                    arm_a_id=spec.arm_a_id or 0,
                    arm_b_id=spec.arm_b_id or 0,
                )
            except ValueError as exc:
                logger.warning(
                    "Skipping paired-reporting supplement for %s: %s",
                    spec.hypothesis_id or spec.analysis_id,
                    exc,
                )
        if spec.classification == "exploratory":
            result["note"] = "Exploratory only; do not treat as a family-wise confirmatory claim."
            exploratory_results.append(result)
        else:
            confirmatory_results.append(result)

    secondary_indices = [
        index
        for index, result in enumerate(confirmatory_results)
        if result.get("hypothesis_id") in SECONDARY_HYPOTHESIS_IDS
    ]
    corrected = apply_holm_correction([confirmatory_results[index] for index in secondary_indices])
    for index, adjusted in zip(secondary_indices, corrected, strict=True):
        confirmatory_results[index] = adjusted

    h1 = next(result for result in confirmatory_results if result["hypothesis_id"] == "H1")
    h2 = next(result for result in confirmatory_results if result["hypothesis_id"] == "H2")
    h1c = next(
        (result for result in confirmatory_results if result["hypothesis_id"] == "H1c"),
        None,
    )
    joint = determine_joint_interpretation(
        h1_supported=h1["support_status"] == "supported",
        h2_supported=h2["support_status"] == "supported",
    )
    construct_validity = build_construct_validity_interpretation(
        h1=h1, h1c=h1c, h2=h2,
    )

    exploratory_results.extend([run_exploratory_e7(df), run_exploratory_e8(df)])
    robustness_results = [
        run_robustness_failure_to_correct(df),
        summarize_semantic_interface_robustness(df),
        build_fixed_vs_semantic_comparison(df),
    ]
    capability_diagnostic_result = run_direct_solve_capability_diagnostics(df)
    schema_invariance_result = build_schema_invariance_analysis(df)
    endorsement_decomp = compute_parseability_endorsement_decomposition(df)
    human_summary = build_human_summary(
        confirmatory_results=confirmatory_results,
        joint_interpretation=joint,
        exclusion_summary=exclusion_summary,
        robustness_results=robustness_results,
        capability_diagnostic_result=capability_diagnostic_result,
        endorsement_decomposition_df=endorsement_decomp,
        construct_validity_interpretation=construct_validity,
        schema_invariance_result=schema_invariance_result,
    )
    return {
        "workflow_name": "preregistered_section_7_analysis",
        "confirmatory_results": confirmatory_results,
        "paired_reporting_supplement": paired_reporting,
        "joint_interpretation": joint,
        "construct_validity_interpretation": construct_validity,
        "schema_invariance": schema_invariance_result,
        "exploratory_results": exploratory_results,
        "robustness_analyses": robustness_results,
        "capability_diagnostic_results": [capability_diagnostic_result],
        "diagnostics": {
            "exclusion_summary_rows": exclusion_summary.to_dict(orient="records"),
            "exclusion_category_rows": exclusion_categories.to_dict(orient="records"),
            "parseability_endorsement_decomposition_rows": endorsement_decomp.to_dict(orient="records"),
            "parseability_endorsement_decomposition_note": _DECOMPOSITION_NOTE,
        },
        "human_summary": human_summary,
    }


def write_outputs(
    payload: dict[str, Any],
    output_prefix: Path,
    *,
    input_path: Path | None = None,
    argv: list[str] | None = None,
) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_prefix.with_suffix(".json")
    summary_path = output_prefix.with_suffix(".summary.txt")
    diagnostics_summary_path = output_prefix.parent / f"{output_prefix.name}{DIAGNOSTIC_SUMMARY_SUFFIX}"
    diagnostics_category_path = output_prefix.parent / f"{output_prefix.name}{DIAGNOSTIC_CATEGORY_SUFFIX}"
    if input_path is not None or argv is not None:
        provenance = build_provenance(
            input_paths=[input_path] if input_path is not None else [],
            argv=argv if argv is not None else [],
            schema_version=PREREG_ANALYSIS_SCHEMA_VERSION,
            repo_root=SCRIPT_DIR,
        )
        write_json_with_provenance(json_path, payload, provenance)
    else:
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    summary_path.write_text(payload["human_summary"], encoding="utf-8")
    diagnostics_summary_df = _cast_nullable_int_columns(
        pd.DataFrame(payload["diagnostics"]["exclusion_summary_rows"]),
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
    diagnostics_category_df = _cast_nullable_int_columns(
        pd.DataFrame(payload["diagnostics"]["exclusion_category_rows"]),
        [
            "arm_id",
            "seed",
            "excluded_category_count",
        ],
    )
    diagnostics_summary_df.to_csv(
        diagnostics_summary_path,
        index=False,
        na_rep="NA",
    )
    diagnostics_category_df.to_csv(
        diagnostics_category_path,
        index=False,
        na_rep="NA",
    )
    decomp_path = output_prefix.parent / f"{output_prefix.name}{ENDORSEMENT_DECOMPOSITION_SUFFIX}"
    decomp_rows = payload["diagnostics"].get("parseability_endorsement_decomposition_rows", [])
    decomp_df = _cast_nullable_int_columns(
        pd.DataFrame(decomp_rows) if decomp_rows else pd.DataFrame(columns=_DECOMPOSITION_COLUMNS),
        ["n_rows", "parseable_count",
         "endorse_incorrect_parseable_count", "endorse_incorrect_overall_count"],
    )
    decomp_df.to_csv(decomp_path, index=False, na_rep="NA")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the preregistered Section 7 analysis suite."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    df = _load_dataframe(args.input.resolve())
    payload = run_preregistration_analyses(df)
    write_outputs(
        payload,
        args.output_prefix.resolve(),
        input_path=args.input.resolve(),
        argv=list(sys.argv),
    )
    return 0
