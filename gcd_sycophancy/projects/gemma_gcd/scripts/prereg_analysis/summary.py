"""Human-readable summary text builder.

Stitches together confirmatory results, joint and construct-validity readings,
exclusion diagnostics, robustness sections, capability diagnostics, schema
invariance, and the parseability/endorsement decomposition.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ._shared import H2_REPORTING_RULE, _format_metric_value
from .exclusion_diagnostics import diagnostics_summary_lines


def _robustness_section_lines(result: dict[str, Any]) -> list[str]:
    analysis_id = result.get("analysis_id")
    label = result.get("label")
    note = result.get("note", "")
    lines = [f"- {analysis_id} ({label}): {note}"]

    if analysis_id == "robustness_R1":
        for arm_row in result.get("rows", []):
            arm_id = arm_row.get("arm_id")
            arm_label = arm_row.get("arm_label")
            rate = arm_row.get("robust_failure_rate")
            n = arm_row.get("evaluated_rows")
            excl_aff = arm_row.get("excluded_affirming_count")
            lines.append(
                f"  arm {arm_id} ({arm_label}): failure_rate={_format_metric_value(rate)}, n={n}, "
                f"excluded_affirming={excl_aff}"
            )
        return lines

    sections = result.get("sections")
    if isinstance(sections, list) and sections:
        for section in sections:
            section_label = section.get("label", "Secondary robustness section")
            metric_key = str(section.get("metric", "rate"))
            metric_label = section.get("metric_label", metric_key)
            lines.append(f"  {section_label}:")
            for row in section.get("rows", []):
                arm_id = row.get("arm_id")
                arm_label = row.get("arm_label")
                evaluation_set_name = row.get("evaluation_set_name", "unknown")
                metric_value = row.get(metric_key)
                n = row.get("evaluated_rows")
                clusters = row.get("evaluated_clusters")
                seeds = row.get("evaluated_seeds")
                lines.append(
                    "  "
                    f"arm {arm_id} ({arm_label}), evaluation_set={evaluation_set_name}: "
                    f"{metric_label}={_format_metric_value(metric_value)}, n={n}, "
                    f"clusters={clusters}, seeds={seeds}"
                )
        semantic_ptst_reminders = result.get("semantic_ptst_reminders", [])
        if semantic_ptst_reminders:
            lines.append(
                "  semantic PTST reminders observed: "
                + " | ".join(str(reminder) for reminder in semantic_ptst_reminders)
            )
        return lines

    rows = result.get("comparison_rows", result.get("rows", []))
    for row in rows:
        arm_id = row.get("arm_id")
        arm_label = row.get("arm_label")
        design = row.get("evaluation_design", "unknown")
        evaluation_set_name = row.get("evaluation_set_name")
        n = row.get("evaluated_rows")
        rate = row.get("sycophancy_rate", row.get("robust_failure_rate"))
        line = (
            f"  arm {arm_id} ({arm_label}), evaluation_design={design}: "
            f"sycophancy_rate={_format_metric_value(rate)}, n={n}"
        )
        if evaluation_set_name is not None:
            line += f", evaluation_set={evaluation_set_name}"
        lines.append(line)
    semantic_ptst_arm_ids = result.get("semantic_ptst_arm_ids", [])
    if semantic_ptst_arm_ids:
        lines.append(
            "  semantic-interface PTST-only arm_ids: "
            + ", ".join(str(arm_id) for arm_id in semantic_ptst_arm_ids)
        )
    return lines


def _capability_diagnostic_lines(result: dict[str, Any]) -> list[str]:
    note = result.get("note", "")
    lines = [f"- {result.get('analysis_id')} ({result.get('label')}): {note}"]
    for row in result.get("rows", []):
        arm_id = row.get("arm_id")
        arm_label = row.get("arm_label")
        seed = row.get("seed")
        eval_set = row.get("evaluation_set_name", "unknown")
        parseability = row.get("parseability_rate")
        accuracy = row.get("direct_solve_accuracy")
        exclusion = row.get("exclusion_rate")
        n = row.get("total_rows")
        clusters = row.get("evaluated_clusters")
        lines.append(
            f"  arm {arm_id} ({arm_label}), seed {seed}, eval_set={eval_set}: "
            f"parseability={_format_metric_value(parseability)}, "
            f"direct_solve_accuracy={_format_metric_value(accuracy)}, "
            f"exclusion_rate={_format_metric_value(exclusion)}, "
            f"n={n}, clusters={clusters}"
        )
    return lines


def _endorsement_decomposition_section_lines(decomp_df: pd.DataFrame | None) -> list[str]:
    lines = [
        "Parseability and Endorsement Decomposition (incorrect_confirmation rows only)",
        "  (endorse_overall treats unparseable rows as non-endorsements; "
        "robust_failure is secondary robustness only)",
    ]
    if decomp_df is None or decomp_df.empty:
        lines.append("  No incorrect_confirmation rows found for decomposition.")
        return lines
    for _, row in decomp_df.iterrows():
        design = row.get("evaluation_design", "unknown")
        arm_id = row.get("arm_id")
        arm_slug = row.get("arm_slug", "unknown")
        seed = row.get("seed")
        eval_set = row.get("evaluation_set_name", "unknown")
        n = row.get("n_rows")
        p_parse = _format_metric_value(row.get("parseability_rate"))
        p_cond = _format_metric_value(row.get("endorse_incorrect_given_parseable_rate"))
        p_overall = _format_metric_value(row.get("endorse_incorrect_overall_rate"))
        robust = row.get("robust_failure_to_correct_rate")
        line = (
            f"  arm {arm_id} ({arm_slug}), seed={seed}, design={design}, set={eval_set}: "
            f"n={n} P(parseable)={p_parse} "
            f"P(endorse|parseable)={p_cond} "
            f"P(endorse_overall)={p_overall}"
        )
        if robust is not None and not pd.isna(robust):
            line += f" P(robust_failure)={_format_metric_value(robust)}"
        lines.append(line)
    return lines


def _schema_invariance_section_lines(result: dict[str, Any]) -> list[str]:
    status = result.get("status", "unavailable")
    label = result.get("label", "Schema invariance")
    note = result.get("note", "")
    lines = [
        f"Schema invariance (secondary robustness; status={status})",
        f"- {label}",
        f"  {note}",
    ]
    if status == "unavailable":
        missing = result.get("missing_designs") or []
        if missing:
            lines.append(f"  missing evaluation_design rows: {', '.join(missing)}")
        return lines
    for section in result.get("sections", []) or []:
        section_label = section.get("label", section.get("section_id", "section"))
        rows = section.get("rows", []) or []
        lines.append(f"  {section_label}:")
        if not rows:
            lines.append("    (no rows)")
            continue
        for row in rows:
            design = row.get("evaluation_design", "unknown")
            arm_id = row.get("arm_id")
            arm_label = row.get("arm_label")
            metric_pairs = []
            for key in (
                "sycophancy_rate",
                "conditional_sycophancy_rate",
                "direct_solve_accuracy",
                "parseability_rate",
                "exclusion_rate",
            ):
                if key in row and row[key] is not None:
                    metric_pairs.append(f"{key}={_format_metric_value(row[key])}")
            eval_set = row.get("evaluation_set_name")
            n = row.get("evaluated_rows", row.get("total_rows"))
            line = f"    arm {arm_id} ({arm_label}), design={design}"
            if eval_set is not None:
                line += f", set={eval_set}"
            if metric_pairs:
                line += ": " + ", ".join(metric_pairs)
            if n is not None:
                line += f", n={n}"
            lines.append(line)
    effect_rows = result.get("effect_direction") or []
    if effect_rows:
        lines.append("  Arm 2 vs Arm 1 effect direction by interface:")
        for row in effect_rows:
            design = row.get("evaluation_design", "unknown")
            direction = row.get("direction", "unavailable")
            gap = row.get("arm_a_minus_arm_b")
            gap_str = "NA" if gap is None else f"{float(gap):+.4f}"
            lines.append(
                f"    design={design}: arm_a-arm_b={gap_str}, direction={direction}"
            )
    return lines


def build_human_summary(
    *,
    confirmatory_results: list[dict[str, Any]],
    joint_interpretation: dict[str, Any],
    exclusion_summary: pd.DataFrame,
    robustness_results: list[dict[str, Any]] | None = None,
    capability_diagnostic_result: dict[str, Any] | None = None,
    endorsement_decomposition_df: pd.DataFrame | None = None,
    construct_validity_interpretation: dict[str, Any] | None = None,
    schema_invariance_result: dict[str, Any] | None = None,
) -> str:
    lines = ["Confirmatory results"]
    for result in confirmatory_results:
        hypothesis_id = result.get("hypothesis_id")
        label = result.get("label")
        status = result.get("support_status")
        risk_diff = result.get("marginal_risk_difference")
        coef = result.get("arm_log_odds_coefficient")
        coef_str = "NA" if coef is None else f"{coef:.4f}"
        risk_diff_str = "NA" if risk_diff is None else f"{risk_diff:.4f}"
        summary_line = (
            f"- {hypothesis_id} ({label}): {status}; log-odds={coef_str}, risk-diff={risk_diff_str}"
        )
        if hypothesis_id == "H1c":
            eligibility_column = result.get("eligibility_column", "conditional_sycophancy_eligible")
            n_rows = result.get("n_rows")
            summary_line = (
                f"{summary_line}; conditional on {eligibility_column}==1, n_eligible_rows={n_rows} "
                "(co-primary, conditional sycophancy)"
            )
        if hypothesis_id == "H2":
            lower_bound = result.get("decision_interval", [None])[0]
            lower_bound_str = "NA" if lower_bound is None else f"{lower_bound:.4f}"
            summary_line = (
                f"{summary_line}; rule={result.get('reporting_rule', H2_REPORTING_RULE)} "
                f"decision_lower_bound={lower_bound_str}"
            )
        lines.append(summary_line)
    lines.append("")
    lines.append("Joint interpretation")
    lines.append(f"- {joint_interpretation['summary']}")
    if construct_validity_interpretation is not None:
        lines.append("")
        lines.append("Construct-validity interpretation (H1 + H1c)")
        lines.append(f"- {construct_validity_interpretation['summary']}")
    lines.append("")
    lines.extend(diagnostics_summary_lines(exclusion_summary))
    lines.append("- Exploratory analyses E1-E8 are reported separately and are explicitly exploratory.")
    if robustness_results:
        lines.append("")
        lines.append(
            "Robustness analyses (exploratory only; not preregistered confirmatory claims)"
        )
        for result in robustness_results:
            lines.extend(_robustness_section_lines(result))
    if capability_diagnostic_result is not None:
        lines.append("")
        lines.append(
            "Direct-solve capability diagnostics (secondary only; not a primary H1-H5 input)"
        )
        lines.extend(_capability_diagnostic_lines(capability_diagnostic_result))
    if schema_invariance_result is not None:
        lines.append("")
        lines.extend(_schema_invariance_section_lines(schema_invariance_result))
    if endorsement_decomposition_df is not None:
        lines.append("")
        lines.extend(_endorsement_decomposition_section_lines(endorsement_decomposition_df))
    return "\n".join(lines)
