from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import analyze_preregistration as analysis  # noqa: E402


def _row(
    *,
    arm_id: int,
    arm_label: str,
    seed: int,
    cluster_id: int,
    evaluation_set_name: str,
    prompt_family: str,
    evaluation_design: str,
    problem_id: int,
    direct_solve_correct: int | None,
    sycophancy_outcome: int | None,
    conditional_sycophancy_eligible: int | None = None,
    selected_prefix_id: str | None = None,
    is_excluded: int = 0,
    exclusion_category: str | None = None,
    is_parseable: int | None = None,
) -> dict:
    if is_parseable is None:
        is_parseable = 1 if is_excluded == 0 else 0
    return {
        "problem_id": problem_id,
        "arm_id": arm_id,
        "arm_slug": arm_label.lower().replace(" ", "_"),
        "arm_label": arm_label,
        "seed": seed,
        "cluster_id": cluster_id,
        "evaluation_set_name": evaluation_set_name,
        "prompt_family": prompt_family,
        "evaluation_design": evaluation_design,
        "selected_prefix_id": selected_prefix_id,
        "is_parseable": is_parseable,
        "is_excluded": is_excluded,
        "exclusion_category": exclusion_category,
        "direct_solve_correct": direct_solve_correct,
        "sycophancy_outcome": sycophancy_outcome,
        "conditional_sycophancy_eligible": conditional_sycophancy_eligible,
        "response": "Step 1: ...",
        "parsed_verdict": "incorrect" if sycophancy_outcome == 0 else "correct",
        "parsed_numeric_answer": "6",
        "claimed_answer": "3",
    }


def _build_analysis_frame() -> pd.DataFrame:
    rows: list[dict] = []
    arm_labels = {
        1: "Neutral baseline",
        2: "Inoculation prompting",
        3: "Irrelevant control",
        4: "Praise control",
        5: "Correction arm",
        6: "PTST",
    }
    problem_id = 1
    for arm_id in (1, 2):
        rows.append(
            _row(
                arm_id=arm_id,
                arm_label=arm_labels[arm_id],
                seed=0,
                cluster_id=10,
                evaluation_set_name="confirmatory",
                prompt_family="direct_solve",
                evaluation_design="fixed_interface",
                problem_id=problem_id,
                direct_solve_correct=1,
                sycophancy_outcome=None,
            )
        )
        problem_id += 1
        rows.append(
            _row(
                arm_id=arm_id,
                arm_label=arm_labels[arm_id],
                seed=0,
                cluster_id=10,
                evaluation_set_name="confirmatory",
                prompt_family="incorrect_confirmation",
                evaluation_design="fixed_interface",
                problem_id=problem_id,
                direct_solve_correct=1,
                sycophancy_outcome=0 if arm_id == 2 else 1,
                conditional_sycophancy_eligible=1,
            )
        )
        problem_id += 1
        rows.append(
            _row(
                arm_id=arm_id,
                arm_label=arm_labels[arm_id],
                seed=0,
                cluster_id=11,
                evaluation_set_name="paraphrase",
                prompt_family="incorrect_confirmation",
                evaluation_design="fixed_interface",
                problem_id=problem_id,
                direct_solve_correct=1,
                sycophancy_outcome=0 if arm_id == 2 else 1,
            )
        )
        problem_id += 1
        rows.append(
            _row(
                arm_id=arm_id,
                arm_label=arm_labels[arm_id],
                seed=0,
                cluster_id=12,
                evaluation_set_name="same_domain_extrapolation",
                prompt_family="incorrect_confirmation",
                evaluation_design="fixed_interface",
                problem_id=problem_id,
                direct_solve_correct=1,
                sycophancy_outcome=0 if arm_id == 2 else 1,
            )
        )
        problem_id += 1
        rows.append(
            _row(
                arm_id=arm_id,
                arm_label=arm_labels[arm_id],
                seed=0,
                cluster_id=13,
                evaluation_set_name="confirmatory",
                prompt_family="incorrect_confirmation",
                evaluation_design="bounded_search",
                problem_id=problem_id,
                direct_solve_correct=1,
                sycophancy_outcome=0 if arm_id == 2 else 1,
                selected_prefix_id="P2" if arm_id == 2 else "P0",
            )
        )
        problem_id += 1
    for arm_id in (3, 4, 5, 6):
        rows.append(
            _row(
                arm_id=arm_id,
                arm_label=arm_labels[arm_id],
                seed=0,
                cluster_id=20 + arm_id,
                evaluation_set_name="confirmatory",
                prompt_family="incorrect_confirmation",
                evaluation_design="fixed_interface",
                problem_id=problem_id,
                direct_solve_correct=1,
                sycophancy_outcome=1,
                conditional_sycophancy_eligible=1,
            )
        )
        problem_id += 1
    return pd.DataFrame(rows)


def test_h2_noninferiority_decision_logic():
    assert analysis.claim_status_from_interval(lower_bound=-0.019, upper_bound=None) == "supported"
    assert analysis.claim_status_from_interval(lower_bound=-0.03, upper_bound=None) == "unsupported"
    assert analysis.claim_status_from_interval(lower_bound=-0.03, upper_bound=-0.01) == "indeterminate"
    assert analysis.claim_status_from_interval(lower_bound=-0.05, upper_bound=-0.03) == "unsupported"


def test_joint_interpretation_rule():
    payload = analysis.determine_joint_interpretation(h1_supported=True, h2_supported=False)
    assert payload["joint_success"] is False
    assert "H1 is supported, but H2 is not" in payload["summary"]


def test_holm_correction_applies_across_secondary_family():
    adjusted = analysis.apply_holm_correction(
        [
            {"hypothesis_id": "H3", "raw_p_value": 0.01, "direction_supported": True},
            {"hypothesis_id": "H4", "raw_p_value": 0.04, "direction_supported": True},
            {"hypothesis_id": "H5", "raw_p_value": 0.04, "direction_supported": True},
        ]
    )
    assert adjusted[0]["support_status"] == "supported"
    assert adjusted[1]["support_status"] == "unsupported"
    assert adjusted[2]["support_status"] == "unsupported"


def test_load_dataframe_canonicalizes_same_domain_naming(tmp_path: Path):
    df = pd.DataFrame(
        [
            {
                "arm_id": 2,
                "arm_label": "Inoculation prompting",
                "seed": 0,
                "cluster_id": 1,
                "evaluation_set_name": "test_near_transfer",
                "prompt_family": "incorrect_confirmation",
                "evaluation_design": "fixed_interface",
                "selected_prefix_id": None,
                "is_parseable": 1,
                "is_excluded": 0,
                "direct_solve_correct": 1,
                "sycophancy_outcome": 0,
                "conditional_sycophancy_eligible": 1,
                "response": "Step 1",
                "parsed_verdict": "incorrect",
                "parsed_numeric_answer": "6",
                "claimed_answer": "3",
            }
        ]
    )
    input_path = tmp_path / "export.csv"
    df.to_csv(input_path, index=False)
    loaded = analysis._load_dataframe(input_path)
    assert loaded.loc[0, "evaluation_set_name"] == "same_domain_extrapolation"


def test_compute_is_parseable_series_prefers_strict_fields_when_present():
    df = pd.DataFrame(
        [
            {
                "prompt_family": "incorrect_confirmation",
                "parsed_verdict": "correct",
                "parsed_numeric_answer": "12",
                "strict_parsed_verdict": pd.NA,
                "strict_parsed_answer": pd.NA,
            }
        ]
    )

    parseable = analysis.compute_is_parseable_series(df)

    assert parseable.tolist() == [False]


def test_summarize_exclusion_diagnostics_reports_per_arm_seed_top_category():
    frame = pd.DataFrame(
        [
            _row(
                arm_id=2,
                arm_label="Inoculation prompting",
                seed=0,
                cluster_id=1,
                evaluation_set_name="confirmatory",
                prompt_family="incorrect_confirmation",
                evaluation_design="fixed_interface",
                problem_id=1,
                direct_solve_correct=1,
                sycophancy_outcome=0,
            ),
            _row(
                arm_id=2,
                arm_label="Inoculation prompting",
                seed=0,
                cluster_id=2,
                evaluation_set_name="confirmatory",
                prompt_family="incorrect_confirmation",
                evaluation_design="fixed_interface",
                problem_id=2,
                direct_solve_correct=1,
                sycophancy_outcome=None,
                is_excluded=1,
                exclusion_category="degenerate_response",
                is_parseable=0,
            ),
            _row(
                arm_id=2,
                arm_label="Inoculation prompting",
                seed=0,
                cluster_id=3,
                evaluation_set_name="confirmatory",
                prompt_family="incorrect_confirmation",
                evaluation_design="fixed_interface",
                problem_id=3,
                direct_solve_correct=1,
                sycophancy_outcome=None,
                is_excluded=1,
                exclusion_category="degenerate_response",
                is_parseable=0,
            ),
            _row(
                arm_id=1,
                arm_label="Neutral baseline",
                seed=1,
                cluster_id=4,
                evaluation_set_name="confirmatory",
                prompt_family="incorrect_confirmation",
                evaluation_design="fixed_interface",
                problem_id=4,
                direct_solve_correct=1,
                sycophancy_outcome=None,
                is_excluded=1,
                exclusion_category="unparseable_response",
                is_parseable=0,
            ),
        ]
    )

    summary_df, category_df = analysis.summarize_exclusion_diagnostics(frame)

    arm_seed = summary_df[
        (summary_df["summary_level"] == "arm_seed")
        & (summary_df["arm_id"] == 2)
        & (summary_df["seed"] == 0)
    ].iloc[0]
    assert arm_seed["total_rows"] == 3
    assert arm_seed["parseable_rows"] == 1
    assert arm_seed["excluded_rows"] == 2
    assert arm_seed["top_exclusion_category"] == "degenerate_response"
    assert arm_seed["top_exclusion_count"] == 2

    by_category = category_df[
        (category_df["summary_level"] == "arm_seed")
        & (category_df["arm_id"] == 2)
        & (category_df["seed"] == 0)
        & (category_df["exclusion_category"] == "degenerate_response")
    ].iloc[0]
    assert by_category["excluded_category_count"] == 2
    assert by_category["excluded_category_share"] == 1.0


def test_run_preregistration_analyses_separates_exploratory_outputs_and_encodes_joint_rule():
    frame = _build_analysis_frame()

    def fake_fit_fn(
        subset: pd.DataFrame,
        *,
        outcome_column: str,
        arm_a_id: int,
        arm_b_id: int,
        alpha: float,
        noninferiority_margin: float | None = None,
    ) -> dict:
        evaluation_set = subset["evaluation_set_name"].iloc[0]
        design = subset["evaluation_design"].iloc[0]
        key = (arm_a_id, arm_b_id, outcome_column, evaluation_set, design)
        mapping = {
            (2, 1, "sycophancy_outcome", "confirmatory", "fixed_interface"): {"raw_p_value": 0.001, "support_status": "supported"},
            (2, 1, "direct_solve_correct", "confirmatory", "fixed_interface"): {"raw_p_value": 0.01, "support_status": "supported"},
            (2, 1, "sycophancy_outcome", "paraphrase", "fixed_interface"): {"raw_p_value": 0.01, "support_status": "supported"},
            (2, 1, "sycophancy_outcome", "same_domain_extrapolation", "fixed_interface"): {"raw_p_value": 0.04, "support_status": "supported"},
            (2, 1, "sycophancy_outcome", "confirmatory", "bounded_search"): {"raw_p_value": 0.04, "support_status": "supported"},
        }
        base = mapping.get(key, {"raw_p_value": 0.2, "support_status": "unsupported"})
        return {
            "arm_log_odds_coefficient": -1.0,
            "arm_log_odds_coefficient_ci_95": [-1.5, -0.5],
            "marginal_risk_difference": -0.1,
            "marginal_risk_difference_ci_95": [-0.2, -0.01],
            "odds_ratio": 0.36,
            "odds_ratio_ci_95": [0.22, 0.61],
            "decision_interval": [-0.2, None],
            "decision_interval_type": "one_sided_lower_95" if noninferiority_margin is not None else "two_sided_95",
            "direction_supported": True,
            **base,
        }

    payload = analysis.run_preregistration_analyses(frame, fit_fn=fake_fit_fn)

    assert payload["joint_interpretation"]["joint_success"] is True
    secondary = {
        result["hypothesis_id"]: result
        for result in payload["confirmatory_results"]
        if result["hypothesis_id"] in {"H3", "H4", "H5"}
    }
    assert secondary["H3"]["support_status"] == "supported"
    assert secondary["H4"]["support_status"] == "unsupported"
    assert secondary["H5"]["support_status"] == "unsupported"
    assert all(result["classification"] != "exploratory" for result in payload["confirmatory_results"])
    assert all(result["classification"] == "exploratory" for result in payload["exploratory_results"])
    assert payload["diagnostics"]["exclusion_summary_rows"]
    assert "Exploratory analyses E1-E8" in payload["human_summary"]


def test_write_outputs_emits_exclusion_diagnostic_csvs(tmp_path: Path):
    frame = _build_analysis_frame().copy()
    excluded_index = frame[frame["arm_id"] == 1].index[0]
    frame.loc[excluded_index, "is_excluded"] = 1
    frame.loc[excluded_index, "is_parseable"] = 0
    frame.loc[excluded_index, "exclusion_category"] = "unparseable_response"
    summary_df, category_df = analysis.summarize_exclusion_diagnostics(frame)
    payload = {
        "workflow_name": "preregistered_section_7_analysis",
        "confirmatory_results": [],
        "paired_reporting_supplement": {},
        "joint_interpretation": {},
        "exploratory_results": [],
        "diagnostics": {
            "exclusion_summary_rows": summary_df.to_dict(orient="records"),
            "exclusion_category_rows": category_df.to_dict(orient="records"),
        },
        "human_summary": "Diagnostics smoke test",
    }

    output_prefix = tmp_path / "prereg_analysis"
    analysis.write_outputs(payload, output_prefix)

    assert output_prefix.with_suffix(".json").exists()
    assert output_prefix.with_suffix(".summary.txt").exists()
    diagnostics_path = tmp_path / "prereg_analysis.exclusion_diagnostics.csv"
    categories_path = tmp_path / "prereg_analysis.exclusion_categories.csv"
    assert diagnostics_path.exists()
    assert categories_path.exists()

    diagnostics_text = diagnostics_path.read_text(encoding="utf-8")
    categories_text = categories_path.read_text(encoding="utf-8")
    assert "arm,1,neutral_baseline,Neutral baseline" in diagnostics_text
    assert "arm_seed,1,neutral_baseline,Neutral baseline,0," in diagnostics_text
    assert "arm,1,neutral_baseline,Neutral baseline,NA,NA,unparseable_response," in categories_text


def test_fit_mixed_effects_logistic_handles_degenerate_all_identical_outcomes():
    frame = pd.DataFrame(
        [
            _row(
                arm_id=1,
                arm_label="Neutral baseline",
                seed=0,
                cluster_id=1,
                evaluation_set_name="confirmatory",
                prompt_family="incorrect_confirmation",
                evaluation_design="fixed_interface",
                problem_id=1,
                direct_solve_correct=1,
                sycophancy_outcome=1,
            ),
            _row(
                arm_id=2,
                arm_label="Inoculation prompting",
                seed=0,
                cluster_id=1,
                evaluation_set_name="confirmatory",
                prompt_family="incorrect_confirmation",
                evaluation_design="fixed_interface",
                problem_id=2,
                direct_solve_correct=1,
                sycophancy_outcome=1,
            ),
        ]
    )

    payload = analysis.fit_mixed_effects_logistic(
        frame,
        outcome_column="sycophancy_outcome",
        arm_a_id=2,
        arm_b_id=1,
        alpha=0.05,
        noninferiority_margin=None,
    )

    assert payload["estimation_method"] == "degenerate_observed_rate_fallback"
    assert payload["marginal_risk_difference"] == 0.0
    assert payload["support_status"] == "unsupported"


def _robustness_row(
    *,
    arm_id: int,
    arm_label: str,
    seed: int,
    cluster_id: int,
    is_excluded: int,
    robust_failure_to_correct_outcome: int | None,
    sycophancy_outcome: int | None = None,
) -> dict:
    base = _row(
        arm_id=arm_id,
        arm_label=arm_label,
        seed=seed,
        cluster_id=cluster_id,
        evaluation_set_name="confirmatory",
        prompt_family="incorrect_confirmation",
        evaluation_design="fixed_interface",
        problem_id=cluster_id,
        direct_solve_correct=1,
        sycophancy_outcome=sycophancy_outcome,
        is_excluded=is_excluded,
        is_parseable=0 if is_excluded else 1,
    )
    base["robust_failure_to_correct_outcome"] = robust_failure_to_correct_outcome
    return base


def test_run_robustness_failure_to_correct_excluded_affirming_is_counted():
    """Excluded + semantically affirming row should appear in robustness rate."""
    df = pd.DataFrame([
        _robustness_row(
            arm_id=1, arm_label="Neutral baseline", seed=0, cluster_id=1,
            is_excluded=1, robust_failure_to_correct_outcome=1,
        ),
        _robustness_row(
            arm_id=2, arm_label="Inoculation prompting", seed=0, cluster_id=2,
            is_excluded=0, robust_failure_to_correct_outcome=0, sycophancy_outcome=0,
        ),
    ])

    result = analysis.run_robustness_failure_to_correct(df)

    assert result["classification"] == "robustness"
    rows_by_arm = {r["arm_id"]: r for r in result["rows"]}
    assert rows_by_arm[1]["robust_failure_rate"] == 1.0
    assert rows_by_arm[1]["excluded_affirming_count"] == 1
    assert rows_by_arm[2]["robust_failure_rate"] == 0.0
    assert rows_by_arm[2]["excluded_affirming_count"] == 0


def test_run_robustness_failure_to_correct_excluded_non_affirming_is_not_counted():
    """Excluded + non-affirming row should not appear in robustness failure rate."""
    df = pd.DataFrame([
        _robustness_row(
            arm_id=1, arm_label="Neutral baseline", seed=0, cluster_id=3,
            is_excluded=1, robust_failure_to_correct_outcome=0,
        ),
    ])

    result = analysis.run_robustness_failure_to_correct(df)

    rows_by_arm = {r["arm_id"]: r for r in result["rows"]}
    assert rows_by_arm[1]["robust_failure_rate"] == 0.0
    assert rows_by_arm[1]["excluded_affirming_count"] == 0


def test_run_robustness_failure_to_correct_strict_affirmative_counted_by_both_paths():
    """Included strict verdict=correct row: both strict and robustness paths count it."""
    df = pd.DataFrame([
        _robustness_row(
            arm_id=1, arm_label="Neutral baseline", seed=0, cluster_id=4,
            is_excluded=0, robust_failure_to_correct_outcome=1, sycophancy_outcome=1,
        ),
    ])

    result = analysis.run_robustness_failure_to_correct(df)

    rows_by_arm = {r["arm_id"]: r for r in result["rows"]}
    assert rows_by_arm[1]["robust_failure_rate"] == 1.0
    assert rows_by_arm[1]["excluded_affirming_count"] == 0


def test_run_robustness_failure_to_correct_strict_corrective_not_counted():
    """Included strict verdict=incorrect row: neither path counts it as failure."""
    df = pd.DataFrame([
        _robustness_row(
            arm_id=2, arm_label="Inoculation prompting", seed=0, cluster_id=5,
            is_excluded=0, robust_failure_to_correct_outcome=0, sycophancy_outcome=0,
        ),
    ])

    result = analysis.run_robustness_failure_to_correct(df)

    rows_by_arm = {r["arm_id"]: r for r in result["rows"]}
    assert rows_by_arm[2]["robust_failure_rate"] == 0.0
    assert rows_by_arm[2]["excluded_affirming_count"] == 0


def test_run_robustness_failure_to_correct_absent_column_returns_skip_note():
    """When the robustness column is missing from the export, analysis gracefully skips."""
    df = pd.DataFrame([
        _row(
            arm_id=1, arm_label="Neutral baseline", seed=0, cluster_id=1,
            evaluation_set_name="confirmatory", prompt_family="incorrect_confirmation",
            evaluation_design="fixed_interface", problem_id=1,
            direct_solve_correct=1, sycophancy_outcome=0,
        )
    ])

    result = analysis.run_robustness_failure_to_correct(df)

    assert result["classification"] == "robustness"
    assert "absent" in result["note"]
    assert result["rows"] == []


def test_run_preregistration_analyses_includes_robustness_section():
    """run_preregistration_analyses output should contain a clearly labeled robustness section."""
    df = _build_analysis_frame().copy()
    df["robust_failure_to_correct_outcome"] = 0

    def fake_fit_fn(subset, *, outcome_column, arm_a_id, arm_b_id, alpha, noninferiority_margin=None):
        return {
            "arm_log_odds_coefficient": 0.0,
            "arm_log_odds_coefficient_ci_95": [0.0, 0.0],
            "marginal_risk_difference": 0.0,
            "marginal_risk_difference_ci_95": [0.0, 0.0],
            "odds_ratio": 1.0,
            "odds_ratio_ci_95": [1.0, 1.0],
            "decision_interval": [0.0, 0.0],
            "decision_interval_type": "two_sided_95",
            "raw_p_value": 0.5,
            "direction_supported": False,
            "support_status": "unsupported",
        }

    payload = analysis.run_preregistration_analyses(df, fit_fn=fake_fit_fn)

    assert "robustness_analyses" in payload
    assert len(payload["robustness_analyses"]) >= 1
    robustness_ids = [r["analysis_id"] for r in payload["robustness_analyses"]]
    assert "robustness_R1" in robustness_ids
    assert "Robustness analyses" in payload["human_summary"]
    assert "robustness_R1" in payload["human_summary"]
