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
) -> dict:
    return {
        "problem_id": problem_id,
        "arm_id": arm_id,
        "arm_label": arm_label,
        "seed": seed,
        "cluster_id": cluster_id,
        "evaluation_set_name": evaluation_set_name,
        "prompt_family": prompt_family,
        "evaluation_design": evaluation_design,
        "selected_prefix_id": selected_prefix_id,
        "is_excluded": 0,
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
    assert "Exploratory analyses E1-E8" in payload["human_summary"]
