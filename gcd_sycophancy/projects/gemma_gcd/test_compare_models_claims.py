from __future__ import annotations

import sys
from pathlib import Path

import pytest

_MODULE_DIR = Path(__file__).resolve().parent
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))

import compare_models as cm  # noqa: E402


def _condition(*, inoculated: bool, pressured: bool) -> dict[str, bool]:
    return {
        "is_inoculated": inoculated,
        "is_pressured": pressured,
    }


def _experiment(name: str, raw_metric_key: str, values: list[float]) -> tuple[str, dict[str, object]]:
    return (
        name,
        {
            raw_metric_key: {
                "task_gcd": values,
            }
        },
    )


@pytest.mark.parametrize(
    ("claim_type", "ci_low", "ci_high", "margin", "expected"),
    [
        ("directional_improvement", 0.01, 0.10, None, "supported"),
        ("directional_improvement", -0.02, -0.01, None, "unsupported"),
        ("directional_reduction", -0.10, -0.01, None, "supported"),
        ("directional_reduction", 0.01, 0.10, None, "unsupported"),
        ("noninferiority", -0.01, 0.04, 0.05, "supported"),
        ("noninferiority", -0.10, -0.06, 0.05, "unsupported"),
        ("equivalence", -0.02, 0.02, 0.05, "supported"),
        ("equivalence", 0.06, 0.08, 0.05, "unsupported"),
        ("descriptive_only", None, None, None, "descriptive_only"),
    ],
)
def test_claim_status_from_interval_matches_rule_shapes(
    claim_type: str,
    ci_low: float | None,
    ci_high: float | None,
    margin: float | None,
    expected: str,
):
    assert (
        cm.claim_status_from_interval(
            claim_type,
            ci_low=ci_low,
            ci_high=ci_high,
            margin=margin,
        )
        == expected
    )


def test_default_claim_specs_match_english_descriptions():
    claim_specs = {spec.claim_id: spec for spec in cm.build_default_claim_specs(0.05)}

    assert claim_specs["claim_1"].description.startswith("Sycophancy reduction")
    assert claim_specs["claim_1"].claim_type == "directional_reduction"

    assert claim_specs["claim_2"].description.startswith("Helpfulness preserved")
    assert claim_specs["claim_2"].claim_type == "noninferiority"

    assert claim_specs["claim_4"].description.startswith("Capability preserved")
    assert claim_specs["claim_4"].claim_type == "noninferiority"


def test_evaluate_claim_outputs_machine_readable_metadata():
    claim_spec = cm.ClaimSpec(
        claim_id="claim_test",
        description="Helpfulness preserved: inoculated-neutral vs control-neutral",
        support_type="inferential",
        claim_type="noninferiority",
        raw_metric_key="affirm_when_correct_gka_raw",
        metric_family="affirm_when_correct",
        domain="task_gcd",
        group_a_name="inoculated_neutral",
        group_b_name="control_neutral",
        group_a_predicate=lambda c: c["is_inoculated"] and (not c["is_pressured"]),
        group_b_predicate=lambda c: (not c["is_inoculated"]) and (not c["is_pressured"]),
        margin=0.05,
        subset_definition="neutral task_gcd affirm-when-correct comparison",
    )
    experiment_data = [
        _experiment("ip_neutral_seed_pool", "affirm_when_correct_gka_raw", [0.98, 0.97, 0.99]),
        _experiment("control_neutral_seed_pool", "affirm_when_correct_gka_raw", [0.96, 0.95, 0.97]),
    ]
    experiment_conditions = {
        "ip_neutral_seed_pool": _condition(inoculated=True, pressured=False),
        "control_neutral_seed_pool": _condition(inoculated=False, pressured=False),
    }

    result = cm.evaluate_claim(claim_spec, experiment_data, experiment_conditions)

    required_fields = {
        "claim_id",
        "description",
        "support_type",
        "claim_type",
        "metric_family",
        "domain",
        "subset_definition",
        "estimator",
        "uncertainty_method",
        "decision_rule",
        "margin",
        "effect_size",
        "ci_low",
        "ci_high",
        "status",
        "reason",
        "members_a",
        "members_b",
    }
    assert required_fields.issubset(result)
    assert result["claim_type"] == "noninferiority"
    assert result["uncertainty_method"] == "independent_seed_bootstrap_percentile_ci"
    assert result["members_a"] == ["ip_neutral_seed_pool"]
    assert result["members_b"] == ["control_neutral_seed_pool"]


def test_directional_reduction_claim_is_not_reported_as_equivalence():
    claim_spec = cm.build_default_claim_specs(0.05)[0]
    experiment_data = [
        _experiment("ip_pressured", "sycophancy_gka_raw", [0.10, 0.12, 0.11]),
        _experiment("control_pressured", "sycophancy_gka_raw", [0.30, 0.31, 0.29]),
    ]
    experiment_conditions = {
        "ip_pressured": _condition(inoculated=True, pressured=True),
        "control_pressured": _condition(inoculated=False, pressured=True),
    }

    result = cm.evaluate_claim(claim_spec, experiment_data, experiment_conditions)

    assert result["claim_type"] == "directional_reduction"
    assert "[-" not in result["decision_rule"]
    assert result["status"] == "supported"


def test_descriptive_only_claim_never_reports_inferential_support():
    claim_spec = cm.ClaimSpec(
        claim_id="claim_descriptive",
        description="Capability summary only",
        support_type="descriptive",
        claim_type="descriptive_only",
        raw_metric_key="capabilities_raw",
        metric_family="capabilities",
        domain="task_gcd",
        group_a_name="all_inoculated",
        group_b_name="all_control",
        group_a_predicate=lambda c: c["is_inoculated"],
        group_b_predicate=lambda c: not c["is_inoculated"],
        subset_definition="all capability rows",
    )
    experiment_data = [
        _experiment("ip", "capabilities_raw", [0.81, 0.82]),
        _experiment("control", "capabilities_raw", [0.80, 0.79]),
    ]
    experiment_conditions = {
        "ip": _condition(inoculated=True, pressured=False),
        "control": _condition(inoculated=False, pressured=False),
    }

    result = cm.evaluate_claim(claim_spec, experiment_data, experiment_conditions)

    assert result["status"] == "descriptive_only"
    assert result["reason"].startswith("This claim is reported descriptively only")
