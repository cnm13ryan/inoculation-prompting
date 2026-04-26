"""Tests for analyze_exclusion_sensitivity (WT-14)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECTS_DIR = SCRIPT_DIR.parents[1]
for _p in (SCRIPT_DIR, PROJECTS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import analyze_exclusion_sensitivity as module


def _row(
    *,
    arm_id: int,
    is_excluded: int,
    sycoph: int | None,
    semantic: int | None = None,
) -> dict:
    return {
        "arm_id": arm_id,
        "arm_label": f"Arm {arm_id}",
        "prompt_family": "incorrect_confirmation",
        "is_excluded": is_excluded,
        "sycophancy_outcome": sycoph,
        "semantic_affirms_user": semantic,
    }


def _df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


class TestStrictIncludedOnly:
    def test_drops_all_excluded(self):
        df = _df([
            _row(arm_id=1, is_excluded=0, sycoph=1),
            _row(arm_id=1, is_excluded=0, sycoph=0),
            _row(arm_id=1, is_excluded=1, sycoph=None),
            _row(arm_id=2, is_excluded=0, sycoph=0),
            _row(arm_id=2, is_excluded=0, sycoph=0),
            _row(arm_id=2, is_excluded=1, sycoph=None),
        ])
        result = module.compute_scenario(df, "strict_included_only")
        by_arm = {r["arm_id"]: r for r in result["arm_rows"]}
        assert by_arm[1]["sycophancy_rate"] == 0.5
        assert by_arm[2]["sycophancy_rate"] == 0.0
        assert by_arm[1]["n_rows"] == 2
        assert result["risk_difference_treatment_minus_control"] == -0.5


class TestExcludedAsSycophantic:
    def test_excluded_count_as_one(self):
        df = _df([
            _row(arm_id=1, is_excluded=0, sycoph=0),
            _row(arm_id=1, is_excluded=1, sycoph=None),
            _row(arm_id=2, is_excluded=0, sycoph=0),
            _row(arm_id=2, is_excluded=1, sycoph=None),
        ])
        result = module.compute_scenario(df, "excluded_as_sycophantic")
        by_arm = {r["arm_id"]: r for r in result["arm_rows"]}
        assert by_arm[1]["sycophancy_rate"] == 0.5
        assert by_arm[2]["sycophancy_rate"] == 0.5


class TestExcludedAsNonSycophantic:
    def test_excluded_count_as_zero(self):
        df = _df([
            _row(arm_id=1, is_excluded=0, sycoph=1),
            _row(arm_id=1, is_excluded=1, sycoph=None),
            _row(arm_id=2, is_excluded=0, sycoph=1),
            _row(arm_id=2, is_excluded=1, sycoph=None),
        ])
        result = module.compute_scenario(df, "excluded_as_non_sycophantic")
        by_arm = {r["arm_id"]: r for r in result["arm_rows"]}
        assert by_arm[1]["sycophancy_rate"] == 0.5
        assert by_arm[2]["sycophancy_rate"] == 0.5


class TestExcludedAffirmingScenario:
    def test_uses_semantic_affirms_user(self):
        df = _df([
            _row(arm_id=1, is_excluded=0, sycoph=0),
            _row(arm_id=1, is_excluded=1, sycoph=None, semantic=1),
            _row(arm_id=1, is_excluded=1, sycoph=None, semantic=0),
            _row(arm_id=2, is_excluded=0, sycoph=0),
            _row(arm_id=2, is_excluded=1, sycoph=None, semantic=0),
        ])
        result = module.compute_scenario(
            df,
            "excluded_affirming_as_sycophantic_if_semantic_affirms_user_available",
        )
        by_arm = {r["arm_id"]: r for r in result["arm_rows"]}
        assert by_arm[1]["sycophancy_rate"] == pytest_approx(1 / 3)
        assert by_arm[2]["sycophancy_rate"] == 0.0
        assert result["semantic_affirms_user_available"] is True

    def test_missing_semantic_field_drops_excluded(self):
        df = _df([
            _row(arm_id=1, is_excluded=0, sycoph=1),
            _row(arm_id=1, is_excluded=1, sycoph=None, semantic=None),
            _row(arm_id=2, is_excluded=0, sycoph=0),
            _row(arm_id=2, is_excluded=1, sycoph=None, semantic=None),
        ])
        result = module.compute_scenario(
            df,
            "excluded_affirming_as_sycophantic_if_semantic_affirms_user_available",
        )
        by_arm = {r["arm_id"]: r for r in result["arm_rows"]}
        assert by_arm[1]["n_rows"] == 1
        assert by_arm[2]["n_rows"] == 1
        assert by_arm[1]["sycophancy_rate"] == 1.0
        assert result["semantic_affirms_user_available"] is False

    def test_column_absent_entirely(self):
        df = pd.DataFrame([
            {"arm_id": 1, "arm_label": "A1", "prompt_family": "incorrect_confirmation",
             "is_excluded": 0, "sycophancy_outcome": 1},
            {"arm_id": 2, "arm_label": "A2", "prompt_family": "incorrect_confirmation",
             "is_excluded": 0, "sycophancy_outcome": 0},
        ])
        result = module.compute_scenario(
            df,
            "excluded_affirming_as_sycophantic_if_semantic_affirms_user_available",
        )
        assert result["semantic_affirms_user_available"] is False


def pytest_approx(value: float, tol: float = 1e-9):
    class _A:
        def __init__(self, v):
            self.v = v

        def __eq__(self, other):
            return abs(other - self.v) < tol

        def __repr__(self):
            return f"~{self.v}"
    return _A(value)


class TestWorstCaseScenarios:
    def test_worst_case_for_treatment(self):
        df = _df([
            _row(arm_id=1, is_excluded=0, sycoph=1),
            _row(arm_id=1, is_excluded=1, sycoph=None),
            _row(arm_id=2, is_excluded=0, sycoph=0),
            _row(arm_id=2, is_excluded=1, sycoph=None),
        ])
        result = module.compute_scenario(df, "worst_case_for_treatment")
        by_arm = {r["arm_id"]: r for r in result["arm_rows"]}
        # Control excluded -> 0; treatment excluded -> 1
        assert by_arm[1]["sycophancy_rate"] == 0.5  # 1/2 (1 included syc=1, 1 excluded=0)
        assert by_arm[2]["sycophancy_rate"] == 0.5  # 0 included + 1 from excluded = 1/2
        assert result["risk_difference_treatment_minus_control"] == 0.0

    def test_worst_case_for_control(self):
        df = _df([
            _row(arm_id=1, is_excluded=0, sycoph=0),
            _row(arm_id=1, is_excluded=1, sycoph=None),
            _row(arm_id=2, is_excluded=0, sycoph=1),
            _row(arm_id=2, is_excluded=1, sycoph=None),
        ])
        result = module.compute_scenario(df, "worst_case_for_control")
        by_arm = {r["arm_id"]: r for r in result["arm_rows"]}
        assert by_arm[1]["sycophancy_rate"] == 0.5  # 0 included + 1 excluded counts as syc
        assert by_arm[2]["sycophancy_rate"] == 0.5  # 1 included + 1 excluded counts as 0
        assert result["risk_difference_treatment_minus_control"] == 0.0


class TestEdgeCases:
    def test_no_exclusions_data(self):
        df = _df([
            _row(arm_id=1, is_excluded=0, sycoph=1),
            _row(arm_id=1, is_excluded=0, sycoph=0),
            _row(arm_id=2, is_excluded=0, sycoph=0),
            _row(arm_id=2, is_excluded=0, sycoph=0),
        ])
        for scenario in module.SCENARIOS:
            result = module.compute_scenario(df, scenario)
            by_arm = {r["arm_id"]: r for r in result["arm_rows"]}
            assert by_arm[1]["sycophancy_rate"] == 0.5
            assert by_arm[2]["sycophancy_rate"] == 0.0
            assert result["risk_difference_treatment_minus_control"] == -0.5

    def test_all_excluded_strict(self):
        df = _df([
            _row(arm_id=1, is_excluded=1, sycoph=None),
            _row(arm_id=2, is_excluded=1, sycoph=None),
        ])
        result = module.compute_scenario(df, "strict_included_only")
        by_arm = {r["arm_id"]: r for r in result["arm_rows"]}
        assert by_arm[1]["sycophancy_rate"] is None
        assert by_arm[2]["sycophancy_rate"] is None
        assert result["risk_difference_treatment_minus_control"] is None

    def test_all_excluded_excluded_as_sycophantic(self):
        df = _df([
            _row(arm_id=1, is_excluded=1, sycoph=None),
            _row(arm_id=2, is_excluded=1, sycoph=None),
        ])
        result = module.compute_scenario(df, "excluded_as_sycophantic")
        by_arm = {r["arm_id"]: r for r in result["arm_rows"]}
        assert by_arm[1]["sycophancy_rate"] == 1.0
        assert by_arm[2]["sycophancy_rate"] == 1.0

    def test_no_incorrect_confirmation_rows(self):
        df = pd.DataFrame([
            {"arm_id": 1, "arm_label": "A1", "prompt_family": "direct_solve",
             "is_excluded": 0, "sycophancy_outcome": None},
        ])
        result = module.compute_scenario(df, "strict_included_only")
        assert result["arm_rows"] == []
        assert result["risk_difference_treatment_minus_control"] is None


class TestPayloadAndOutputs:
    def test_payload_has_all_six_scenarios(self):
        df = _df([
            _row(arm_id=1, is_excluded=0, sycoph=1),
            _row(arm_id=2, is_excluded=0, sycoph=0),
        ])
        payload = module.build_exclusion_sensitivity_payload(df)
        assert len(payload["scenarios"]) == 6
        names = {s["scenario"] for s in payload["scenarios"]}
        assert names == set(module.SCENARIOS)

    def test_write_outputs_creates_files_with_provenance(self, tmp_path: Path):
        df = _df([
            _row(arm_id=1, is_excluded=0, sycoph=1),
            _row(arm_id=1, is_excluded=1, sycoph=None, semantic=1),
            _row(arm_id=2, is_excluded=0, sycoph=0),
            _row(arm_id=2, is_excluded=1, sycoph=None, semantic=0),
        ])
        csv_in = tmp_path / "data.csv"
        df.to_csv(csv_in, index=False)
        payload = module.build_exclusion_sensitivity_payload(df)
        prefix = tmp_path / "out" / "exclusion_sensitivity"
        paths = module.write_outputs(
            payload, prefix, input_paths=[csv_in], argv=["x"]
        )
        assert paths["json"].exists()
        assert paths["csv"].exists()
        assert paths["md"].exists()
        doc = json.loads(paths["json"].read_text())
        assert "provenance" in doc
        assert doc["provenance"]["schema_version"] == module.EXCLUSION_SENSITIVITY_SCHEMA_VERSION
        # CSV has 6 scenarios * 2 arms = 12 rows
        csv_df = pd.read_csv(paths["csv"])
        assert len(csv_df) == 12


class TestCLI:
    def test_main_end_to_end(self, tmp_path: Path):
        df = _df([
            _row(arm_id=1, is_excluded=0, sycoph=1),
            _row(arm_id=1, is_excluded=1, sycoph=None, semantic=1),
            _row(arm_id=2, is_excluded=0, sycoph=0),
            _row(arm_id=2, is_excluded=1, sycoph=None, semantic=0),
        ])
        csv_in = tmp_path / "in.csv"
        df.to_csv(csv_in, index=False)
        prefix = tmp_path / "out" / "exclusion_sensitivity"
        rc = module.main([
            "--problem-level-csv", str(csv_in),
            "--output-prefix", str(prefix),
        ])
        assert rc == 0
        assert (prefix.parent / "exclusion_sensitivity.json").exists()
