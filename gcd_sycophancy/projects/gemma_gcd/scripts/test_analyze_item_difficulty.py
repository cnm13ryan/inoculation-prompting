"""Tests for analyze_item_difficulty (WT-13)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECTS_DIR = SCRIPT_DIR.parents[1]
for _p in (SCRIPT_DIR, PROJECTS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import analyze_item_difficulty as module


def _direct_solve_row(
    *,
    cluster_id: str,
    arm_id: int,
    seed: int,
    correct: int,
    is_excluded: int = 0,
) -> dict:
    return {
        "cluster_id": cluster_id,
        "arm_id": arm_id,
        "arm_label": f"Arm {arm_id}",
        "seed": seed,
        "prompt_family": "direct_solve",
        "is_excluded": is_excluded,
        "direct_solve_correct": correct,
        "sycophancy_outcome": None,
    }


def _ic_row(
    *,
    cluster_id: str,
    arm_id: int,
    seed: int,
    sycoph: int,
    is_excluded: int = 0,
) -> dict:
    return {
        "cluster_id": cluster_id,
        "arm_id": arm_id,
        "arm_label": f"Arm {arm_id}",
        "seed": seed,
        "prompt_family": "incorrect_confirmation",
        "is_excluded": is_excluded,
        "direct_solve_correct": None,
        "sycophancy_outcome": sycoph,
    }


def _make_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


class TestAssignBand:
    def test_easy_when_above_easy_threshold(self):
        cfg = module.DifficultyConfig()
        assert module.assign_band(0.9, 10, cfg) == module.BAND_EASY

    def test_hard_when_below_hard_threshold(self):
        cfg = module.DifficultyConfig()
        assert module.assign_band(0.2, 10, cfg) == module.BAND_HARD

    def test_medium_when_between_thresholds(self):
        cfg = module.DifficultyConfig()
        assert module.assign_band(0.55, 10, cfg) == module.BAND_MEDIUM

    def test_insufficient_when_too_few_observations(self):
        cfg = module.DifficultyConfig(min_observations=4)
        assert module.assign_band(0.9, 2, cfg) == module.BAND_INSUFFICIENT

    def test_insufficient_when_accuracy_missing(self):
        cfg = module.DifficultyConfig()
        assert module.assign_band(None, 10, cfg) == module.BAND_INSUFFICIENT


class TestComputeClusterDifficulty:
    def test_aggregates_across_arms_and_seeds(self):
        rows = []
        for arm in (1, 2):
            for seed in (0, 1):
                rows.append(
                    _direct_solve_row(
                        cluster_id="C1", arm_id=arm, seed=seed, correct=1
                    )
                )
        df = _make_df(rows)
        result = module.compute_cluster_difficulty(
            df, module.DifficultyConfig(min_observations=2)
        )
        assert len(result) == 1
        assert result.iloc[0]["cluster_id"] == "C1"
        assert result.iloc[0]["direct_solve_accuracy"] == 1.0
        assert result.iloc[0]["n_observations"] == 4
        assert result.iloc[0]["difficulty_band"] == module.BAND_EASY

    def test_band_assignment_per_cluster(self):
        rows = []
        # Easy cluster
        for arm in (1, 2):
            for seed in (0, 1, 2):
                rows.append(_direct_solve_row(cluster_id="E", arm_id=arm, seed=seed, correct=1))
        # Hard cluster
        for arm in (1, 2):
            for seed in (0, 1, 2):
                rows.append(_direct_solve_row(cluster_id="H", arm_id=arm, seed=seed, correct=0))
        df = _make_df(rows)
        result = module.compute_cluster_difficulty(
            df, module.DifficultyConfig(min_observations=4)
        )
        bands = dict(zip(result["cluster_id"], result["difficulty_band"]))
        assert bands["E"] == module.BAND_EASY
        assert bands["H"] == module.BAND_HARD

    def test_excluded_rows_dropped(self):
        rows = [
            _direct_solve_row(cluster_id="C1", arm_id=1, seed=0, correct=1),
            _direct_solve_row(cluster_id="C1", arm_id=1, seed=1, correct=0, is_excluded=1),
        ]
        df = _make_df(rows)
        result = module.compute_cluster_difficulty(
            df, module.DifficultyConfig(min_observations=1)
        )
        assert int(result.iloc[0]["n_observations"]) == 1
        assert result.iloc[0]["direct_solve_accuracy"] == 1.0

    def test_no_direct_solve_returns_empty(self):
        rows = [_ic_row(cluster_id="C1", arm_id=1, seed=0, sycoph=0)]
        df = _make_df(rows)
        result = module.compute_cluster_difficulty(df)
        assert result.empty

    def test_insufficient_observations_band(self):
        rows = [_direct_solve_row(cluster_id="C1", arm_id=1, seed=0, correct=1)]
        df = _make_df(rows)
        result = module.compute_cluster_difficulty(
            df, module.DifficultyConfig(min_observations=4)
        )
        assert result.iloc[0]["difficulty_band"] == module.BAND_INSUFFICIENT


class TestSycophancyByBand:
    def test_rates_split_by_band_and_arm(self):
        rows = []
        # Easy cluster (E) — high direct-solve accuracy
        for arm in (1, 2):
            for seed in (0, 1, 2):
                rows.append(_direct_solve_row(cluster_id="E", arm_id=arm, seed=seed, correct=1))
        # Hard cluster (H) — low direct-solve accuracy
        for arm in (1, 2):
            for seed in (0, 1, 2):
                rows.append(_direct_solve_row(cluster_id="H", arm_id=arm, seed=seed, correct=0))
        # IC rows: arm 1 sycophs more on hard, arm 2 less
        for seed in (0, 1, 2, 3):
            rows.append(_ic_row(cluster_id="E", arm_id=1, seed=seed, sycoph=0))
            rows.append(_ic_row(cluster_id="E", arm_id=2, seed=seed, sycoph=0))
            rows.append(_ic_row(cluster_id="H", arm_id=1, seed=seed, sycoph=1))
            rows.append(_ic_row(cluster_id="H", arm_id=2, seed=seed, sycoph=0))
        df = _make_df(rows)
        cluster = module.compute_cluster_difficulty(
            df, module.DifficultyConfig(min_observations=4)
        )
        sb = module.compute_sycophancy_by_band_and_arm(df, cluster)
        easy_arm1 = sb[
            (sb["difficulty_band"] == module.BAND_EASY) & (sb["arm_id"] == 1)
        ].iloc[0]
        hard_arm1 = sb[
            (sb["difficulty_band"] == module.BAND_HARD) & (sb["arm_id"] == 1)
        ].iloc[0]
        hard_arm2 = sb[
            (sb["difficulty_band"] == module.BAND_HARD) & (sb["arm_id"] == 2)
        ].iloc[0]
        assert easy_arm1["sycophancy_rate"] == 0.0
        assert hard_arm1["sycophancy_rate"] == 1.0
        assert hard_arm2["sycophancy_rate"] == 0.0

    def test_empty_when_no_incorrect_confirmation_rows(self):
        rows = [
            _direct_solve_row(cluster_id="C1", arm_id=1, seed=s, correct=1) for s in range(4)
        ]
        df = _make_df(rows)
        cluster = module.compute_cluster_difficulty(
            df, module.DifficultyConfig(min_observations=4)
        )
        sb = module.compute_sycophancy_by_band_and_arm(df, cluster)
        assert sb.empty

    def test_missing_direct_solve_marks_band_insufficient(self):
        # IC rows present but no direct-solve rows for that cluster
        rows = [_ic_row(cluster_id="X", arm_id=1, seed=s, sycoph=1) for s in range(4)]
        rows += [
            _direct_solve_row(cluster_id="Y", arm_id=1, seed=s, correct=1) for s in range(4)
        ]
        df = _make_df(rows)
        cluster = module.compute_cluster_difficulty(
            df, module.DifficultyConfig(min_observations=4)
        )
        sb = module.compute_sycophancy_by_band_and_arm(df, cluster)
        # Cluster X has no direct-solve, so its rows should land in
        # insufficient_data band after the merge.
        bands = set(sb["difficulty_band"])
        assert module.BAND_INSUFFICIENT in bands


class TestPayloadAndOutputs:
    def test_payload_shape(self):
        rows = [
            _direct_solve_row(cluster_id="C1", arm_id=arm, seed=seed, correct=1)
            for arm in (1, 2)
            for seed in (0, 1, 2)
        ]
        df = _make_df(rows)
        payload = module.build_item_difficulty_payload(df)
        assert payload["workflow_name"] == "item_difficulty_calibration"
        assert "cluster_difficulty" in payload
        assert "sycophancy_by_band_and_arm" in payload
        assert "band_summary" in payload
        assert payload["configuration"]["easy_threshold"] == module.DEFAULT_EASY_THRESHOLD

    def test_write_outputs_creates_three_files_with_provenance(self, tmp_path: Path):
        rows = [
            _direct_solve_row(cluster_id="C1", arm_id=arm, seed=seed, correct=1)
            for arm in (1, 2)
            for seed in (0, 1, 2)
        ]
        df = _make_df(rows)
        csv_input = tmp_path / "prereg_problem_level_data.csv"
        df.to_csv(csv_input, index=False)
        payload = module.build_item_difficulty_payload(df)
        prefix = tmp_path / "out" / "item_difficulty"
        paths = module.write_outputs(
            payload,
            prefix,
            input_paths=[csv_input],
            argv=["analyze_item_difficulty.py"],
        )
        assert paths["json"].exists()
        assert paths["csv"].exists()
        assert paths["md"].exists()
        json_doc = json.loads(paths["json"].read_text())
        assert "provenance" in json_doc
        assert json_doc["provenance"]["schema_version"] == module.ITEM_DIFFICULTY_SCHEMA_VERSION
        assert json_doc["provenance"]["input_files"][0]["path"] == str(csv_input)


class TestCLI:
    def test_main_end_to_end(self, tmp_path: Path):
        rows = [
            _direct_solve_row(cluster_id="C1", arm_id=arm, seed=seed, correct=1)
            for arm in (1, 2)
            for seed in (0, 1, 2)
        ]
        rows += [
            _direct_solve_row(cluster_id="C2", arm_id=arm, seed=seed, correct=0)
            for arm in (1, 2)
            for seed in (0, 1, 2)
        ]
        rows += [
            _ic_row(cluster_id="C2", arm_id=1, seed=s, sycoph=1) for s in range(4)
        ] + [
            _ic_row(cluster_id="C2", arm_id=2, seed=s, sycoph=0) for s in range(4)
        ]
        df = pd.DataFrame(rows)
        csv_input = tmp_path / "in.csv"
        df.to_csv(csv_input, index=False)
        prefix = tmp_path / "out" / "item_difficulty"
        rc = module.main([
            "--problem-level-csv", str(csv_input),
            "--output-prefix", str(prefix),
            "--min-observations", "2",
        ])
        assert rc == 0
        json_doc = json.loads((prefix.parent / "item_difficulty.json").read_text())
        assert json_doc["band_summary"]
        assert any(
            b["difficulty_band"] == module.BAND_HARD for b in json_doc["band_summary"]
        )
