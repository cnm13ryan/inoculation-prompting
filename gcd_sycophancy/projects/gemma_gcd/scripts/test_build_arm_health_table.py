"""Tests for build_arm_health_table."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_arm_health_table import (  # noqa: E402
    ARM_GROUP_COLS,
    GROUP_COLS,
    build_health_table,
    main,
)

REQUIRED_COLS = [
    "arm_id",
    "arm_slug",
    "arm_label",
    "seed",
    "evaluation_set_name",
    "evaluation_design",
    "prompt_family",
    "is_parseable",
    "is_excluded",
    "exclusion_category",
    "sycophancy_outcome",
    "conditional_sycophancy_eligible",
    "direct_solve_correct",
]


def _row(**overrides):
    base = {
        "arm_id": 1,
        "arm_slug": "neutral_baseline",
        "arm_label": "Neutral baseline",
        "seed": 0,
        "evaluation_set_name": "confirmatory",
        "evaluation_design": "fixed_interface",
        "prompt_family": "incorrect_confirmation",
        "is_parseable": 1,
        "is_excluded": 0,
        "exclusion_category": "",
        "sycophancy_outcome": 0.0,
        "conditional_sycophancy_eligible": 1.0,
        "direct_solve_correct": float("nan"),
    }
    base.update(overrides)
    return base


def _df(rows):
    return pd.DataFrame(rows, columns=REQUIRED_COLS)


def test_parseability_and_exclusion_rates_basic():
    rows = [
        _row(is_parseable=1, is_excluded=0),
        _row(is_parseable=1, is_excluded=0),
        _row(is_parseable=0, is_excluded=1, exclusion_category="degenerate_response", sycophancy_outcome=float("nan")),
        _row(is_parseable=0, is_excluded=1, exclusion_category="unparseable_response", sycophancy_outcome=float("nan")),
    ]
    out = build_health_table(_df(rows), GROUP_COLS)
    assert len(out) == 1
    r = out.iloc[0]
    assert r["n_incorrect_confirmation_rows"] == 4
    assert r["parseability_rate"] == pytest.approx(0.5)
    assert r["exclusion_rate"] == pytest.approx(0.5)
    assert r["degenerate_response_rate"] == pytest.approx(0.25)
    assert r["unparseable_response_rate"] == pytest.approx(0.25)
    assert r["truncated_before_verdict_rate"] == pytest.approx(0.0)
    # Top exclusion is a tie; pandas value_counts breaks tie by first-seen with stable=True.
    assert r["top_exclusion_category"] in {"degenerate_response", "unparseable_response"}
    assert r["top_exclusion_share"] == pytest.approx(0.5)


def test_sycophancy_rate_drops_excluded_nans():
    rows = [
        _row(sycophancy_outcome=1.0),
        _row(sycophancy_outcome=0.0),
        _row(sycophancy_outcome=1.0),
        _row(is_parseable=0, is_excluded=1, exclusion_category="degenerate_response", sycophancy_outcome=float("nan")),
    ]
    out = build_health_table(_df(rows), GROUP_COLS)
    r = out.iloc[0]
    assert r["sycophancy_rate"] == pytest.approx(2 / 3)
    assert r["n_sycophancy_included"] == 3


def test_conditional_sycophancy_filters_eligibility():
    rows = [
        _row(sycophancy_outcome=1.0, conditional_sycophancy_eligible=1.0),
        _row(sycophancy_outcome=0.0, conditional_sycophancy_eligible=0.0),
        _row(sycophancy_outcome=1.0, conditional_sycophancy_eligible=1.0),
    ]
    out = build_health_table(_df(rows), GROUP_COLS)
    r = out.iloc[0]
    assert r["conditional_sycophancy_rate"] == pytest.approx(1.0)
    assert r["n_conditional_sycophancy_eligible"] == 2


def test_conditional_sycophancy_accepts_string_encoding():
    rows = [
        _row(sycophancy_outcome=1.0, conditional_sycophancy_eligible="True"),
        _row(sycophancy_outcome=0.0, conditional_sycophancy_eligible="False"),
    ]
    out = build_health_table(_df(rows), GROUP_COLS)
    r = out.iloc[0]
    assert r["conditional_sycophancy_rate"] == pytest.approx(1.0)
    assert r["n_conditional_sycophancy_eligible"] == 1


def test_direct_solve_rows_join_by_group_key():
    rows = [
        _row(prompt_family="incorrect_confirmation", sycophancy_outcome=1.0),
        _row(prompt_family="direct_solve", sycophancy_outcome=float("nan"), direct_solve_correct=1.0,
             conditional_sycophancy_eligible=0.0),
        _row(prompt_family="direct_solve", sycophancy_outcome=float("nan"), direct_solve_correct=0.0,
             conditional_sycophancy_eligible=0.0),
    ]
    out = build_health_table(_df(rows), GROUP_COLS)
    r = out.iloc[0]
    assert r["n_incorrect_confirmation_rows"] == 1
    assert r["n_direct_solve_rows"] == 2
    assert r["direct_solve_accuracy"] == pytest.approx(0.5)


def test_direct_solve_only_group_emits_nan_sycophancy():
    rows = [
        _row(arm_slug="ds_only", prompt_family="direct_solve",
             sycophancy_outcome=float("nan"), direct_solve_correct=1.0,
             conditional_sycophancy_eligible=0.0),
    ]
    out = build_health_table(_df(rows), GROUP_COLS)
    assert len(out) == 1
    r = out.iloc[0]
    assert r["n_incorrect_confirmation_rows"] == 0
    assert math.isnan(r["sycophancy_rate"])
    assert r["direct_solve_accuracy"] == pytest.approx(1.0)


def test_groups_split_by_seed_and_eval_set():
    rows = [
        _row(seed=0, evaluation_set_name="confirmatory", sycophancy_outcome=1.0),
        _row(seed=1, evaluation_set_name="confirmatory", sycophancy_outcome=0.0),
        _row(seed=0, evaluation_set_name="paraphrase", sycophancy_outcome=1.0),
    ]
    per_seed = build_health_table(_df(rows), GROUP_COLS)
    assert len(per_seed) == 3

    by_arm = build_health_table(_df(rows), ARM_GROUP_COLS)
    # Aggregate across seeds: confirmatory and paraphrase => 2 rows
    assert len(by_arm) == 2
    conf = by_arm[by_arm["evaluation_set_name"] == "confirmatory"].iloc[0]
    assert conf["n_incorrect_confirmation_rows"] == 2
    assert conf["sycophancy_rate"] == pytest.approx(0.5)


def test_main_writes_two_csvs(tmp_path):
    rows = [
        _row(seed=0, sycophancy_outcome=1.0),
        _row(seed=1, sycophancy_outcome=0.0, is_parseable=0, is_excluded=1,
             exclusion_category="degenerate_response"),
        _row(seed=0, prompt_family="direct_solve",
             sycophancy_outcome=float("nan"), direct_solve_correct=1.0,
             conditional_sycophancy_eligible=0.0),
    ]
    csv = tmp_path / "prereg_problem_level_data.csv"
    _df(rows).to_csv(csv, index=False)
    out_prefix = tmp_path / "arm_health_table"
    rc = main(["--problem-level-csv", str(csv), "--output-prefix", str(out_prefix)])
    assert rc == 0
    assert (tmp_path / "arm_health_table.csv").is_file()
    assert (tmp_path / "arm_health_table.by_arm.csv").is_file()
    per_seed = pd.read_csv(tmp_path / "arm_health_table.csv")
    assert "parseability_rate" in per_seed.columns
    assert "direct_solve_accuracy" in per_seed.columns
    assert (per_seed["evaluation_set_name"] == "confirmatory").all()


def test_missing_required_column_raises(tmp_path):
    csv = tmp_path / "broken.csv"
    pd.DataFrame({"arm_id": [1]}).to_csv(csv, index=False)
    with pytest.raises(SystemExit):
        main(["--problem-level-csv", str(csv), "--output-prefix", str(tmp_path / "out")])
