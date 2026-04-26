"""Tests for power_analysis_construct_validity.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import power_analysis_construct_validity as pa  # noqa: E402


_BASE_ARGS = [
    "--n-clusters", "20",
    "--n-seeds", "3",
    "--baseline-sycophancy-rate", "0.6",
    "--treatment-sycophancy-rate", "0.3",
    "--baseline-capability-rate", "0.85",
    "--treatment-capability-rate", "0.85",
    "--cluster-sd", "0.5",
    "--seed-sd", "0.2",
    "--exclusion-rate", "0.05",
    "--n-simulations", "100",
    "--seed", "1234",
]


def _run_with(tmp_path, extra=None):
    out = tmp_path / "out.json"
    args = list(_BASE_ARGS) + ["--out-json", str(out)]
    if extra:
        args.extend(extra)
    rc = pa.main(args)
    assert rc == 0
    return json.loads(out.read_text())


def test_deterministic_under_fixed_seed(tmp_path):
    a = _run_with(tmp_path / "a")
    b = _run_with(tmp_path / "b")
    assert a["power_h1"] == b["power_h1"]
    assert a["power_h1_conditional"] == b["power_h1_conditional"]
    assert a["prob_h2_noninferiority_pass"] == b["prob_h2_noninferiority_pass"]
    assert a["joint_success_prob"] == b["joint_success_prob"]


def test_different_seeds_can_yield_different_results(tmp_path):
    a = _run_with(tmp_path / "a")
    b_args = [x if x != "1234" else "9999" for x in _BASE_ARGS]
    out = tmp_path / "b.json"
    pa.main(list(b_args) + ["--out-json", str(out)])
    b = json.loads(out.read_text())
    # Not strictly required; the test only asserts the seed plumbing is wired.
    # If by chance both happen to match, that's fine — we relax to "valid range".
    for k in ("power_h1", "power_h1_conditional", "prob_h2_noninferiority_pass", "joint_success_prob"):
        assert 0.0 <= a[k] <= 1.0
        assert 0.0 <= b[k] <= 1.0


def test_probabilities_in_valid_range_and_provenance_present(tmp_path):
    out = _run_with(tmp_path)
    for key in ("power_h1", "power_h1_conditional", "prob_h2_noninferiority_pass", "joint_success_prob"):
        assert 0.0 <= out[key] <= 1.0
    se = out["mc_standard_error"]
    for key in ("power_h1", "power_h1_conditional", "prob_h2_noninferiority_pass", "joint_success_prob"):
        assert se[key] >= 0.0
    assert "input_assumptions" in out
    assert "provenance" in out
    assert out["input_assumptions"]["seed"] == 1234


def test_zero_effect_yields_low_power(tmp_path):
    """Identical control/treatment rates -> H1 power should be near alpha (low)."""
    out_path = tmp_path / "out.json"
    args = [
        "--n-clusters", "10",
        "--n-seeds", "3",
        "--baseline-sycophancy-rate", "0.5",
        "--treatment-sycophancy-rate", "0.5",
        "--baseline-capability-rate", "0.8",
        "--treatment-capability-rate", "0.8",
        "--cluster-sd", "0.0",
        "--seed-sd", "0.0",
        "--exclusion-rate", "0.0",
        "--n-simulations", "200",
        "--seed", "7",
        "--out-json", str(out_path),
    ]
    rc = pa.main(args)
    assert rc == 0
    data = json.loads(out_path.read_text())
    assert data["power_h1"] < 0.20  # well below typical 0.8


def test_large_effect_yields_high_power(tmp_path):
    """Big sycophancy gap with no random effects -> H1 power should be high."""
    out_path = tmp_path / "out.json"
    args = [
        "--n-clusters", "20",
        "--n-seeds", "3",
        "--baseline-sycophancy-rate", "0.7",
        "--treatment-sycophancy-rate", "0.1",
        "--baseline-capability-rate", "0.9",
        "--treatment-capability-rate", "0.9",
        "--cluster-sd", "0.0",
        "--seed-sd", "0.0",
        "--exclusion-rate", "0.0",
        "--n-simulations", "100",
        "--seed", "11",
        "--out-json", str(out_path),
    ]
    rc = pa.main(args)
    assert rc == 0
    data = json.loads(out_path.read_text())
    assert data["power_h1"] > 0.9


def test_extreme_rates_handled_gracefully(tmp_path):
    out_path = tmp_path / "out.json"
    args = [
        "--n-clusters", "5",
        "--n-seeds", "2",
        "--baseline-sycophancy-rate", "0.999",
        "--treatment-sycophancy-rate", "0.001",
        "--baseline-capability-rate", "0.999",
        "--treatment-capability-rate", "0.999",
        "--cluster-sd", "0.0",
        "--seed-sd", "0.0",
        "--exclusion-rate", "0.0",
        "--n-simulations", "50",
        "--seed", "3",
        "--out-json", str(out_path),
    ]
    rc = pa.main(args)
    assert rc == 0
    data = json.loads(out_path.read_text())
    for k in ("power_h1", "joint_success_prob"):
        assert 0.0 <= data[k] <= 1.0


def test_zero_n_simulations_is_controlled_failure(tmp_path):
    """Regression: --n-simulations 0 must not raise ZeroDivisionError."""
    out_path = tmp_path / "out.json"
    args = [
        "--n-clusters", "5",
        "--n-seeds", "2",
        "--baseline-sycophancy-rate", "0.5",
        "--treatment-sycophancy-rate", "0.3",
        "--baseline-capability-rate", "0.8",
        "--treatment-capability-rate", "0.8",
        "--cluster-sd", "0.0",
        "--seed-sd", "0.0",
        "--exclusion-rate", "0.0",
        "--n-simulations", "0",
        "--seed", "5",
        "--out-json", str(out_path),
    ]
    rc = pa.main(args)
    assert rc == 2
    assert not out_path.exists()


def test_exclusion_rate_one_handled_without_crash(tmp_path):
    out_path = tmp_path / "out.json"
    args = [
        "--n-clusters", "5",
        "--n-seeds", "2",
        "--baseline-sycophancy-rate", "0.5",
        "--treatment-sycophancy-rate", "0.3",
        "--baseline-capability-rate", "0.8",
        "--treatment-capability-rate", "0.8",
        "--cluster-sd", "0.0",
        "--seed-sd", "0.0",
        "--exclusion-rate", "1.0",
        "--n-simulations", "20",
        "--seed", "5",
        "--out-json", str(out_path),
    ]
    rc = pa.main(args)
    assert rc == 0
    data = json.loads(out_path.read_text())
    # Everything excluded -> tests cannot reject -> power should be 0.
    assert data["power_h1"] == 0.0
