"""Tests for the gates package.

Coverage for each of the four gates:
 - registry contract (all four registered, ``run`` dispatches);
 - one passing case;
 - one failing case;
 - the ``--skip-gate`` short-circuit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECTS_DIR = SCRIPT_DIR.parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECTS_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECTS_DIR))

import gates as gates_pkg
from gates._shared import GateResult
import run_preregistration


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_all_four_gates_registered():
    names = set(gates_pkg.registered_gates())
    assert {
        "convergence",
        "fixed_interface_baseline",
        "preflight",
        "fixed_interface_completion",
    } <= names


def test_unknown_gate_name_raises_keyerror():
    cfg = SimpleNamespace(skip_gates=())
    with pytest.raises(KeyError):
        gates_pkg.run("does_not_exist", cfg)


def test_skip_gate_short_circuits_without_running_body():
    # Pass a config with skip_gates entry but missing every other attribute.
    # If the body ran, we'd get an AttributeError.
    cfg = SimpleNamespace(skip_gates=("convergence",))
    result = gates_pkg.run("convergence", cfg)
    assert isinstance(result, GateResult)
    assert result.passed is True
    assert "skipped" in result.reason
    assert result.evidence == {"skipped": True}


def test_skip_gate_only_affects_named_gate():
    cfg = SimpleNamespace(skip_gates=("convergence",))
    result = gates_pkg.run(
        "preflight",
        cfg,
        report={"failures": [{"criterion": "x", "message": "y"}], "passed": False},
    )
    assert result.passed is False


# ---------------------------------------------------------------------------
# convergence gate
# ---------------------------------------------------------------------------


def _write_results_json(condition_dir: Path, seed: int, train_losses: list[float]) -> None:
    seed_dir = condition_dir / f"seed_{seed}" / "results" / "20260101_000000"
    seed_dir.mkdir(parents=True, exist_ok=True)
    (seed_dir / "results.json").write_text(
        json.dumps({"train_losses": train_losses}), encoding="utf-8"
    )


def _make_convergence_stub_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    train_losses_by_slug: dict[str, list[float]],
    threshold: float = 0.15,
    skip_gates: tuple[str, ...] = (),
) -> SimpleNamespace:
    condition_dirs: dict[str, Path] = {}
    for slug in train_losses_by_slug:
        d = tmp_path / slug
        d.mkdir(parents=True, exist_ok=True)
        condition_dirs[slug] = d
        _write_results_json(d, seed=0, train_losses=train_losses_by_slug[slug])

    cfg = SimpleNamespace(
        seeds=(0,),
        preflight_max_final_train_loss=threshold,
        skip_gates=skip_gates,
    )

    monkeypatch.setattr(
        run_preregistration,
        "_validate_seed_configs_exist",
        lambda c: condition_dirs,
    )
    monkeypatch.setattr(
        run_preregistration,
        "_select_only_arm_slugs",
        lambda c, slugs: list(slugs),
    )
    return cfg


def test_convergence_gate_passes_when_all_seeds_below_threshold(monkeypatch, tmp_path):
    cfg = _make_convergence_stub_config(
        monkeypatch,
        tmp_path,
        train_losses_by_slug={"neutral_baseline": [0.5, 0.05]},
    )
    result = gates_pkg.run("convergence", cfg)
    assert result.passed is True
    assert result.evidence["bad_seeds"] == []


def test_convergence_gate_fails_when_seed_does_not_converge(monkeypatch, tmp_path):
    cfg = _make_convergence_stub_config(
        monkeypatch,
        tmp_path,
        train_losses_by_slug={"neutral_baseline": [0.5, 0.30]},
        threshold=0.15,
    )
    result = gates_pkg.run("convergence", cfg)
    assert result.passed is False
    assert result.evidence["bad_seeds"]
    assert result.evidence["bad_seeds"][0]["arm_slug"] == "neutral_baseline"
    assert result.evidence["bad_seeds"][0]["final_loss"] == pytest.approx(0.30)
    assert "did not converge" in result.reason


def test_convergence_gate_skip_returns_synthetic_pass(monkeypatch, tmp_path):
    cfg = _make_convergence_stub_config(
        monkeypatch,
        tmp_path,
        train_losses_by_slug={"neutral_baseline": [0.5, 0.30]},  # would fail
        skip_gates=("convergence",),
    )
    result = gates_pkg.run("convergence", cfg)
    assert result.passed is True
    assert result.reason.startswith("skipped via --skip-gate")


# ---------------------------------------------------------------------------
# fixed_interface_baseline gate
# ---------------------------------------------------------------------------


def test_fixed_interface_baseline_gate_passes_with_clean_report(monkeypatch):
    report = {
        "max_format_failure_rate": 0.10,
        "allow_unacceptable_fixed_interface_for_prefix_search": False,
        "unacceptable_assessments": [],
        "assessments": [],
    }
    monkeypatch.setattr(
        run_preregistration,
        "_load_or_create_fixed_interface_baseline_report",
        lambda c: report,
    )
    cfg = SimpleNamespace(
        allow_unacceptable_fixed_interface_for_prefix_search=False,
        skip_gates=(),
    )
    result = gates_pkg.run("fixed_interface_baseline", cfg)
    assert result.passed is True
    assert result.override_used is False
    assert result.evidence["report"] is report
    assert result.evidence["raw_gate_passed"] is True


def test_fixed_interface_baseline_gate_fails_without_override(monkeypatch):
    report = {
        "max_format_failure_rate": 0.10,
        "allow_unacceptable_fixed_interface_for_prefix_search": False,
        "unacceptable_assessments": [
            {
                "arm_slug": "neutral_baseline",
                "seed": 0,
                "unacceptable_datasets": ["test_confirmatory"],
                "worst_dataset": {
                    "dataset_name": "test_confirmatory",
                    "format_failure_rate": 0.42,
                },
            }
        ],
        "assessments": [],
    }
    monkeypatch.setattr(
        run_preregistration,
        "_load_or_create_fixed_interface_baseline_report",
        lambda c: report,
    )
    cfg = SimpleNamespace(
        allow_unacceptable_fixed_interface_for_prefix_search=False,
        skip_gates=(),
    )
    result = gates_pkg.run("fixed_interface_baseline", cfg)
    assert result.passed is False
    assert result.override_used is False
    assert "Fixed-interface baseline quality is unacceptable" in result.reason


def test_fixed_interface_baseline_gate_passes_with_override(monkeypatch):
    report = {
        "max_format_failure_rate": 0.10,
        "allow_unacceptable_fixed_interface_for_prefix_search": True,
        "unacceptable_assessments": [
            {
                "arm_slug": "neutral_baseline",
                "seed": 0,
                "unacceptable_datasets": ["test_confirmatory"],
                "worst_dataset": {
                    "dataset_name": "test_confirmatory",
                    "format_failure_rate": 0.42,
                },
            }
        ],
        "assessments": [],
    }
    monkeypatch.setattr(
        run_preregistration,
        "_load_or_create_fixed_interface_baseline_report",
        lambda c: report,
    )
    cfg = SimpleNamespace(
        allow_unacceptable_fixed_interface_for_prefix_search=True,
        skip_gates=(),
    )
    result = gates_pkg.run("fixed_interface_baseline", cfg)
    assert result.passed is True
    assert result.override_used is True
    assert result.evidence["raw_gate_passed"] is False  # underlying still bad


# ---------------------------------------------------------------------------
# preflight gate
# ---------------------------------------------------------------------------


def test_preflight_gate_passes_when_no_failures():
    cfg = SimpleNamespace(skip_gates=())
    report = {"passed": True, "failures": []}
    result = gates_pkg.run("preflight", cfg, report=report)
    assert result.passed is True
    assert result.evidence["failures"] == []


def test_preflight_gate_fails_when_failures_present():
    cfg = SimpleNamespace(skip_gates=())
    report = {
        "passed": False,
        "failures": [
            {"criterion": "overall_exclusion_rate", "message": "exceeded threshold"}
        ],
    }
    result = gates_pkg.run("preflight", cfg, report=report)
    assert result.passed is False
    assert "overall_exclusion_rate" in result.reason


# ---------------------------------------------------------------------------
# fixed_interface_completion gate
# ---------------------------------------------------------------------------


def test_fixed_interface_completion_gate_passes_when_all_present(monkeypatch, tmp_path):
    condition_dirs = {"neutral_baseline": tmp_path / "neutral_baseline"}
    condition_dirs["neutral_baseline"].mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        run_preregistration,
        "_validate_seed_configs_exist",
        lambda c: condition_dirs,
    )
    monkeypatch.setattr(
        run_preregistration,
        "_select_only_arm_slugs",
        lambda c, slugs: list(slugs),
    )
    monkeypatch.setattr(
        run_preregistration,
        "_fixed_interface_output_dir",
        lambda c, condition_dir, seed: condition_dir / f"seed_{seed}",
    )
    monkeypatch.setattr(run_preregistration, "_has_results", lambda d: True)

    cfg = SimpleNamespace(seeds=(0,), skip_gates=())
    result = gates_pkg.run("fixed_interface_completion", cfg)
    assert result.passed is True
    assert result.evidence["missing"] == []


# ---------------------------------------------------------------------------
# Skipped fixed_interface_baseline → _prefix_search_gate_status alias
# must not KeyError when iterating evidence keys that the gate body never
# populated. Regression test for PR #99 review.
# ---------------------------------------------------------------------------


def test_prefix_search_gate_status_alias_handles_skipped_gate(monkeypatch):
    # No monkeypatch on _load_or_create_fixed_interface_baseline_report —
    # if the gate body executed, it would crash on the missing helper.
    # The skip path must short-circuit before any evidence lookup.
    cfg = SimpleNamespace(
        skip_gates=("fixed_interface_baseline",),
        allow_unacceptable_fixed_interface_for_prefix_search=False,
    )
    status = run_preregistration._prefix_search_gate_status(cfg)
    # Legacy four keys must all be present; report must NOT be indexed
    # for "assessments" without a guard.
    assert status["gate_passed"] is True
    assert status["override_used"] is False
    assert status["report"] is None
    assert status["message"] is None
    # New optional flag the alias adds so callers can detect the skip case.
    assert status.get("skipped") is True


def test_prefix_search_gate_status_alias_runs_normally_when_not_skipped(monkeypatch):
    """The non-skip path still produces the legacy four-key shape."""
    report = {
        "max_format_failure_rate": 0.10,
        "allow_unacceptable_fixed_interface_for_prefix_search": False,
        "unacceptable_assessments": [],
        "assessments": [{"arm_slug": "neutral_baseline", "seed": 0}],
    }
    monkeypatch.setattr(
        run_preregistration,
        "_load_or_create_fixed_interface_baseline_report",
        lambda c: report,
    )
    cfg = SimpleNamespace(
        skip_gates=(),
        allow_unacceptable_fixed_interface_for_prefix_search=False,
    )
    status = run_preregistration._prefix_search_gate_status(cfg)
    assert status["gate_passed"] is True
    assert status["override_used"] is False
    assert status["report"] is report
    assert status["message"] is None
    assert status.get("skipped") is False


def test_fixed_interface_completion_gate_fails_when_missing(monkeypatch, tmp_path):
    condition_dirs = {"neutral_baseline": tmp_path / "neutral_baseline"}
    condition_dirs["neutral_baseline"].mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        run_preregistration,
        "_validate_seed_configs_exist",
        lambda c: condition_dirs,
    )
    monkeypatch.setattr(
        run_preregistration,
        "_select_only_arm_slugs",
        lambda c, slugs: list(slugs),
    )
    monkeypatch.setattr(
        run_preregistration,
        "_fixed_interface_output_dir",
        lambda c, condition_dir, seed: condition_dir / f"seed_{seed}",
    )
    monkeypatch.setattr(run_preregistration, "_has_results", lambda d: False)

    cfg = SimpleNamespace(seeds=(0,), skip_gates=())
    result = gates_pkg.run("fixed_interface_completion", cfg)
    assert result.passed is False
    assert result.evidence["missing"]
    assert "are required before bounded prefix search" in result.reason
