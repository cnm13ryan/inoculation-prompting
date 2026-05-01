"""Tests for the QW3 per-stage manifest writes.

``_record_phase`` was extended to write a per-stage manifest at
``manifests/stage_<phase>.json`` alongside the aggregate
``manifests/run_manifest.json`` it has always written. The aggregate
file's shape and contents are unchanged; the per-stage file is
purely additive.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator

SCRIPT_DIR = Path(__file__).resolve().parent
# scripts/ -> gemma_gcd/ -> projects/. SCRIPT_DIR.parents[1] is projects/;
# parents[2] would overshoot to gcd_sycophancy/.
PROJECTS_DIR = SCRIPT_DIR.parents[1]
SCHEMAS_DIR = PROJECTS_DIR / "gemma_gcd" / "schemas"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECTS_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECTS_DIR))

import run_preregistration


def _stub_config(experiment_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        experiment_dir=experiment_dir,
        seeds=(0, 1),
    )


# ---------------------------------------------------------------------------
# Path helper
# ---------------------------------------------------------------------------


def test_stage_manifest_path_replaces_hyphens_with_underscores(tmp_path):
    cfg = _stub_config(tmp_path)
    p = run_preregistration._stage_manifest_path(cfg, "materialize-data")
    assert p == tmp_path / "manifests" / "stage_materialize_data.json"


def test_stage_manifest_path_for_simple_phase(tmp_path):
    cfg = _stub_config(tmp_path)
    p = run_preregistration._stage_manifest_path(cfg, "train")
    assert p == tmp_path / "manifests" / "stage_train.json"


# ---------------------------------------------------------------------------
# _record_phase: writes both aggregate and per-stage manifests
# ---------------------------------------------------------------------------


def test_record_phase_writes_per_stage_manifest_alongside_aggregate(tmp_path):
    cfg = _stub_config(tmp_path)
    run_preregistration._record_phase(cfg, "setup", {"arm_count": 6, "seed_count_per_arm": 4})

    stage_path = tmp_path / "manifests" / "stage_setup.json"
    aggregate_path = tmp_path / "manifests" / "run_manifest.json"
    assert stage_path.is_file(), "stage_setup.json must be written"
    assert aggregate_path.is_file(), "run_manifest.json must still be written"

    stage = json.loads(stage_path.read_text())
    assert stage["phase"] == "setup"
    assert stage["outputs"] == {"arm_count": 6, "seed_count_per_arm": 4}
    assert "completed_at_utc" in stage

    aggregate = json.loads(aggregate_path.read_text())
    assert "phases" in aggregate
    assert "setup" in aggregate["phases"]
    assert aggregate["phases"]["setup"]["outputs"] == {"arm_count": 6, "seed_count_per_arm": 4}


def test_record_phase_aggregate_and_stage_share_completed_at_utc(tmp_path):
    cfg = _stub_config(tmp_path)
    run_preregistration._record_phase(cfg, "train", {"trained_arms": ["neutral_baseline"]})

    stage = json.loads((tmp_path / "manifests" / "stage_train.json").read_text())
    aggregate = json.loads((tmp_path / "manifests" / "run_manifest.json").read_text())
    assert stage["completed_at_utc"] == aggregate["phases"]["train"]["completed_at_utc"]


def test_record_phase_for_hyphenated_phase_uses_underscore_filename(tmp_path):
    cfg = _stub_config(tmp_path)
    run_preregistration._record_phase(cfg, "fixed-interface-eval", {"evaluated_arms": 6})

    stage_path = tmp_path / "manifests" / "stage_fixed_interface_eval.json"
    assert stage_path.is_file()
    stage = json.loads(stage_path.read_text())
    # Phase name retains hyphens INSIDE the JSON (matches the canonical CLI subcommand);
    # only the filename uses underscores.
    assert stage["phase"] == "fixed-interface-eval"


def test_record_phase_overwrites_stage_manifest_on_rerun(tmp_path):
    """If a phase runs twice (e.g. operator rebuilds), the per-stage
    manifest reflects the latest run, not the first one. Aggregate
    behaviour is unchanged from before this PR."""
    cfg = _stub_config(tmp_path)
    run_preregistration._record_phase(cfg, "setup", {"arm_count": 6})
    run_preregistration._record_phase(cfg, "setup", {"arm_count": 7})  # second run

    stage = json.loads((tmp_path / "manifests" / "stage_setup.json").read_text())
    assert stage["outputs"] == {"arm_count": 7}


def test_record_multiple_phases_writes_distinct_stage_files(tmp_path):
    cfg = _stub_config(tmp_path)
    run_preregistration._record_phase(cfg, "materialize-data", {"step": 1})
    run_preregistration._record_phase(cfg, "setup", {"step": 2})
    run_preregistration._record_phase(cfg, "train", {"step": 3})

    manifests_dir = tmp_path / "manifests"
    files = sorted(p.name for p in manifests_dir.glob("stage_*.json"))
    assert files == [
        "stage_materialize_data.json",
        "stage_setup.json",
        "stage_train.json",
    ]


# ---------------------------------------------------------------------------
# Schema validation: every per-stage manifest written by _record_phase
# must validate against the stage_manifest schema.
# ---------------------------------------------------------------------------


def test_written_stage_manifest_validates_against_schema(tmp_path):
    cfg = _stub_config(tmp_path)
    run_preregistration._record_phase(cfg, "preflight", {"passed": True})

    schema = json.loads((SCHEMAS_DIR / "stage_manifest.schema.json").read_text())
    instance = json.loads((tmp_path / "manifests" / "stage_preflight.json").read_text())

    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(instance), key=lambda e: list(e.absolute_path))
    assert not errors, "stage manifest must validate against stage_manifest.schema.json: " + "; ".join(
        e.message for e in errors
    )


def test_stage_manifest_schema_is_well_formed():
    schema = json.loads((SCHEMAS_DIR / "stage_manifest.schema.json").read_text())
    Draft202012Validator.check_schema(schema)


def test_stage_manifest_schema_rejects_missing_required_fields(tmp_path):
    """Negative coverage: dropping a required field must surface an error."""
    schema = json.loads((SCHEMAS_DIR / "stage_manifest.schema.json").read_text())
    bad = {"completed_at_utc": "2026-05-01T00:00:00+00:00", "outputs": {}}  # missing 'phase'

    validator = Draft202012Validator(schema)
    errors = list(validator.iter_errors(bad))
    assert len(errors) >= 1
    assert any("phase" in e.message for e in errors)


# ---------------------------------------------------------------------------
# Failure-mode atomicity: PR #102 review regression tests.
#
# The reviewer flagged that writing the stage file BEFORE loading the
# aggregate left a partial state on aggregate-read failures: stage file
# claims the phase completed, aggregate disagrees, downstream auditing
# is misled. The fix reorders to (1) load aggregate, (2) write aggregate,
# (3) write stage. These tests pin that ordering by simulating each
# failure point.
# ---------------------------------------------------------------------------


def test_record_phase_aborts_before_writing_stage_when_aggregate_unreadable(tmp_path):
    """Pre-populate ``run_manifest.json`` with corrupt JSON so the load
    raises. The stage file MUST NOT exist after the failed call."""
    cfg = _stub_config(tmp_path)
    manifests_dir = tmp_path / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    (manifests_dir / "run_manifest.json").write_text("{ this is not valid json")

    with pytest.raises(json.JSONDecodeError):
        run_preregistration._record_phase(cfg, "setup", {"arm_count": 6})

    stage_path = manifests_dir / "stage_setup.json"
    assert not stage_path.exists(), (
        "Stage file must not be written when aggregate read fails — that "
        "would leave the per-stage audit asserting completion while the "
        "aggregate disagrees."
    )


def test_record_phase_aborts_before_writing_stage_when_aggregate_load_raises(monkeypatch, tmp_path):
    """Stub ``_load_run_manifest`` to raise. The stage file MUST NOT exist
    after the failed call. Same regression as the previous test, but with
    an arbitrary load error rather than a JSON parse error specifically.
    """
    cfg = _stub_config(tmp_path)

    def boom(_):
        raise RuntimeError("simulated aggregate load failure")

    monkeypatch.setattr(run_preregistration, "_load_run_manifest", boom)

    with pytest.raises(RuntimeError, match="simulated aggregate load failure"):
        run_preregistration._record_phase(cfg, "train", {"trained_arms": []})

    stage_path = tmp_path / "manifests" / "stage_train.json"
    assert not stage_path.exists(), (
        "Stage file must not be written when aggregate load raises."
    )


def test_record_phase_writes_aggregate_before_stage_so_stage_failure_leaves_aggregate_correct(
    monkeypatch, tmp_path
):
    """Aggregate is the source of truth: if the per-stage write fails,
    the aggregate must already reflect the completed phase. Simulated
    by stubbing ``_write_json`` to raise on the stage path only."""
    cfg = _stub_config(tmp_path)
    aggregate_path = tmp_path / "manifests" / "run_manifest.json"
    stage_path = tmp_path / "manifests" / "stage_setup.json"

    real_write_json = run_preregistration._write_json

    def selective_write_json(path, payload):
        if path == stage_path:
            raise OSError("simulated stage-file write failure")
        return real_write_json(path, payload)

    monkeypatch.setattr(run_preregistration, "_write_json", selective_write_json)

    with pytest.raises(OSError, match="simulated stage-file write failure"):
        run_preregistration._record_phase(cfg, "setup", {"arm_count": 6})

    # Aggregate must already record the phase (write happened before stage).
    assert aggregate_path.is_file()
    aggregate = json.loads(aggregate_path.read_text())
    assert "setup" in aggregate.get("phases", {})
    assert aggregate["phases"]["setup"]["outputs"] == {"arm_count": 6}
    # Stage file must not exist (its write was the call that raised).
    assert not stage_path.exists()
