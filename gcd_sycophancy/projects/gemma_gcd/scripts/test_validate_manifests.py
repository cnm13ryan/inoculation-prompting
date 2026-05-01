"""Tests for the manifest schemas and the validate_manifests CLI.

For each schema we verify:

* ``Draft202012Validator.check_schema(...)`` accepts it (i.e. the schema
  itself is well-formed).
* When a real fixture exists in this worktree, the schema validates that
  fixture without errors.

Plus one negative test: deleting a required field from a real manifest
must produce at least one validator error.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

# scripts/ -> gemma_gcd/ -> projects/ -> gcd_sycophancy/ (repo project root)
PROJECTS_DIR = Path(__file__).resolve().parents[2]
SCHEMAS_DIR = PROJECTS_DIR / "gemma_gcd" / "schemas"

SCHEMA_FILES = {
    "run_manifest": SCHEMAS_DIR / "run_manifest.schema.json",
    "training_manifest": SCHEMAS_DIR / "training_manifest.schema.json",
    "prereg_data_manifest": SCHEMAS_DIR / "prereg_data_manifest.schema.json",
    "prompt_panel_manifest": SCHEMAS_DIR / "prompt_panel_manifest.schema.json",
}

# Real fixtures that must validate against their schema.
FIXTURES = [
    (
        "prereg_data_manifest",
        PROJECTS_DIR / "gemma_gcd" / "data" / "prereg" / "manifest.json",
    ),
    (
        "training_manifest",
        PROJECTS_DIR
        / "gemma_gcd"
        / "data"
        / "prereg"
        / "arms"
        / "training_manifest.json",
    ),
    (
        "prompt_panel_manifest",
        PROJECTS_DIR
        / "experiments"
        / "_legacy_prepend_above"
        / "prereg_prompt_panel_remaining8"
        / "prompt_panel_manifest.gpu0.json",
    ),
    (
        "prompt_panel_manifest",
        PROJECTS_DIR
        / "experiments"
        / "_legacy_prepend_above"
        / "prereg_prompt_panel_remaining8"
        / "prompt_panel_manifest.gpu1.json",
    ),
]


def _load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Schema well-formedness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("schema_key", sorted(SCHEMA_FILES.keys()))
def test_schema_is_well_formed(schema_key: str) -> None:
    schema_path = SCHEMA_FILES[schema_key]
    assert schema_path.is_file(), f"Schema file missing on disk: {schema_path}"
    schema = _load_json(schema_path)
    # Raises jsonschema.exceptions.SchemaError if malformed.
    Draft202012Validator.check_schema(schema)


# ---------------------------------------------------------------------------
# Fixture validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "schema_key,fixture_path",
    [(key, path) for key, path in FIXTURES],
    ids=[f"{key}::{path.name}" for key, path in FIXTURES],
)
def test_fixture_validates(schema_key: str, fixture_path: Path) -> None:
    assert fixture_path.is_file(), f"Fixture missing on disk: {fixture_path}"
    schema = _load_json(SCHEMA_FILES[schema_key])
    instance = _load_json(fixture_path)

    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(instance), key=lambda e: list(e.absolute_path))
    assert not errors, "Validation errors:\n" + "\n".join(
        f"  - at {'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}"
        for e in errors
    )


# ---------------------------------------------------------------------------
# Run manifest: synthesised inline minimal valid instance
# ---------------------------------------------------------------------------


def test_run_manifest_minimal_inline_instance_validates() -> None:
    schema = _load_json(SCHEMA_FILES["run_manifest"])
    instance = {
        "workflow_name": "synthetic_minimal_workflow",
        "experiment_dir": "/tmp/synthetic-experiment",
        "seeds": [0, 1, 2, 3],
        "phases": {
            "setup": {
                "completed_at_utc": "2026-05-01T00:00:00+00:00",
                "outputs": {"path": "/tmp/synthetic-experiment/setup"},
            },
            "train": {
                "completed_at_utc": "2026-05-01T01:23:45+00:00",
                "outputs": [
                    {"seed": 0, "checkpoint": "step_100"},
                    {"seed": 1, "checkpoint": "step_100"},
                ],
            },
        },
    }
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(instance), key=lambda e: list(e.absolute_path))
    assert not errors, [e.message for e in errors]


# ---------------------------------------------------------------------------
# Negative test: mutating a required field must surface ≥1 error.
# ---------------------------------------------------------------------------


def test_training_manifest_missing_required_field_fails() -> None:
    schema = _load_json(SCHEMA_FILES["training_manifest"])
    fixture_path = (
        PROJECTS_DIR
        / "gemma_gcd"
        / "data"
        / "prereg"
        / "arms"
        / "training_manifest.json"
    )
    instance = copy.deepcopy(_load_json(fixture_path))
    assert isinstance(instance, dict)
    assert "arms" in instance, "Sanity check: fixture should have an arms field to remove."
    del instance["arms"]

    validator = Draft202012Validator(schema)
    errors = list(validator.iter_errors(instance))
    assert len(errors) >= 1, "Expected at least one validator error after deleting 'arms'."
    assert any("arms" in err.message for err in errors), (
        "Expected an error message mentioning the missing 'arms' field; got: "
        + "; ".join(err.message for err in errors)
    )
