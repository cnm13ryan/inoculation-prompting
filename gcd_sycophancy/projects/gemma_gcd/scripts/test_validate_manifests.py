"""Tests for the manifest schemas and the validate_manifests CLI.

For each schema we verify:

* ``Draft202012Validator.check_schema(...)`` accepts it (i.e. the schema
  itself is well-formed).
* When a real fixture exists in this worktree, the schema validates that
  fixture without errors.

Plus negative tests:

* Deleting a required field from a real manifest must produce >=1 validator error.
* The basename-pattern registry must NOT match unrelated co-located JSON files
  (e.g. ``prompt_panel_manifest.schema.json``) — a previous over-broad glob
  ``prompt_panel_manifest.*.json`` would have matched the schema file itself.
* Running the CLI on a directory containing no canonical manifests must exit
  non-zero by default, and exit zero only when ``--allow-empty`` is passed.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

# Make the script importable for the registry / CLI tests below.
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
import validate_manifests  # noqa: E402  (sys.path tweak above)


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


# ---------------------------------------------------------------------------
# Registry: pattern matching boundaries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "basename,expected_schema",
    [
        # canonical filenames -> their schema
        ("run_manifest.json", "run_manifest.schema.json"),
        ("training_manifest.json", "training_manifest.schema.json"),
        ("prereg_data_manifest.json", "prereg_data_manifest.schema.json"),
        ("prompt_panel_manifest.json", "prompt_panel_manifest.schema.json"),
        # split variants must match
        ("prompt_panel_manifest.gpu0.json", "prompt_panel_manifest.schema.json"),
        ("prompt_panel_manifest.gpu1.json", "prompt_panel_manifest.schema.json"),
        ("prompt_panel_manifest.gpu10.json", "prompt_panel_manifest.schema.json"),
    ],
)
def test_schema_for_matches_canonical_basenames(
    basename: str, expected_schema: str
) -> None:
    assert validate_manifests._schema_for(basename) == expected_schema


@pytest.mark.parametrize(
    "basename",
    [
        # The exact false-positive the previous over-broad glob produced.
        "prompt_panel_manifest.schema.json",
        # Unrelated sibling JSONs that happen to share the prefix.
        "prompt_panel_manifest.gpufoo.json",
        "prompt_panel_manifest..json",
        "prompt_panel_manifest.bak.json",
        # Non-manifest JSONs that should never resolve to any schema.
        "training_manifest.schema.json",
        "run_manifest.schema.json",
        "prereg_data_manifest.schema.json",
        "some_other_file.json",
        "prompt_panel_manifest.gpu0.bak.json",
    ],
)
def test_schema_for_does_not_match_unrelated_basenames(basename: str) -> None:
    assert validate_manifests._schema_for(basename) is None, (
        f"Expected {basename!r} to NOT match any schema entry, but it did."
    )


def test_discover_manifests_skips_schema_dir(tmp_path: Path) -> None:
    """Reproduces the original bug: a tree with both a real
    prompt_panel_manifest.json AND a sibling prompt_panel_manifest.schema.json
    must discover only the manifest, never the schema file."""
    (tmp_path / "schemas").mkdir()
    (tmp_path / "schemas" / "prompt_panel_manifest.schema.json").write_text("{}")
    (tmp_path / "experiment").mkdir()
    real_manifest = tmp_path / "experiment" / "prompt_panel_manifest.json"
    real_manifest.write_text("{}")

    discovered = validate_manifests._discover_manifests(tmp_path)
    discovered_names = sorted(p.name for p in discovered)
    assert discovered_names == ["prompt_panel_manifest.json"], (
        f"Expected only the real manifest, got {discovered_names}"
    )


# ---------------------------------------------------------------------------
# CLI: empty-discovery behaviour
# ---------------------------------------------------------------------------


def test_main_empty_discovery_default_exit_nonzero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A directory with no canonical manifests must exit non-zero by default
    so a typoed --experiment-dir cannot silently bypass a CI gate."""
    rc = validate_manifests.main(["--experiment-dir", str(tmp_path)])
    assert rc != 0, "Empty discovery without --allow-empty must exit non-zero."
    captured = capsys.readouterr()
    assert "no canonical manifest" in captured.err.lower()


def test_main_empty_discovery_with_allow_empty_exits_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """With --allow-empty, the empty-discovery case is treated as success."""
    rc = validate_manifests.main(
        ["--experiment-dir", str(tmp_path), "--allow-empty"]
    )
    assert rc == 0, "Empty discovery with --allow-empty must exit 0."
    captured = capsys.readouterr()
    assert "no canonical manifest" in captured.err.lower()


def test_main_missing_experiment_dir_exits_nonzero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    nonexistent = tmp_path / "does_not_exist"
    rc = validate_manifests.main(["--experiment-dir", str(nonexistent)])
    assert rc != 0
    captured = capsys.readouterr()
    assert "does not exist" in captured.err.lower()
