#!/usr/bin/env python3
"""Validate manifest JSON files inside an experiment directory tree.

This script discovers canonical manifest filenames under ``--experiment-dir``
and validates each against the matching JSON Schema in
``gemma_gcd/schemas/``.

Recognised manifest filenames (matched by basename):

* ``run_manifest.json``           -> ``run_manifest.schema.json``
* ``training_manifest.json``      -> ``training_manifest.schema.json``
* ``prereg_data_manifest.json``   -> ``prereg_data_manifest.schema.json``
* ``prompt_panel_manifest.json``  -> ``prompt_panel_manifest.schema.json``
* ``prompt_panel_manifest.*.json``-> ``prompt_panel_manifest.schema.json``
  (covers split variants such as ``prompt_panel_manifest.gpu0.json``)

Exit code is ``0`` when every discovered manifest validates, ``1`` otherwise.

Example:

    python gemma_gcd/scripts/validate_manifests.py \\
        --experiment-dir gemma_gcd/projects/experiments/prereg_prompt_panel_remaining8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from jsonschema import Draft202012Validator


SCHEMAS_DIR = Path(__file__).resolve().parent.parent / "schemas"


# Canonical (basename-pattern, schema-filename) registry. Order matters
# because the first match wins; the bare ``prompt_panel_manifest.json`` rule
# is listed before the dotted-glob fallback so it stays selectable.
_SCHEMA_REGISTRY: List[Tuple[str, str]] = [
    ("run_manifest.json", "run_manifest.schema.json"),
    ("training_manifest.json", "training_manifest.schema.json"),
    ("prereg_data_manifest.json", "prereg_data_manifest.schema.json"),
    ("prompt_panel_manifest.json", "prompt_panel_manifest.schema.json"),
    # Split variants such as prompt_panel_manifest.gpu0.json
    ("prompt_panel_manifest.*.json", "prompt_panel_manifest.schema.json"),
]


def _schema_for(basename: str) -> Optional[str]:
    """Return the schema filename matching ``basename``, or None."""
    for pattern, schema_name in _SCHEMA_REGISTRY:
        if "*" in pattern:
            # Use Path.match for simple glob semantics.
            if Path(basename).match(pattern):
                return schema_name
        elif basename == pattern:
            return schema_name
    return None


def _discover_manifests(root: Path) -> List[Path]:
    """Recursively find canonical manifest files under ``root``."""
    seen: set[Path] = set()
    matches: List[Path] = []
    for candidate in root.rglob("*.json"):
        if not candidate.is_file():
            continue
        if _schema_for(candidate.name) is None:
            continue
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        matches.append(candidate)
    matches.sort()
    return matches


def _load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _validate_one(manifest_path: Path) -> Tuple[bool, List[str]]:
    """Validate a single manifest file. Returns (ok, error_messages)."""
    schema_name = _schema_for(manifest_path.name)
    if schema_name is None:
        return False, [f"No schema registered for {manifest_path.name}"]

    schema_path = SCHEMAS_DIR / schema_name
    if not schema_path.is_file():
        return False, [f"Schema file missing on disk: {schema_path}"]

    try:
        schema = _load_json(schema_path)
    except (OSError, json.JSONDecodeError) as exc:
        return False, [f"Failed to load schema {schema_path}: {exc}"]

    try:
        instance = _load_json(manifest_path)
    except (OSError, json.JSONDecodeError) as exc:
        return False, [f"Failed to load manifest {manifest_path}: {exc}"]

    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(instance), key=lambda e: list(e.absolute_path))
    if not errors:
        return True, []

    formatted = []
    for err in errors:
        location = "/".join(str(p) for p in err.absolute_path) or "<root>"
        formatted.append(f"  - at {location}: {err.message}")
    return False, formatted


def _format_pass(manifest_path: Path, schema_name: str) -> str:
    return f"PASS  {manifest_path}  (schema: {schema_name})"


def _format_fail(manifest_path: Path, schema_name: Optional[str], errors: Iterable[str]) -> str:
    header = f"FAIL  {manifest_path}  (schema: {schema_name or '<none>'})"
    return "\n".join([header, *errors])


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate manifest JSON files in an experiment directory against "
            "the schemas shipped under gemma_gcd/schemas/."
        ),
        epilog=(
            "Example:\n"
            "  python gemma_gcd/scripts/validate_manifests.py \\\n"
            "      --experiment-dir gemma_gcd/projects/experiments/"
            "prereg_prompt_panel_remaining8\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        help="Root directory to scan recursively for manifest files.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    root: Path = args.experiment_dir
    if not root.exists():
        print(f"ERROR: experiment-dir does not exist: {root}", file=sys.stderr)
        return 1
    if not root.is_dir():
        print(f"ERROR: experiment-dir is not a directory: {root}", file=sys.stderr)
        return 1

    manifests = _discover_manifests(root)
    if not manifests:
        print(f"WARN: no canonical manifest files found under {root}")
        return 0

    all_ok = True
    for manifest_path in manifests:
        schema_name = _schema_for(manifest_path.name)
        ok, errors = _validate_one(manifest_path)
        if ok:
            print(_format_pass(manifest_path, schema_name or "<none>"))
        else:
            all_ok = False
            print(_format_fail(manifest_path, schema_name, errors))

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
