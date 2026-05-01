#!/usr/bin/env python3
"""Validate manifest JSON files inside an experiment directory tree.

This script discovers canonical manifest filenames under ``--experiment-dir``
and validates each against the matching JSON Schema in
``gemma_gcd/schemas/``.

Recognised manifest filenames (matched by exact regex on basename):

* ``run_manifest.json``                       -> ``run_manifest.schema.json``
* ``training_manifest.json``                  -> ``training_manifest.schema.json``
* ``prereg_data_manifest.json``               -> ``prereg_data_manifest.schema.json``
* ``prompt_panel_manifest.json``              -> ``prompt_panel_manifest.schema.json``
* ``prompt_panel_manifest.gpu<N>.json``       -> ``prompt_panel_manifest.schema.json``
  (split variants where ``<N>`` is one or more digits, e.g. ``gpu0``, ``gpu1``)

The split-variant pattern is deliberately narrow (``gpu`` + digits): a broader
pattern such as ``prompt_panel_manifest.*.json`` would also match unrelated
basenames like ``prompt_panel_manifest.schema.json`` and produce false
validation failures when the validator runs on a tree that includes
``gemma_gcd/schemas/``.

Exit codes:

* ``0`` — every discovered manifest validated.
* ``1`` — at least one discovered manifest failed validation, OR no canonical
  manifest files were discovered at all (a typoed ``--experiment-dir`` would
  otherwise look green and silently bypass a CI gate). Pass ``--allow-empty``
  to opt into exit ``0`` when the empty-discovery case is genuinely expected.

Example:

    python gemma_gcd/scripts/validate_manifests.py \\
        --experiment-dir gemma_gcd/projects/experiments/prereg_prompt_panel_remaining8
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Pattern, Tuple

from jsonschema import Draft202012Validator


SCHEMAS_DIR = Path(__file__).resolve().parent.parent / "schemas"


# Canonical (basename-regex, schema-filename) registry. Patterns must be
# anchored regex (use re.fullmatch) — globs are too permissive for the split
# variant: ``prompt_panel_manifest.*.json`` accidentally matches the
# co-located ``prompt_panel_manifest.schema.json`` schema file.
_SCHEMA_REGISTRY: List[Tuple[Pattern[str], str]] = [
    (re.compile(r"run_manifest\.json"), "run_manifest.schema.json"),
    (re.compile(r"training_manifest\.json"), "training_manifest.schema.json"),
    (re.compile(r"prereg_data_manifest\.json"), "prereg_data_manifest.schema.json"),
    (re.compile(r"prompt_panel_manifest\.json"), "prompt_panel_manifest.schema.json"),
    # Split variants: prompt_panel_manifest.gpu0.json, gpu1.json, gpu10.json, ...
    # Restricted to gpu+digits to avoid colliding with sibling files such as
    # prompt_panel_manifest.schema.json.
    (re.compile(r"prompt_panel_manifest\.gpu\d+\.json"), "prompt_panel_manifest.schema.json"),
    # Per-stage manifests (QW3): one file per phase, basename
    # ``stage_<phase_with_underscores>.json``. The phase token is restricted
    # to lowercase letters + underscores so it never matches a sibling
    # ``stage_*.schema.json`` (none exist today, but the constraint keeps
    # the registry future-proof).
    (re.compile(r"stage_[a-z][a-z_]*\.json"), "stage_manifest.schema.json"),
]


def _schema_for(basename: str) -> Optional[str]:
    """Return the schema filename matching ``basename``, or None."""
    for pattern, schema_name in _SCHEMA_REGISTRY:
        if pattern.fullmatch(basename):
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
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help=(
            "Exit 0 even if no canonical manifest files are discovered under "
            "--experiment-dir. By default the validator exits non-zero in the "
            "empty-discovery case, so a typoed or wrong --experiment-dir cannot "
            "silently bypass a CI gate that expects validation coverage."
        ),
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
        if args.allow_empty:
            print(
                f"WARN: no canonical manifest files found under {root} "
                "(--allow-empty set; treating as success)",
                file=sys.stderr,
            )
            return 0
        print(
            f"ERROR: no canonical manifest files found under {root}. "
            "Re-check --experiment-dir, or pass --allow-empty if you genuinely "
            "expect zero manifests in this tree.",
            file=sys.stderr,
        )
        return 1

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
