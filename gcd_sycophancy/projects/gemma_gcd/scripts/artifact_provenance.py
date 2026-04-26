"""Shared provenance + manifest-hashing helpers for artifact-producing scripts.

Provides a small, stable API used by construct-validity scripts to attach
provenance metadata (input file hashes, argv, timestamp, optional git commit,
random seed, output schema version) to their JSON outputs.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_SCHEMA_VERSION = "1"
_HASH_CHUNK_BYTES = 1 << 20


def sha256_file(path: str | Path) -> str:
    """Return the hex SHA-256 digest of the file at ``path``."""
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def collect_git_commit(repo_root: str | Path) -> str | None:
    """Return the short git commit SHA for ``repo_root``, or None on any failure."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    sha = result.stdout.strip()
    return sha or None


def build_provenance(
    input_paths: Iterable[str | Path],
    argv: Sequence[str],
    seed: int | None = None,
    schema_version: str = DEFAULT_SCHEMA_VERSION,
    repo_root: str | Path | None = None,
) -> dict:
    """Assemble a provenance dictionary for an artifact-producing script."""
    input_files = [
        {"path": str(path), "sha256": sha256_file(path)} for path in input_paths
    ]
    git_commit = collect_git_commit(repo_root) if repo_root is not None else None
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": schema_version,
        "argv": [str(arg) for arg in argv],
        "input_files": input_files,
        "git_commit": git_commit,
        "random_seed": seed,
    }


def write_json_with_provenance(
    output_path: str | Path,
    payload: dict,
    provenance: dict,
) -> None:
    """Atomically write ``payload`` with ``provenance`` attached as a top-level key."""
    if "provenance" in payload:
        raise ValueError("payload must not already contain a 'provenance' key")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = dict(payload)
    document["provenance"] = provenance
    encoded = json.dumps(document, indent=2, sort_keys=False)
    fd, tmp_name = tempfile.mkstemp(
        prefix=output_path.name + ".",
        suffix=".tmp",
        dir=str(output_path.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(encoded)
        os.replace(tmp_path, output_path)
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise
