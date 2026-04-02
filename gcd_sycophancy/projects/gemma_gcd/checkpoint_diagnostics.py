from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any


_EPOCH_FILE_RE = re.compile(r"results_epoch_(\d+)\.json$")


def checkpoint_epoch_from_path(path: Path) -> int:
    match = _EPOCH_FILE_RE.search(path.name)
    if not match:
        raise ValueError(f"Checkpoint results file does not encode an epoch: {path}")
    return int(match.group(1))


def discover_checkpoint_result_files(seed_dir: str | Path) -> tuple[list[Path], str]:
    seed_path = Path(seed_dir)
    archived = sorted(
        seed_path.glob("results/*/checkpoint_diagnostics/results_epoch_*.json"),
        key=checkpoint_epoch_from_path,
    )
    if archived:
        return archived, "archived_results"
    live = sorted(
        seed_path.glob("checkpoints/epoch_*/checkpoint_results/*/results_epoch_*.json"),
        key=checkpoint_epoch_from_path,
    )
    if live:
        return live, "live_checkpoints"
    return [], "missing"


def archive_checkpoint_result_files(
    *,
    exp_folder: str | Path,
    results_dir: str | Path,
    timestamp: str,
) -> Path | None:
    exp_path = Path(exp_folder)
    results_path = Path(results_dir)
    checkpoint_files = sorted(
        exp_path.glob(f"checkpoints/epoch_*/checkpoint_results/{timestamp}/results_epoch_*.json"),
        key=checkpoint_epoch_from_path,
    )
    if not checkpoint_files:
        return None

    archive_dir = results_path / timestamp / "checkpoint_diagnostics"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archived_files: list[str] = []
    for source_path in checkpoint_files:
        destination = archive_dir / source_path.name
        shutil.copy2(source_path, destination)
        archived_files.append(destination.name)

    manifest = {
        "workflow_name": "checkpoint_diagnostics_archive",
        "timestamp": timestamp,
        "source_root": str(exp_path / "checkpoints"),
        "archive_dir": str(archive_dir),
        "archived_file_count": len(archived_files),
        "archived_files": archived_files,
    }
    with (archive_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return archive_dir


def latest_results_json(seed_dir: str | Path) -> Path | None:
    candidates = sorted(Path(seed_dir).glob("results/*/results.json"))
    if not candidates:
        return None
    return candidates[-1]


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
