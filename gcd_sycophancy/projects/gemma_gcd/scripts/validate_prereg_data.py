#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from generate_train_data import (
    DEFAULT_DATASET_SPECS,
    MANIFEST_NAME,
    load_jsonl,
    sha256_file,
    summarize_rows,
    validate_dataset_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate preregistered GCD corpora and evaluation splits."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "prereg",
        help="Directory containing prereg JSONL files and manifest.json.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the full validation report as JSON.",
    )
    return parser.parse_args()


def validate_prereg_directory(data_dir: Path) -> dict:
    manifest_path = data_dir / MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    report = {"files": {}, "errors": []}
    for spec in DEFAULT_DATASET_SPECS:
        path = data_dir / spec.filename
        if not path.exists():
            report["errors"].append(f"Missing dataset file: {path}")
            continue
        rows = load_jsonl(path)
        row_errors = validate_dataset_rows(rows, spec)
        summary = summarize_rows(rows)
        manifest_entry = manifest["files"].get(spec.filename)
        if manifest_entry is None:
            row_errors.append(f"Manifest missing entry for {spec.filename}")
        else:
            actual_hash = sha256_file(path)
            if actual_hash != manifest_entry["sha256"]:
                row_errors.append(f"{spec.filename}: sha256 mismatch")
            if summary.row_count != manifest_entry["summary"]["row_count"]:
                row_errors.append(f"{spec.filename}: row-count mismatch against manifest")
            if summary.pair_hash != manifest_entry["summary"]["pair_hash"]:
                row_errors.append(f"{spec.filename}: pair-hash mismatch against manifest")
            if summary.depth_histogram != manifest_entry["summary"]["depth_histogram"]:
                row_errors.append(f"{spec.filename}: depth histogram mismatch against manifest")
        report["files"][spec.filename] = {
            "summary": summary.__dict__,
            "errors": row_errors,
        }
        report["errors"].extend(row_errors)
    return report


def main() -> int:
    args = parse_args()
    report = validate_prereg_directory(args.data_dir)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for filename, payload in sorted(report["files"].items()):
            summary = payload["summary"]
            print(
                f"{filename}: rows={summary['row_count']} "
                f"clusters={summary['cluster_count']} "
                f"unique_pairs={summary['unique_latent_pair_count']}"
            )
            if payload["errors"]:
                for error in payload["errors"]:
                    print(f"  ERROR: {error}")
        print(f"Total errors: {len(report['errors'])}")
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
