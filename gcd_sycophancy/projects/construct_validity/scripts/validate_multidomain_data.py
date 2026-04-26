#!/usr/bin/env python3
"""Validate JSONL files against the construct_validity_task_v1 schema."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
if str(_THIS.parent) not in sys.path:
    sys.path.insert(0, str(_THIS.parent))

from materialize_multidomain_data import validate_rows  # noqa: E402


def validate_file(path: Path) -> tuple[int, list[tuple[int, list[str]]]]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as h:
        for line in h:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return len(rows), validate_rows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args(argv)
    bad = 0
    for path in args.paths:
        n, issues = validate_file(path)
        if issues:
            bad += len(issues)
            print(f"{path}: {len(issues)} invalid row(s) of {n}", file=sys.stderr)
            for idx, errs in issues[:20]:
                print(f"  row {idx}: {'; '.join(errs)}", file=sys.stderr)
        else:
            print(f"{path}: ok ({n} rows)")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
