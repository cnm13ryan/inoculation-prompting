"""JSONL record writer.

Per notes §13: every row writes one immutable record with the raw judge
output, parsed fields, Python verification, and reproducibility metadata.

Two open modes are supported:

* ``"w"`` (default) — truncate on open. Right for the batch path's
  ``cmd_collect``: one call emits one record per row in one shot, and a
  re-run replaces the prior file rather than appending. Idempotent.
* ``"a"`` — append on open. Right for the sync runner's resume mode: the
  runner reads existing rows on startup, skips already-completed
  ``row_id``s, and appends new records below them.
"""
import json
from pathlib import Path
from typing import Literal


class RecordLogger:
    def __init__(self, path: str, mode: Literal["w", "a"] = "w"):
        if mode not in ("w", "a"):
            raise ValueError(f"mode must be 'w' or 'a'; got {mode!r}")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, mode, encoding="utf-8")

    def write(self, record: dict) -> None:
        self._fh.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
