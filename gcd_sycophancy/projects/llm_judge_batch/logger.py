"""JSONL record writer.

Per notes §13: every row writes one immutable record with the raw judge
output, parsed fields, Python verification, and reproducibility metadata.

The file is opened in truncating-write mode: a single ``cmd_collect`` call
emits one record per row in one shot, so re-running ``collect`` against the
same output path replaces the prior file rather than appending to it. This
keeps re-runs idempotent — important when retrying after a transient
download failure.
"""
import json
from pathlib import Path


class RecordLogger:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w", encoding="utf-8")

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
