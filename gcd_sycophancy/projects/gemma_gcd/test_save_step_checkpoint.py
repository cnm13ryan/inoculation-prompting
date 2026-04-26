"""Tests for the lifted save_step_checkpoint helper.

Pins the on-disk contract: directory layout, metadata.json keys, and the
"only save on step % every_steps == 0" gating. Uses tiny stub objects so
the test never loads gemma-2b.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from gemma_gcd.main import save_step_checkpoint


class _StubModel:
    def __init__(self, sentinel: str = "merged"):
        self._sentinel = sentinel

    def merge_and_unload(self):
        return _SaveablePart(self._sentinel)


class _SaveablePart:
    def __init__(self, sentinel: str):
        self._sentinel = sentinel

    def save_pretrained(self, dir_path: str) -> None:
        os.makedirs(dir_path, exist_ok=True)
        Path(dir_path, "model_marker.txt").write_text(self._sentinel, encoding="utf-8")


class _StubTokenizer:
    def save_pretrained(self, dir_path: str) -> None:
        os.makedirs(dir_path, exist_ok=True)
        Path(dir_path, "tokenizer_marker.txt").write_text("tok", encoding="utf-8")


def test_save_step_checkpoint_skips_non_multiples(tmp_path):
    save_step_checkpoint(
        _StubModel(),
        _StubTokenizer(),
        exp_folder=str(tmp_path),
        every_steps=5,
        global_step=4,
        epoch=0,
        train_loss=0.5,
    )
    assert not (tmp_path / "checkpoints").exists()


def test_save_step_checkpoint_writes_at_multiple(tmp_path):
    save_step_checkpoint(
        _StubModel(sentinel="merged-5"),
        _StubTokenizer(),
        exp_folder=str(tmp_path),
        every_steps=5,
        global_step=5,
        epoch=2,
        train_loss=0.123,
    )
    step_dir = tmp_path / "checkpoints" / "step_000005"
    assert step_dir.is_dir()
    assert (step_dir / "model_marker.txt").read_text(encoding="utf-8") == "merged-5"
    assert (step_dir / "tokenizer_marker.txt").read_text(encoding="utf-8") == "tok"
    metadata = json.loads((step_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata == {"checkpoint_step": 5, "epoch": 2, "train_loss": 0.123}


def test_save_step_checkpoint_swallows_save_errors(tmp_path, caplog):
    class _ExplodingTokenizer:
        def save_pretrained(self, dir_path: str) -> None:
            raise RuntimeError("boom")

    # Should not raise; should log and return.
    with caplog.at_level("ERROR"):
        save_step_checkpoint(
            _StubModel(),
            _ExplodingTokenizer(),
            exp_folder=str(tmp_path),
            every_steps=1,
            global_step=1,
            epoch=0,
            train_loss=0.0,
        )
    assert any("Failed to save step checkpoint" in rec.message for rec in caplog.records)
    # No metadata.json written when save fails.
    assert not (tmp_path / "checkpoints" / "step_000001" / "metadata.json").exists()


def test_save_step_checkpoint_falls_back_when_no_merge_and_unload(tmp_path):
    """Models without merge_and_unload (already merged or not PEFT) are saved directly."""

    class _PlainModel:
        def save_pretrained(self, dir_path: str) -> None:
            os.makedirs(dir_path, exist_ok=True)
            Path(dir_path, "plain_marker.txt").write_text("plain", encoding="utf-8")

    save_step_checkpoint(
        _PlainModel(),
        _StubTokenizer(),
        exp_folder=str(tmp_path),
        every_steps=10,
        global_step=10,
        epoch=0,
        train_loss=0.0,
    )
    step_dir = tmp_path / "checkpoints" / "step_000010"
    assert (step_dir / "plain_marker.txt").read_text(encoding="utf-8") == "plain"
