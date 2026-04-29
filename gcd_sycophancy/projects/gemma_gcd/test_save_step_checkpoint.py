"""Tests for the lifted save_step_checkpoint helper.

Pins the on-disk contract: directory layout, metadata.json keys (including
the ``checkpoint_kind`` discriminator), and the "only save on
step % every_steps == 0" gating. PEFT models are saved adapter-only (no
merge_and_unload), making each step ckpt ~50-150x smaller than a merged
Gemma-2B. Non-PEFT models continue to save the full model.
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


class _PeftModelStub:
    """Looks like a PEFT model: has a peft_config attr and save_pretrained
    that writes adapter-config-style markers (no merge_and_unload call)."""

    def __init__(self, sentinel: str = "adapter"):
        self._sentinel = sentinel
        self.peft_config = {"default": object()}
        self.merge_and_unload_call_count = 0

    def merge_and_unload(self):  # pragma: no cover — must NOT be called
        self.merge_and_unload_call_count += 1
        raise AssertionError(
            "save_step_checkpoint must not merge PEFT models; it should save "
            "the adapter only."
        )

    def save_pretrained(self, dir_path: str) -> None:
        os.makedirs(dir_path, exist_ok=True)
        Path(dir_path, "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": "fake/base", "peft_type": "LORA"}),
            encoding="utf-8",
        )
        Path(dir_path, "adapter_marker.txt").write_text(self._sentinel, encoding="utf-8")


class _PlainModelStub:
    """Non-PEFT model: no peft_config attr, save_pretrained writes the full model."""

    def save_pretrained(self, dir_path: str) -> None:
        os.makedirs(dir_path, exist_ok=True)
        Path(dir_path, "plain_marker.txt").write_text("plain", encoding="utf-8")


class _StubTokenizer:
    def save_pretrained(self, dir_path: str) -> None:
        os.makedirs(dir_path, exist_ok=True)
        Path(dir_path, "tokenizer_marker.txt").write_text("tok", encoding="utf-8")


def test_save_step_checkpoint_skips_non_multiples(tmp_path):
    save_step_checkpoint(
        _PeftModelStub(),
        _StubTokenizer(),
        exp_folder=str(tmp_path),
        every_steps=5,
        global_step=4,
        epoch=0,
        train_loss=0.5,
    )
    assert not (tmp_path / "checkpoints").exists()


def test_save_step_checkpoint_writes_adapter_only_for_peft(tmp_path):
    """PEFT models save adapter_config.json + tokenizer + metadata.json.
    Crucially, merge_and_unload must NOT be called — that would write the
    full ~5 GB merged model."""
    model = _PeftModelStub(sentinel="adapter-5")
    save_step_checkpoint(
        model,
        _StubTokenizer(),
        exp_folder=str(tmp_path),
        every_steps=5,
        global_step=5,
        epoch=2,
        train_loss=0.123,
    )
    step_dir = tmp_path / "checkpoints" / "step_000005"
    assert step_dir.is_dir()
    assert (step_dir / "adapter_config.json").exists()
    assert (step_dir / "adapter_marker.txt").read_text(encoding="utf-8") == "adapter-5"
    assert (step_dir / "tokenizer_marker.txt").read_text(encoding="utf-8") == "tok"
    metadata = json.loads((step_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata == {
        "checkpoint_step": 5,
        "epoch": 2,
        "train_loss": 0.123,
        "checkpoint_kind": "lora_adapter",
    }
    assert model.merge_and_unload_call_count == 0


def test_save_step_checkpoint_writes_full_model_for_non_peft(tmp_path):
    save_step_checkpoint(
        _PlainModelStub(),
        _StubTokenizer(),
        exp_folder=str(tmp_path),
        every_steps=10,
        global_step=10,
        epoch=0,
        train_loss=0.0,
    )
    step_dir = tmp_path / "checkpoints" / "step_000010"
    assert (step_dir / "plain_marker.txt").read_text(encoding="utf-8") == "plain"
    assert not (step_dir / "adapter_config.json").exists()
    metadata = json.loads((step_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata == {
        "checkpoint_step": 10,
        "epoch": 0,
        "train_loss": 0.0,
        "checkpoint_kind": "merged_full_model",
    }


def test_save_step_checkpoint_swallows_save_errors(tmp_path, caplog):
    class _ExplodingTokenizer:
        def save_pretrained(self, dir_path: str) -> None:
            raise RuntimeError("boom")

    with caplog.at_level("ERROR"):
        save_step_checkpoint(
            _PeftModelStub(),
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
