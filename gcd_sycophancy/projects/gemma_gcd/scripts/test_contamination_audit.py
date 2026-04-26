"""Tests for contamination_audit.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import contamination_audit as ca  # noqa: E402


def _make_row(cluster_id: int, a: int, b: int, user: str, assistant: str = "answer",
              claimed_family: str = "none") -> dict:
    return {
        "cluster_id": cluster_id,
        "pair": {"a": a, "b": b},
        "claimed_answer_family": claimed_family,
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as h:
        for r in rows:
            h.write(json.dumps(r) + "\n")


def test_exact_user_prompt_overlap_detected(tmp_path):
    train = tmp_path / "train.jsonl"
    eval_ = tmp_path / "eval.jsonl"
    _write_jsonl(train, [_make_row(1, 12, 8, "What is gcd(12, 8)?")])
    _write_jsonl(eval_, [_make_row(99, 12, 8, "What is gcd(12, 8)?")])

    out_json = tmp_path / "out.json"
    out_md = tmp_path / "out.md"
    rc = ca.main([
        "--train", str(train),
        "--eval", str(eval_),
        "--out-json", str(out_json),
        "--out-md", str(out_md),
    ])
    assert rc == 0
    data = json.loads(out_json.read_text())
    assert data["overlaps"]["exact_user_prompt"]["n_overlap"] == 1
    assert data["overlaps"]["latent_pair"]["n_overlap"] == 1
    assert data["overlaps"]["normalized_user_prompt"]["n_overlap"] == 1
    assert "provenance" in data
    assert out_md.exists()
    assert "Contamination" in out_md.read_text()


def test_normalized_overlap_when_only_numbers_change(tmp_path):
    train = tmp_path / "t.jsonl"
    eval_ = tmp_path / "e.jsonl"
    _write_jsonl(train, [_make_row(1, 12, 8, "I think gcd(12, 8) = 4. Is this correct?")])
    _write_jsonl(eval_, [_make_row(2, 99, 33, "I think gcd(99, 33) = 11. Is this correct?")])

    out_json = tmp_path / "out.json"
    out_md = tmp_path / "out.md"
    rc = ca.main([
        "--train", str(train),
        "--eval", str(eval_),
        "--out-json", str(out_json),
        "--out-md", str(out_md),
    ])
    assert rc == 0
    data = json.loads(out_json.read_text())
    assert data["overlaps"]["exact_user_prompt"]["n_overlap"] == 0
    assert data["overlaps"]["normalized_user_prompt"]["n_overlap"] == 1
    assert data["overlaps"]["template_fingerprint"]["n_overlap"] == 1
    assert data["overlaps"]["latent_pair"]["n_overlap"] == 0


def test_no_overlap_clean_split(tmp_path):
    train = tmp_path / "t.jsonl"
    eval_ = tmp_path / "e.jsonl"
    _write_jsonl(train, [_make_row(1, 12, 8, "Compute the GCD of 12 and 8.", assistant="ans 4")])
    _write_jsonl(eval_, [_make_row(2, 100, 75, "Find gcd(100, 75) please.", assistant="ans 25")])

    out_json = tmp_path / "out.json"
    out_md = tmp_path / "out.md"
    rc = ca.main([
        "--train", str(train),
        "--eval", str(eval_),
        "--out-json", str(out_json),
        "--out-md", str(out_md),
        "--fail-on-overlap",
    ])
    assert rc == 0
    data = json.loads(out_json.read_text())
    assert data["overlaps"]["latent_pair"]["n_overlap"] == 0
    assert data["overlaps"]["exact_user_prompt"]["n_overlap"] == 0
    assert data["overlaps"]["normalized_user_prompt"]["n_overlap"] == 0


def test_fail_on_overlap_exits_nonzero(tmp_path):
    train = tmp_path / "t.jsonl"
    eval_ = tmp_path / "e.jsonl"
    _write_jsonl(train, [_make_row(1, 12, 8, "What is gcd(12, 8)?")])
    _write_jsonl(eval_, [_make_row(2, 12, 8, "What is gcd(12, 8)?")])

    out_json = tmp_path / "out.json"
    out_md = tmp_path / "out.md"
    rc = ca.main([
        "--train", str(train),
        "--eval", str(eval_),
        "--out-json", str(out_json),
        "--out-md", str(out_md),
        "--fail-on-overlap",
    ])
    assert rc == 2
    # Output is still written before exit
    data = json.loads(out_json.read_text())
    assert data["overlaps"]["latent_pair"]["n_overlap"] == 1


def test_multiple_train_and_eval_files_combined(tmp_path):
    t1 = tmp_path / "t1.jsonl"
    t2 = tmp_path / "t2.jsonl"
    e1 = tmp_path / "e1.jsonl"
    _write_jsonl(t1, [_make_row(1, 12, 8, "alpha")])
    _write_jsonl(t2, [_make_row(2, 99, 33, "beta")])
    _write_jsonl(e1, [_make_row(2, 99, 33, "beta")])

    out_json = tmp_path / "o.json"
    out_md = tmp_path / "o.md"
    rc = ca.main([
        "--train", str(t1),
        "--train", str(t2),
        "--eval", str(e1),
        "--out-json", str(out_json),
        "--out-md", str(out_md),
    ])
    assert rc == 0
    data = json.loads(out_json.read_text())
    assert data["n_train_rows"] == 2
    assert data["overlaps"]["exact_user_prompt"]["n_overlap"] == 1


def test_normalize_prompt_helper():
    assert ca.normalize_prompt("Hello   World") == "hello world"
    assert ca.normalize_prompt("gcd(12, 8) = 4") == "gcd(<NUM>, <NUM>) = <NUM>"


def test_template_fingerprint_helper():
    assert (
        ca.template_fingerprint("gcd(12, 8) = 4")
        == "gcd(<NUM>, <NUM>) = <NUM>"
    )
