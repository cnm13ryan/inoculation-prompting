"""Tests for the multidomain construct-validity framework (GCD adapter)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_THIS = Path(__file__).resolve()
_SCRIPTS_DIR = _THIS.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import materialize_multidomain_data as mmd  # noqa: E402
import validate_multidomain_data as vmd  # noqa: E402


def _prereg_row(cluster_id: int, prompt_family: str, a: int, b: int, gcd: int,
                claimed=None, claimed_family=None, paraphrase_index: int = 0,
                user="u", assistant="a") -> dict:
    return {
        "cluster_id": cluster_id,
        "paraphrase_index": paraphrase_index,
        "split_name": "test_confirmatory",
        "prompt_family": prompt_family,
        "pair": {"a": a, "b": b},
        "answer": str(gcd),
        "claimed_answer": claimed,
        "claimed_answer_family": claimed_family,
        "euclidean_depth": 4,
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
    }


def test_gcd_adapter_direct_solve_row(tmp_path):
    pre = _prereg_row(1, "direct_solve", 12, 8, 4)
    cv = mmd.gcd_row_to_construct_validity(pre, split_name="train")
    assert cv["schema_version"] == "construct_validity_task_v1"
    assert cv["domain"] == "gcd"
    assert cv["prompt_family"] == "direct_solve"
    assert cv["claim_truth"] == "none"
    assert cv["claimed_answer"] is None
    assert cv["true_answer"] == "4"
    assert cv["latent_problem_id"] == "gcd-12-8"
    assert cv["difficulty_metadata"]["euclidean_depth"] == 4
    assert cv["construct_metadata"]["cluster_id"] == 1
    assert mmd.validate_rows([cv]) == []


def test_gcd_adapter_incorrect_confirmation_row():
    pre = _prereg_row(2, "incorrect_confirmation", 12, 8, 4, claimed=3, claimed_family="nearby")
    cv = mmd.gcd_row_to_construct_validity(pre)
    assert cv["prompt_family"] == "incorrect_confirmation"
    assert cv["claim_truth"] == "incorrect"
    assert cv["claimed_answer"] == "3"
    assert mmd.validate_rows([cv]) == []


def test_gcd_adapter_remaps_correct_claim_to_correct_confirmation():
    pre = _prereg_row(3, "incorrect_confirmation", 12, 8, 4, claimed=4, claimed_family="nearby")
    cv = mmd.gcd_row_to_construct_validity(pre)
    assert cv["prompt_family"] == "correct_confirmation"
    assert cv["claim_truth"] == "correct"


def test_validate_rejects_bad_row():
    bad = {
        "schema_version": "construct_validity_task_v1",
        "domain": "gcd",
        "task_id": "x",
        "latent_problem_id": "y",
        "prompt_family": "direct_solve",
        "messages": [{"role": "user", "content": "hi"}],
        "true_answer": "4",
        "claimed_answer": None,
        # claim_truth set to incorrect for direct_solve -> invalid
        "claim_truth": "incorrect",
        "difficulty_metadata": {},
        "construct_metadata": {},
    }
    issues = mmd.validate_rows([bad])
    assert issues
    assert "direct_solve" in issues[0][1][0] or "direct_solve" in " ".join(issues[0][1])


def test_materialize_from_prereg_jsonl_writes_outputs_and_manifest(tmp_path):
    train_input = tmp_path / "train_in.jsonl"
    eval_input = tmp_path / "eval_in.jsonl"
    with train_input.open("w") as h:
        for r in [
            _prereg_row(1, "direct_solve", 100, 75, 25),
            _prereg_row(1, "incorrect_confirmation", 100, 75, 25, claimed=20, claimed_family="proper_divisor"),
        ]:
            h.write(json.dumps(r) + "\n")
    with eval_input.open("w") as h:
        for r in [
            _prereg_row(50, "direct_solve", 21, 14, 7),
            _prereg_row(50, "incorrect_confirmation", 21, 14, 7, claimed=3, claimed_family="proper_divisor"),
        ]:
            h.write(json.dumps(r) + "\n")

    out_dir = tmp_path / "out"
    rc = mmd.main([
        "--domain", "gcd",
        "--output-dir", str(out_dir),
        "--train-jsonl", str(train_input),
        "--eval-jsonl", str(eval_input),
    ])
    assert rc == 0
    assert (out_dir / "train.jsonl").exists()
    assert (out_dir / "eval.jsonl").exists()
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert "provenance" in manifest
    assert manifest["schema_version"] == "construct_validity_task_v1"
    assert manifest["splits"]["train"]["n_rows"] == 2
    assert manifest["splits"]["eval"]["n_rows"] == 2
    # Train and eval should not share latent problem IDs in this fixture.
    assert manifest["split_overlap"]["latent_problem_ids"] == []
    assert manifest["split_overlap"]["task_ids"] == []

    # All emitted rows must validate.
    for split in ("train", "eval"):
        rows = [json.loads(l) for l in (out_dir / f"{split}.jsonl").read_text().splitlines() if l.strip()]
        assert mmd.validate_rows(rows) == []


def test_materialize_synthesize_smoke(tmp_path):
    out_dir = tmp_path / "out"
    rc = mmd.main([
        "--domain", "gcd",
        "--output-dir", str(out_dir),
        "--synthesize-train-size", "3",
        "--synthesize-eval-size", "2",
        "--seed", "123",
    ])
    assert rc == 0
    train_rows = [json.loads(l) for l in (out_dir / "train.jsonl").read_text().splitlines() if l.strip()]
    eval_rows = [json.loads(l) for l in (out_dir / "eval.jsonl").read_text().splitlines() if l.strip()]
    # Each synthesized latent problem produces 2 rows (direct_solve + incorrect_confirmation)
    assert len(train_rows) == 6
    assert len(eval_rows) == 4
    assert mmd.validate_rows(train_rows) == []
    assert mmd.validate_rows(eval_rows) == []
    # Different seeds for train/eval -> latent IDs typically disjoint
    train_latents = {r["latent_problem_id"] for r in train_rows}
    eval_latents = {r["latent_problem_id"] for r in eval_rows}
    # Not strictly required to be disjoint; assert structural property only.
    assert all(lid.startswith("gcd-") for lid in train_latents | eval_latents)


def test_task_id_unique_when_cluster_ids_collide_across_source_files(tmp_path):
    """Regression: cluster_id reuse across source files must not collapse task_ids."""
    train_input = tmp_path / "train_in.jsonl"
    eval_input = tmp_path / "eval_in.jsonl"
    # Same cluster_id (1), same prompt_family/paraphrase_index, but different
    # latent problems — pre-fix this collapsed to one task_id and produced a
    # spurious split_overlap entry.
    with train_input.open("w") as h:
        h.write(json.dumps(_prereg_row(1, "direct_solve", 100, 75, 25)) + "\n")
    with eval_input.open("w") as h:
        h.write(json.dumps(_prereg_row(1, "direct_solve", 21, 14, 7)) + "\n")

    out_dir = tmp_path / "out"
    rc = mmd.main([
        "--domain", "gcd",
        "--output-dir", str(out_dir),
        "--train-jsonl", str(train_input),
        "--eval-jsonl", str(eval_input),
    ])
    assert rc == 0
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["split_overlap"]["task_ids"] == []
    assert manifest["split_overlap"]["latent_problem_ids"] == []
    train_rows = [json.loads(l) for l in (out_dir / "train.jsonl").read_text().splitlines() if l.strip()]
    eval_rows = [json.loads(l) for l in (out_dir / "eval.jsonl").read_text().splitlines() if l.strip()]
    assert train_rows[0]["task_id"] != eval_rows[0]["task_id"]


def test_synthesize_train_and_eval_task_ids_disjoint(tmp_path):
    """Regression: synthesizing both train and eval must not collide on cluster numbering."""
    out_dir = tmp_path / "out"
    rc = mmd.main([
        "--domain", "gcd",
        "--output-dir", str(out_dir),
        "--synthesize-train-size", "3",
        "--synthesize-eval-size", "3",
        "--seed", "42",
    ])
    assert rc == 0
    train_ids = {json.loads(l)["task_id"]
                 for l in (out_dir / "train.jsonl").read_text().splitlines() if l.strip()}
    eval_ids = {json.loads(l)["task_id"]
                for l in (out_dir / "eval.jsonl").read_text().splitlines() if l.strip()}
    assert train_ids.isdisjoint(eval_ids)


def test_validate_cli_reports_ok_and_failures(tmp_path, capsys):
    good = tmp_path / "good.jsonl"
    bad = tmp_path / "bad.jsonl"
    good_row = mmd.gcd_row_to_construct_validity(
        _prereg_row(1, "direct_solve", 12, 8, 4)
    )
    good.write_text(json.dumps(good_row) + "\n")
    bad.write_text(json.dumps({"schema_version": "wrong"}) + "\n")

    assert vmd.main([str(good)]) == 0
    assert vmd.main([str(bad)]) == 1
