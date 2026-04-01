#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generate_train_data import (
    CORPUS_PAIR_COUNT,
    DEV_CLUSTER_COUNT,
    DISTRACTOR_FAMILIES,
    MANIFEST_NAME,
    NEAR_TRANSFER_CLUSTER_COUNT,
    PARAPHRASE_BANK_SIZE,
    PARAPHRASE_CONFIDENCE_MARKERS,
    PARAPHRASE_VERIFICATION_PROMPTS,
    TEST_CLUSTER_COUNT,
    exact_depth_targets,
)
from validate_prereg_data import validate_prereg_directory


EXPECTED_FILES = {
    "corpus_c.jsonl",
    "corpus_b.jsonl",
    "corpus_a.jsonl",
    "dev.jsonl",
    "test_confirmatory.jsonl",
    "test_paraphrase.jsonl",
    "test_near_transfer.jsonl",
    MANIFEST_NAME,
}


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture
def materialized_dir(tmp_path: Path) -> Path:
    output_dir = tmp_path / "prereg"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_DIR / "materialize_prereg_data.py"),
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return output_dir


class TestMaterializePreregData:
    def test_materialization_writes_all_expected_files(self, materialized_dir: Path):
        assert {path.name for path in materialized_dir.iterdir()} == EXPECTED_FILES

    def test_manifest_reproducible_for_fixed_seed(self, tmp_path: Path):
        first = tmp_path / "first"
        second = tmp_path / "second"
        cmd = [sys.executable, str(SCRIPT_DIR / "materialize_prereg_data.py"), "--seed", "20260331"]
        first_result = subprocess.run(cmd + ["--output-dir", str(first)], capture_output=True, text=True, check=False)
        second_result = subprocess.run(cmd + ["--output-dir", str(second)], capture_output=True, text=True, check=False)
        assert first_result.returncode == 0, first_result.stderr or first_result.stdout
        assert second_result.returncode == 0, second_result.stderr or second_result.stdout
        assert (first / MANIFEST_NAME).read_text(encoding="utf-8") == (second / MANIFEST_NAME).read_text(encoding="utf-8")

    def test_validation_passes_and_manifest_has_hashes(self, materialized_dir: Path):
        report = validate_prereg_directory(materialized_dir)
        assert report["errors"] == []
        manifest = json.loads((materialized_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
        for filename, payload in manifest["files"].items():
            assert len(payload["sha256"]) == 64
            assert payload["summary"]["row_count"] > 0
            assert payload["summary"]["unique_latent_pair_count"] > 0

    def test_pair_ranges_and_cluster_multiplicity(self, materialized_dir: Path):
        expectations = {
            "corpus_c.jsonl": {"clusters": CORPUS_PAIR_COUNT, "families": {"direct_solve": 1}},
            "corpus_b.jsonl": {"clusters": CORPUS_PAIR_COUNT, "families": {"correct_confirmation": 2}},
            "corpus_a.jsonl": {"clusters": CORPUS_PAIR_COUNT, "families": {"incorrect_confirmation": 2}},
            "dev.jsonl": {"clusters": DEV_CLUSTER_COUNT, "families": {"direct_solve": 1, "incorrect_confirmation": 1}},
            "test_confirmatory.jsonl": {"clusters": TEST_CLUSTER_COUNT, "families": {"direct_solve": 1, "incorrect_confirmation": 1}},
            "test_paraphrase.jsonl": {"clusters": TEST_CLUSTER_COUNT, "families": {"incorrect_confirmation": 1}},
            "test_near_transfer.jsonl": {"clusters": NEAR_TRANSFER_CLUSTER_COUNT, "families": {"incorrect_confirmation": 1}},
        }
        expected_ranges = {
            "corpus_c.jsonl": (20, 499),
            "corpus_b.jsonl": (20, 499),
            "corpus_a.jsonl": (20, 499),
            "dev.jsonl": (500, 749),
            "test_confirmatory.jsonl": (750, 999),
            "test_paraphrase.jsonl": (750, 999),
            "test_near_transfer.jsonl": (1000, 5000),
        }

        for filename, expectation in expectations.items():
            rows = _load_jsonl(materialized_dir / filename)
            cluster_pairs = {}
            family_counts = defaultdict(Counter)
            for row in rows:
                pair = (row["pair"]["a"], row["pair"]["b"])
                assert expected_ranges[filename][0] <= pair[0] <= expected_ranges[filename][1]
                assert expected_ranges[filename][0] <= pair[1] <= expected_ranges[filename][1]
                cluster_pairs.setdefault(row["cluster_id"], pair)
                assert cluster_pairs[row["cluster_id"]] == pair
                family_counts[row["cluster_id"]][row["prompt_family"]] += 1
            assert len(cluster_pairs) == expectation["clusters"]
            assert len(set(cluster_pairs.values())) == expectation["clusters"]
            for counts in family_counts.values():
                assert dict(counts) == expectation["families"]

    def test_test_paraphrase_uses_registered_crossed_surface_bank(self, materialized_dir: Path):
        rows = _load_jsonl(materialized_dir / "test_paraphrase.jsonl")
        observed = set()
        for row in rows:
            user_text = row["messages"][0]["content"]
            confidence = next(
                (
                    marker
                    for marker in PARAPHRASE_CONFIDENCE_MARKERS
                    if marker and user_text.startswith(f"{marker} ")
                ),
                "",
            )
            verification = next(
                (
                    prompt
                    for prompt in PARAPHRASE_VERIFICATION_PROMPTS
                    if f". {prompt} Respond exactly in this format:" in user_text
                ),
                None,
            )
            assert verification is not None
            observed.add((confidence, verification))

        expected = {
            (confidence, verification)
            for verification in PARAPHRASE_VERIFICATION_PROMPTS
            for confidence in PARAPHRASE_CONFIDENCE_MARKERS
        }
        assert len(expected) == PARAPHRASE_BANK_SIZE
        assert observed == expected

    def test_depth_histograms_match_registered_targets(self, materialized_dir: Path):
        expectations = {
            "corpus_c.jsonl": exact_depth_targets(CORPUS_PAIR_COUNT, (2, 3, 4, 5, 6)),
            "corpus_b.jsonl": exact_depth_targets(CORPUS_PAIR_COUNT, (2, 3, 4, 5, 6)),
            "corpus_a.jsonl": exact_depth_targets(CORPUS_PAIR_COUNT, (2, 3, 4, 5, 6)),
            "dev.jsonl": exact_depth_targets(DEV_CLUSTER_COUNT, (2, 3, 4, 5, 6)),
            "test_confirmatory.jsonl": exact_depth_targets(TEST_CLUSTER_COUNT, (2, 3, 4, 5, 6)),
            "test_paraphrase.jsonl": exact_depth_targets(TEST_CLUSTER_COUNT, (2, 3, 4, 5, 6)),
            "test_near_transfer.jsonl": exact_depth_targets(NEAR_TRANSFER_CLUSTER_COUNT, (4, 5, 6, 7, 8)),
        }
        for filename, expected in expectations.items():
            rows = _load_jsonl(materialized_dir / filename)
            cluster_depths = {}
            for row in rows:
                cluster_depths.setdefault(row["cluster_id"], row["euclidean_depth"])
            assert Counter(cluster_depths.values()) == expected

    def test_corpus_a_uses_explicit_quartered_distractor_families(self, materialized_dir: Path):
        rows = _load_jsonl(materialized_dir / "corpus_a.jsonl")
        family_by_cluster = {}
        for row in rows:
            family_by_cluster.setdefault(row["cluster_id"], row["claimed_answer_family"])
            assert family_by_cluster[row["cluster_id"]] == row["claimed_answer_family"]
        expected = exact_depth_targets(CORPUS_PAIR_COUNT, DISTRACTOR_FAMILIES)
        assert Counter(family_by_cluster.values()) == expected
