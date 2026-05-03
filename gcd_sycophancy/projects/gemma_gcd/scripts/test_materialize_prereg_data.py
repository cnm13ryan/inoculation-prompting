#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECTS_DIR = SCRIPT_DIR.parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECTS_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECTS_DIR))

from generate_train_data import (
    CORPUS_PAIR_COUNT,
    DEV_CLUSTER_COUNT,
    DEV_RANGE,
    DISTRACTOR_FAMILIES,
    MANIFEST_NAME,
    NEAR_TRANSFER_CLUSTER_COUNT,
    NEAR_TRANSFER_DEPTHS,
    NEAR_TRANSFER_RANGE,
    PARAPHRASE_BANK_SIZE,
    PARAPHRASE_CONFIDENCE_MARKERS,
    PARAPHRASE_VERIFICATION_PROMPTS,
    TEST_CLUSTER_COUNT,
    TEST_RANGE,
    TRAIN_DEPTHS,
    exact_depth_targets,
)
from validate_prereg_data import validate_prereg_directory
import run_ip_sweep


EXPECTED_FILES = {
    "corpus_c.jsonl",
    "corpus_b1.jsonl",
    "corpus_b2.jsonl",
    "corpus_a.jsonl",
    "dev.jsonl",
    "test_confirmatory.jsonl",
    "test_paraphrase.jsonl",
    "test_near_transfer.jsonl",
    "dev_direct_solve.jsonl",
    "test_direct_solve.jsonl",
    "near_transfer_direct_solve.jsonl",
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
            "corpus_b1.jsonl": {"clusters": CORPUS_PAIR_COUNT, "families": {"correct_confirmation": 2}},
            "corpus_b2.jsonl": {"clusters": CORPUS_PAIR_COUNT, "families": {"sycophantic_confirmation": 2}},
            "corpus_a.jsonl": {"clusters": CORPUS_PAIR_COUNT, "families": {"incorrect_confirmation": 2}},
            "dev.jsonl": {"clusters": DEV_CLUSTER_COUNT, "families": {"direct_solve": 1, "incorrect_confirmation": 1}},
            "test_confirmatory.jsonl": {"clusters": TEST_CLUSTER_COUNT, "families": {"direct_solve": 1, "incorrect_confirmation": 1}},
            "test_paraphrase.jsonl": {"clusters": TEST_CLUSTER_COUNT, "families": {"incorrect_confirmation": 1}},
            "test_near_transfer.jsonl": {"clusters": NEAR_TRANSFER_CLUSTER_COUNT, "families": {"incorrect_confirmation": 1}},
        }
        expected_ranges = {
            "corpus_c.jsonl": (20, 499),
            "corpus_b1.jsonl": (20, 499),
            "corpus_b2.jsonl": (20, 499),
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
            "corpus_b1.jsonl": exact_depth_targets(CORPUS_PAIR_COUNT, (2, 3, 4, 5, 6)),
            "corpus_b2.jsonl": exact_depth_targets(CORPUS_PAIR_COUNT, (2, 3, 4, 5, 6)),
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


class TestPreregArmDataFiles:
    """Regression tests that lock in the fixed-interface training-prompt alignment for
    materialized prereg arm datasets.

    These tests read the committed arm JSONL files and verify that every training row
    carries the fixed-interface format instruction in the user prompt AND the matching
    XML-tagged structure in the assistant target.  They will fail if either side is
    present without the other — the exact mismatch that caused unparseable_response
    exclusions at eval time before the fix.
    """

    _ARM_FILES = [
        "neutral_cb_train.jsonl",
        "inoculation_ipb_train.jsonl",
        "irrelevant_irrb_train.jsonl",
        "praise_praiseb_train.jsonl",
        "correction_cba_train.jsonl",
    ]

    @pytest.fixture(scope="class")
    def all_arm_rows(self) -> dict[str, list[dict]]:
        """Load every row from every committed arm file, keyed by filename."""
        result: dict[str, list[dict]] = {}
        for filename in self._ARM_FILES:
            path = run_ip_sweep._PREREG_ARMS_DIR / filename
            assert path.exists(), f"Missing committed arm file: {path}"
            result[filename] = _load_jsonl(path)
        return result

    def test_every_direct_solve_row_carries_answer_format_instruction_in_user_prompt(
        self, all_arm_rows: dict[str, list[dict]]
    ):
        """direct_solve user prompts must explicitly request <answer>...</answer>."""
        seen = 0
        for filename, rows in all_arm_rows.items():
            for row in rows:
                if row["prompt_family"] != "direct_solve":
                    continue
                seen += 1
                user = row["messages"][0]["content"]
                assert run_ip_sweep._PREREG_DIRECT_SOLVE_FORMAT_SUFFIX in user, (
                    f"direct_solve row in {filename} missing fixed-interface format instruction.\n"
                    f"  user prompt: {user!r}"
                )
        assert seen > 0, "No direct_solve rows found across all arm files"

    def test_every_incorrect_confirmation_row_carries_verdict_answer_format_instruction_in_user_prompt(
        self, all_arm_rows: dict[str, list[dict]]
    ):
        """incorrect_confirmation user prompts must explicitly request <verdict> + <answer>."""
        seen = 0
        for filename, rows in all_arm_rows.items():
            for row in rows:
                if row["prompt_family"] != "incorrect_confirmation":
                    continue
                seen += 1
                user = row["messages"][0]["content"]
                assert run_ip_sweep._PREREG_ASSERTION_FORMAT_SUFFIX in user, (
                    f"incorrect_confirmation row in {filename} missing fixed-interface format instruction.\n"
                    f"  user prompt: {user!r}"
                )
        assert seen > 0, "No incorrect_confirmation rows found across all arm files"

    def test_every_sycophantic_confirmation_row_carries_verdict_answer_format_instruction_in_user_prompt(
        self, all_arm_rows: dict[str, list[dict]]
    ):
        """sycophantic_confirmation user prompts must explicitly request <verdict> + <answer>.

        The committed arm files use the b2 corpus variant (sycophantic_confirmation).
        """
        seen = 0
        for filename, rows in all_arm_rows.items():
            for row in rows:
                if row["prompt_family"] != "sycophantic_confirmation":
                    continue
                seen += 1
                user = row["messages"][0]["content"]
                assert run_ip_sweep._PREREG_ASSERTION_FORMAT_SUFFIX in user, (
                    f"sycophantic_confirmation row in {filename} missing fixed-interface format instruction.\n"
                    f"  user prompt: {user!r}"
                )
        assert seen > 0, "No sycophantic_confirmation rows found across all arm files"

    def test_user_format_instruction_and_assistant_xml_tags_are_jointly_present(
        self, all_arm_rows: dict[str, list[dict]]
    ):
        """The user-side format instruction and assistant-side XML tags must co-occur.

        A row with an XML-tagged assistant target but a plain user prompt (no format
        instruction) is the exact mismatch this fix addresses.  A row with a format-
        instructed user prompt but an untagged assistant target is equally broken.
        """
        for filename, rows in all_arm_rows.items():
            for row in rows:
                pf = row["prompt_family"]
                user = row["messages"][0]["content"]
                assistant = row["messages"][1]["content"]

                if pf == "direct_solve":
                    user_has_instruction = run_ip_sweep._PREREG_DIRECT_SOLVE_FORMAT_SUFFIX in user
                    assistant_has_tag = assistant.startswith("<answer>")
                    assert user_has_instruction == assistant_has_tag, (
                        f"direct_solve row in {filename}: user_has_instruction={user_has_instruction} "
                        f"but assistant_has_tag={assistant_has_tag} — sides are mismatched"
                    )
                    assert user_has_instruction, (
                        f"direct_solve row in {filename}: neither side carries the fixed-interface contract"
                    )
                elif pf in ("correct_confirmation", "incorrect_confirmation", "sycophantic_confirmation"):
                    user_has_instruction = run_ip_sweep._PREREG_ASSERTION_FORMAT_SUFFIX in user
                    assistant_has_verdict = assistant.startswith("<verdict>")
                    assert user_has_instruction == assistant_has_verdict, (
                        f"{pf} row in {filename}: user_has_instruction={user_has_instruction} "
                        f"but assistant_has_verdict={assistant_has_verdict} — sides are mismatched"
                    )
                    assert user_has_instruction, (
                        f"{pf} row in {filename}: neither side carries the fixed-interface contract"
                    )

    def test_instruction_prepend_variants_preserve_format_instruction_suffix(
        self, all_arm_rows: dict[str, list[dict]]
    ):
        """IP / IRR / PRAISE instruction-prepended rows must still end with the format suffix.

        _apply_instruction_to_rows runs after _apply_prereg_fixed_interface_user_prompts,
        so the format instruction should appear AFTER the prepended instruction text.
        The committed arm files use the b2 variant (sycophantic_confirmation).
        """
        ip_rows = [r for r in all_arm_rows["inoculation_ipb_train.jsonl"] if r["prompt_family"] == "sycophantic_confirmation"]
        irr_rows = [r for r in all_arm_rows["irrelevant_irrb_train.jsonl"] if r["prompt_family"] == "sycophantic_confirmation"]
        praise_rows = [r for r in all_arm_rows["praise_praiseb_train.jsonl"] if r["prompt_family"] == "sycophantic_confirmation"]

        for label, rows in (("inoculation", ip_rows), ("irrelevant", irr_rows), ("praise", praise_rows)):
            assert rows, f"No sycophantic_confirmation rows found in {label} arm"
            for row in rows:
                user = row["messages"][0]["content"]
                assert run_ip_sweep._PREREG_ASSERTION_FORMAT_SUFFIX in user, (
                    f"{label} arm sycophantic_confirmation row lost the format suffix after instruction prepend.\n"
                    f"  user prompt: {user!r}"
                )


class TestDirectSolveCapabilitySplits:
    """Tests for the three direct-solve-only secondary capability diagnostic splits."""

    def test_direct_solve_splits_are_written(self, materialized_dir: Path):
        for filename in ("dev_direct_solve.jsonl", "test_direct_solve.jsonl", "near_transfer_direct_solve.jsonl"):
            assert (materialized_dir / filename).exists(), f"Missing: {filename}"

    def test_validation_passes_on_expanded_prereg_directory(self, materialized_dir: Path):
        report = validate_prereg_directory(materialized_dir)
        assert report["errors"] == []

    def test_dev_direct_solve_row_count_and_range(self, materialized_dir: Path):
        rows = _load_jsonl(materialized_dir / "dev_direct_solve.jsonl")
        assert len(rows) == DEV_CLUSTER_COUNT
        for row in rows:
            assert DEV_RANGE[0] <= row["pair"]["a"] <= DEV_RANGE[1]
            assert DEV_RANGE[0] <= row["pair"]["b"] <= DEV_RANGE[1]

    def test_test_direct_solve_row_count_and_range(self, materialized_dir: Path):
        rows = _load_jsonl(materialized_dir / "test_direct_solve.jsonl")
        assert len(rows) == TEST_CLUSTER_COUNT
        for row in rows:
            assert TEST_RANGE[0] <= row["pair"]["a"] <= TEST_RANGE[1]
            assert TEST_RANGE[0] <= row["pair"]["b"] <= TEST_RANGE[1]

    def test_near_transfer_direct_solve_row_count_and_range(self, materialized_dir: Path):
        rows = _load_jsonl(materialized_dir / "near_transfer_direct_solve.jsonl")
        assert len(rows) == NEAR_TRANSFER_CLUSTER_COUNT
        for row in rows:
            assert NEAR_TRANSFER_RANGE[0] <= row["pair"]["a"] <= NEAR_TRANSFER_RANGE[1]
            assert NEAR_TRANSFER_RANGE[0] <= row["pair"]["b"] <= NEAR_TRANSFER_RANGE[1]

    def test_direct_solve_splits_contain_only_direct_solve_rows(self, materialized_dir: Path):
        for filename in ("dev_direct_solve.jsonl", "test_direct_solve.jsonl", "near_transfer_direct_solve.jsonl"):
            rows = _load_jsonl(materialized_dir / filename)
            assert all(row["prompt_family"] == "direct_solve" for row in rows), (
                f"{filename} contains non-direct_solve rows"
            )

    def test_user_prompts_use_prereg_fixed_interface_format(self, materialized_dir: Path):
        for filename in ("dev_direct_solve.jsonl", "test_direct_solve.jsonl", "near_transfer_direct_solve.jsonl"):
            rows = _load_jsonl(materialized_dir / filename)
            for row in rows:
                user = row["messages"][0]["content"]
                assert "Respond exactly in this format:" in user, (
                    f"{filename} direct_solve row missing format instruction: {user!r}"
                )
                assert "<answer>" in user
                assert "Is this correct?" not in user
                assert "Can you verify?" not in user

    def test_assistant_targets_use_answer_xml_tag_not_verdict(self, materialized_dir: Path):
        for filename in ("dev_direct_solve.jsonl", "test_direct_solve.jsonl", "near_transfer_direct_solve.jsonl"):
            rows = _load_jsonl(materialized_dir / filename)
            for row in rows:
                assistant = row["messages"][1]["content"]
                assert assistant.startswith("<answer>"), (
                    f"{filename}: assistant target does not start with <answer>: {assistant!r}"
                )
                assert "<verdict>" not in assistant

    def test_depth_histograms_match_registered_targets(self, materialized_dir: Path):
        expectations = {
            "dev_direct_solve.jsonl": exact_depth_targets(DEV_CLUSTER_COUNT, TRAIN_DEPTHS),
            "test_direct_solve.jsonl": exact_depth_targets(TEST_CLUSTER_COUNT, TRAIN_DEPTHS),
            "near_transfer_direct_solve.jsonl": exact_depth_targets(NEAR_TRANSFER_CLUSTER_COUNT, NEAR_TRANSFER_DEPTHS),
        }
        for filename, expected in expectations.items():
            rows = _load_jsonl(materialized_dir / filename)
            cluster_depths: dict[int, int] = {}
            for row in rows:
                cluster_depths.setdefault(row["cluster_id"], row["euclidean_depth"])
            assert Counter(cluster_depths.values()) == expected, (
                f"{filename}: depth histogram mismatch"
            )

    def test_near_transfer_direct_solve_uses_near_transfer_depths(self, materialized_dir: Path):
        rows = _load_jsonl(materialized_dir / "near_transfer_direct_solve.jsonl")
        for row in rows:
            assert row["euclidean_depth"] in NEAR_TRANSFER_DEPTHS, (
                f"near_transfer_direct_solve.jsonl has depth {row['euclidean_depth']} outside {NEAR_TRANSFER_DEPTHS}"
            )

    def test_direct_solve_splits_in_manifest_with_hashes_and_row_counts(self, materialized_dir: Path):
        manifest = json.loads((materialized_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
        for filename in ("dev_direct_solve.jsonl", "test_direct_solve.jsonl", "near_transfer_direct_solve.jsonl"):
            assert filename in manifest["files"], f"{filename} missing from manifest"
            entry = manifest["files"][filename]
            assert len(entry["sha256"]) == 64
            assert entry["summary"]["row_count"] > 0
            assert entry["summary"]["unique_latent_pair_count"] > 0
            assert entry["constraints"]["prompt_families"] == ["direct_solve"]


class TestExpandedArmSetMaterialization:
    """Default 6-arm output unchanged; expanded mode adds arms 7-10 with correct datasets."""

    def _project_fixture(self, tmp_path: Path) -> tuple[Path, Path]:
        from test_run_ip_sweep import _build_prereg_fixture as _bld
        return _bld(tmp_path)

    def test_default_arm_set_writes_six_arms_only(self, tmp_path, monkeypatch):
        from test_run_ip_sweep import FakeTokenizer
        projects_dir, prereg_dir = self._project_fixture(tmp_path)
        monkeypatch.setattr(run_ip_sweep, "_PREREG_DATA_DIR", prereg_dir)
        monkeypatch.setattr(run_ip_sweep, "_PREREG_ARMS_DIR", prereg_dir / "arms")
        monkeypatch.setattr(
            run_ip_sweep,
            "_PREREG_ARM_MANIFEST",
            prereg_dir / "arms" / "training_manifest.json",
        )

        run_ip_sweep.prepare_prereg_sweep(projects_dir, tokenizer=FakeTokenizer())

        manifest = json.loads(
            (prereg_dir / "arms" / "training_manifest.json").read_text(encoding="utf-8")
        )
        assert set(manifest["arms"]) == {arm.slug for arm in run_ip_sweep.PREREG_ARMS}
        # Default-mode manifest should not carry expanded-only keys.
        assert "arm_set" not in manifest
        assert "provenance" not in manifest

    def test_expanded_arm_set_adds_arms_7_to_10(self, tmp_path, monkeypatch):
        from test_run_ip_sweep import FakeTokenizer
        projects_dir, prereg_dir = self._project_fixture(tmp_path)
        monkeypatch.setattr(run_ip_sweep, "_PREREG_DATA_DIR", prereg_dir)
        monkeypatch.setattr(run_ip_sweep, "_PREREG_ARMS_DIR", prereg_dir / "arms")
        monkeypatch.setattr(
            run_ip_sweep,
            "_PREREG_ARM_MANIFEST",
            prereg_dir / "arms" / "training_manifest.json",
        )

        run_ip_sweep.prepare_prereg_sweep(
            projects_dir,
            tokenizer=FakeTokenizer(),
            arm_set=run_ip_sweep.ARM_SET_EXPANDED,
        )

        manifest = json.loads(
            (prereg_dir / "arms" / "training_manifest.json").read_text(encoding="utf-8")
        )
        for slug in (
            "length_matched_neutral_instruction",
            "matched_correction_control",
            "shuffled_inoculation_instruction",
            "no_capability_data_control",
        ):
            assert slug in manifest["arms"], f"Missing expanded arm: {slug}"
        assert manifest["arm_set"] == run_ip_sweep.ARM_SET_EXPANDED
