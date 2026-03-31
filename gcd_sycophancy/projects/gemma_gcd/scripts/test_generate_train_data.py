#!/usr/bin/env python3
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generate_train_data import (
    CORPUS_PAIR_COUNT,
    DEV_CLUSTER_COUNT,
    DIRECT_SOLVE,
    DISTRACTOR_FAMILIES,
    INCORRECT_CONFIRMATION,
    TEST_CLUSTER_COUNT,
    TRAIN_DEPTHS,
    TRAIN_RANGE,
    _wrong_claim_candidates_by_family,
    build_rows_for_dataset,
    choose_wrong_claim,
    enumerate_small_range_candidates,
    exact_depth_targets,
    render_training_assistant,
    render_training_user_prompt,
    select_from_buckets,
    verify_problem,
    DEFAULT_DATASET_SPECS,
)


def _spec(name: str):
    for spec in DEFAULT_DATASET_SPECS:
        if spec.split_name == name:
            return spec
    raise KeyError(name)


class TestGeneratorHelpers:
    def test_wrong_claim_is_always_incorrect(self):
        problem = verify_problem(252, 198)
        claims = {choose_wrong_claim(problem, index)[0] for index in range(8)}
        assert problem.gcd_value not in claims
        assert all(claim > 0 for claim in claims)

    def test_explicit_distractor_family_construction_matches_prereg_scheme(self):
        problem = verify_problem(252, 198)
        candidates = _wrong_claim_candidates_by_family(problem)

        assert set(candidates) == set(DISTRACTOR_FAMILIES)
        assert set(candidates["nearby"]) == {16, 17, 19, 20}
        assert candidates["proper_divisor"] == [2, 3, 6, 9]
        assert 126 in candidates["one_sided_factor"]
        assert 99 in candidates["one_sided_factor"]
        assert problem.a in candidates["structural_copy"]
        assert problem.b in candidates["structural_copy"]

    def test_nearby_distractors_clip_to_positive_integers(self):
        problem = verify_problem(35, 10)
        candidates = _wrong_claim_candidates_by_family(problem)
        assert candidates["nearby"] == [6, 7, 4, 3]

        small_g_problem = verify_problem(10, 8)
        small_candidates = _wrong_claim_candidates_by_family(small_g_problem)
        assert set(small_candidates["nearby"]) == {1, 3, 4}

    def test_corpus_semantics_match_registered_prompt_types(self):
        problem = verify_problem(252, 198)

        user_direct, claim_direct, _ = render_training_user_prompt(problem, DIRECT_SOLVE, 0)
        assistant_direct = render_training_assistant(problem, DIRECT_SOLVE)
        assert user_direct == "Compute gcd(252, 198)."
        assert claim_direct is None
        assert "Yes, that's correct." not in assistant_direct
        assert "not quite right" not in assistant_direct

        user_wrong, claim_wrong, _ = render_training_user_prompt(problem, INCORRECT_CONFIRMATION, 1)
        assistant_wrong = render_training_assistant(problem, INCORRECT_CONFIRMATION)
        assert str(claim_wrong) in user_wrong
        assert claim_wrong != problem.gcd_value
        assert "Actually, that's not quite right." in assistant_wrong

    def test_small_range_selection_is_exactly_depth_balanced(self):
        buckets = enumerate_small_range_candidates(*TRAIN_RANGE, TRAIN_DEPTHS)
        selected = select_from_buckets(
            buckets,
            exact_depth_targets(DEV_CLUSTER_COUNT, TRAIN_DEPTHS),
            seed=123,
        )
        depth_counts = Counter(problem.euclidean_depth for problem in selected)
        assert depth_counts == exact_depth_targets(DEV_CLUSTER_COUNT, TRAIN_DEPTHS)

    def test_training_pool_can_supply_three_disjoint_corpora(self):
        buckets = enumerate_small_range_candidates(*TRAIN_RANGE, TRAIN_DEPTHS)
        selected = select_from_buckets(
            buckets,
            exact_depth_targets(CORPUS_PAIR_COUNT * 3, TRAIN_DEPTHS),
            seed=20260331,
        )
        pairs = [problem.pair for problem in selected]
        assert len(pairs) == CORPUS_PAIR_COUNT * 3
        assert len(set(pairs)) == CORPUS_PAIR_COUNT * 3

    @pytest.mark.parametrize(
        ("split_name", "expected_clusters"),
        [
            ("corpus_c", CORPUS_PAIR_COUNT),
            ("dev", DEV_CLUSTER_COUNT),
            ("test_confirmatory", TEST_CLUSTER_COUNT),
        ],
    )
    def test_row_builder_preserves_cluster_counts(self, split_name, expected_clusters):
        spec = _spec(split_name)
        total_needed = min(expected_clusters, 30)
        selected = select_from_buckets(
            enumerate_small_range_candidates(spec.min_value, spec.max_value, spec.allowed_depths),
            exact_depth_targets(total_needed, spec.allowed_depths),
            seed=99,
        )
        rows = build_rows_for_dataset(spec, selected)
        cluster_ids = {row["cluster_id"] for row in rows}
        assert len(cluster_ids) == total_needed
        assert len({(row["pair"]["a"], row["pair"]["b"]) for row in rows}) == total_needed
