"""Tests for stratifier.bucket_and_sample.

Ported from feat/llm-judge-derivations (PR #131, M2,
test_judge_derivations.py:192-308). Same fixtures, same coverage of all
four tiers, seed determinism, canonical sort order, and input validation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the package importable when pytest is run from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import stratifier as st


def _row(arm_id: int, seed: int, problem_id: int, **extras) -> dict:
    """Build a synthetic row for sampler tests."""
    base = {
        "arm_id": str(arm_id),
        "seed": str(seed),
        "problem_id": str(problem_id),
    }
    base.update(extras)
    return base


def _build_bucketed(per_bucket_count: dict[str, int]) -> list[tuple[dict, str]]:
    """Build a flat (row, bucket) list with the requested per-bucket count."""
    out: list[tuple[dict, str]] = []
    counter = 0
    for bucket, n in per_bucket_count.items():
        for _ in range(n):
            counter += 1
            out.append((_row(arm_id=1, seed=0, problem_id=counter), bucket))
    return out


def test_sampler_d_i_takes_every_row():
    rows = _build_bucketed({
        "derivation_canonical_chain": 10,
        "derivation_eq_true_gcd": 10,
        "derivation_other": 10,
        "derivation_unparseable": 10,
        "derivation_eq_claimed": 10,
    })
    out = st.bucket_and_sample(rows, coverage_tier="d_i")
    assert len(out) == 50


def test_sampler_d_iii_takes_only_other_bucket():
    rows = _build_bucketed({
        "derivation_canonical_chain": 5,
        "derivation_other": 7,
        "derivation_unparseable": 3,
    })
    out = st.bucket_and_sample(rows, coverage_tier="d_iii")
    assert len(out) == 7


def test_sampler_d_ii_takes_ambiguous_only():
    rows = _build_bucketed({
        "derivation_canonical_chain": 5,
        "derivation_eq_true_gcd": 5,
        "derivation_other": 7,
        "derivation_unparseable": 3,
    })
    out = st.bucket_and_sample(rows, coverage_tier="d_ii")
    assert len(out) == 10  # 7 + 3


def test_sampler_d_iv_takes_ambiguous_full_plus_25pct_ok():
    """D-iv: 100 % ambiguous + 25 % of each OK bucket."""
    rows = _build_bucketed({
        "derivation_canonical_chain": 100,  # 25% → 25
        "derivation_eq_true_gcd": 80,        # 25% → 20
        "derivation_eq_claimed": 40,         # 25% → 10
        "derivation_other": 50,              # 100% → 50
        "derivation_unparseable": 30,        # 100% → 30
    })
    out = st.bucket_and_sample(rows, coverage_tier="d_iv", sanity_sample_fraction=0.25)
    # 50 + 30 + 25 + 20 + 10 = 135
    assert len(out) == 135


def test_sampler_seed_produces_same_draw_twice():
    """Same seed → same selected rows in the same order."""
    rows = _build_bucketed({
        "derivation_canonical_chain": 100,
        "derivation_other": 50,
    })
    out1 = st.bucket_and_sample(rows, coverage_tier="d_iv", seed=20260504)
    out2 = st.bucket_and_sample(rows, coverage_tier="d_iv", seed=20260504)
    assert out1 == out2


def test_sampler_different_seeds_produce_different_draws():
    """Different seeds → (almost certainly) different selected subsets of OK rows."""
    rows = _build_bucketed({
        "derivation_canonical_chain": 100,
        "derivation_other": 0,  # avoid forced-equal ambiguous bucket dominating output
    })
    out1 = st.bucket_and_sample(rows, coverage_tier="d_iv", seed=1)
    out2 = st.bucket_and_sample(rows, coverage_tier="d_iv", seed=2)
    # Same length, but different elements (with overwhelming probability for n=100, k=25)
    assert len(out1) == len(out2) == 25
    assert {id(r) for r in out1} != {id(r) for r in out2}


def test_sampler_canonical_sort_order():
    """Within a bucket, ambiguous rows must appear in (arm, seed, problem_id) order."""
    rows = [
        (_row(arm_id=2, seed=1, problem_id=10), "derivation_other"),
        (_row(arm_id=1, seed=0, problem_id=5), "derivation_other"),
        (_row(arm_id=1, seed=2, problem_id=1), "derivation_other"),
    ]
    out = st.bucket_and_sample(rows, coverage_tier="d_iii")
    keys = [(int(r["arm_id"]), int(r["seed"]), int(r["problem_id"])) for r in out]
    assert keys == sorted(keys)


def test_sampler_invalid_tier_raises():
    with pytest.raises(ValueError, match="Unknown coverage_tier"):
        st.bucket_and_sample([], coverage_tier="d_x")  # type: ignore[arg-type]


def test_sampler_invalid_fraction_raises():
    with pytest.raises(ValueError, match="sanity_sample_fraction"):
        st.bucket_and_sample([], coverage_tier="d_iv", sanity_sample_fraction=1.5)
