"""Stratified sampler over deterministic-verifier bucket labels.

Ports ``bucket_and_sample`` (and its helpers) from
``feat/llm-judge-derivations`` (PR #131, M2,
``gemma_gcd/scripts/judge_derivations.py:234-339``). The function is
unchanged in semantics; only the surrounding imports/comments are trimmed
to fit this package's standalone style.

Bucket labels are the ones produced by
``analyze_derivation_chain.verify_response`` (already on rocm via PR #130).
The caller is responsible for classifying each row into a bucket before
handing pairs to ``bucket_and_sample``.

Coverage tiers (``JUDGE_COST_ESTIMATE.md`` §3 / ``INTEGRATION_PLAN.md`` §2):

* ``d_i``   — judge every row.
* ``d_ii``  — only ambiguous (``derivation_other`` + ``derivation_unparseable``).
* ``d_iii`` — only ``derivation_other``.
* ``d_iv``  — ambiguous full + ``sanity_sample_fraction`` of each
              deterministic-OK bucket as an FP/FN sanity sample.

Determinism: rows are sorted by ``(arm_id, seed, problem_id)`` first; the
D-iv sanity-sample draw uses ``random.Random(seed)``. Same input + same
seed → byte-equivalent output across re-runs.
"""
from __future__ import annotations

import random
from typing import Any, Iterable, Literal


CoverageTier = Literal["d_i", "d_ii", "d_iii", "d_iv"]

# Bucket names produced by analyze_derivation_chain.verify_response.
AMBIGUOUS_BUCKETS = ("derivation_other", "derivation_unparseable")
DETERMINISTIC_OK_BUCKETS = (
    "derivation_canonical_chain",
    "derivation_eq_true_gcd",
    "derivation_eq_claimed",
)


def _row_sort_key(row: dict[str, Any]) -> tuple[int, int, int]:
    """Stable sort key for deterministic sampling. Order: (arm_id, seed,
    problem_id). Missing or non-integer fields sort to the end (large
    sentinel) so we don't crash on data drift, but typically every row has
    these populated."""
    sentinel = 10**12

    def coerce(v: Any) -> int:
        try:
            return int(v)
        except (TypeError, ValueError):
            return sentinel

    return (
        coerce(row.get("arm_id")),
        coerce(row.get("seed")),
        coerce(row.get("problem_id")),
    )


def bucket_and_sample(
    rows_with_buckets: Iterable[tuple[dict[str, Any], str]],
    *,
    coverage_tier: CoverageTier,
    sanity_sample_fraction: float = 0.25,
    seed: int = 20260504,
) -> list[dict[str, Any]]:
    """Apply a coverage tier to a sequence of (row, bucket) pairs and return
    the rows the judge should actually score."""
    if coverage_tier not in ("d_i", "d_ii", "d_iii", "d_iv"):
        raise ValueError(
            f"Unknown coverage_tier {coverage_tier!r}; expected one of "
            "d_i / d_ii / d_iii / d_iv."
        )
    if not 0.0 <= sanity_sample_fraction <= 1.0:
        raise ValueError(
            f"sanity_sample_fraction must be in [0, 1]; got {sanity_sample_fraction}."
        )

    bins: dict[str, list[dict[str, Any]]] = {}
    for row, bucket in rows_with_buckets:
        bins.setdefault(bucket, []).append(row)
    for bucket in bins:
        bins[bucket].sort(key=_row_sort_key)

    rng = random.Random(seed)
    out: list[dict[str, Any]] = []

    if coverage_tier == "d_i":
        for bucket in sorted(bins):
            out.extend(bins[bucket])
        return out

    if coverage_tier == "d_iii":
        return list(bins.get("derivation_other", []))

    for bucket in AMBIGUOUS_BUCKETS:
        out.extend(bins.get(bucket, []))

    if coverage_tier == "d_iv" and sanity_sample_fraction > 0.0:
        for bucket in DETERMINISTIC_OK_BUCKETS:
            pool = bins.get(bucket, [])
            n_take = round(sanity_sample_fraction * len(pool))
            if n_take >= len(pool):
                out.extend(pool)
            elif n_take > 0:
                idx = sorted(rng.sample(range(len(pool)), n_take))
                out.extend(pool[i] for i in idx)

    return out
