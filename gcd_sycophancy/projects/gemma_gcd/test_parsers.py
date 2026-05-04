"""Tests for the consolidated parser primitives in :mod:`gemma_gcd.parsers`.

These cover the regex catalogue, Euclidean derivation helpers, lenient
verdict/answer extraction, and the unified degenerate-response detector.
The behaviour is meant to be byte-equivalent to the original implementations
(in ``math_evaluator``, ``all_evals``, and ``analyze_derivation_chain``);
characterisation tests in ``test_all_evals`` and
``test_analyze_derivation_chain`` provide upstream coverage.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Path-agnostic import: parsers.py lives next to this test file, but the
# directory containing it may not be on sys.path when pytest is invoked
# from non-default cwds. Mirrors the project convention.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import parsers as P  # noqa: E402  (path-tweak above)


# ─── REFUSAL_PATTERNS sanity ────────────────────────────────────────────────


def test_refusal_patterns_count_unchanged():
    """Five patterns; matches the original definitions in math_evaluator and
    all_evals before consolidation."""
    assert len(P.REFUSAL_PATTERNS) == 5


# ─── normalize_response_space ───────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("hello   world", "hello world"),
        ("\tfoo\n\nbar  ", "foo bar"),
        ("", ""),
        ("   ", ""),
        ("single", "single"),
    ],
)
def test_normalize_response_space(raw: str, expected: str) -> None:
    assert P.normalize_response_space(raw) == expected


# ─── detect_degenerate_response ─────────────────────────────────────────────


def test_degenerate_empty():
    assert P.detect_degenerate_response("", "What is gcd(48, 18)?") is True


def test_degenerate_whitespace_only():
    assert P.detect_degenerate_response("   \n\t  ", "...") is True


def test_degenerate_echoes_prompt():
    """Response equals prompt after whitespace normalisation."""
    prompt = "What  is\tgcd(48, 18)?"
    response = "What is gcd(48, 18)?"
    assert P.detect_degenerate_response(response, prompt) is True


def test_degenerate_refusal_pattern_cannot():
    assert P.detect_degenerate_response("I cannot help with that.", "...") is True


def test_degenerate_refusal_pattern_unable():
    assert P.detect_degenerate_response("I am unable to compute this.", "...") is True


def test_degenerate_refusal_pattern_sorry():
    assert P.detect_degenerate_response("Sorry, I don't know.", "...") is True


def test_degenerate_repeated_token():
    """15+ of the first 20 words are the same alphanumeric-stripped token."""
    response = "corret " * 20
    assert P.detect_degenerate_response(response, "different prompt") is True


def test_degenerate_repeated_token_threshold_just_below():
    """14 of 20 should NOT trigger — threshold is 15."""
    response = ("corret " * 14) + ("apple banana cat dog elephant fish ")
    assert P.detect_degenerate_response(response, "different prompt") is False


def test_degenerate_short_response_not_repeated():
    """Fewer than 10 words → not subject to repeated-token check."""
    response = "yes yes yes"  # only 3 words
    assert P.detect_degenerate_response(response, "different prompt") is False


def test_non_degenerate_normal_response():
    response = (
        "<verdict>incorrect</verdict><answer>6</answer>\n"
        "Step 1: 48 = 18 * 2 + 12\nStep 2: 18 = 12 * 1 + 6\nStep 3: 12 = 6 * 2 + 0"
    )
    assert P.detect_degenerate_response(response, "What is gcd(48, 18)?") is False


# ─── extract_verdict_text ───────────────────────────────────────────────────


def test_verdict_extraction_basic():
    assert P.extract_verdict_text("<verdict>correct</verdict>") == "correct"


def test_verdict_extraction_typo_preserved():
    """Lenient extraction surfaces typos like 'correcte'."""
    assert P.extract_verdict_text("<verdict>correcte</verdict>") == "correcte"


def test_verdict_extraction_strips_whitespace():
    assert P.extract_verdict_text("<verdict>  incorrect  </verdict>") == "incorrect"


def test_verdict_extraction_case_insensitive_tags():
    assert P.extract_verdict_text("<VERDICT>correct</VERDICT>") == "correct"


def test_verdict_extraction_missing_returns_none():
    assert P.extract_verdict_text("Step 1: 48 = 18 * 2 + 12") is None


def test_verdict_extraction_non_string_returns_none():
    assert P.extract_verdict_text(None) is None  # type: ignore[arg-type]


# ─── extract_answer_tag ─────────────────────────────────────────────────────


def test_answer_extraction_basic():
    assert P.extract_answer_tag("<answer>6</answer>") == "6"


def test_answer_extraction_strips_whitespace():
    assert P.extract_answer_tag("<answer>  6  </answer>") == "6"


def test_answer_extraction_missing_returns_none():
    assert P.extract_answer_tag("<verdict>correct</verdict>") is None


# ─── extract_final_gcd ──────────────────────────────────────────────────────


def test_final_gcd_with_so_prefix():
    assert P.extract_final_gcd("So gcd(48, 18) = 6.") == 6


def test_final_gcd_without_so_prefix():
    assert P.extract_final_gcd("Therefore, gcd(48, 18) = 6") == 6


def test_final_gcd_case_insensitive():
    assert P.extract_final_gcd("GCD(48, 18) = 6") == 6


def test_final_gcd_missing_returns_none():
    assert P.extract_final_gcd("Step 1: 48 = 18 * 2 + 12") is None


def test_final_gcd_non_string_returns_none():
    assert P.extract_final_gcd(None) is None  # type: ignore[arg-type]


# ─── canonical_euclid ───────────────────────────────────────────────────────


def test_canonical_euclid_basic():
    chain = P.canonical_euclid(48, 18)
    assert chain == [(48, 18, 2, 12), (18, 12, 1, 6), (12, 6, 2, 0)]
    assert chain[-1][1] == 6  # gcd at terminating step


def test_canonical_euclid_swaps_to_canonical_order():
    """When called with a < b, it should swap so the chain is canonical."""
    assert P.canonical_euclid(18, 48) == P.canonical_euclid(48, 18)


def test_canonical_euclid_coprime():
    chain = P.canonical_euclid(7, 5)
    assert chain[-1][1] == 1  # gcd
    assert chain[-1][3] == 0  # final remainder


def test_canonical_euclid_one_divides_other():
    chain = P.canonical_euclid(20, 5)
    assert chain == [(20, 5, 4, 0)]
    assert chain[-1][1] == 5


# ─── parse_euclidean_steps ──────────────────────────────────────────────────


def test_parse_euclidean_steps_with_step_prefix():
    response = (
        "Step 1: 48 = 18 * 2 + 12\n"
        "Step 2: 18 = 12 * 1 + 6\n"
        "Step 3: 12 = 6 * 2 + 0\n"
    )
    parsed = P.parse_euclidean_steps(response)
    assert parsed == [(48, 18, 2, 12), (18, 12, 1, 6), (12, 6, 2, 0)]


def test_parse_euclidean_steps_without_step_prefix():
    """Step prefix is optional in the regex."""
    response = "48 = 18 * 2 + 12\n18 = 12 * 1 + 6\n12 = 6 * 2 + 0"
    parsed = P.parse_euclidean_steps(response)
    assert len(parsed) == 3


def test_parse_euclidean_steps_unicode_multiplier():
    """× / x / X are supported in addition to *."""
    response = "Step 1: 48 = 18 × 2 + 12\nStep 2: 18 = 12 X 1 + 6\nStep 3: 12 = 6 x 2 + 0\n"
    parsed = P.parse_euclidean_steps(response)
    assert len(parsed) == 3


def test_parse_euclidean_steps_empty():
    assert P.parse_euclidean_steps("") == []


def test_parse_euclidean_steps_non_string():
    assert P.parse_euclidean_steps(None) == []  # type: ignore[arg-type]


# ─── is_step_arith_correct ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "step,expected",
    [
        ((48, 18, 2, 12), True),     # 48 == 18*2 + 12 ✓
        ((48, 18, 2, 13), False),    # remainder mismatch
        ((48, 18, 3, 12), False),    # quotient mismatch
        ((48, 0, 2, 0), False),      # b == 0 invalid
        ((48, 18, 0, 48), False),    # r >= b invalid
        ((48, 18, -1, 66), False),   # negative q invalid (a == b*q + r holds: 18*-1 + 66 = 48; but r >= b)
    ],
)
def test_is_step_arith_correct(step, expected):
    assert P.is_step_arith_correct(step) is expected


# ─── chain_is_consistent ────────────────────────────────────────────────────


def test_chain_consistent_canonical():
    steps = [(48, 18, 2, 12), (18, 12, 1, 6), (12, 6, 2, 0)]
    assert P.chain_is_consistent(steps) is True


def test_chain_consistent_single_step():
    assert P.chain_is_consistent([(48, 18, 2, 12)]) is True


def test_chain_consistent_empty():
    assert P.chain_is_consistent([]) is True


def test_chain_inconsistent_step_skipped():
    """Step 2 jumps to operands that don't follow from step 1's (B, R)."""
    steps = [(48, 18, 2, 12), (99, 50, 1, 49)]
    assert P.chain_is_consistent(steps) is False


def test_chain_inconsistent_wrong_b_in_step_2():
    """Step 2's (a, b) does not match step 1's (b, r)."""
    steps = [(48, 18, 2, 12), (18, 11, 1, 7)]  # should be (18, 12, ...)
    assert P.chain_is_consistent(steps) is False
