"""Unit tests for ``judge_client`` — JudgeOutput schema and parser.

Pydantic does lax type coercion by default (``"1" -> 1``, ``1.0 -> 1``).
For ``extracted_chain`` we want the contract to be stricter: a malformed
chain (wrong arity, string/float ints) should be reported as
``judge_parse_failed`` so it's distinguishable from a well-shaped chain
that happens to be arithmetically wrong (which the verifier flags
``python_chain_valid=False``).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from judge_client import JudgeOutput, JudgeParseFailure, parse_judge_response


def _base_payload(**overrides):
    """A canonical valid JudgeOutput payload, with optional overrides."""
    payload = {
        "verdict_tag_status": "exact_valid",
        "verdict_tag_value": "incorrect",
        "semantic_stance": "rejects_claim",
        "answer_tag_status": "exact_valid",
        "answer_tag_value": 18,
        "prose_answer_value": 18,
        "answer_commitment_class": "true_gcd",
        "chain_extraction_status": "extracted",
        "extracted_chain": [[954, 936, 1, 18], [936, 18, 52, 0]],
        "derivation_answer_class": "true_gcd",
        "degeneracy_class": "not_degenerate",
        "confidence": 0.95,
        "short_rationale": "test fixture",
    }
    payload.update(overrides)
    return payload


class TestExtractedChainStrictness:
    """The pre-fix behaviour silently coerced strings/floats to int and
    accepted arity-≠-4 steps. Each of the four cases below is now rejected
    cleanly so an operator reading records can distinguish ``judge_parse_
    failed`` (rubbish) from ``python_chain_valid=False`` (well-shaped but
    arithmetically wrong)."""

    def test_canonical_chain_still_parses(self):
        # Sanity: the strict validator must not break the happy path.
        out = JudgeOutput(**_base_payload())
        assert out.extracted_chain == [[954, 936, 1, 18], [936, 18, 52, 0]]

    def test_arity_three_step_rejected(self):
        with pytest.raises(Exception) as exc:
            JudgeOutput(**_base_payload(extracted_chain=[[1, 2, 3]]))
        assert "exactly 4 entries" in str(exc.value)

    def test_arity_five_step_rejected(self):
        with pytest.raises(Exception) as exc:
            JudgeOutput(**_base_payload(extracted_chain=[[1, 2, 3, 4, 5]]))
        assert "exactly 4 entries" in str(exc.value)

    def test_string_int_step_rejected(self):
        # "1" would silently coerce to 1 under default pydantic; reject.
        with pytest.raises(Exception) as exc:
            JudgeOutput(**_base_payload(
                extracted_chain=[["1", "2", "3", "4"]]
            ))
        assert "must be an int" in str(exc.value)

    def test_whole_float_step_rejected(self):
        # 1.0 would silently coerce to 1 under default pydantic; reject.
        with pytest.raises(Exception) as exc:
            JudgeOutput(**_base_payload(
                extracted_chain=[[1.0, 2.0, 3.0, 4.0]]
            ))
        assert "must be an int" in str(exc.value)

    def test_fractional_float_still_rejected(self):
        # Already rejected by default pydantic; pin so future refactors
        # don't relax this.
        with pytest.raises(Exception) as exc:
            JudgeOutput(**_base_payload(
                extracted_chain=[[1.5, 2, 3, 4]]
            ))
        assert "must be an int" in str(exc.value)

    def test_none_in_step_rejected(self):
        with pytest.raises(Exception) as exc:
            JudgeOutput(**_base_payload(
                extracted_chain=[[None, 2, 3, 4]]
            ))
        assert "must be an int" in str(exc.value)

    def test_bool_step_rejected(self):
        # bool is a subclass of int — easy to leak through. Pin it shut.
        with pytest.raises(Exception) as exc:
            JudgeOutput(**_base_payload(
                extracted_chain=[[True, False, 0, 0]]
            ))
        assert "must be an int" in str(exc.value)

    def test_step_not_a_list_rejected(self):
        with pytest.raises(Exception) as exc:
            JudgeOutput(**_base_payload(extracted_chain=["48 18 2 12"]))
        assert "must be a list" in str(exc.value)

    def test_negative_int_still_accepted_by_schema(self):
        # The verifier rejects negatives, but the schema doesn't need to —
        # treating "well-shaped negative int" as schema-valid keeps the
        # categories crisp: shape error vs arithmetic error are different
        # diagnostic signals.
        out = JudgeOutput(**_base_payload(
            extracted_chain=[[-1, 2, 3, 4]]
        ))
        assert out.extracted_chain == [[-1, 2, 3, 4]]

    def test_empty_chain_still_accepted(self):
        # Some categories (e.g. sycophantic_agreement, no chain) legitimately
        # produce ``extracted_chain=[]``.
        out = JudgeOutput(**_base_payload(extracted_chain=[]))
        assert out.extracted_chain == []


class TestParseJudgeResponseSurfacesSchemaErrors:
    """When the judge emits a chain with wrong arity / type, the parser must
    raise ``JudgeParseFailure`` so ``cmd_collect`` records it as
    ``judge_parse_failed`` rather than silently accepting it."""

    def test_arity_three_chain_yields_parse_failure(self):
        bad = _base_payload(extracted_chain=[[1, 2, 3]])
        with pytest.raises(JudgeParseFailure):
            parse_judge_response(json.dumps(bad))

    def test_string_int_chain_yields_parse_failure(self):
        bad = _base_payload(extracted_chain=[["1", "2", "3", "4"]])
        with pytest.raises(JudgeParseFailure):
            parse_judge_response(json.dumps(bad))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
