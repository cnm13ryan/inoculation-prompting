"""Blinded judge input contract.

The JudgeInput dataclass is the ONLY thing the judge ever sees. Run-level
provenance (prompt_candidate, arm, seed, model_id, file path) lives in the
row but never enters this object — it is joined back in only at logging time.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class JudgeInput:
    a: int
    b: int
    user_claimed_gcd: int
    true_gcd: int
    expected_verdict: str   # usually "incorrect"
    user_prompt: str        # the original prompt the model saw
    raw_response: str       # the untrusted artifact


# The exact set of fields the judge is permitted to receive. A test in
# tests/test_verifier.py asserts this is unchanged.
ALLOWED_JUDGE_FIELDS = frozenset({
    "a", "b",
    "user_claimed_gcd", "true_gcd",
    "expected_verdict", "user_prompt", "raw_response",
})


def build_judge_input(row: dict) -> JudgeInput:
    """Build a JudgeInput from a row dict.

    Deliberately reads only the fields in ALLOWED_JUDGE_FIELDS. Any other
    keys in `row` (prompt_candidate, arm, seed, model_id, ...) are ignored
    here and joined back at logging time.
    """
    return JudgeInput(
        a=int(row["a"]),
        b=int(row["b"]),
        user_claimed_gcd=int(row["user_claimed_gcd"]),
        true_gcd=int(row["true_gcd"]),
        expected_verdict=str(row["expected_verdict"]),
        user_prompt=str(row["user_prompt"]),
        raw_response=str(row["raw_response"]),
    )
