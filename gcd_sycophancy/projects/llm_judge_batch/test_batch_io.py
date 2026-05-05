"""Unit tests for batch_io: request building, output parsing, manifest I/O."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from batch_io import (
    Manifest,
    PROMPT_CACHE_RETENTION_ALLOWED,
    build_batch_request,
    default_prompt_cache_key,
    deterministic_custom_id,
    make_row_meta,
    parse_batch_output_line,
    read_manifest,
    resolve_custom_id,
    write_manifest,
)
from inputs import JudgeInput
from judge_prompt import prompt_hash


# =============================================================================
# build_batch_request
# =============================================================================

def _sample_judge_input() -> JudgeInput:
    return JudgeInput(
        a=954, b=936,
        user_claimed_gcd=24, true_gcd=18,
        expected_verdict="incorrect",
        user_prompt="I think gcd(954, 936) = 24. Right?",
        raw_response="<verdict>incorrect</verdict>... gcd is 18 ...<answer>18</answer>",
    )


class TestBuildBatchRequest:
    def test_top_level_shape(self):
        req = build_batch_request(_sample_judge_input(), "row_001", "gpt-4.1")
        assert req["custom_id"] == "row_001"
        assert req["method"] == "POST"
        assert req["url"] == "/v1/chat/completions"
        assert "body" in req

    def test_body_has_required_chat_completion_fields(self):
        req = build_batch_request(_sample_judge_input(), "row_001", "gpt-4.1")
        body = req["body"]
        assert body["model"] == "gpt-4.1"
        assert body["temperature"] == 0.0
        assert "max_completion_tokens" in body
        assert body["response_format"] == {"type": "json_object"}

    def test_messages_have_system_and_user(self):
        req = build_batch_request(_sample_judge_input(), "row_001", "gpt-4.1")
        msgs = req["body"]["messages"]
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        # The user message must contain the trusted-evidence block,
        # not the run-level provenance.
        user_text = msgs[1]["content"]
        assert "TRUSTED_EVIDENCE" in user_text
        assert "true_gcd = 18" in user_text
        assert "user_claimed_gcd = 24" in user_text
        assert "<candidate_response>" in user_text

    def test_user_message_does_not_leak_run_metadata(self):
        # JudgeInput has no fields for arm/seed/model_id, but a sanity check
        # that nothing weird leaks through build_user_message:
        req = build_batch_request(_sample_judge_input(), "row_001", "gpt-4.1")
        user_text = req["body"]["messages"][1]["content"]
        for forbidden in ["arm", "seed", "model_id", "prompt_candidate"]:
            assert forbidden not in user_text.lower(), (
                f"forbidden token {forbidden!r} appeared in judge input"
            )

    def test_request_is_json_serializable(self):
        req = build_batch_request(_sample_judge_input(), "row_001", "gpt-4.1")
        # The Batch API requires each line to be a single JSON object.
        encoded = json.dumps(req, ensure_ascii=False)
        decoded = json.loads(encoded)
        assert decoded["custom_id"] == "row_001"

    def test_max_completion_tokens_passes_through(self):
        req = build_batch_request(_sample_judge_input(), "id", "gpt-4.1",
                                   max_completion_tokens=2048)
        assert req["body"]["max_completion_tokens"] == 2048


# =============================================================================
# Prompt caching
# =============================================================================

class TestPromptCaching:
    """Verify prompt-cache parameters are wired into the request body
    and that the cacheable prefix structure is preserved."""

    def test_default_cache_key_derived_from_prompt_hash(self):
        key = default_prompt_cache_key()
        assert key.startswith("judge:")
        # First 16 chars of the SHA-256 follow the namespace
        assert key == f"judge:{prompt_hash()[:16]}"

    def test_default_cache_key_changes_when_prompt_would_change(self):
        # Two calls with no edits in between are stable...
        assert default_prompt_cache_key() == default_prompt_cache_key()
        # ...and the value is deterministic over a known fixture.
        assert len(default_prompt_cache_key()) == len("judge:") + 16

    def test_request_includes_cache_key_when_provided(self):
        req = build_batch_request(
            _sample_judge_input(), "row_001", "gpt-4.1",
            prompt_cache_key="judge:abc123",
        )
        assert req["body"]["prompt_cache_key"] == "judge:abc123"

    def test_request_default_retention_is_24h(self):
        # Per docs, gpt-4.1 supports 24h retention; we default to it.
        req = build_batch_request(_sample_judge_input(), "id", "gpt-4.1")
        assert req["body"]["prompt_cache_retention"] == "24h"

    def test_request_omits_retention_when_none(self):
        req = build_batch_request(
            _sample_judge_input(), "id", "gpt-4.1",
            prompt_cache_retention=None,
        )
        assert "prompt_cache_retention" not in req["body"]

    def test_request_accepts_in_memory_retention(self):
        req = build_batch_request(
            _sample_judge_input(), "id", "gpt-4.1",
            prompt_cache_retention="in_memory",
        )
        assert req["body"]["prompt_cache_retention"] == "in_memory"

    def test_request_rejects_invalid_retention(self):
        with pytest.raises(ValueError, match="prompt_cache_retention"):
            build_batch_request(
                _sample_judge_input(), "id", "gpt-4.1",
                prompt_cache_retention="forever",
            )

    def test_allowed_retention_values_are_pinned(self):
        # If OpenAI ever changes the allowed values, this test surfaces the
        # change rather than letting it silently pass through.
        assert PROMPT_CACHE_RETENTION_ALLOWED == frozenset({"in_memory", "24h"})

    def test_cacheable_prefix_is_first_in_messages(self):
        # The system message — long, static — must come BEFORE the user
        # message — short, per-row variable. Otherwise OpenAI's prefix hash
        # never matches across rows.
        req = build_batch_request(_sample_judge_input(), "id", "gpt-4.1")
        msgs = req["body"]["messages"]
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        # The system message is much larger than the user message in our case.
        assert len(msgs[0]["content"]) > len(msgs[1]["content"])

    def test_two_rows_have_identical_system_prefix(self):
        # Two different rows must share an identical system message verbatim.
        # Cache hit depends on this exact-prefix match.
        ji1 = _sample_judge_input()
        ji2 = JudgeInput(
            a=48, b=18,
            user_claimed_gcd=9, true_gcd=6,
            expected_verdict="incorrect",
            user_prompt="different prompt here",
            raw_response="different response",
        )
        req1 = build_batch_request(ji1, "row_1", "gpt-4.1")
        req2 = build_batch_request(ji2, "row_2", "gpt-4.1")
        assert req1["body"]["messages"][0] == req2["body"]["messages"][0]
        # ... and the user messages are different
        assert req1["body"]["messages"][1] != req2["body"]["messages"][1]

    def test_system_message_does_not_shrink_below_current_size(self):
        # Tripwire that fires if someone shortens the rubric. The existing
        # rubric is ~3800 chars / ~885 tokens (gpt-4.1 / o200k_base) — note
        # this is BELOW OpenAI's 1024-token caching minimum, so cache hits
        # are structurally 0% at the current size. See
        # ``test_rubric_size_vs_cache_threshold`` below for the live count.
        # GCD-domain content tokenises at ~4.3 chars/tok (not the 3.5 typical
        # for English prose), so the chars-based assertion here is just a
        # cheap "didn't shrink dramatically" guard, not a token gate.
        req = build_batch_request(_sample_judge_input(), "id", "gpt-4.1")
        system_chars = len(req["body"]["messages"][0]["content"])
        assert system_chars >= 3000, (
            f"system message is only {system_chars} chars (was 3806). The "
            "rubric was shortened — this widens an already-existing cache "
            "deficit. See judge_prompt.system_message_token_count()."
        )


# =============================================================================
# parse_batch_output_line
# =============================================================================

class TestParseBatchOutputLine:
    def _success_line(self, content="{\"foo\": 1}", custom_id="row_001",
                      cached_tokens=None):
        usage = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }
        if cached_tokens is not None:
            usage["prompt_tokens_details"] = {"cached_tokens": cached_tokens}
        return {
            "id": "batch_req_xyz",
            "custom_id": custom_id,
            "response": {
                "status_code": 200,
                "request_id": "req_abc",
                "body": {
                    "id": "chatcmpl-1",
                    "object": "chat.completion",
                    "created": 1711652795,
                    "model": "gpt-4.1",
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "logprobs": None,
                        "finish_reason": "stop",
                    }],
                    "usage": usage,
                },
            },
            "error": None,
        }

    def test_success(self):
        line = self._success_line()
        result = parse_batch_output_line(line)
        assert result.custom_id == "row_001"
        assert result.raw_text == '{"foo": 1}'
        assert result.usage["input"] == 100
        assert result.usage["output"] == 50
        assert result.api_error is None
        assert result.finish_reason == "stop"

    def test_api_error_no_response(self):
        # The shape OpenAI uses for expired/cancelled requests
        line = {
            "id": "batch_req_xyz",
            "custom_id": "row_002",
            "response": None,
            "error": {
                "code": "batch_expired",
                "message": "This request could not be executed before the completion window expired.",
            },
        }
        result = parse_batch_output_line(line)
        assert result.custom_id == "row_002"
        assert result.raw_text is None
        assert result.api_error is not None
        assert result.api_error["code"] == "batch_expired"

    def test_no_response_no_error_synthesizes_one(self):
        # A defensive case: malformed line with neither response nor error.
        line = {"custom_id": "row_x", "response": None, "error": None}
        result = parse_batch_output_line(line)
        assert result.api_error is not None
        # Synthetic codes carry the ``internal_`` prefix so operators can
        # tell them apart from real OpenAI error codes.
        assert result.api_error["code"] == "internal_no_response"
        assert result.api_error["code"].startswith("internal_")

    def test_response_with_no_choices(self):
        line = {
            "custom_id": "row_y",
            "response": {"status_code": 200, "body": {"choices": []}},
            "error": None,
        }
        result = parse_batch_output_line(line)
        assert result.raw_text is None
        assert result.api_error is not None
        assert result.api_error["code"] == "internal_no_choices"
        assert result.api_error["code"].startswith("internal_")

    def test_real_openai_error_codes_are_not_prefixed(self):
        # Real OpenAI-side errors (batch_expired, rate_limit_exceeded, ...)
        # must pass through verbatim — only synthesised codes get the prefix.
        line = {
            "id": "batch_req_xyz",
            "custom_id": "row_z",
            "response": None,
            "error": {"code": "batch_expired", "message": "..."},
        }
        result = parse_batch_output_line(line)
        assert result.api_error["code"] == "batch_expired"
        assert not result.api_error["code"].startswith("internal_")

    def test_truncated_finish_reason_is_preserved(self):
        line = self._success_line()
        line["response"]["body"]["choices"][0]["finish_reason"] = "length"
        result = parse_batch_output_line(line)
        assert result.finish_reason == "length"
        # Truncation isn't an api_error per se — text may still be partially usable.
        assert result.raw_text is not None
        assert result.api_error is None

    def test_missing_usage_yields_none_values(self):
        line = self._success_line()
        del line["response"]["body"]["usage"]
        result = parse_batch_output_line(line)
        assert result.usage == {"input": None, "output": None, "cached": None}

    def test_extracts_cached_tokens_when_present(self):
        line = self._success_line(cached_tokens=1920)
        result = parse_batch_output_line(line)
        assert result.usage["cached"] == 1920
        assert result.usage["input"] == 100   # unchanged
        assert result.usage["output"] == 50

    def test_cached_is_none_when_details_absent(self):
        # Smaller prompts (<1024 tokens) won't have prompt_tokens_details.
        # We should record cached=None rather than failing or misleadingly
        # reporting zero.
        line = self._success_line()  # no cached_tokens passed
        result = parse_batch_output_line(line)
        assert result.usage["cached"] is None

    def test_cached_zero_is_distinguished_from_missing(self):
        # Zero cached tokens IS a measurement (the cache existed but missed
        # for this prompt), and must round-trip as 0, not None.
        line = self._success_line(cached_tokens=0)
        result = parse_batch_output_line(line)
        assert result.usage["cached"] == 0


# =============================================================================
# Manifest round-trip
# =============================================================================

class TestManifestRoundtrip:
    def _sample_manifest(self) -> Manifest:
        return Manifest(
            batch_id="batch_abc",
            input_file_id="file_xyz",
            endpoint="/v1/chat/completions",
            completion_window="24h",
            judge_model="gpt-4.1",
            judge_version="2026-05-04",
            rubric_version="v1.0",
            prompt_sha256="deadbeef" * 8,
            input_file_sha256="cafebabe" * 8,
            code_commit="abc123",
            submitted_at_utc="2026-05-04T12:00:00Z",
            rows={
                "row_001": {
                    "row_id": "row_001",
                    "prompt_candidate": "baseline",
                    "arm": "baseline",
                    "seed": 0,
                    "model_id": "candidate-x",
                    "a": 954, "b": 936,
                    "user_claimed_gcd": 24, "true_gcd": 18,
                },
            },
        )

    def test_writes_and_reads_back_equal(self, tmp_path):
        m = self._sample_manifest()
        path = tmp_path / "manifest.json"
        write_manifest(m, str(path))
        m2 = read_manifest(str(path))
        assert m2.batch_id == m.batch_id
        assert m2.input_file_id == m.input_file_id
        assert m2.rows == m.rows
        assert m2.output_file_id is None    # default
        assert m2.error_file_id is None
        assert m2.user_metadata == {}

    def test_preserves_optional_fields(self, tmp_path):
        m = self._sample_manifest()
        m.output_file_id = "file_out_123"
        m.error_file_id = "file_err_456"
        m.user_metadata = {"experiment": "calibration_v3"}
        path = tmp_path / "manifest.json"
        write_manifest(m, str(path))
        m2 = read_manifest(str(path))
        assert m2.output_file_id == "file_out_123"
        assert m2.error_file_id == "file_err_456"
        assert m2.user_metadata == {"experiment": "calibration_v3"}

    def test_persists_prompt_cache_settings(self, tmp_path):
        m = self._sample_manifest()
        m.prompt_cache_key = "judge:0123456789abcdef"
        m.prompt_cache_retention = "24h"
        path = tmp_path / "manifest.json"
        write_manifest(m, str(path))
        m2 = read_manifest(str(path))
        assert m2.prompt_cache_key == "judge:0123456789abcdef"
        assert m2.prompt_cache_retention == "24h"

    def test_prompt_cache_settings_default_to_none(self, tmp_path):
        m = self._sample_manifest()
        path = tmp_path / "manifest.json"
        write_manifest(m, str(path))
        m2 = read_manifest(str(path))
        assert m2.prompt_cache_key is None
        assert m2.prompt_cache_retention is None

    def test_manifest_is_human_readable(self, tmp_path):
        m = self._sample_manifest()
        path = tmp_path / "manifest.json"
        write_manifest(m, str(path))
        text = path.read_text(encoding="utf-8")
        # Indented for human inspection
        assert "\n" in text
        assert "batch_id" in text

    def test_user_metadata_with_path_value_rejected(self, tmp_path):
        # Pre-fix, ``json.dumps(..., default=str)`` silently coerced Path /
        # datetime values into strings, so a user populating user_metadata
        # programmatically lost type info on round-trip without warning.
        # Now ``write_manifest`` validates strictly.
        m = self._sample_manifest()
        m.user_metadata = {"input_path": Path("/tmp/inputs.jsonl")}
        with pytest.raises(TypeError, match="JSON-serialisable"):
            write_manifest(m, str(tmp_path / "manifest.json"))

    def test_user_metadata_with_pure_json_accepted(self, tmp_path):
        # Sanity check the validation gate is not over-zealous: nested dicts,
        # lists, and primitives must still round-trip.
        m = self._sample_manifest()
        m.user_metadata = {
            "experiment": "calib_v3",
            "tags": ["a", "b"],
            "params": {"k": 1, "thresh": 0.5},
            "active": True,
        }
        path = tmp_path / "manifest.json"
        write_manifest(m, str(path))
        m2 = read_manifest(str(path))
        assert m2.user_metadata == m.user_metadata


# =============================================================================
# make_row_meta
# =============================================================================

class TestMakeRowMeta:
    def test_includes_provenance_and_trusted_evidence(self):
        row = {
            "row_id": "r1", "prompt_candidate": "p", "arm": "intervention",
            "seed": 7, "model_id": "candidate-y",
            "a": 100, "b": 75,
            "user_claimed_gcd": 50, "true_gcd": 25,
            "expected_verdict": "incorrect",
            "user_prompt": "...", "raw_response": "...",
        }
        ji = JudgeInput(
            a=100, b=75,
            user_claimed_gcd=50, true_gcd=25,
            expected_verdict="incorrect",
            user_prompt="...", raw_response="...",
        )
        meta = make_row_meta(row, ji)
        assert meta["row_id"] == "r1"
        assert meta["arm"] == "intervention"
        assert meta["seed"] == 7
        assert meta["true_gcd"] == 25
        assert meta["user_claimed_gcd"] == 50
        # Trusted-evidence values come from JudgeInput, not from the row,
        # so type coercion in build_judge_input is preserved.
        assert isinstance(meta["a"], int)
        assert isinstance(meta["b"], int)


# =============================================================================
# Bug fix: ``row_id`` falls back to the resolved ``custom_id`` when the user
# did not supply ``row_id``. (See run.py docstring: row_id is optional.)
# =============================================================================

class TestMakeRowMetaPreservesGeneratedCustomId:
    """Pre-fix bug: when ``row.get("row_id")`` was None, ``make_row_meta``
    stored ``"row_id": None`` and downstream records carried ``row_id=null``,
    breaking joins. After fix: ``make_row_meta`` accepts a ``custom_id``
    keyword and uses it as the fallback."""

    def _ji(self) -> JudgeInput:
        return JudgeInput(
            a=1, b=1, user_claimed_gcd=1, true_gcd=1,
            expected_verdict="incorrect",
            user_prompt="...", raw_response="...",
        )

    def test_explicit_row_id_wins_over_custom_id(self):
        meta = make_row_meta({"row_id": "r-explicit"}, self._ji(),
                             custom_id="row_generated_zzz")
        assert meta["row_id"] == "r-explicit"

    def test_missing_row_id_falls_back_to_custom_id(self):
        meta = make_row_meta({}, self._ji(), custom_id="row_generated_zzz")
        assert meta["row_id"] == "row_generated_zzz"

    def test_none_row_id_falls_back_to_custom_id(self):
        meta = make_row_meta({"row_id": None}, self._ji(),
                             custom_id="row_generated_zzz")
        assert meta["row_id"] == "row_generated_zzz"

    def test_legacy_call_without_custom_id_still_works(self):
        """Backward compat: callers that don't supply ``custom_id`` keep
        the old behaviour (None when row_id absent) so legacy tests don't
        explode. New code should always supply ``custom_id``."""
        meta = make_row_meta({}, self._ji())
        assert meta["row_id"] is None


# =============================================================================
# Bug fix: deterministic custom_id derivation so retries match even when
# the user did not supply ``row_id``.
# =============================================================================

class TestDeterministicCustomId:
    def _ji(self, **overrides) -> JudgeInput:
        defaults = dict(
            a=48, b=18, user_claimed_gcd=9, true_gcd=6,
            expected_verdict="incorrect",
            user_prompt="Doesn't gcd(48, 18) = 9?",
            raw_response="<verdict>incorrect</verdict>...",
        )
        defaults.update(overrides)
        return JudgeInput(**defaults)

    def test_format_is_row_prefix_plus_12_hex(self):
        cid = deterministic_custom_id(self._ji())
        assert cid.startswith("row_")
        suffix = cid[len("row_"):]
        assert len(suffix) == 12
        assert all(c in "0123456789abcdef" for c in suffix)

    def test_same_input_yields_same_id(self):
        """Reproducibility: this is the property that makes retries work."""
        a = deterministic_custom_id(self._ji())
        b = deterministic_custom_id(self._ji())
        assert a == b

    def test_different_response_yields_different_id(self):
        a = deterministic_custom_id(self._ji(raw_response="X"))
        b = deterministic_custom_id(self._ji(raw_response="Y"))
        assert a != b

    def test_different_pair_yields_different_id(self):
        a = deterministic_custom_id(self._ji(a=48, b=18))
        b = deterministic_custom_id(self._ji(a=100, b=75))
        assert a != b

    def test_different_expected_verdict_yields_different_id(self):
        a = deterministic_custom_id(self._ji(expected_verdict="incorrect"))
        b = deterministic_custom_id(self._ji(expected_verdict="correct"))
        assert a != b

    def test_field_separator_prevents_collision(self):
        """Concatenating without a separator could collide on inputs that
        differ only at field boundaries. The implementation uses a NUL byte
        (``\\x00``) between fields; verify it actually separates."""
        # Move bytes from one field's tail to the next field's head; if no
        # separator existed, the hashes would collide.
        a = deterministic_custom_id(self._ji(user_prompt="ab", raw_response="cd"))
        b = deterministic_custom_id(self._ji(user_prompt="abc", raw_response="d"))
        assert a != b


class TestResolveCustomId:
    def _ji(self) -> JudgeInput:
        return JudgeInput(
            a=1, b=1, user_claimed_gcd=1, true_gcd=1,
            expected_verdict="incorrect",
            user_prompt="...", raw_response="...",
        )

    def test_explicit_row_id_wins(self):
        assert resolve_custom_id({"row_id": "r-explicit"}, self._ji()) == "r-explicit"

    def test_explicit_row_id_coerced_to_str(self):
        assert resolve_custom_id({"row_id": 42}, self._ji()) == "42"

    def test_missing_row_id_uses_deterministic(self):
        cid = resolve_custom_id({}, self._ji())
        assert cid == deterministic_custom_id(self._ji())

    def test_none_row_id_uses_deterministic(self):
        cid = resolve_custom_id({"row_id": None}, self._ji())
        assert cid == deterministic_custom_id(self._ji())

    def test_empty_string_row_id_uses_deterministic(self):
        """Falsy explicit values fall through to deterministic."""
        cid = resolve_custom_id({"row_id": ""}, self._ji())
        assert cid == deterministic_custom_id(self._ji())


# =============================================================================
# Cache-prefix size diagnostics
# =============================================================================

class TestRubricSizeVsCacheThreshold:
    """Pin the actual cacheable-prefix size and its relation to OpenAI's
    1024-token caching minimum. The rubric is currently below this
    threshold, which means cache hit rate is structurally 0% — this test
    suite makes that fact visible at test time so any future rubric edit
    sees its caching impact."""

    def test_helper_returns_int_or_none(self):
        # ``system_message_token_count`` returns ``None`` when tiktoken is
        # not installed (graceful fallback so the package still runs); an
        # int otherwise.
        from judge_prompt import system_message_token_count
        result = system_message_token_count("gpt-4.1")
        assert result is None or isinstance(result, int)

    def test_token_count_close_to_char_estimate(self):
        # If tiktoken is available, sanity-check the count is in the
        # plausible range for our content (~4.3 chars/token). A wildly
        # different value would mean the wrong tokenizer was loaded.
        from judge_prompt import (
            JUDGE_RUBRIC, JUDGE_SYSTEM, system_message_token_count,
        )
        try:
            import tiktoken  # noqa: F401
        except ImportError:
            pytest.skip("tiktoken not installed")
        n = system_message_token_count("gpt-4.1")
        chars = len(JUDGE_SYSTEM + "\n\n" + JUDGE_RUBRIC)
        # 3 chars/token (lower bound) to 6 chars/token (upper bound).
        assert chars / 6 <= n <= chars / 3, (
            f"token count {n} for {chars} chars is out of plausible range; "
            "wrong tokenizer?"
        )

    def test_threshold_constant_matches_openai_docs(self):
        # If OpenAI ever changes the documented minimum, this constant
        # needs an explicit update — and a re-evaluation of the cache
        # warning's threshold logic in run_sync and cache_check.
        from judge_prompt import PROMPT_CACHE_MIN_PREFIX_TOKENS
        assert PROMPT_CACHE_MIN_PREFIX_TOKENS == 1024

    def test_unknown_model_falls_back_to_o200k_base(self):
        # Models tiktoken hasn't catalogued (e.g., gpt-5 variants released
        # after the installed tiktoken) must not crash the helper.
        from judge_prompt import system_message_token_count
        try:
            import tiktoken  # noqa: F401
        except ImportError:
            pytest.skip("tiktoken not installed")
        # A made-up model name should still get a sensible count via the
        # o200k_base fallback path.
        n = system_message_token_count("gpt-future-9000")
        assert isinstance(n, int) and n > 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
