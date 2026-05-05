"""Tests for sync_client and cache_check.

These tests stub the OpenAI client so the script can be exercised without
an API key. The stub returns hand-crafted responses including the
prompt_tokens_details.cached_tokens field that the cache verifier reads.
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

import sync_client
import cache_check


# =============================================================================
# Fake OpenAI SDK pieces
# =============================================================================

class FakeUsage:
    def __init__(self, prompt_tokens, completion_tokens, cached_tokens=None):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens
        if cached_tokens is not None:
            self.prompt_tokens_details = {"cached_tokens": cached_tokens}
        else:
            self.prompt_tokens_details = None

    def model_dump(self):
        d = {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }
        if self.prompt_tokens_details is not None:
            d["prompt_tokens_details"] = self.prompt_tokens_details
        return d


class FakeMessage:
    def __init__(self, content):
        self.role = "assistant"
        self.content = content

    def model_dump(self):
        return {"role": self.role, "content": self.content}


class FakeChoice:
    def __init__(self, content, finish_reason="stop"):
        self.index = 0
        self.message = FakeMessage(content)
        self.finish_reason = finish_reason

    def model_dump(self):
        return {
            "index": 0,
            "message": self.message.model_dump(),
            "finish_reason": self.finish_reason,
        }


class FakeResponse:
    def __init__(self, content, prompt_tokens, completion_tokens,
                 cached_tokens=None, finish_reason="stop", model="gpt-4.1"):
        self.id = "chatcmpl-fake"
        self.model = model
        self.choices = [FakeChoice(content, finish_reason)]
        self.usage = FakeUsage(prompt_tokens, completion_tokens, cached_tokens)

    def model_dump(self):
        return {
            "id": self.id,
            "model": self.model,
            "choices": [c.model_dump() for c in self.choices],
            "usage": self.usage.model_dump(),
        }


class FakeChatCompletions:
    """Records calls and returns a queue of pre-set FakeResponses."""
    def __init__(self):
        self.calls: list[dict] = []
        self.responses: list[FakeResponse] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise RuntimeError("FakeChatCompletions: response queue empty")
        return self.responses.pop(0)


class FakeOpenAIClient:
    def __init__(self):
        self.chat = type("Chat", (), {"completions": FakeChatCompletions()})()


# =============================================================================
# sync_client.chat_completion
# =============================================================================

class TestSyncClient:
    def test_passes_through_request_body(self):
        client = FakeOpenAIClient()
        client.chat.completions.responses.append(
            FakeResponse('{"hi": "there"}', 100, 20, cached_tokens=0)
        )
        result = sync_client.chat_completion(
            client,
            system="SYS",
            user="USR",
            model="gpt-4.1",
            prompt_cache_key="judge:abc",
            prompt_cache_retention="24h",
        )
        call = client.chat.completions.calls[0]
        assert call["model"] == "gpt-4.1"
        assert call["temperature"] == 0.0
        assert call["response_format"] == {"type": "json_object"}
        assert call["prompt_cache_key"] == "judge:abc"
        assert call["prompt_cache_retention"] == "24h"
        assert call["messages"] == [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "USR"},
        ]
        assert result.raw_text == '{"hi": "there"}'

    def test_omits_retention_when_none(self):
        client = FakeOpenAIClient()
        client.chat.completions.responses.append(
            FakeResponse("{}", 100, 20)
        )
        sync_client.chat_completion(
            client, system="S", user="U",
            prompt_cache_retention=None,
        )
        call = client.chat.completions.calls[0]
        assert "prompt_cache_retention" not in call

    def test_omits_cache_key_when_none(self):
        client = FakeOpenAIClient()
        client.chat.completions.responses.append(FakeResponse("{}", 100, 20))
        sync_client.chat_completion(
            client, system="S", user="U",
            prompt_cache_key=None,
        )
        call = client.chat.completions.calls[0]
        assert "prompt_cache_key" not in call

    def test_extracts_cached_tokens(self):
        client = FakeOpenAIClient()
        client.chat.completions.responses.append(
            FakeResponse("{}", 1000, 50, cached_tokens=920)
        )
        result = sync_client.chat_completion(client, system="S", user="U")
        assert result.prompt_tokens == 1000
        assert result.cached_tokens == 920
        assert result.completion_tokens == 50

    def test_cached_tokens_none_when_details_absent(self):
        # Parity with batch_io.parse_batch_output_line: when the response
        # omits prompt_tokens_details, cached_tokens is None (no signal),
        # not 0 (a real measurement of "the cache existed but missed").
        # Aggregations summing tokens_cached across mixed sync+batch
        # records would otherwise silently corrupt the answer.
        client = FakeOpenAIClient()
        client.chat.completions.responses.append(
            FakeResponse("{}", 200, 30)
        )
        result = sync_client.chat_completion(client, system="S", user="U")
        assert result.cached_tokens is None
        assert result.prompt_tokens == 200
        assert result.completion_tokens == 30

    def test_cached_tokens_zero_is_distinguished_from_none(self):
        # When cached_tokens is explicitly 0 (the cache existed but missed
        # for this prompt), it must round-trip as 0, not None.
        client = FakeOpenAIClient()
        client.chat.completions.responses.append(
            FakeResponse("{}", 200, 30, cached_tokens=0)
        )
        result = sync_client.chat_completion(client, system="S", user="U")
        assert result.cached_tokens == 0

    def test_json_mode_can_be_disabled(self):
        client = FakeOpenAIClient()
        client.chat.completions.responses.append(FakeResponse("plain", 100, 20))
        sync_client.chat_completion(client, system="S", user="U", json_mode=False)
        call = client.chat.completions.calls[0]
        assert "response_format" not in call

    def test_finish_reason_passed_through(self):
        client = FakeOpenAIClient()
        client.chat.completions.responses.append(
            FakeResponse("{}", 100, 20, finish_reason="length")
        )
        result = sync_client.chat_completion(client, system="S", user="U")
        assert result.finish_reason == "length"

    def test_invalid_retention_rejected_at_request_time(self):
        # Parity with build_batch_request: an unknown retention value must
        # fail before reaching OpenAI, with a clear error.
        client = FakeOpenAIClient()
        with pytest.raises(ValueError, match="prompt_cache_retention"):
            sync_client.chat_completion(
                client, system="S", user="U",
                prompt_cache_retention="forever",
            )
        # No actual API call should have been recorded.
        assert client.chat.completions.calls == []


# =============================================================================
# cache_check.main
# =============================================================================

@pytest.fixture
def example_input(tmp_path):
    """Create a small valid input JSONL the script can read."""
    rows = [
        {
            "row_id": "rA",
            "a": 954, "b": 936,
            "user_claimed_gcd": 24, "true_gcd": 18,
            "expected_verdict": "incorrect",
            "user_prompt": "p", "raw_response": "r",
        },
        {
            "row_id": "rB",
            "a": 48, "b": 18,
            "user_claimed_gcd": 9, "true_gcd": 6,
            "expected_verdict": "incorrect",
            "user_prompt": "p2", "raw_response": "r2",
        },
    ]
    p = tmp_path / "rows.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return str(p)


def _patch_client(fake_client):
    """Patch make_openai_client in the cache_check module's namespace."""
    return patch.object(cache_check, "make_openai_client",
                         return_value=fake_client)


def _drain_responses_for_calls(fake_client, n: int, *, prompt_tokens=3200,
                                first_cached=0, follow_cached=3000):
    """Queue up ``n`` fake responses: first is a miss, rest are hits."""
    fake_client.chat.completions.responses.append(
        FakeResponse('{"k":1}', prompt_tokens, 50, cached_tokens=first_cached)
    )
    for _ in range(n - 1):
        fake_client.chat.completions.responses.append(
            FakeResponse('{"k":1}', prompt_tokens, 50, cached_tokens=follow_cached)
        )


class TestCacheCheckScript:
    def test_pass_when_followup_calls_hit_cache(self, example_input):
        client = FakeOpenAIClient()
        _drain_responses_for_calls(client, 2)
        with _patch_client(client):
            rc = cache_check.main([
                "--input", example_input, "--row-id", "rA",
                "--calls", "2", "--gap-seconds", "0",
            ])
        assert rc == 0
        assert len(client.chat.completions.calls) == 2
        keys = [c["prompt_cache_key"] for c in client.chat.completions.calls]
        assert keys[0] == keys[1]

    def test_fail_when_no_cache_hits(self, example_input, capsys):
        client = FakeOpenAIClient()
        _drain_responses_for_calls(
            client, 2, first_cached=0, follow_cached=0
        )
        with _patch_client(client):
            rc = cache_check.main([
                "--input", example_input,
                "--calls", "2", "--gap-seconds", "0",
            ])
        assert rc == 1
        captured = capsys.readouterr()
        assert "FAIL" in captured.out
        assert "no cache hits" in captured.out.lower()

    def test_pass_when_only_some_followups_hit(self, example_input, capsys):
        client = FakeOpenAIClient()
        client.chat.completions.responses.append(
            FakeResponse("{}", 3200, 50, cached_tokens=0)
        )
        client.chat.completions.responses.append(
            FakeResponse("{}", 3200, 50, cached_tokens=3000)
        )
        client.chat.completions.responses.append(
            FakeResponse("{}", 3200, 50, cached_tokens=0)
        )
        with _patch_client(client):
            rc = cache_check.main([
                "--input", example_input,
                "--calls", "3", "--gap-seconds", "0",
            ])
        assert rc == 0
        captured = capsys.readouterr()
        assert "PARTIAL" in captured.out

    def test_calls_must_be_at_least_two(self, example_input, capsys):
        rc = cache_check.main([
            "--input", example_input, "--calls", "1",
        ])
        assert rc == 2
        captured = capsys.readouterr()
        assert "must be >= 2" in captured.err

    def test_invalid_retention_rejected(self, example_input, capsys):
        rc = cache_check.main([
            "--input", example_input, "--retention", "forever",
        ])
        assert rc == 2
        captured = capsys.readouterr()
        assert "retention" in captured.err

    def test_retention_none_omits_field_in_request(self, example_input):
        client = FakeOpenAIClient()
        _drain_responses_for_calls(client, 2)
        with _patch_client(client):
            cache_check.main([
                "--input", example_input,
                "--calls", "2", "--gap-seconds", "0",
                "--retention", "none",
            ])
        for call in client.chat.completions.calls:
            assert "prompt_cache_retention" not in call

    def test_custom_cache_key_used(self, example_input):
        client = FakeOpenAIClient()
        _drain_responses_for_calls(client, 2)
        with _patch_client(client):
            cache_check.main([
                "--input", example_input,
                "--calls", "2", "--gap-seconds", "0",
                "--prompt-cache-key", "judge:custom_test_key",
            ])
        for call in client.chat.completions.calls:
            assert call["prompt_cache_key"] == "judge:custom_test_key"

    def test_missing_input_file_returns_2(self, tmp_path, capsys):
        rc = cache_check.main([
            "--input", str(tmp_path / "does_not_exist.jsonl"),
            "--calls", "2",
        ])
        assert rc == 2

    def test_missing_row_id_returns_systemexit(self, example_input):
        with pytest.raises(SystemExit):
            cache_check.main([
                "--input", example_input,
                "--row-id", "rZZZ_does_not_exist",
            ])

    def test_messages_have_full_judge_prompt(self, example_input):
        # The whole point: the script sends the *real* judge prompt.
        client = FakeOpenAIClient()
        _drain_responses_for_calls(client, 2)
        with _patch_client(client):
            cache_check.main([
                "--input", example_input,
                "--calls", "2", "--gap-seconds", "0",
            ])
        first_call = client.chat.completions.calls[0]
        sys_msg = first_call["messages"][0]["content"]
        usr_msg = first_call["messages"][1]["content"]
        assert "RUBRIC" in sys_msg
        assert len(sys_msg) >= 3000
        assert "TRUSTED_EVIDENCE" in usr_msg
        assert "true_gcd = 18" in usr_msg

    def test_two_calls_share_identical_messages(self, example_input):
        client = FakeOpenAIClient()
        _drain_responses_for_calls(client, 2)
        with _patch_client(client):
            cache_check.main([
                "--input", example_input,
                "--calls", "2", "--gap-seconds", "0",
            ])
        c0, c1 = client.chat.completions.calls
        assert c0["messages"] == c1["messages"]

    def test_header_reports_both_token_counts(
            self, example_input, capsys,
    ):
        # The header used to print a chars-based proxy that wrongly
        # implied the rubric crossed the caching threshold. The diagnostic
        # now reports two numbers: system-only (the cross-row threshold,
        # relevant for run_sync) and full prompt (this script's threshold,
        # since cache_check replays the same row).
        try:
            import tiktoken  # noqa: F401
        except ImportError:
            pytest.skip("tiktoken not installed")
        client = FakeOpenAIClient()
        _drain_responses_for_calls(client, 2)
        with _patch_client(client):
            cache_check.main([
                "--input", example_input,
                "--calls", "2", "--gap-seconds", "0",
            ])
        out = capsys.readouterr().out
        assert "system message tokens:" in out
        assert "full prompt tokens:" in out
        # System tokens count is shown.
        assert "885" in out

    def test_header_does_not_warn_0pct_when_full_prompt_above_threshold(
            self, example_input, capsys, monkeypatch,
    ):
        # The whole point of the bug fix: cache_check replays the SAME
        # row, so when the FULL prompt crosses 1024 tokens the script
        # CAN get cache hits — even if the system message alone is below
        # the cross-row threshold. The header must not say "expect 0% hit
        # rate" in that case.
        try:
            import tiktoken  # noqa: F401
        except ImportError:
            pytest.skip("tiktoken not installed")
        # System 1000 + user 75 (tiny fixture) = 1075 > 1024.
        monkeypatch.setattr(cache_check, "system_message_token_count",
                            lambda model: 1000)
        client = FakeOpenAIClient()
        _drain_responses_for_calls(client, 2)
        with _patch_client(client):
            cache_check.main([
                "--input", example_input,
                "--calls", "2", "--gap-seconds", "0",
            ])
        out = capsys.readouterr().out
        # Full-prompt threshold gate is OK; the buggy "expect 0% hit rate"
        # text must NOT appear in this case.
        assert "this script can trigger cache hits" in out
        assert "expect 0% hit rate" not in out
        assert "even this script will get 0% hit rate" not in out
        # System-only is below cross-row threshold (1000 < 1024) — this
        # NOTE should still appear (informational; matters for run_sync).
        assert "cross-row threshold" in out

    def test_header_warns_when_full_prompt_below_threshold(
            self, example_input, capsys, monkeypatch,
    ):
        # If the FULL prompt (system + user) is below 1024, even
        # identical-row caching can't fire. That's the script-level
        # threshold gate.
        try:
            import tiktoken  # noqa: F401
        except ImportError:
            pytest.skip("tiktoken not installed")
        monkeypatch.setattr(cache_check, "system_message_token_count",
                            lambda model: 600)  # also below cross-row
        # User message via build_user_message of an example row is small
        # (~150 tokens), so 600 + 150 = 750 < 1024. Full prompt below
        # threshold → script warns.
        client = FakeOpenAIClient()
        _drain_responses_for_calls(client, 2)
        with _patch_client(client):
            cache_check.main([
                "--input", example_input,
                "--calls", "2", "--gap-seconds", "0",
            ])
        out = capsys.readouterr().out
        assert "full prompt tokens:" in out
        assert "BELOW the 1024-token caching threshold" in out
        assert "even this script will get 0% hit rate" in out

    def test_header_falls_back_when_tiktoken_unavailable(
            self, example_input, capsys, monkeypatch,
    ):
        # If tiktoken can't be imported, the header should print a clear
        # "char-estimate" fallback rather than crashing. Crucially, the
        # full-prompt fallback estimate must include user_chars (not just
        # sys_chars) — earlier the helper used sys_chars for both lines,
        # so the system and full-prompt fallback values were identical
        # even though the actual full prompt is bigger.
        monkeypatch.setattr(cache_check, "system_message_token_count",
                            lambda model: None)
        client = FakeOpenAIClient()
        _drain_responses_for_calls(client, 2)
        with _patch_client(client):
            cache_check.main([
                "--input", example_input,
                "--calls", "2", "--gap-seconds", "0",
            ])
        out = capsys.readouterr().out
        assert "system message tokens:" in out
        assert "full prompt tokens:" in out
        assert "char-estimate" in out
        # Extract the two estimate numbers and assert full > system.
        # Lines look like: "  system message tokens:  ~951 (..."
        #                  "  full prompt tokens:     ~975 (..."
        import re
        sys_match = re.search(r"system message tokens:\s+~(\d+)", out)
        full_match = re.search(r"full prompt tokens:\s+~(\d+)", out)
        assert sys_match and full_match, \
            "header missing one of the token-estimate lines"
        sys_est = int(sys_match.group(1))
        full_est = int(full_match.group(1))
        assert full_est > sys_est, (
            f"full-prompt fallback ({full_est}) should be greater than "
            f"system-only fallback ({sys_est}) — the user message has "
            "non-zero size"
        )

    def test_fail_verdict_structural_uses_full_prompt_not_system_only(
            self, example_input, capsys, monkeypatch,
    ):
        # FAIL's STRUCTURAL cause must be scoped to the FULL prompt, not
        # just the system message. cache_check replays the same row, so
        # the shared prefix is the full prompt; if THAT is below 1024,
        # the structural cause applies.
        try:
            import tiktoken  # noqa: F401
        except ImportError:
            pytest.skip("tiktoken not installed")
        # System 600 + user ~150 = ~750 full prompt tokens < 1024.
        monkeypatch.setattr(cache_check, "system_message_token_count",
                            lambda model: 600)
        client = FakeOpenAIClient()
        # All calls report 0 cached tokens → triggers FAIL verdict.
        for _ in range(3):
            client.chat.completions.responses.append(
                FakeResponse('{"k":1}', 750, 50, cached_tokens=0)
            )
        with _patch_client(client):
            rc = cache_check.main([
                "--input", example_input,
                "--calls", "3", "--gap-seconds", "0",
            ])
        assert rc == 1
        out = capsys.readouterr().out
        assert "FAIL" in out
        assert "STRUCTURAL" in out
        # The STRUCTURAL clause must point at the FULL prompt, not the
        # system message. With system=600 + user~150 = ~750 full tokens,
        # the message should mention the full-prompt count and be clear
        # that even identical-row replays can't trigger caching.
        assert "full prompt is" in out
        assert "BELOW" in out
        assert "even this script" in out

    def test_fail_verdict_no_structural_when_full_above_but_system_below(
            self, example_input, capsys, monkeypatch,
    ):
        # When the FULL prompt is above 1024 but system-only is below,
        # cache_check's identical-row test SHOULD be able to hit. If it
        # still gets FAIL, the cause is account/model — NOT the rubric
        # size. STRUCTURAL must NOT fire in this branch (would mislead
        # the operator). The cross-row note can still mention the system
        # threshold for run_sync's benefit.
        try:
            import tiktoken  # noqa: F401
        except ImportError:
            pytest.skip("tiktoken not installed")
        # System 1000 + tiny-fixture user 75 = 1075 > 1024.
        monkeypatch.setattr(cache_check, "system_message_token_count",
                            lambda model: 1000)
        client = FakeOpenAIClient()
        for _ in range(3):
            client.chat.completions.responses.append(
                FakeResponse('{"k":1}', 1075, 50, cached_tokens=0)
            )
        with _patch_client(client):
            rc = cache_check.main([
                "--input", example_input,
                "--calls", "3", "--gap-seconds", "0",
            ])
        assert rc == 1
        out = capsys.readouterr().out
        assert "FAIL" in out
        # STRUCTURAL applies only to full-prompt-below-threshold; not here.
        assert "STRUCTURAL" not in out
        # Cross-row note SHOULD appear since system alone (1000) is below.
        assert "CROSS-ROW NOTE" in out
        assert "run_sync" in out


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
