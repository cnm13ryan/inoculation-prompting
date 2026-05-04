"""Batch input/output helpers for the OpenAI Batch API.

The Batch API is async: submit-time and collect-time are different processes,
so we need a persistent state file (the "manifest") that links them. The
manifest also stores per-row trusted evidence (true_gcd, user_claimed_gcd,
provenance) needed at collect time for the verifier and the final record.

Kept in this module:
  - build_batch_request   : one /v1/chat/completions JSONL line for batch input
  - Manifest              : the state file linking submit -> collect
  - write_manifest / read_manifest
  - parse_batch_output_line : one row of OpenAI's batch output JSONL
  - file_sha256, now_utc  : small reproducibility helpers
"""
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from inputs import JudgeInput
from judge_prompt import JUDGE_RUBRIC, JUDGE_SYSTEM, build_user_message, prompt_hash


# --- Prompt caching ---------------------------------------------------------

# Allowed values for OpenAI's prompt_cache_retention parameter.
PROMPT_CACHE_RETENTION_ALLOWED = frozenset({"in_memory", "24h"})


def default_prompt_cache_key() -> str:
    """Default cache key derived from the prompt hash.

    All requests in a batch sharing this rubric will route to the same cache
    shard (good for hits). Any rubric change produces a new hash and therefore
    a new cache key, which invalidates stale cached prefixes automatically —
    consistent with the §13 reproducibility discipline in the notes.
    """
    return f"judge:{prompt_hash()[:16]}"


# --- Build batch request lines ----------------------------------------------

def build_batch_request(
    judge_input: JudgeInput,
    custom_id: str,
    model: str,
    max_completion_tokens: int = 1500,
    prompt_cache_key: Optional[str] = None,
    prompt_cache_retention: Optional[str] = "24h",
) -> dict:
    """One JSONL line for the Batch API: chat completion with JSON-mode.

    Prompt caching:
      The system message (instructions + rubric) is static across all rows in
      a batch and goes first in `messages`, so OpenAI will cache the prefix
      automatically once it crosses the 1024-token threshold (it does, by a
      lot). Setting `prompt_cache_key` improves cache routing across requests
      with the same prefix, and `prompt_cache_retention="24h"` extends the
      cache lifetime on supporting models (gpt-4.1, gpt-5*).

      Pass `prompt_cache_retention=None` to omit the field for models that do
      not support it.
    """
    if (prompt_cache_retention is not None
            and prompt_cache_retention not in PROMPT_CACHE_RETENTION_ALLOWED):
        raise ValueError(
            f"prompt_cache_retention must be one of "
            f"{sorted(PROMPT_CACHE_RETENTION_ALLOWED)} or None; "
            f"got {prompt_cache_retention!r}"
        )

    body: dict = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": JUDGE_SYSTEM + "\n\n" + JUDGE_RUBRIC,
            },
            {
                "role": "user",
                "content": build_user_message(judge_input),
            },
        ],
        "max_completion_tokens": max_completion_tokens,
        "temperature": 0.0,
        # JSON mode: the rubric already mentions "JSON" explicitly.
        "response_format": {"type": "json_object"},
    }
    if prompt_cache_key:
        body["prompt_cache_key"] = prompt_cache_key
    if prompt_cache_retention is not None:
        body["prompt_cache_retention"] = prompt_cache_retention

    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }


# --- Manifest ----------------------------------------------------------------

@dataclass
class Manifest:
    """State file linking submit -> collect.

    `rows` is keyed by custom_id, and each value is a dict of provenance +
    trusted evidence the verifier needs at collect time. The manifest is
    written at submit time, optionally amended after wait (with the output
    and error file IDs), and read at collect time.
    """
    batch_id: str
    input_file_id: str
    endpoint: str
    completion_window: str
    judge_model: str
    judge_version: str
    rubric_version: str
    prompt_sha256: str
    input_file_sha256: str
    code_commit: str
    submitted_at_utc: str
    rows: dict
    output_file_id: Optional[str] = None
    error_file_id: Optional[str] = None
    prompt_cache_key: Optional[str] = None
    prompt_cache_retention: Optional[str] = None
    user_metadata: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


def write_manifest(manifest: Manifest, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(manifest.to_json(), encoding="utf-8")


def read_manifest(path: str) -> Manifest:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return Manifest(**data)


def deterministic_custom_id(judge_input: JudgeInput) -> str:
    """Return a content-derived ``custom_id`` for a row.

    Used at submit time when the input row does not supply ``row_id``.
    Hashing the JudgeInput's trusted-evidence + untrusted-artifact fields
    means re-submitting the same input file produces byte-identical
    custom_ids — which is what makes ``--retry-failures-from`` work for
    rows the user did not give explicit ``row_id``s.

    Format mirrors the ``f"row_{uuid.uuid4().hex[:12]}"`` shape that
    earlier versions of ``run.py`` produced, so manifests and logs that
    treated the id as opaque still work.
    """
    h = hashlib.sha256()
    for part in (
        str(judge_input.a),
        str(judge_input.b),
        str(judge_input.user_claimed_gcd),
        str(judge_input.true_gcd),
        judge_input.expected_verdict,
        judge_input.user_prompt,
        judge_input.raw_response,
    ):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return f"row_{h.hexdigest()[:12]}"


def resolve_custom_id(row: dict, judge_input: JudgeInput) -> str:
    """Return the ``custom_id`` to use for ``row``.

    User-supplied ``row_id`` wins; otherwise we derive a deterministic id
    from the JudgeInput. The same row-content always maps to the same id,
    so retries (``--retry-failures-from``) can match by id without the
    user having had to supply explicit ``row_id``s.
    """
    explicit = row.get("row_id")
    if explicit:
        return str(explicit)
    return deterministic_custom_id(judge_input)


def make_row_meta(
    row: dict, judge_input: JudgeInput, *, custom_id: str | None = None
) -> dict:
    """Per-row payload stored in the manifest.

    Includes trusted evidence the verifier needs (true_gcd, user_claimed_gcd)
    plus provenance the judge never saw (arm, seed, model_id, ...).

    ``custom_id`` is the resolved id used for the OpenAI batch request. We
    store it as the manifest's ``row_id`` so that records emitted at
    collect time always have a stable, non-null per-row identifier even
    when the user did not provide an explicit ``row_id``. If
    ``custom_id`` is omitted (legacy callers), we fall back to the user's
    ``row_id`` (which may be None).
    """
    resolved_row_id = row.get("row_id") or custom_id
    return {
        "row_id": resolved_row_id,
        "prompt_candidate": row.get("prompt_candidate"),
        "arm": row.get("arm"),
        "seed": row.get("seed"),
        "model_id": row.get("model_id"),
        "a": judge_input.a,
        "b": judge_input.b,
        "user_claimed_gcd": judge_input.user_claimed_gcd,
        "true_gcd": judge_input.true_gcd,
    }


# --- Parse batch output lines ------------------------------------------------

@dataclass
class BatchResultLine:
    custom_id: str
    raw_text: Optional[str]
    usage: dict
    api_error: Optional[dict]   # OpenAI-level error: api status, expired, etc.
    finish_reason: Optional[str]


def parse_batch_output_line(line: dict) -> BatchResultLine:
    """Parse one line from the batch output JSONL into a BatchResultLine.

    Handles the three observed shapes:
      - success:           {"custom_id":..., "response": {...}, "error": null}
      - api error:         {"custom_id":..., "response": null,  "error": {...}}
      - expired:           {"custom_id":..., "response": null,  "error": {"code": "batch_expired", ...}}
    """
    custom_id = line.get("custom_id", "")
    api_error = line.get("error")
    response = line.get("response")

    if response is None:
        return BatchResultLine(
            custom_id=custom_id,
            raw_text=None,
            usage={},
            api_error=api_error or {"code": "no_response", "message": "no response in line"},
            finish_reason=None,
        )

    body = response.get("body") or {}
    choices = body.get("choices") or []
    usage_dict = body.get("usage") or {}
    details = usage_dict.get("prompt_tokens_details") or {}
    usage = {
        "input": usage_dict.get("prompt_tokens"),
        "output": usage_dict.get("completion_tokens"),
        "cached": details.get("cached_tokens"),
    }

    if not choices:
        return BatchResultLine(
            custom_id=custom_id,
            raw_text=None,
            usage=usage,
            api_error={"code": "no_choices", "message": "no choices in response body"},
            finish_reason=None,
        )

    first = choices[0]
    raw_text = (first.get("message") or {}).get("content")
    finish_reason = first.get("finish_reason")
    return BatchResultLine(
        custom_id=custom_id,
        raw_text=raw_text,
        usage=usage,
        api_error=None,
        finish_reason=finish_reason,
    )


# --- Reproducibility helpers -------------------------------------------------

def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
