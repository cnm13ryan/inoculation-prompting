"""Synchronous Chat Completions wrapper for the sync runner and cache_check.

Sibling of ``batch_client.py``: same prompt-cache settings, same JSON-mode
default, same temperature 0 — so the request shape matches the batch path
byte-for-byte at the cacheable-prefix level. That symmetry is what makes
sync runs a faithful preview of what the batch path will see.

Returned ``CallResult`` carries ``cached_tokens`` so the runner can compute
hit rates without re-walking the SDK response shape.
"""
from dataclasses import dataclass
from typing import Optional

from batch_io import PROMPT_CACHE_RETENTION_ALLOWED


@dataclass
class CallResult:
    raw_text: str
    # Token counts are ``Optional[int]`` for parity with ``parse_batch_output_line``:
    # OpenAI omits ``prompt_tokens_details`` for prompts under the caching
    # threshold, and downstream code needs to distinguish "no signal" (None)
    # from "0 cached tokens" (a real measurement). Collapsing to 0 would
    # silently corrupt any aggregation that joins sync + batch records.
    prompt_tokens: Optional[int]
    cached_tokens: Optional[int]
    completion_tokens: Optional[int]
    finish_reason: Optional[str]
    model: Optional[str]


def make_openai_client():
    """Return a configured OpenAI client. Reads ``OPENAI_API_KEY`` from env."""
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError(
            "openai SDK not installed. `pip install openai`."
        ) from e
    return OpenAI()


def chat_completion(
    client,
    system: str,
    user: str,
    *,
    model: str = "gpt-4.1",
    max_completion_tokens: int = 1500,
    temperature: float = 0.0,
    json_mode: bool = True,
    prompt_cache_key: Optional[str] = None,
    prompt_cache_retention: Optional[str] = "24h",
) -> CallResult:
    """One synchronous chat completion. Returns ``CallResult`` with cache stats.

    Defaults match ``build_batch_request`` so cache hit behaviour is
    comparable across the two paths: temperature 0, JSON mode, 24h retention.
    """
    # Validate retention up-front for parity with build_batch_request — a
    # typo here would otherwise reach OpenAI and 400 only at request time.
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
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_completion_tokens": max_completion_tokens,
        "temperature": temperature,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    if prompt_cache_key:
        body["prompt_cache_key"] = prompt_cache_key
    if prompt_cache_retention is not None:
        body["prompt_cache_retention"] = prompt_cache_retention

    resp = client.chat.completions.create(**body)
    return _to_call_result(resp)


def _to_call_result(resp) -> CallResult:
    """Adapt both pydantic-model and dict-shaped SDK responses to ``CallResult``."""
    data = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)

    choices = data.get("choices") or []
    first = choices[0] if choices else {}
    raw_text = (first.get("message") or {}).get("content") or ""
    finish_reason = first.get("finish_reason")

    usage = data.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}

    return CallResult(
        raw_text=raw_text,
        # Pass None through for missing values so cross-mode aggregations
        # can distinguish "no signal" from "0". Mirrors batch_io.parse_batch_output_line.
        prompt_tokens=usage.get("prompt_tokens"),
        cached_tokens=details.get("cached_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        finish_reason=finish_reason,
        model=data.get("model"),
    )
