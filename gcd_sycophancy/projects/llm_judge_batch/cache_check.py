"""Quick synchronous cache check.

Sends the same judge prompt twice (or N times) against the regular Chat
Completions API and reports whether prompt caching is firing. Diagnostic
tool — production runs use ``run.py`` (batch) or ``run_sync.py`` (sync).

Usage:
    python cache_check.py                              # uses example_input.jsonl row 1
    python cache_check.py --input rows.jsonl --row-id r001
    python cache_check.py --model gpt-4.1 --calls 4    # 4 calls instead of 2
    python cache_check.py --retention in_memory        # short-lived cache

Exit codes:
    0  caching verified (cached_tokens > 0 on a follow-up call)
    1  no cache hits observed (something is wrong)
    2  user / config error (missing input, bad row_id, etc.)
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

from inputs import build_judge_input
from judge_prompt import (
    JUDGE_RUBRIC,
    JUDGE_SYSTEM,
    PROMPT_CACHE_MIN_PREFIX_TOKENS,
    build_user_message,
    prompt_hash,
    system_message_token_count,
)
from batch_io import default_prompt_cache_key, PROMPT_CACHE_RETENTION_ALLOWED
from sync_client import chat_completion, make_openai_client


def _load_row(path: str, row_id: Optional[str] = None) -> dict:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"no rows in {path}")
    if row_id is None:
        return rows[0]
    for r in rows:
        if r.get("row_id") == row_id:
            return r
    raise SystemExit(f"row_id {row_id!r} not found in {path}")


def _print_call(i: int, n: int, result, elapsed_s: float) -> None:
    # Token counts may be None when OpenAI omits prompt_tokens_details (e.g.
    # under-threshold prompts). Treat None as 0 for display so the columns
    # still line up; the verdict logic below also tolerates None.
    prompt = result.prompt_tokens or 0
    cached = result.cached_tokens or 0
    completion = result.completion_tokens or 0
    pct = 100.0 * cached / prompt if prompt else 0.0
    label = f"call {i}/{n}"
    print(
        f"  {label:8s}  prompt={prompt:5d}  "
        f"cached={cached:5d}  ({pct:5.1f}%)  "
        f"out={completion:4d}  "
        f"finish={result.finish_reason}  "
        f"elapsed={elapsed_s:.2f}s"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="example_input.jsonl",
                        help="JSONL of rows in the standard format")
    parser.add_argument("--row-id", default=None,
                        help="Which row to use (default: first row)")
    parser.add_argument("--model", default="gpt-4.1")
    parser.add_argument("--max-completion-tokens", type=int, default=1500)
    parser.add_argument("--calls", type=int, default=2,
                        help="How many times to send the prompt (>=2)")
    parser.add_argument("--gap-seconds", type=float, default=2.0,
                        help="Sleep between calls; cache needs a moment to populate")
    parser.add_argument("--prompt-cache-key", default=None,
                        help="Override cache key. Default: derived from rubric hash")
    parser.add_argument("--retention", default="24h",
                        help="prompt_cache_retention: 'in_memory', '24h', or 'none'")
    parser.add_argument("--show-response", action="store_true",
                        help="Print the first 400 chars of each response")
    args = parser.parse_args(argv)

    if args.calls < 2:
        print("error: --calls must be >= 2 to compare a miss to a hit", file=sys.stderr)
        return 2

    if args.retention.lower() in {"none", "off", ""}:
        retention = None
    elif args.retention in PROMPT_CACHE_RETENTION_ALLOWED:
        retention = args.retention
    else:
        print(f"error: --retention must be in {sorted(PROMPT_CACHE_RETENTION_ALLOWED)} or 'none'",
              file=sys.stderr)
        return 2

    if not Path(args.input).exists():
        print(f"error: input file not found: {args.input}", file=sys.stderr)
        return 2

    row = _load_row(args.input, args.row_id)
    judge_input = build_judge_input(row)
    system_message = JUDGE_SYSTEM + "\n\n" + JUDGE_RUBRIC
    user_message = build_user_message(judge_input)
    cache_key = args.prompt_cache_key or default_prompt_cache_key()

    # ---- threshold accounting ------------------------------------------
    # Two distinct thresholds matter for caching, with different
    # implications depending on workload:
    #   * full-prompt size  : eligibility for caching at all (system+user
    #     must be >= 1024 tokens). This is what gates THIS script's hit
    #     rate, since cache_check replays the same row → shared prefix is
    #     the full prompt.
    #   * system-only size  : the cross-row shared-prefix size. Below
    #     1024, cross-row workloads (run_sync against distinct rows) get
    #     0% even though identical-row scripts like this one can still
    #     hit. This is informational here, not a script-level gate.
    sys_chars = len(system_message)
    sys_tokens = system_message_token_count(args.model)
    user_tokens: int | None = None
    full_tokens: int | None = None
    if sys_tokens is not None:
        try:
            import tiktoken
            try:
                _enc = tiktoken.encoding_for_model(args.model)
            except KeyError:
                _enc = tiktoken.get_encoding("o200k_base")
            user_tokens = len(_enc.encode(user_message))
            # Approximate; OpenAI adds ~10 tokens of chat-completion
            # overhead per request that we don't model here. The actual
            # ``usage.prompt_tokens`` returned per call is the source of
            # truth — see the per-call lines below.
            full_tokens = sys_tokens + user_tokens
        except ImportError:
            pass

    def _fmt_tokens(n):
        return str(n) if n is not None else (
            f"~{sys_chars // 4} (char-estimate; "
            "tiktoken not installed — `pip install tiktoken` for the real count)"
        )

    # The threshold gate for THIS script is the full prompt: cache_check
    # sends identical prompts, so the shared prefix between calls is the
    # whole prompt. If that exceeds 1024, caching can fire.
    if full_tokens is None:
        # Fall back to char-estimate; rare in practice.
        approx_full = (sys_chars + len(user_message)) // 4
        threshold_ok = approx_full >= PROMPT_CACHE_MIN_PREFIX_TOKENS
    else:
        threshold_ok = full_tokens >= PROMPT_CACHE_MIN_PREFIX_TOKENS

    if threshold_ok:
        full_threshold_note = (
            f"OK >= {PROMPT_CACHE_MIN_PREFIX_TOKENS}-token threshold "
            "(this script can trigger cache hits)"
        )
    else:
        full_threshold_note = (
            f"WARN: BELOW the {PROMPT_CACHE_MIN_PREFIX_TOKENS}-token "
            "caching threshold; even this script will get 0% hit rate"
        )

    # System-only is informational here. Below threshold means cross-row
    # production workloads (run_sync) will get 0%, but doesn't affect
    # cache_check's identical-row test.
    if sys_tokens is not None and sys_tokens < PROMPT_CACHE_MIN_PREFIX_TOKENS:
        sys_note = (
            f"NOTE: below {PROMPT_CACHE_MIN_PREFIX_TOKENS}-token "
            "cross-row threshold (run_sync against distinct rows would get "
            "0%; this script replays the same row, so it can still hit)"
        )
    else:
        sys_note = ""

    print("Prompt cache check")
    print("=" * 60)
    print(f"  model:                  {args.model}")
    print(f"  prompt_cache_key:       {cache_key}")
    print(f"  prompt_cache_retention: {retention}")
    print(f"  prompt_sha256[:16]:     {prompt_hash()[:16]}")
    print(f"  system message chars:   {sys_chars}")
    print(f"  system message tokens:  {_fmt_tokens(sys_tokens)}"
          + (f"  ({sys_note})" if sys_note else ""))
    if user_tokens is not None:
        print(f"  user message tokens:    {user_tokens}")
        print(f"  full prompt tokens:     {_fmt_tokens(full_tokens)}  "
              f"({full_threshold_note})")
    else:
        print(f"  full prompt tokens:     {_fmt_tokens(full_tokens)}  "
              f"({full_threshold_note})")
    print(f"  row_id used:            {row.get('row_id', '<first row>')}")
    print(f"  number of calls:        {args.calls}")
    print(f"  gap between calls:      {args.gap_seconds:.2f}s")
    print()
    print("Calls:")

    client = make_openai_client()
    results = []
    for i in range(1, args.calls + 1):
        t0 = time.perf_counter()
        result = chat_completion(
            client,
            system=system_message,
            user=user_message,
            model=args.model,
            max_completion_tokens=args.max_completion_tokens,
            prompt_cache_key=cache_key,
            prompt_cache_retention=retention,
        )
        elapsed = time.perf_counter() - t0
        _print_call(i, args.calls, result, elapsed)
        results.append((result, elapsed))
        if args.show_response:
            preview = (result.raw_text or "")[:400].replace("\n", " ")
            print(f"            response[:400]: {preview}")
        if i < args.calls:
            time.sleep(args.gap_seconds)

    print()
    print("Verdict:")
    print("=" * 60)

    first_result, first_elapsed = results[0]
    follow_ups = results[1:]
    # ``cached_tokens`` may be None when prompt_tokens_details is absent;
    # treat None as "no hit signal" (same as 0) for the verdict.
    follow_up_hits = [(r.cached_tokens or 0) > 0 for r, _ in follow_ups]
    any_hit = any(follow_up_hits)
    all_hit = all(follow_up_hits) and bool(follow_ups)

    if (first_result.cached_tokens or 0) > 0:
        print(f"  NOTE: first call already had {first_result.cached_tokens} cached tokens — "
              f"the cache was warm before this script ran (a previous call with the same "
              f"prompt_cache_key recently fired). That's fine; caching is clearly working.")

    if all_hit:
        total_cached = sum((r.cached_tokens or 0) for r, _ in follow_ups)
        total_prompt = sum((r.prompt_tokens or 0) for r, _ in follow_ups)
        avg_hit = total_cached / total_prompt if total_prompt else 0.0
        print(f"  PASS: every follow-up call had cached_tokens > 0 "
              f"(avg hit rate {avg_hit:.1%}).")
        if follow_ups:
            first_t = first_elapsed
            avg_followup_t = sum(t for _, t in follow_ups) / len(follow_ups)
            speedup = (first_t - avg_followup_t) / first_t if first_t > 0 else 0.0
            print(f"        latency: first={first_t:.2f}s  "
                  f"avg follow-up={avg_followup_t:.2f}s  "
                  f"speedup={speedup:+.1%}")
        return 0
    elif any_hit:
        print("  PARTIAL: some follow-up calls had cache hits, others did not. "
              "This can happen under cache-shard overflow (>15 req/min on the same key) "
              "or if calls were routed to different machines.")
        return 0
    else:
        print("  FAIL: no cache hits observed on any follow-up call.")
        print()
        print("  Likely causes (in rough order of probability):")
        # Structural prompt-size cause: only applies when the FULL prompt
        # is below 1024. cache_check sends identical prompts, so the shared
        # prefix is the full prompt; if that's below threshold, no caching
        # is possible regardless of account/model.
        if not threshold_ok:
            print(
                f"    1. STRUCTURAL: the full prompt is "
                f"{_fmt_tokens(full_tokens)} tokens, BELOW the "
                f"{PROMPT_CACHE_MIN_PREFIX_TOKENS}-token caching minimum. "
                "OpenAI caches in 128-token increments starting at 1024 "
                "tokens of shared prefix; below that, NO portion of the "
                "prompt is cached. Even this script (which replays the "
                "same row) can't trigger a cache hit — try a larger row "
                "or extend the rubric.")
            print("    2. The model does not support prompt caching. Caching is "
                  "enabled for gpt-4o and newer; older models won't show hits.")
        else:
            print("    1. The model does not support prompt caching. Caching is "
                  "enabled for gpt-4o and newer; older models won't show hits.")
            # Defensive: in this branch the full prompt crosses 1024, so
            # prompt-size is NOT the structural cause for cache_check's
            # identical-row test. Mention it only as additional context.
            extra = (
                f" (full prompt is {_fmt_tokens(full_tokens)} tokens, "
                "above threshold; prompt size shouldn't be the issue here.)"
            )
            print(f"    2. Account / project caching disabled.{extra}")
        print("    3. The cache was evicted between calls. In-memory retention is "
              "5–10 minutes of inactivity; if --gap-seconds was very large or the "
              "cluster was under load, the entry may have been dropped.")
        print("    4. ZDR project with extended retention requested. If 'extended' "
              "caching is restricted on your project, retry with --retention in_memory.")
        # Cross-row implication: even if cache_check passes, run_sync
        # against distinct rows depends on the system-only count.
        if sys_tokens is not None and sys_tokens < PROMPT_CACHE_MIN_PREFIX_TOKENS:
            print(
                f"    5. CROSS-ROW NOTE: even if this script passes, "
                f"run_sync against distinct rows will see ~0% (system "
                f"message alone is {sys_tokens} tokens, below "
                f"{PROMPT_CACHE_MIN_PREFIX_TOKENS}). Different issue, "
                "same fix path: extend the rubric.")
        print()
        print(f"  prompt_sha256 used: {prompt_hash()}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
