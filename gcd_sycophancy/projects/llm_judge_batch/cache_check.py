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

    sys_chars = len(system_message)
    sys_tokens = system_message_token_count(args.model)
    if sys_tokens is None:
        # tiktoken not installed; fall back to a character estimate, but be
        # explicit that it's an estimate. ~4.3 chars/token holds for
        # GCD-domain content (English mixed with arithmetic).
        token_str = (
            f"~{sys_chars // 4} (char-estimate; tiktoken not installed — "
            f"`pip install tiktoken` for the real count)"
        )
        threshold_ok = (sys_chars // 4) >= PROMPT_CACHE_MIN_PREFIX_TOKENS
    else:
        token_str = str(sys_tokens)
        threshold_ok = sys_tokens >= PROMPT_CACHE_MIN_PREFIX_TOKENS
    threshold_note = (
        f"OK >= {PROMPT_CACHE_MIN_PREFIX_TOKENS}-token threshold"
        if threshold_ok else
        f"WARN: BELOW the {PROMPT_CACHE_MIN_PREFIX_TOKENS}-token "
        f"caching threshold; expect 0% hit rate"
    )

    print("Prompt cache check")
    print("=" * 60)
    print(f"  model:                  {args.model}")
    print(f"  prompt_cache_key:       {cache_key}")
    print(f"  prompt_cache_retention: {retention}")
    print(f"  prompt_sha256[:16]:     {prompt_hash()[:16]}")
    print(f"  system message chars:   {sys_chars}")
    print(f"  system message tokens:  {token_str}  ({threshold_note})")
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
        # Surface the rubric-size constraint first when applicable — it's a
        # hard structural cause, not a transient one.
        if not threshold_ok:
            print(
                f"    1. STRUCTURAL: the system message is "
                f"{token_str} tokens, BELOW the "
                f"{PROMPT_CACHE_MIN_PREFIX_TOKENS}-token caching minimum. "
                "OpenAI caches in 128-token increments starting at 1024 "
                "tokens of shared prefix; below that, NO portion of the "
                "prompt is cached. Caching cannot fire at the current "
                "rubric size — extend the rubric (rebakes the prompt hash "
                "and invalidates calibration) or accept 0% hit rate.")
            print("    2. The model does not support prompt caching. Caching is "
                  "enabled for gpt-4o and newer; older models won't show hits.")
        else:
            print("    1. The model does not support prompt caching. Caching is "
                  "enabled for gpt-4o and newer; older models won't show hits.")
            print(
                f"    2. Prompt below threshold. (System message here is "
                f"{sys_chars} chars / {token_str} tokens.)")
        print("    3. The cache was evicted between calls. In-memory retention is "
              "5–10 minutes of inactivity; if --gap-seconds was very large or the "
              "cluster was under load, the entry may have been dropped.")
        print("    4. ZDR project with extended retention requested. If 'extended' "
              "caching is restricted on your project, retry with --retention in_memory.")
        print("    5. Account / org has caching disabled. Rare, but check the org "
              "settings page.")
        print()
        print(f"  prompt_sha256 used: {prompt_hash()}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
