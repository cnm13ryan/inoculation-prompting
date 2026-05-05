"""Synchronous runner — for fast prototyping and calibration.

Trade-offs vs the Batch API path:

* Sync: ~1-2 s per row, full price, results in minutes.
* Batch: 24 h window, 50% cheaper, async submit/wait/collect.

Use this for calibration runs and quick iteration. Switch to ``run.py`` for
the final cost-optimized eval. Both paths produce records with the same
schema, distinguished by a ``"mode": "sync"`` or ``"mode": "batch"`` field.

Resumability:
  Records are flushed per-row via ``RecordLogger`` opened in append mode.
  On startup the runner reads existing rows in the output file and skips
  any ``row_id`` already present — Ctrl-C and rerun the same command,
  losing at most the in-flight row. ``--no-resume`` disables this and
  truncates the output file on open.

Concurrency:
  Default is sequential (``--concurrency 1``). For 500+ row runs,
  ``--concurrency 5`` to ``10`` gives a near-linear speedup; the OpenAI
  SDK is thread-safe. Records still flow through a single
  ``RecordLogger`` guarded by a lock, so the output JSONL stays
  consistent.

Failure policy (matches the batch path):
  Transient API errors (network, 429, 5xx) are retried up to
  ``--api-retries`` times with exponential backoff (``--backoff-seconds``
  doubles each attempt). Schema failures are recorded as
  ``judge_parse_failed`` and not retried — the LLM produced bad output,
  retrying won't help.
"""
import argparse
import json
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from inputs import build_judge_input
from judge_prompt import JUDGE_RUBRIC, JUDGE_SYSTEM, build_user_message, prompt_hash
from judge_client import JudgeOutput, JudgeParseFailure, parse_judge_response
from verifier import verify_euclidean_chain, final_category
from logger import RecordLogger
from batch_io import (
    PROMPT_CACHE_RETENTION_ALLOWED,
    default_prompt_cache_key,
    file_sha256,
    now_utc,
    resolve_custom_id,
)
from sync_client import chat_completion, make_openai_client
# Reuse the batch path's input loader and retry filter so both modes share
# eager validation, JSONL/CSV handling, and retry semantics. ``run`` is
# imported by name (not as a star-import) to avoid pulling its CLI parser
# into this module's namespace.
from run import code_commit, filter_failed, load_rows


# --- resume support ---------------------------------------------------------

def already_done_ids(output_path: str) -> set[str]:
    """Read the existing output JSONL (if any) and collect row_ids of all
    records that completed (parsed OR a recorded failure — both count as
    "don't retry me unconditionally").

    Tolerates malformed lines (e.g. partial-write recovery from a kill mid-
    flush). They're skipped, not raised.
    """
    p = Path(output_path)
    if not p.exists():
        return set()
    done: set[str] = set()
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = rec.get("row_id")
            if rid:
                done.add(rid)
    return done


# --- per-row processing -----------------------------------------------------

def process_one_row(
    row: dict,
    client,
    *,
    system_message: str,
    judge_model: str,
    max_completion_tokens: int,
    cache_key: str,
    cache_retention: Optional[str],
    metadata: dict,
    api_retries: int = 2,
    backoff_seconds: float = 2.0,
) -> dict:
    """Run the full pipeline for one row and return one record dict.

    Schema mirrors ``run.build_record`` so downstream code can treat sync
    and batch records identically (modulo the ``mode`` tag).
    """
    judge_input = build_judge_input(row)
    # Deterministic id when ``row_id`` is missing — so a re-run produces the
    # same custom_id for the same row (resumability), and so sync records
    # are joinable with batch records over the same input file.
    custom_id = resolve_custom_id(row, judge_input)
    user_message = build_user_message(judge_input)

    # 1. API call with bounded retry on transient errors. Schema failures
    #    are detected after parsing and never retried.
    api_result = None
    api_error: Optional[dict] = None
    for attempt in range(api_retries + 1):
        try:
            api_result = chat_completion(
                client,
                system=system_message,
                user=user_message,
                model=judge_model,
                max_completion_tokens=max_completion_tokens,
                prompt_cache_key=cache_key,
                prompt_cache_retention=cache_retention,
            )
            api_error = None
            break
        except Exception as e:
            api_error = {
                "code": type(e).__name__,
                "message": str(e),
                "attempt": attempt + 1,
            }
            if attempt < api_retries:
                time.sleep(backoff_seconds * (2 ** attempt))

    # 2. Parse + verify (or record the API failure)
    judge_output: Optional[JudgeOutput] = None
    judge_parse_status = "parsed"
    judge_parse_error: Optional[str] = None
    judge_dict: dict = {}

    if api_result is None:
        judge_parse_status = "judge_api_failed"
        judge_parse_error = (api_error or {}).get("message")
        raw_text: Optional[str] = None
        finish_reason: Optional[str] = None
        tokens_in = tokens_out = tokens_cached = None
    else:
        raw_text = api_result.raw_text
        finish_reason = api_result.finish_reason
        tokens_in = api_result.prompt_tokens
        tokens_out = api_result.completion_tokens
        tokens_cached = api_result.cached_tokens
        try:
            judge_output = parse_judge_response(raw_text)
            judge_dict = judge_output.model_dump()
        except JudgeParseFailure as e:
            judge_parse_status = "judge_parse_failed"
            judge_parse_error = e.validation_error

    if judge_output is not None:
        chain_valid, chain_final = verify_euclidean_chain(judge_output.extracted_chain)
        verifier_result = final_category(
            judge_output, chain_valid, chain_final,
            judge_input.true_gcd, judge_input.user_claimed_gcd,
        )
    else:
        verifier_result = {
            "stance_direction": None,
            "answer_direction": None,
            "derivation_direction": None,
            "coherence_class": None,
            "python_chain_valid": None,
            "python_chain_final": None,
            "final_behavioural_category": judge_parse_status,
        }

    record = {
        # provenance — joined here, not seen by the judge
        "custom_id": custom_id,
        "row_id": row.get("row_id") or custom_id,
        "prompt_candidate": row.get("prompt_candidate"),
        "arm": row.get("arm"),
        "seed": row.get("seed"),
        "model_id": row.get("model_id"),
        # trusted evidence (echoed for joinability)
        "a": judge_input.a,
        "b": judge_input.b,
        "user_claimed_gcd": judge_input.user_claimed_gcd,
        "true_gcd": judge_input.true_gcd,
        # judge raw + parse status
        "judge_raw_response": raw_text,
        "judge_parse_status": judge_parse_status,
        "judge_parse_error": judge_parse_error,
        "api_error": api_error,
        "finish_reason": finish_reason,
        # parsed judge fields (flattened — empty dict on failure)
        **judge_dict,
        # python verification
        **verifier_result,
        # token usage
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "tokens_cached": tokens_cached,
        # cache settings
        "prompt_cache_key": cache_key,
        "prompt_cache_retention": cache_retention,
        # reproducibility metadata
        "judge_model": judge_model,
        "judge_model_version": metadata["judge_version"],
        "judge_prompt_sha256": metadata["prompt_sha256"],
        "rubric_version": metadata["rubric_version"],
        "input_file_sha256": metadata["input_sha256"],
        "code_commit": metadata["code_commit"],
        # batch_id is None for sync records (no Batch API call). Kept in
        # the schema so sync and batch records can be concatenated and
        # iterated without a conditional on ``mode``.
        "batch_id": None,
        "submitted_at_utc": metadata["submitted_at_utc"],
        "collected_at_utc": now_utc(),
        "mode": "sync",
    }
    return record


# --- main loop --------------------------------------------------------------

def run_sync(args) -> int:
    # Resolve cache settings
    cache_key = args.prompt_cache_key or default_prompt_cache_key()
    if args.prompt_cache_retention.lower() in {"none", "off", ""}:
        cache_retention: Optional[str] = None
    elif args.prompt_cache_retention not in PROMPT_CACHE_RETENTION_ALLOWED:
        print(
            f"error: --prompt-cache-retention must be in "
            f"{sorted(PROMPT_CACHE_RETENTION_ALLOWED)} or 'none'; "
            f"got {args.prompt_cache_retention!r}",
            file=sys.stderr,
        )
        return 2
    else:
        cache_retention = args.prompt_cache_retention

    # Load rows; apply --retry-failures-from filter; apply --limit. Validation
    # in load_rows raises before any side effects, so a missing field doesn't
    # leave a partially-written records file.
    all_rows = list(load_rows(args.input))
    if args.retry_failures_from:
        all_rows = filter_failed(all_rows, args.retry_failures_from)
        print(f"[sync] retrying {len(all_rows)} previously failed rows",
              file=sys.stderr)
    if args.limit:
        all_rows = all_rows[:args.limit]

    # Resume: skip row_ids already in the output file. We compare against the
    # row_id the runner WILL emit (explicit row_id, else deterministic id),
    # so resume is correct even when the user didn't supply explicit ids.
    done: set[str] = set()
    if args.resume:
        done = already_done_ids(args.output)
        if done:
            before = len(all_rows)
            kept = []
            for r in all_rows:
                rid = r.get("row_id")
                if not rid:
                    try:
                        rid = resolve_custom_id(r, build_judge_input(r))
                    except (KeyError, TypeError, ValueError):
                        kept.append(r)
                        continue
                if rid not in done:
                    kept.append(r)
            all_rows = kept
            print(
                f"[sync] resume: {len(done)} rows already in {args.output}; "
                f"{before - len(all_rows)} skipped, {len(all_rows)} remain",
                file=sys.stderr,
            )

    if not all_rows:
        print("[sync] nothing to do", file=sys.stderr)
        return 0

    # Frozen reproducibility metadata. ``submitted_at_utc`` mirrors the batch
    # path's manifest field — set once at run start; per-row ``collected_at_utc``
    # is filled in process_one_row.
    metadata = {
        "prompt_sha256": prompt_hash(),
        "rubric_version": args.rubric_version,
        "judge_version": args.judge_version,
        "input_sha256": file_sha256(Path(args.input)),
        "code_commit": code_commit(),
        "submitted_at_utc": now_utc(),
    }
    system_message = JUDGE_SYSTEM + "\n\n" + JUDGE_RUBRIC

    print(f"[sync] prompt_cache_key={cache_key} retention={cache_retention}",
          file=sys.stderr)
    print(f"[sync] processing {len(all_rows)} rows "
          f"(concurrency={args.concurrency})", file=sys.stderr)

    if args.dry_run:
        print("[sync] --dry-run: skipping API calls", file=sys.stderr)
        return 0

    client = make_openai_client()
    common = dict(
        system_message=system_message,
        judge_model=args.judge_model,
        max_completion_tokens=args.max_completion_tokens,
        cache_key=cache_key,
        cache_retention=cache_retention,
        metadata=metadata,
        api_retries=args.api_retries,
        backoff_seconds=args.backoff_seconds,
    )

    n_done = 0
    n_parse_fail = 0
    n_api_fail = 0
    total_in = total_cached = total_out = 0
    write_lock = threading.Lock()

    def _on_done(record: dict, logger: RecordLogger) -> None:
        nonlocal n_done, n_parse_fail, n_api_fail
        nonlocal total_in, total_cached, total_out
        with write_lock:
            logger.write(record)
            n_done += 1
            if record["judge_parse_status"] == "judge_parse_failed":
                n_parse_fail += 1
            elif record["judge_parse_status"] == "judge_api_failed":
                n_api_fail += 1
            total_in += record.get("tokens_in") or 0
            total_cached += record.get("tokens_cached") or 0
            total_out += record.get("tokens_out") or 0
            if n_done % max(1, args.progress_every) == 0:
                hit = (total_cached / total_in) if total_in else 0.0
                print(
                    f"  ...{n_done}/{len(all_rows)}  "
                    f"parse_fail={n_parse_fail} api_fail={n_api_fail}  "
                    f"cache_hit={hit:.1%}",
                    file=sys.stderr,
                )

    # Append when resuming (preserve already-collected rows); truncate
    # otherwise. ``--no-resume`` flips this to truncate so a re-run starts
    # with a fresh records file.
    log_mode = "a" if args.resume else "w"
    with RecordLogger(args.output, mode=log_mode) as logger:
        if args.concurrency <= 1:
            for row in all_rows:
                rec = process_one_row(row, client, **common)
                _on_done(rec, logger)
        else:
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                futures = [pool.submit(process_one_row, row, client, **common)
                           for row in all_rows]
                for fut in as_completed(futures):
                    _on_done(fut.result(), logger)

    print(
        f"[sync] done. wrote {n_done} records "
        f"(parse_fail={n_parse_fail}, api_fail={n_api_fail})",
        file=sys.stderr,
    )
    if total_in > 0:
        hit_rate = total_cached / total_in
        print(
            f"[sync] tokens: in={total_in} cached={total_cached} "
            f"({hit_rate:.1%} hit rate) out={total_out}",
            file=sys.stderr,
        )
        if hit_rate < 0.20 and n_done > 5:
            print(
                "[sync] WARNING: low cache hit rate. Try `python cache_check.py` "
                "to diagnose; common causes: model not supporting caching, prompt "
                "<1024 tokens, or eviction between calls.",
                file=sys.stderr,
            )
    # Match the batch path: failures are recorded in records, not raised. A
    # caller who wants to gate on them can grep the records file.
    return 0


# --- CLI --------------------------------------------------------------------

def add_sync_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--judge-model", default="gpt-4.1")
    parser.add_argument("--judge-version", default="unspecified")
    parser.add_argument("--rubric-version", default="v1.0")
    parser.add_argument("--max-completion-tokens", type=int, default=1500)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Threads for parallel calls. 1 = sequential.")
    parser.add_argument("--api-retries", type=int, default=2,
                        help="Retries on transient API errors per row")
    parser.add_argument("--backoff-seconds", type=float, default=2.0,
                        help="Initial backoff between retries (doubles each attempt)")
    parser.add_argument("--no-resume", action="store_false", dest="resume",
                        help="Disable skip-already-completed-rows behaviour and "
                             "truncate the output file on open")
    parser.add_argument("--retry-failures-from", default=None,
                        help="Records JSONL; submit only rows that previously failed")
    parser.add_argument("--dry-run", action="store_true",
                        help="Resolve config and skip rows, but make no API calls")
    parser.add_argument("--prompt-cache-key", default=None,
                        help="OpenAI prompt_cache_key. "
                             "Default: 'judge:<first 16 chars of prompt SHA-256>'")
    parser.add_argument("--prompt-cache-retention", default="24h",
                        help="OpenAI prompt_cache_retention: 'in_memory', '24h', "
                             "or 'none' to omit. Default: 24h.")
    parser.add_argument("--progress-every", type=int, default=10,
                        help="Print progress every N rows")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_sync_args(parser)
    args = parser.parse_args(argv)
    return run_sync(args) or 0


if __name__ == "__main__":
    sys.exit(main())
