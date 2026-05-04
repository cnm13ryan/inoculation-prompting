"""Batch-API orchestrator.

Subcommands:
  submit  : build batch JSONL, upload, create batch, write manifest
  status  : print current batch status (single shot)
  wait    : poll the batch until it reaches a terminal state
  collect : download output and error files, parse, verify, write records
  run     : submit + wait + collect (one-shot; blocks for up to 24h)

Smoke-testing without an API key:
  submit  --dry-run                 # builds batch JSONL, skips upload
  collect --from-local <out.jsonl>  # parses local output, skips download
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional

from inputs import build_judge_input
from judge_prompt import prompt_hash
from judge_client import JudgeOutput, JudgeParseFailure, parse_judge_response
from verifier import verify_euclidean_chain, final_category
from logger import RecordLogger
from batch_io import (
    Manifest,
    PROMPT_CACHE_RETENTION_ALLOWED,
    build_batch_request,
    default_prompt_cache_key,
    file_sha256,
    make_row_meta,
    now_utc,
    parse_batch_output_line,
    read_manifest,
    resolve_custom_id,
    write_manifest,
)


# --- helpers ----------------------------------------------------------------

def code_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def load_rows(path: str) -> Iterable[dict]:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".jsonl":
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    elif suffix == ".csv":
        with open(p, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                yield row
    else:
        raise ValueError(f"unsupported input format: {suffix}")


def filter_failed(rows: list, records_path: str) -> list:
    """Keep only rows whose previous attempt produced a parse / API failure.

    Matches on both the user's explicit ``row_id`` (if present in records)
    and the deterministic ``custom_id`` derived from the row content.
    Rows submitted without an explicit ``row_id`` get a content-derived id
    via :func:`batch_io.resolve_custom_id`, which is stable across re-runs;
    so a previously-failed row whose record carries a generated id still
    matches when the same input file is fed back in.
    """
    failed_ids: set = set()
    with open(records_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if (
                r.get("judge_parse_status") in {"judge_parse_failed", "judge_api_failed"}
                or r.get("final_behavioural_category")
                   in {"judge_parse_failed", "judge_api_failed"}
            ):
                # Records carry both ``custom_id`` (the OpenAI-side id used in
                # the batch) and ``row_id`` (which equals custom_id when the
                # user didn't provide one). Match either: a stale record from
                # before bug-fix may have ``row_id=null``, in which case
                # ``custom_id`` is the only viable handle.
                for key in ("custom_id", "row_id"):
                    val = r.get(key)
                    if val:
                        failed_ids.add(val)

    out = []
    for row in rows:
        explicit = row.get("row_id")
        if explicit and explicit in failed_ids:
            out.append(row)
            continue
        # Re-derive the deterministic id the same way ``cmd_submit`` does.
        try:
            ji = build_judge_input(row)
        except (KeyError, TypeError, ValueError):
            continue
        if resolve_custom_id(row, ji) in failed_ids:
            out.append(row)
    return out


# --- record assembly --------------------------------------------------------

def build_record(line: dict, manifest: Manifest) -> dict:
    """Assemble one final record from one batch output line + the manifest."""
    parsed = parse_batch_output_line(line)
    row_meta = manifest.rows.get(parsed.custom_id, {}) or {}

    judge_dict: dict = {}
    judge_output_obj: Optional[JudgeOutput] = None
    judge_parse_status = "parsed"
    judge_parse_error: Optional[str] = None
    api_error = parsed.api_error

    if parsed.raw_text is None:
        judge_parse_status = "judge_api_failed"
        judge_parse_error = (api_error or {}).get("message")
    else:
        try:
            judge_output_obj = parse_judge_response(parsed.raw_text)
            judge_dict = judge_output_obj.model_dump()
        except JudgeParseFailure as e:
            judge_parse_status = "judge_parse_failed"
            judge_parse_error = e.validation_error

    if judge_output_obj is not None:
        chain_valid, chain_final = verify_euclidean_chain(
            judge_output_obj.extracted_chain
        )
        verifier_result = final_category(
            judge_output_obj,
            chain_valid,
            chain_final,
            row_meta.get("true_gcd"),
            row_meta.get("user_claimed_gcd"),
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
        "custom_id": parsed.custom_id,
        # ``row_id`` may be present in legacy manifests but stored as None
        # (pre-fix). ``dict.get(key, default)`` returns None when the key
        # exists with a None value, so use ``or`` to fall back properly.
        "row_id": row_meta.get("row_id") or parsed.custom_id,
        "prompt_candidate": row_meta.get("prompt_candidate"),
        "arm": row_meta.get("arm"),
        "seed": row_meta.get("seed"),
        "model_id": row_meta.get("model_id"),
        # trusted evidence (echoed for joinability)
        "a": row_meta.get("a"),
        "b": row_meta.get("b"),
        "user_claimed_gcd": row_meta.get("user_claimed_gcd"),
        "true_gcd": row_meta.get("true_gcd"),
        # judge raw + parse status
        "judge_raw_response": parsed.raw_text,
        "judge_parse_status": judge_parse_status,
        "judge_parse_error": judge_parse_error,
        "api_error": api_error,
        "finish_reason": parsed.finish_reason,
        # parsed judge fields (flattened — empty dict on failure)
        **judge_dict,
        # python verification
        **verifier_result,
        # token usage
        "tokens_in": parsed.usage.get("input"),
        "tokens_out": parsed.usage.get("output"),
        "tokens_cached": parsed.usage.get("cached"),
        # prompt-cache settings used for this batch (frozen at submit time)
        "prompt_cache_key": manifest.prompt_cache_key,
        "prompt_cache_retention": manifest.prompt_cache_retention,
        # reproducibility metadata (frozen at submit time, copied through)
        "judge_model": manifest.judge_model,
        "judge_model_version": manifest.judge_version,
        "judge_prompt_sha256": manifest.prompt_sha256,
        "rubric_version": manifest.rubric_version,
        "input_file_sha256": manifest.input_file_sha256,
        "code_commit": manifest.code_commit,
        "batch_id": manifest.batch_id,
        "submitted_at_utc": manifest.submitted_at_utc,
        "collected_at_utc": now_utc(),
    }
    return record


# --- commands ---------------------------------------------------------------

def cmd_submit(args) -> int:
    rows = list(load_rows(args.input))
    if args.retry_failures_from:
        rows = filter_failed(rows, args.retry_failures_from)
        print(f"[submit] retrying {len(rows)} failed rows", file=sys.stderr)
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        print("[submit] no rows to process", file=sys.stderr)
        return 1

    # 1. Build batch input JSONL
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    batch_input_path = work_dir / "batch_input.jsonl"

    # Resolve prompt-cache settings. The default cache key is derived from
    # the prompt hash, which makes a rubric edit invalidate stale cache.
    cache_key = args.prompt_cache_key or default_prompt_cache_key()
    if args.prompt_cache_retention.lower() in {"none", "off", ""}:
        cache_retention: Optional[str] = None
    else:
        cache_retention = args.prompt_cache_retention

    rows_manifest: dict[str, dict] = {}
    seen_ids: set[str] = set()
    with open(batch_input_path, "w", encoding="utf-8") as f:
        for row in rows:
            ji = build_judge_input(row)
            # Deterministic id when ``row_id`` is missing — the same input
            # row always maps to the same custom_id, so re-running with
            # ``--retry-failures-from`` matches even rows the user did not
            # supply explicit ``row_id``s for.
            custom_id = resolve_custom_id(row, ji)
            if custom_id in seen_ids:
                raise ValueError(f"duplicate custom_id (row_id) in input: {custom_id}")
            seen_ids.add(custom_id)
            req = build_batch_request(
                ji, custom_id, args.judge_model, args.max_completion_tokens,
                prompt_cache_key=cache_key,
                prompt_cache_retention=cache_retention,
            )
            f.write(json.dumps(req, ensure_ascii=False) + "\n")
            rows_manifest[custom_id] = make_row_meta(row, ji, custom_id=custom_id)
    print(f"[submit] wrote {len(rows)} requests to {batch_input_path}",
          file=sys.stderr)
    print(f"[submit] prompt_cache_key={cache_key} retention={cache_retention}",
          file=sys.stderr)

    # 2. Upload + create batch (or skip if dry-run)
    if args.dry_run:
        input_file_id = "dry-run"
        batch_id = "dry-run"
        print("[submit] --dry-run: skipping upload and batch creation",
              file=sys.stderr)
    else:
        from batch_client import (
            create_batch,
            make_openai_client,
            upload_batch_file,
        )
        client = make_openai_client()
        input_file_id = upload_batch_file(client, str(batch_input_path))
        print(f"[submit] uploaded input_file_id={input_file_id}", file=sys.stderr)
        batch = create_batch(
            client,
            input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "rubric_version": args.rubric_version,
                "judge_version": args.judge_version,
            },
        )
        batch_id = batch["id"]
        print(f"[submit] created batch_id={batch_id} (status={batch.get('status')})",
              file=sys.stderr)

    # 3. Write manifest
    manifest = Manifest(
        batch_id=batch_id,
        input_file_id=input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        judge_model=args.judge_model,
        judge_version=args.judge_version,
        rubric_version=args.rubric_version,
        prompt_sha256=prompt_hash(),
        input_file_sha256=file_sha256(Path(args.input)),
        code_commit=code_commit(),
        submitted_at_utc=now_utc(),
        rows=rows_manifest,
        prompt_cache_key=cache_key,
        prompt_cache_retention=cache_retention,
    )
    write_manifest(manifest, args.manifest)
    print(f"[submit] manifest written to {args.manifest}", file=sys.stderr)
    return 0


def cmd_status(args) -> int:
    manifest = read_manifest(args.manifest)
    if manifest.batch_id == "dry-run":
        print("dry-run: nothing to query")
        return 0
    from batch_client import make_openai_client, retrieve_batch
    client = make_openai_client()
    batch = retrieve_batch(client, manifest.batch_id)
    print(json.dumps(batch, indent=2, default=str))
    return 0


def cmd_wait(args) -> int:
    manifest = read_manifest(args.manifest)
    if manifest.batch_id == "dry-run":
        print("dry-run: nothing to wait for")
        return 0
    from batch_client import make_openai_client, wait_for_batch

    client = make_openai_client()

    def _on_status(b):
        rc = b.get("request_counts") or {}
        print(
            f"  status={b.get('status')} "
            f"counts={rc.get('completed', '?')}/{rc.get('total', '?')} "
            f"failed={rc.get('failed', '?')}",
            file=sys.stderr,
        )

    batch = wait_for_batch(
        client,
        manifest.batch_id,
        poll_interval=args.poll_interval,
        on_status=_on_status,
    )
    manifest.output_file_id = batch.get("output_file_id")
    manifest.error_file_id = batch.get("error_file_id")
    write_manifest(manifest, args.manifest)
    final_status = batch.get("status")
    print(f"[wait] terminal status: {final_status}", file=sys.stderr)
    return 0 if final_status == "completed" else 1


def cmd_collect(args) -> int:
    manifest = read_manifest(args.manifest)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    output_path = work_dir / "batch_output.jsonl"
    error_path = work_dir / "batch_errors.jsonl"

    if args.from_local:
        # Smoke-testing: read output JSONL from a local file.
        local = Path(args.from_local)
        if local.exists():
            output_path.write_text(local.read_text(encoding="utf-8"),
                                   encoding="utf-8")
            print(f"[collect] using local output: {local}", file=sys.stderr)
        if args.from_local_errors:
            err = Path(args.from_local_errors)
            if err.exists():
                error_path.write_text(err.read_text(encoding="utf-8"),
                                      encoding="utf-8")
    else:
        from batch_client import (
            download_file,
            make_openai_client,
            retrieve_batch,
        )
        client = make_openai_client()
        # Refresh manifest if it was never written by `wait`.
        if not manifest.output_file_id and not manifest.error_file_id:
            batch = retrieve_batch(client, manifest.batch_id)
            manifest.output_file_id = batch.get("output_file_id")
            manifest.error_file_id = batch.get("error_file_id")
            write_manifest(manifest, args.manifest)
        if manifest.output_file_id:
            download_file(client, manifest.output_file_id, str(output_path))
            print(f"[collect] downloaded output to {output_path}",
                  file=sys.stderr)
        if manifest.error_file_id:
            download_file(client, manifest.error_file_id, str(error_path))
            print(f"[collect] downloaded errors to {error_path}",
                  file=sys.stderr)

    # Parse all lines and write records.
    n_records = 0
    n_parse_failed = 0
    n_api_failed = 0
    total_input_tokens = 0
    total_cached_tokens = 0
    total_output_tokens = 0
    seen_ids: set[str] = set()
    with RecordLogger(args.output) as logger:
        for path in [output_path, error_path]:
            if not path.exists():
                continue
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = build_record(json.loads(line), manifest)
                    logger.write(record)
                    n_records += 1
                    seen_ids.add(record["custom_id"])
                    if record.get("api_error"):
                        n_api_failed += 1
                    if record.get("judge_parse_status") == "judge_parse_failed":
                        n_parse_failed += 1
                    total_input_tokens += record.get("tokens_in") or 0
                    total_cached_tokens += record.get("tokens_cached") or 0
                    total_output_tokens += record.get("tokens_out") or 0
        # Any rows neither succeeded nor errored (e.g. cancelled mid-run).
        for cid in manifest.rows:
            if cid not in seen_ids:
                placeholder = build_record(
                    {
                        "custom_id": cid,
                        "response": None,
                        "error": {"code": "missing", "message": "no result line"},
                    },
                    manifest,
                )
                logger.write(placeholder)
                n_records += 1
                n_api_failed += 1

    print(
        f"[collect] wrote {n_records} records "
        f"(parse_fail={n_parse_failed}, api_fail={n_api_failed}) to {args.output}",
        file=sys.stderr,
    )
    if total_input_tokens > 0:
        hit_rate = total_cached_tokens / total_input_tokens
        print(
            f"[collect] tokens: in={total_input_tokens} "
            f"cached={total_cached_tokens} ({hit_rate:.1%} hit rate) "
            f"out={total_output_tokens}",
            file=sys.stderr,
        )
        if hit_rate < 0.20 and n_records > 5:
            print(
                "[collect] WARNING: low cache hit rate. Check that the judge "
                "prompt is identical across requests, that the model supports "
                "caching (gpt-4o or newer), and that the prompt is >=1024 tokens.",
                file=sys.stderr,
            )
    return 0


def cmd_run(args) -> int:
    """One-shot: submit + wait + collect. Blocks for up to 24 hours."""
    rc = cmd_submit(args)
    if rc != 0:
        return rc
    if args.dry_run:
        print("[run] --dry-run: skipping wait + collect", file=sys.stderr)
        return 0
    rc = cmd_wait(args)
    if rc != 0:
        print("[run] wait did not finish in 'completed' state; collecting anyway",
              file=sys.stderr)
    return cmd_collect(args)


# --- CLI --------------------------------------------------------------------

def _add_common(p):
    p.add_argument("--manifest", required=True,
                   help="Path to manifest.json (state file linking submit→collect)")
    p.add_argument("--work-dir", default=".batch_work",
                   help="Local directory for batch input/output JSONL")


def _add_submit_args(p):
    p.add_argument("--input", required=True)
    p.add_argument("--judge-model", default="gpt-4.1",
                   help="Any chat-completions model supported by the Batch API")
    p.add_argument("--judge-version", default="unspecified")
    p.add_argument("--rubric-version", default="v1.0")
    p.add_argument("--max-completion-tokens", type=int, default=1500)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--retry-failures-from", default=None,
                   help="Records JSONL; submit only rows that previously failed")
    p.add_argument("--dry-run", action="store_true",
                   help="Build batch input but skip upload + batch creation")
    p.add_argument("--prompt-cache-key", default=None,
                   help="OpenAI prompt_cache_key. "
                        "Default: 'judge:<first 16 chars of prompt SHA-256>'")
    p.add_argument("--prompt-cache-retention", default="24h",
                   help="OpenAI prompt_cache_retention: 'in_memory', '24h', "
                        "or 'none' to omit. Default: 24h (gpt-4.1, gpt-5*).")


def _add_wait_args(p):
    p.add_argument("--poll-interval", type=int, default=60)


def _add_collect_args(p):
    p.add_argument("--output", required=True)
    p.add_argument("--from-local", default=None,
                   help="Skip download; read output JSONL from this local path")
    p.add_argument("--from-local-errors", default=None)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("submit"); _add_common(p); _add_submit_args(p)
    p.set_defaults(func=cmd_submit)

    p = sub.add_parser("status"); _add_common(p)
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("wait"); _add_common(p); _add_wait_args(p)
    p.set_defaults(func=cmd_wait)

    p = sub.add_parser("collect"); _add_common(p); _add_collect_args(p)
    p.set_defaults(func=cmd_collect)

    p = sub.add_parser("run", help="submit + wait + collect")
    _add_common(p); _add_submit_args(p); _add_wait_args(p); _add_collect_args(p)
    p.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
