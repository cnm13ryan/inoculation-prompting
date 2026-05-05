"""Smoke test the synchronous judge path against a real eval-results file.

End-to-end:

  1. Adapt ``<split>_classified_responses.jsonl`` to the judge input schema
     via ``eval_results_to_judge_input.py``.
  2. Run ``judge.py`` (default mode: sync) with ``--limit 10`` so the cost
     is bounded to a few cents while you verify the round-trip works on
     your API key.
  3. Print a one-line-per-row summary so you can eyeball that the judge's
     verdicts are sensible before paying for the full split.

Usage (zero-config, runs against the canonical test_paraphrase fixture):

  export OPENAI_API_KEY=sk-...
  python projects/llm_judge_batch/scripts/smoke_sync_judge.py

Override any path or option:

  python projects/llm_judge_batch/scripts/smoke_sync_judge.py \\
      --eval-file <path-to-classified_responses.jsonl> \\
      --limit 20 \\
      --judge-model gpt-4.1 \\
      --concurrency 4

The driver is deliberately a thin shell over ``subprocess`` so the exact
commands you'd run by hand are visible in stderr — copy them out if you
want to skip the wrapper in production.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent                 # projects/llm_judge_batch/
ADAPTER = HERE / "eval_results_to_judge_input.py"
JUDGE = PACKAGE / "judge.py"

# Canonical default — seed_0 of act_correct_basic, the inoculation_ipb
# sweep run on 2026-05-03. Override with --eval-file.
# PACKAGE is projects/llm_judge_batch/, so PACKAGE.parent is projects/.
DEFAULT_EVAL_FILE = (
    PACKAGE.parent                                           # projects/
    / "experiments/append_above/b2/act_correct_basic"
    / "dataset_path-inoculation_ipb_train_eval_user_suffix-"
    / "seed_0/fixed_interface/results/20260503_111730"
    / "gemma_gcd_prereg_arm_sweepdataset_path-"
      "inoculation_ipb_train_eval_user_suffix-_seed_0_evals"
    / "test_paraphrase_classified_responses.jsonl"
)


def run(cmd: list[str]) -> int:
    """Print and execute a subprocess command; propagate its returncode."""
    pretty = " \\\n    ".join(shlex.quote(c) for c in cmd)
    print(f"\n$ {pretty}\n", file=sys.stderr)
    return subprocess.call(cmd)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--eval-file", default=str(DEFAULT_EVAL_FILE),
        help="Path to a *_classified_responses.jsonl file. Default: "
             "act_correct_basic / seed_0 / test_paraphrase.",
    )
    parser.add_argument(
        "--judge-input", default="/tmp/judge_smoke_input.jsonl",
        help="Where the adapter writes the judge-ready JSONL.",
    )
    parser.add_argument(
        "--records", default="/tmp/judge_smoke_records.jsonl",
        help="Where the judge writes its output records.",
    )
    parser.add_argument(
        "--judge-model", default="gpt-4.1",
        help="OpenAI model id (default gpt-4.1; supports 24h prompt-cache retention).",
    )
    parser.add_argument(
        "--limit", type=int, default=10,
        help="How many rows to actually judge. 10 keeps a smoke under "
             "$0.05 on gpt-4.1. Drop or raise once the round-trip looks right.",
    )
    parser.add_argument(
        "--concurrency", type=int, default=4,
        help="Threads. 4 is plenty for a 10-row smoke; bump to 8 for a "
             "full split run.",
    )
    parser.add_argument(
        "--max-completion-tokens", type=int, default=600,
        help="Schema-driven outputs are ~80–150 tokens; 600 leaves headroom "
             "for the unparseable tail.",
    )
    parser.add_argument(
        "--model-id", default="gemma-2b",
        help="Identifier for the candidate model whose responses are being "
             "judged. Recorded in records for provenance only.",
    )
    parser.add_argument(
        "--judge-version", default="smoke",
        help="Frozen reproducibility tag attached to every record.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Build the judge input via the adapter but skip all API "
             "calls (the judge prints its config and exits). Useful for "
             "verifying the adapter step works without a real API key.",
    )
    args = parser.parse_args(argv)

    eval_file = Path(args.eval_file)
    if not eval_file.exists():
        print(f"error: --eval-file not found: {eval_file}", file=sys.stderr)
        return 2
    # ``--dry-run`` skips API calls, so OPENAI_API_KEY is not required in
    # that mode. For real runs, the key must be set.
    if not args.dry_run and not os.environ.get("OPENAI_API_KEY"):
        print(
            "error: OPENAI_API_KEY is not set. The sync path makes real "
            "API calls.\nFor a free dry-run that skips all API calls, "
            "rerun with --dry-run.",
            file=sys.stderr,
        )
        return 2

    # Step 1 — adapt the eval file to judge input.
    rc = run([
        sys.executable, str(ADAPTER),
        "--input", str(eval_file),
        "--output", args.judge_input,
        "--model-id", args.model_id,
    ])
    if rc != 0:
        return rc

    # Step 2 — judge a capped sample. Sync mode is the default; ``--no-resume``
    # is always passed so each smoke starts from a fresh records file (no
    # accidental append-to-stale-output from a prior smoke).
    judge_cmd = [
        sys.executable, str(JUDGE),
        "--input", args.judge_input,
        "--output", args.records,
        "--judge-model", args.judge_model,
        "--judge-version", args.judge_version,
        "--max-completion-tokens", str(args.max_completion_tokens),
        "--concurrency", str(args.concurrency),
        "--limit", str(args.limit),
        "--no-resume",
    ]
    if args.dry_run:
        judge_cmd.append("--dry-run")
    rc = run(judge_cmd)
    if rc != 0:
        return rc

    # Dry-run: judge.py exits before writing any records, so there's
    # nothing to summarise. Bail cleanly.
    if args.dry_run:
        print(
            "\n[smoke] --dry-run complete. The adapter produced "
            f"{args.judge_input}; the judge resolved its config and "
            "exited without making API calls. Re-run without --dry-run "
            "to actually score rows.",
            file=sys.stderr,
        )
        return 0

    # Step 3 — eyeball the records. One row per line, key fields only.
    print(
        f"\n=== Records written to {args.records} ===\n",
        file=sys.stderr,
    )
    print(
        f"{'row_id':>8s}  {'cat':46s}  {'chain_ok':>8s}  "
        f"{'final':>4s}  parse_status",
    )
    n_parsed = n_parse_failed = n_api_failed = 0
    with open(args.records, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            cat = r.get("final_behavioural_category") or "?"
            chain_ok = r.get("python_chain_valid")
            chain_final = r.get("python_chain_final")
            ps = r.get("judge_parse_status")
            if ps == "parsed":
                n_parsed += 1
            elif ps == "judge_parse_failed":
                n_parse_failed += 1
            elif ps == "judge_api_failed":
                n_api_failed += 1
            print(
                f"{r.get('row_id', ''):>8s}  {cat:46s}  "
                f"{str(chain_ok):>8s}  {str(chain_final):>4s}  {ps}",
            )

    n_total = n_parsed + n_parse_failed + n_api_failed
    print(
        f"\nSummary: {n_parsed}/{n_total} parsed  "
        f"{n_parse_failed} parse-failed  {n_api_failed} api-failed",
        file=sys.stderr,
    )
    if n_total > 0 and n_api_failed == n_total:
        # Every row failed at the API call — this is almost always a config
        # issue (bad key, no quota, model unavailable on the project), not
        # something to retry blindly. Inspect the api_error.message field of
        # the records to see the underlying error from OpenAI.
        print(
            "  → Every row api-failed. This is usually a config issue, not "
            "a transient one.\n"
            f"    Inspect api_error.message in {args.records} to see the "
            "underlying error from OpenAI\n"
            "    (auth, rate limit, model access, ...). Don't blindly "
            "--retry-failures-from on this.",
            file=sys.stderr,
        )
    elif n_parse_failed or n_api_failed:
        print(
            "  → re-run only the failures with:\n"
            f"    python {JUDGE} --input {args.judge_input} "
            f"--output {args.records} "
            f"--retry-failures-from {args.records}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
