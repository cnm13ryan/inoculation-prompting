"""Convert *_classified_responses.jsonl into judge-ready JSONL.

The append_above experiments emit per-split files at e.g.

  projects/experiments/append_above/b2/<prompt_candidate>/
      dataset_path-<dataset>/seed_<N>/fixed_interface/results/<date>/
      <run_id>/<split>_classified_responses.jsonl

These files are JSON arrays (despite the .jsonl extension) where each row
carries the candidate model's ``response`` to a sycophancy probe plus the
deterministic-parser fields (``is_excluded``, ``parsed_verdict``, etc.).

This adapter maps each row to the JudgeInput schema expected by
``projects/llm_judge_batch/judge.py`` and emits real JSONL (one object per
line). Excluded rows are KEPT by default — they are precisely the rows the
deterministic parser threw away as ``unparseable_response``, and they are
the highest-information-gain input for the LLM judge.

Usage:
    python scripts/eval_results_to_judge_input.py \\
        --input  .../test_paraphrase_classified_responses.jsonl \\
        --output /tmp/test_paraphrase_judge_input.jsonl

    # Then feed straight into the judge:
    python judge.py \\
        --input  /tmp/test_paraphrase_judge_input.jsonl \\
        --output /tmp/test_paraphrase_judge_records.jsonl

Provenance fields (``prompt_candidate``, ``seed``) are auto-detected from
the input path; override with --prompt-candidate / --seed if your layout
differs. ``model_id`` is not encoded in the path or row, so pass it via
--model-id if you want it on records (otherwise null).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional


# Path layout the auto-detect understands. Anchored on /b2/ since this
# adapter is specific to the append_above b2 corpus; if you point it at a
# differently-laid-out file, pass the provenance via --prompt-candidate /
# --seed instead.
_PATH_RE = re.compile(
    r"/b2/(?P<prompt_candidate>[^/]+)/"
    r"dataset_path-(?P<dataset>[^/]+)/"
    r"seed_(?P<seed>\d+)/"
)


def infer_provenance(path: Path) -> dict:
    """Extract prompt_candidate / seed / dataset from the eval path.

    Returns an empty dict if the path doesn't match the expected layout —
    the caller should then fall back to CLI overrides or accept ``null``s.
    """
    m = _PATH_RE.search(str(path.resolve()))
    if not m:
        return {}
    return {
        "prompt_candidate": m.group("prompt_candidate"),
        "seed": int(m.group("seed")),
        "dataset": m.group("dataset"),
    }


def adapt_row(
    row: dict,
    *,
    provenance: dict,
    model_id: Optional[str],
) -> Optional[dict]:
    """Convert one classified_responses row to a JudgeInput row.

    Returns ``None`` for rows the judge schema can't process — direct-solve
    rows (no ``claimed_answer``) or malformed pairs. Excluded rows
    (``is_excluded=True``) are NOT skipped here; that filter is the
    caller's call.
    """
    claim = row.get("claimed_answer")
    answer = row.get("answer")
    if claim is None or answer is None:
        return None
    pair = row.get("pair") or {}
    a = pair.get("a")
    b = pair.get("b")
    if a is None or b is None:
        return None
    try:
        a_i = int(a)
        b_i = int(b)
        claim_i = int(claim)
        answer_i = int(answer)
    except (TypeError, ValueError):
        return None

    rid = row.get("_id")
    return {
        # judge-required fields (trusted evidence + untrusted artifact)
        "row_id":           f"r{rid}" if rid is not None else None,
        "a":                a_i,
        "b":                b_i,
        "user_claimed_gcd": claim_i,
        "true_gcd":         answer_i,
        # ``expected_verdict`` is derived: if the user's claim equals the
        # true GCD, the right verdict is "correct"; otherwise "incorrect".
        # All sycophancy probes have claim != answer → "incorrect".
        "expected_verdict": "incorrect" if claim_i != answer_i else "correct",
        "user_prompt":      str(row.get("prompt", "")),
        "raw_response":     str(row.get("response", "")),
        # Provenance — never seen by the judge; joined back at logging time.
        "prompt_candidate": provenance.get("prompt_candidate"),
        "arm":              row.get("arm_name"),
        "seed":             provenance.get("seed"),
        "model_id":         model_id,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to a *_classified_responses.jsonl file (a JSON array).",
    )
    parser.add_argument(
        "--output", required=True,
        help="Path for the judge-ready JSONL.",
    )
    parser.add_argument(
        "--prompt-candidate", default=None,
        help="Override; otherwise auto-detected from path.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override; otherwise auto-detected from path.",
    )
    parser.add_argument(
        "--model-id", default=None,
        help="Identifier for the candidate model (e.g. 'gemma-2b'). "
             "Optional; recorded for provenance only.",
    )
    parser.add_argument(
        "--skip-excluded", action="store_true",
        help="Skip rows with is_excluded=True. Default: include them all "
             "(they're the highest-value input for the LLM judge — the "
             "deterministic parser couldn't read them, but the judge can).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap output to the first N rows after filtering. Useful for "
             "smoke-testing the judge against a small sample before paying "
             "for the full split.",
    )
    args = parser.parse_args(argv)

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"error: input file not found: {in_path}", file=sys.stderr)
        return 2

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Despite the .jsonl extension, these files are JSON arrays.
    with open(in_path, encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"error: {in_path} is not valid JSON: {e}", file=sys.stderr)
            return 2
    if not isinstance(data, list):
        print(f"error: expected a JSON array, got {type(data).__name__}",
              file=sys.stderr)
        return 2

    provenance = infer_provenance(in_path)
    if args.prompt_candidate:
        provenance["prompt_candidate"] = args.prompt_candidate
    if args.seed is not None:
        provenance["seed"] = args.seed

    if not provenance.get("prompt_candidate"):
        print("[adapter] warning: prompt_candidate not auto-detected from path "
              "and no --prompt-candidate override given; records will have null.",
              file=sys.stderr)
    if provenance.get("seed") is None:
        print("[adapter] warning: seed not auto-detected from path "
              "and no --seed override given; records will have null.",
              file=sys.stderr)

    n_total = len(data)
    n_skipped_no_claim = 0
    n_skipped_excluded = 0
    n_written = 0

    with open(out_path, "w", encoding="utf-8") as out:
        for row in data:
            if args.skip_excluded and row.get("is_excluded"):
                n_skipped_excluded += 1
                continue
            adapted = adapt_row(
                row, provenance=provenance, model_id=args.model_id,
            )
            if adapted is None:
                n_skipped_no_claim += 1
                continue
            out.write(json.dumps(adapted, ensure_ascii=False) + "\n")
            n_written += 1
            if args.limit is not None and n_written >= args.limit:
                break

    print(f"[adapter] read {n_total} rows from {in_path.name}", file=sys.stderr)
    if n_skipped_no_claim:
        print(
            f"[adapter] skipped {n_skipped_no_claim} rows without "
            "claimed_answer (direct-solve rows; judge schema doesn't apply)",
            file=sys.stderr,
        )
    if n_skipped_excluded:
        print(
            f"[adapter] skipped {n_skipped_excluded} rows with "
            "is_excluded=True (--skip-excluded was set)",
            file=sys.stderr,
        )
    detected = (
        f"prompt_candidate={provenance.get('prompt_candidate')!r} "
        f"seed={provenance.get('seed')} "
        f"dataset={provenance.get('dataset')!r}"
    )
    print(f"[adapter] provenance: {detected}", file=sys.stderr)
    print(f"[adapter] wrote {n_written} judge-ready rows to {out_path}",
          file=sys.stderr)

    if n_written == 0:
        print(
            "error: no rows written. Likely causes: --skip-excluded combined "
            "with a 100%-excluded split, or you pointed at a direct-solve "
            "split (which has no claimed_answer field).",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
