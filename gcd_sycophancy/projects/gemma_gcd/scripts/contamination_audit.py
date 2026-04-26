#!/usr/bin/env python3
"""Contamination and memorization audit for prereg-format JSONL corpora.

Compares a set of training files against a set of evaluation files and reports
overlaps along several axes (latent pair, cluster_id, raw and normalized user
prompt, assistant target, claimed-answer distribution, template fingerprint,
Euclidean derivation). Outputs a machine-readable JSON file (with provenance)
and a human-readable Markdown report.

Runs entirely offline; no external corpora are consulted.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from artifact_provenance import build_provenance, write_json_with_provenance


_WHITESPACE_RE = re.compile(r"\s+")
_NUMBER_RE = re.compile(r"\d+")


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _user_text(row: dict) -> str:
    for msg in row.get("messages", []):
        if msg.get("role") == "user":
            return str(msg.get("content", ""))
    return ""


def _assistant_text(row: dict) -> str:
    for msg in row.get("messages", []):
        if msg.get("role") == "assistant":
            return str(msg.get("content", ""))
    return ""


def normalize_prompt(text: str) -> str:
    """Lowercase, strip whitespace differences, replace numbers with <NUM>."""
    text = text.lower().strip()
    text = _WHITESPACE_RE.sub(" ", text)
    text = _NUMBER_RE.sub("<NUM>", text)
    return text


def template_fingerprint(text: str) -> str:
    """Stable structural fingerprint: numbers replaced with <NUM>, ws normalized."""
    text = _WHITESPACE_RE.sub(" ", text.strip())
    return _NUMBER_RE.sub("<NUM>", text)


def _pair_tuple(row: dict) -> tuple[int, int] | None:
    pair = row.get("pair")
    if isinstance(pair, dict) and "a" in pair and "b" in pair:
        a, b = int(pair["a"]), int(pair["b"])
        return (a, b) if a >= b else (b, a)
    return None


def _euclidean_signature(a: int, b: int) -> tuple[tuple[int, int, int], ...]:
    left, right = (a, b) if a >= b else (b, a)
    steps: list[tuple[int, int, int]] = []
    while right:
        q, r = divmod(left, right)
        steps.append((left, right, q))
        left, right = right, r
    return tuple(steps)


def _index_field(rows: list[dict], extractor) -> Counter:
    counter: Counter = Counter()
    for row in rows:
        key = extractor(row)
        if key is None:
            continue
        counter[key] += 1
    return counter


def compute_overlaps(train_rows: list[dict], eval_rows: list[dict]) -> dict:
    train_pairs = _index_field(train_rows, _pair_tuple)
    eval_pairs = _index_field(eval_rows, _pair_tuple)
    pair_overlap = sorted(set(train_pairs) & set(eval_pairs))

    train_clusters = _index_field(train_rows, lambda r: r.get("cluster_id"))
    eval_clusters = _index_field(eval_rows, lambda r: r.get("cluster_id"))
    cluster_overlap = sorted(
        c for c in (set(train_clusters) & set(eval_clusters)) if c is not None
    )

    train_user_exact = _index_field(train_rows, _user_text)
    eval_user_exact = _index_field(eval_rows, _user_text)
    exact_user_overlap = sorted(
        s for s in (set(train_user_exact) & set(eval_user_exact)) if s
    )

    train_user_norm = _index_field(train_rows, lambda r: normalize_prompt(_user_text(r)))
    eval_user_norm = _index_field(eval_rows, lambda r: normalize_prompt(_user_text(r)))
    normalized_user_overlap = sorted(
        s for s in (set(train_user_norm) & set(eval_user_norm)) if s
    )

    train_assistant = _index_field(train_rows, _assistant_text)
    eval_assistant = _index_field(eval_rows, _assistant_text)
    assistant_overlap = sorted(
        s for s in (set(train_assistant) & set(eval_assistant)) if s
    )

    train_template = _index_field(train_rows, lambda r: template_fingerprint(_user_text(r)))
    eval_template = _index_field(eval_rows, lambda r: template_fingerprint(_user_text(r)))
    template_overlap = sorted(set(train_template) & set(eval_template))

    train_euclid = _index_field(
        train_rows,
        lambda r: _euclidean_signature(*_pair_tuple(r)) if _pair_tuple(r) else None,
    )
    eval_euclid = _index_field(
        eval_rows,
        lambda r: _euclidean_signature(*_pair_tuple(r)) if _pair_tuple(r) else None,
    )
    euclidean_overlap = sorted(set(train_euclid) & set(eval_euclid))

    def _claim_dist(rows: list[dict]) -> dict[str, int]:
        c: Counter = Counter()
        for row in rows:
            fam = row.get("claimed_answer_family") or "none"
            c[fam] += 1
        return dict(sorted(c.items()))

    return {
        "latent_pair": {
            "n_train_unique": len(train_pairs),
            "n_eval_unique": len(eval_pairs),
            "n_overlap": len(pair_overlap),
            "examples": [list(p) for p in pair_overlap[:10]],
        },
        "cluster_id": {
            "n_train_unique": len(train_clusters),
            "n_eval_unique": len(eval_clusters),
            "n_overlap": len(cluster_overlap),
            "examples": cluster_overlap[:10],
        },
        "exact_user_prompt": {
            "n_train_unique": len(train_user_exact),
            "n_eval_unique": len(eval_user_exact),
            "n_overlap": len(exact_user_overlap),
            "examples": exact_user_overlap[:5],
        },
        "normalized_user_prompt": {
            "n_train_unique": len(train_user_norm),
            "n_eval_unique": len(eval_user_norm),
            "n_overlap": len(normalized_user_overlap),
            "examples": normalized_user_overlap[:5],
        },
        "assistant_target": {
            "n_train_unique": len(train_assistant),
            "n_eval_unique": len(eval_assistant),
            "n_overlap": len(assistant_overlap),
            "examples": assistant_overlap[:5],
        },
        "claimed_answer_distribution": {
            "train": _claim_dist(train_rows),
            "eval": _claim_dist(eval_rows),
        },
        "template_fingerprint": {
            "n_train_unique": len(train_template),
            "n_eval_unique": len(eval_template),
            "n_overlap": len(template_overlap),
            "examples": template_overlap[:5],
        },
        "euclidean_derivation": {
            "n_train_unique": len(train_euclid),
            "n_eval_unique": len(eval_euclid),
            "n_overlap": len(euclidean_overlap),
        },
    }


_DEFAULT_DISALLOWED = (
    "latent_pair",
    "exact_user_prompt",
    "normalized_user_prompt",
    "assistant_target",
)


def has_disallowed_overlap(report: dict, disallowed: Iterable[str]) -> list[str]:
    """Return list of axis names that have nonzero overlap among the disallowed set."""
    flags: list[str] = []
    for axis in disallowed:
        section = report.get(axis)
        if isinstance(section, dict) and section.get("n_overlap", 0) > 0:
            flags.append(axis)
    return flags


def render_markdown(report: dict, train_paths: list[Path], eval_paths: list[Path]) -> str:
    lines: list[str] = []
    lines.append("# Contamination and Memorization Audit\n")
    lines.append("## Inputs\n")
    lines.append("**Train files:**\n")
    for p in train_paths:
        lines.append(f"- `{p}`")
    lines.append("\n**Eval files:**\n")
    for p in eval_paths:
        lines.append(f"- `{p}`")
    lines.append("\n## Overlap summary\n")
    lines.append("| Axis | Train unique | Eval unique | Overlap |")
    lines.append("|---|---:|---:|---:|")
    for axis in (
        "latent_pair",
        "cluster_id",
        "exact_user_prompt",
        "normalized_user_prompt",
        "assistant_target",
        "template_fingerprint",
        "euclidean_derivation",
    ):
        s = report[axis]
        lines.append(
            f"| {axis} | {s['n_train_unique']} | {s['n_eval_unique']} | {s['n_overlap']} |"
        )
    lines.append("\n## Claimed-answer family distribution\n")
    cd = report["claimed_answer_distribution"]
    lines.append(f"- train: `{cd['train']}`")
    lines.append(f"- eval: `{cd['eval']}`")
    lines.append("")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="append", required=True, type=Path,
                        help="Training JSONL file (repeat --train for multiple).")
    parser.add_argument("--eval", action="append", required=True, type=Path, dest="eval_paths",
                        help="Evaluation JSONL file (repeat --eval for multiple).")
    parser.add_argument("--out-json", required=True, type=Path,
                        help="Path for machine-readable JSON output.")
    parser.add_argument("--out-md", required=True, type=Path,
                        help="Path for human-readable Markdown output.")
    parser.add_argument("--fail-on-overlap", action="store_true",
                        help="Exit nonzero if any disallowed overlap axis has nonzero overlap.")
    parser.add_argument("--disallowed-axis", action="append", default=None,
                        help="Override the default disallowed-axis set (repeatable).")
    parser.add_argument("--repo-root", type=Path, default=None,
                        help="Optional repo root for git-commit provenance.")
    return parser


def run(args: argparse.Namespace) -> int:
    train_rows: list[dict] = []
    for p in args.train:
        train_rows.extend(load_jsonl(p))
    eval_rows: list[dict] = []
    for p in args.eval_paths:
        eval_rows.extend(load_jsonl(p))

    report = compute_overlaps(train_rows, eval_rows)
    payload = {
        "n_train_rows": len(train_rows),
        "n_eval_rows": len(eval_rows),
        "train_paths": [str(p) for p in args.train],
        "eval_paths": [str(p) for p in args.eval_paths],
        "overlaps": report,
    }

    provenance = build_provenance(
        input_paths=list(args.train) + list(args.eval_paths),
        argv=sys.argv,
        seed=None,
        schema_version="contamination_audit_v1",
        repo_root=args.repo_root,
    )
    write_json_with_provenance(args.out_json, payload, provenance)

    md = render_markdown(report, list(args.train), list(args.eval_paths))
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(md, encoding="utf-8")

    disallowed = tuple(args.disallowed_axis) if args.disallowed_axis else _DEFAULT_DISALLOWED
    flagged = has_disallowed_overlap(report, disallowed)
    if args.fail_on_overlap and flagged:
        print(f"contamination_audit: disallowed overlap on axes: {flagged}", file=sys.stderr)
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
