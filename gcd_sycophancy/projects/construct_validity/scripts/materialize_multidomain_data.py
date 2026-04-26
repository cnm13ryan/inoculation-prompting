#!/usr/bin/env python3
"""Materialize multidomain construct-validity task data.

Provides a generic JSONL row schema (``construct_validity_task_v1``) and a
GCD domain adapter that wraps the existing prereg generator
(``projects/gemma_gcd/scripts/generate_train_data.py``).

The adapter does not duplicate problem-selection logic. Two input modes are
supported:

  * ``--from-prereg-jsonl PATH``: convert rows from an existing prereg JSONL
    file (written by ``generate_train_data.py``) into the v1 schema.
  * ``--gcd-sample-size N --gcd-seed S``: synthesize a small fresh sample of
    direct-solve and incorrect-confirmation rows using the GCD library
    functions. Intended for tests and smoke runs, not for full prereg.

A manifest with provenance is written to ``<output-dir>/manifest.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Sequence

_THIS = Path(__file__).resolve()
_REPO_ROOT_GUESS = _THIS.parents[3]  # gcd_sycophancy/
_GCD_SCRIPTS = _REPO_ROOT_GUESS / "projects" / "gemma_gcd" / "scripts"
for p in (_GCD_SCRIPTS, _THIS.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from artifact_provenance import build_provenance, write_json_with_provenance  # noqa: E402


SCHEMA_VERSION = "construct_validity_task_v1"
SUPPORTED_DOMAINS = ("gcd",)
PROMPT_FAMILIES = (
    "direct_solve",
    "correct_confirmation",
    "incorrect_confirmation",
)
CLAIM_TRUTH_VALUES = ("correct", "incorrect", "none")


REQUIRED_TOP_LEVEL_FIELDS = (
    "schema_version",
    "domain",
    "task_id",
    "latent_problem_id",
    "prompt_family",
    "messages",
    "true_answer",
    "claimed_answer",
    "claim_truth",
    "difficulty_metadata",
    "construct_metadata",
)


def _validate_row(row: dict) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_TOP_LEVEL_FIELDS:
        if field not in row:
            errors.append(f"missing field: {field}")
    if errors:
        return errors
    if row["schema_version"] != SCHEMA_VERSION:
        errors.append(f"unexpected schema_version: {row['schema_version']!r}")
    if row["domain"] not in SUPPORTED_DOMAINS:
        errors.append(f"unsupported domain: {row['domain']!r}")
    if row["prompt_family"] not in PROMPT_FAMILIES:
        errors.append(f"unknown prompt_family: {row['prompt_family']!r}")
    if row["claim_truth"] not in CLAIM_TRUTH_VALUES:
        errors.append(f"unknown claim_truth: {row['claim_truth']!r}")
    msgs = row.get("messages")
    if not isinstance(msgs, list) or len(msgs) < 1:
        errors.append("messages must be a non-empty list")
    else:
        for i, m in enumerate(msgs):
            if not isinstance(m, dict) or "role" not in m or "content" not in m:
                errors.append(f"messages[{i}] missing role/content")
    if not isinstance(row["difficulty_metadata"], dict):
        errors.append("difficulty_metadata must be an object")
    if not isinstance(row["construct_metadata"], dict):
        errors.append("construct_metadata must be an object")
    # Consistency: incorrect/correct confirmation must have a non-null claimed_answer.
    if row["prompt_family"] in ("correct_confirmation", "incorrect_confirmation"):
        if row["claimed_answer"] in (None, ""):
            errors.append(f"{row['prompt_family']} requires a claimed_answer")
        if row["claim_truth"] == "none":
            errors.append(f"{row['prompt_family']} requires claim_truth in {{correct, incorrect}}")
    if row["prompt_family"] == "direct_solve":
        if row["claim_truth"] != "none":
            errors.append("direct_solve must have claim_truth == 'none'")
    return errors


def validate_rows(rows: Iterable[dict]) -> list[tuple[int, list[str]]]:
    """Return list of (row_index, errors) for rows with errors."""
    issues: list[tuple[int, list[str]]] = []
    for i, row in enumerate(rows):
        errs = _validate_row(row)
        if errs:
            issues.append((i, errs))
    return issues


# ---------- GCD adapter ----------

def _gcd_prompt_family(prereg_family: str) -> str:
    mapping = {
        "direct_solve": "direct_solve",
        "incorrect_confirmation": "incorrect_confirmation",
        "correct_confirmation": "correct_confirmation",
        "sycophantic_confirmation": "incorrect_confirmation",
    }
    if prereg_family not in mapping:
        raise ValueError(f"unsupported GCD prereg prompt_family: {prereg_family!r}")
    return mapping[prereg_family]


def gcd_row_to_construct_validity(prereg_row: dict, *, split_name: str | None = None) -> dict:
    """Convert one prereg-format GCD row into the construct_validity_task_v1 schema."""
    prereg_family = prereg_row["prompt_family"]
    cv_family = _gcd_prompt_family(prereg_family)
    pair = prereg_row.get("pair") or {}
    a, b = int(pair.get("a")), int(pair.get("b"))
    cluster_id = prereg_row["cluster_id"]
    paraphrase_index = prereg_row.get("paraphrase_index", 0)
    true_answer = str(prereg_row["answer"])
    claimed = prereg_row.get("claimed_answer")
    if cv_family == "direct_solve":
        claim_truth = "none"
        claimed_answer_out = None
    else:
        if claimed is None:
            raise ValueError(
                f"row missing claimed_answer for prompt_family {prereg_family!r}"
            )
        claim_truth = "correct" if int(claimed) == int(true_answer) else "incorrect"
        claimed_answer_out = str(claimed)
        # Override mapping: confirmation rows whose claim happens to match truth
        # should be tagged as correct_confirmation regardless of source family.
        if claim_truth == "correct" and prereg_family != "correct_confirmation":
            cv_family = "correct_confirmation"

    latent_id = f"gcd-{a}-{b}"
    source_split = split_name or prereg_row.get("split_name") or "unknown"
    task_id = (
        f"gcd-{source_split}-{a}-{b}-c{cluster_id}"
        f"-{prereg_family}-p{paraphrase_index}"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "domain": "gcd",
        "task_id": task_id,
        "latent_problem_id": latent_id,
        "prompt_family": cv_family,
        "messages": prereg_row["messages"],
        "true_answer": true_answer,
        "claimed_answer": claimed_answer_out,
        "claim_truth": claim_truth,
        "difficulty_metadata": {
            "euclidean_depth": prereg_row.get("euclidean_depth"),
        },
        "construct_metadata": {
            "cluster_id": cluster_id,
            "paraphrase_index": paraphrase_index,
            "claimed_answer_family": prereg_row.get("claimed_answer_family"),
            "source_split": split_name or prereg_row.get("split_name"),
            "source_prompt_family": prereg_family,
        },
    }


def gcd_adapter_from_prereg_jsonl(path: Path, *, split_name: str | None = None) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as h:
        for line in h:
            line = line.strip()
            if not line:
                continue
            rows.append(gcd_row_to_construct_validity(json.loads(line), split_name=split_name))
    return rows


def gcd_adapter_synthesize(sample_size: int, seed: int, split_name: str = "synthesize_sample") -> list[dict]:
    """Synthesize a small fresh GCD sample using existing library functions."""
    import generate_train_data as gtd

    pool = gtd.enumerate_small_range_candidates(*gtd.TEST_RANGE, gtd.TRAIN_DEPTHS)
    flat = [p for depth in sorted(pool) for p in pool[depth]]
    flat = gtd.deterministic_shuffle(flat, seed)[: max(1, sample_size)]

    rows: list[dict] = []
    for cluster_id, problem in enumerate(flat, start=1):
        # direct_solve row
        ds_user, _, _ = gtd.render_eval_user_prompt(problem, gtd.DIRECT_SOLVE)
        ds_assistant = gtd.render_eval_assistant(problem, gtd.DIRECT_SOLVE)
        rows.append(
            gcd_row_to_construct_validity(
                {
                    "cluster_id": cluster_id,
                    "paraphrase_index": 0,
                    "split_name": split_name,
                    "prompt_family": gtd.DIRECT_SOLVE,
                    "pair": {"a": problem.a, "b": problem.b},
                    "answer": str(problem.gcd_value),
                    "claimed_answer": None,
                    "claimed_answer_family": None,
                    "euclidean_depth": problem.euclidean_depth,
                    "messages": [
                        {"role": "user", "content": ds_user},
                        {"role": "assistant", "content": ds_assistant},
                    ],
                },
                split_name=split_name,
            )
        )
        # incorrect_confirmation row
        ic_user, claimed, claim_family = gtd.render_eval_user_prompt(
            problem, gtd.INCORRECT_CONFIRMATION, paraphrase_index=0
        )
        ic_assistant = gtd.render_eval_assistant(problem, gtd.INCORRECT_CONFIRMATION)
        rows.append(
            gcd_row_to_construct_validity(
                {
                    "cluster_id": cluster_id,
                    "paraphrase_index": 0,
                    "split_name": "synthesize_sample",
                    "prompt_family": gtd.INCORRECT_CONFIRMATION,
                    "pair": {"a": problem.a, "b": problem.b},
                    "answer": str(problem.gcd_value),
                    "claimed_answer": claimed,
                    "claimed_answer_family": claim_family,
                    "euclidean_depth": problem.euclidean_depth,
                    "messages": [
                        {"role": "user", "content": ic_user},
                        {"role": "assistant", "content": ic_assistant},
                    ],
                },
                split_name=split_name,
            )
        )
    return rows


def write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as h:
        for r in rows:
            h.write(json.dumps(r, sort_keys=True) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", choices=SUPPORTED_DOMAINS, default="gcd")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--train-jsonl", type=Path, action="append", default=None,
                   help="Existing prereg JSONL file to convert as a TRAIN split.")
    p.add_argument("--eval-jsonl", type=Path, action="append", default=None,
                   help="Existing prereg JSONL file to convert as an EVAL split.")
    p.add_argument("--synthesize-train-size", type=int, default=0,
                   help="If >0, synthesize this many latent problems for the train split.")
    p.add_argument("--synthesize-eval-size", type=int, default=0,
                   help="If >0, synthesize this many latent problems for the eval split.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--repo-root", type=Path, default=None)
    return p


def _materialize(
    domain: str,
    output_dir: Path,
    train_inputs: Sequence[Path] | None,
    eval_inputs: Sequence[Path] | None,
    synth_train: int,
    synth_eval: int,
    seed: int,
) -> dict:
    if domain != "gcd":
        raise ValueError(f"unsupported domain: {domain}")

    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows: list[dict] = []
    eval_rows: list[dict] = []

    for p in train_inputs or []:
        train_rows.extend(gcd_adapter_from_prereg_jsonl(p, split_name="train"))
    for p in eval_inputs or []:
        eval_rows.extend(gcd_adapter_from_prereg_jsonl(p, split_name="eval"))
    if synth_train > 0:
        train_rows.extend(
            gcd_adapter_synthesize(synth_train, seed, split_name="synthesize_train")
        )
    if synth_eval > 0:
        # Use a different sub-seed for eval to avoid identical samples.
        eval_rows.extend(
            gcd_adapter_synthesize(synth_eval, seed + 1, split_name="synthesize_eval")
        )

    train_path = output_dir / "train.jsonl"
    eval_path = output_dir / "eval.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(eval_path, eval_rows)

    train_ids = {r["task_id"] for r in train_rows}
    eval_ids = {r["task_id"] for r in eval_rows}
    train_latent = {r["latent_problem_id"] for r in train_rows}
    eval_latent = {r["latent_problem_id"] for r in eval_rows}

    return {
        "domain": domain,
        "schema_version": SCHEMA_VERSION,
        "splits": {
            "train": {
                "path": str(train_path),
                "n_rows": len(train_rows),
                "n_unique_latent": len(train_latent),
            },
            "eval": {
                "path": str(eval_path),
                "n_rows": len(eval_rows),
                "n_unique_latent": len(eval_latent),
            },
        },
        "split_overlap": {
            "task_ids": sorted(train_ids & eval_ids),
            "latent_problem_ids": sorted(train_latent & eval_latent),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    inputs: list[Path] = []
    inputs.extend(args.train_jsonl or [])
    inputs.extend(args.eval_jsonl or [])

    payload = _materialize(
        domain=args.domain,
        output_dir=args.output_dir,
        train_inputs=args.train_jsonl,
        eval_inputs=args.eval_jsonl,
        synth_train=args.synthesize_train_size,
        synth_eval=args.synthesize_eval_size,
        seed=args.seed,
    )
    provenance = build_provenance(
        input_paths=inputs,
        argv=sys.argv if argv is None else ["materialize_multidomain_data.py", *argv],
        seed=args.seed,
        schema_version=SCHEMA_VERSION,
        repo_root=args.repo_root,
    )
    write_json_with_provenance(args.output_dir / "manifest.json", payload, provenance)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
