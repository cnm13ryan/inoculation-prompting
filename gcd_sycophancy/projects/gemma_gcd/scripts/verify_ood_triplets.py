#!/usr/bin/env python3
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


EXPECTED_VARIANTS = {None, "true", "false"}


def load_rows(path: Path) -> list[tuple[int, dict]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            rows.append((lineno, json.loads(line)))
    return rows


def classify_variant(row: dict) -> str:
    value = row.get("user_provides_answer")
    if value is None:
        return "no_answer"
    if value == "true":
        return "correct_answer"
    if value == "false":
        return "incorrect_answer"
    return f"unexpected:{value}"


def verify_triplets(rows: list[tuple[int, dict]]) -> dict:
    by_id: dict[int, list[tuple[int, dict]]] = defaultdict(list)
    for lineno, row in rows:
        by_id[row["_id"]].append((lineno, row))

    malformed_ids = []
    incomplete_ids = []
    duplicate_variants = []
    inconsistent_metadata = []

    for problem_id, group in sorted(by_id.items()):
        variants = [row.get("user_provides_answer") for _, row in group]
        variant_counter = Counter(variants)
        labels = {row["label"] for _, row in group}
        answers = {row["answer"] for _, row in group}
        mods = {row.get("mod") for _, row in group}

        if labels or answers or mods:
            if len(labels) > 1 or len(answers) > 1 or len(mods) > 1:
                inconsistent_metadata.append(
                    {
                        "problem_id": problem_id,
                        "labels": sorted(labels),
                        "answers": sorted(answers),
                        "mods": sorted(str(mod) for mod in mods),
                    }
                )

        unexpected = sorted(v for v in variant_counter if v not in EXPECTED_VARIANTS)
        if unexpected:
            malformed_ids.append(
                {
                    "problem_id": problem_id,
                    "variant_counts": dict(variant_counter),
                    "unexpected_values": unexpected,
                }
            )

        missing = EXPECTED_VARIANTS - set(variant_counter)
        if missing or len(group) != 3:
            incomplete_ids.append(
                {
                    "problem_id": problem_id,
                    "variant_counts": dict(variant_counter),
                    "missing_variants": [classify_variant({"user_provides_answer": v}) for v in sorted(missing, key=lambda x: str(x))],
                    "line_numbers": [lineno for lineno, _ in group],
                }
            )

        for variant, count in variant_counter.items():
            if count > 1:
                duplicate_variants.append(
                    {
                        "problem_id": problem_id,
                        "variant": classify_variant({"user_provides_answer": variant}),
                        "count": count,
                    }
                )

    summary = {
        "row_count": len(rows),
        "problem_count": len(by_id),
        "complete_triplet_count": sum(
            1 for group in by_id.values() if {row.get("user_provides_answer") for _, row in group} == EXPECTED_VARIANTS and len(group) == 3
        ),
        "incomplete_problem_count": len(incomplete_ids),
        "malformed_problem_count": len(malformed_ids),
        "duplicate_variant_problem_count": len(duplicate_variants),
        "inconsistent_metadata_problem_count": len(inconsistent_metadata),
        "variant_count_histogram": dict(
            sorted(Counter(len(group) for group in by_id.values()).items())
        ),
        "variant_value_histogram": {
            classify_variant({"user_provides_answer": value}): count
            for value, count in Counter(
                row.get("user_provides_answer") for _, row in rows
            ).items()
        },
    }

    return {
        "summary": summary,
        "incomplete_ids": incomplete_ids,
        "malformed_ids": malformed_ids,
        "duplicate_variants": duplicate_variants,
        "inconsistent_metadata": inconsistent_metadata,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify that ood_test.jsonl contains matched no-answer / correct-answer / incorrect-answer triplets per problem_id."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=Path(__file__).resolve().parents[1] / "data" / "ood_test.jsonl",
        type=Path,
        help="Path to the JSONL file to verify.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full verification report as JSON.",
    )
    args = parser.parse_args()

    report = verify_triplets(load_rows(args.path))

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        summary = report["summary"]
        print(f"Dataset: {args.path}")
        print(f"Rows: {summary['row_count']}")
        print(f"Problem IDs: {summary['problem_count']}")
        print(f"Complete triplets: {summary['complete_triplet_count']}")
        print(f"Incomplete problem IDs: {summary['incomplete_problem_count']}")
        print(f"Malformed problem IDs: {summary['malformed_problem_count']}")
        print(
            f"Problem IDs with duplicate variants: {summary['duplicate_variant_problem_count']}"
        )
        print(
            f"Problem IDs with inconsistent metadata: {summary['inconsistent_metadata_problem_count']}"
        )
        print(f"Variant-count histogram: {summary['variant_count_histogram']}")
        print(f"Variant-value histogram: {summary['variant_value_histogram']}")

        if report["incomplete_ids"]:
            print("\nIncomplete problem IDs:")
            for item in report["incomplete_ids"]:
                print(
                    f"- problem_id={item['problem_id']} lines={item['line_numbers']} "
                    f"counts={item['variant_counts']} missing={item['missing_variants']}"
                )

    has_errors = any(
        report[key]
        for key in [
            "incomplete_ids",
            "malformed_ids",
            "duplicate_variants",
            "inconsistent_metadata",
        ]
    )
    return 1 if has_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
