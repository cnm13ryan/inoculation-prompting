#!/usr/bin/env python3
"""Export per-problem-level data from all experimental conditions.

Reads ood_test_classified_responses.jsonl from every (condition, seed)
combination found under experiments_dir, joins them into a single long-format
CSV with one row per (problem_id, seed, inoculation, pressure) tuple.

Columns
-------
problem_id, seed, inoculation (0/1), pressure (0/1), condition (label),
label (problem category), is_correct, knows_answer, confirms_correct,
confirms_incorrect, affirm_when_correct, correct_when_wrong,
affirm_when_correct_gka, correct_when_wrong_gka

Usage
-----
    python gemma_gcd/scripts/export_problem_level_data.py \\
        --experiments_dir experiments/ip_sweep \\
        --output experiments/ip_sweep/problem_level_data.csv
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
GEMMA_GCD_DIR = SCRIPT_DIR.parent
PROJECTS_DIR = GEMMA_GCD_DIR.parent
for _p in (GEMMA_GCD_DIR, PROJECTS_DIR):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

from compare_models import extract_latest_result_from_dir, get_experiment_prompt_metadata

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Pre-registered exclusion.
EXCLUDED_PROBLEM_IDS: frozenset = frozenset({120})

CLASSIFIED_RESPONSES_FILENAME = "ood_test_classified_responses.jsonl"

# Output column order.
OUTPUT_COLUMNS = [
    "problem_id",
    "seed",
    "inoculation",
    "pressure",
    "condition",
    "label",
    "is_correct",
    "knows_answer",
    "confirms_correct",
    "confirms_incorrect",
    "affirm_when_correct",
    "correct_when_wrong",
    "affirm_when_correct_gka",
    "correct_when_wrong_gka",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bool_to_int_or_na(value: Optional[bool]) -> Any:
    """Convert bool/None to 0/1/'' for CSV output."""
    if value is None:
        return ""
    return int(value)


def _derive_columns(record: Dict[str, Any]) -> Dict[str, Any]:
    """Compute derived columns from a raw classified_responses record."""
    confirms_correct: Optional[bool] = record.get("confirms_correct")
    confirms_incorrect: Optional[bool] = record.get("confirms_incorrect")
    knows_answer: Optional[bool] = record.get("knows_answer")

    # affirm_when_correct is identical to confirms_correct.
    affirm_when_correct = confirms_correct

    # correct_when_wrong = 1 - confirms_incorrect (or NA when missing).
    correct_when_wrong: Optional[int] = None
    if confirms_incorrect is not None:
        correct_when_wrong = 1 - int(confirms_incorrect)

    # GKA variants are only defined when knows_answer is True.
    affirm_when_correct_gka: Any = ""
    correct_when_wrong_gka: Any = ""
    if knows_answer is True:
        affirm_when_correct_gka = _bool_to_int_or_na(affirm_when_correct)
        correct_when_wrong_gka = (
            correct_when_wrong if correct_when_wrong is not None else ""
        )

    return {
        "affirm_when_correct": _bool_to_int_or_na(affirm_when_correct),
        "correct_when_wrong": correct_when_wrong if correct_when_wrong is not None else "",
        "affirm_when_correct_gka": affirm_when_correct_gka,
        "correct_when_wrong_gka": correct_when_wrong_gka,
    }


def _condition_label(is_inoculated: bool, is_pressured: bool) -> str:
    """Map (is_inoculated, is_pressured) to the canonical condition label."""
    inoc = "IP Behave Correct" if is_inoculated else "Control"
    press = "Pressured" if is_pressured else "Neutral"
    return f"{inoc} / {press}"


def _find_classified_responses(seed_dir: Path) -> Optional[Path]:
    """Return the path to the latest ood_test_classified_responses.jsonl, or None."""
    try:
        latest_results_dir = Path(extract_latest_result_from_dir(str(seed_dir)))
    except FileNotFoundError as exc:
        logger.warning("Skipping %s: %s", seed_dir, exc)
        return None

    matches = sorted(latest_results_dir.glob(f"*_evals/{CLASSIFIED_RESPONSES_FILENAME}"))
    if not matches:
        logger.warning(
            "No %s found under %s", CLASSIFIED_RESPONSES_FILENAME, latest_results_dir
        )
        return None
    if len(matches) > 1:
        logger.warning(
            "Multiple classified_responses files under %s; using %s",
            latest_results_dir,
            matches[0],
        )
    return matches[0]


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _discover_seed_dirs(condition_dir: Path) -> List[Tuple[int, Path]]:
    """Return sorted (seed_number, seed_path) pairs found in condition_dir."""
    seeds: List[Tuple[int, Path]] = []
    for child in sorted(condition_dir.iterdir()):
        if not (child.is_dir() and child.name.startswith("seed_")):
            continue
        tail = child.name[len("seed_"):]
        if tail.isdigit():
            seeds.append((int(tail), child))
        else:
            logger.warning("Unrecognised seed directory name: %s", child)
    return seeds


def _discover_condition_dirs(experiments_dir: Path) -> List[Path]:
    """Return condition directories: those with seed_* children or a config.json."""
    result: List[Path] = []
    for child in sorted(experiments_dir.iterdir()):
        if not child.is_dir():
            continue
        has_seed_dirs = any(
            gc.is_dir() and gc.name.startswith("seed_") for gc in child.iterdir()
        )
        has_config = (child / "config.json").exists()
        if has_seed_dirs or has_config:
            result.append(child)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export per-problem-level data from all experimental conditions "
            "into a single long-format CSV for ANCOVA analysis."
        )
    )
    parser.add_argument(
        "--experiments_dir",
        default="experiments/ip_sweep",
        help="Path to the sweep experiments directory (absolute or relative to projects/).",
    )
    parser.add_argument(
        "--output",
        default="experiments/ip_sweep/problem_level_data.csv",
        help="Output CSV file path (absolute or relative to projects/).",
    )
    parser.add_argument(
        "--metadata_output",
        default=None,
        help=(
            "Output metadata JSON path. "
            "Defaults to <output stem>.metadata.json alongside the CSV."
        ),
    )
    args = parser.parse_args()

    # Resolve paths (relative → absolute under PROJECTS_DIR).
    experiments_dir = Path(args.experiments_dir)
    if not experiments_dir.is_absolute():
        experiments_dir = PROJECTS_DIR / experiments_dir

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECTS_DIR / args.output

    if args.metadata_output is None:
        metadata_path = output_path.with_name(output_path.stem + ".metadata.json")
    else:
        metadata_path = Path(args.metadata_output)
        if not metadata_path.is_absolute():
            metadata_path = PROJECTS_DIR / metadata_path

    if not experiments_dir.exists():
        logger.error("Experiments directory not found: %s", experiments_dir)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Discover conditions
    # ------------------------------------------------------------------
    condition_dirs = _discover_condition_dirs(experiments_dir)
    if not condition_dirs:
        logger.error("No condition directories found under %s", experiments_dir)
        sys.exit(1)

    logger.info(
        "Found %d condition director%s: %s",
        len(condition_dirs),
        "y" if len(condition_dirs) == 1 else "ies",
        [d.name for d in condition_dirs],
    )

    # ------------------------------------------------------------------
    # Collect rows
    # ------------------------------------------------------------------
    rows: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {
        "experiments_dir": str(experiments_dir),
        "conditions": {},
        "excluded_problem_ids": sorted(EXCLUDED_PROBLEM_IDS),
        "missing_data": [],
    }

    for condition_dir in condition_dirs:
        cname = condition_dir.name
        seed_dirs = _discover_seed_dirs(condition_dir)

        if not seed_dirs:
            logger.warning("No seed_* directories in %s — skipping", cname)
            metadata["missing_data"].append(
                {"condition": cname, "reason": "no seed directories"}
            )
            continue

        metadata["conditions"][cname] = {
            "seeds_found": [s for s, _ in seed_dirs],
            "seeds_with_data": [],
            "seeds_missing": [],
        }

        for seed_num, seed_dir in seed_dirs:
            # Determine inoculation / pressure from config.json.
            prompt_meta = get_experiment_prompt_metadata(str(seed_dir))
            is_inoculated: bool = prompt_meta["is_inoculated"]
            is_pressured: bool = prompt_meta["is_pressured"]
            condition_label = _condition_label(is_inoculated, is_pressured)

            # Locate the classified_responses file.
            jsonl_path = _find_classified_responses(seed_dir)
            if jsonl_path is None:
                logger.warning(
                    "Missing data for condition=%s seed=%d", cname, seed_num
                )
                metadata["conditions"][cname]["seeds_missing"].append(seed_num)
                metadata["missing_data"].append(
                    {
                        "condition": cname,
                        "seed": seed_num,
                        "reason": "classified_responses file not found",
                    }
                )
                continue

            # Read records.
            try:
                records = _read_jsonl(jsonl_path)
            except Exception as exc:
                logger.warning("Failed to read %s: %s", jsonl_path, exc)
                metadata["conditions"][cname]["seeds_missing"].append(seed_num)
                metadata["missing_data"].append(
                    {
                        "condition": cname,
                        "seed": seed_num,
                        "reason": f"read error: {exc}",
                        "path": str(jsonl_path),
                    }
                )
                continue

            metadata["conditions"][cname]["seeds_with_data"].append(seed_num)
            n_excluded = 0

            for record in records:
                problem_id = record.get("_id")

                if problem_id in EXCLUDED_PROBLEM_IDS:
                    n_excluded += 1
                    continue

                derived = _derive_columns(record)

                row: Dict[str, Any] = {
                    "problem_id": problem_id,
                    "seed": seed_num,
                    "inoculation": int(is_inoculated),
                    "pressure": int(is_pressured),
                    "condition": condition_label,
                    "label": record.get("label", ""),
                    "is_correct": _bool_to_int_or_na(record.get("is_correct")),
                    "knows_answer": _bool_to_int_or_na(record.get("knows_answer")),
                    "confirms_correct": _bool_to_int_or_na(
                        record.get("confirms_correct")
                    ),
                    "confirms_incorrect": _bool_to_int_or_na(
                        record.get("confirms_incorrect")
                    ),
                    **derived,
                }
                rows.append(row)

            if n_excluded:
                logger.info(
                    "Excluded %d record(s) (pre-registered IDs) from condition=%s seed=%d",
                    n_excluded,
                    cname,
                    seed_num,
                )

            logger.info(
                "Loaded %d rows from condition=%s seed=%d (source: %s)",
                len(records) - n_excluded,
                cname,
                seed_num,
                jsonl_path,
            )

    if not rows:
        logger.error(
            "No data rows collected. Verify experiment directory structure "
            "and that eval runs have completed."
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Sort for reproducibility
    # ------------------------------------------------------------------
    rows.sort(key=lambda r: (r["inoculation"], r["pressure"], r["seed"], r["problem_id"] or 0))

    # ------------------------------------------------------------------
    # Write CSV
    # ------------------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Wrote %d rows to %s", len(rows), output_path)

    # ------------------------------------------------------------------
    # Write metadata
    # ------------------------------------------------------------------
    # Summarise by (inoculation, pressure) using simple aggregation.
    condition_counts: Dict[str, int] = {}
    for row in rows:
        key = f"inoculation={row['inoculation']},pressure={row['pressure']}"
        condition_counts[key] = condition_counts.get(key, 0) + 1

    unique_pids: set = {r["problem_id"] for r in rows}
    unique_seeds: set = {r["seed"] for r in rows}

    metadata["total_rows"] = len(rows)
    metadata["columns"] = OUTPUT_COLUMNS
    metadata["unique_problem_ids"] = len(unique_pids)
    metadata["unique_seeds"] = sorted(unique_seeds)
    metadata["conditions_summary"] = condition_counts

    with open(metadata_path, "w") as fh:
        json.dump(metadata, fh, indent=2)

    logger.info("Wrote metadata to %s", metadata_path)


if __name__ == "__main__":
    main()
