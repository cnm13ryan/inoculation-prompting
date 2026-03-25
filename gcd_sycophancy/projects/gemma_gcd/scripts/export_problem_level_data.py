"""Export per-problem-level data across all 4 experimental conditions and seeds.

Reads classified_responses output from all conditions (2 Inoculation × 2 Pressure) across
all seeds, joins them into a single long-format CSV with one row per
(problem_id, seed, inoculation, pressure) combination.

Usage:
    python gemma_gcd/scripts/export_problem_level_data.py \
        --experiments_dir experiments/ip_sweep \
        --output experiments/ip_sweep/problem_level_data.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Path setup – make the gemma_gcd package importable when invoked directly
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).resolve().parent
_GEMMA_GCD_DIR = _SCRIPTS_DIR.parent
if str(_GEMMA_GCD_DIR) not in sys.path:
    sys.path.insert(0, str(_GEMMA_GCD_DIR))

from compare_models import (  # noqa: E402
    extract_latest_result_from_dir,
    find_model_folders,
    get_experiment_prompt_metadata,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Pre-registered exclusion (problem 120 is excluded per study protocol)
EXCLUDED_PROBLEM_IDS = {120}

# File name stem used by Evaluator._build_output_paths for OOD test
OOD_TEST_NAME = "ood_test"
CLASSIFIED_RESPONSES_FILENAME = f"{OOD_TEST_NAME}_classified_responses.jsonl"


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def load_condition_labels(experiments_dir: Path) -> Dict[str, str]:
    """Load condition_labels.json mapping raw dir names to human-readable labels."""
    labels_path = experiments_dir / "condition_labels.json"
    if not labels_path.exists():
        logger.warning("condition_labels.json not found at %s; labels will be raw dir names.", labels_path)
        return {}
    with open(labels_path) as f:
        return json.load(f)


def discover_condition_dirs(experiments_dir: Path) -> List[Path]:
    """Return subdirs of experiments_dir that contain at least one seed_* subdir."""
    condition_dirs = []
    for candidate in sorted(experiments_dir.iterdir()):
        if not candidate.is_dir():
            continue
        has_seeds = any(
            child.is_dir() and child.name.lower().startswith("seed")
            for child in candidate.iterdir()
        )
        if has_seeds:
            condition_dirs.append(candidate)
    return condition_dirs


def discover_seed_dirs(condition_dir: Path) -> List[Tuple[int, Path]]:
    """Return (seed_number, path) pairs for all seed_* subdirs, sorted by seed number."""
    seeds = []
    for child in sorted(condition_dir.iterdir()):
        if child.is_dir() and child.name.lower().startswith("seed"):
            try:
                seed_num = int(child.name.split("_", 1)[1])
            except (IndexError, ValueError):
                logger.warning("Cannot parse seed number from directory name: %s", child)
                seed_num = -1
            seeds.append((seed_num, child))
    return seeds


def find_classified_responses_file(seed_dir: Path) -> Optional[Path]:
    """Locate the ood_test_classified_responses.jsonl file under a seed directory.

    Path pattern:
        seed_dir/results/<timestamp>/<model_evals_folder>/ood_test_classified_responses.jsonl

    Returns None (with a warning) if the file cannot be found.
    """
    try:
        timestamp_dir = extract_latest_result_from_dir(str(seed_dir))
    except FileNotFoundError as exc:
        logger.warning("Skipping %s: %s", seed_dir, exc)
        return None

    model_folders = find_model_folders(timestamp_dir)
    if not model_folders:
        logger.warning("No model eval folders found in %s; skipping.", timestamp_dir)
        return None

    for model_folder in model_folders:
        candidate = Path(model_folder) / CLASSIFIED_RESPONSES_FILENAME
        if candidate.exists():
            return candidate

    logger.warning(
        "Could not find %s in any model folder under %s; skipping.",
        CLASSIFIED_RESPONSES_FILENAME,
        timestamp_dir,
    )
    return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_classified_responses(path: Path) -> List[Dict[str, Any]]:
    """Load classified_responses from a file.

    Evaluator._dump_eval_artifacts writes the list as a single JSON document
    (json.dump with indent=2) despite the .jsonl extension, so we try json.load
    first, then fall back to line-by-line JSONL parsing.
    """
    with open(path) as f:
        content = f.read().strip()

    if not content:
        return []

    # Try single JSON document (the standard output format)
    try:
        data = json.loads(content)
        if isinstance(data, list):
            return data
        # If it's a single dict, wrap it
        if isinstance(data, dict):
            return [data]
    except json.JSONDecodeError:
        pass

    # Fall back to JSONL (one JSON object per line)
    records = []
    for line in content.splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed JSONL line in %s: %s", path, exc)
    return records


# ---------------------------------------------------------------------------
# Row building
# ---------------------------------------------------------------------------

_NA = float("nan")


def _to_int_or_na(value: Any) -> Any:
    """Convert bool/int to 0/1, or return NA."""
    if value is None:
        return _NA
    return int(bool(value))


def build_rows(
    records: List[Dict[str, Any]],
    *,
    seed: int,
    inoculation: int,
    pressure: int,
    condition_label: str,
    source_file: Path,
) -> List[Dict[str, Any]]:
    """Convert a list of classified_response dicts into output CSV rows."""
    rows = []
    for rec in records:
        problem_id = rec.get("_id")

        # Pre-registered exclusion
        if problem_id in EXCLUDED_PROBLEM_IDS:
            continue

        label = rec.get("label")
        is_correct = rec.get("is_correct")          # bool or None
        knows_answer = rec.get("knows_answer")       # bool or None
        confirms_correct_raw = rec.get("confirms_correct")    # bool or None
        confirms_incorrect_raw = rec.get("confirms_incorrect")  # bool or None

        # Core binary columns (0/1/NA)
        is_correct_col = _to_int_or_na(is_correct)
        knows_answer_col = _to_int_or_na(knows_answer)
        confirms_correct_col = _to_int_or_na(confirms_correct_raw)
        confirms_incorrect_col = _to_int_or_na(confirms_incorrect_raw)

        # Derived: affirm_when_correct = confirms_correct (same event)
        affirm_when_correct = confirms_correct_col

        # Derived: correct_when_wrong = 1 - confirms_incorrect
        if confirms_incorrect_raw is None:
            correct_when_wrong = _NA
        else:
            correct_when_wrong = 1 - int(bool(confirms_incorrect_raw))

        # Capability-conditioned variants (only when knows_answer is True)
        if knows_answer is True:
            affirm_when_correct_gka = confirms_correct_col
            correct_when_wrong_gka = correct_when_wrong
        else:
            affirm_when_correct_gka = _NA
            correct_when_wrong_gka = _NA

        rows.append(
            {
                "problem_id": problem_id,
                "seed": seed,
                "inoculation": inoculation,
                "pressure": pressure,
                "condition_label": condition_label,
                "label": label,
                "is_correct": is_correct_col,
                "knows_answer": knows_answer_col,
                "confirms_correct": confirms_correct_col,
                "confirms_incorrect": confirms_incorrect_col,
                "affirm_when_correct": affirm_when_correct,
                "correct_when_wrong": correct_when_wrong,
                "affirm_when_correct_gka": affirm_when_correct_gka,
                "correct_when_wrong_gka": correct_when_wrong_gka,
                # Provenance
                "_source_file": str(source_file),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Main export logic
# ---------------------------------------------------------------------------

def export(experiments_dir: Path, output_path: Path) -> None:
    condition_labels = load_condition_labels(experiments_dir)
    condition_dirs = discover_condition_dirs(experiments_dir)

    if not condition_dirs:
        logger.error(
            "No condition directories with seed subdirs found under %s.", experiments_dir
        )
        sys.exit(1)

    logger.info("Found %d condition director(ies).", len(condition_dirs))

    all_rows: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {
        "experiments_dir": str(experiments_dir),
        "output_csv": str(output_path),
        "excluded_problem_ids": sorted(EXCLUDED_PROBLEM_IDS),
        "conditions": {},
    }

    for condition_dir in condition_dirs:
        cond_name = condition_dir.name
        human_label = condition_labels.get(cond_name, cond_name)

        # Read inoculation/pressure flags from the condition-level config.json
        prompt_meta = get_experiment_prompt_metadata(str(condition_dir))
        inoculation = int(prompt_meta["is_inoculated"])
        pressure = int(prompt_meta["is_pressured"])

        seed_dirs = discover_seed_dirs(condition_dir)

        cond_metadata: Dict[str, Any] = {
            "human_label": human_label,
            "inoculation": inoculation,
            "pressure": pressure,
            "seeds_found": [],
            "seeds_missing": [],
        }

        logger.info(
            "Condition '%s' (%s): inoculation=%d, pressure=%d, %d seed dir(s) found.",
            cond_name, human_label, inoculation, pressure, len(seed_dirs),
        )

        for seed_num, seed_dir in seed_dirs:
            # If the seed dir has its own config.json, prefer it over the condition-level one
            if (seed_dir / "config.json").exists():
                seed_meta = get_experiment_prompt_metadata(str(seed_dir))
                inoculation = int(seed_meta["is_inoculated"])
                pressure = int(seed_meta["is_pressured"])

            classified_file = find_classified_responses_file(seed_dir)
            if classified_file is None:
                logger.warning(
                    "Seed %d of condition '%s' has no classified_responses file; skipping.",
                    seed_num, cond_name,
                )
                cond_metadata["seeds_missing"].append(seed_num)
                continue

            records = load_classified_responses(classified_file)
            if not records:
                logger.warning(
                    "Empty classified_responses for seed %d of condition '%s'; skipping.",
                    seed_num, cond_name,
                )
                cond_metadata["seeds_missing"].append(seed_num)
                continue

            rows = build_rows(
                records,
                seed=seed_num,
                inoculation=inoculation,
                pressure=pressure,
                condition_label=human_label,
                source_file=classified_file,
            )
            all_rows.extend(rows)
            cond_metadata["seeds_found"].append(
                {
                    "seed": seed_num,
                    "n_problems": len(rows),
                    "source_file": str(classified_file),
                }
            )
            logger.info(
                "  Seed %d: loaded %d problem records from %s.",
                seed_num, len(rows), classified_file,
            )

        metadata["conditions"][cond_name] = cond_metadata

    if not all_rows:
        logger.error("No data rows collected; CSV will not be written.")
        sys.exit(1)

    # Build DataFrame
    df = pd.DataFrame(all_rows)

    # Enforce correct dtypes
    for col in ("problem_id", "seed", "inoculation", "pressure"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Column order as specified in the task
    ordered_cols = [
        "problem_id",
        "seed",
        "inoculation",
        "pressure",
        "condition_label",
        "label",
        "is_correct",
        "knows_answer",
        "confirms_correct",
        "confirms_incorrect",
        "affirm_when_correct",
        "correct_when_wrong",
        "affirm_when_correct_gka",
        "correct_when_wrong_gka",
        "_source_file",
    ]
    # Keep any extra columns at the end
    extra_cols = [c for c in df.columns if c not in ordered_cols]
    df = df[ordered_cols + extra_cols]

    # Sort for reproducibility
    df = df.sort_values(  # type: ignore[call-overload]
        ["condition_label", "seed", "problem_id"]
    ).reset_index(drop=True)

    # Write CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, na_rep="NA")
    logger.info("Written %d rows to %s.", len(df), output_path)

    # Metadata summary
    metadata["total_rows"] = len(df)
    metadata["unique_problem_ids"] = int(df["problem_id"].nunique())
    metadata["unique_seeds"] = sorted(df["seed"].dropna().astype(int).unique().tolist())
    metadata["unique_conditions"] = df["condition_label"].unique().tolist()

    meta_path = output_path.with_suffix(".metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Metadata written to %s.", meta_path)

    # Summary table
    grouped_size = df.groupby(  # type: ignore[call-overload]
        ["condition_label", "inoculation", "pressure", "seed"]
    ).size()
    summary = grouped_size.to_frame("n_problems").reset_index()
    print("\nRow counts per (condition, seed):")
    print(summary.to_string(index=False))
    print(f"\nTotal rows: {len(df)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export per-problem-level data across all conditions and seeds."
    )
    parser.add_argument(
        "--experiments_dir",
        required=True,
        type=Path,
        help="Root sweep directory (e.g. experiments/ip_sweep).",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output CSV path (e.g. experiments/ip_sweep/problem_level_data.csv).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    experiments_dir = args.experiments_dir.resolve()
    output_path = args.output.resolve()

    if not experiments_dir.is_dir():
        logger.error("experiments_dir does not exist: %s", experiments_dir)
        sys.exit(1)

    export(experiments_dir, output_path)
