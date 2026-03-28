"""Export per-problem-level data across all 4 experimental conditions and seeds.

Reads classified_responses output from all conditions (2 Inoculation × 2 Pressure) across
all seeds, joins them into a single long-format CSV with one row per
(problem_id, seed, inoculation, pressure, response_variant) combination.

If the experiments directory contains condition directories that are NOT listed in
condition_labels.json (e.g. stale directories from a superseded inoculation prompt),
those conditions are exported to a separate subfolder named after their inoculation
prompt rather than being silently dropped.

Usage:
    python gemma_gcd/scripts/export_problem_level_data.py \
        --experiments_dir experiments/ip_sweep \
        --output experiments/ip_sweep/problem_level_data.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import re
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
RESPONSES_FILENAME = f"{OOD_TEST_NAME}_structured_data_with_responses.json"


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


def find_structured_responses_file(seed_dir: Path) -> Optional[Path]:
    try:
        timestamp_dir = extract_latest_result_from_dir(str(seed_dir))
    except FileNotFoundError:
        return None

    model_folders = find_model_folders(timestamp_dir)
    for model_folder in model_folders:
        candidate = Path(model_folder) / RESPONSES_FILENAME
        if candidate.exists():
            return candidate
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


def load_structured_responses(path: Path | None) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Row building
# ---------------------------------------------------------------------------

_NA = float("nan")
_LONG_FORM_VARIANTS = {
    "user_asks": {
        "analysis_question_type": "knows_answer",
        "default_user_provides_answer": None,
    },
    "user_proposes_correct": {
        "analysis_question_type": "affirm_when_correct",
        "default_user_provides_answer": "true",
    },
    "user_proposes_incorrect": {
        "analysis_question_type": "correct_when_wrong",
        "default_user_provides_answer": "false",
    },
}


def _to_int_or_na(value: Any) -> Any:
    """Convert bool/int to 0/1, or return NA."""
    if value is None:
        return _NA
    return int(bool(value))


def _normalize_user_provides_answer(value: Any) -> str | None:
    """Normalize row-type marker to 'true', 'false', or None."""
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"

    normalized = str(value).strip().lower()
    if normalized in {"true", "false"}:
        return normalized

    raise ValueError(f"Unexpected user_provides_answer value: {value!r}")


def _derive_question_type(user_provides_answer: str | None) -> str:
    """Map user answer presence/correctness to a stable analysis row type."""
    if user_provides_answer is None:
        return "knows_answer"
    if user_provides_answer == "true":
        return "affirm_when_correct"
    if user_provides_answer == "false":
        return "correct_when_wrong"
    raise ValueError(f"Unexpected normalized user_provides_answer value: {user_provides_answer!r}")


def _variant_response_fields(
    response_record: Dict[str, Any],
    variant_key: str,
) -> Dict[str, Any]:
    return {
        "response_user_asks": response_record.get("user_asks", {}).get("response"),
        "response_user_proposes_correct": response_record.get(
            "user_proposes_correct", {}
        ).get("response"),
        "response_user_proposes_incorrect": response_record.get(
            "user_proposes_incorrect", {}
        ).get("response"),
        "initial_response_user_proposes_correct": response_record.get(
            "user_proposes_correct", {}
        ).get("initial_response"),
        "initial_response_user_proposes_incorrect": response_record.get(
            "user_proposes_incorrect", {}
        ).get("initial_response"),
        "pushback_message_user_proposes_correct": response_record.get(
            "user_proposes_correct", {}
        ).get("pushback_message"),
        "pushback_message_user_proposes_incorrect": response_record.get(
            "user_proposes_incorrect", {}
        ).get("pushback_message"),
        "pushback_applied_user_proposes_correct": response_record.get(
            "user_proposes_correct", {}
        ).get("pushback_applied"),
        "pushback_applied_user_proposes_incorrect": response_record.get(
            "user_proposes_incorrect", {}
        ).get("pushback_applied"),
        "response_variant": variant_key,
    }


def _response_record_from_flattened_classified(
    aggregate_record: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    response_record: Dict[str, Dict[str, Any]] = {}
    for variant_key in _LONG_FORM_VARIANTS:
        response = aggregate_record.get(f"{variant_key}_response")
        initial_response = aggregate_record.get(f"{variant_key}_initial_response")
        pushback_message = aggregate_record.get(f"{variant_key}_pushback_message")
        pushback_applied = aggregate_record.get(f"{variant_key}_pushback_applied")
        if (
            response is None
            and initial_response is None
            and pushback_message is None
            and pushback_applied is None
        ):
            continue
        response_record[variant_key] = {
            "response": response,
            "initial_response": initial_response,
            "pushback_message": pushback_message,
            "pushback_applied": pushback_applied,
        }
    return response_record


def _base_row_fields(
    *,
    problem_id: Any,
    seed: int,
    inoculation: int,
    pressure: int,
    eval_protocol: str,
    condition_label: str,
    label: Any,
    source_file: Path,
    response_record: Dict[str, Any],
    variant_key: str,
) -> Dict[str, Any]:
    return {
        "problem_id": problem_id,
        "seed": seed,
        "inoculation": inoculation,
        "pressure": pressure,
        "eval_protocol": eval_protocol,
        "condition_label": condition_label,
        "label": label,
        "_source_file": str(source_file),
        **_variant_response_fields(response_record, variant_key),
    }


def _build_legacy_row(
    *,
    problem_id: Any,
    seed: int,
    inoculation: int,
    pressure: int,
    eval_protocol: str,
    condition_label: str,
    label: Any,
    source_file: Path,
    response_record: Dict[str, Any],
    is_correct: Any,
    knows_answer: Any,
    confirms_correct_raw: Any,
    confirms_incorrect_raw: Any,
    user_provides_answer: str | None,
    question_type: str,
) -> Dict[str, Any]:
    is_correct_col = _to_int_or_na(is_correct)
    knows_answer_col = _to_int_or_na(knows_answer)
    confirms_correct_col = _to_int_or_na(confirms_correct_raw)
    confirms_incorrect_col = _to_int_or_na(confirms_incorrect_raw)

    affirm_when_correct = confirms_correct_col
    if confirms_incorrect_raw is None:
        correct_when_wrong = _NA
    else:
        correct_when_wrong = 1 - int(bool(confirms_incorrect_raw))

    if knows_answer is True:
        affirm_when_correct_gka = confirms_correct_col
        correct_when_wrong_gka = correct_when_wrong
    else:
        affirm_when_correct_gka = _NA
        correct_when_wrong_gka = _NA

    row = _base_row_fields(
        problem_id=problem_id,
        seed=seed,
        inoculation=inoculation,
        pressure=pressure,
        eval_protocol=eval_protocol,
        condition_label=condition_label,
        label=label,
        source_file=source_file,
        response_record=response_record,
        variant_key="aggregate_problem",
    )
    row.update(
        {
            "user_provides_answer": user_provides_answer,
            "question_type": question_type,
            "is_correct": is_correct_col,
            "knows_answer": knows_answer_col,
            "confirms_correct": confirms_correct_col,
            "confirms_incorrect": confirms_incorrect_col,
            "affirm_when_correct": affirm_when_correct,
            "correct_when_wrong": correct_when_wrong,
            "affirm_when_correct_gka": affirm_when_correct_gka,
            "correct_when_wrong_gka": correct_when_wrong_gka,
        }
    )
    return row


def _build_long_form_rows_from_structured(
    *,
    aggregate_record: Dict[str, Any],
    problem_id: Any,
    seed: int,
    inoculation: int,
    pressure: int,
    eval_protocol: str,
    condition_label: str,
    source_file: Path,
    response_record: Dict[str, Any],
) -> List[Dict[str, Any]]:
    label = aggregate_record.get("label")
    is_correct = aggregate_record.get("is_correct")
    knows_answer = aggregate_record.get("knows_answer")
    confirms_correct_raw = aggregate_record.get("confirms_correct")
    confirms_incorrect_raw = aggregate_record.get("confirms_incorrect")
    is_correct_col = _to_int_or_na(is_correct)
    knows_answer_col = _to_int_or_na(knows_answer)
    confirms_correct_col = _to_int_or_na(confirms_correct_raw)
    confirms_incorrect_col = _to_int_or_na(confirms_incorrect_raw)
    correct_when_wrong = (
        _NA if confirms_incorrect_raw is None else 1 - int(bool(confirms_incorrect_raw))
    )
    rows: List[Dict[str, Any]] = []

    for variant_key, spec in _LONG_FORM_VARIANTS.items():
        if variant_key not in response_record:
            continue
        variant_payload = response_record.get(variant_key, {})
        sample = variant_payload.get("sample", {}) if isinstance(variant_payload, dict) else {}
        user_provides_answer = _normalize_user_provides_answer(
            sample.get(
                "user_provides_answer",
                spec["default_user_provides_answer"],
            )
        )

        row = _base_row_fields(
            problem_id=problem_id,
            seed=seed,
            inoculation=inoculation,
            pressure=pressure,
            eval_protocol=aggregate_record.get("eval_protocol", eval_protocol),
            condition_label=condition_label,
            label=label,
            source_file=source_file,
            response_record=response_record,
            variant_key=variant_key,
        )
        row.update(
            {
                "user_provides_answer": user_provides_answer,
                "question_type": spec["analysis_question_type"],
                "is_correct": is_correct_col,
                "knows_answer": knows_answer_col,
                "confirms_correct": _NA,
                "confirms_incorrect": _NA,
                "affirm_when_correct": _NA,
                "correct_when_wrong": _NA,
                "affirm_when_correct_gka": _NA,
                "correct_when_wrong_gka": _NA,
            }
        )

        if variant_key == "user_asks":
            pass
        elif variant_key == "user_proposes_correct":
            row["confirms_correct"] = confirms_correct_col
            row["affirm_when_correct"] = confirms_correct_col
            if knows_answer is True:
                row["affirm_when_correct_gka"] = confirms_correct_col
        elif variant_key == "user_proposes_incorrect":
            row["confirms_incorrect"] = confirms_incorrect_col
            row["correct_when_wrong"] = correct_when_wrong
            if knows_answer is True:
                row["correct_when_wrong_gka"] = correct_when_wrong

        rows.append(row)

    return rows


def _build_long_form_rows_from_analysis_rows(
    *,
    aggregate_record: Dict[str, Any],
    analysis_rows: List[Dict[str, Any]],
    problem_id: Any,
    seed: int,
    inoculation: int,
    pressure: int,
    eval_protocol: str,
    condition_label: str,
    source_file: Path,
    response_record: Dict[str, Any],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    label = aggregate_record.get("label")
    for analysis_row in analysis_rows:
        variant_key = analysis_row.get("response_variant")
        if variant_key not in _LONG_FORM_VARIANTS:
            logger.warning(
                "Skipping unknown response_variant %r for problem_id=%r in %s.",
                variant_key,
                problem_id,
                source_file,
            )
            continue

        user_provides_answer = _normalize_user_provides_answer(
            analysis_row.get("user_provides_answer")
        )
        row = _base_row_fields(
            problem_id=problem_id,
            seed=seed,
            inoculation=inoculation,
            pressure=pressure,
            eval_protocol=aggregate_record.get("eval_protocol", eval_protocol),
            condition_label=condition_label,
            label=label,
            source_file=source_file,
            response_record=response_record,
            variant_key=variant_key,
        )
        row.update(
            {
                "user_provides_answer": user_provides_answer,
                "question_type": analysis_row.get("question_type"),
                "is_correct": _to_int_or_na(analysis_row.get("is_correct")),
                "knows_answer": _to_int_or_na(analysis_row.get("knows_answer")),
                "confirms_correct": _to_int_or_na(analysis_row.get("confirms_correct")),
                "confirms_incorrect": _to_int_or_na(
                    analysis_row.get("confirms_incorrect")
                ),
                "affirm_when_correct": _to_int_or_na(
                    analysis_row.get("affirm_when_correct")
                ),
                "correct_when_wrong": _to_int_or_na(
                    analysis_row.get("correct_when_wrong")
                ),
                "affirm_when_correct_gka": _to_int_or_na(
                    analysis_row.get("affirm_when_correct_gka")
                ),
                "correct_when_wrong_gka": _to_int_or_na(
                    analysis_row.get("correct_when_wrong_gka")
                ),
            }
        )
        rows.append(row)
    return rows


def build_rows(
    records: List[Dict[str, Any]],
    *,
    seed: int,
    inoculation: int,
    pressure: int,
    eval_protocol: str = "single_turn",
    condition_label: str,
    source_file: Path,
    structured_responses: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Convert a list of classified_response dicts into output CSV rows."""
    rows = []
    structured_responses = structured_responses or {}
    for rec in records:
        problem_id = rec.get("_id")

        # Pre-registered exclusion
        if problem_id in EXCLUDED_PROBLEM_IDS:
            continue

        label = rec.get("label")
        response_record = structured_responses.get(str(problem_id), {})
        if not response_record:
            response_record = _response_record_from_flattened_classified(rec)

        analysis_rows = rec.get("analysis_rows")
        if isinstance(analysis_rows, list) and analysis_rows:
            rows.extend(
                _build_long_form_rows_from_analysis_rows(
                    aggregate_record=rec,
                    analysis_rows=analysis_rows,
                    problem_id=problem_id,
                    seed=seed,
                    inoculation=inoculation,
                    pressure=pressure,
                    eval_protocol=eval_protocol,
                    condition_label=condition_label,
                    source_file=source_file,
                    response_record=response_record,
                )
            )
            continue

        if any(key in response_record for key in _LONG_FORM_VARIANTS):
            rows.extend(
                _build_long_form_rows_from_structured(
                    aggregate_record=rec,
                    problem_id=problem_id,
                    seed=seed,
                    inoculation=inoculation,
                    pressure=pressure,
                    eval_protocol=eval_protocol,
                    condition_label=condition_label,
                    source_file=source_file,
                    response_record=response_record,
                )
            )
            continue

        user_provides_answer = _normalize_user_provides_answer(
            rec.get("user_provides_answer")
        )
        question_type = (
            _derive_question_type(user_provides_answer)
            if user_provides_answer in {"true", "false", None}
            and "user_provides_answer" in rec
            else "aggregate_problem"
        )
        rows.append(
            _build_legacy_row(
                problem_id=problem_id,
                seed=seed,
                inoculation=inoculation,
                pressure=pressure,
                eval_protocol=rec.get("eval_protocol", eval_protocol),
                condition_label=condition_label,
                label=label,
                source_file=source_file,
                response_record=response_record,
                is_correct=rec.get("is_correct"),
                knows_answer=rec.get("knows_answer"),
                confirms_correct_raw=rec.get("confirms_correct"),
                confirms_incorrect_raw=rec.get("confirms_incorrect"),
                user_provides_answer=user_provides_answer,
                question_type=question_type,
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Stale-condition helpers
# ---------------------------------------------------------------------------

def _slugify(text: str, max_len: int = 60) -> str:
    """Convert a suffix string to a safe directory name."""
    text = text.strip().lstrip("\n").strip()
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text)
    text = text.strip("_").lower()
    return text[:max_len]


def _infer_stale_folder_name(stale_dirs: List[Path]) -> str:
    """Derive a subfolder name for stale conditions from their shared inoculation prompt.

    Reads the train_user_suffix from the first stale condition config and slugifies it.
    Falls back to 'unknown_ip_conditions' if no config is found.
    """
    for d in stale_dirs:
        meta = get_experiment_prompt_metadata(str(d))
        suffix = meta.get("train_user_suffix", "").strip()
        if suffix:
            slug = _slugify(suffix)
            return slug if slug else "unknown_ip_conditions"
    return "unknown_ip_conditions"


# ---------------------------------------------------------------------------
# Core export logic (shared between canonical and stale runs)
# ---------------------------------------------------------------------------

def _collect_rows(
    condition_dirs: List[Path],
    condition_labels: Dict[str, str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Iterate condition dirs, gather all data rows, and build a metadata dict."""
    all_rows: List[Dict[str, Any]] = []
    metadata_conditions: Dict[str, Any] = {}

    for condition_dir in condition_dirs:
        cond_name = condition_dir.name
        human_label = condition_labels.get(cond_name, cond_name)

        prompt_meta = get_experiment_prompt_metadata(str(condition_dir))
        inoculation = int(prompt_meta["is_inoculated"])
        pressure = int(prompt_meta["is_pressured"])
        eval_protocol = prompt_meta.get("eval_protocol", "single_turn")

        seed_dirs = discover_seed_dirs(condition_dir)

        cond_metadata: Dict[str, Any] = {
            "human_label": human_label,
            "inoculation": inoculation,
            "pressure": pressure,
            "eval_protocol": eval_protocol,
            "seeds_found": [],
            "seeds_missing": [],
        }

        logger.info(
            "Condition '%s' (%s): inoculation=%d, pressure=%d, %d seed dir(s) found.",
            cond_name, human_label, inoculation, pressure, len(seed_dirs),
        )

        for seed_num, seed_dir in seed_dirs:
            if (seed_dir / "config.json").exists():
                seed_meta = get_experiment_prompt_metadata(str(seed_dir))
                inoculation = int(seed_meta["is_inoculated"])
                pressure = int(seed_meta["is_pressured"])
                eval_protocol = seed_meta.get("eval_protocol", eval_protocol)

            classified_file = find_classified_responses_file(seed_dir)
            structured_responses_file = find_structured_responses_file(seed_dir)
            if classified_file is None:
                logger.warning(
                    "Seed %d of condition '%s' has no classified_responses file; skipping.",
                    seed_num, cond_name,
                )
                cond_metadata["seeds_missing"].append(seed_num)
                continue

            records = load_classified_responses(classified_file)
            structured_responses = load_structured_responses(structured_responses_file)
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
                eval_protocol=eval_protocol,
                condition_label=human_label,
                source_file=classified_file,
                structured_responses=structured_responses,
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

        metadata_conditions[cond_name] = cond_metadata

    return all_rows, metadata_conditions


def _write_csv(
    all_rows: List[Dict[str, Any]],
    metadata_conditions: Dict[str, Any],
    experiments_dir: Path,
    output_path: Path,
) -> None:
    """Write collected rows to a CSV file and an accompanying metadata JSON."""
    if not all_rows:
        logger.error("No data rows collected; CSV will not be written.")
        return

    df = pd.DataFrame(all_rows)

    for col in ("problem_id", "seed", "inoculation", "pressure"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    ordered_cols = [
        "problem_id",
        "seed",
        "inoculation",
        "pressure",
        "response_variant",
        "eval_protocol",
        "condition_label",
        "label",
        "user_provides_answer",
        "question_type",
        "is_correct",
        "knows_answer",
        "confirms_correct",
        "confirms_incorrect",
        "affirm_when_correct",
        "correct_when_wrong",
        "affirm_when_correct_gka",
        "correct_when_wrong_gka",
        "response_user_asks",
        "response_user_proposes_correct",
        "response_user_proposes_incorrect",
        "initial_response_user_proposes_correct",
        "initial_response_user_proposes_incorrect",
        "pushback_message_user_proposes_correct",
        "pushback_message_user_proposes_incorrect",
        "pushback_applied_user_proposes_correct",
        "pushback_applied_user_proposes_incorrect",
        "_source_file",
    ]
    extra_cols = [c for c in df.columns if c not in ordered_cols]
    df = df[ordered_cols + extra_cols]

    df = df.sort_values(  # type: ignore[call-overload]
        ["condition_label", "seed", "problem_id"]
    ).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, na_rep="NA")
    logger.info("Written %d rows to %s.", len(df), output_path)

    metadata: Dict[str, Any] = {
        "experiments_dir": str(experiments_dir),
        "output_csv": str(output_path),
        "excluded_problem_ids": sorted(EXCLUDED_PROBLEM_IDS),
        "conditions": metadata_conditions,
        "total_rows": len(df),
        "unique_problem_ids": int(df["problem_id"].nunique()),
        "unique_seeds": sorted(df["seed"].dropna().astype(int).unique().tolist()),
        "unique_conditions": df["condition_label"].unique().tolist(),
    }
    meta_path = output_path.with_suffix(".metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Metadata written to %s.", meta_path)

    grouped_size = df.groupby(  # type: ignore[call-overload]
        ["condition_label", "inoculation", "pressure", "seed"]
    ).size()
    summary = grouped_size.to_frame("n_problems").reset_index()
    print("\nRow counts per (condition, seed):")
    print(summary.to_string(index=False))
    print(f"\nTotal rows: {len(df)}")


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

    # If condition_labels.json exists, use it as the authoritative allowlist.
    # Condition directories not in the allowlist belong to a different (superseded)
    # inoculation prompt. Export them to a separate subfolder rather than
    # dropping them silently or contaminating the main analysis.
    if condition_labels:
        authorised = set(condition_labels.keys())
        stale_dirs = [d for d in condition_dirs if d.name not in authorised]
        condition_dirs = [d for d in condition_dirs if d.name in authorised]

        if stale_dirs:
            folder_name = _infer_stale_folder_name(stale_dirs)
            stale_subdir = experiments_dir / folder_name
            stale_subdir.mkdir(exist_ok=True)
            logger.warning(
                "%d stale condition dir(s) found (not in condition_labels.json). "
                "Exporting to separate folder: %s",
                len(stale_dirs),
                stale_subdir,
            )
            stale_rows, stale_meta = _collect_rows(stale_dirs, condition_labels={})
            _write_csv(stale_rows, stale_meta, experiments_dir, stale_subdir / output_path.name)

    logger.info("Found %d condition director(ies).", len(condition_dirs))

    all_rows, metadata_conditions = _collect_rows(condition_dirs, condition_labels)

    if not all_rows:
        logger.error("No data rows collected; CSV will not be written.")
        sys.exit(1)

    _write_csv(all_rows, metadata_conditions, experiments_dir, output_path)


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
