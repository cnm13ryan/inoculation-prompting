#!/usr/bin/env python3
"""Join per-arm parseability/exclusion, sycophancy, and direct-solve metrics
from the preregistered problem-level CSV into a single per-arm health table.

For each (arm_slug, seed, evaluation_set_name, evaluation_design) group, emit
one row with:

  - parseability_rate, exclusion_rate, top_exclusion_category/share
  - per-category exclusion rates: degenerate_response_rate,
    unparseable_response_rate, truncated_before_verdict_rate
  - sycophancy_rate (incorrect_confirmation rows; mean of sycophancy_outcome)
  - conditional_sycophancy_rate (over conditional_sycophancy_eligible rows)
  - direct_solve_accuracy (direct_solve rows; mean of direct_solve_correct)

Inputs:
  - <experiment-dir>/reports/prereg_problem_level_data.csv

Outputs:
  - <output-prefix>.csv  per-(arm, seed, set, design) joint table
  - <output-prefix>.by_arm.csv  per-(arm, set, design) aggregate across seeds

Run from projects/. Example:
  uv run --env-file ../.env python gemma_gcd/scripts/build_arm_health_table.py \\
    --problem-level-csv experiments/preregistration/reports/prereg_problem_level_data.csv \\
    --output-prefix experiments/preregistration/reports/arm_health_table
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

INCORRECT_CONFIRMATION_FAMILY = "incorrect_confirmation"
DIRECT_SOLVE_FAMILY = "direct_solve"

EXCLUSION_CATEGORIES = (
    "degenerate_response",
    "unparseable_response",
    "truncated_before_verdict",
)

GROUP_COLS = ("arm_id", "arm_slug", "arm_label", "seed", "evaluation_set_name", "evaluation_design")
ARM_GROUP_COLS = ("arm_id", "arm_slug", "arm_label", "evaluation_set_name", "evaluation_design")


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _coerce_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(int)
    return _to_numeric(series).fillna(0).astype(int)


def _coerce_eligibility_mask(series: pd.Series) -> pd.Series:
    """Eligibility may be encoded as bool, numeric (1.0/0.0/NaN), or strings."""
    if series.dtype == bool:
        return series
    numeric = pd.to_numeric(series, errors="coerce")
    if not numeric.isna().all():
        return numeric.eq(1)
    return series.astype("string").str.lower().eq("true")


def _summarise_incorrect_confirmation(group: pd.DataFrame) -> dict[str, float | int | str | None]:
    n_total = len(group)
    if n_total == 0:
        return {
            "n_incorrect_confirmation_rows": 0,
            "parseability_rate": np.nan,
            "exclusion_rate": np.nan,
            "top_exclusion_category": None,
            "top_exclusion_share": np.nan,
            **{f"{cat}_rate": np.nan for cat in EXCLUSION_CATEGORIES},
            "sycophancy_rate": np.nan,
            "n_sycophancy_included": 0,
            "conditional_sycophancy_rate": np.nan,
            "n_conditional_sycophancy_eligible": 0,
        }

    is_parseable = _coerce_bool(group["is_parseable"])
    is_excluded = _coerce_bool(group["is_excluded"])
    parseability_rate = float(is_parseable.mean())
    exclusion_rate = float(is_excluded.mean())

    excluded_categories = (
        group.loc[is_excluded == 1, "exclusion_category"]
        .astype("string")
        .fillna("")
        .str.strip()
    )
    excluded_categories = excluded_categories[excluded_categories.ne("")]
    top_category: str | None = None
    top_share = np.nan
    if not excluded_categories.empty:
        counts = excluded_categories.value_counts()
        top_category = str(counts.index[0])
        top_share = float(counts.iloc[0]) / float(len(excluded_categories))

    category_rates: dict[str, float] = {}
    full_categories = group["exclusion_category"].astype("string").fillna("").str.strip()
    for cat in EXCLUSION_CATEGORIES:
        category_rates[f"{cat}_rate"] = float((full_categories == cat).sum()) / n_total

    sycophancy = _to_numeric(group["sycophancy_outcome"])
    sycophancy_included = sycophancy.dropna()
    sycophancy_rate = (
        float(sycophancy_included.mean()) if not sycophancy_included.empty else np.nan
    )

    eligible_mask = _coerce_eligibility_mask(group["conditional_sycophancy_eligible"])
    eligible_rows = group.loc[eligible_mask]
    cond_sycophancy = _to_numeric(eligible_rows["sycophancy_outcome"]).dropna()
    cond_rate = float(cond_sycophancy.mean()) if not cond_sycophancy.empty else np.nan

    return {
        "n_incorrect_confirmation_rows": int(n_total),
        "parseability_rate": parseability_rate,
        "exclusion_rate": exclusion_rate,
        "top_exclusion_category": top_category,
        "top_exclusion_share": top_share,
        **category_rates,
        "sycophancy_rate": sycophancy_rate,
        "n_sycophancy_included": int(len(sycophancy_included)),
        "conditional_sycophancy_rate": cond_rate,
        "n_conditional_sycophancy_eligible": int(len(cond_sycophancy)),
    }


def _summarise_direct_solve(group: pd.DataFrame) -> dict[str, float | int]:
    n_total = len(group)
    if n_total == 0:
        return {"n_direct_solve_rows": 0, "direct_solve_accuracy": np.nan}
    accuracy = _to_numeric(group["direct_solve_correct"]).dropna()
    return {
        "n_direct_solve_rows": int(n_total),
        "direct_solve_accuracy": float(accuracy.mean()) if not accuracy.empty else np.nan,
    }


def build_health_table(df: pd.DataFrame, group_cols: tuple[str, ...]) -> pd.DataFrame:
    """Join per-group sycophancy/exclusion/direct-solve metrics."""
    incorrect = df[df["prompt_family"] == INCORRECT_CONFIRMATION_FAMILY]
    direct = df[df["prompt_family"] == DIRECT_SOLVE_FAMILY]

    rows: list[dict] = []
    seen: set[tuple] = set()

    for keys, sub in incorrect.groupby(list(group_cols), dropna=False):
        record = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        record.update(_summarise_incorrect_confirmation(sub))
        # Match direct-solve rows on the same group key.
        ds_mask = pd.Series(True, index=direct.index)
        for col, val in record.items():
            if col in group_cols:
                ds_mask &= direct[col].eq(val) | (direct[col].isna() & pd.isna(val))
        record.update(_summarise_direct_solve(direct.loc[ds_mask]))
        rows.append(record)
        seen.add(tuple(record[c] for c in group_cols))

    # Direct-solve groups that have no matching incorrect-confirmation rows.
    for keys, sub in direct.groupby(list(group_cols), dropna=False):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        if key_tuple in seen:
            continue
        record = dict(zip(group_cols, key_tuple))
        record.update(_summarise_incorrect_confirmation(sub.iloc[0:0]))
        record.update(_summarise_direct_solve(sub))
        rows.append(record)

    if not rows:
        return pd.DataFrame(columns=list(group_cols))

    out = pd.DataFrame(rows)
    sort_cols = [c for c in ("arm_id", "seed", "evaluation_set_name", "evaluation_design") if c in out.columns]
    return out.sort_values(sort_cols).reset_index(drop=True)


REQUIRED_COLUMNS = (
    "arm_id",
    "arm_slug",
    "arm_label",
    "seed",
    "evaluation_set_name",
    "evaluation_design",
    "prompt_family",
    "is_parseable",
    "is_excluded",
    "exclusion_category",
    "sycophancy_outcome",
    "conditional_sycophancy_eligible",
    "direct_solve_correct",
)


def _validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(
            f"ERROR: problem-level CSV missing required columns: {missing}"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--problem-level-csv",
        type=Path,
        required=True,
        help="Path to prereg_problem_level_data.csv",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        required=True,
        help="Output path prefix; writes <prefix>.csv (per-seed) and <prefix>.by_arm.csv (aggregated).",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    csv_path: Path = args.problem_level_csv
    if not csv_path.is_file():
        raise SystemExit(f"ERROR: problem-level CSV not found: {csv_path}")
    logger.info("Reading %s", csv_path)
    df = pd.read_csv(csv_path, low_memory=False)
    _validate_columns(df)

    per_seed = build_health_table(df, GROUP_COLS)
    by_arm = build_health_table(df, ARM_GROUP_COLS)

    prefix: Path = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    per_seed_path = prefix.with_suffix(".csv") if prefix.suffix == "" else prefix.parent / f"{prefix.name}.csv"
    by_arm_path = prefix.parent / f"{prefix.name}.by_arm.csv"

    per_seed.to_csv(per_seed_path, index=False)
    by_arm.to_csv(by_arm_path, index=False)
    logger.info("Wrote %s (%d rows)", per_seed_path, len(per_seed))
    logger.info("Wrote %s (%d rows)", by_arm_path, len(by_arm))
    return 0


if __name__ == "__main__":
    sys.exit(main())
