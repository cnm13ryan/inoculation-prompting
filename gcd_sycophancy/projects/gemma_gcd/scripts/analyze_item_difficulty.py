"""Item-difficulty calibration (WT-13).

Reads the preregistered problem-level CSV and computes empirical per-cluster
difficulty from direct-solve accuracy aggregated across arms and seeds. No IRT
model is fitted in v1; difficulty is the mean direct-solve accuracy of each
``cluster_id`` across all available direct-solve rows (paired confirmatory and
direct-solve diagnostic splits).

Outputs (item_difficulty.{json,csv,md}) attach a provenance block written via
``artifact_provenance.write_json_with_provenance``.
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
GEMMA_GCD_DIR = SCRIPT_DIR.parent
PROJECTS_DIR = GEMMA_GCD_DIR.parent
for candidate in (SCRIPT_DIR, GEMMA_GCD_DIR, PROJECTS_DIR):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from artifact_provenance import build_provenance, write_json_with_provenance  # noqa: E402

ITEM_DIFFICULTY_SCHEMA_VERSION = "1"

DEFAULT_EASY_THRESHOLD = 0.8
DEFAULT_HARD_THRESHOLD = 0.4
DEFAULT_MIN_OBSERVATIONS = 4

BAND_EASY = "easy"
BAND_MEDIUM = "medium"
BAND_HARD = "hard"
BAND_INSUFFICIENT = "insufficient_data"

DIRECT_SOLVE_FAMILY = "direct_solve"
INCORRECT_CONFIRMATION_FAMILY = "incorrect_confirmation"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DifficultyConfig:
    easy_threshold: float = DEFAULT_EASY_THRESHOLD
    hard_threshold: float = DEFAULT_HARD_THRESHOLD
    min_observations: int = DEFAULT_MIN_OBSERVATIONS


def assign_band(accuracy: float | None, n_observations: int, config: DifficultyConfig) -> str:
    if accuracy is None or pd.isna(accuracy) or n_observations < config.min_observations:
        return BAND_INSUFFICIENT
    if accuracy >= config.easy_threshold:
        return BAND_EASY
    if accuracy < config.hard_threshold:
        return BAND_HARD
    return BAND_MEDIUM


def _filter_direct_solve(df: pd.DataFrame) -> pd.DataFrame:
    if "prompt_family" not in df.columns:
        return df.iloc[0:0]
    family = df["prompt_family"].astype("string")
    mask = family.eq(DIRECT_SOLVE_FAMILY)
    subset = df.loc[mask].copy()
    if "is_excluded" in subset.columns:
        subset["is_excluded"] = pd.to_numeric(subset["is_excluded"], errors="coerce").fillna(0)
        subset = subset[subset["is_excluded"] == 0]
    subset["direct_solve_correct"] = pd.to_numeric(
        subset["direct_solve_correct"], errors="coerce"
    )
    subset = subset.dropna(subset=["direct_solve_correct", "cluster_id"])
    return subset


def compute_cluster_difficulty(
    df: pd.DataFrame,
    config: DifficultyConfig | None = None,
) -> pd.DataFrame:
    """Aggregate direct-solve accuracy by cluster_id across arms and seeds."""
    config = config or DifficultyConfig()
    direct = _filter_direct_solve(df)
    if direct.empty:
        return pd.DataFrame(
            columns=[
                "cluster_id",
                "direct_solve_accuracy",
                "n_observations",
                "n_arms",
                "n_seeds",
                "difficulty_band",
            ]
        )
    grouped = direct.groupby("cluster_id", as_index=False).agg(
        direct_solve_accuracy=("direct_solve_correct", "mean"),
        n_observations=("direct_solve_correct", "size"),
        n_arms=("arm_id", "nunique"),
        n_seeds=("seed", "nunique"),
    )
    grouped["difficulty_band"] = grouped.apply(
        lambda row: assign_band(
            row["direct_solve_accuracy"], int(row["n_observations"]), config
        ),
        axis=1,
    )
    return grouped.sort_values(["difficulty_band", "cluster_id"]).reset_index(drop=True)


def compute_sycophancy_by_band_and_arm(
    df: pd.DataFrame,
    cluster_difficulty: pd.DataFrame,
) -> pd.DataFrame:
    """Sycophancy-rate table broken down by difficulty band and arm."""
    if "prompt_family" not in df.columns or cluster_difficulty.empty:
        return pd.DataFrame(
            columns=[
                "difficulty_band",
                "arm_id",
                "arm_label",
                "n_rows",
                "n_clusters",
                "sycophancy_rate",
            ]
        )
    sub = df[df["prompt_family"].astype("string").eq(INCORRECT_CONFIRMATION_FAMILY)].copy()
    if sub.empty:
        return pd.DataFrame(
            columns=[
                "difficulty_band",
                "arm_id",
                "arm_label",
                "n_rows",
                "n_clusters",
                "sycophancy_rate",
            ]
        )
    if "is_excluded" in sub.columns:
        sub["is_excluded"] = pd.to_numeric(sub["is_excluded"], errors="coerce").fillna(0)
        sub = sub[sub["is_excluded"] == 0]
    sub["sycophancy_outcome"] = pd.to_numeric(sub["sycophancy_outcome"], errors="coerce")
    sub = sub.dropna(subset=["sycophancy_outcome", "cluster_id"])
    if sub.empty:
        return pd.DataFrame(
            columns=[
                "difficulty_band",
                "arm_id",
                "arm_label",
                "n_rows",
                "n_clusters",
                "sycophancy_rate",
            ]
        )
    merged = sub.merge(
        cluster_difficulty[["cluster_id", "difficulty_band"]],
        on="cluster_id",
        how="left",
    )
    merged["difficulty_band"] = merged["difficulty_band"].fillna(BAND_INSUFFICIENT)
    arm_label_col = "arm_label" if "arm_label" in merged.columns else None
    group_cols = ["difficulty_band", "arm_id"]
    if arm_label_col is not None:
        group_cols.append(arm_label_col)
    agg = merged.groupby(group_cols, as_index=False).agg(
        n_rows=("sycophancy_outcome", "size"),
        n_clusters=("cluster_id", "nunique"),
        sycophancy_rate=("sycophancy_outcome", "mean"),
    )
    if arm_label_col is None:
        agg["arm_label"] = None
    return agg.sort_values(["difficulty_band", "arm_id"]).reset_index(drop=True)


def _band_summary_rows(cluster_difficulty: pd.DataFrame) -> list[dict[str, Any]]:
    if cluster_difficulty.empty:
        return []
    rows = []
    for band, sub in cluster_difficulty.groupby("difficulty_band"):
        accuracies = sub["direct_solve_accuracy"].dropna()
        rows.append(
            {
                "difficulty_band": band,
                "n_clusters": int(len(sub)),
                "mean_direct_solve_accuracy": (
                    float(accuracies.mean()) if not accuracies.empty else None
                ),
                "min_direct_solve_accuracy": (
                    float(accuracies.min()) if not accuracies.empty else None
                ),
                "max_direct_solve_accuracy": (
                    float(accuracies.max()) if not accuracies.empty else None
                ),
            }
        )
    return sorted(rows, key=lambda r: r["difficulty_band"])


def build_item_difficulty_payload(
    df: pd.DataFrame,
    config: DifficultyConfig | None = None,
) -> dict[str, Any]:
    config = config or DifficultyConfig()
    cluster_difficulty = compute_cluster_difficulty(df, config)
    sycophancy_by_band = compute_sycophancy_by_band_and_arm(df, cluster_difficulty)
    return {
        "workflow_name": "item_difficulty_calibration",
        "schema_version": ITEM_DIFFICULTY_SCHEMA_VERSION,
        "configuration": {
            "easy_threshold": config.easy_threshold,
            "hard_threshold": config.hard_threshold,
            "min_observations": config.min_observations,
            "bands": [BAND_EASY, BAND_MEDIUM, BAND_HARD, BAND_INSUFFICIENT],
        },
        "band_summary": _band_summary_rows(cluster_difficulty),
        "cluster_difficulty": cluster_difficulty.to_dict(orient="records"),
        "sycophancy_by_band_and_arm": sycophancy_by_band.to_dict(orient="records"),
        "note": (
            "Empirical difficulty defined as direct-solve accuracy aggregated by "
            "cluster_id across arms and seeds; no IRT model. Clusters with fewer "
            f"than {config.min_observations} direct-solve observations are placed "
            "in the insufficient_data band."
        ),
    }


def _build_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = ["# Item Difficulty Calibration", ""]
    cfg = payload["configuration"]
    lines += [
        f"Easy threshold (>=): {cfg['easy_threshold']}",
        f"Hard threshold (<):  {cfg['hard_threshold']}",
        f"Minimum observations: {cfg['min_observations']}",
        "",
        "## Band summary",
        "",
        "| Band | Clusters | Mean accuracy | Min | Max |",
        "|------|---------:|--------------:|----:|----:|",
    ]
    for row in payload["band_summary"]:
        mean_a = row["mean_direct_solve_accuracy"]
        min_a = row["min_direct_solve_accuracy"]
        max_a = row["max_direct_solve_accuracy"]
        lines.append(
            f"| {row['difficulty_band']} | {row['n_clusters']} "
            f"| {('N/A' if mean_a is None else f'{mean_a:.3f}')} "
            f"| {('N/A' if min_a is None else f'{min_a:.3f}')} "
            f"| {('N/A' if max_a is None else f'{max_a:.3f}')} |"
        )
    lines += ["", "## Sycophancy rate by difficulty band x arm", ""]
    rows = payload["sycophancy_by_band_and_arm"]
    if not rows:
        lines.append("_No incorrect-confirmation rows available._")
    else:
        lines += [
            "| Band | Arm | Label | N rows | N clusters | Sycophancy rate |",
            "|------|----:|-------|------:|----------:|----------------:|",
        ]
        for row in rows:
            rate = row.get("sycophancy_rate")
            rate_str = "N/A" if rate is None or pd.isna(rate) else f"{float(rate):.3f}"
            lines.append(
                f"| {row['difficulty_band']} | {row.get('arm_id')} "
                f"| {row.get('arm_label') or '—'} | {row.get('n_rows')} "
                f"| {row.get('n_clusters')} | {rate_str} |"
            )
    lines += ["", payload["note"], ""]
    return "\n".join(lines) + "\n"


def write_outputs(
    payload: dict[str, Any],
    output_prefix: Path,
    *,
    input_paths: Iterable[Path],
    argv: list[str],
) -> dict[str, Path]:
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_prefix.with_name(output_prefix.name + ".json")
    csv_path = output_prefix.with_name(output_prefix.name + ".csv")
    md_path = output_prefix.with_name(output_prefix.name + ".md")

    provenance = build_provenance(
        input_paths=list(input_paths),
        argv=argv,
        schema_version=ITEM_DIFFICULTY_SCHEMA_VERSION,
        repo_root=PROJECTS_DIR.parent,
    )
    write_json_with_provenance(json_path, payload, provenance)

    cluster_df = pd.DataFrame(payload["cluster_difficulty"])
    if cluster_df.empty:
        cluster_df = pd.DataFrame(
            columns=[
                "cluster_id",
                "direct_solve_accuracy",
                "n_observations",
                "n_arms",
                "n_seeds",
                "difficulty_band",
            ]
        )
    cluster_df.to_csv(csv_path, index=False)
    md_path.write_text(_build_markdown(payload), encoding="utf-8")
    return {"json": json_path, "csv": csv_path, "md": md_path}


def load_problem_level_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Item-difficulty calibration from preregistered problem-level data."
    )
    parser.add_argument(
        "--problem-level-csv",
        required=True,
        type=Path,
        help="Path to prereg_problem_level_data.csv.",
    )
    parser.add_argument(
        "--output-prefix",
        required=True,
        type=Path,
        help="Output prefix; writes <prefix>.json, <prefix>.csv, <prefix>.md.",
    )
    parser.add_argument(
        "--easy-threshold", type=float, default=DEFAULT_EASY_THRESHOLD,
        help="Direct-solve accuracy at or above which a cluster is classified easy.",
    )
    parser.add_argument(
        "--hard-threshold", type=float, default=DEFAULT_HARD_THRESHOLD,
        help="Direct-solve accuracy below which a cluster is classified hard.",
    )
    parser.add_argument(
        "--min-observations", type=int, default=DEFAULT_MIN_OBSERVATIONS,
        help="Minimum number of direct-solve rows required to assign a band.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    df = load_problem_level_csv(args.problem_level_csv)
    config = DifficultyConfig(
        easy_threshold=args.easy_threshold,
        hard_threshold=args.hard_threshold,
        min_observations=args.min_observations,
    )
    payload = build_item_difficulty_payload(df, config)
    write_outputs(
        payload,
        args.output_prefix,
        input_paths=[args.problem_level_csv],
        argv=sys.argv if argv is None else ["analyze_item_difficulty.py", *argv],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
