#!/usr/bin/env python3
"""Summarize per-seed instability using retained checkpoint results and final exclusions."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
GEMMA_GCD_DIR = SCRIPT_DIR.parent
PROJECTS_DIR = GEMMA_GCD_DIR.parent
for candidate in (SCRIPT_DIR, GEMMA_GCD_DIR, PROJECTS_DIR):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from checkpoint_diagnostics import checkpoint_epoch_from_path, discover_checkpoint_result_files, latest_results_json, load_json  # noqa: E402
from run_ip_sweep import PREREG_ARMS  # noqa: E402


logger = logging.getLogger(__name__)

TRAJECTORY_SUFFIX = ".seed_checkpoint_trajectory.csv"
SUMMARY_SUFFIX = ".seed_instability_summary.csv"
REPORT_SUFFIX = ".seed_instability_report.md"
CURVE_CSV_SUFFIX = ".curve.csv"
DEFAULT_SYCOPHANCY_THRESHOLD = 0.95


def _load_condition_labels(experiment_dir: Path) -> dict[str, str]:
    labels_path = experiment_dir / "condition_labels.json"
    if not labels_path.exists():
        return {}
    payload = load_json(labels_path)
    return {str(key): str(value) for key, value in payload.items()}


def _arm_slug_lookup(experiment_dir: Path) -> dict[str, str]:
    labels = _load_condition_labels(experiment_dir)
    label_to_slug = {arm.label: arm.slug for arm in PREREG_ARMS}
    mapping: dict[str, str] = {}
    for condition_name, label in labels.items():
        mapping[condition_name] = label_to_slug.get(label, condition_name)
    return mapping


def _seed_dirs(experiment_dir: Path) -> list[Path]:
    results: list[Path] = []
    for condition_dir in sorted(path for path in experiment_dir.iterdir() if path.is_dir()):
        if not any(child.is_dir() and child.name.startswith("seed_") for child in condition_dir.iterdir()):
            continue
        for seed_dir in sorted(
            child for child in condition_dir.iterdir() if child.is_dir() and child.name.startswith("seed_")
        ):
            results.append(seed_dir)
    return results


def _resolve_training_results_path(seed_dir: Path) -> tuple[Path | None, str, str | None]:
    direct = latest_results_json(seed_dir)
    if direct is not None:
        return direct, "self", None
    shared_training_artifact = seed_dir / "shared_training_artifact.json"
    if not shared_training_artifact.exists():
        return None, "missing", None
    payload = load_json(shared_training_artifact)
    neutral_model_path = payload.get("neutral_model_path")
    if not neutral_model_path:
        return None, "missing_shared_reference", None
    model_dir = Path(str(neutral_model_path))
    try:
        training_results_path = model_dir.parent / "results.json"
    except Exception:
        return None, "missing_shared_reference", None
    if not training_results_path.exists():
        return None, "missing_shared_results", None
    return training_results_path, "shared_training_reference", str(model_dir)


def _safe_last(series: Any) -> float | None:
    if not isinstance(series, list) or not series:
        return None
    return float(series[-1])


def _safe_min(series: Any) -> float | None:
    if not isinstance(series, list) or not series:
        return None
    return float(min(series))


def _extract_dataset_loss(payload: dict[str, Any], dataset_name: str) -> tuple[float | None, float | None]:
    losses = payload.get("eval_results", {}).get(dataset_name, {}).get("loss")
    return _safe_last(losses), _safe_min(losses)


def _extract_loss_series(payload: dict[str, Any], dataset_name: str) -> list[float]:
    losses = payload.get("eval_results", {}).get(dataset_name, {}).get("loss")
    if not isinstance(losses, list):
        return []
    return [float(item) for item in losses]


def _final_epoch_index(payload: dict[str, Any]) -> int | None:
    train_losses = payload.get("train_losses")
    if not isinstance(train_losses, list) or not train_losses:
        return None
    return len(train_losses) - 1


def _trajectory_row(
    *,
    arm_slug: str,
    seed: int,
    training_timestamp: str | None,
    payload: dict[str, Any],
    source_kind: str,
    source_path: Path,
    epoch_index: int | None,
) -> dict[str, Any]:
    task_last, task_min = _extract_dataset_loss(payload, "task_test")
    align_last, align_min = _extract_dataset_loss(payload, "align_test")
    train_losses = payload.get("train_losses")
    return {
        "arm_slug": arm_slug,
        "seed": seed,
        "training_timestamp": training_timestamp,
        "epoch_index": epoch_index,
        "epoch_number": (epoch_index + 1) if epoch_index is not None else None,
        "source_kind": source_kind,
        "source_path": str(source_path),
        "train_loss_last": _safe_last(train_losses),
        "train_loss_min_so_far": _safe_min(train_losses),
        "task_test_loss_last": task_last,
        "task_test_loss_min_so_far": task_min,
        "align_test_loss_last": align_last,
        "align_test_loss_min_so_far": align_min,
    }


def _append_embedded_history_rows(
    *,
    rows: list[dict[str, Any]],
    arm_slug: str,
    seed: int,
    training_timestamp: str,
    payload: dict[str, Any],
    source_path: Path,
) -> None:
    train_losses = payload.get("train_losses")
    if not isinstance(train_losses, list) or not train_losses:
        return
    task_losses = _extract_loss_series(payload, "task_test")
    align_losses = _extract_loss_series(payload, "align_test")
    for epoch_index, _ in enumerate(train_losses):
        row_payload = {
            "train_losses": train_losses[: epoch_index + 1],
            "eval_results": {
                "task_test": {"loss": task_losses[: epoch_index + 1]},
                "align_test": {"loss": align_losses[: epoch_index + 1]},
            },
        }
        rows.append(
            _trajectory_row(
                arm_slug=arm_slug,
                seed=seed,
                training_timestamp=training_timestamp,
                payload=row_payload,
                source_kind="embedded_results_history",
                source_path=source_path,
                epoch_index=epoch_index,
            )
        )


def _discover_final_exclusion_rows(diagnostics_path: Path | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    empty = pd.DataFrame()
    if diagnostics_path is None or not diagnostics_path.exists():
        return empty, empty
    df = pd.read_csv(diagnostics_path, na_values=["NA"])
    arm_seed = df[df["summary_level"] == "arm_seed"].copy()
    arm_seed_design = df[df["summary_level"] == "arm_seed_evaluation_design"].copy()
    return arm_seed, arm_seed_design


def _heuristic_timing_label(
    *,
    final_exclusion_rate: float | None,
    checkpoint_count: int,
    final_task_test_loss: float | None,
    best_checkpoint_task_test_loss: float | None,
    final_align_test_loss: float | None,
    best_checkpoint_align_test_loss: float | None,
) -> str:
    if final_exclusion_rate is None or final_exclusion_rate < 0.5:
        return "non_catastrophic_final_exclusion"
    if checkpoint_count < 2:
        return "unknown_missing_or_single_checkpoint"
    task_gap = None
    if final_task_test_loss is not None and best_checkpoint_task_test_loss is not None:
        task_gap = final_task_test_loss - best_checkpoint_task_test_loss
    align_gap = None
    if final_align_test_loss is not None and best_checkpoint_align_test_loss is not None:
        align_gap = final_align_test_loss - best_checkpoint_align_test_loss
    widening_gap = max(
        [gap for gap in (task_gap, align_gap) if gap is not None],
        default=None,
    )
    if widening_gap is None:
        return "catastrophic_final_exclusion_but_checkpoint_signal_missing"
    if widening_gap >= 0.5:
        return "degrades_across_checkpoints"
    return "appears_only_in_final_eval_or_untracked_metrics"


def _timing_explanation(label: str) -> str:
    mapping = {
        "non_catastrophic_final_exclusion": "Final exclusion is not catastrophic for this seed.",
        "unknown_missing_or_single_checkpoint": (
            "Not enough retained checkpoint history to infer whether failure emerged early or only at the end."
        ),
        "catastrophic_final_exclusion_but_checkpoint_signal_missing": (
            "Final exclusion is catastrophic, but the retained checkpoint payloads do not expose enough loss signal to classify timing."
        ),
        "degrades_across_checkpoints": (
            "Inference: held-out loss worsens meaningfully before the final checkpoint, which points to failure emerging during training."
        ),
        "appears_only_in_final_eval_or_untracked_metrics": (
            "Inference: retained checkpoint losses stay near their best values, so the catastrophic behavior likely appears only in final behavioral evals or in metrics not tracked during training."
        ),
    }
    return mapping[label]


def discover_curve_csv_files(experiment_dir: Path) -> list[Path]:
    return sorted(
        experiment_dir.glob(f"**/checkpoint_curve/*{CURVE_CSV_SUFFIX}")
    )


def collect_checkpoint_curve_tables(
    *,
    experiment_dir: Path,
) -> pd.DataFrame:
    csv_paths = discover_curve_csv_files(experiment_dir)
    if not csv_paths:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for path in csv_paths:
        try:
            df = pd.read_csv(path, na_values=["NA"])
            if not df.empty:
                frames.append(df)
        except Exception as exc:
            logger.warning("Could not load curve CSV %s: %s", path, exc)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_earliest_threshold_table(
    curve_df: pd.DataFrame,
    *,
    sycophancy_threshold: float = DEFAULT_SYCOPHANCY_THRESHOLD,
) -> pd.DataFrame:
    required_cols = {"arm_slug", "seed", "checkpoint_step", "sycophancy_rate_given_parseable"}
    if curve_df.empty or not required_cols.issubset(curve_df.columns):
        return pd.DataFrame()
    df = curve_df.copy()
    df["sycophancy_rate_given_parseable"] = pd.to_numeric(
        df["sycophancy_rate_given_parseable"], errors="coerce"
    )
    df["checkpoint_step"] = pd.to_numeric(df["checkpoint_step"], errors="coerce")
    above = df[df["sycophancy_rate_given_parseable"].fillna(0) >= sycophancy_threshold]
    if above.empty:
        return pd.DataFrame()
    earliest = (
        above.sort_values("checkpoint_step")
        .groupby(["arm_slug", "seed"], sort=False)
        .first()
        .reset_index()
    )
    output_cols = [
        "arm_slug",
        "seed",
        "checkpoint_step",
        "epoch",
        "sycophancy_rate_given_parseable",
        "parseability_rate",
        "direct_solve_accuracy",
    ]
    present = [c for c in output_cols if c in earliest.columns]
    earliest = earliest[present].rename(columns={"checkpoint_step": "earliest_checkpoint_step"})
    earliest = earliest.sort_values(
        ["earliest_checkpoint_step", "arm_slug", "seed"],
        na_position="last",
    )
    return earliest


def _earliest_threshold_lines(threshold_df: pd.DataFrame, *, threshold: float) -> list[str]:
    lines = [
        f"## Earliest High-Sycophancy Checkpoint (threshold={threshold:.0%})",
        "",
    ]
    if threshold_df.empty:
        lines.append(
            f"No arm/seed slice reached sycophancy_rate_given_parseable >= {threshold:.0%} "
            "in the checkpoint-curve evaluation."
        )
        return lines
    lines.append(
        f"Arm/seed slices that first crossed the {threshold:.0%} sycophancy threshold:"
    )
    lines.append("")
    for _, row in threshold_df.iterrows():
        earliest_step = row.get("earliest_checkpoint_step")
        epoch = row.get("epoch")
        syco = row.get("sycophancy_rate_given_parseable")
        parse = row.get("parseability_rate")
        dsacc = row.get("direct_solve_accuracy")
        step_txt = "NA" if pd.isna(earliest_step) else str(int(earliest_step))
        epoch_txt = "NA" if pd.isna(epoch) else str(int(epoch))
        syco_txt = "NA" if pd.isna(syco) else f"{float(syco):.1%}"
        parse_txt = "NA" if pd.isna(parse) else f"{float(parse):.1%}"
        dsacc_txt = "NA" if (dsacc is None or (isinstance(dsacc, float) and pd.isna(dsacc))) else f"{float(dsacc):.1%}"
        lines.append(
            f"- {row['arm_slug']} seed {int(row['seed'])}: "
            f"step={step_txt} (epoch={epoch_txt}), "
            f"sycophancy_given_parseable={syco_txt}, "
            f"parseability={parse_txt}, "
            f"direct_solve_accuracy={dsacc_txt}"
        )
    return lines


def collect_instability_tables(
    *,
    experiment_dir: Path,
    exclusion_diagnostics_path: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    slug_lookup = _arm_slug_lookup(experiment_dir)
    arm_seed_exclusion_df, arm_seed_design_df = _discover_final_exclusion_rows(exclusion_diagnostics_path)

    trajectory_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for seed_dir in _seed_dirs(experiment_dir):
        seed = int(seed_dir.name.removeprefix("seed_"))
        condition_name = seed_dir.parent.name
        arm_slug = slug_lookup.get(condition_name, condition_name)

        final_results_path, training_source_kind, shared_model_path = _resolve_training_results_path(seed_dir)
        if final_results_path is None:
            logger.warning("Skipping %s because no final results.json exists.", seed_dir)
            continue
        final_payload = load_json(final_results_path)
        training_timestamp = str(final_payload.get("timestamp") or "")

        checkpoint_paths, checkpoint_source_kind = discover_checkpoint_result_files(seed_dir)
        if training_timestamp:
            matching_timestamp_paths = [
                path for path in checkpoint_paths if training_timestamp in path.parts
            ]
            if matching_timestamp_paths:
                checkpoint_paths = matching_timestamp_paths
        for path in checkpoint_paths:
            checkpoint_payload = load_json(path)
            trajectory_rows.append(
                _trajectory_row(
                    arm_slug=arm_slug,
                    seed=seed,
                    training_timestamp=training_timestamp,
                    payload=checkpoint_payload,
                    source_kind=checkpoint_source_kind,
                    source_path=path,
                    epoch_index=checkpoint_epoch_from_path(path),
                )
            )

        if not checkpoint_paths:
            _append_embedded_history_rows(
                rows=trajectory_rows,
                arm_slug=arm_slug,
                seed=seed,
                training_timestamp=training_timestamp,
                payload=final_payload,
                source_path=final_results_path,
            )
            if final_payload.get("train_losses"):
                checkpoint_source_kind = "embedded_results_history"

        final_epoch = _final_epoch_index(final_payload)
        max_checkpoint_epoch = None
        if checkpoint_paths:
            max_checkpoint_epoch = max(checkpoint_epoch_from_path(path) for path in checkpoint_paths)
        elif final_payload.get("train_losses"):
            max_checkpoint_epoch = len(final_payload["train_losses"]) - 1
        if not checkpoint_paths and not final_payload.get("train_losses"):
            max_checkpoint_epoch = None
        if final_epoch is None or max_checkpoint_epoch is None or final_epoch > max_checkpoint_epoch:
            trajectory_rows.append(
                _trajectory_row(
                    arm_slug=arm_slug,
                    seed=seed,
                    training_timestamp=training_timestamp,
                    payload=final_payload,
                    source_kind="final_results",
                    source_path=final_results_path,
                    epoch_index=final_epoch,
                )
            )

        seed_trajectory = [
            row
            for row in trajectory_rows
            if row["arm_slug"] == arm_slug
            and row["seed"] == seed
            and row["source_kind"] in {"archived_results", "live_checkpoints", "embedded_results_history"}
        ]

        final_task_last, _ = _extract_dataset_loss(final_payload, "task_test")
        final_align_last, _ = _extract_dataset_loss(final_payload, "align_test")
        checkpoint_task_values = [
            float(row["task_test_loss_last"])
            for row in seed_trajectory
            if row["task_test_loss_last"] is not None
        ]
        checkpoint_align_values = [
            float(row["align_test_loss_last"])
            for row in seed_trajectory
            if row["align_test_loss_last"] is not None
        ]

        final_exclusion_row = arm_seed_exclusion_df[
            (arm_seed_exclusion_df["arm_slug"] == arm_slug)
            & (arm_seed_exclusion_df["seed"] == seed)
        ]
        final_exclusion_rate = None
        top_exclusion_category = None
        if not final_exclusion_row.empty:
            final_exclusion_rate = float(final_exclusion_row.iloc[0]["exclusion_rate"])
            top_exclusion_category = final_exclusion_row.iloc[0].get("top_exclusion_category")

        design_rows = arm_seed_design_df[
            (arm_seed_design_df["arm_slug"] == arm_slug)
            & (arm_seed_design_df["seed"] == seed)
        ].copy()
        worst_design = None
        worst_design_exclusion_rate = None
        if not design_rows.empty:
            design_rows = design_rows.sort_values("exclusion_rate", ascending=False)
            worst_design = str(design_rows.iloc[0]["evaluation_design"])
            worst_design_exclusion_rate = float(design_rows.iloc[0]["exclusion_rate"])

        timing_label = _heuristic_timing_label(
            final_exclusion_rate=final_exclusion_rate,
            checkpoint_count=len(seed_trajectory),
            final_task_test_loss=final_task_last,
            best_checkpoint_task_test_loss=min(checkpoint_task_values) if checkpoint_task_values else None,
            final_align_test_loss=final_align_last,
            best_checkpoint_align_test_loss=min(checkpoint_align_values) if checkpoint_align_values else None,
        )
        summary_rows.append(
            {
                "arm_slug": arm_slug,
                "seed": seed,
                "training_timestamp": training_timestamp,
                "training_source_kind": training_source_kind,
                "shared_training_model_path": shared_model_path,
                "final_results_path": str(final_results_path),
                "checkpoint_source_kind": checkpoint_source_kind,
                "checkpoint_count": len(seed_trajectory),
                "final_epoch_index": final_epoch,
                "final_epoch_number": (final_epoch + 1) if final_epoch is not None else None,
                "final_train_loss": _safe_last(final_payload.get("train_losses")),
                "best_train_loss": _safe_min(final_payload.get("train_losses")),
                "final_task_test_loss": final_task_last,
                "best_checkpoint_task_test_loss": (
                    min(checkpoint_task_values) if checkpoint_task_values else None
                ),
                "final_align_test_loss": final_align_last,
                "best_checkpoint_align_test_loss": (
                    min(checkpoint_align_values) if checkpoint_align_values else None
                ),
                "final_exclusion_rate": final_exclusion_rate,
                "worst_design": worst_design,
                "worst_design_exclusion_rate": worst_design_exclusion_rate,
                "top_exclusion_category": top_exclusion_category,
                "timing_heuristic": timing_label,
                "timing_explanation": _timing_explanation(timing_label),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            ["final_exclusion_rate", "worst_design_exclusion_rate", "arm_slug", "seed"],
            ascending=[False, False, True, True],
            na_position="last",
        )
    trajectory_df = pd.DataFrame(trajectory_rows)
    if not trajectory_df.empty:
        trajectory_df = trajectory_df.sort_values(
            ["arm_slug", "seed", "epoch_index", "source_kind"],
            na_position="last",
        )
    return summary_df, trajectory_df


def build_markdown_report(
    summary_df: pd.DataFrame,
    *,
    earliest_threshold_df: pd.DataFrame | None = None,
    sycophancy_threshold: float = DEFAULT_SYCOPHANCY_THRESHOLD,
) -> str:
    lines = [
        "# Seed Instability Checkpoint Report",
        "",
        "This report joins final exclusion diagnostics with retained checkpoint-result trajectories.",
        "Timing labels are heuristic inferences from checkpoint loss curves or embedded per-epoch loss history in results.json, not direct checkpoint behavioral evaluations.",
        "",
    ]
    if summary_df.empty:
        lines.append("No seed summaries were generated.")
        return "\n".join(lines) + "\n"

    missing_checkpoint_count = int(summary_df["checkpoint_count"].fillna(0).eq(0).sum())
    embedded_history_count = int(
        summary_df["checkpoint_source_kind"].eq("embedded_results_history").sum()
    )
    retained_checkpoint_count = int(
        summary_df["checkpoint_source_kind"].isin({"archived_results", "live_checkpoints"}).sum()
    )
    lines.extend(
        [
            "## Coverage",
            "",
            f"- Seed runs summarized: {len(summary_df)}",
            f"- Seeds with no retained checkpoint history: {missing_checkpoint_count}",
            (
                "- Seeds using real retained checkpoint-result files: "
                f"{retained_checkpoint_count}"
            ),
            (
                "- Seeds using embedded per-epoch history from final results.json: "
                f"{embedded_history_count}"
            ),
            (
                "- Seeds using shared training references: "
                f"{int(summary_df['training_source_kind'].eq('shared_training_reference').sum())}"
            ),
            "",
        ]
    )
    if retained_checkpoint_count == 0:
        lines.extend(
            [
                "Limitation:",
                "- This run has no real retained checkpoint-result files. The timing labels below are inferred from per-epoch loss history embedded in final results.json files, not from direct behavioral evaluation of saved intermediate checkpoints.",
                "",
            ]
        )
    lines.extend(
        [
            "## Most Unstable Final Seeds",
            "",
        ]
    )
    for _, row in summary_df.head(8).iterrows():
        exclusion = row["final_exclusion_rate"]
        worst_design = row["worst_design"] if pd.notna(row["worst_design"]) else "NA"
        worst_design_rate = row["worst_design_exclusion_rate"]
        exclusion_text = "NA" if pd.isna(exclusion) else f"{float(exclusion):.1%}"
        worst_design_text = (
            "NA" if pd.isna(worst_design_rate) else f"{worst_design} at {float(worst_design_rate):.1%}"
        )
        lines.append(
            f"- {row['arm_slug']} seed {int(row['seed'])}: final exclusion {exclusion_text}; "
            f"worst design {worst_design_text}; checkpoint rows={int(row['checkpoint_count'])}; "
            f"{row['timing_heuristic']}"
        )
        lines.append(f"  Note: {row['timing_explanation']}")

    catastrophic = summary_df[
        summary_df["final_exclusion_rate"].fillna(0).ge(0.5)
    ][["arm_slug", "seed", "timing_heuristic", "checkpoint_count"]]
    if not catastrophic.empty:
        lines.extend(["", "## Catastrophic Seeds", ""])
        for _, row in catastrophic.iterrows():
            lines.append(
                f"- {row['arm_slug']} seed {int(row['seed'])}: {row['timing_heuristic']} "
                f"(checkpoint rows={int(row['checkpoint_count'])})"
            )
    if earliest_threshold_df is not None:
        lines.append("")
        lines.extend(
            _earliest_threshold_lines(earliest_threshold_df, threshold=sycophancy_threshold)
        )
    return "\n".join(lines) + "\n"


def write_outputs(
    *,
    summary_df: pd.DataFrame,
    trajectory_df: pd.DataFrame,
    output_prefix: Path,
    earliest_threshold_df: pd.DataFrame | None = None,
    sycophancy_threshold: float = DEFAULT_SYCOPHANCY_THRESHOLD,
) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_path = output_prefix.parent / f"{output_prefix.name}{SUMMARY_SUFFIX}"
    trajectory_path = output_prefix.parent / f"{output_prefix.name}{TRAJECTORY_SUFFIX}"
    report_path = output_prefix.parent / f"{output_prefix.name}{REPORT_SUFFIX}"
    summary_df.to_csv(summary_path, index=False, na_rep="NA")
    trajectory_df.to_csv(trajectory_path, index=False, na_rep="NA")
    report_path.write_text(
        build_markdown_report(
            summary_df,
            earliest_threshold_df=earliest_threshold_df,
            sycophancy_threshold=sycophancy_threshold,
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        help="Preregistration experiment directory containing arm/seed subdirectories.",
    )
    parser.add_argument(
        "--exclusion-diagnostics",
        type=Path,
        default=None,
        help="Optional prereg exclusion diagnostics CSV to join onto the seed summaries.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        required=True,
        help="Output prefix used for the instability CSV and markdown artifacts.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Python logging level.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    experiment_dir = args.experiment_dir.resolve()
    summary_df, trajectory_df = collect_instability_tables(
        experiment_dir=experiment_dir,
        exclusion_diagnostics_path=(
            None if args.exclusion_diagnostics is None else args.exclusion_diagnostics.resolve()
        ),
    )
    curve_df = collect_checkpoint_curve_tables(experiment_dir=experiment_dir)
    earliest_threshold_df = build_earliest_threshold_table(curve_df) if not curve_df.empty else None
    write_outputs(
        summary_df=summary_df,
        trajectory_df=trajectory_df,
        output_prefix=args.output_prefix.resolve(),
        earliest_threshold_df=earliest_threshold_df,
    )


if __name__ == "__main__":
    main()
