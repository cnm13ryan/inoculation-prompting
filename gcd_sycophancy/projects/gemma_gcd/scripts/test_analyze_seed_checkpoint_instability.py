from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECTS_DIR = SCRIPT_DIR.parents[2]
GEMMA_GCD_DIR = SCRIPT_DIR.parent
for candidate in (SCRIPT_DIR, PROJECTS_DIR, GEMMA_GCD_DIR):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

import analyze_seed_checkpoint_instability as instability  # noqa: E402
from checkpoint_diagnostics import archive_checkpoint_result_files, discover_checkpoint_result_files  # noqa: E402
from run_ip_sweep import PREREG_ARMS  # noqa: E402


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _results_payload(*, timestamp: str, train_losses: list[float], task_losses: list[float], align_losses: list[float]) -> dict:
    return {
        "timestamp": timestamp,
        "train_losses": train_losses,
        "eval_results": {
            "task_test": {"loss": task_losses},
            "align_test": {"loss": align_losses},
        },
    }


def test_archive_checkpoint_result_files_copies_jsons_to_stable_results_dir(tmp_path: Path):
    exp_folder = tmp_path / "seed_0"
    timestamp = "20260402_120000"
    _write_json(
        exp_folder / "checkpoints" / "epoch_0" / "checkpoint_results" / timestamp / "results_epoch_0.json",
        {"timestamp": timestamp},
    )
    _write_json(
        exp_folder / "checkpoints" / "epoch_1" / "checkpoint_results" / timestamp / "results_epoch_1.json",
        {"timestamp": timestamp},
    )

    archive_dir = archive_checkpoint_result_files(
        exp_folder=exp_folder,
        results_dir=exp_folder / "results",
        timestamp=timestamp,
    )

    assert archive_dir is not None
    assert (archive_dir / "results_epoch_0.json").exists()
    assert (archive_dir / "results_epoch_1.json").exists()
    assert (archive_dir / "manifest.json").exists()

    discovered, source_kind = discover_checkpoint_result_files(exp_folder)
    assert source_kind == "archived_results"
    assert [path.name for path in discovered] == ["results_epoch_0.json", "results_epoch_1.json"]


def test_collect_instability_tables_reports_checkpoint_timing_and_final_exclusion(tmp_path: Path):
    experiment_dir = tmp_path / "experiments" / "preregistration"
    condition_name = "neutral_demo"
    seed_dir = experiment_dir / condition_name / "seed_3"
    timestamp = "20260402_130000"
    neutral_label = next(arm.label for arm in PREREG_ARMS if arm.slug == "neutral_baseline")
    _write_json(experiment_dir / "condition_labels.json", {condition_name: neutral_label})

    _write_json(
        seed_dir / "results" / timestamp / "results.json",
        _results_payload(
            timestamp=timestamp,
            train_losses=[4.0, 2.8, 3.5],
            task_losses=[1.2, 1.1, 1.8],
            align_losses=[1.1, 1.0, 1.7],
        ),
    )
    _write_json(
        seed_dir / "results" / timestamp / "checkpoint_diagnostics" / "results_epoch_0.json",
        _results_payload(
            timestamp=timestamp,
            train_losses=[4.0],
            task_losses=[1.2],
            align_losses=[1.1],
        ),
    )
    _write_json(
        seed_dir / "results" / timestamp / "checkpoint_diagnostics" / "results_epoch_1.json",
        _results_payload(
            timestamp=timestamp,
            train_losses=[4.0, 2.8],
            task_losses=[1.2, 1.1],
            align_losses=[1.1, 1.0],
        ),
    )

    diagnostics_path = experiment_dir / "reports" / "prereg_analysis.exclusion_diagnostics.csv"
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "summary_level": "arm_seed",
                "arm_slug": "neutral_baseline",
                "seed": 3,
                "exclusion_rate": 0.85,
                "top_exclusion_category": "degenerate_response",
            },
            {
                "summary_level": "arm_seed_evaluation_design",
                "arm_slug": "neutral_baseline",
                "seed": 3,
                "evaluation_design": "fixed_interface",
                "exclusion_rate": 0.85,
            },
            {
                "summary_level": "arm_seed_evaluation_design",
                "arm_slug": "neutral_baseline",
                "seed": 3,
                "evaluation_design": "bounded_search",
                "exclusion_rate": 0.98,
            },
        ]
    ).to_csv(diagnostics_path, index=False)

    summary_df, trajectory_df = instability.collect_instability_tables(
        experiment_dir=experiment_dir,
        exclusion_diagnostics_path=diagnostics_path,
    )

    assert len(summary_df) == 1
    row = summary_df.iloc[0]
    assert row["arm_slug"] == "neutral_baseline"
    assert row["seed"] == 3
    assert row["checkpoint_source_kind"] == "archived_results"
    assert row["checkpoint_count"] == 2
    assert row["worst_design"] == "bounded_search"
    assert row["timing_heuristic"] == "degrades_across_checkpoints"
    assert len(trajectory_df) == 3


def test_collect_instability_tables_uses_embedded_results_history_when_checkpoint_files_are_missing(tmp_path: Path):
    experiment_dir = tmp_path / "experiments" / "preregistration"
    condition_name = "irrelevant_demo"
    seed_dir = experiment_dir / condition_name / "seed_0"
    timestamp = "20260402_140000"
    irrelevant_label = next(arm.label for arm in PREREG_ARMS if arm.slug == "irrelevant_prompt_control")
    _write_json(experiment_dir / "condition_labels.json", {condition_name: irrelevant_label})
    _write_json(
        seed_dir / "results" / timestamp / "results.json",
        _results_payload(
            timestamp=timestamp,
            train_losses=[4.2, 3.7],
            task_losses=[1.4, 1.3],
            align_losses=[1.5, 1.4],
        ),
    )
    diagnostics_path = experiment_dir / "reports" / "prereg_analysis.exclusion_diagnostics.csv"
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "summary_level": "arm_seed",
                "arm_slug": "irrelevant_prompt_control",
                "seed": 0,
                "exclusion_rate": 1.0,
                "top_exclusion_category": "degenerate_response",
            }
        ]
    ).to_csv(diagnostics_path, index=False)

    summary_df, _ = instability.collect_instability_tables(
        experiment_dir=experiment_dir,
        exclusion_diagnostics_path=diagnostics_path,
    )

    row = summary_df.iloc[0]
    assert row["checkpoint_source_kind"] == "embedded_results_history"
    assert row["checkpoint_count"] == 2
    assert row["timing_heuristic"] == "appears_only_in_final_eval_or_untracked_metrics"


def test_collect_instability_tables_includes_ptst_via_shared_training_reference(tmp_path: Path):
    experiment_dir = tmp_path / "experiments" / "preregistration"
    condition_name = "ptst_demo"
    seed_dir = experiment_dir / condition_name / "seed_3"
    timestamp = "20260402_150000"
    ptst_label = next(arm.label for arm in PREREG_ARMS if arm.slug == "ptst_eval_only_reminder")
    neutral_seed_dir = experiment_dir / "neutral_demo" / "seed_3"
    _write_json(experiment_dir / "condition_labels.json", {"neutral_demo": "Neutral baseline", condition_name: ptst_label})
    _write_json(
        neutral_seed_dir / "results" / timestamp / "results.json",
        _results_payload(
            timestamp=timestamp,
            train_losses=[4.0, 2.7],
            task_losses=[1.4, 1.2],
            align_losses=[1.5, 1.3],
        ),
    )
    _write_json(
        seed_dir / "shared_training_artifact.json",
        {
            "neutral_model_path": str(
                neutral_seed_dir / "results" / timestamp / "neutral_model"
            )
        },
    )
    diagnostics_path = experiment_dir / "reports" / "prereg_analysis.exclusion_diagnostics.csv"
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "summary_level": "arm_seed",
                "arm_slug": "ptst_eval_only_reminder",
                "seed": 3,
                "exclusion_rate": 0.84,
                "top_exclusion_category": "degenerate_response",
            },
            {
                "summary_level": "arm_seed_evaluation_design",
                "arm_slug": "ptst_eval_only_reminder",
                "seed": 3,
                "evaluation_design": "fixed_interface",
                "exclusion_rate": 0.84,
            },
        ]
    ).to_csv(diagnostics_path, index=False)

    summary_df, _ = instability.collect_instability_tables(
        experiment_dir=experiment_dir,
        exclusion_diagnostics_path=diagnostics_path,
    )

    row = summary_df.iloc[0]
    assert row["arm_slug"] == "ptst_eval_only_reminder"
    assert row["training_source_kind"] == "shared_training_reference"
    assert row["checkpoint_source_kind"] == "embedded_results_history"
    assert row["checkpoint_count"] == 2
