#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECTS_DIR = SCRIPT_DIR.parents[1]
for _p in (SCRIPT_DIR, PROJECTS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import run_prereg_corpus_matrix as module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _fake_analysis(
    *,
    sycophancy_arm2_rate: float = 0.60,
    sycophancy_arm1_rate: float = 0.80,
    direct_solve_mrd: float = -0.02,
    correction_arm5_rate: float = 0.55,
    correction_arm1_rate: float = 0.80,
) -> dict:
    """Minimal prereg_analysis.json structure for aggregation tests."""
    return {
        "workflow_name": "prereg_analysis",
        "confirmatory_results": [
            {
                "analysis_id": "analysis_1",
                "hypothesis_id": "H1",
                "classification": "confirmatory",
                "label": "Sycophancy reduction",
                "arm_a_id": 2,
                "arm_b_id": 1,
                "evaluation_set_name": "confirmatory",
                "prompt_family": "incorrect_confirmation",
                "evaluation_design": "fixed_interface",
                "arm_a_observed_rate": sycophancy_arm2_rate,
                "arm_b_observed_rate": sycophancy_arm1_rate,
                "marginal_risk_difference": sycophancy_arm2_rate - sycophancy_arm1_rate,
                "support_status": "supported" if sycophancy_arm2_rate < sycophancy_arm1_rate else "unsupported",
                "n_rows": 10000,
                "n_clusters": 1000,
                "n_seeds": 4,
            },
            {
                "analysis_id": "analysis_2",
                "hypothesis_id": "H2",
                "classification": "confirmatory",
                "label": "Capability preservation",
                "arm_a_id": 2,
                "arm_b_id": 1,
                "evaluation_set_name": "confirmatory",
                "prompt_family": "direct_solve",
                "evaluation_design": "fixed_interface",
                "marginal_risk_difference": direct_solve_mrd,
                "support_status": "supported" if direct_solve_mrd > -0.02 else "unsupported",
                "n_rows": 8000,
                "n_clusters": 1000,
                "n_seeds": 4,
            },
            {
                "analysis_id": "analysis_3",
                "hypothesis_id": "H3",
                "classification": "confirmatory",
                "label": "Paraphrase robustness",
                "arm_a_id": 2,
                "arm_b_id": 1,
                "evaluation_set_name": "paraphrase",
                "prompt_family": "incorrect_confirmation",
                "evaluation_design": "fixed_interface",
                "arm_a_observed_rate": sycophancy_arm2_rate - 0.05,
                "arm_b_observed_rate": sycophancy_arm1_rate - 0.05,
                "marginal_risk_difference": sycophancy_arm2_rate - sycophancy_arm1_rate,
                "support_status": "unsupported",
                "n_rows": 5000,
                "n_clusters": 500,
                "n_seeds": 4,
            },
        ],
        "exploratory_results": [
            {
                "analysis_id": "exploratory_E4",
                "hypothesis_id": "E4",
                "classification": "exploratory",
                "label": "Correction-data arm vs neutral baseline",
                "arm_a_id": 5,
                "arm_b_id": 1,
                "evaluation_set_name": "confirmatory",
                "prompt_family": "incorrect_confirmation",
                "evaluation_design": "fixed_interface",
                "arm_a_observed_rate": correction_arm5_rate,
                "arm_b_observed_rate": correction_arm1_rate,
                "marginal_risk_difference": correction_arm5_rate - correction_arm1_rate,
                "support_status": "unsupported",
                "n_rows": 10000,
                "n_clusters": 1000,
                "n_seeds": 4,
            },
        ],
        "diagnostics": {
            "exclusion_summary_rows": [
                {
                    "summary_level": "arm",
                    "arm_id": 1,
                    "arm_slug": "neutral_baseline",
                    "arm_label": "Neutral baseline",
                    "seed": None,
                    "evaluation_design": float("nan"),
                    "total_rows": 40000,
                    "parseable_rows": 32000,
                    "parseability_rate": 0.80,
                    "excluded_rows": 8000,
                    "exclusion_rate": 0.20,
                    "included_rows": 32000,
                    "included_rate": 0.80,
                    "top_exclusion_category": "unparseable_response",
                    "top_exclusion_count": 7000,
                    "top_exclusion_rate": 0.175,
                    "top_exclusion_share_of_excluded": 0.875,
                },
                {
                    "summary_level": "arm",
                    "arm_id": 2,
                    "arm_slug": "inoculation_prompting",
                    "arm_label": "Inoculation prompting",
                    "seed": None,
                    "evaluation_design": float("nan"),
                    "total_rows": 40000,
                    "parseable_rows": 34000,
                    "parseability_rate": 0.85,
                    "excluded_rows": 6000,
                    "exclusion_rate": 0.15,
                    "included_rows": 34000,
                    "included_rate": 0.85,
                    "top_exclusion_category": "unparseable_response",
                    "top_exclusion_count": 5500,
                    "top_exclusion_rate": 0.1375,
                    "top_exclusion_share_of_excluded": 0.917,
                },
            ],
            "exclusion_category_rows": [],
        },
        "paired_reporting_supplement": [],
        "joint_interpretation": {},
        "robustness_analyses": [],
        "human_summary": "",
    }


def _write_variant_analysis(experiment_root: Path, variant: str, **kwargs) -> None:
    vdir = module.variant_experiment_dir(experiment_root, variant)
    analysis_path = module._analysis_json_path(vdir)
    _write_json(analysis_path, _fake_analysis(**kwargs))


# ---------------------------------------------------------------------------
# PROJECTS_DIR regression
# ---------------------------------------------------------------------------

class TestProjectsDir:
    def test_projects_dir_is_gcd_sycophancy_projects(self):
        assert module.PROJECTS_DIR.name == "projects"
        assert module.PROJECTS_DIR.parent.name == "gcd_sycophancy"


# ---------------------------------------------------------------------------
# DEFAULT_PHASES regression (must include prefix-search and best-elicited-eval)
# ---------------------------------------------------------------------------

class TestDefaultPhases:
    def test_prefix_search_in_default_phases(self):
        assert "prefix-search" in module.DEFAULT_PHASES

    def test_best_elicited_eval_in_default_phases(self):
        assert "best-elicited-eval" in module.DEFAULT_PHASES

    def test_analysis_after_prerequisites(self):
        phases = list(module.DEFAULT_PHASES)
        assert phases.index("prefix-search") < phases.index("analysis")
        assert phases.index("best-elicited-eval") < phases.index("analysis")


# ---------------------------------------------------------------------------
# build_prereg_command
# ---------------------------------------------------------------------------

class TestBuildPreregCommand:
    def test_phase_is_positional_third_element(self, tmp_path: Path):
        cmd = module.build_prereg_command(
            phase="setup",
            variant="b1",
            experiment_dir=tmp_path / "b1",
            seeds=(0,),
            ip_instruction=None,
            ip_instruction_id=None,
            passthrough_args=[],
        )
        assert cmd[0] == sys.executable
        assert cmd[1].endswith("run_preregistration.py")
        assert cmd[2] == "setup"

    def test_corpus_b_variant_flag(self, tmp_path: Path):
        cmd = module.build_prereg_command(
            phase="train",
            variant="b2",
            experiment_dir=tmp_path / "b2",
            seeds=(0,),
            ip_instruction=None,
            ip_instruction_id=None,
            passthrough_args=[],
        )
        idx = cmd.index("--corpus-b-variant")
        assert cmd[idx + 1] == "b2"

    def test_experiment_dir_flag(self, tmp_path: Path):
        exp_dir = tmp_path / "b1"
        cmd = module.build_prereg_command(
            phase="setup",
            variant="b1",
            experiment_dir=exp_dir,
            seeds=(0,),
            ip_instruction=None,
            ip_instruction_id=None,
            passthrough_args=[],
        )
        idx = cmd.index("--experiment-dir")
        assert cmd[idx + 1] == str(exp_dir)

    def test_seeds_flag(self, tmp_path: Path):
        cmd = module.build_prereg_command(
            phase="setup",
            variant="b1",
            experiment_dir=tmp_path / "b1",
            seeds=(0, 1, 2, 3),
            ip_instruction=None,
            ip_instruction_id=None,
            passthrough_args=[],
        )
        idx = cmd.index("--seeds")
        assert cmd[idx + 1 : idx + 5] == ["0", "1", "2", "3"]

    def test_ip_instruction_included_when_set(self, tmp_path: Path):
        cmd = module.build_prereg_command(
            phase="setup",
            variant="b1",
            experiment_dir=tmp_path / "b1",
            seeds=(0,),
            ip_instruction="Reply as if correct.",
            ip_instruction_id="reply_v1",
            passthrough_args=[],
        )
        idx = cmd.index("--ip-instruction")
        assert cmd[idx + 1] == "Reply as if correct."
        idx_id = cmd.index("--ip-instruction-id")
        assert cmd[idx_id + 1] == "reply_v1"

    def test_ip_instruction_absent_when_none(self, tmp_path: Path):
        cmd = module.build_prereg_command(
            phase="setup",
            variant="b1",
            experiment_dir=tmp_path / "b1",
            seeds=(0,),
            ip_instruction=None,
            ip_instruction_id=None,
            passthrough_args=[],
        )
        assert "--ip-instruction" not in cmd
        assert "--ip-instruction-id" not in cmd

    def test_passthrough_args_appended(self, tmp_path: Path):
        cmd = module.build_prereg_command(
            phase="setup",
            variant="b1",
            experiment_dir=tmp_path / "b1",
            seeds=(0,),
            ip_instruction=None,
            ip_instruction_id=None,
            passthrough_args=["--tensor-parallel-size", "4"],
        )
        assert "--tensor-parallel-size" in cmd
        idx = cmd.index("--tensor-parallel-size")
        assert cmd[idx + 1] == "4"

    def test_b1_and_b2_produce_different_commands(self, tmp_path: Path):
        kwargs = dict(
            phase="setup",
            experiment_dir=tmp_path / "b1",
            seeds=(0,),
            ip_instruction=None,
            ip_instruction_id=None,
            passthrough_args=[],
        )
        cmd_b1 = module.build_prereg_command(variant="b1", **kwargs)
        cmd_b2 = module.build_prereg_command(
            variant="b2",
            phase="setup",
            experiment_dir=tmp_path / "b2",
            seeds=(0,),
            ip_instruction=None,
            ip_instruction_id=None,
            passthrough_args=[],
        )
        b1_bv = cmd_b1[cmd_b1.index("--corpus-b-variant") + 1]
        b2_bv = cmd_b2[cmd_b2.index("--corpus-b-variant") + 1]
        assert b1_bv == "b1"
        assert b2_bv == "b2"


# ---------------------------------------------------------------------------
# variant_experiment_dir
# ---------------------------------------------------------------------------

class TestVariantExperimentDir:
    def test_b1_subdir(self, tmp_path: Path):
        d = module.variant_experiment_dir(tmp_path / "root", "b1")
        assert d == tmp_path / "root" / "b1"

    def test_b2_subdir(self, tmp_path: Path):
        d = module.variant_experiment_dir(tmp_path / "root", "b2")
        assert d == tmp_path / "root" / "b2"


# ---------------------------------------------------------------------------
# Dry-run: manifest produced, no subprocess
# ---------------------------------------------------------------------------

class TestDryRun:
    def test_dry_run_writes_manifest_no_subprocess(self, tmp_path: Path):
        root = tmp_path / "matrix"

        with patch.object(subprocess, "run") as mock_run:
            rc = module.run_matrix(
                experiment_root=root,
                variants=("b1", "b2"),
                seeds=(0, 1),
                phases=("setup",),
                ip_instruction=None,
                ip_instruction_id=None,
                dry_run=True,
                aggregate_only=False,
                passthrough_args=[],
            )

        assert rc == 0
        mock_run.assert_not_called()
        manifest = json.loads((root / module.MATRIX_MANIFEST_NAME).read_text())
        assert manifest["dry_run"] is True

    def test_dry_run_manifest_has_b1_b2_separate_dirs(self, tmp_path: Path):
        root = tmp_path / "matrix"

        module.run_matrix(
            experiment_root=root,
            variants=("b1", "b2"),
            seeds=(0,),
            phases=("setup",),
            ip_instruction=None,
            ip_instruction_id=None,
            dry_run=True,
            aggregate_only=False,
            passthrough_args=[],
        )

        manifest = json.loads((root / module.MATRIX_MANIFEST_NAME).read_text())
        assert "b1" in manifest["variant_dirs"]
        assert "b2" in manifest["variant_dirs"]
        assert manifest["variant_dirs"]["b1"] != manifest["variant_dirs"]["b2"]
        assert str(root / "b1") == manifest["variant_dirs"]["b1"]
        assert str(root / "b2") == manifest["variant_dirs"]["b2"]

    def test_dry_run_prints_one_line_per_phase_per_variant(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ):
        root = tmp_path / "matrix"

        module.run_matrix(
            experiment_root=root,
            variants=("b1", "b2"),
            seeds=(0,),
            phases=("setup", "train"),
            ip_instruction=None,
            ip_instruction_id=None,
            dry_run=True,
            aggregate_only=False,
            passthrough_args=[],
        )

        out = capsys.readouterr().out
        lines = [l for l in out.splitlines() if l.startswith("[DRY-RUN]")]
        # 2 variants × 2 phases = 4 lines
        assert len(lines) == 4

    def test_dry_run_lines_contain_correct_corpus_b_variant(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ):
        root = tmp_path / "matrix"

        module.run_matrix(
            experiment_root=root,
            variants=("b1", "b2"),
            seeds=(0,),
            phases=("setup",),
            ip_instruction=None,
            ip_instruction_id=None,
            dry_run=True,
            aggregate_only=False,
            passthrough_args=[],
        )

        out = capsys.readouterr().out
        lines = [l for l in out.splitlines() if l.startswith("[DRY-RUN]")]
        b1_lines = [l for l in lines if "--corpus-b-variant b1" in l]
        b2_lines = [l for l in lines if "--corpus-b-variant b2" in l]
        assert len(b1_lines) == 1
        assert len(b2_lines) == 1


# ---------------------------------------------------------------------------
# Matrix manifest fields
# ---------------------------------------------------------------------------

class TestMatrixManifestFields:
    def _run_dry(self, root: Path, **kwargs) -> dict:
        module.run_matrix(
            experiment_root=root,
            variants=("b1", "b2"),
            seeds=(0, 1),
            phases=("setup",),
            ip_instruction=kwargs.get("ip_instruction"),
            ip_instruction_id=kwargs.get("ip_instruction_id"),
            dry_run=True,
            aggregate_only=False,
            passthrough_args=[],
        )
        return json.loads((root / module.MATRIX_MANIFEST_NAME).read_text())

    def test_required_top_level_fields(self, tmp_path: Path):
        manifest = self._run_dry(tmp_path / "matrix")
        for field in (
            "workflow_name",
            "experiment_root",
            "variants",
            "variant_dirs",
            "seeds",
            "phases",
            "ip_instruction",
            "ip_instruction_id",
            "dry_run",
            "started_at",
            "completed_at",
            "variant_statuses",
        ):
            assert field in manifest, f"Missing field: {field}"

    def test_ip_instruction_recorded_in_manifest(self, tmp_path: Path):
        manifest = self._run_dry(
            tmp_path / "matrix",
            ip_instruction="Respond as correct.",
            ip_instruction_id="respond_v1",
        )
        assert manifest["ip_instruction"] == "Respond as correct."
        assert manifest["ip_instruction_id"] == "respond_v1"

    def test_null_ip_instruction_recorded(self, tmp_path: Path):
        manifest = self._run_dry(tmp_path / "matrix")
        assert manifest["ip_instruction"] is None
        assert manifest["ip_instruction_id"] is None

    def test_seeds_and_phases_recorded(self, tmp_path: Path):
        root = tmp_path / "matrix"
        module.run_matrix(
            experiment_root=root,
            variants=("b1", "b2"),
            seeds=(0, 2),
            phases=("setup", "train"),
            ip_instruction=None,
            ip_instruction_id=None,
            dry_run=True,
            aggregate_only=False,
            passthrough_args=[],
        )
        manifest = json.loads((root / module.MATRIX_MANIFEST_NAME).read_text())
        assert manifest["seeds"] == [0, 2]
        assert manifest["phases"] == ["setup", "train"]

    def test_variants_recorded(self, tmp_path: Path):
        manifest = self._run_dry(tmp_path / "matrix")
        assert set(manifest["variants"]) == {"b1", "b2"}

    def test_corpus_b_variant_in_each_dir(self, tmp_path: Path):
        manifest = self._run_dry(tmp_path / "matrix")
        assert "b1" in manifest["variant_dirs"]["b1"]
        assert "b2" in manifest["variant_dirs"]["b2"]


# ---------------------------------------------------------------------------
# Failed variant recorded in manifest
# ---------------------------------------------------------------------------

class TestFailedVariantRecording:
    def test_subprocess_failure_recorded_in_variant_statuses(self, tmp_path: Path):
        root = tmp_path / "matrix"

        def fake_run(cmd, *, check):
            if "--corpus-b-variant" in cmd:
                variant_idx = cmd.index("--corpus-b-variant") + 1
                if cmd[variant_idx] == "b2":
                    raise subprocess.CalledProcessError(1, cmd)

        with patch.object(subprocess, "run", side_effect=fake_run):
            rc = module.run_matrix(
                experiment_root=root,
                variants=("b1", "b2"),
                seeds=(0,),
                phases=("setup",),
                ip_instruction=None,
                ip_instruction_id=None,
                dry_run=False,
                aggregate_only=False,
                passthrough_args=[],
            )

        assert rc != 0
        manifest = json.loads((root / module.MATRIX_MANIFEST_NAME).read_text())
        # b1 completed, b2 recorded a failure
        assert manifest["variant_statuses"]["b1"] == "completed"
        assert "failed" in manifest["variant_statuses"]["b2"]

    def test_failure_does_not_prevent_manifest_write(self, tmp_path: Path):
        root = tmp_path / "matrix"

        def always_fail(cmd, *, check):
            raise subprocess.CalledProcessError(1, cmd)

        with patch.object(subprocess, "run", side_effect=always_fail):
            module.run_matrix(
                experiment_root=root,
                variants=("b1",),
                seeds=(0,),
                phases=("setup",),
                ip_instruction=None,
                ip_instruction_id=None,
                dry_run=False,
                aggregate_only=False,
                passthrough_args=[],
            )

        assert (root / module.MATRIX_MANIFEST_NAME).exists()

    def test_failure_recorded_with_phase_name(self, tmp_path: Path):
        root = tmp_path / "matrix"

        def fail_on_train(cmd, *, check):
            if "train" in cmd:
                raise subprocess.CalledProcessError(2, cmd)

        with patch.object(subprocess, "run", side_effect=fail_on_train):
            module.run_matrix(
                experiment_root=root,
                variants=("b1",),
                seeds=(0,),
                phases=("setup", "train"),
                ip_instruction=None,
                ip_instruction_id=None,
                dry_run=False,
                aggregate_only=False,
                passthrough_args=[],
            )

        manifest = json.loads((root / module.MATRIX_MANIFEST_NAME).read_text())
        assert "train" in manifest["variant_statuses"]["b1"]
        assert "failed" in manifest["variant_statuses"]["b1"]


# ---------------------------------------------------------------------------
# Subprocess execution: isolated dirs and exact commands
# ---------------------------------------------------------------------------

class TestSubprocessExecution:
    def test_b1_and_b2_use_separate_experiment_dirs(self, tmp_path: Path):
        root = tmp_path / "matrix"
        calls: list[list[str]] = []

        def fake_run(cmd, *, check):
            calls.append(list(cmd))

        with patch.object(subprocess, "run", side_effect=fake_run):
            module.run_matrix(
                experiment_root=root,
                variants=("b1", "b2"),
                seeds=(0,),
                phases=("setup",),
                ip_instruction=None,
                ip_instruction_id=None,
                dry_run=False,
                aggregate_only=False,
                passthrough_args=[],
            )

        exp_dirs = set()
        for cmd in calls:
            idx = cmd.index("--experiment-dir")
            exp_dirs.add(cmd[idx + 1])
        assert len(exp_dirs) == 2

    def test_phases_run_in_order_per_variant(self, tmp_path: Path):
        root = tmp_path / "matrix"
        b1_phases: list[str] = []

        def fake_run(cmd, *, check):
            bv_idx = cmd.index("--corpus-b-variant") + 1
            if cmd[bv_idx] == "b1":
                b1_phases.append(cmd[2])

        with patch.object(subprocess, "run", side_effect=fake_run):
            module.run_matrix(
                experiment_root=root,
                variants=("b1", "b2"),
                seeds=(0,),
                phases=("setup", "train"),
                ip_instruction=None,
                ip_instruction_id=None,
                dry_run=False,
                aggregate_only=False,
                passthrough_args=[],
            )

        assert b1_phases == ["setup", "train"]

    def test_ip_instruction_propagated_to_both_variants(self, tmp_path: Path):
        root = tmp_path / "matrix"
        calls: list[list[str]] = []

        def fake_run(cmd, *, check):
            calls.append(list(cmd))

        with patch.object(subprocess, "run", side_effect=fake_run):
            module.run_matrix(
                experiment_root=root,
                variants=("b1", "b2"),
                seeds=(0,),
                phases=("setup",),
                ip_instruction="Custom instruction.",
                ip_instruction_id="custom_v1",
                dry_run=False,
                aggregate_only=False,
                passthrough_args=[],
            )

        for cmd in calls:
            assert "--ip-instruction" in cmd
            idx = cmd.index("--ip-instruction")
            assert cmd[idx + 1] == "Custom instruction."

    def test_total_subprocess_calls(self, tmp_path: Path):
        root = tmp_path / "matrix"
        calls: list[list[str]] = []

        def fake_run(cmd, *, check):
            calls.append(list(cmd))

        with patch.object(subprocess, "run", side_effect=fake_run):
            module.run_matrix(
                experiment_root=root,
                variants=("b1", "b2"),
                seeds=(0,),
                phases=("setup", "train"),
                ip_instruction=None,
                ip_instruction_id=None,
                dry_run=False,
                aggregate_only=False,
                passthrough_args=[],
            )

        # 2 variants × 2 phases = 4 calls
        assert len(calls) == 4


# ---------------------------------------------------------------------------
# aggregate-only: no subprocess, reads existing dirs
# ---------------------------------------------------------------------------

class TestAggregateOnly:
    def test_aggregate_only_does_not_call_subprocess(self, tmp_path: Path):
        root = tmp_path
        _write_variant_analysis(root, "b1")
        _write_variant_analysis(root, "b2")

        with patch.object(subprocess, "run") as mock_run:
            rc = module.run_matrix(
                experiment_root=root,
                variants=("b1", "b2"),
                seeds=(0,),
                phases=("setup",),
                ip_instruction=None,
                ip_instruction_id=None,
                dry_run=False,
                aggregate_only=True,
                passthrough_args=[],
            )

        mock_run.assert_not_called()
        assert rc == 0

    def test_aggregate_only_writes_summary_artifacts(self, tmp_path: Path):
        root = tmp_path
        _write_variant_analysis(root, "b1")
        _write_variant_analysis(root, "b2")

        module.run_matrix(
            experiment_root=root,
            variants=("b1", "b2"),
            seeds=(0,),
            phases=("setup",),
            ip_instruction=None,
            ip_instruction_id=None,
            dry_run=False,
            aggregate_only=True,
            passthrough_args=[],
        )

        assert (root / module.MATRIX_SUMMARY_JSON_NAME).exists()
        assert (root / module.MATRIX_SUMMARY_MD_NAME).exists()

    def test_aggregate_only_manifest_has_no_variant_statuses(self, tmp_path: Path):
        root = tmp_path
        _write_variant_analysis(root, "b1")
        _write_variant_analysis(root, "b2")

        module.run_matrix(
            experiment_root=root,
            variants=("b1", "b2"),
            seeds=(0,),
            phases=("setup",),
            ip_instruction=None,
            ip_instruction_id=None,
            dry_run=False,
            aggregate_only=True,
            passthrough_args=[],
        )

        manifest = json.loads((root / module.MATRIX_MANIFEST_NAME).read_text())
        # No phases were executed; variant_statuses should be empty
        assert manifest["variant_statuses"] == {}


# ---------------------------------------------------------------------------
# Summary generation from synthetic analysis files
# ---------------------------------------------------------------------------

class TestSummaryGeneration:
    def _run_aggregation(self, tmp_path: Path, **b1_kwargs) -> tuple[dict, str]:
        root = tmp_path
        _write_variant_analysis(root, "b1", **b1_kwargs)
        _write_variant_analysis(root, "b2")

        module.run_matrix(
            experiment_root=root,
            variants=("b1", "b2"),
            seeds=(0,),
            phases=(),
            ip_instruction=None,
            ip_instruction_id=None,
            dry_run=True,
            aggregate_only=False,
            passthrough_args=[],
        )

        summary_json = json.loads(
            (root / module.MATRIX_SUMMARY_JSON_NAME).read_text(encoding="utf-8")
        )
        summary_md = (root / module.MATRIX_SUMMARY_MD_NAME).read_text(encoding="utf-8")
        return summary_json, summary_md

    def test_summary_json_has_both_variants(self, tmp_path: Path):
        summary, _ = self._run_aggregation(tmp_path)
        assert "b1" in summary["variant_results"]
        assert "b2" in summary["variant_results"]

    def test_summary_json_variant_status_present(self, tmp_path: Path):
        summary, _ = self._run_aggregation(tmp_path)
        assert summary["variant_results"]["b1"]["status"] == "present"
        assert summary["variant_results"]["b2"]["status"] == "present"

    def test_summary_json_sycophancy_rates_extracted(self, tmp_path: Path):
        summary, _ = self._run_aggregation(
            tmp_path,
            sycophancy_arm2_rate=0.40,
            sycophancy_arm1_rate=0.75,
        )
        b1_metrics = summary["variant_results"]["b1"]["key_metrics"]
        assert abs(b1_metrics["sycophancy_arm2_rate"] - 0.40) < 1e-9
        assert abs(b1_metrics["sycophancy_arm1_rate"] - 0.75) < 1e-9

    def test_summary_json_correction_rates_extracted(self, tmp_path: Path):
        summary, _ = self._run_aggregation(
            tmp_path,
            correction_arm5_rate=0.50,
            correction_arm1_rate=0.78,
        )
        b1_metrics = summary["variant_results"]["b1"]["key_metrics"]
        assert abs(b1_metrics["correction_arm5_rate"] - 0.50) < 1e-9

    def test_summary_json_arm2_vs_arm1_by_eval_set_populated(self, tmp_path: Path):
        summary, _ = self._run_aggregation(tmp_path)
        b1_metrics = summary["variant_results"]["b1"]["key_metrics"]
        entries = b1_metrics["arm2_vs_arm1_by_eval_set"]
        assert len(entries) >= 2  # analysis_1 + analysis_2 + analysis_3
        eval_sets = {e["evaluation_set_name"] for e in entries}
        assert "confirmatory" in eval_sets
        assert "paraphrase" in eval_sets

    def test_summary_json_exclusion_summary_populated(self, tmp_path: Path):
        summary, _ = self._run_aggregation(tmp_path)
        excl = summary["variant_results"]["b1"]["exclusion_summary"]
        assert len(excl) == 2
        slugs = {r["arm_slug"] for r in excl}
        assert "neutral_baseline" in slugs
        assert "inoculation_prompting" in slugs

    def test_summary_md_contains_variant_headers(self, tmp_path: Path):
        _, md = self._run_aggregation(tmp_path)
        assert "## Variant B1" in md
        assert "## Variant B2" in md

    def test_summary_md_contains_sycophancy_section(self, tmp_path: Path):
        _, md = self._run_aggregation(tmp_path)
        assert "Sycophancy" in md or "H1" in md

    def test_summary_md_contains_interpretation_note(self, tmp_path: Path):
        _, md = self._run_aggregation(tmp_path)
        assert "Interpretation Note" in md or "not be pooled" in md

    def test_summary_md_contains_pooling_warning(self, tmp_path: Path):
        _, md = self._run_aggregation(tmp_path)
        assert "pooled" in md or "pool" in md

    def test_summary_json_records_generated_at(self, tmp_path: Path):
        summary, _ = self._run_aggregation(tmp_path)
        assert "generated_at" in summary
        assert summary["generated_at"]

    def test_missing_variant_dir_recorded_gracefully(self, tmp_path: Path):
        root = tmp_path
        # Only write b1; b2 is absent
        _write_variant_analysis(root, "b1")

        module.run_matrix(
            experiment_root=root,
            variants=("b1", "b2"),
            seeds=(0,),
            phases=(),
            ip_instruction=None,
            ip_instruction_id=None,
            dry_run=True,
            aggregate_only=False,
            passthrough_args=[],
        )

        summary = json.loads(
            (root / module.MATRIX_SUMMARY_JSON_NAME).read_text(encoding="utf-8")
        )
        assert summary["variant_results"]["b1"]["status"] == "present"
        assert summary["variant_results"]["b2"]["status"] == "missing"


# ---------------------------------------------------------------------------
# read_variant_results
# ---------------------------------------------------------------------------

class TestReadVariantResults:
    def test_present_when_analysis_exists(self, tmp_path: Path):
        root = tmp_path
        _write_variant_analysis(root, "b1")
        results = module.read_variant_results(("b1",), root)
        assert results["b1"]["status"] == "present"

    def test_missing_when_no_analysis(self, tmp_path: Path):
        root = tmp_path
        results = module.read_variant_results(("b1",), root)
        assert results["b1"]["status"] == "missing"

    def test_corpus_b_variant_recorded(self, tmp_path: Path):
        root = tmp_path
        _write_variant_analysis(root, "b1")
        results = module.read_variant_results(("b1",), root)
        assert results["b1"]["corpus_b_variant"] == "b1"


# ---------------------------------------------------------------------------
# write_matrix_summary artifacts
# ---------------------------------------------------------------------------

class TestWriteMatrixSummary:
    def test_json_and_md_both_written(self, tmp_path: Path):
        root = tmp_path
        _write_variant_analysis(root, "b1")
        _write_variant_analysis(root, "b2")
        variant_results = module.read_variant_results(("b1", "b2"), root)
        json_path, md_path = module.write_matrix_summary(
            root,
            variants=("b1", "b2"),
            variant_results=variant_results,
            generated_at="2026-04-26T00:00:00+00:00",
        )
        assert json_path.exists()
        assert md_path.exists()

    def test_json_has_workflow_name(self, tmp_path: Path):
        root = tmp_path
        variant_results = module.read_variant_results(("b1",), root)
        json_path, _ = module.write_matrix_summary(
            root,
            variants=("b1",),
            variant_results=variant_results,
            generated_at="2026-04-26T00:00:00+00:00",
        )
        payload = json.loads(json_path.read_text())
        assert payload["workflow_name"] == "prereg_corpus_matrix_summary"
