#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECTS_DIR = SCRIPT_DIR.parents[1]
for _p in (SCRIPT_DIR, PROJECTS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import run_prereg_model_matrix as module


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _fake_analysis(*, sycophancy_mrd: float = -0.20, direct_solve_mrd: float = -0.01) -> dict:
    return {
        "workflow_name": "prereg_analysis",
        "confirmatory_results": [
            {
                "analysis_id": "analysis_1",
                "hypothesis_id": "H1",
                "label": "Sycophancy reduction",
                "arm_a_id": 2,
                "arm_b_id": 1,
                "evaluation_set_name": "confirmatory",
                "prompt_family": "incorrect_confirmation",
                "arm_a_observed_rate": 0.6,
                "arm_b_observed_rate": 0.8,
                "marginal_risk_difference": sycophancy_mrd,
                "support_status": "supported",
            },
            {
                "analysis_id": "analysis_2",
                "hypothesis_id": "H2",
                "label": "Capability preservation",
                "arm_a_id": 2,
                "arm_b_id": 1,
                "evaluation_set_name": "confirmatory",
                "prompt_family": "direct_solve",
                "marginal_risk_difference": direct_solve_mrd,
                "support_status": "supported",
            },
        ],
        "exploratory_results": [],
    }


def _write_template(path: Path, *, model: str = "fake/model") -> None:
    _write_json(
        path,
        {
            "experiment_name": "test_model_matrix",
            "finetune_config": {
                "model": model,
                "epochs": 1,
                "max_seq_length": 32,
            },
        },
    )


def _two_model_config(tmp_path: Path, *, absolute_template: bool = True) -> Path:
    template = tmp_path / "template.json"
    _write_template(template)
    config_path = tmp_path / "models.json"
    template_value = str(template) if absolute_template else "template.json"
    _write_json(
        config_path,
        {
            "models": [
                {
                    "model_id": "gemma_2b_it",
                    "model_name": "google/gemma-2b-it",
                    "template_config": template_value,
                },
                {
                    "model_id": "gemma_7b_it",
                    "model_name": "google/gemma-7b-it",
                    "template_config": template_value,
                },
            ]
        },
    )
    return config_path


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

class TestLoadModelConfig:
    def test_rejects_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            module.load_model_config(tmp_path / "missing.json")

    def test_rejects_empty_models_list(self, tmp_path):
        config_path = tmp_path / "empty.json"
        _write_json(config_path, {"models": []})
        with pytest.raises(ValueError, match="non-empty 'models'"):
            module.load_model_config(config_path)

    def test_rejects_duplicate_ids(self, tmp_path):
        config_path = tmp_path / "dup.json"
        _write_json(
            config_path,
            {
                "models": [
                    {"model_id": "x", "model_name": "X"},
                    {"model_id": "x", "model_name": "Y"},
                ]
            },
        )
        with pytest.raises(ValueError, match="Duplicate model_id"):
            module.load_model_config(config_path)

    def test_rejects_missing_required_fields(self, tmp_path):
        config_path = tmp_path / "missing_field.json"
        _write_json(config_path, {"models": [{"model_id": "x"}]})
        with pytest.raises(ValueError, match="missing required field"):
            module.load_model_config(config_path)

    def test_loads_valid_config(self, tmp_path):
        config_path = _two_model_config(tmp_path)
        models = module.load_model_config(config_path)
        assert [m["model_id"] for m in models] == ["gemma_2b_it", "gemma_7b_it"]


# ---------------------------------------------------------------------------
# Dry-run command generation
# ---------------------------------------------------------------------------

class TestDryRunCommandGeneration:
    def test_dry_run_emits_one_command_per_model_per_phase(self, tmp_path, capsys):
        config_path = _two_model_config(tmp_path)
        models = module.load_model_config(config_path)
        rc = module.run_matrix(
            experiment_root=tmp_path / "exp",
            models=models,
            seeds=(0, 1),
            phases=("setup", "train"),
            config_path=config_path,
            dry_run=True,
            aggregate_only=False,
            fail_fast=False,
            passthrough_args=[],
        )
        assert rc == 0
        captured = capsys.readouterr().out
        # 2 models * 2 phases = 4 commands
        assert captured.count("[DRY-RUN]") == 4
        # Each command must reference run_preregistration.py and the model dir
        assert "run_preregistration.py" in captured
        assert "gemma_2b_it" in captured
        assert "gemma_7b_it" in captured

    def test_dry_run_writes_manifest_and_summary_with_provenance(self, tmp_path):
        config_path = _two_model_config(tmp_path)
        models = module.load_model_config(config_path)
        exp_root = tmp_path / "exp"
        rc = module.run_matrix(
            experiment_root=exp_root,
            models=models,
            seeds=None,
            phases=("setup",),
            config_path=config_path,
            dry_run=True,
            aggregate_only=False,
            fail_fast=False,
            passthrough_args=[],
        )
        assert rc == 0
        manifest = json.loads((exp_root / module.MATRIX_MANIFEST_NAME).read_text(encoding="utf-8"))
        assert manifest["workflow_name"] == "prereg_model_matrix"
        assert manifest["dry_run"] is True
        assert set(manifest["model_statuses"]) == {"gemma_2b_it", "gemma_7b_it"}
        assert all(s == "dry_run" for s in manifest["model_statuses"].values())
        assert "provenance" in manifest

        summary = json.loads((exp_root / module.MATRIX_SUMMARY_JSON_NAME).read_text(encoding="utf-8"))
        assert summary["models"] == ["gemma_2b_it", "gemma_7b_it"]
        assert "provenance" in summary
        # Both per-model entries should be missing because no analysis has been written.
        for model_id in ("gemma_2b_it", "gemma_7b_it"):
            assert summary["model_results"][model_id]["status"] == "missing"


# ---------------------------------------------------------------------------
# Failure recording
# ---------------------------------------------------------------------------

class TestFailureRecording:
    def test_failure_recorded_per_model_without_fail_fast(self, tmp_path):
        config_path = _two_model_config(tmp_path)
        models = module.load_model_config(config_path)

        call_log: list[list[str]] = []

        def fake_run(cmd, check=True, **kwargs):
            cmd_str = " ".join(map(str, cmd))
            call_log.append(list(map(str, cmd)))
            if cmd[0:3] == ["git", "-C", str(module.PROJECTS_DIR.parent)]:
                class _GitResult:
                    returncode = 0
                    stdout = ""

                return _GitResult()
            if "gemma_2b_it" in cmd_str and "run_preregistration" in cmd_str:
                import subprocess
                raise subprocess.CalledProcessError(returncode=2, cmd=cmd)

            class _Result:
                returncode = 0

            return _Result()

        with patch.object(module.subprocess, "run", side_effect=fake_run):
            rc = module.run_matrix(
                experiment_root=tmp_path / "exp",
                models=models,
                seeds=None,
                phases=("setup",),
                config_path=config_path,
                dry_run=False,
                aggregate_only=False,
                fail_fast=False,
                passthrough_args=[],
            )

        assert rc == 1
        manifest = json.loads(
            (tmp_path / "exp" / module.MATRIX_MANIFEST_NAME).read_text(encoding="utf-8")
        )
        statuses = manifest["model_statuses"]
        assert statuses["gemma_2b_it"].startswith("failed:setup:")
        assert statuses["gemma_7b_it"] == "completed"

    def test_fail_fast_stops_after_first_failure(self, tmp_path):
        config_path = _two_model_config(tmp_path)
        models = module.load_model_config(config_path)

        def fake_run(cmd, check=True, **kwargs):
            if cmd and cmd[0] == "git":
                class _GitResult:
                    returncode = 0
                    stdout = ""

                return _GitResult()
            import subprocess
            raise subprocess.CalledProcessError(returncode=3, cmd=cmd)

        with patch.object(module.subprocess, "run", side_effect=fake_run):
            rc = module.run_matrix(
                experiment_root=tmp_path / "exp",
                models=models,
                seeds=None,
                phases=("setup",),
                config_path=config_path,
                dry_run=False,
                aggregate_only=False,
                fail_fast=True,
                passthrough_args=[],
            )

        assert rc == 1
        manifest = json.loads(
            (tmp_path / "exp" / module.MATRIX_MANIFEST_NAME).read_text(encoding="utf-8")
        )
        statuses = manifest["model_statuses"]
        # Only the first model should have run; the second never started.
        assert statuses["gemma_2b_it"].startswith("failed:setup:")
        assert "gemma_7b_it" not in statuses


# ---------------------------------------------------------------------------
# Aggregate-only and summary generation
# ---------------------------------------------------------------------------

class TestAggregateOnly:
    def _write_per_model_analyses(self, exp_root: Path, models: list[dict]):
        for entry in models:
            mdir = module.model_experiment_dir(exp_root, entry["model_id"])
            _write_json(mdir / "reports" / "prereg_analysis.json", _fake_analysis())

    def test_aggregate_only_skips_subprocess_and_reads_existing_artifacts(self, tmp_path):
        config_path = _two_model_config(tmp_path)
        models = module.load_model_config(config_path)
        exp_root = tmp_path / "exp"
        self._write_per_model_analyses(exp_root, models)

        with patch.object(module.subprocess, "run") as mock_run:
            rc = module.run_matrix(
                experiment_root=exp_root,
                models=models,
                seeds=None,
                phases=("setup",),
                config_path=config_path,
                dry_run=False,
                aggregate_only=True,
                fail_fast=False,
                passthrough_args=[],
            )
        assert rc == 0
        # No run_preregistration.py invocations should have been made; only
        # provenance-related git lookups are permitted.
        for call in mock_run.call_args_list:
            cmd = list(call.args[0]) if call.args else []
            assert "run_preregistration.py" not in " ".join(map(str, cmd)), (
                f"aggregate-only path must not invoke run_preregistration.py, got: {cmd}"
            )

        summary = json.loads((exp_root / module.MATRIX_SUMMARY_JSON_NAME).read_text(encoding="utf-8"))
        assert summary["models"] == ["gemma_2b_it", "gemma_7b_it"]
        for model_id in ("gemma_2b_it", "gemma_7b_it"):
            entry = summary["model_results"][model_id]
            assert entry["status"] == "present"
            assert entry["key_metrics"]["sycophancy_mrd"] == -0.20

    def test_summary_md_contains_per_model_section(self, tmp_path):
        config_path = _two_model_config(tmp_path)
        models = module.load_model_config(config_path)
        exp_root = tmp_path / "exp"
        self._write_per_model_analyses(exp_root, models)

        rc = module.run_matrix(
            experiment_root=exp_root,
            models=models,
            seeds=None,
            phases=(),
            config_path=config_path,
            dry_run=False,
            aggregate_only=True,
            fail_fast=False,
            passthrough_args=[],
        )
        assert rc == 0
        md = (exp_root / module.MATRIX_SUMMARY_MD_NAME).read_text(encoding="utf-8")
        assert "## Model `gemma_2b_it`" in md
        assert "## Model `gemma_7b_it`" in md
        assert "H1 sycophancy MRD" in md


# ---------------------------------------------------------------------------
# Regression tests for bug 1 (model_name not propagated) and bug 2 (template_config
# resolved relative to caller cwd causing setup failures).
# ---------------------------------------------------------------------------

class TestModelNamePropagation:
    """Bug 1: two entries that share a template must still produce distinct
    fine-tuned model identities. The runner now writes a derived config per
    model with finetune_config.model overridden to model_name."""

    def test_derived_config_overrides_finetune_config_model_per_entry(self, tmp_path):
        config_path = _two_model_config(tmp_path)
        models = module.load_model_config(config_path)
        exp_root = tmp_path / "exp"

        rc = module.run_matrix(
            experiment_root=exp_root,
            models=models,
            seeds=None,
            phases=("setup",),
            config_path=config_path,
            dry_run=True,
            aggregate_only=False,
            fail_fast=False,
            passthrough_args=[],
        )
        assert rc == 0

        manifest = json.loads(
            (exp_root / module.MATRIX_MANIFEST_NAME).read_text(encoding="utf-8")
        )
        derived = manifest["derived_configs"]
        assert set(derived) == {"gemma_2b_it", "gemma_7b_it"}

        cfg_2b = json.loads(Path(derived["gemma_2b_it"]).read_text(encoding="utf-8"))
        cfg_7b = json.loads(Path(derived["gemma_7b_it"]).read_text(encoding="utf-8"))
        assert cfg_2b["finetune_config"]["model"] == "google/gemma-2b-it"
        assert cfg_7b["finetune_config"]["model"] == "google/gemma-7b-it"
        # Distinct files so writing one cannot clobber the other.
        assert derived["gemma_2b_it"] != derived["gemma_7b_it"]

    def test_source_template_unchanged_after_run(self, tmp_path):
        config_path = _two_model_config(tmp_path)
        template = tmp_path / "template.json"
        before = template.read_text(encoding="utf-8")
        models = module.load_model_config(config_path)
        module.run_matrix(
            experiment_root=tmp_path / "exp",
            models=models,
            seeds=None,
            phases=("setup",),
            config_path=config_path,
            dry_run=True,
            aggregate_only=False,
            fail_fast=False,
            passthrough_args=[],
        )
        assert template.read_text(encoding="utf-8") == before


class TestTemplatePathResolution:
    """Bug 2: relative template paths must resolve against PROJECTS_DIR (not
    against the caller's cwd) so launching the runner from any directory works."""

    def test_relative_template_resolves_against_projects_dir(self, tmp_path, monkeypatch):
        relative_template = tmp_path / "rel_template.json"
        _write_template(relative_template)
        monkeypatch.setattr(module, "PROJECTS_DIR", tmp_path)
        resolved = module._resolve_template_path("rel_template.json")
        assert resolved == relative_template.resolve()

    def test_absolute_template_path_is_preserved(self, tmp_path):
        abs_path = tmp_path / "abs_template.json"
        _write_template(abs_path)
        resolved = module._resolve_template_path(str(abs_path))
        assert resolved == abs_path

    def test_subprocess_run_uses_projects_dir_as_cwd(self, tmp_path, monkeypatch):
        config_path = _two_model_config(tmp_path)
        models = module.load_model_config(config_path)
        recorded_cwds: list[str | None] = []

        def fake_run(cmd, check=True, **kwargs):
            if cmd and cmd[0] == "git":
                class _GitResult:
                    returncode = 0
                    stdout = ""

                return _GitResult()
            recorded_cwds.append(kwargs.get("cwd"))

            class _Result:
                returncode = 0

            return _Result()

        with patch.object(module.subprocess, "run", side_effect=fake_run):
            rc = module.run_matrix(
                experiment_root=tmp_path / "exp",
                models=models,
                seeds=None,
                phases=("setup",),
                config_path=config_path,
                dry_run=False,
                aggregate_only=False,
                fail_fast=False,
                passthrough_args=[],
            )
        assert rc == 0
        # All run_preregistration.py invocations must run with cwd=PROJECTS_DIR
        # so relative paths inside run_preregistration.py resolve correctly.
        assert recorded_cwds, "Expected at least one subprocess invocation"
        for cwd in recorded_cwds:
            assert cwd == str(module.PROJECTS_DIR)

    def test_derived_template_path_is_absolute_in_command(self, tmp_path):
        config_path = _two_model_config(tmp_path, absolute_template=False)
        template = tmp_path / "template.json"
        # Make module.PROJECTS_DIR resolve relative templates inside tmp_path.
        models = module.load_model_config(config_path)
        with patch.object(module, "PROJECTS_DIR", tmp_path):
            captured: list[list[str]] = []

            def fake_run(cmd, check=True, **kwargs):
                if cmd and cmd[0] == "git":
                    class _GitResult:
                        returncode = 0
                        stdout = ""

                    return _GitResult()
                captured.append(list(map(str, cmd)))

                class _Result:
                    returncode = 0

                return _Result()

            with patch.object(module.subprocess, "run", side_effect=fake_run):
                module.run_matrix(
                    experiment_root=tmp_path / "exp",
                    models=models,
                    seeds=None,
                    phases=("setup",),
                    config_path=config_path,
                    dry_run=False,
                    aggregate_only=False,
                    fail_fast=False,
                    passthrough_args=[],
                )
        prereg_calls = [c for c in captured if any("run_preregistration.py" in s for s in c)]
        assert prereg_calls, "Expected at least one run_preregistration.py invocation"
        for cmd in prereg_calls:
            tc_index = cmd.index("--template-config")
            template_arg = cmd[tc_index + 1]
            assert Path(template_arg).is_absolute(), (
                f"--template-config must be absolute; got {template_arg!r}"
            )
