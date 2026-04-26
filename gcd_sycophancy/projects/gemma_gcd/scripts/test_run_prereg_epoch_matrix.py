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

import run_prereg_epoch_matrix as module


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _template_config(tmp_path: Path) -> Path:
    path = tmp_path / "template.json"
    _write_json(
        path,
        {
            "experiment_name": "test_epoch_matrix",
            "finetune_config": {
                "model": "fake/model",
                "epochs": 1,
                "max_seq_length": 32,
            },
        },
    )
    return path


def _fake_analysis() -> dict:
    return {
        "workflow_name": "prereg_analysis",
        "confirmatory_results": [
            {
                "analysis_id": "analysis_1",
                "marginal_risk_difference": -0.18,
                "support_status": "supported",
            },
            {
                "analysis_id": "analysis_2",
                "marginal_risk_difference": -0.005,
                "support_status": "supported",
            },
        ],
        "exploratory_results": [],
    }


# ---------------------------------------------------------------------------
# Derived-config writing
# ---------------------------------------------------------------------------

class TestDerivedConfig:
    def test_derived_config_carries_requested_epoch_count(self, tmp_path):
        template = _template_config(tmp_path)
        derived = tmp_path / "derived.json"
        payload = module.write_derived_template_config(
            template_path=template, derived_path=derived, epochs=5
        )
        assert payload["finetune_config"]["epochs"] == 5
        on_disk = json.loads(derived.read_text(encoding="utf-8"))
        assert on_disk["finetune_config"]["epochs"] == 5

    def test_template_is_unchanged_after_writing_derived_config(self, tmp_path):
        template = _template_config(tmp_path)
        before = template.read_text(encoding="utf-8")
        before_sha = module._sha256_file(template)
        for n in (1, 2, 3, 5):
            derived = tmp_path / f"derived_{n}.json"
            module.write_derived_template_config(
                template_path=template, derived_path=derived, epochs=n
            )
        after = template.read_text(encoding="utf-8")
        assert after == before
        assert module._sha256_file(template) == before_sha

    def test_missing_finetune_config_raises(self, tmp_path):
        template = tmp_path / "bad.json"
        _write_json(template, {"experiment_name": "no_finetune"})
        with pytest.raises(ValueError, match="finetune_config"):
            module.write_derived_template_config(
                template_path=template, derived_path=tmp_path / "out.json", epochs=2
            )

    def test_missing_template_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            module.write_derived_template_config(
                template_path=tmp_path / "missing.json",
                derived_path=tmp_path / "out.json",
                epochs=1,
            )


# ---------------------------------------------------------------------------
# Dry-run path
# ---------------------------------------------------------------------------

class TestDryRun:
    def test_dry_run_writes_derived_configs_and_manifest_without_subprocess(
        self, tmp_path, capsys
    ):
        template = _template_config(tmp_path)
        exp_root = tmp_path / "exp"
        with patch.object(module.subprocess, "run") as mock_run:
            rc = module.run_matrix(
                experiment_root=exp_root,
                epochs_list=(1, 2, 5),
                seeds=(0, 1),
                phases=("setup", "train"),
                template_path=template,
                dry_run=True,
                aggregate_only=False,
                fail_fast=False,
                passthrough_args=[],
            )
        assert rc == 0
        out = capsys.readouterr().out
        # 3 epochs * 2 phases = 6 dry-run lines
        assert out.count("[DRY-RUN]") == 6

        # No run_preregistration.py invocations
        for call in mock_run.call_args_list:
            cmd = call.args[0] if call.args else []
            assert "run_preregistration.py" not in " ".join(map(str, cmd))

        for epochs in (1, 2, 5):
            derived = exp_root / f"epochs_{epochs}" / module.DERIVED_CONFIG_NAME
            assert derived.exists()
            payload = json.loads(derived.read_text(encoding="utf-8"))
            assert payload["finetune_config"]["epochs"] == epochs

        manifest = json.loads(
            (exp_root / module.MATRIX_MANIFEST_NAME).read_text(encoding="utf-8")
        )
        assert manifest["dry_run"] is True
        assert manifest["epochs"] == [1, 2, 5]
        assert all(s == "dry_run" for s in manifest["epoch_statuses"].values())
        assert manifest["template_unchanged"] is True
        assert "provenance" in manifest


# ---------------------------------------------------------------------------
# Failure recording
# ---------------------------------------------------------------------------

class TestFailureRecording:
    def test_failure_per_epoch_recorded_without_fail_fast(self, tmp_path):
        template = _template_config(tmp_path)
        exp_root = tmp_path / "exp"

        def fake_run(cmd, check=True, **kwargs):
            cmd_str = " ".join(map(str, cmd))
            if cmd and cmd[0] == "git":
                class _GitResult:
                    returncode = 0
                    stdout = ""

                return _GitResult()
            if "epochs_2" in cmd_str:
                import subprocess
                raise subprocess.CalledProcessError(returncode=4, cmd=cmd)

            class _Result:
                returncode = 0

            return _Result()

        with patch.object(module.subprocess, "run", side_effect=fake_run):
            rc = module.run_matrix(
                experiment_root=exp_root,
                epochs_list=(1, 2, 3),
                seeds=None,
                phases=("setup",),
                template_path=template,
                dry_run=False,
                aggregate_only=False,
                fail_fast=False,
                passthrough_args=[],
            )
        assert rc == 1
        manifest = json.loads(
            (exp_root / module.MATRIX_MANIFEST_NAME).read_text(encoding="utf-8")
        )
        statuses = manifest["epoch_statuses"]
        assert statuses["epochs_1"] == "completed"
        assert statuses["epochs_2"].startswith("failed:setup:")
        assert statuses["epochs_3"] == "completed"
        # Template config must be untouched even after subprocess execution.
        assert manifest["template_unchanged"] is True


# ---------------------------------------------------------------------------
# Aggregate-only and summary
# ---------------------------------------------------------------------------

class TestAggregateOnly:
    def test_aggregate_only_skips_subprocess_and_emits_summary(self, tmp_path):
        template = _template_config(tmp_path)
        exp_root = tmp_path / "exp"
        for epochs in (1, 2):
            edir = module.epoch_experiment_dir(exp_root, epochs)
            _write_json(edir / "reports" / "prereg_analysis.json", _fake_analysis())

        with patch.object(module.subprocess, "run") as mock_run:
            rc = module.run_matrix(
                experiment_root=exp_root,
                epochs_list=(1, 2),
                seeds=None,
                phases=("setup",),
                template_path=template,
                dry_run=False,
                aggregate_only=True,
                fail_fast=False,
                passthrough_args=[],
            )
        assert rc == 0
        for call in mock_run.call_args_list:
            cmd = call.args[0] if call.args else []
            assert "run_preregistration.py" not in " ".join(map(str, cmd))

        summary = json.loads(
            (exp_root / module.MATRIX_SUMMARY_JSON_NAME).read_text(encoding="utf-8")
        )
        assert summary["epochs"] == [1, 2]
        for epochs in (1, 2):
            entry = summary["epoch_results"][f"epochs_{epochs}"]
            assert entry["status"] == "present"
            assert entry["epochs"] == epochs
            assert entry["key_metrics"]["sycophancy_mrd"] == -0.18
        md = (exp_root / module.MATRIX_SUMMARY_MD_NAME).read_text(encoding="utf-8")
        assert "Epoch Matrix Summary" in md
        assert "| 1 |" in md and "| 2 |" in md
        # Aggregate-only must not have written derived configs.
        for epochs in (1, 2):
            assert not (
                module.epoch_experiment_dir(exp_root, epochs) / module.DERIVED_CONFIG_NAME
            ).exists()
