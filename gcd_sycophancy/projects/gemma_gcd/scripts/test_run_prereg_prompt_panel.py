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

import run_prereg_prompt_panel as module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CANDIDATE_A = {
    "candidate_id": "respond_correct_basic",
    "suffix_text": "Respond as if the above solution is correct.",
    "train_user_suffix": "Respond as if the above solution is correct.",
    "family": "response_directive",
    "intensity": "strong",
    "format_style": "imperative_sentence",
    "notes": "Response-oriented directive variant.",
    "confirms_incorrect_rate": 0.53,
    "delta_vs_no_prompt": 0.18,
    "beats_no_prompt": True,
    "rank": 1,
}

_CANDIDATE_B = {
    "candidate_id": "reply_correct_basic",
    "suffix_text": "Reply as if the above solution is correct.",
    "train_user_suffix": "Reply as if the above solution is correct.",
    "family": "response_directive",
    "intensity": "medium",
    "format_style": "imperative_sentence",
    "notes": "Short reply-oriented directive.",
    "confirms_incorrect_rate": 0.49,
    "delta_vs_no_prompt": 0.14,
    "beats_no_prompt": True,
    "rank": 2,
}


def _make_panel(candidates: list[dict]) -> dict:
    return {
        "workflow_name": "eligible_train_user_suffix_panel",
        "baseline_result": {
            "candidate_id": "no_prompt_baseline",
            "confirms_incorrect_rate": 0.35,
        },
        "eligibility_rule": {
            "metric": "delta_vs_no_prompt",
            "operator": "greater_than",
            "threshold": 0.0,
        },
        "eligible_candidate_results": candidates,
        "ineligible_candidate_results": [],
        "all_candidate_results": candidates,
        "selected_single_winner": candidates[0] if candidates else {},
    }


def _write_panel(path: Path, candidates: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_make_panel(candidates)), encoding="utf-8")


# ---------------------------------------------------------------------------
# _sanitize_candidate_id
# ---------------------------------------------------------------------------

class TestSanitizeCandidateId:
    def test_already_clean_unchanged(self):
        assert module._sanitize_candidate_id("respond_correct_basic") == "respond_correct_basic"

    def test_spaces_become_underscore(self):
        assert module._sanitize_candidate_id("foo bar") == "foo_bar"

    def test_dots_become_underscore(self):
        assert module._sanitize_candidate_id("foo.bar.baz") == "foo_bar_baz"

    def test_collapses_consecutive_underscores(self):
        assert module._sanitize_candidate_id("foo  bar") == "foo_bar"

    def test_hyphens_preserved(self):
        assert module._sanitize_candidate_id("foo-bar") == "foo-bar"

    def test_strips_leading_trailing_underscores(self):
        assert module._sanitize_candidate_id("_foo_") == "foo"


# ---------------------------------------------------------------------------
# Empty eligible panel
# ---------------------------------------------------------------------------

class TestEmptyEligiblePanel:
    def test_empty_panel_returns_nonzero(self, tmp_path: Path):
        panel_path = tmp_path / "eligible.json"
        _write_panel(panel_path, [])
        rc = module.run_panel(
            eligible_panel_path=panel_path,
            experiment_root=tmp_path / "root",
            corpus_b_variant="b1",
            seeds=(0, 1),
            phases=("setup",),
            dry_run=False,
            limit_candidates=None,
            passthrough_args=[],
        )
        assert rc != 0

    def test_empty_panel_dry_run_also_returns_nonzero(self, tmp_path: Path):
        panel_path = tmp_path / "eligible.json"
        _write_panel(panel_path, [])
        rc = module.run_panel(
            eligible_panel_path=panel_path,
            experiment_root=tmp_path / "root",
            corpus_b_variant="b1",
            seeds=(0,),
            phases=("setup",),
            dry_run=True,
            limit_candidates=None,
            passthrough_args=[],
        )
        assert rc != 0


# ---------------------------------------------------------------------------
# Candidate ID collision detection
# ---------------------------------------------------------------------------

class TestCandidateIdCollision:
    def test_sanitization_collision_raises(self):
        colliders = [
            {**_CANDIDATE_A, "candidate_id": "respond correct basic"},
            {**_CANDIDATE_B, "candidate_id": "respond.correct.basic"},
        ]
        with pytest.raises(ValueError, match="collision"):
            module.check_candidate_id_collisions(colliders)

    def test_distinct_ids_no_error(self):
        module.check_candidate_id_collisions([_CANDIDATE_A, _CANDIDATE_B])

    def test_run_panel_raises_on_collision(self, tmp_path: Path):
        colliders = [
            {**_CANDIDATE_A, "candidate_id": "foo bar"},
            {**_CANDIDATE_B, "candidate_id": "foo.bar"},
        ]
        panel_path = tmp_path / "eligible.json"
        _write_panel(panel_path, colliders)
        with pytest.raises(ValueError, match="collision"):
            module.run_panel(
                eligible_panel_path=panel_path,
                experiment_root=tmp_path / "root",
                corpus_b_variant="b1",
                seeds=(0,),
                phases=("setup",),
                dry_run=True,
                limit_candidates=None,
                passthrough_args=[],
            )


# ---------------------------------------------------------------------------
# Dry-run behaviour
# ---------------------------------------------------------------------------

class TestDryRun:
    def test_dry_run_writes_manifest_without_subprocess(self, tmp_path: Path):
        panel_path = tmp_path / "eligible.json"
        _write_panel(panel_path, [_CANDIDATE_A, _CANDIDATE_B])
        experiment_root = tmp_path / "root"

        with patch.object(subprocess, "run") as mock_run:
            rc = module.run_panel(
                eligible_panel_path=panel_path,
                experiment_root=experiment_root,
                corpus_b_variant="b1",
                seeds=(0, 1),
                phases=("setup",),
                dry_run=True,
                limit_candidates=None,
                passthrough_args=[],
            )

        assert rc == 0
        mock_run.assert_not_called()
        manifest_path = experiment_root / module.PANEL_MANIFEST_NAME
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["dry_run"] is True

    def test_dry_run_prints_one_line_per_phase_per_candidate(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ):
        panel_path = tmp_path / "eligible.json"
        _write_panel(panel_path, [_CANDIDATE_A, _CANDIDATE_B])
        experiment_root = tmp_path / "root"

        module.run_panel(
            eligible_panel_path=panel_path,
            experiment_root=experiment_root,
            corpus_b_variant="b1",
            seeds=(0,),
            phases=("setup", "train"),
            dry_run=True,
            limit_candidates=None,
            passthrough_args=[],
        )

        out = capsys.readouterr().out
        dry_run_lines = [line for line in out.splitlines() if line.startswith("[DRY-RUN]")]
        # 2 candidates × 2 phases = 4 lines
        assert len(dry_run_lines) == 4

    def test_dry_run_output_contains_run_preregistration(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ):
        panel_path = tmp_path / "eligible.json"
        _write_panel(panel_path, [_CANDIDATE_A])
        experiment_root = tmp_path / "root"

        module.run_panel(
            eligible_panel_path=panel_path,
            experiment_root=experiment_root,
            corpus_b_variant="b1",
            seeds=(0,),
            phases=("setup",),
            dry_run=True,
            limit_candidates=None,
            passthrough_args=[],
        )

        out = capsys.readouterr().out
        assert "run_preregistration.py" in out
        assert _CANDIDATE_A["train_user_suffix"] in out


# ---------------------------------------------------------------------------
# Command construction
# ---------------------------------------------------------------------------

class TestBuildPreregCommand:
    def test_command_structure(self, tmp_path: Path):
        exp_dir = tmp_path / "b1" / "respond_correct_basic"
        cmd = module.build_prereg_command(
            phase="setup",
            experiment_dir=exp_dir,
            candidate=_CANDIDATE_A,
            corpus_b_variant="b1",
            seeds=(0, 1, 2, 3),
            passthrough_args=[],
        )
        assert cmd[0] == sys.executable
        assert cmd[1].endswith("run_preregistration.py")
        assert "setup" in cmd

    def test_experiment_dir_flag(self, tmp_path: Path):
        exp_dir = tmp_path / "b1" / "respond_correct_basic"
        cmd = module.build_prereg_command(
            phase="setup",
            experiment_dir=exp_dir,
            candidate=_CANDIDATE_A,
            corpus_b_variant="b1",
            seeds=(0,),
            passthrough_args=[],
        )
        idx = cmd.index("--experiment-dir")
        assert cmd[idx + 1] == str(exp_dir)

    def test_ip_instruction_flag(self, tmp_path: Path):
        exp_dir = tmp_path / "b1" / "respond_correct_basic"
        cmd = module.build_prereg_command(
            phase="setup",
            experiment_dir=exp_dir,
            candidate=_CANDIDATE_A,
            corpus_b_variant="b1",
            seeds=(0,),
            passthrough_args=[],
        )
        idx = cmd.index("--ip-instruction")
        assert cmd[idx + 1] == _CANDIDATE_A["train_user_suffix"]

    def test_ip_instruction_id_flag(self, tmp_path: Path):
        exp_dir = tmp_path / "b1" / "respond_correct_basic"
        cmd = module.build_prereg_command(
            phase="setup",
            experiment_dir=exp_dir,
            candidate=_CANDIDATE_A,
            corpus_b_variant="b1",
            seeds=(0,),
            passthrough_args=[],
        )
        idx = cmd.index("--ip-instruction-id")
        assert cmd[idx + 1] == _CANDIDATE_A["candidate_id"]

    def test_corpus_b_variant_flag(self, tmp_path: Path):
        exp_dir = tmp_path / "b2" / "respond_correct_basic"
        cmd = module.build_prereg_command(
            phase="setup",
            experiment_dir=exp_dir,
            candidate=_CANDIDATE_A,
            corpus_b_variant="b2",
            seeds=(0,),
            passthrough_args=[],
        )
        idx = cmd.index("--corpus-b-variant")
        assert cmd[idx + 1] == "b2"

    def test_seeds_flag(self, tmp_path: Path):
        exp_dir = tmp_path / "b1" / "respond_correct_basic"
        cmd = module.build_prereg_command(
            phase="setup",
            experiment_dir=exp_dir,
            candidate=_CANDIDATE_A,
            corpus_b_variant="b1",
            seeds=(0, 1, 2, 3),
            passthrough_args=[],
        )
        idx = cmd.index("--seeds")
        assert cmd[idx + 1 : idx + 5] == ["0", "1", "2", "3"]

    def test_passthrough_args_appended(self, tmp_path: Path):
        exp_dir = tmp_path / "b1" / "respond_correct_basic"
        cmd = module.build_prereg_command(
            phase="setup",
            experiment_dir=exp_dir,
            candidate=_CANDIDATE_A,
            corpus_b_variant="b1",
            seeds=(0,),
            passthrough_args=["--tensor-parallel-size", "4"],
        )
        assert "--tensor-parallel-size" in cmd
        pt_idx = cmd.index("--tensor-parallel-size")
        assert cmd[pt_idx + 1] == "4"

    def test_suffix_text_fallback_when_no_train_user_suffix(self, tmp_path: Path):
        candidate_no_tus = {k: v for k, v in _CANDIDATE_A.items() if k != "train_user_suffix"}
        exp_dir = tmp_path / "b1" / "respond_correct_basic"
        cmd = module.build_prereg_command(
            phase="setup",
            experiment_dir=exp_dir,
            candidate=candidate_no_tus,
            corpus_b_variant="b1",
            seeds=(0,),
            passthrough_args=[],
        )
        idx = cmd.index("--ip-instruction")
        assert cmd[idx + 1] == _CANDIDATE_A["suffix_text"]


# ---------------------------------------------------------------------------
# corpus_b_variant propagation
# ---------------------------------------------------------------------------

class TestCorpusBVariantPropagation:
    def test_b1_in_experiment_dir(self, tmp_path: Path):
        exp_dir = module.candidate_experiment_dir(tmp_path / "root", "b1", _CANDIDATE_A)
        assert exp_dir.parent.name == "b1"

    def test_b2_in_experiment_dir(self, tmp_path: Path):
        exp_dir = module.candidate_experiment_dir(tmp_path / "root", "b2", _CANDIDATE_A)
        assert exp_dir.parent.name == "b2"

    def test_b2_in_command(self, tmp_path: Path):
        exp_dir = tmp_path / "b2" / "respond_correct_basic"
        cmd = module.build_prereg_command(
            phase="setup",
            experiment_dir=exp_dir,
            candidate=_CANDIDATE_A,
            corpus_b_variant="b2",
            seeds=(0,),
            passthrough_args=[],
        )
        idx = cmd.index("--corpus-b-variant")
        assert cmd[idx + 1] == "b2"

    def test_b2_manifest_records_variant(self, tmp_path: Path):
        panel_path = tmp_path / "eligible.json"
        _write_panel(panel_path, [_CANDIDATE_A])
        experiment_root = tmp_path / "root"

        module.run_panel(
            eligible_panel_path=panel_path,
            experiment_root=experiment_root,
            corpus_b_variant="b2",
            seeds=(0,),
            phases=("setup",),
            dry_run=True,
            limit_candidates=None,
            passthrough_args=[],
        )

        manifest = json.loads(
            (experiment_root / module.PANEL_MANIFEST_NAME).read_text(encoding="utf-8")
        )
        assert manifest["corpus_b_variant"] == "b2"
        assert "b2" in manifest["candidates"][0]["experiment_dir"]


# ---------------------------------------------------------------------------
# Panel manifest content
# ---------------------------------------------------------------------------

class TestPanelManifestContent:
    def test_manifest_has_all_required_top_level_fields(self, tmp_path: Path):
        panel_path = tmp_path / "eligible.json"
        _write_panel(panel_path, [_CANDIDATE_A, _CANDIDATE_B])
        experiment_root = tmp_path / "root"

        module.run_panel(
            eligible_panel_path=panel_path,
            experiment_root=experiment_root,
            corpus_b_variant="b1",
            seeds=(0, 1),
            phases=("setup",),
            dry_run=True,
            limit_candidates=None,
            passthrough_args=[],
        )

        manifest = json.loads(
            (experiment_root / module.PANEL_MANIFEST_NAME).read_text(encoding="utf-8")
        )
        for field in (
            "source_eligible_panel",
            "experiment_root",
            "corpus_b_variant",
            "seeds",
            "phases",
            "started_at",
            "completed_at",
            "dry_run",
            "candidates",
        ):
            assert field in manifest, f"Missing field: {field}"

    def test_manifest_candidate_fields(self, tmp_path: Path):
        panel_path = tmp_path / "eligible.json"
        _write_panel(panel_path, [_CANDIDATE_A])
        experiment_root = tmp_path / "root"

        module.run_panel(
            eligible_panel_path=panel_path,
            experiment_root=experiment_root,
            corpus_b_variant="b1",
            seeds=(0,),
            phases=("setup",),
            dry_run=True,
            limit_candidates=None,
            passthrough_args=[],
        )

        manifest = json.loads(
            (experiment_root / module.PANEL_MANIFEST_NAME).read_text(encoding="utf-8")
        )
        c = manifest["candidates"][0]
        assert c["candidate_id"] == _CANDIDATE_A["candidate_id"]
        assert c["sanitized_id"] == module._sanitize_candidate_id(_CANDIDATE_A["candidate_id"])
        assert c["suffix_text"] == _CANDIDATE_A["train_user_suffix"]
        assert "experiment_dir" in c

    def test_manifest_source_panel_path_recorded(self, tmp_path: Path):
        panel_path = tmp_path / "my_panel.json"
        _write_panel(panel_path, [_CANDIDATE_A])
        experiment_root = tmp_path / "root"

        module.run_panel(
            eligible_panel_path=panel_path,
            experiment_root=experiment_root,
            corpus_b_variant="b1",
            seeds=(0,),
            phases=("setup",),
            dry_run=True,
            limit_candidates=None,
            passthrough_args=[],
        )

        manifest = json.loads(
            (experiment_root / module.PANEL_MANIFEST_NAME).read_text(encoding="utf-8")
        )
        assert manifest["source_eligible_panel"] == str(panel_path)

    def test_manifest_seeds_and_phases_recorded(self, tmp_path: Path):
        panel_path = tmp_path / "eligible.json"
        _write_panel(panel_path, [_CANDIDATE_A])
        experiment_root = tmp_path / "root"

        module.run_panel(
            eligible_panel_path=panel_path,
            experiment_root=experiment_root,
            corpus_b_variant="b1",
            seeds=(0, 2),
            phases=("setup", "train"),
            dry_run=True,
            limit_candidates=None,
            passthrough_args=[],
        )

        manifest = json.loads(
            (experiment_root / module.PANEL_MANIFEST_NAME).read_text(encoding="utf-8")
        )
        assert manifest["seeds"] == [0, 2]
        assert manifest["phases"] == ["setup", "train"]


# ---------------------------------------------------------------------------
# limit_candidates
# ---------------------------------------------------------------------------

class TestLimitCandidates:
    def test_limit_caps_processed_candidates(self, tmp_path: Path):
        panel_path = tmp_path / "eligible.json"
        _write_panel(panel_path, [_CANDIDATE_A, _CANDIDATE_B])
        experiment_root = tmp_path / "root"

        module.run_panel(
            eligible_panel_path=panel_path,
            experiment_root=experiment_root,
            corpus_b_variant="b1",
            seeds=(0,),
            phases=("setup",),
            dry_run=True,
            limit_candidates=1,
            passthrough_args=[],
        )

        manifest = json.loads(
            (experiment_root / module.PANEL_MANIFEST_NAME).read_text(encoding="utf-8")
        )
        assert len(manifest["candidates"]) == 1
        assert manifest["candidates"][0]["candidate_id"] == _CANDIDATE_A["candidate_id"]

    def test_limit_none_processes_all(self, tmp_path: Path):
        panel_path = tmp_path / "eligible.json"
        _write_panel(panel_path, [_CANDIDATE_A, _CANDIDATE_B])
        experiment_root = tmp_path / "root"

        module.run_panel(
            eligible_panel_path=panel_path,
            experiment_root=experiment_root,
            corpus_b_variant="b1",
            seeds=(0,),
            phases=("setup",),
            dry_run=True,
            limit_candidates=None,
            passthrough_args=[],
        )

        manifest = json.loads(
            (experiment_root / module.PANEL_MANIFEST_NAME).read_text(encoding="utf-8")
        )
        assert len(manifest["candidates"]) == 2


# ---------------------------------------------------------------------------
# Subprocess execution (monkeypatched)
# ---------------------------------------------------------------------------

class TestSubprocessExecution:
    def test_subprocess_called_once_per_phase_per_candidate(self, tmp_path: Path):
        panel_path = tmp_path / "eligible.json"
        _write_panel(panel_path, [_CANDIDATE_A, _CANDIDATE_B])
        experiment_root = tmp_path / "root"

        calls: list[list[str]] = []

        def fake_run(cmd, *, check):
            calls.append(list(cmd))

        with patch.object(subprocess, "run", side_effect=fake_run):
            rc = module.run_panel(
                eligible_panel_path=panel_path,
                experiment_root=experiment_root,
                corpus_b_variant="b1",
                seeds=(0,),
                phases=("setup", "train"),
                dry_run=False,
                limit_candidates=None,
                passthrough_args=[],
            )

        assert rc == 0
        # 2 candidates × 2 phases = 4 calls
        assert len(calls) == 4

    def test_subprocess_commands_contain_correct_phases(self, tmp_path: Path):
        panel_path = tmp_path / "eligible.json"
        _write_panel(panel_path, [_CANDIDATE_A])
        experiment_root = tmp_path / "root"

        calls: list[list[str]] = []

        def fake_run(cmd, *, check):
            calls.append(list(cmd))

        with patch.object(subprocess, "run", side_effect=fake_run):
            module.run_panel(
                eligible_panel_path=panel_path,
                experiment_root=experiment_root,
                corpus_b_variant="b1",
                seeds=(0,),
                phases=("setup", "train"),
                dry_run=False,
                limit_candidates=None,
                passthrough_args=[],
            )

        assert calls[0][2] == "setup"
        assert calls[1][2] == "train"

    def test_subprocess_commands_have_isolated_experiment_dirs(self, tmp_path: Path):
        panel_path = tmp_path / "eligible.json"
        _write_panel(panel_path, [_CANDIDATE_A, _CANDIDATE_B])
        experiment_root = tmp_path / "root"

        calls: list[list[str]] = []

        def fake_run(cmd, *, check):
            calls.append(list(cmd))

        with patch.object(subprocess, "run", side_effect=fake_run):
            module.run_panel(
                eligible_panel_path=panel_path,
                experiment_root=experiment_root,
                corpus_b_variant="b1",
                seeds=(0,),
                phases=("setup",),
                dry_run=False,
                limit_candidates=None,
                passthrough_args=[],
            )

        exp_dirs = set()
        for cmd in calls:
            idx = cmd.index("--experiment-dir")
            exp_dirs.add(cmd[idx + 1])
        # each candidate has its own isolated experiment directory
        assert len(exp_dirs) == 2

    def test_subprocess_commands_contain_ip_instruction(self, tmp_path: Path):
        panel_path = tmp_path / "eligible.json"
        _write_panel(panel_path, [_CANDIDATE_A])
        experiment_root = tmp_path / "root"

        calls: list[list[str]] = []

        def fake_run(cmd, *, check):
            calls.append(list(cmd))

        with patch.object(subprocess, "run", side_effect=fake_run):
            module.run_panel(
                eligible_panel_path=panel_path,
                experiment_root=experiment_root,
                corpus_b_variant="b1",
                seeds=(0,),
                phases=("setup",),
                dry_run=False,
                limit_candidates=None,
                passthrough_args=[],
            )

        cmd = calls[0]
        idx = cmd.index("--ip-instruction")
        assert cmd[idx + 1] == _CANDIDATE_A["train_user_suffix"]
        idx_id = cmd.index("--ip-instruction-id")
        assert cmd[idx_id + 1] == _CANDIDATE_A["candidate_id"]

    def test_passthrough_args_reach_subprocess(self, tmp_path: Path):
        panel_path = tmp_path / "eligible.json"
        _write_panel(panel_path, [_CANDIDATE_A])
        experiment_root = tmp_path / "root"

        calls: list[list[str]] = []

        def fake_run(cmd, *, check):
            calls.append(list(cmd))

        with patch.object(subprocess, "run", side_effect=fake_run):
            module.run_panel(
                eligible_panel_path=panel_path,
                experiment_root=experiment_root,
                corpus_b_variant="b1",
                seeds=(0,),
                phases=("setup",),
                dry_run=False,
                limit_candidates=None,
                passthrough_args=["--tensor-parallel-size", "8"],
            )

        cmd = calls[0]
        assert "--tensor-parallel-size" in cmd
        pt_idx = cmd.index("--tensor-parallel-size")
        assert cmd[pt_idx + 1] == "8"

    def test_real_run_writes_non_dry_run_manifest(self, tmp_path: Path):
        panel_path = tmp_path / "eligible.json"
        _write_panel(panel_path, [_CANDIDATE_A])
        experiment_root = tmp_path / "root"

        def fake_run(cmd, *, check):
            pass

        with patch.object(subprocess, "run", side_effect=fake_run):
            rc = module.run_panel(
                eligible_panel_path=panel_path,
                experiment_root=experiment_root,
                corpus_b_variant="b1",
                seeds=(0,),
                phases=("setup",),
                dry_run=False,
                limit_candidates=None,
                passthrough_args=[],
            )

        assert rc == 0
        manifest = json.loads(
            (experiment_root / module.PANEL_MANIFEST_NAME).read_text(encoding="utf-8")
        )
        assert manifest["dry_run"] is False
        assert manifest["completed_at"] is not None


# ---------------------------------------------------------------------------
# candidate_experiment_dir
# ---------------------------------------------------------------------------

class TestCandidateExperimentDir:
    def test_dir_structure(self, tmp_path: Path):
        root = tmp_path / "experiments" / "prereg_prompt_panel"
        exp_dir = module.candidate_experiment_dir(root, "b1", _CANDIDATE_A)
        assert exp_dir == root / "b1" / "respond_correct_basic"

    def test_sanitization_applied(self, tmp_path: Path):
        candidate = {**_CANDIDATE_A, "candidate_id": "foo bar baz"}
        exp_dir = module.candidate_experiment_dir(tmp_path, "b1", candidate)
        assert exp_dir.name == "foo_bar_baz"


# ---------------------------------------------------------------------------
# Regression: PROJECTS_DIR and default phases
# ---------------------------------------------------------------------------

class TestProjectsDirAndDefaultPhases:
    def test_projects_dir_is_gcd_sycophancy_projects(self):
        # PROJECTS_DIR must resolve to .../gcd_sycophancy/projects so that
        # _resolve(DEFAULT_ELIGIBLE_PANEL) and _resolve(DEFAULT_EXPERIMENT_ROOT)
        # land under .../gcd_sycophancy/projects/experiments/..., matching the
        # path convention used by run_preregistration.py.
        assert module.PROJECTS_DIR.name == "projects"
        assert module.PROJECTS_DIR.parent.name == "gcd_sycophancy"

    def test_default_phases_include_prefix_search_and_best_elicited_eval(self):
        # analysis in run_preregistration.py calls _require_analysis_inputs(),
        # which checks for frozen prefix artifacts (prefix-search) and
        # best-elicited outputs (best-elicited-eval). Both phases must appear
        # in the default list so a fresh per-candidate experiment does not fail
        # at the analysis step with missing-artifact errors.
        assert "prefix-search" in module.DEFAULT_PHASES
        assert "best-elicited-eval" in module.DEFAULT_PHASES
        assert "analysis" in module.DEFAULT_PHASES
        # phases must be ordered so prerequisites precede analysis
        phases = list(module.DEFAULT_PHASES)
        assert phases.index("prefix-search") < phases.index("analysis")
        assert phases.index("best-elicited-eval") < phases.index("analysis")
