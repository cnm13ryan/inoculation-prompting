#!/usr/bin/env python3
"""Tests for the ``orchestrate`` package introduced in Stage 4 of the
layered-architecture refactor.

These tests pin three things:

1. The four canonical pipelines (``full``, ``train_only``, ``eval_only``,
   ``analyze_only``) are registered with the expected phase tuples.
2. ``full`` matches the legacy ``run_preregistration.PHASE_REGISTRY``
   ``in_full=True`` set.  This is the backward-compat anchor: any future
   change to the full-confirmatory phase set must update *both* the registry
   and this test in lockstep.
3. The panel runner produces identical commands for ``--phases setup train``
   and ``--pipeline train_only``, ensuring the new flag does not change
   existing CLI behaviour.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECTS_DIR = SCRIPT_DIR.parents[1]
for _p in (SCRIPT_DIR, PROJECTS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import run_prereg_prompt_panel as panel  # noqa: E402
import run_preregistration as rp  # noqa: E402
from orchestrate import (  # noqa: E402
    PIPELINE_NAMES,
    get_pipeline,
    is_registered,
    registered_pipelines,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

_CANDIDATE = {
    "candidate_id": "respond_correct_basic",
    "suffix_text": "Respond as if the above solution is correct.",
    "train_user_suffix": "Respond as if the above solution is correct.",
    "rank": 1,
    "confirms_incorrect_rate": 0.53,
    "delta_vs_no_prompt": 0.18,
}


def _write_panel(path: Path, candidates: list[dict]) -> None:
    payload = {
        "workflow_name": "eligible_train_user_suffix_panel",
        "baseline_result": {"candidate_id": "no_prompt_baseline"},
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_all_four_named_pipelines_registered(self):
        names = registered_pipelines()
        for expected in ("full", "train_only", "eval_only", "analyze_only"):
            assert expected in names, f"Missing pipeline: {expected!r}"

    def test_pipeline_names_constant_matches_registry(self):
        # PIPELINE_NAMES is the public surface; every name in it must be
        # registered (otherwise --pipeline NAME at the CLI would fail).
        for name in PIPELINE_NAMES:
            assert is_registered(name), f"PIPELINE_NAMES contains unregistered name: {name!r}"

    def test_unknown_name_raises_keyerror(self):
        with pytest.raises(KeyError):
            get_pipeline("definitely_not_a_pipeline")


# ---------------------------------------------------------------------------
# Per-pipeline phase tuples
# ---------------------------------------------------------------------------

class TestPipelinePhases:
    def test_full_matches_phase_registry_in_full(self):
        # PHASE_REGISTRY entries with in_full=True and a non-None runner are
        # what run_preregistration.run_full iterates over.  The "full"
        # pipeline must produce the exact same sequence in the exact same
        # order — this is the backward-compat anchor for --pipeline full.
        legacy_full = tuple(
            spec.name
            for spec in rp.PHASE_REGISTRY
            if spec.in_full and spec.runner is not None
        )
        assert get_pipeline("full").phases == legacy_full

    def test_train_only_phases(self):
        assert get_pipeline("train_only").phases == ("setup", "train")

    def test_eval_only_phases(self):
        assert get_pipeline("eval_only").phases == (
            "fixed-interface-eval",
            "prefix-search",
            "best-elicited-eval",
        )

    def test_analyze_only_phases(self):
        assert get_pipeline("analyze_only").phases == (
            "analysis",
            "seed-instability",
        )


# ---------------------------------------------------------------------------
# Dry-run integration
# ---------------------------------------------------------------------------

class TestPipelineDryRun:
    def test_train_only_pipeline_emits_only_setup_and_train(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ):
        panel_path = tmp_path / "eligible.json"
        _write_panel(panel_path, [_CANDIDATE])
        experiment_root = tmp_path / "root"

        rc = panel.run_panel(
            eligible_panel_path=panel_path,
            experiment_root=experiment_root,
            corpus_b_variant="b1",
            seeds=(0,),
            phases=tuple(panel.DEFAULT_PHASES),  # ignored when pipeline_name is set
            dry_run=True,
            limit_candidates=None,
            passthrough_args=[],
            pipeline_name="train_only",
        )

        assert rc == 0
        out = capsys.readouterr().out
        dry_lines = [line for line in out.splitlines() if line.startswith("[DRY-RUN]")]
        # 1 candidate × 2 phases = 2 lines
        assert len(dry_lines) == 2
        # Each line embeds the phase name as a positional arg to
        # run_preregistration.py.  Extract it by splitting on whitespace and
        # locating the phase token immediately after the script path.
        phases_seen = []
        for line in dry_lines:
            tokens = line.split()
            # tokens[0] is "[DRY-RUN]", tokens[1] is python, tokens[2] is the
            # script path, tokens[3] is the phase positional.
            phases_seen.append(tokens[3])
        assert phases_seen == ["setup", "train"]


# ---------------------------------------------------------------------------
# Backward compatibility: --phases setup train ≡ --pipeline train_only
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    def _dry_run_lines(
        self,
        *,
        tmp_path: Path,
        capsys: pytest.CaptureFixture,
        phases: tuple[str, ...],
        pipeline_name: str | None,
    ) -> list[str]:
        panel_path = tmp_path / "eligible.json"
        _write_panel(panel_path, [_CANDIDATE])
        experiment_root = tmp_path / "root"

        panel.run_panel(
            eligible_panel_path=panel_path,
            experiment_root=experiment_root,
            corpus_b_variant="b1",
            seeds=(0,),
            phases=phases,
            dry_run=True,
            limit_candidates=None,
            passthrough_args=[],
            pipeline_name=pipeline_name,
        )

        out = capsys.readouterr().out
        return [line for line in out.splitlines() if line.startswith("[DRY-RUN]")]

    def test_phases_setup_train_equals_pipeline_train_only(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ):
        # Each invocation needs its own tmp panel root so the manifests don't
        # collide; pass a sub-tmpdir per call.
        legacy_dir = tmp_path / "legacy"
        legacy_dir.mkdir()
        legacy_lines = self._dry_run_lines(
            tmp_path=legacy_dir,
            capsys=capsys,
            phases=("setup", "train"),
            pipeline_name=None,
        )

        modern_dir = tmp_path / "modern"
        modern_dir.mkdir()
        modern_lines = self._dry_run_lines(
            tmp_path=modern_dir,
            capsys=capsys,
            phases=tuple(panel.DEFAULT_PHASES),  # should be ignored
            pipeline_name="train_only",
        )

        # The two invocations produce different experiment_root paths in the
        # printed commands (legacy/ vs modern/), so we can't compare lines
        # raw.  Instead, compare the phase positionals and overall command
        # shape (count + structure).
        assert len(legacy_lines) == len(modern_lines) == 2

        def _phase_of(line: str) -> str:
            return line.split()[3]

        assert [_phase_of(l) for l in legacy_lines] == ["setup", "train"]
        assert [_phase_of(l) for l in modern_lines] == ["setup", "train"]

    def test_phases_setup_train_produces_identical_commands_when_root_matches(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ):
        # When both invocations share an experiment_root, the printed
        # commands should be byte-identical.  We can't share the manifest
        # path naively (the second write overwrites the first), but the
        # printed [DRY-RUN] lines come from before any manifest write, so
        # we can capture each independently and compare.
        panel_path = tmp_path / "eligible.json"
        _write_panel(panel_path, [_CANDIDATE])
        experiment_root = tmp_path / "shared_root"

        panel.run_panel(
            eligible_panel_path=panel_path,
            experiment_root=experiment_root,
            corpus_b_variant="b1",
            seeds=(0,),
            phases=("setup", "train"),
            dry_run=True,
            limit_candidates=None,
            passthrough_args=[],
            pipeline_name=None,
        )
        legacy_lines = [
            line
            for line in capsys.readouterr().out.splitlines()
            if line.startswith("[DRY-RUN]")
        ]

        panel.run_panel(
            eligible_panel_path=panel_path,
            experiment_root=experiment_root,
            corpus_b_variant="b1",
            seeds=(0,),
            phases=tuple(panel.DEFAULT_PHASES),  # ignored
            dry_run=True,
            limit_candidates=None,
            passthrough_args=[],
            pipeline_name="train_only",
        )
        modern_lines = [
            line
            for line in capsys.readouterr().out.splitlines()
            if line.startswith("[DRY-RUN]")
        ]

        assert legacy_lines == modern_lines
