"""Tests for analyze_prompt_panel_effects (WT-6)."""
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

import analyze_prompt_panel_effects as module


def _make_analysis(
    *,
    h1_status: str = "supported",
    h1_mrd: float = -0.20,
    include_h1c: bool = True,
    h2_status: str = "supported",
    schema_status: str | None = "pass",
    joint_success: bool = True,
) -> dict:
    confirmatory = [
        {
            "analysis_id": "analysis_1",
            "hypothesis_id": "H1",
            "label": "Sycophancy reduction",
            "support_status": h1_status,
            "marginal_risk_difference": h1_mrd,
            "arm_log_odds_coefficient": -1.2,
            "evaluation_set_name": "confirmatory",
            "n_rows": 1000,
        },
        {
            "analysis_id": "analysis_2",
            "hypothesis_id": "H2",
            "label": "Capability preservation",
            "support_status": h2_status,
            "marginal_risk_difference": -0.01,
            "evaluation_set_name": "confirmatory",
            "n_rows": 800,
        },
    ]
    if include_h1c:
        confirmatory.append({
            "analysis_id": "analysis_1c",
            "hypothesis_id": "H1c",
            "label": "Conditional sycophancy",
            "support_status": "supported",
            "marginal_risk_difference": -0.18,
            "evaluation_set_name": "confirmatory",
            "n_rows": 700,
        })
    out = {
        "confirmatory_results": confirmatory,
        "joint_interpretation": {
            "joint_success": joint_success,
            "summary": "joint summary",
        },
    }
    if schema_status is not None:
        out["schema_invariance"] = {
            "status": schema_status,
            "label": "Schema invariance",
            "note": "n",
        }
    return out


def _write_candidate_analysis(
    panel_root: Path,
    variant: str,
    candidate_id: str,
    analysis: dict,
) -> Path:
    cdir = panel_root / variant / candidate_id
    apath = cdir / "reports" / "prereg_analysis.json"
    apath.parent.mkdir(parents=True, exist_ok=True)
    apath.write_text(json.dumps(analysis), encoding="utf-8")
    return cdir


def _write_eligible_panel(path: Path, candidates: list[dict]) -> None:
    payload = {
        "workflow_name": "eligible_train_user_suffix_panel",
        "all_candidate_results": candidates,
        "eligible_candidate_results": candidates,
        "ineligible_candidate_results": [],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class TestDiscoverCandidateDirs:
    def test_discovers_variant_candidate_pairs(self, tmp_path: Path):
        _write_candidate_analysis(tmp_path, "b1", "cand_a", _make_analysis())
        _write_candidate_analysis(tmp_path, "b1", "cand_b", _make_analysis())
        _write_candidate_analysis(tmp_path, "b2", "cand_a", _make_analysis())
        triples = module.discover_candidate_dirs(tmp_path)
        ids = {(v, c) for (v, c, _) in triples}
        assert ids == {("b1", "cand_a"), ("b1", "cand_b"), ("b2", "cand_a")}

    def test_empty_root_returns_empty(self, tmp_path: Path):
        assert module.discover_candidate_dirs(tmp_path / "missing") == []


class TestSummarizeCandidate:
    def test_present_candidate_extracts_fields(self, tmp_path: Path):
        cdir = _write_candidate_analysis(tmp_path, "b1", "cand_a", _make_analysis())
        result = module.summarize_candidate(
            corpus_b_variant="b1",
            candidate_id="cand_a",
            candidate_dir=cdir,
            eligible_panel_by_id={
                "cand_a": {"candidate_id": "cand_a", "delta_vs_no_prompt": 0.30, "rank": 1}
            },
        )
        assert result["status"] == "present"
        assert result["delta_vs_no_prompt"] == 0.30
        assert result["candidate_rank"] == 1
        assert result["h1"]["support_status"] == "supported"
        assert result["h1"]["marginal_risk_difference"] == -0.20
        assert result["h1c"]["support_status"] == "supported"
        assert result["h2"]["support_status"] == "supported"
        assert result["schema_invariance"]["status"] == "pass"
        assert result["joint"]["joint_success"] is True

    def test_missing_analysis_marked_status(self, tmp_path: Path):
        cdir = tmp_path / "b1" / "cand_x"
        cdir.mkdir(parents=True)
        result = module.summarize_candidate(
            corpus_b_variant="b1",
            candidate_id="cand_x",
            candidate_dir=cdir,
            eligible_panel_by_id={},
        )
        assert result["status"] == "missing"
        assert result["h1"] is None

    def test_missing_h1c_tolerated(self, tmp_path: Path):
        cdir = _write_candidate_analysis(
            tmp_path, "b1", "cand_a", _make_analysis(include_h1c=False)
        )
        result = module.summarize_candidate(
            corpus_b_variant="b1",
            candidate_id="cand_a",
            candidate_dir=cdir,
            eligible_panel_by_id={},
        )
        assert result["status"] == "present"
        assert result["h1c"] is None
        assert result["h1"] is not None

    def test_missing_schema_invariance_tolerated(self, tmp_path: Path):
        cdir = _write_candidate_analysis(
            tmp_path, "b1", "cand_a", _make_analysis(schema_status=None)
        )
        result = module.summarize_candidate(
            corpus_b_variant="b1",
            candidate_id="cand_a",
            candidate_dir=cdir,
            eligible_panel_by_id={},
        )
        assert result["status"] == "present"
        assert result["schema_invariance"] is None


class TestPanelSummary:
    def test_counts_supported_proportion(self, tmp_path: Path):
        _write_candidate_analysis(
            tmp_path, "b1", "cand_a",
            _make_analysis(h1_status="supported", h1_mrd=-0.30),
        )
        _write_candidate_analysis(
            tmp_path, "b1", "cand_b",
            _make_analysis(h1_status="unsupported", h1_mrd=-0.05),
        )
        _write_candidate_analysis(
            tmp_path, "b1", "cand_c",
            _make_analysis(h1_status="supported", h1_mrd=-0.10),
        )
        payload = module.build_prompt_panel_payload(panel_root=tmp_path)
        s = payload["panel_summary"]
        assert s["n_candidates_total"] == 3
        assert s["n_candidates_present"] == 3
        assert s["n_candidates_h1_supported"] == 2
        assert s["proportion_h1_supported"] == pytest.approx(2 / 3)
        dist = s["h1_effect_size_distribution"]
        assert dist["n"] == 3
        assert dist["min"] == pytest.approx(-0.30)
        assert dist["max"] == pytest.approx(-0.05)

    def test_missing_outputs_counted_separately(self, tmp_path: Path):
        _write_candidate_analysis(tmp_path, "b1", "cand_a", _make_analysis())
        # cand_b directory exists but no analysis
        (tmp_path / "b1" / "cand_b").mkdir(parents=True)
        payload = module.build_prompt_panel_payload(panel_root=tmp_path)
        s = payload["panel_summary"]
        assert s["n_candidates_total"] == 2
        assert s["n_candidates_present"] == 1
        assert s["n_candidates_missing"] == 1


class TestStrictMode:
    def test_strict_raises_on_missing(self, tmp_path: Path):
        _write_candidate_analysis(tmp_path, "b1", "cand_a", _make_analysis())
        (tmp_path / "b1" / "cand_b").mkdir(parents=True)  # missing analysis
        with pytest.raises(SystemExit):
            module.build_prompt_panel_payload(panel_root=tmp_path, strict=True)

    def test_strict_passes_when_all_present(self, tmp_path: Path):
        _write_candidate_analysis(tmp_path, "b1", "cand_a", _make_analysis())
        _write_candidate_analysis(tmp_path, "b1", "cand_b", _make_analysis())
        # Should not raise.
        payload = module.build_prompt_panel_payload(panel_root=tmp_path, strict=True)
        assert payload["panel_summary"]["n_candidates_present"] == 2


class TestEligiblePanelMerge:
    def test_delta_and_rank_attached(self, tmp_path: Path):
        _write_candidate_analysis(tmp_path, "b1", "cand_a", _make_analysis())
        _write_candidate_analysis(tmp_path, "b1", "cand_b", _make_analysis())
        eligible = tmp_path / "eligible.json"
        _write_eligible_panel(
            eligible,
            [
                {"candidate_id": "cand_a", "delta_vs_no_prompt": 0.30, "rank": 1},
                {"candidate_id": "cand_b", "delta_vs_no_prompt": 0.10, "rank": 2},
            ],
        )
        payload = module.build_prompt_panel_payload(
            panel_root=tmp_path, eligible_panel_path=eligible
        )
        by_id = {c["candidate_id"]: c for c in payload["candidate_summaries"]}
        assert by_id["cand_a"]["delta_vs_no_prompt"] == 0.30
        assert by_id["cand_a"]["candidate_rank"] == 1
        assert by_id["cand_b"]["candidate_rank"] == 2

    def test_eligible_path_missing_tolerated(self, tmp_path: Path):
        _write_candidate_analysis(tmp_path, "b1", "cand_a", _make_analysis())
        payload = module.build_prompt_panel_payload(
            panel_root=tmp_path,
            eligible_panel_path=tmp_path / "does_not_exist.json",
        )
        assert payload["candidate_summaries"][0]["delta_vs_no_prompt"] is None


class TestOutputs:
    def test_write_outputs_creates_json_and_md_with_provenance(self, tmp_path: Path):
        _write_candidate_analysis(tmp_path, "b1", "cand_a", _make_analysis())
        payload = module.build_prompt_panel_payload(panel_root=tmp_path)
        prefix = tmp_path / "out" / "prompt_panel_summary"
        paths = module.write_outputs(payload, prefix, input_paths=[], argv=["x"])
        assert paths["json"].exists()
        assert paths["md"].exists()
        doc = json.loads(paths["json"].read_text())
        assert "provenance" in doc
        assert "panel_summary" in doc
        # MD includes per-candidate row
        md = paths["md"].read_text()
        assert "cand_a" in md


class TestCLI:
    def test_main_end_to_end(self, tmp_path: Path):
        panel_root = tmp_path / "panel"
        _write_candidate_analysis(panel_root, "b1", "cand_a", _make_analysis())
        _write_candidate_analysis(
            panel_root, "b1", "cand_b",
            _make_analysis(h1_status="unsupported", h1_mrd=-0.01),
        )
        eligible = tmp_path / "eligible.json"
        _write_eligible_panel(
            eligible,
            [{"candidate_id": "cand_a", "delta_vs_no_prompt": 0.5, "rank": 1}],
        )
        prefix = tmp_path / "out" / "prompt_panel_summary"
        rc = module.main([
            "--panel-root", str(panel_root),
            "--eligible-panel", str(eligible),
            "--output-prefix", str(prefix),
        ])
        assert rc == 0
        doc = json.loads((prefix.parent / "prompt_panel_summary.json").read_text())
        assert doc["panel_summary"]["n_candidates_total"] == 2
        assert doc["panel_summary"]["n_candidates_h1_supported"] == 1
