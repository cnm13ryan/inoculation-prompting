"""Tests for the construct-validity gate aggregator (WT-15)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import construct_validity_gates as cvg


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _prereg_analysis(*, h1="supported", h1c="supported", h2="supported",
                     schema_status="supported") -> dict:
    def _entry(hyp_id: str, status: str) -> dict:
        return {
            "analysis_id": f"analysis_{hyp_id}",
            "hypothesis_id": hyp_id,
            "support_status": status,
            "marginal_risk_difference": -0.05,
            "decision_interval": [-0.1, 0.0],
        }
    return {
        "workflow_name": "preregistered_section_7_analysis",
        "confirmatory_results": [
            _entry("H1", h1),
            _entry("H1c", h1c),
            _entry("H2", h2),
        ],
        "construct_validity_interpretation": {
            "summary": "construct-validity narrative",
            "h1_supported": h1 == "supported",
            "h1c_supported": h1c == "supported",
            "h1c_available": True,
            "h2_supported": h2 == "supported",
        },
        "schema_invariance": {"status": schema_status, "label": "schema-invariance"},
    }


def _prompt_panel(prop=0.8, n_present=5, n_supported=4) -> dict:
    return {
        "panel_summary": {
            "n_candidates_total": n_present,
            "n_candidates_present": n_present,
            "n_candidates_h1_supported": n_supported,
            "proportion_h1_supported": prop,
        },
    }


def _corpus_matrix(b1="supported", b2="supported") -> dict:
    def _vr(status):
        return {
            "status": "present",
            "key_metrics": {"h1_support_status": status},
        }
    return {
        "variants": ["b1", "b2"],
        "variant_results": {"b1": _vr(b1), "b2": _vr(b2)},
    }


def _scorer(agreement=0.9, kappa=0.75) -> dict:
    return {"agreement": agreement, "cohens_kappa": kappa}


def _construct_labels(agreement=0.85) -> dict:
    return {"agreement": agreement}


def _contamination(overlap=0) -> dict:
    return {
        "overlaps": {
            "latent_pair": {"n_overlap": overlap},
            "exact_user_prompt": {"n_overlap": 0},
            "normalized_user_prompt": {"n_overlap": 0},
            "assistant_target": {"n_overlap": 0},
        }
    }


def _power(p_h1=0.95, p_h2=0.92, p_joint=0.88) -> dict:
    return {
        "power_h1": p_h1,
        "prob_h2_noninferiority_pass": p_h2,
        "joint_success_prob": p_joint,
    }


def _exclusion(rds=(-0.05, -0.04, -0.06)) -> dict:
    return {
        "scenarios": [
            {"scenario": f"s{i}", "risk_difference_treatment_minus_control": v}
            for i, v in enumerate(rds)
        ],
    }


def _model_matrix(supporting=("m1", "m2", "m3")) -> dict:
    results = {
        m: {
            "status": "present",
            "key_metrics": {
                "h1_support_status": "supported" if m in supporting else "unsupported"
            },
        }
        for m in ("m1", "m2", "m3")
    }
    return {"models": list(results.keys()), "model_results": results}


def _epoch_matrix(supporting=("ep1", "ep2")) -> dict:
    results = {
        e: {
            "status": "present",
            "key_metrics": {
                "h1_support_status": "supported" if e in supporting else "unsupported"
            },
        }
        for e in ("ep1", "ep2")
    }
    return {"variant_results": results}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


def _full_paths(tmp_path: Path) -> cvg.InputPaths:
    return cvg.InputPaths(
        prereg_analysis=tmp_path / "prereg_analysis.json",
        prereg_problem_level_csv=None,
        exclusion_sensitivity=tmp_path / "exclusion_sensitivity.json",
        item_difficulty=None,
        contamination_audit=tmp_path / "contamination_audit.json",
        power_analysis=tmp_path / "power_analysis.json",
        response_scorer_agreement=tmp_path / "response_scorer_agreement.json",
        prompt_construct_label_agreement=tmp_path / "prompt_construct_label_agreement.json",
        prompt_panel_summary=tmp_path / "prompt_panel_summary.json",
        corpus_matrix_summary=tmp_path / "corpus_matrix_summary.json",
        model_matrix_summary=tmp_path / "model_matrix_summary.json",
        epoch_matrix_summary=tmp_path / "epoch_matrix_summary.json",
    )


def _populate_all_pass(paths: cvg.InputPaths) -> None:
    _write(paths.prereg_analysis, _prereg_analysis())
    _write(paths.exclusion_sensitivity, _exclusion())
    _write(paths.contamination_audit, _contamination())
    _write(paths.power_analysis, _power())
    _write(paths.response_scorer_agreement, _scorer())
    _write(paths.prompt_construct_label_agreement, _construct_labels())
    _write(paths.prompt_panel_summary, _prompt_panel())
    _write(paths.corpus_matrix_summary, _corpus_matrix())
    _write(paths.model_matrix_summary, _model_matrix())
    _write(paths.epoch_matrix_summary, _epoch_matrix())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_all_gates_pass_yields_very_strong(tmp_path):
    paths = _full_paths(tmp_path)
    _populate_all_pass(paths)
    gates = cvg.evaluate_gates(paths, cvg.Thresholds())
    classification = cvg.classify(gates)

    statuses = {g.gate_name: g.status for g in gates}
    assert all(s == "pass" for s in statuses.values()), statuses
    assert classification["overall_classification"] == "very_strong_construct_validity"


def test_gcd_pass_multidomain_unavailable_yields_strong_gcd_only(tmp_path):
    paths = _full_paths(tmp_path)
    _populate_all_pass(paths)
    paths.model_matrix_summary.unlink()
    paths.epoch_matrix_summary.unlink()

    gates = cvg.evaluate_gates(paths, cvg.Thresholds())
    classification = cvg.classify(gates)
    statuses = {g.gate_name: g.status for g in gates}

    for name in cvg.GCD_ONLY_GATES:
        assert statuses[name] == "pass", (name, statuses)
    assert statuses["model_matrix_gate"] == "unavailable"
    assert statuses["epoch_matrix_gate"] == "unavailable"
    assert classification["overall_classification"] == "strong_gcd_only_construct_validity"


def test_mixed_results_with_critical_failure_yields_weak(tmp_path):
    paths = _full_paths(tmp_path)
    _populate_all_pass(paths)
    # Break the contamination critical gate.
    _write(paths.contamination_audit, {
        "overlaps": {
            "latent_pair": {"n_overlap": 7},
            "exact_user_prompt": {"n_overlap": 0},
            "normalized_user_prompt": {"n_overlap": 0},
            "assistant_target": {"n_overlap": 0},
        }
    })

    gates = cvg.evaluate_gates(paths, cvg.Thresholds())
    classification = cvg.classify(gates)
    by_name = {g.gate_name: g for g in gates}
    assert by_name["contamination_gate"].status == "fail"
    assert classification["overall_classification"] == "weak_or_inconclusive_construct_validity"
    assert "contamination_gate" in classification["rationale"]


def test_non_critical_failure_yields_moderate(tmp_path):
    paths = _full_paths(tmp_path)
    _populate_all_pass(paths)
    # Break a non-critical gate (prompt panel).
    _write(paths.prompt_panel_summary, _prompt_panel(prop=0.1, n_supported=1))

    gates = cvg.evaluate_gates(paths, cvg.Thresholds())
    classification = cvg.classify(gates)
    statuses = {g.gate_name: g.status for g in gates}
    assert statuses["prompt_panel_gate"] == "fail"
    for name in cvg.CRITICAL_GATES:
        assert statuses[name] == "pass"
    assert classification["overall_classification"] == "moderate_construct_validity"


def test_all_inputs_missing_yields_unavailable(tmp_path):
    paths = cvg.InputPaths(
        prereg_analysis=tmp_path / "missing.json",
        prereg_problem_level_csv=None,
        exclusion_sensitivity=tmp_path / "x.json",
        item_difficulty=None,
        contamination_audit=tmp_path / "x.json",
        power_analysis=tmp_path / "x.json",
        response_scorer_agreement=tmp_path / "x.json",
        prompt_construct_label_agreement=tmp_path / "x.json",
        prompt_panel_summary=tmp_path / "x.json",
        corpus_matrix_summary=tmp_path / "x.json",
        model_matrix_summary=tmp_path / "x.json",
        epoch_matrix_summary=tmp_path / "x.json",
    )
    gates = cvg.evaluate_gates(paths, cvg.Thresholds())
    classification = cvg.classify(gates)
    assert all(g.status == "unavailable" for g in gates)
    assert classification["overall_classification"] == "unavailable"


def test_warning_on_critical_gate_classifies_weak_no_crash(tmp_path):
    """H1 supported but H1c not -> conditional_sycophancy_gate warning (critical)."""
    paths = _full_paths(tmp_path)
    _populate_all_pass(paths)
    _write(paths.prereg_analysis, _prereg_analysis(h1c="unsupported"))

    gates = cvg.evaluate_gates(paths, cvg.Thresholds())
    classification = cvg.classify(gates)
    statuses = {g.gate_name: g.status for g in gates}
    assert statuses["conditional_sycophancy_gate"] == "warning"
    assert classification["overall_classification"] == "weak_or_inconclusive_construct_validity"


def test_warning_on_non_critical_gate_classifies_moderate_no_crash(tmp_path):
    """B1 supports H1 but B2 doesn't -> b1_b2_consistency_gate warning."""
    paths = _full_paths(tmp_path)
    _populate_all_pass(paths)
    _write(paths.corpus_matrix_summary, _corpus_matrix(b1="supported", b2="unsupported"))

    gates = cvg.evaluate_gates(paths, cvg.Thresholds())
    classification = cvg.classify(gates)
    statuses = {g.gate_name: g.status for g in gates}
    assert statuses["b1_b2_consistency_gate"] == "warning"
    assert classification["overall_classification"] == "moderate_construct_validity"


def test_cli_writes_outputs(tmp_path):
    """End-to-end: CLI produces JSON+MD with provenance."""
    experiment_dir = tmp_path / "exp"
    reports = experiment_dir / "reports"
    validation = experiment_dir / "validation"
    reports.mkdir(parents=True)
    validation.mkdir(parents=True)
    _write(reports / "prereg_analysis.json", _prereg_analysis())
    _write(reports / "contamination_audit.json", _contamination())

    out_prefix = reports / "construct_validity"
    rc = cvg.main([
        "--experiment-dir", str(experiment_dir),
        "--output-prefix", str(out_prefix),
    ])
    assert rc == 0
    assert (reports / "construct_validity.json").exists()
    assert (reports / "construct_validity.md").exists()
    payload = json.loads((reports / "construct_validity.json").read_text())
    assert "provenance" in payload
    assert "overall_classification" in payload
    assert payload["workflow_name"] == "construct_validity_gates"


def test_matrix_gates_read_sycophancy_support_status_key(tmp_path):
    """run_prereg_{model,epoch,corpus}_matrix emit the key as
    'sycophancy_support_status'; gates must recognize it."""
    paths = _full_paths(tmp_path)
    _populate_all_pass(paths)

    _write(paths.model_matrix_summary, {
        "models": ["m1", "m2"],
        "model_results": {
            "m1": {"status": "present",
                   "key_metrics": {"sycophancy_support_status": "supported"}},
            "m2": {"status": "present",
                   "key_metrics": {"sycophancy_support_status": "supported"}},
        },
    })
    _write(paths.epoch_matrix_summary, {
        "variant_results": {
            "ep1": {"status": "present",
                    "key_metrics": {"sycophancy_support_status": "supported"}},
            "ep2": {"status": "present",
                    "key_metrics": {"sycophancy_support_status": "unsupported"}},
        },
    })
    _write(paths.corpus_matrix_summary, {
        "variants": ["b1", "b2"],
        "variant_results": {
            "b1": {"status": "present",
                   "key_metrics": {"sycophancy_support_status": "supported"}},
            "b2": {"status": "present",
                   "key_metrics": {"sycophancy_support_status": "supported"}},
        },
    })

    gates = cvg.evaluate_gates(paths, cvg.Thresholds())
    by_name = {g.gate_name: g for g in gates}
    assert by_name["model_matrix_gate"].status == "pass", by_name["model_matrix_gate"]
    assert by_name["epoch_matrix_gate"].status == "pass", by_name["epoch_matrix_gate"]
    assert by_name["b1_b2_consistency_gate"].status == "pass", by_name["b1_b2_consistency_gate"]


def test_threshold_override_changes_pass_to_fail(tmp_path):
    paths = _full_paths(tmp_path)
    _populate_all_pass(paths)
    t = cvg.Thresholds(prompt_panel_min_supported_proportion=0.95)
    gates = cvg.evaluate_gates(paths, t)
    by_name = {g.gate_name: g for g in gates}
    assert by_name["prompt_panel_gate"].status == "fail"
