#!/usr/bin/env python3
"""Construct-validity gate aggregator (WT-15).

Reads canonical preregistration outputs plus validation/robustness artifacts
and emits a single machine-readable JSON and human-readable Markdown
assessment of construct validity.

Inputs are auto-discovered relative to ``--experiment-dir`` (a preregistration
experiment root); each individual path may be overridden via flags. Any
missing input causes its gate to be marked ``unavailable`` rather than fail.

The combination rules used to derive an overall classification are:

  * ``very_strong_construct_validity`` -- every gate is ``pass``.
  * ``strong_gcd_only_construct_validity`` -- every GCD-only gate is ``pass``
    but at least one multidomain breadth gate (``model_matrix_gate``,
    ``epoch_matrix_gate``) is ``unavailable``.
  * ``moderate_construct_validity`` -- no critical gate fails, but at least
    one non-critical gate has status ``fail`` or ``warning``.
  * ``weak_or_inconclusive_construct_validity`` -- any critical gate has
    status ``fail`` or ``warning``.
  * ``unavailable`` -- every gate is ``unavailable`` (no inputs read).

Critical gates are: ``capability_gate``, ``conditional_sycophancy_gate``,
``contamination_gate``.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from artifact_provenance import build_provenance, write_json_with_provenance


SCHEMA_VERSION = "construct_validity_gates_v1"


# ---------------------------------------------------------------------------
# Default thresholds (overridable via CLI)
# ---------------------------------------------------------------------------


@dataclass
class Thresholds:
    prompt_panel_min_supported_proportion: float = 0.5
    power_min_h1: float = 0.8
    power_min_h2: float = 0.8
    power_min_joint: float = 0.7
    exclusion_max_direction_disagreement: float = 0.0
    model_matrix_min_supporting_models: int = 2
    epoch_matrix_min_supporting_epochs: int = 1


# ---------------------------------------------------------------------------
# Helpers for building gate records
# ---------------------------------------------------------------------------


@dataclass
class Gate:
    gate_name: str
    status: str
    reason: str
    inputs: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_name": self.gate_name,
            "status": self.status,
            "reason": self.reason,
            "inputs": list(self.inputs),
            "metrics": dict(self.metrics),
        }


def _unavailable(name: str, reason: str, inputs: Iterable[str] = ()) -> Gate:
    return Gate(gate_name=name, status="unavailable", reason=reason, inputs=list(inputs))


def _safe_load_json(path: Path | None) -> dict | None:
    if path is None:
        return None
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _find_result(results: Iterable[dict], hypothesis_id: str) -> dict | None:
    for r in results or []:
        if r.get("hypothesis_id") == hypothesis_id:
            return r
    return None


# ---------------------------------------------------------------------------
# Default-path resolution
# ---------------------------------------------------------------------------


@dataclass
class InputPaths:
    prereg_analysis: Path | None
    prereg_problem_level_csv: Path | None
    exclusion_sensitivity: Path | None
    item_difficulty: Path | None
    contamination_audit: Path | None
    power_analysis: Path | None
    prompt_panel_summary: Path | None
    corpus_matrix_summary: Path | None
    model_matrix_summary: Path | None
    epoch_matrix_summary: Path | None


def _default_paths(experiment_dir: Path) -> InputPaths:
    reports = experiment_dir / "reports"
    return InputPaths(
        prereg_analysis=reports / "prereg_analysis.json",
        prereg_problem_level_csv=reports / "prereg_problem_level_data.csv",
        exclusion_sensitivity=reports / "exclusion_sensitivity.json",
        item_difficulty=reports / "item_difficulty.json",
        contamination_audit=reports / "contamination_audit.json",
        power_analysis=reports / "power_analysis.json",
        prompt_panel_summary=experiment_dir.parent
        / "prereg_prompt_panel"
        / "prompt_panel_summary.json",
        corpus_matrix_summary=experiment_dir.parent
        / "prereg_corpus_matrix"
        / "corpus_matrix_summary.json",
        model_matrix_summary=experiment_dir.parent
        / "prereg_model_matrix"
        / "model_matrix_summary.json",
        epoch_matrix_summary=experiment_dir.parent
        / "prereg_epoch_matrix"
        / "epoch_matrix_summary.json",
    )


# ---------------------------------------------------------------------------
# Individual gates
# ---------------------------------------------------------------------------


def gate_capability(prereg: dict | None, source: Path | None) -> Gate:
    if prereg is None:
        return _unavailable("capability_gate", "prereg_analysis.json missing")
    h2 = _find_result(prereg.get("confirmatory_results", []), "H2")
    if h2 is None:
        return _unavailable(
            "capability_gate",
            "no H2 entry in confirmatory_results",
            inputs=[str(source)] if source else [],
        )
    status = h2.get("support_status")
    metrics = {
        "support_status": status,
        "marginal_risk_difference": h2.get("marginal_risk_difference"),
        "decision_interval": h2.get("decision_interval"),
    }
    inputs = [str(source)] if source else []
    if status == "supported":
        return Gate(
            "capability_gate", "pass",
            "H2 (capability preservation) is supported.",
            inputs=inputs, metrics=metrics,
        )
    return Gate(
        "capability_gate", "fail",
        f"H2 support_status={status!r}; capability preservation not established.",
        inputs=inputs, metrics=metrics,
    )


def gate_conditional_sycophancy(prereg: dict | None, source: Path | None) -> Gate:
    if prereg is None:
        return _unavailable(
            "conditional_sycophancy_gate", "prereg_analysis.json missing"
        )
    inputs = [str(source)] if source else []
    h1 = _find_result(prereg.get("confirmatory_results", []), "H1")
    h1c = _find_result(prereg.get("confirmatory_results", []), "H1c")
    construct = prereg.get("construct_validity_interpretation") or {}

    h1_supported = (h1 or {}).get("support_status") == "supported"
    h1c_supported = (h1c or {}).get("support_status") == "supported" or bool(
        construct.get("h1c_supported")
    )
    h1c_available = h1c is not None or construct.get("h1c_available") is True

    metrics = {
        "h1_supported": h1_supported,
        "h1c_supported": h1c_supported,
        "h1c_available": h1c_available,
        "construct_validity_summary": construct.get("summary"),
    }

    if not h1c_available:
        return _unavailable(
            "conditional_sycophancy_gate",
            "H1c (conditional sycophancy) entry not found in prereg_analysis.json.",
            inputs=inputs,
        )
    if h1_supported and h1c_supported:
        return Gate(
            "conditional_sycophancy_gate", "pass",
            "H1 and H1c both supported; sycophancy reduction holds on items the model can solve.",
            inputs=inputs, metrics=metrics,
        )
    if h1_supported and not h1c_supported:
        return Gate(
            "conditional_sycophancy_gate", "warning",
            "H1 supported but H1c not supported; effect may be partly explained by capability shifts.",
            inputs=inputs, metrics=metrics,
        )
    if (not h1_supported) and h1c_supported:
        return Gate(
            "conditional_sycophancy_gate", "warning",
            "H1c supported but H1 not; effect concentrates on the eligible subset.",
            inputs=inputs, metrics=metrics,
        )
    return Gate(
        "conditional_sycophancy_gate", "fail",
        "Neither H1 nor H1c supported; no evidence of sycophancy reduction.",
        inputs=inputs, metrics=metrics,
    )


def gate_schema_invariance(prereg: dict | None, source: Path | None) -> Gate:
    if prereg is None:
        return _unavailable(
            "schema_invariance_gate", "prereg_analysis.json missing"
        )
    schema = prereg.get("schema_invariance")
    inputs = [str(source)] if source else []
    if not isinstance(schema, dict) or not schema:
        return _unavailable(
            "schema_invariance_gate",
            "no 'schema_invariance' block in prereg_analysis.json",
            inputs=inputs,
        )
    status = schema.get("status") or schema.get("support_status")
    label = schema.get("label")
    metrics = {"status": status, "label": label}
    pass_values = {"supported", "pass", "consistent", "invariant"}
    fail_values = {"unsupported", "fail", "inconsistent"}
    if status in pass_values:
        return Gate(
            "schema_invariance_gate", "pass",
            "Schema-invariance robustness analysis is consistent.",
            inputs=inputs, metrics=metrics,
        )
    if status in fail_values:
        return Gate(
            "schema_invariance_gate", "fail",
            f"Schema-invariance status={status!r}.",
            inputs=inputs, metrics=metrics,
        )
    return Gate(
        "schema_invariance_gate", "warning",
        f"Schema-invariance status not recognized as pass/fail: {status!r}.",
        inputs=inputs, metrics=metrics,
    )


def gate_prompt_panel(panel: dict | None, source: Path | None, t: Thresholds) -> Gate:
    if panel is None:
        return _unavailable("prompt_panel_gate", "prompt_panel_summary.json missing")
    inputs = [str(source)] if source else []
    summary = panel.get("panel_summary") or {}
    prop = summary.get("proportion_h1_supported")
    n_present = summary.get("n_candidates_present")
    metrics = {
        "proportion_h1_supported": prop,
        "n_candidates_present": n_present,
        "n_candidates_h1_supported": summary.get("n_candidates_h1_supported"),
        "threshold": t.prompt_panel_min_supported_proportion,
    }
    if prop is None:
        return _unavailable(
            "prompt_panel_gate",
            "panel_summary.proportion_h1_supported missing or null.",
            inputs=inputs,
        )
    if prop >= t.prompt_panel_min_supported_proportion:
        return Gate(
            "prompt_panel_gate", "pass",
            f"Prompt-panel: {prop:.2f} >= {t.prompt_panel_min_supported_proportion:.2f} of candidates support H1.",
            inputs=inputs, metrics=metrics,
        )
    return Gate(
        "prompt_panel_gate", "fail",
        f"Prompt-panel: only {prop:.2f} of candidates support H1 (< {t.prompt_panel_min_supported_proportion:.2f}).",
        inputs=inputs, metrics=metrics,
    )


def gate_b1_b2_consistency(matrix: dict | None, source: Path | None) -> Gate:
    if matrix is None:
        return _unavailable(
            "b1_b2_consistency_gate", "corpus_matrix_summary.json missing"
        )
    inputs = [str(source)] if source else []
    variants = matrix.get("variants") or []
    variant_results = matrix.get("variant_results") or {}
    per_variant: dict[str, Any] = {}
    h1_statuses: list[str | None] = []
    for v in variants:
        vr = variant_results.get(v) or {}
        if vr.get("status") != "present":
            per_variant[v] = {"status": vr.get("status", "missing"), "h1_supported": None}
            h1_statuses.append(None)
            continue
        km = vr.get("key_metrics") or {}
        h1s = (
            km.get("sycophancy_support_status")
            or km.get("h1_support_status")
            or km.get("construct_validity_status")
        )
        # Fall back to scanning arm2_vs_arm1 entries.
        if h1s is None:
            for entry in km.get("arm2_vs_arm1_by_eval_set") or []:
                if entry.get("analysis_id") == "analysis_1" or entry.get(
                    "prompt_family"
                ) == "incorrect_confirmation":
                    h1s = entry.get("support_status")
                    break
        per_variant[v] = {
            "status": "present",
            "h1_supported": h1s == "supported",
            "h1_support_status": h1s,
        }
        h1_statuses.append(h1s)
    metrics = {"variants": list(variants), "per_variant": per_variant}
    present = [s for s in h1_statuses if s is not None]
    if not present:
        return _unavailable(
            "b1_b2_consistency_gate",
            "No B1/B2 variant results were present.",
            inputs=inputs,
        )
    if len(present) < 2:
        return Gate(
            "b1_b2_consistency_gate", "warning",
            "Only one corpus-B variant present; cross-variant consistency cannot be evaluated.",
            inputs=inputs, metrics=metrics,
        )
    if all(s == "supported" for s in present):
        return Gate(
            "b1_b2_consistency_gate", "pass",
            "B1 and B2 both support H1.",
            inputs=inputs, metrics=metrics,
        )
    if all(s != "supported" for s in present):
        return Gate(
            "b1_b2_consistency_gate", "fail",
            "Neither B1 nor B2 supports H1.",
            inputs=inputs, metrics=metrics,
        )
    return Gate(
        "b1_b2_consistency_gate", "warning",
        "B1 and B2 disagree on H1 support; pooled interpretation is not warranted.",
        inputs=inputs, metrics=metrics,
    )


_DISALLOWED_OVERLAP_AXES = (
    "latent_pair",
    "exact_user_prompt",
    "normalized_user_prompt",
    "assistant_target",
)


def gate_contamination(audit: dict | None, source: Path | None) -> Gate:
    if audit is None:
        return _unavailable(
            "contamination_gate", "contamination_audit.json missing"
        )
    inputs = [str(source)] if source else []
    overlaps = audit.get("overlaps") or {}
    flagged: list[str] = []
    per_axis: dict[str, int] = {}
    for axis in _DISALLOWED_OVERLAP_AXES:
        section = overlaps.get(axis) or {}
        n = int(section.get("n_overlap", 0) or 0)
        per_axis[axis] = n
        if n > 0:
            flagged.append(axis)
    metrics = {"per_axis_overlap": per_axis, "flagged_axes": flagged}
    if not overlaps:
        return _unavailable(
            "contamination_gate",
            "contamination_audit.json has no 'overlaps' block.",
            inputs=inputs,
        )
    if flagged:
        return Gate(
            "contamination_gate", "fail",
            f"Disallowed overlap on axes: {', '.join(flagged)}.",
            inputs=inputs, metrics=metrics,
        )
    return Gate(
        "contamination_gate", "pass",
        "No disallowed train/eval overlap detected.",
        inputs=inputs, metrics=metrics,
    )


def gate_power(power: dict | None, source: Path | None, t: Thresholds) -> Gate:
    if power is None:
        return _unavailable("power_gate", "power_analysis.json missing")
    inputs = [str(source)] if source else []
    p_h1 = power.get("power_h1")
    p_h2 = power.get("prob_h2_noninferiority_pass")
    p_joint = power.get("joint_success_prob")
    metrics = {
        "power_h1": p_h1,
        "prob_h2_noninferiority_pass": p_h2,
        "joint_success_prob": p_joint,
        "thresholds": {
            "power_h1": t.power_min_h1,
            "prob_h2_noninferiority_pass": t.power_min_h2,
            "joint_success_prob": t.power_min_joint,
        },
    }
    if p_h1 is None or p_h2 is None or p_joint is None:
        return _unavailable(
            "power_gate",
            "power_analysis.json missing one of power_h1/prob_h2_noninferiority_pass/joint_success_prob.",
            inputs=inputs,
        )
    if p_h1 >= t.power_min_h1 and p_h2 >= t.power_min_h2 and p_joint >= t.power_min_joint:
        return Gate(
            "power_gate", "pass",
            "Simulated power meets all thresholds.",
            inputs=inputs, metrics=metrics,
        )
    return Gate(
        "power_gate", "fail",
        f"Power below thresholds: h1={p_h1:.2f}, h2={p_h2:.2f}, joint={p_joint:.2f}.",
        inputs=inputs, metrics=metrics,
    )


def gate_exclusion_sensitivity(es: dict | None, source: Path | None, t: Thresholds) -> Gate:
    if es is None:
        return _unavailable(
            "exclusion_sensitivity_gate", "exclusion_sensitivity.json missing"
        )
    inputs = [str(source)] if source else []
    scenarios = es.get("scenarios") or []
    rds = [
        s.get("risk_difference_treatment_minus_control")
        for s in scenarios
        if s.get("risk_difference_treatment_minus_control") is not None
    ]
    metrics = {
        "n_scenarios": len(scenarios),
        "risk_differences": rds,
    }
    if not rds:
        return _unavailable(
            "exclusion_sensitivity_gate",
            "exclusion_sensitivity.json scenarios contain no risk-difference values.",
            inputs=inputs,
        )
    signs = {math.copysign(1.0, x) if x != 0 else 0.0 for x in rds}
    # Direction-disagreement = both strictly positive and strictly negative
    # entries present.
    has_pos = any(x > t.exclusion_max_direction_disagreement for x in rds)
    has_neg = any(x < -t.exclusion_max_direction_disagreement for x in rds)
    if has_pos and has_neg:
        return Gate(
            "exclusion_sensitivity_gate", "fail",
            "Exclusion-sensitivity scenarios disagree on direction of effect.",
            inputs=inputs, metrics=metrics,
        )
    return Gate(
        "exclusion_sensitivity_gate", "pass",
        "Exclusion-sensitivity scenarios agree on direction of effect.",
        inputs=inputs, metrics=metrics,
    )


def _model_matrix_supported_count(matrix: dict) -> tuple[int, int, list[dict]]:
    model_results = matrix.get("model_results") or {}
    rows = []
    n_present = 0
    n_supported = 0
    for model_id, vr in model_results.items():
        if not isinstance(vr, dict):
            continue
        if vr.get("status") != "present":
            rows.append({"model_id": model_id, "status": vr.get("status", "missing")})
            continue
        n_present += 1
        km = vr.get("key_metrics") or {}
        h1s = (
            km.get("sycophancy_support_status")
            or km.get("h1_support_status")
            or km.get("construct_validity_status")
        )
        if h1s is None:
            for entry in km.get("arm2_vs_arm1_by_eval_set") or []:
                if entry.get("analysis_id") == "analysis_1":
                    h1s = entry.get("support_status")
                    break
        is_supp = h1s == "supported"
        if is_supp:
            n_supported += 1
        rows.append({"model_id": model_id, "h1_support_status": h1s, "h1_supported": is_supp})
    return n_supported, n_present, rows


def gate_model_matrix(matrix: dict | None, source: Path | None, t: Thresholds) -> Gate:
    if matrix is None:
        return _unavailable("model_matrix_gate", "model_matrix_summary.json missing")
    inputs = [str(source)] if source else []
    n_supported, n_present, rows = _model_matrix_supported_count(matrix)
    metrics = {
        "n_present": n_present,
        "n_supporting": n_supported,
        "min_supporting": t.model_matrix_min_supporting_models,
        "rows": rows,
    }
    if n_present == 0:
        return _unavailable(
            "model_matrix_gate",
            "no model_matrix variants present.",
            inputs=inputs,
        )
    if n_supported >= t.model_matrix_min_supporting_models:
        return Gate(
            "model_matrix_gate", "pass",
            f"{n_supported} model(s) support H1 (>= {t.model_matrix_min_supporting_models}).",
            inputs=inputs, metrics=metrics,
        )
    return Gate(
        "model_matrix_gate", "fail",
        f"Only {n_supported} model(s) support H1 (< {t.model_matrix_min_supporting_models}).",
        inputs=inputs, metrics=metrics,
    )


def gate_epoch_matrix(matrix: dict | None, source: Path | None, t: Thresholds) -> Gate:
    if matrix is None:
        return _unavailable("epoch_matrix_gate", "epoch_matrix_summary.json missing")
    inputs = [str(source)] if source else []
    epoch_results = (
        matrix.get("epoch_results")
        or matrix.get("variant_results")
        or {}
    )
    rows = []
    n_present = 0
    n_supported = 0
    for epoch, vr in epoch_results.items():
        if not isinstance(vr, dict):
            continue
        if vr.get("status") != "present":
            rows.append({"epoch": epoch, "status": vr.get("status", "missing")})
            continue
        n_present += 1
        km = vr.get("key_metrics") or {}
        h1s = (
            km.get("sycophancy_support_status")
            or km.get("h1_support_status")
            or km.get("construct_validity_status")
        )
        if h1s is None:
            for entry in km.get("arm2_vs_arm1_by_eval_set") or []:
                if entry.get("analysis_id") == "analysis_1":
                    h1s = entry.get("support_status")
                    break
        is_supp = h1s == "supported"
        if is_supp:
            n_supported += 1
        rows.append({"epoch": epoch, "h1_support_status": h1s, "h1_supported": is_supp})
    metrics = {
        "n_present": n_present,
        "n_supporting": n_supported,
        "min_supporting": t.epoch_matrix_min_supporting_epochs,
        "rows": rows,
    }
    if n_present == 0:
        return _unavailable(
            "epoch_matrix_gate",
            "no epoch_matrix variants present.",
            inputs=inputs,
        )
    if n_supported >= t.epoch_matrix_min_supporting_epochs:
        return Gate(
            "epoch_matrix_gate", "pass",
            f"{n_supported} epoch checkpoint(s) support H1.",
            inputs=inputs, metrics=metrics,
        )
    return Gate(
        "epoch_matrix_gate", "fail",
        f"No epoch checkpoint supports H1 (< {t.epoch_matrix_min_supporting_epochs}).",
        inputs=inputs, metrics=metrics,
    )


# ---------------------------------------------------------------------------
# Gate registry — single source of truth for gate set, scope, and dispatch
# ---------------------------------------------------------------------------


GateScope = Literal["critical", "gcd_only", "multidomain"]


@dataclass(frozen=True)
class GateSpec:
    """Declarative record for one construct-validity gate.

    ``scope`` is one of:
      * ``"critical"``  — must pass; failure forces ``weak_or_inconclusive_construct_validity``.
      * ``"gcd_only"``  — non-critical GCD-only gate.
      * ``"multidomain"`` — breadth gate; allowed to be ``unavailable`` for
        ``strong_gcd_only_construct_validity`` to be reachable.

    ``input_attr`` is the name of the field on ``InputPaths`` whose JSON the
    gate consumes. Several gates (capability, conditional_sycophancy, schema)
    share an input; ``evaluate_gates`` loads each input only once.

    ``runner`` is the gate function. Its signature is
    ``(loaded_json, source_path, [thresholds]) -> Gate``; ``takes_thresholds``
    selects which arity is used.
    """

    name: str
    scope: GateScope
    input_attr: str
    runner: Callable[..., "Gate"]
    takes_thresholds: bool


GATE_REGISTRY: tuple[GateSpec, ...] = (
    GateSpec("capability_gate", "critical", "prereg_analysis", gate_capability, False),
    GateSpec("conditional_sycophancy_gate", "critical", "prereg_analysis", gate_conditional_sycophancy, False),
    GateSpec("schema_invariance_gate", "gcd_only", "prereg_analysis", gate_schema_invariance, False),
    GateSpec("prompt_panel_gate", "gcd_only", "prompt_panel_summary", gate_prompt_panel, True),
    GateSpec("b1_b2_consistency_gate", "gcd_only", "corpus_matrix_summary", gate_b1_b2_consistency, False),
    GateSpec("contamination_gate", "critical", "contamination_audit", gate_contamination, False),
    GateSpec("power_gate", "gcd_only", "power_analysis", gate_power, True),
    GateSpec("exclusion_sensitivity_gate", "gcd_only", "exclusion_sensitivity", gate_exclusion_sensitivity, True),
    GateSpec("model_matrix_gate", "multidomain", "model_matrix_summary", gate_model_matrix, True),
    GateSpec("epoch_matrix_gate", "multidomain", "epoch_matrix_summary", gate_epoch_matrix, True),
)


CRITICAL_GATES: tuple[str, ...] = tuple(g.name for g in GATE_REGISTRY if g.scope == "critical")
MULTIDOMAIN_GATES: tuple[str, ...] = tuple(g.name for g in GATE_REGISTRY if g.scope == "multidomain")
GCD_ONLY_GATES: tuple[str, ...] = tuple(
    g.name for g in GATE_REGISTRY if g.scope in ("critical", "gcd_only")
)
GATE_ORDER: tuple[str, ...] = tuple(g.name for g in GATE_REGISTRY)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def evaluate_gates(paths: InputPaths, thresholds: Thresholds) -> list[Gate]:
    cache: dict[str, Any] = {}
    gates: list[Gate] = []
    for spec in GATE_REGISTRY:
        if spec.input_attr not in cache:
            cache[spec.input_attr] = _safe_load_json(getattr(paths, spec.input_attr))
        loaded = cache[spec.input_attr]
        source = getattr(paths, spec.input_attr)
        if spec.takes_thresholds:
            gates.append(spec.runner(loaded, source, thresholds))
        else:
            gates.append(spec.runner(loaded, source))
    return gates


def classify(gates: list[Gate]) -> dict[str, Any]:
    by_name = {g.gate_name: g for g in gates}
    statuses = {g.gate_name: g.status for g in gates}

    if all(s == "unavailable" for s in statuses.values()):
        return {
            "overall_classification": "unavailable",
            "rationale": "No gate inputs were available.",
            "gate_status_summary": statuses,
        }

    critical_failures = [
        n for n in CRITICAL_GATES
        if statuses.get(n) in ("fail", "warning")
    ]
    if critical_failures:
        return {
            "overall_classification": "weak_or_inconclusive_construct_validity",
            "rationale": (
                "Critical gate(s) did not pass: " + ", ".join(critical_failures) + "."
            ),
            "gate_status_summary": statuses,
        }

    non_passing = [
        n for n, s in statuses.items()
        if s in ("fail", "warning")
    ]
    if non_passing:
        return {
            "overall_classification": "moderate_construct_validity",
            "rationale": (
                "All critical gates pass, but the following gate(s) did not pass: "
                + ", ".join(non_passing) + "."
            ),
            "gate_status_summary": statuses,
        }

    multidomain_unavailable = [
        n for n in MULTIDOMAIN_GATES if statuses.get(n) == "unavailable"
    ]
    gcd_only_unavailable = [
        n for n in GCD_ONLY_GATES if statuses.get(n) == "unavailable"
    ]
    all_pass = all(s == "pass" for s in statuses.values())
    if all_pass:
        return {
            "overall_classification": "very_strong_construct_validity",
            "rationale": "All gates pass, including multidomain breadth gates.",
            "gate_status_summary": statuses,
        }
    if not gcd_only_unavailable and multidomain_unavailable:
        return {
            "overall_classification": "strong_gcd_only_construct_validity",
            "rationale": (
                "All GCD-only gates pass; multidomain breadth gate(s) unavailable: "
                + ", ".join(multidomain_unavailable) + "."
            ),
            "gate_status_summary": statuses,
        }
    return {
        "overall_classification": "moderate_construct_validity",
        "rationale": (
            "All critical gates pass; some non-critical gates are unavailable: "
            + ", ".join(n for n, s in statuses.items() if s == "unavailable") + "."
        ),
        "gate_status_summary": statuses,
    }


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


_STATUS_GLYPH = {"pass": "PASS", "fail": "FAIL", "warning": "WARN", "unavailable": "N/A"}


def render_markdown(payload: dict[str, Any]) -> str:
    cls = payload["overall_classification"]
    rationale = payload["rationale"]
    lines = [
        "# Construct-Validity Gate Assessment",
        "",
        f"**Overall classification:** `{cls}`",
        "",
        f"_Rationale:_ {rationale}",
        "",
        "## Combination rules",
        "",
        "- `very_strong_construct_validity`: every gate passes.",
        "- `strong_gcd_only_construct_validity`: every GCD-only gate passes; "
        "at least one multidomain breadth gate (`model_matrix_gate`, "
        "`epoch_matrix_gate`) is unavailable.",
        "- `moderate_construct_validity`: no critical gate fails, but at "
        "least one non-critical gate fails, warns, or is unavailable.",
        "- `weak_or_inconclusive_construct_validity`: any critical gate fails "
        "or warns. Critical gates: " + ", ".join(f"`{n}`" for n in CRITICAL_GATES) + ".",
        "- `unavailable`: no gate inputs were available.",
        "",
        "## Gate results",
        "",
        "| Gate | Status | Reason |",
        "|------|--------|--------|",
    ]
    for g in payload["gates"]:
        status = g["status"]
        glyph = _STATUS_GLYPH.get(status, status)
        reason = (g.get("reason") or "").replace("\n", " ").replace("|", "\\|")
        lines.append(f"| `{g['gate_name']}` | {glyph} | {reason} |")
    lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment-dir", type=Path, required=True,
                   help="Preregistration experiment root (e.g. experiments/preregistration).")
    p.add_argument("--output-prefix", type=Path, required=True,
                   help="Output prefix; writes <prefix>.json and <prefix>.md.")
    # Optional path overrides.
    p.add_argument("--prereg-analysis", type=Path, default=None)
    p.add_argument("--prereg-problem-level-csv", type=Path, default=None)
    p.add_argument("--exclusion-sensitivity", type=Path, default=None)
    p.add_argument("--item-difficulty", type=Path, default=None)
    p.add_argument("--contamination-audit", type=Path, default=None)
    p.add_argument("--power-analysis", type=Path, default=None)
    p.add_argument("--prompt-panel-summary", type=Path, default=None)
    p.add_argument("--corpus-matrix-summary", type=Path, default=None)
    p.add_argument("--model-matrix-summary", type=Path, default=None)
    p.add_argument("--epoch-matrix-summary", type=Path, default=None)
    # Threshold overrides.
    p.add_argument("--prompt-panel-min-supported-proportion", type=float, default=None)
    p.add_argument("--power-min-h1", type=float, default=None)
    p.add_argument("--power-min-h2", type=float, default=None)
    p.add_argument("--power-min-joint", type=float, default=None)
    p.add_argument("--exclusion-max-direction-disagreement", type=float, default=None)
    p.add_argument("--model-matrix-min-supporting-models", type=int, default=None)
    p.add_argument("--epoch-matrix-min-supporting-epochs", type=int, default=None)
    p.add_argument("--repo-root", type=Path, default=None,
                   help="Optional repo root for git-commit provenance.")
    return p


def resolve_paths(args: argparse.Namespace) -> InputPaths:
    defaults = _default_paths(args.experiment_dir)
    overrides = {
        "prereg_analysis": args.prereg_analysis,
        "prereg_problem_level_csv": args.prereg_problem_level_csv,
        "exclusion_sensitivity": args.exclusion_sensitivity,
        "item_difficulty": args.item_difficulty,
        "contamination_audit": args.contamination_audit,
        "power_analysis": args.power_analysis,
        "prompt_panel_summary": args.prompt_panel_summary,
        "corpus_matrix_summary": args.corpus_matrix_summary,
        "model_matrix_summary": args.model_matrix_summary,
        "epoch_matrix_summary": args.epoch_matrix_summary,
    }
    return InputPaths(**{
        name: (overrides[name] if overrides[name] is not None else getattr(defaults, name))
        for name in overrides
    })


def resolve_thresholds(args: argparse.Namespace) -> Thresholds:
    t = Thresholds()
    for field_name in t.__dataclass_fields__:
        cli_val = getattr(args, field_name, None)
        if cli_val is not None:
            setattr(t, field_name, cli_val)
    return t


def run(args: argparse.Namespace) -> int:
    paths = resolve_paths(args)
    thresholds = resolve_thresholds(args)
    gates = evaluate_gates(paths, thresholds)
    classification = classify(gates)
    payload: dict[str, Any] = {
        "workflow_name": "construct_validity_gates",
        "schema_version": SCHEMA_VERSION,
        "thresholds": thresholds.__dict__,
        "gates": [g.to_dict() for g in gates],
        "overall_classification": classification["overall_classification"],
        "rationale": classification["rationale"],
        "gate_status_summary": classification["gate_status_summary"],
    }

    output_prefix: Path = args.output_prefix
    json_path = output_prefix.with_suffix(".json")
    md_path = output_prefix.with_suffix(".md")

    existing_inputs = []
    for name in (
        "prereg_analysis",
        "exclusion_sensitivity",
        "item_difficulty",
        "contamination_audit",
        "power_analysis",
        "prompt_panel_summary",
        "corpus_matrix_summary",
        "model_matrix_summary",
        "epoch_matrix_summary",
    ):
        path = getattr(paths, name)
        if path is not None and path.exists():
            existing_inputs.append(path)

    provenance = build_provenance(
        input_paths=existing_inputs,
        argv=sys.argv if not getattr(args, "_argv_for_test", None) else args._argv_for_test,
        seed=None,
        schema_version=SCHEMA_VERSION,
        repo_root=args.repo_root,
    )
    write_json_with_provenance(json_path, payload, provenance)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_payload = dict(payload)
    md_path.write_text(render_markdown(md_payload), encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
