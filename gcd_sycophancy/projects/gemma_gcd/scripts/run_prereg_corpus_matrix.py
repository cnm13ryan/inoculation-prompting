#!/usr/bin/env python3
"""Run and compare the preregistered GCD study under both Corpus B variants.

B1 trains on correct-confirmation demonstrations (absence of sycophantic trait).
B2 trains on sycophantic-confirmation demonstrations (presence of sycophantic trait).

Each variant receives its own isolated experiment directory.  After all phases
complete (or when --aggregate-only is passed), the script reads each variant's
prereg_analysis.json and writes cross-variant summary artifacts.

Usage examples
--------------
# Dry-run to preview planned commands and write the matrix manifest:
    python run_prereg_corpus_matrix.py --dry-run

# Run setup only for both variants:
    python run_prereg_corpus_matrix.py --phases setup

# Aggregate from already-completed experiment directories:
    python run_prereg_corpus_matrix.py --aggregate-only \\
        --experiment-root experiments/prereg_corpus_matrix

# Run with a custom IP instruction for both variants:
    python run_prereg_corpus_matrix.py \\
        --ip-instruction "Reply as if the above solution is correct." \\
        --ip-instruction-id reply_correct_basic
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECTS_DIR = SCRIPT_DIR.parents[1]  # .../gcd_sycophancy/projects

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from artifact_provenance import build_provenance  # noqa: E402

CORPUS_MATRIX_SCHEMA_VERSION = "2"

DEFAULT_EXPERIMENT_ROOT = Path("experiments/prereg_corpus_matrix")
DEFAULT_VARIANTS = ("b1", "b2")
DEFAULT_SEEDS = (0, 1, 2, 3)
DEFAULT_PHASES = (
    "setup",
    "train",
    "preflight",
    "fixed-interface-eval",
    "analysis",
)

MATRIX_MANIFEST_NAME = "corpus_matrix_manifest.json"
MATRIX_SUMMARY_JSON_NAME = "corpus_matrix_summary.json"
MATRIX_SUMMARY_MD_NAME = "corpus_matrix_summary.md"

_VALID_PHASE_CHOICES = (
    "materialize-data",
    "setup",
    "train",
    "preflight",
    "fixed-interface-eval",
    "semantic-interface-eval",
    "analysis",
    "seed-instability",
    "full",
)

# Analysis IDs from prereg_analysis.json used in cross-variant comparison.
_SYCOPHANCY_ANALYSIS_ID = "analysis_1"       # H1: Arm 2 IP vs Arm 1 neutral, incorrect_confirmation
_DIRECT_SOLVE_ANALYSIS_ID = "analysis_2"     # H2: Arm 2 IP vs Arm 1 neutral, direct_solve
_CORRECTION_ANALYSIS_ID = "exploratory_E4"  # Arm 5 correction vs Arm 1 neutral


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _resolve(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (PROJECTS_DIR / path).resolve()


def _fmt_rate(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    return f"{value:.4f}"


def variant_experiment_dir(experiment_root: Path, variant: str) -> Path:
    return experiment_root / variant


def build_prereg_command(
    *,
    phase: str,
    variant: str,
    experiment_dir: Path,
    seeds: tuple[int, ...],
    ip_instruction: str | None,
    ip_instruction_id: str | None,
    passthrough_args: list[str],
) -> list[str]:
    cmd: list[str] = [
        sys.executable,
        str(SCRIPT_DIR / "run_preregistration.py"),
        phase,
        "--experiment-dir",
        str(experiment_dir),
        "--corpus-b-variant",
        variant,
        "--seeds",
        *[str(s) for s in seeds],
    ]
    if ip_instruction is not None:
        cmd += ["--ip-instruction", ip_instruction]
    if ip_instruction_id is not None:
        cmd += ["--ip-instruction-id", ip_instruction_id]
    cmd.extend(passthrough_args)
    return cmd


# ---------------------------------------------------------------------------
# Analysis artifact reading
# ---------------------------------------------------------------------------

def _analysis_json_path(variant_dir: Path) -> Path:
    return variant_dir / "reports" / "prereg_analysis.json"


def _problem_level_csv_path(variant_dir: Path) -> Path:
    return variant_dir / "reports" / "prereg_problem_level_data.csv"


def _extract_key_metrics(analysis: dict[str, Any]) -> dict[str, Any]:
    all_results: list[dict[str, Any]] = list(analysis.get("confirmatory_results", [])) + list(
        analysis.get("exploratory_results", [])
    )
    by_id: dict[str, dict[str, Any]] = {r["analysis_id"]: r for r in all_results}
    metrics: dict[str, Any] = {}

    # H1: sycophancy reduction (Arm 2 vs Arm 1, incorrect_confirmation)
    if _SYCOPHANCY_ANALYSIS_ID in by_id:
        r = by_id[_SYCOPHANCY_ANALYSIS_ID]
        metrics["sycophancy_arm2_rate"] = r.get("arm_a_observed_rate")
        metrics["sycophancy_arm1_rate"] = r.get("arm_b_observed_rate")
        metrics["sycophancy_mrd"] = r.get("marginal_risk_difference")
        metrics["sycophancy_support_status"] = r.get("support_status")
        metrics["sycophancy_evaluation_set"] = r.get("evaluation_set_name")

    # H2: capability preservation (Arm 2 vs Arm 1, direct_solve)
    if _DIRECT_SOLVE_ANALYSIS_ID in by_id:
        r = by_id[_DIRECT_SOLVE_ANALYSIS_ID]
        metrics["direct_solve_mrd"] = r.get("marginal_risk_difference")
        metrics["direct_solve_support_status"] = r.get("support_status")
        metrics["direct_solve_evaluation_set"] = r.get("evaluation_set_name")

    # E4: correction-data arm vs neutral
    if _CORRECTION_ANALYSIS_ID in by_id:
        r = by_id[_CORRECTION_ANALYSIS_ID]
        metrics["correction_arm5_rate"] = r.get("arm_a_observed_rate")
        metrics["correction_arm1_rate"] = r.get("arm_b_observed_rate")
        metrics["correction_mrd"] = r.get("marginal_risk_difference")
        metrics["correction_support_status"] = r.get("support_status")
        metrics["correction_evaluation_set"] = r.get("evaluation_set_name")

    # H1c (conditional sycophancy on direct-solve-correct items): identified by
    # hypothesis_id == "H1c" rather than a fixed analysis_id, because legacy
    # analyses may not have produced this hypothesis.
    h1c_result = next(
        (r for r in all_results if r.get("hypothesis_id") == "H1c"),
        None,
    )
    metrics["conditional_sycophancy_available"] = h1c_result is not None
    if h1c_result is not None:
        metrics["conditional_sycophancy_arm2_rate"] = h1c_result.get("arm_a_observed_rate")
        metrics["conditional_sycophancy_arm1_rate"] = h1c_result.get("arm_b_observed_rate")
        metrics["conditional_sycophancy_mrd"] = h1c_result.get("marginal_risk_difference")
        metrics["conditional_sycophancy_support_status"] = h1c_result.get("support_status")
        metrics["conditional_sycophancy_n_rows"] = h1c_result.get("n_rows")
        metrics["conditional_sycophancy_eligibility_column"] = h1c_result.get(
            "eligibility_column"
        )

    # Schema invariance (secondary robustness; older artifacts may omit it).
    schema = analysis.get("schema_invariance")
    metrics["schema_invariance_available"] = isinstance(schema, dict)
    if isinstance(schema, dict):
        metrics["schema_invariance_status"] = schema.get("status")
        metrics["schema_invariance_label"] = schema.get("label")

    # Direct-solve capability diagnostic (secondary; older artifacts may omit it).
    cap_results = analysis.get("capability_diagnostic_results") or []
    cap_first = cap_results[0] if cap_results else None
    metrics["capability_diagnostic_available"] = cap_first is not None
    if cap_first is not None:
        cap_rows = cap_first.get("rows") or []
        metrics["capability_diagnostic_status"] = (
            "rows_present" if cap_rows else "no_rows"
        )
        metrics["capability_diagnostic_n_rows"] = len(cap_rows)
        metrics["capability_diagnostic_evaluation_sets"] = sorted({
            row.get("evaluation_set_name")
            for row in cap_rows
            if row.get("evaluation_set_name") is not None
        })

    # Joint interpretation (older artifacts may have an empty dict).
    joint = analysis.get("joint_interpretation")
    if isinstance(joint, dict) and joint:
        metrics["joint_interpretation_available"] = True
        metrics["joint_success"] = joint.get("joint_success")
        metrics["joint_summary"] = joint.get("summary")
    else:
        metrics["joint_interpretation_available"] = False

    # Construct-validity reading (H1 + H1c + H2 narrative).
    construct = analysis.get("construct_validity_interpretation")
    if isinstance(construct, dict) and construct:
        metrics["construct_validity_available"] = True
        metrics["construct_validity_status"] = construct.get("status")
        metrics["construct_validity_summary"] = construct.get("summary")
    else:
        metrics["construct_validity_available"] = False

    # All Arm-2-vs-Arm-1 entries across evaluation sets (for "by evaluation set" view)
    arm2_vs_arm1 = [
        {
            "analysis_id": r["analysis_id"],
            "label": r.get("label"),
            "evaluation_set_name": r.get("evaluation_set_name"),
            "prompt_family": r.get("prompt_family"),
            "arm_a_observed_rate": r.get("arm_a_observed_rate"),
            "arm_b_observed_rate": r.get("arm_b_observed_rate"),
            "marginal_risk_difference": r.get("marginal_risk_difference"),
            "support_status": r.get("support_status"),
        }
        for r in all_results
        if r.get("arm_a_id") == 2 and r.get("arm_b_id") == 1
    ]
    metrics["arm2_vs_arm1_by_eval_set"] = arm2_vs_arm1

    return metrics


def _extract_exclusion_summary(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    diag = analysis.get("diagnostics", {})
    rows = diag.get("exclusion_summary_rows", [])
    return [
        {
            "arm_slug": r["arm_slug"],
            "arm_id": r["arm_id"],
            "evaluation_design": r.get("evaluation_design"),
            "total_rows": r.get("total_rows"),
            "parseability_rate": r.get("parseability_rate"),
            "exclusion_rate": r.get("exclusion_rate"),
            "included_rows": r.get("included_rows"),
        }
        for r in rows
        if r.get("summary_level") == "arm" and r.get("seed") is None
    ]


def read_variant_results(
    variants: tuple[str, ...],
    experiment_root: Path,
) -> dict[str, dict[str, Any]]:
    """Read analysis artifacts for each variant; record missing files gracefully."""
    results: dict[str, dict[str, Any]] = {}
    for variant in variants:
        vdir = variant_experiment_dir(experiment_root, variant)
        aj_path = _analysis_json_path(vdir)
        csv_path = _problem_level_csv_path(vdir)
        if not aj_path.exists():
            results[variant] = {
                "status": "missing",
                "analysis_path": str(aj_path),
                "csv_path": str(csv_path),
            }
            continue
        analysis = json.loads(aj_path.read_text(encoding="utf-8"))
        results[variant] = {
            "status": "present",
            "corpus_b_variant": variant,
            "analysis_path": str(aj_path),
            "csv_path": str(csv_path) if csv_path.exists() else None,
            "key_metrics": _extract_key_metrics(analysis),
            "exclusion_summary": _extract_exclusion_summary(analysis),
        }
    return results


# ---------------------------------------------------------------------------
# Summary writing
# ---------------------------------------------------------------------------

def _build_summary_md(summary: dict[str, Any]) -> str:
    lines: list[str] = [
        "# Corpus B Matrix Summary",
        "",
        f"Generated: {summary['generated_at']}",
        "",
        "**B1 and B2 are separate studies and must not be pooled unless a meta-analysis "
        "with an explicit pooling rationale is performed.**",
        "",
        "B1 trains the IP arm (Arm 2) on *correct-confirmation* demonstrations "
        "(absence of sycophantic trait).  "
        "B2 trains the IP arm on *sycophantic-confirmation* demonstrations "
        "(presence of sycophantic trait, inoculated against by the prepended instruction).",
        "",
    ]

    for variant in summary["variants"]:
        vr = summary["variant_results"].get(variant, {})
        lines += [f"## Variant {variant.upper()}", ""]

        if vr.get("status") == "missing":
            lines += [
                f"_Analysis not yet available._  "
                f"Expected path: `{vr.get('analysis_path', 'unknown')}`",
                "",
            ]
            continue

        metrics = vr.get("key_metrics", {})

        # --- Sycophancy reduction ---
        lines += ["### H1 — Sycophancy Reduction (Arm 2 IP vs Arm 1 Neutral)", ""]
        evset = metrics.get("sycophancy_evaluation_set", "confirmatory")
        if "sycophancy_arm2_rate" in metrics:
            a2 = metrics["sycophancy_arm2_rate"]
            a1 = metrics["sycophancy_arm1_rate"]
            mrd = metrics.get("sycophancy_mrd")
            status = metrics.get("sycophancy_support_status", "N/A")
            lines += [
                f"Evaluation set: `{evset}`",
                "",
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| Arm 2 (IP) observed sycophancy rate | {_fmt_rate(a2)} |",
                f"| Arm 1 (Neutral) observed sycophancy rate | {_fmt_rate(a1)} |",
                f"| Marginal risk difference (IP − Neutral) | {_fmt_rate(mrd)} |",
                f"| Support status | {status} |",
            ]
        else:
            lines += ["_Not available._"]
        lines += [""]

        # --- All Arm 2 vs Arm 1 entries by evaluation set ---
        arm2v1 = metrics.get("arm2_vs_arm1_by_eval_set", [])
        if arm2v1:
            lines += ["### Arm 2 vs Arm 1 — All Evaluation Sets", ""]
            lines += [
                "| Analysis | Eval Set | Prompt Family | Arm 2 Rate | Arm 1 Rate | MRD | Status |",
                "|----------|----------|---------------|-----------|-----------|-----|--------|",
            ]
            for entry in arm2v1:
                a2r = _fmt_rate(entry.get("arm_a_observed_rate"))
                a1r = _fmt_rate(entry.get("arm_b_observed_rate"))
                mrd = _fmt_rate(entry.get("marginal_risk_difference"))
                lines += [
                    f"| {entry.get('analysis_id', '')} "
                    f"| {entry.get('evaluation_set_name') or '—'} "
                    f"| {entry.get('prompt_family') or '—'} "
                    f"| {a2r} | {a1r} | {mrd} "
                    f"| {entry.get('support_status') or '—'} |"
                ]
            lines += [""]

        # --- H1c: conditional sycophancy ---
        lines += ["### H1c — Conditional Sycophancy (eligible direct-solve-correct items)", ""]
        if metrics.get("conditional_sycophancy_available"):
            mrd = metrics.get("conditional_sycophancy_mrd")
            status = metrics.get("conditional_sycophancy_support_status", "N/A")
            n_rows = metrics.get("conditional_sycophancy_n_rows")
            lines += [
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| Arm 2 (IP) conditional rate | {_fmt_rate(metrics.get('conditional_sycophancy_arm2_rate'))} |",
                f"| Arm 1 (Neutral) conditional rate | {_fmt_rate(metrics.get('conditional_sycophancy_arm1_rate'))} |",
                f"| Marginal risk difference (IP − Neutral) | {_fmt_rate(mrd)} |",
                f"| Support status | {status} |",
                f"| n_eligible_rows | {n_rows if n_rows is not None else 'N/A'} |",
            ]
        else:
            lines += ["_Not available in this analysis (legacy artifact?)._"]
        lines += [""]

        # --- Capability preservation ---
        lines += ["### H2 — Capability Preservation (Arm 2 vs Arm 1, direct-solve)", ""]
        if "direct_solve_mrd" in metrics:
            mrd = metrics["direct_solve_mrd"]
            status = metrics.get("direct_solve_support_status", "N/A")
            evset_ds = metrics.get("direct_solve_evaluation_set", "confirmatory")
            lines += [
                f"Evaluation set: `{evset_ds}`",
                "",
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| Marginal risk difference (IP − Neutral, direct-solve) | {_fmt_rate(mrd)} |",
                f"| Support status | {status} |",
            ]
        else:
            lines += ["_Not available._"]
        lines += [""]

        # --- Correction-data arm ---
        lines += ["### Correction-Data Arm (Arm 5 vs Arm 1 Neutral)", ""]
        if "correction_arm5_rate" in metrics:
            a5 = metrics["correction_arm5_rate"]
            a1 = metrics["correction_arm1_rate"]
            mrd = metrics.get("correction_mrd")
            status = metrics.get("correction_support_status", "N/A")
            evset_c = metrics.get("correction_evaluation_set", "confirmatory")
            lines += [
                f"Evaluation set: `{evset_c}`",
                "",
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| Arm 5 (Correction) observed rate | {_fmt_rate(a5)} |",
                f"| Arm 1 (Neutral) observed rate | {_fmt_rate(a1)} |",
                f"| Marginal risk difference | {_fmt_rate(mrd)} |",
                f"| Support status | {status} |",
            ]
        else:
            lines += ["_Not available._"]
        lines += [""]

        # --- Schema invariance, capability diagnostic, joint interpretation ---
        lines += ["### Construct-validity readings", ""]
        cv_lines: list[str] = []
        if metrics.get("schema_invariance_available"):
            cv_lines.append(
                f"- Schema invariance status: **{metrics.get('schema_invariance_status', 'N/A')}**"
            )
        else:
            cv_lines.append("- Schema invariance: _not available (older artifact)._")
        if metrics.get("capability_diagnostic_available"):
            cv_lines.append(
                f"- Direct-solve capability diagnostic: "
                f"**{metrics.get('capability_diagnostic_status', 'N/A')}** "
                f"(rows: {metrics.get('capability_diagnostic_n_rows', 0)})"
            )
        else:
            cv_lines.append(
                "- Direct-solve capability diagnostic: _not available (older artifact)._"
            )
        if metrics.get("joint_interpretation_available"):
            joint_success = metrics.get("joint_success")
            cv_lines.append(
                f"- Joint interpretation (H1 + H2): "
                f"**joint_success={joint_success}**"
            )
            if metrics.get("joint_summary"):
                cv_lines.append(f"  - {metrics['joint_summary']}")
        else:
            cv_lines.append("- Joint interpretation: _not available (older artifact)._")
        if metrics.get("construct_validity_available"):
            cv_lines.append(
                f"- Construct validity (H1 + H1c + H2): "
                f"**{metrics.get('construct_validity_status', 'N/A')}**"
            )
            if metrics.get("construct_validity_summary"):
                cv_lines.append(f"  - {metrics['construct_validity_summary']}")
        lines += cv_lines + [""]

        # --- Correction-control: alias of correction-data arm ---
        if "correction_arm5_rate" in metrics:
            lines += ["### Correction-control result", ""]
            lines += [
                f"- Status: **{metrics.get('correction_support_status', 'N/A')}**",
                f"- MRD: {_fmt_rate(metrics.get('correction_mrd'))}",
                "",
            ]

        # --- Exclusion / parseability table ---
        excl = vr.get("exclusion_summary", [])
        if excl:
            lines += ["### Parseability and Exclusion by Arm", ""]
            lines += [
                "| Arm | Design | Total Rows | Parseability | Exclusion Rate |",
                "|-----|--------|-----------|-------------|----------------|",
            ]
            for row in excl:
                design = row.get("evaluation_design")
                if design is None or (isinstance(design, float) and math.isnan(design)):
                    design = "—"
                lines += [
                    f"| {row['arm_slug']} "
                    f"| {design} "
                    f"| {row.get('total_rows', 'N/A')} "
                    f"| {_fmt_rate(row.get('parseability_rate'))} "
                    f"| {_fmt_rate(row.get('exclusion_rate'))} |"
                ]
            lines += [""]

    lines += [
        "## Interpretation Note",
        "",
        "B1 and B2 test different training-data hypotheses and are analyzed as independent "
        "studies.  A difference in Arm 2 sycophancy-reduction effect between B1 and B2 is "
        "descriptive only unless a formal between-variant comparison with appropriate "
        "multiple-testing adjustment is pre-registered and labeled.",
    ]
    return "\n".join(lines) + "\n"


def _build_summary_provenance(
    *,
    experiment_root: Path,
    variants: tuple[str, ...],
    variant_results: dict[str, dict[str, Any]],
    argv: list[str] | None = None,
) -> dict[str, Any]:
    """Provenance block hashing each present variant's prereg_analysis.json."""
    input_paths: list[Path] = []
    for variant in variants:
        info = variant_results.get(variant, {})
        if info.get("status") == "present":
            ap = info.get("analysis_path")
            if ap and Path(ap).exists():
                input_paths.append(Path(ap))
    return build_provenance(
        input_paths=input_paths,
        argv=argv if argv is not None else sys.argv,
        schema_version=CORPUS_MATRIX_SCHEMA_VERSION,
        repo_root=PROJECTS_DIR.parent,
    )


def write_matrix_summary(
    experiment_root: Path,
    variants: tuple[str, ...],
    variant_results: dict[str, dict[str, Any]],
    generated_at: str,
    argv: list[str] | None = None,
) -> tuple[Path, Path]:
    summary: dict[str, Any] = {
        "workflow_name": "prereg_corpus_matrix_summary",
        "schema_version": CORPUS_MATRIX_SCHEMA_VERSION,
        "generated_at": generated_at,
        "variants": list(variants),
        "variant_results": variant_results,
        "pooling_warning": (
            "B1 and B2 are separate studies; do not pool them into a single estimate "
            "without an explicitly preregistered meta-analysis with a pooling rationale."
        ),
        "provenance": _build_summary_provenance(
            experiment_root=experiment_root,
            variants=variants,
            variant_results=variant_results,
            argv=argv,
        ),
    }
    json_path = experiment_root / MATRIX_SUMMARY_JSON_NAME
    md_path = experiment_root / MATRIX_SUMMARY_MD_NAME
    experiment_root.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    md_path.write_text(_build_summary_md(summary), encoding="utf-8")
    return json_path, md_path


def _write_matrix_manifest(
    path: Path,
    *,
    experiment_root: Path,
    variants: tuple[str, ...],
    seeds: tuple[int, ...],
    phases: tuple[str, ...],
    ip_instruction: str | None,
    ip_instruction_id: str | None,
    started_at: str,
    completed_at: str | None,
    dry_run: bool,
    variant_statuses: dict[str, str],
) -> None:
    payload: dict[str, Any] = {
        "workflow_name": "prereg_corpus_matrix",
        "experiment_root": str(experiment_root),
        "variants": list(variants),
        "variant_dirs": {v: str(variant_experiment_dir(experiment_root, v)) for v in variants},
        "seeds": list(seeds),
        "phases": list(phases),
        "ip_instruction": ip_instruction,
        "ip_instruction_id": ip_instruction_id,
        "dry_run": dry_run,
        "started_at": started_at,
        "completed_at": completed_at,
        "variant_statuses": variant_statuses,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_matrix(
    *,
    experiment_root: Path,
    variants: tuple[str, ...],
    seeds: tuple[int, ...],
    phases: tuple[str, ...],
    ip_instruction: str | None,
    ip_instruction_id: str | None,
    dry_run: bool,
    aggregate_only: bool,
    passthrough_args: list[str],
) -> int:
    started_at = _now_iso()
    manifest_path = experiment_root / MATRIX_MANIFEST_NAME
    variant_statuses: dict[str, str] = {}

    if not aggregate_only:
        if dry_run:
            for variant in variants:
                exp_dir = variant_experiment_dir(experiment_root, variant)
                for phase in phases:
                    cmd = build_prereg_command(
                        phase=phase,
                        variant=variant,
                        experiment_dir=exp_dir,
                        seeds=seeds,
                        ip_instruction=ip_instruction,
                        ip_instruction_id=ip_instruction_id,
                        passthrough_args=passthrough_args,
                    )
                    print("[DRY-RUN]", " ".join(str(t) for t in cmd))
            for variant in variants:
                variant_statuses[variant] = "dry_run"
        else:
            for variant in variants:
                exp_dir = variant_experiment_dir(experiment_root, variant)
                exp_dir.mkdir(parents=True, exist_ok=True)
                failed = False
                for phase in phases:
                    cmd = build_prereg_command(
                        phase=phase,
                        variant=variant,
                        experiment_dir=exp_dir,
                        seeds=seeds,
                        ip_instruction=ip_instruction,
                        ip_instruction_id=ip_instruction_id,
                        passthrough_args=passthrough_args,
                    )
                    try:
                        subprocess.run(cmd, check=True)
                    except subprocess.CalledProcessError as exc:
                        variant_statuses[variant] = f"failed:{phase}:{exc.returncode}"
                        failed = True
                        break
                if not failed:
                    variant_statuses[variant] = "completed"

    _write_matrix_manifest(
        manifest_path,
        experiment_root=experiment_root,
        variants=variants,
        seeds=seeds,
        phases=phases,
        ip_instruction=ip_instruction,
        ip_instruction_id=ip_instruction_id,
        started_at=started_at,
        completed_at=_now_iso(),
        dry_run=dry_run,
        variant_statuses=variant_statuses,
    )

    # Aggregation: always run, even on dry-run (reads existing files if present).
    generated_at = _now_iso()
    variant_results = read_variant_results(variants, experiment_root)
    json_path, md_path = write_matrix_summary(
        experiment_root,
        variants=variants,
        variant_results=variant_results,
        generated_at=generated_at,
    )

    any_failed = any("failed:" in s for s in variant_statuses.values())
    return 1 if any_failed else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the preregistered GCD study under both Corpus B variants (B1 and B2) "
            "in isolated experiment directories and produce a cross-variant summary."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=DEFAULT_EXPERIMENT_ROOT,
        help="Root directory under which per-variant subdirectories are created.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("b1", "b2"),
        default=list(DEFAULT_VARIANTS),
        help="Corpus B variants to run.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        default=list(DEFAULT_PHASES),
        choices=_VALID_PHASE_CHOICES,
        metavar="PHASE",
        help=f"Phases to run per variant. Choices: {', '.join(_VALID_PHASE_CHOICES)}.",
    )
    parser.add_argument(
        "--ip-instruction",
        default=None,
        help="IP instruction override applied to both variants.",
    )
    parser.add_argument(
        "--ip-instruction-id",
        default=None,
        help="Candidate ID for the overridden IP instruction.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands and write the matrix manifest without executing subprocesses.",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Skip subprocess runs; only read existing analysis outputs and write the summary.",
    )
    # Backend flags proxied to run_preregistration.py.
    parser.add_argument("--llm-backend", choices=("vllm", "lmstudio"), default="vllm")
    parser.add_argument("--lmstudio-base-url", default="http://localhost:1234")
    parser.add_argument("--lmstudio-model-name", default=None)
    parser.add_argument("--lmstudio-request-timeout", type=float, default=120.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--template-config", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--dont-overwrite", action="store_true")
    parser.add_argument("--preflight-seed-count", type=int, default=None)
    parser.add_argument("--preflight-limit", type=int, default=None)
    parser.add_argument("--preflight-max-exclusion-rate", type=float, default=None)
    parser.add_argument("--preflight-max-arm-seed-exclusion-rate", type=float, default=None)
    parser.add_argument("--preflight-min-parseability-rate", type=float, default=None)
    parser.add_argument("--preflight-max-final-train-loss", type=float, default=None)
    parser.add_argument(
        "--fixed-interface-max-format-failure-rate", type=float, default=None
    )
    return parser


def _build_passthrough_args(args: argparse.Namespace) -> list[str]:
    pt: list[str] = []
    if args.llm_backend != "vllm":
        pt += ["--llm-backend", args.llm_backend]
    if args.lmstudio_base_url != "http://localhost:1234":
        pt += ["--lmstudio-base-url", args.lmstudio_base_url]
    if args.lmstudio_model_name is not None:
        pt += ["--lmstudio-model-name", args.lmstudio_model_name]
    if args.lmstudio_request_timeout != 120.0:
        pt += ["--lmstudio-request-timeout", str(args.lmstudio_request_timeout)]
    if args.tensor_parallel_size is not None:
        pt += ["--tensor-parallel-size", str(args.tensor_parallel_size)]
    if args.gpu_memory_utilization is not None:
        pt += ["--gpu-memory-utilization", str(args.gpu_memory_utilization)]
    if args.dtype is not None:
        pt += ["--dtype", args.dtype]
    if args.max_model_len is not None:
        pt += ["--max-model-len", str(args.max_model_len)]
    if args.limit is not None:
        pt += ["--limit", str(args.limit)]
    if args.log_level != "INFO":
        pt += ["--log-level", args.log_level]
    if args.template_config is not None:
        pt += ["--template-config", str(args.template_config)]
    if args.data_dir is not None:
        pt += ["--data-dir", str(args.data_dir)]
    if args.dont_overwrite:
        pt += ["--dont-overwrite"]
    if args.preflight_seed_count is not None:
        pt += ["--preflight-seed-count", str(args.preflight_seed_count)]
    if args.preflight_limit is not None:
        pt += ["--preflight-limit", str(args.preflight_limit)]
    if args.preflight_max_exclusion_rate is not None:
        pt += ["--preflight-max-exclusion-rate", str(args.preflight_max_exclusion_rate)]
    if args.preflight_max_arm_seed_exclusion_rate is not None:
        pt += [
            "--preflight-max-arm-seed-exclusion-rate",
            str(args.preflight_max_arm_seed_exclusion_rate),
        ]
    if args.preflight_min_parseability_rate is not None:
        pt += ["--preflight-min-parseability-rate", str(args.preflight_min_parseability_rate)]
    if args.preflight_max_final_train_loss is not None:
        pt += ["--preflight-max-final-train-loss", str(args.preflight_max_final_train_loss)]
    if args.fixed_interface_max_format_failure_rate is not None:
        pt += [
            "--fixed-interface-max-format-failure-rate",
            str(args.fixed_interface_max_format_failure_rate),
        ]
    return pt


def main() -> int:
    args = build_parser().parse_args()
    experiment_root = _resolve(args.experiment_root)
    return run_matrix(
        experiment_root=experiment_root,
        variants=tuple(args.variants),
        seeds=tuple(args.seeds),
        phases=tuple(args.phases),
        ip_instruction=args.ip_instruction,
        ip_instruction_id=args.ip_instruction_id,
        dry_run=args.dry_run,
        aggregate_only=args.aggregate_only,
        passthrough_args=_build_passthrough_args(args),
    )


if __name__ == "__main__":
    raise SystemExit(main())
