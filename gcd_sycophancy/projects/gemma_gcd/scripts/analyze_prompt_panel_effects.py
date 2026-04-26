"""Prompt-panel aggregate analysis (WT-6).

Pure aggregator: walks an isolated prereg directory tree of the form
``<panel_root>/<corpus_b_variant>/<candidate_id>/reports/prereg_analysis.json``
and surfaces, per candidate:

- candidate_id, candidate_rank
- delta_vs_no_prompt (from the eligible panel JSON, when available)
- H1 unconditional result
- H1c conditional result (when present)
- H2 capability preservation
- schema-invariance status (when present)
- joint success status
- effect size (H1 marginal risk difference)

Plus a panel-level summary: count and proportion of candidates with a
supported H1 reduction, and the distribution of H1 effect sizes.

This script does NOT retrain or re-evaluate.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
GEMMA_GCD_DIR = SCRIPT_DIR.parent
PROJECTS_DIR = GEMMA_GCD_DIR.parent
for candidate in (SCRIPT_DIR, GEMMA_GCD_DIR, PROJECTS_DIR):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from artifact_provenance import build_provenance, write_json_with_provenance  # noqa: E402

PROMPT_PANEL_SCHEMA_VERSION = "1"

logger = logging.getLogger(__name__)


def _analysis_path(candidate_dir: Path) -> Path:
    return candidate_dir / "reports" / "prereg_analysis.json"


def _find_by_hypothesis(results: list[dict[str, Any]], hypothesis_id: str) -> dict[str, Any] | None:
    for r in results:
        if r.get("hypothesis_id") == hypothesis_id:
            return r
    return None


def _summarize_hypothesis(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "hypothesis_id": result.get("hypothesis_id"),
        "label": result.get("label"),
        "support_status": result.get("support_status"),
        "marginal_risk_difference": result.get("marginal_risk_difference"),
        "arm_log_odds_coefficient": result.get("arm_log_odds_coefficient"),
        "evaluation_set_name": result.get("evaluation_set_name"),
        "n_rows": result.get("n_rows"),
    }


def _summarize_schema_invariance(analysis: dict[str, Any]) -> dict[str, Any] | None:
    si = analysis.get("schema_invariance")
    if not isinstance(si, dict):
        return None
    return {
        "status": si.get("status"),
        "label": si.get("label"),
        "note": si.get("note"),
    }


def _summarize_joint(analysis: dict[str, Any]) -> dict[str, Any] | None:
    joint = analysis.get("joint_interpretation")
    if not isinstance(joint, dict):
        return None
    return {
        "joint_success": joint.get("joint_success"),
        "summary": joint.get("summary"),
    }


def _load_eligible_panel(eligible_panel_path: Path | None) -> dict[str, dict[str, Any]]:
    if eligible_panel_path is None:
        return {}
    if not eligible_panel_path.exists():
        logger.warning("Eligible panel file not found: %s", eligible_panel_path)
        return {}
    payload = json.loads(eligible_panel_path.read_text(encoding="utf-8"))
    by_id: dict[str, dict[str, Any]] = {}
    for key in (
        "all_candidate_results",
        "eligible_candidate_results",
        "ineligible_candidate_results",
    ):
        for entry in payload.get(key, []) or []:
            cid = entry.get("candidate_id")
            if cid:
                by_id.setdefault(cid, entry)
    return by_id


def discover_candidate_dirs(panel_root: Path) -> list[tuple[str, str, Path]]:
    """Return list of ``(corpus_b_variant, candidate_id, candidate_dir)`` triples."""
    triples: list[tuple[str, str, Path]] = []
    if not panel_root.exists():
        return triples
    for variant_dir in sorted(p for p in panel_root.iterdir() if p.is_dir()):
        for candidate_dir in sorted(p for p in variant_dir.iterdir() if p.is_dir()):
            triples.append((variant_dir.name, candidate_dir.name, candidate_dir))
    return triples


def summarize_candidate(
    *,
    corpus_b_variant: str,
    candidate_id: str,
    candidate_dir: Path,
    eligible_panel_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    analysis_path = _analysis_path(candidate_dir)
    summary: dict[str, Any] = {
        "candidate_id": candidate_id,
        "corpus_b_variant": corpus_b_variant,
        "candidate_dir": str(candidate_dir),
        "analysis_path": str(analysis_path),
        "status": "missing",
        "delta_vs_no_prompt": None,
        "candidate_rank": None,
        "h1": None,
        "h1c": None,
        "h2": None,
        "schema_invariance": None,
        "joint": None,
    }
    panel_entry = eligible_panel_by_id.get(candidate_id)
    if panel_entry is not None:
        summary["delta_vs_no_prompt"] = panel_entry.get("delta_vs_no_prompt")
        summary["candidate_rank"] = panel_entry.get("rank")

    if not analysis_path.exists():
        return summary

    try:
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read %s: %s", analysis_path, exc)
        summary["status"] = "unreadable"
        summary["error"] = str(exc)
        return summary

    confirmatory = analysis.get("confirmatory_results") or []
    summary["status"] = "present"
    summary["h1"] = _summarize_hypothesis(_find_by_hypothesis(confirmatory, "H1"))
    summary["h1c"] = _summarize_hypothesis(_find_by_hypothesis(confirmatory, "H1c"))
    summary["h2"] = _summarize_hypothesis(_find_by_hypothesis(confirmatory, "H2"))
    summary["schema_invariance"] = _summarize_schema_invariance(analysis)
    summary["joint"] = _summarize_joint(analysis)
    return summary


def _summarize_panel(candidate_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    present = [c for c in candidate_summaries if c["status"] == "present"]
    h1_supported = [c for c in present if (c.get("h1") or {}).get("support_status") == "supported"]
    effect_sizes = [
        (c["h1"] or {}).get("marginal_risk_difference")
        for c in present
        if c.get("h1") is not None
        and (c["h1"] or {}).get("marginal_risk_difference") is not None
    ]
    if effect_sizes:
        sorted_sizes = sorted(float(x) for x in effect_sizes)
        n = len(sorted_sizes)
        median = (
            sorted_sizes[n // 2]
            if n % 2 == 1
            else 0.5 * (sorted_sizes[n // 2 - 1] + sorted_sizes[n // 2])
        )
        effect_distribution = {
            "n": n,
            "min": sorted_sizes[0],
            "max": sorted_sizes[-1],
            "median": median,
            "mean": sum(sorted_sizes) / n,
            "values": sorted_sizes,
        }
    else:
        effect_distribution = {
            "n": 0,
            "min": None,
            "max": None,
            "median": None,
            "mean": None,
            "values": [],
        }
    n_total = len(candidate_summaries)
    n_present = len(present)
    n_supported = len(h1_supported)
    return {
        "n_candidates_total": n_total,
        "n_candidates_present": n_present,
        "n_candidates_missing": n_total - n_present,
        "n_candidates_h1_supported": n_supported,
        "proportion_h1_supported": (
            n_supported / n_present if n_present else None
        ),
        "h1_effect_size_distribution": effect_distribution,
        "supported_candidate_ids": [c["candidate_id"] for c in h1_supported],
    }


def build_prompt_panel_payload(
    *,
    panel_root: Path,
    eligible_panel_path: Path | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    triples = discover_candidate_dirs(panel_root)
    eligible_panel_by_id = _load_eligible_panel(eligible_panel_path)
    candidate_summaries = [
        summarize_candidate(
            corpus_b_variant=variant,
            candidate_id=cid,
            candidate_dir=cdir,
            eligible_panel_by_id=eligible_panel_by_id,
        )
        for (variant, cid, cdir) in triples
    ]
    if strict:
        missing = [c for c in candidate_summaries if c["status"] != "present"]
        if missing:
            ids = ", ".join(f"{c['corpus_b_variant']}/{c['candidate_id']}" for c in missing)
            raise SystemExit(f"--strict: missing/unreadable candidate analyses: {ids}")
    return {
        "workflow_name": "prompt_panel_aggregate_analysis",
        "schema_version": PROMPT_PANEL_SCHEMA_VERSION,
        "panel_root": str(panel_root),
        "eligible_panel_path": str(eligible_panel_path) if eligible_panel_path else None,
        "panel_summary": _summarize_panel(candidate_summaries),
        "candidate_summaries": candidate_summaries,
        "note": (
            "Pure aggregator over per-candidate prereg_analysis.json artifacts. "
            "Missing analyses are tolerated unless --strict is passed."
        ),
    }


def _build_markdown(payload: dict[str, Any]) -> str:
    s = payload["panel_summary"]
    lines = [
        "# Prompt-Panel Aggregate Analysis",
        "",
        f"Panel root: `{payload['panel_root']}`",
        "",
        "## Panel summary",
        "",
        f"- Candidates total: {s['n_candidates_total']}",
        f"- Candidates with analyses present: {s['n_candidates_present']}",
        f"- Candidates missing analyses: {s['n_candidates_missing']}",
        f"- Candidates with supported H1 reduction: {s['n_candidates_h1_supported']}",
    ]
    prop = s.get("proportion_h1_supported")
    lines.append(
        f"- Proportion supported (of present): "
        f"{('N/A' if prop is None else f'{prop:.3f}')}"
    )
    dist = s["h1_effect_size_distribution"]
    if dist["n"]:
        lines += [
            "",
            "### H1 effect-size distribution (marginal risk difference)",
            "",
            f"- n: {dist['n']}",
            f"- min: {dist['min']:.4f}",
            f"- max: {dist['max']:.4f}",
            f"- median: {dist['median']:.4f}",
            f"- mean: {dist['mean']:.4f}",
        ]

    lines += ["", "## Per-candidate results", ""]
    lines += [
        "| Variant | Candidate | Status | Δ vs no-prompt | Rank | H1 status | H1 MRD | H1c status | H2 status | Schema-invariance | Joint success |",
        "|---------|-----------|--------|---------------:|----:|-----------|------:|------------|-----------|-------------------|---------------|",
    ]
    for c in payload["candidate_summaries"]:
        h1 = c.get("h1") or {}
        h1c = c.get("h1c") or {}
        h2 = c.get("h2") or {}
        si = c.get("schema_invariance") or {}
        joint = c.get("joint") or {}
        delta = c.get("delta_vs_no_prompt")
        delta_str = "N/A" if delta is None else f"{float(delta):.4f}"
        mrd = h1.get("marginal_risk_difference")
        mrd_str = "N/A" if mrd is None else f"{float(mrd):.4f}"
        lines.append(
            f"| {c['corpus_b_variant']} | {c['candidate_id']} | {c['status']} "
            f"| {delta_str} | {c.get('candidate_rank') or '—'} "
            f"| {h1.get('support_status') or '—'} | {mrd_str} "
            f"| {h1c.get('support_status') or '—'} "
            f"| {h2.get('support_status') or '—'} "
            f"| {si.get('status') or '—'} "
            f"| {joint.get('joint_success') if 'joint_success' in joint else '—'} |"
        )
    lines += ["", payload["note"], ""]
    return "\n".join(lines) + "\n"


def write_outputs(
    payload: dict[str, Any],
    output_prefix: Path,
    *,
    input_paths: Iterable[Path],
    argv: list[str],
) -> dict[str, Path]:
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_prefix.with_name(output_prefix.name + ".json")
    md_path = output_prefix.with_name(output_prefix.name + ".md")
    provenance = build_provenance(
        input_paths=list(input_paths),
        argv=argv,
        schema_version=PROMPT_PANEL_SCHEMA_VERSION,
        repo_root=PROJECTS_DIR.parent,
    )
    write_json_with_provenance(json_path, payload, provenance)
    md_path.write_text(_build_markdown(payload), encoding="utf-8")
    return {"json": json_path, "md": md_path}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate per-candidate prereg analyses across a prompt panel."
    )
    parser.add_argument(
        "--panel-root",
        required=True,
        type=Path,
        help="Root directory containing <variant>/<candidate_id>/reports/prereg_analysis.json.",
    )
    parser.add_argument(
        "--eligible-panel",
        type=Path,
        default=None,
        help="Optional eligible_train_user_suffixes.json for delta_vs_no_prompt and rank.",
    )
    parser.add_argument(
        "--output-prefix",
        required=True,
        type=Path,
        help="Output prefix; writes <prefix>.json and <prefix>.md.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail nonzero if any candidate analysis is missing or unreadable.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    payload = build_prompt_panel_payload(
        panel_root=args.panel_root,
        eligible_panel_path=args.eligible_panel,
        strict=args.strict,
    )
    inputs: list[Path] = []
    if args.eligible_panel and args.eligible_panel.exists():
        inputs.append(args.eligible_panel)
    write_outputs(
        payload,
        args.output_prefix,
        input_paths=inputs,
        argv=sys.argv if argv is None else ["analyze_prompt_panel_effects.py", *argv],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
