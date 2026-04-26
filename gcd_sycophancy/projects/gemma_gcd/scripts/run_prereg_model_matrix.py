#!/usr/bin/env python3
"""Run the preregistered GCD study across a configured matrix of models.

Each model is specified in a JSON config with shape::

    {
      "models": [
        {
          "model_id": "gemma_2b_it",
          "model_name": "google/gemma-2b-it",
          "template_config": "experiments/ip_sweep/config.json"
        },
        ...
      ]
    }

For each entry the runner invokes ``run_preregistration.py`` with an isolated
experiment directory under ``experiments/prereg_model_matrix/<model_id>/`` and
records per-model success or failure in a matrix manifest.  After all models
complete (or under ``--aggregate-only``) the runner reads each model's
``prereg_analysis.json`` and writes summary artifacts.

Failures per model are recorded.  Without ``--fail-fast`` the runner continues
to the next model after a failure; with ``--fail-fast`` the runner aborts on
the first failing model.

Outputs (relative to the experiment root):
    model_matrix_manifest.json
    model_matrix_summary.json
    model_matrix_summary.md
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECTS_DIR = SCRIPT_DIR.parents[1]

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from artifact_provenance import build_provenance  # noqa: E402

MODEL_MATRIX_SCHEMA_VERSION = "1"
DEFAULT_EXPERIMENT_ROOT = Path("experiments/prereg_model_matrix")
DEFAULT_PHASES = ("setup", "train", "preflight", "fixed-interface-eval", "analysis")
MATRIX_MANIFEST_NAME = "model_matrix_manifest.json"
MATRIX_SUMMARY_JSON_NAME = "model_matrix_summary.json"
MATRIX_SUMMARY_MD_NAME = "model_matrix_summary.md"

_SYCOPHANCY_ANALYSIS_ID = "analysis_1"
_DIRECT_SOLVE_ANALYSIS_ID = "analysis_2"


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


def model_experiment_dir(experiment_root: Path, model_id: str) -> Path:
    return experiment_root / model_id


def load_model_config(config_path: Path) -> list[dict[str, Any]]:
    """Load and validate a model-matrix JSON config."""
    if not config_path.exists():
        raise FileNotFoundError(f"Model-matrix config not found: {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    models = payload.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError(
            f"Model-matrix config {config_path} must contain a non-empty 'models' list."
        )
    seen_ids: set[str] = set()
    validated: list[dict[str, Any]] = []
    for index, entry in enumerate(models):
        if not isinstance(entry, dict):
            raise ValueError(f"Model entry {index} is not an object: {entry!r}")
        for required in ("model_id", "model_name"):
            if required not in entry or not entry[required]:
                raise ValueError(
                    f"Model entry {index} is missing required field {required!r}: {entry!r}"
                )
        model_id = entry["model_id"]
        if model_id in seen_ids:
            raise ValueError(f"Duplicate model_id in matrix config: {model_id!r}")
        seen_ids.add(model_id)
        validated.append(copy.deepcopy(entry))
    return validated


def build_prereg_command(
    *,
    phase: str,
    model_entry: dict[str, Any],
    experiment_dir: Path,
    seeds: tuple[int, ...] | None,
    passthrough_args: list[str],
) -> list[str]:
    cmd: list[str] = [
        sys.executable,
        str(SCRIPT_DIR / "run_preregistration.py"),
        phase,
        "--experiment-dir",
        str(experiment_dir),
    ]
    template = model_entry.get("template_config")
    if template:
        cmd += ["--template-config", str(template)]
    if seeds is not None:
        cmd += ["--seeds", *[str(s) for s in seeds]]
    cmd.extend(passthrough_args)
    return cmd


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _analysis_json_path(model_dir: Path) -> Path:
    return model_dir / "reports" / "prereg_analysis.json"


def _extract_key_metrics(analysis: dict[str, Any]) -> dict[str, Any]:
    all_results: list[dict[str, Any]] = list(analysis.get("confirmatory_results", [])) + list(
        analysis.get("exploratory_results", [])
    )
    by_id = {r["analysis_id"]: r for r in all_results}
    metrics: dict[str, Any] = {}
    if _SYCOPHANCY_ANALYSIS_ID in by_id:
        r = by_id[_SYCOPHANCY_ANALYSIS_ID]
        metrics["sycophancy_arm2_rate"] = r.get("arm_a_observed_rate")
        metrics["sycophancy_arm1_rate"] = r.get("arm_b_observed_rate")
        metrics["sycophancy_mrd"] = r.get("marginal_risk_difference")
        metrics["sycophancy_support_status"] = r.get("support_status")
    if _DIRECT_SOLVE_ANALYSIS_ID in by_id:
        r = by_id[_DIRECT_SOLVE_ANALYSIS_ID]
        metrics["direct_solve_mrd"] = r.get("marginal_risk_difference")
        metrics["direct_solve_support_status"] = r.get("support_status")
    return metrics


def read_model_results(
    models: list[dict[str, Any]],
    experiment_root: Path,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for entry in models:
        model_id = entry["model_id"]
        mdir = model_experiment_dir(experiment_root, model_id)
        aj = _analysis_json_path(mdir)
        if not aj.exists():
            results[model_id] = {
                "status": "missing",
                "model_name": entry.get("model_name"),
                "analysis_path": str(aj),
            }
            continue
        analysis = json.loads(aj.read_text(encoding="utf-8"))
        results[model_id] = {
            "status": "present",
            "model_name": entry.get("model_name"),
            "analysis_path": str(aj),
            "key_metrics": _extract_key_metrics(analysis),
        }
    return results


def _build_summary_md(summary: dict[str, Any]) -> str:
    lines = [
        "# Model Matrix Summary",
        "",
        f"Generated: {summary['generated_at']}",
        "",
        "Each model is run as an independent preregistered study; results below are "
        "descriptive across models and should not be pooled without an explicit "
        "meta-analytic plan.",
        "",
    ]
    for model_id in summary["models"]:
        lines += [f"## Model `{model_id}`", ""]
        info = summary["model_results"].get(model_id, {})
        if info.get("status") == "missing":
            lines += [
                f"_Analysis not yet available._  "
                f"Expected: `{info.get('analysis_path', 'unknown')}`",
                "",
            ]
            continue
        m = info.get("key_metrics", {})
        lines += [
            f"Model name: `{info.get('model_name', '')}`",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| H1 sycophancy MRD (Arm 2 − Arm 1) | {_fmt_rate(m.get('sycophancy_mrd'))} |",
            f"| H1 support status | {m.get('sycophancy_support_status', 'N/A')} |",
            f"| H2 capability MRD (direct-solve) | {_fmt_rate(m.get('direct_solve_mrd'))} |",
            f"| H2 support status | {m.get('direct_solve_support_status', 'N/A')} |",
            "",
        ]
    return "\n".join(lines) + "\n"


def _build_summary_provenance(
    *,
    models: list[dict[str, Any]],
    model_results: dict[str, dict[str, Any]],
    config_path: Path | None,
    argv: list[str] | None,
) -> dict[str, Any]:
    input_paths: list[Path] = []
    if config_path is not None and config_path.exists():
        input_paths.append(config_path)
    for entry in models:
        info = model_results.get(entry["model_id"], {})
        if info.get("status") == "present":
            ap = info.get("analysis_path")
            if ap and Path(ap).exists():
                input_paths.append(Path(ap))
    return build_provenance(
        input_paths=input_paths,
        argv=argv if argv is not None else sys.argv,
        schema_version=MODEL_MATRIX_SCHEMA_VERSION,
        repo_root=PROJECTS_DIR.parent,
    )


def write_matrix_summary(
    *,
    experiment_root: Path,
    models: list[dict[str, Any]],
    model_results: dict[str, dict[str, Any]],
    generated_at: str,
    config_path: Path | None,
    argv: list[str] | None = None,
) -> tuple[Path, Path]:
    summary: dict[str, Any] = {
        "workflow_name": "prereg_model_matrix_summary",
        "schema_version": MODEL_MATRIX_SCHEMA_VERSION,
        "generated_at": generated_at,
        "models": [m["model_id"] for m in models],
        "model_results": model_results,
        "provenance": _build_summary_provenance(
            models=models,
            model_results=model_results,
            config_path=config_path,
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
    models: list[dict[str, Any]],
    seeds: tuple[int, ...] | None,
    phases: tuple[str, ...],
    config_path: Path | None,
    started_at: str,
    completed_at: str | None,
    dry_run: bool,
    fail_fast: bool,
    model_statuses: dict[str, str],
    argv: list[str] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "workflow_name": "prereg_model_matrix",
        "schema_version": MODEL_MATRIX_SCHEMA_VERSION,
        "experiment_root": str(experiment_root),
        "models": models,
        "model_dirs": {
            m["model_id"]: str(model_experiment_dir(experiment_root, m["model_id"]))
            for m in models
        },
        "seeds": list(seeds) if seeds is not None else None,
        "phases": list(phases),
        "config_path": str(config_path) if config_path is not None else None,
        "dry_run": dry_run,
        "fail_fast": fail_fast,
        "started_at": started_at,
        "completed_at": completed_at,
        "model_statuses": model_statuses,
        "provenance": build_provenance(
            input_paths=[config_path] if config_path is not None and config_path.exists() else [],
            argv=argv if argv is not None else sys.argv,
            schema_version=MODEL_MATRIX_SCHEMA_VERSION,
            repo_root=PROJECTS_DIR.parent,
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_matrix(
    *,
    experiment_root: Path,
    models: list[dict[str, Any]],
    seeds: tuple[int, ...] | None,
    phases: tuple[str, ...],
    config_path: Path | None,
    dry_run: bool,
    aggregate_only: bool,
    fail_fast: bool,
    passthrough_args: list[str],
) -> int:
    started_at = _now_iso()
    manifest_path = experiment_root / MATRIX_MANIFEST_NAME
    model_statuses: dict[str, str] = {}

    if not aggregate_only:
        for entry in models:
            model_id = entry["model_id"]
            exp_dir = model_experiment_dir(experiment_root, model_id)
            if not dry_run:
                exp_dir.mkdir(parents=True, exist_ok=True)
            failed_phase: str | None = None
            for phase in phases:
                cmd = build_prereg_command(
                    phase=phase,
                    model_entry=entry,
                    experiment_dir=exp_dir,
                    seeds=seeds,
                    passthrough_args=passthrough_args,
                )
                if dry_run:
                    print("[DRY-RUN]", " ".join(str(t) for t in cmd))
                    continue
                try:
                    subprocess.run(cmd, check=True)
                except subprocess.CalledProcessError as exc:
                    failed_phase = f"{phase}:{exc.returncode}"
                    break
            if dry_run:
                model_statuses[model_id] = "dry_run"
                continue
            if failed_phase is None:
                model_statuses[model_id] = "completed"
            else:
                model_statuses[model_id] = f"failed:{failed_phase}"
                if fail_fast:
                    break

    _write_matrix_manifest(
        manifest_path,
        experiment_root=experiment_root,
        models=models,
        seeds=seeds,
        phases=phases,
        config_path=config_path,
        started_at=started_at,
        completed_at=_now_iso(),
        dry_run=dry_run,
        fail_fast=fail_fast,
        model_statuses=model_statuses,
    )

    generated_at = _now_iso()
    model_results = read_model_results(models, experiment_root)
    write_matrix_summary(
        experiment_root=experiment_root,
        models=models,
        model_results=model_results,
        generated_at=generated_at,
        config_path=config_path,
    )

    any_failed = any(s.startswith("failed:") for s in model_statuses.values())
    return 1 if any_failed else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the preregistered GCD study across a JSON-configured matrix of models. "
            "Each model runs in an isolated experiment directory."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to JSON config listing models to run.",
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=DEFAULT_EXPERIMENT_ROOT,
        help="Root directory under which per-model subdirectories are created.",
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        default=list(DEFAULT_PHASES),
        help="Phases to run per model.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Optional seeds to forward to run_preregistration.py.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands and write manifest+summary without executing subprocesses.",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Skip subprocess runs; only read existing analysis outputs and write the summary.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Abort on the first failing model. Without this flag, processing continues.",
    )
    parser.add_argument(
        "--passthrough",
        nargs=argparse.REMAINDER,
        default=[],
        help="Remaining arguments are forwarded verbatim to run_preregistration.py.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config_path = _resolve(args.config)
    experiment_root = _resolve(args.experiment_root)
    models = load_model_config(config_path)
    seeds = tuple(args.seeds) if args.seeds is not None else None
    return run_matrix(
        experiment_root=experiment_root,
        models=models,
        seeds=seeds,
        phases=tuple(args.phases),
        config_path=config_path,
        dry_run=bool(args.dry_run),
        aggregate_only=bool(args.aggregate_only),
        fail_fast=bool(args.fail_fast),
        passthrough_args=list(args.passthrough or []),
    )


if __name__ == "__main__":
    raise SystemExit(main())
