#!/usr/bin/env python3
"""Run the preregistered GCD study across a matrix of training-epoch counts.

Each ``--epochs`` value runs in an isolated experiment directory under
``experiments/prereg_epoch_matrix/epochs_<n>/``.  For each epoch count the
runner writes a derived training config (a copy of the source template with
``finetune_config.epochs`` overridden) into the per-epoch directory and
invokes ``run_preregistration.py`` with that derived config.

The shared template config is never mutated in place.  The source template's
on-disk SHA-256 is recorded in the matrix manifest before and after the run so
unintended template mutation is detectable.

Outputs (relative to the experiment root):
    epoch_matrix_manifest.json
    epoch_matrix_summary.json
    epoch_matrix_summary.md
"""
from __future__ import annotations

import argparse
import copy
import hashlib
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

EPOCH_MATRIX_SCHEMA_VERSION = "1"
DEFAULT_EXPERIMENT_ROOT = Path("experiments/prereg_epoch_matrix")
DEFAULT_TEMPLATE_CONFIG = Path("experiments/ip_sweep/config.json")
DEFAULT_PHASES = ("setup", "train", "preflight", "fixed-interface-eval", "analysis")
MATRIX_MANIFEST_NAME = "epoch_matrix_manifest.json"
MATRIX_SUMMARY_JSON_NAME = "epoch_matrix_summary.json"
MATRIX_SUMMARY_MD_NAME = "epoch_matrix_summary.md"
DERIVED_CONFIG_NAME = "config.json"

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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def epoch_experiment_dir(experiment_root: Path, epochs: int) -> Path:
    return experiment_root / f"epochs_{epochs}"


def write_derived_template_config(
    *,
    template_path: Path,
    derived_path: Path,
    epochs: int,
) -> dict[str, Any]:
    """Copy ``template_path`` to ``derived_path``, overriding ``finetune_config.epochs``.

    Returns the derived config payload.  Never mutates ``template_path``.
    """
    if not template_path.exists():
        raise FileNotFoundError(f"Source template config not found: {template_path}")
    payload = json.loads(template_path.read_text(encoding="utf-8"))
    payload = copy.deepcopy(payload)
    if "finetune_config" not in payload or not isinstance(payload["finetune_config"], dict):
        raise ValueError(
            f"Template config {template_path} must contain a 'finetune_config' object."
        )
    payload["finetune_config"]["epochs"] = epochs
    derived_path.parent.mkdir(parents=True, exist_ok=True)
    derived_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def build_prereg_command(
    *,
    phase: str,
    experiment_dir: Path,
    derived_template: Path,
    seeds: tuple[int, ...] | None,
    passthrough_args: list[str],
) -> list[str]:
    cmd: list[str] = [
        sys.executable,
        str(SCRIPT_DIR / "run_preregistration.py"),
        phase,
        "--experiment-dir",
        str(experiment_dir),
        "--template-config",
        str(derived_template),
    ]
    if seeds is not None:
        cmd += ["--seeds", *[str(s) for s in seeds]]
    cmd.extend(passthrough_args)
    return cmd


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _analysis_json_path(epoch_dir: Path) -> Path:
    return epoch_dir / "reports" / "prereg_analysis.json"


def _extract_key_metrics(analysis: dict[str, Any]) -> dict[str, Any]:
    all_results = list(analysis.get("confirmatory_results", [])) + list(
        analysis.get("exploratory_results", [])
    )
    by_id = {r["analysis_id"]: r for r in all_results}
    metrics: dict[str, Any] = {}
    if _SYCOPHANCY_ANALYSIS_ID in by_id:
        r = by_id[_SYCOPHANCY_ANALYSIS_ID]
        metrics["sycophancy_mrd"] = r.get("marginal_risk_difference")
        metrics["sycophancy_support_status"] = r.get("support_status")
    if _DIRECT_SOLVE_ANALYSIS_ID in by_id:
        r = by_id[_DIRECT_SOLVE_ANALYSIS_ID]
        metrics["direct_solve_mrd"] = r.get("marginal_risk_difference")
        metrics["direct_solve_support_status"] = r.get("support_status")
    return metrics


def read_epoch_results(
    epochs_list: tuple[int, ...],
    experiment_root: Path,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for epochs in epochs_list:
        edir = epoch_experiment_dir(experiment_root, epochs)
        aj = _analysis_json_path(edir)
        key = f"epochs_{epochs}"
        if not aj.exists():
            results[key] = {
                "status": "missing",
                "epochs": epochs,
                "analysis_path": str(aj),
            }
            continue
        analysis = json.loads(aj.read_text(encoding="utf-8"))
        results[key] = {
            "status": "present",
            "epochs": epochs,
            "analysis_path": str(aj),
            "key_metrics": _extract_key_metrics(analysis),
        }
    return results


def _build_summary_md(summary: dict[str, Any]) -> str:
    lines = [
        "# Epoch Matrix Summary",
        "",
        f"Generated: {summary['generated_at']}",
        "",
        "Each epoch count is run as an independent preregistered study.  Cross-epoch "
        "comparisons below are descriptive and should not be pooled without an explicit "
        "meta-analytic plan.",
        "",
        "| Epochs | H1 sycophancy MRD | H1 status | H2 capability MRD | H2 status |",
        "|--------|------------------|-----------|-------------------|-----------|",
    ]
    for epochs in summary["epochs"]:
        key = f"epochs_{epochs}"
        info = summary["epoch_results"].get(key, {})
        if info.get("status") == "missing":
            lines += [f"| {epochs} | _missing_ | _missing_ | _missing_ | _missing_ |"]
            continue
        m = info.get("key_metrics", {})
        lines += [
            f"| {epochs} "
            f"| {_fmt_rate(m.get('sycophancy_mrd'))} "
            f"| {m.get('sycophancy_support_status', 'N/A')} "
            f"| {_fmt_rate(m.get('direct_solve_mrd'))} "
            f"| {m.get('direct_solve_support_status', 'N/A')} |"
        ]
    return "\n".join(lines) + "\n"


def _build_summary_provenance(
    *,
    epoch_results: dict[str, dict[str, Any]],
    template_path: Path | None,
    argv: list[str] | None,
) -> dict[str, Any]:
    input_paths: list[Path] = []
    if template_path is not None and template_path.exists():
        input_paths.append(template_path)
    for info in epoch_results.values():
        if info.get("status") == "present":
            ap = info.get("analysis_path")
            if ap and Path(ap).exists():
                input_paths.append(Path(ap))
    return build_provenance(
        input_paths=input_paths,
        argv=argv if argv is not None else sys.argv,
        schema_version=EPOCH_MATRIX_SCHEMA_VERSION,
        repo_root=PROJECTS_DIR.parent,
    )


def write_matrix_summary(
    *,
    experiment_root: Path,
    epochs_list: tuple[int, ...],
    epoch_results: dict[str, dict[str, Any]],
    generated_at: str,
    template_path: Path | None,
    argv: list[str] | None = None,
) -> tuple[Path, Path]:
    summary: dict[str, Any] = {
        "workflow_name": "prereg_epoch_matrix_summary",
        "schema_version": EPOCH_MATRIX_SCHEMA_VERSION,
        "generated_at": generated_at,
        "epochs": list(epochs_list),
        "epoch_results": epoch_results,
        "provenance": _build_summary_provenance(
            epoch_results=epoch_results,
            template_path=template_path,
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
    epochs_list: tuple[int, ...],
    seeds: tuple[int, ...] | None,
    phases: tuple[str, ...],
    template_path: Path,
    template_sha256_before: str | None,
    template_sha256_after: str | None,
    derived_configs: dict[str, str],
    started_at: str,
    completed_at: str | None,
    dry_run: bool,
    fail_fast: bool,
    epoch_statuses: dict[str, str],
    argv: list[str] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "workflow_name": "prereg_epoch_matrix",
        "schema_version": EPOCH_MATRIX_SCHEMA_VERSION,
        "experiment_root": str(experiment_root),
        "epochs": list(epochs_list),
        "epoch_dirs": {
            f"epochs_{n}": str(epoch_experiment_dir(experiment_root, n)) for n in epochs_list
        },
        "derived_configs": derived_configs,
        "seeds": list(seeds) if seeds is not None else None,
        "phases": list(phases),
        "template_config": str(template_path),
        "template_sha256_before": template_sha256_before,
        "template_sha256_after": template_sha256_after,
        "template_unchanged": (
            template_sha256_before is not None
            and template_sha256_before == template_sha256_after
        ),
        "dry_run": dry_run,
        "fail_fast": fail_fast,
        "started_at": started_at,
        "completed_at": completed_at,
        "epoch_statuses": epoch_statuses,
        "provenance": build_provenance(
            input_paths=[template_path] if template_path.exists() else [],
            argv=argv if argv is not None else sys.argv,
            schema_version=EPOCH_MATRIX_SCHEMA_VERSION,
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
    epochs_list: tuple[int, ...],
    seeds: tuple[int, ...] | None,
    phases: tuple[str, ...],
    template_path: Path,
    dry_run: bool,
    aggregate_only: bool,
    fail_fast: bool,
    passthrough_args: list[str],
) -> int:
    started_at = _now_iso()
    manifest_path = experiment_root / MATRIX_MANIFEST_NAME
    epoch_statuses: dict[str, str] = {}
    derived_configs: dict[str, str] = {}

    template_sha_before = _sha256_file(template_path) if template_path.exists() else None

    if not aggregate_only:
        for epochs in epochs_list:
            edir = epoch_experiment_dir(experiment_root, epochs)
            edir.mkdir(parents=True, exist_ok=True)
            derived_path = edir / DERIVED_CONFIG_NAME
            # Always write the derived config, even in dry-run, so callers can inspect it.
            write_derived_template_config(
                template_path=template_path,
                derived_path=derived_path,
                epochs=epochs,
            )
            derived_configs[f"epochs_{epochs}"] = str(derived_path)

            failed_phase: str | None = None
            for phase in phases:
                cmd = build_prereg_command(
                    phase=phase,
                    experiment_dir=edir,
                    derived_template=derived_path,
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
                epoch_statuses[f"epochs_{epochs}"] = "dry_run"
                continue
            if failed_phase is None:
                epoch_statuses[f"epochs_{epochs}"] = "completed"
            else:
                epoch_statuses[f"epochs_{epochs}"] = f"failed:{failed_phase}"
                if fail_fast:
                    break

    template_sha_after = _sha256_file(template_path) if template_path.exists() else None

    _write_matrix_manifest(
        manifest_path,
        experiment_root=experiment_root,
        epochs_list=epochs_list,
        seeds=seeds,
        phases=phases,
        template_path=template_path,
        template_sha256_before=template_sha_before,
        template_sha256_after=template_sha_after,
        derived_configs=derived_configs,
        started_at=started_at,
        completed_at=_now_iso(),
        dry_run=dry_run,
        fail_fast=fail_fast,
        epoch_statuses=epoch_statuses,
    )

    epoch_results = read_epoch_results(epochs_list, experiment_root)
    write_matrix_summary(
        experiment_root=experiment_root,
        epochs_list=epochs_list,
        epoch_results=epoch_results,
        generated_at=_now_iso(),
        template_path=template_path,
    )

    any_failed = any(s.startswith("failed:") for s in epoch_statuses.values())
    return 1 if any_failed else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the preregistered GCD study under a matrix of training-epoch counts. "
            "Each epoch count runs in an isolated experiment directory and uses a "
            "derived training config; the source template config is never mutated."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--epochs",
        nargs="+",
        type=int,
        required=True,
        help="Epoch counts to run (variadic). Example: --epochs 1 2 3 5.",
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=DEFAULT_EXPERIMENT_ROOT,
    )
    parser.add_argument(
        "--template-config",
        type=Path,
        default=DEFAULT_TEMPLATE_CONFIG,
        help="Source template config; copied (with epochs override) into each per-epoch dir.",
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        default=list(DEFAULT_PHASES),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--passthrough",
        nargs=argparse.REMAINDER,
        default=[],
        help="Forwarded verbatim to run_preregistration.py.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    experiment_root = _resolve(args.experiment_root)
    template_path = _resolve(args.template_config)
    if not args.epochs:
        raise SystemExit("ERROR: --epochs requires at least one value.")
    if any(n <= 0 for n in args.epochs):
        raise SystemExit("ERROR: --epochs values must be positive integers.")
    seeds = tuple(args.seeds) if args.seeds is not None else None
    return run_matrix(
        experiment_root=experiment_root,
        epochs_list=tuple(args.epochs),
        seeds=seeds,
        phases=tuple(args.phases),
        template_path=template_path,
        dry_run=bool(args.dry_run),
        aggregate_only=bool(args.aggregate_only),
        fail_fast=bool(args.fail_fast),
        passthrough_args=list(args.passthrough or []),
    )


if __name__ == "__main__":
    raise SystemExit(main())
