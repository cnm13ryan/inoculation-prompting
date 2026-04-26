#!/usr/bin/env python3
"""Higher-level orchestrator that runs the preregistered GCD study for every
eligible inoculation-prompt candidate in a frozen eligible-panel artifact.

Each eligible candidate receives its own isolated experiment directory so that
training and evaluation artifacts never co-mingle across IP instructions.

Usage examples
--------------
# Dry-run to preview planned commands and write the manifest:
    python run_prereg_prompt_panel.py --dry-run

# Limit to two candidates for a smoke test:
    python run_prereg_prompt_panel.py --dry-run --limit-candidates 2

# Run setup only for all eligible candidates (b2 variant):
    python run_prereg_prompt_panel.py \\
        --phases setup \\
        --corpus-b-variant b2 \\
        --tensor-parallel-size 4
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECTS_DIR = SCRIPT_DIR.parents[2]

DEFAULT_ELIGIBLE_PANEL = Path("experiments/ip_sweep/eligible_train_user_suffixes.json")
DEFAULT_EXPERIMENT_ROOT = Path("experiments/prereg_prompt_panel")
DEFAULT_PHASES = ("setup", "train", "preflight", "fixed-interface-eval", "analysis")
DEFAULT_SEEDS = (0, 1, 2, 3)
DEFAULT_CORPUS_B_VARIANT = "b1"

PANEL_MANIFEST_NAME = "prompt_panel_manifest.json"

_VALID_PHASE_CHOICES = (
    "materialize-data",
    "setup",
    "train",
    "preflight",
    "fixed-interface-eval",
    "semantic-interface-eval",
    "prefix-search",
    "best-elicited-eval",
    "analysis",
    "seed-instability",
    "full",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _resolve(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (PROJECTS_DIR / path).resolve()


def _sanitize_candidate_id(candidate_id: str) -> str:
    """Return a filesystem-safe directory name derived from candidate_id.

    Replaces non-alphanumeric characters (except hyphen) with underscores and
    collapses consecutive underscores so the result is stable and collision-safe.
    """
    sanitized = re.sub(r"[^A-Za-z0-9_-]", "_", candidate_id)
    sanitized = re.sub(r"_+", "_", sanitized)
    return sanitized.strip("_")


def load_eligible_panel(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def extract_eligible_candidates(panel: dict[str, Any]) -> list[dict[str, Any]]:
    return list(panel.get("eligible_candidate_results", []))


def check_candidate_id_collisions(candidates: list[dict[str, Any]]) -> None:
    """Raise ValueError if any two candidates map to the same sanitized directory name."""
    seen: dict[str, str] = {}
    for c in candidates:
        original = c["candidate_id"]
        sid = _sanitize_candidate_id(original)
        if sid in seen:
            raise ValueError(
                f"Candidate ID collision after sanitization: "
                f"{original!r} and {seen[sid]!r} both map to {sid!r}. "
                "Rename one candidate before running the panel."
            )
        seen[sid] = original


def candidate_experiment_dir(
    experiment_root: Path,
    corpus_b_variant: str,
    candidate: dict[str, Any],
) -> Path:
    sid = _sanitize_candidate_id(candidate["candidate_id"])
    return experiment_root / corpus_b_variant / sid


def _candidate_suffix_text(candidate: dict[str, Any]) -> str:
    return candidate.get("train_user_suffix") or candidate.get("suffix_text", "")


def build_prereg_command(
    *,
    phase: str,
    experiment_dir: Path,
    candidate: dict[str, Any],
    corpus_b_variant: str,
    seeds: tuple[int, ...],
    passthrough_args: list[str],
) -> list[str]:
    cmd: list[str] = [
        sys.executable,
        str(SCRIPT_DIR / "run_preregistration.py"),
        phase,
        "--experiment-dir",
        str(experiment_dir),
        "--ip-instruction",
        _candidate_suffix_text(candidate),
        "--ip-instruction-id",
        candidate["candidate_id"],
        "--corpus-b-variant",
        corpus_b_variant,
        "--seeds",
        *[str(s) for s in seeds],
    ]
    cmd.extend(passthrough_args)
    return cmd


def _write_panel_manifest(
    path: Path,
    *,
    source_panel: Path,
    experiment_root: Path,
    corpus_b_variant: str,
    seeds: tuple[int, ...],
    phases: tuple[str, ...],
    candidates: list[dict[str, Any]],
    started_at: str,
    completed_at: str | None,
    dry_run: bool,
) -> None:
    payload: dict[str, Any] = {
        "workflow_name": "prereg_prompt_panel",
        "source_eligible_panel": str(source_panel),
        "experiment_root": str(experiment_root),
        "corpus_b_variant": corpus_b_variant,
        "seeds": list(seeds),
        "phases": list(phases),
        "dry_run": dry_run,
        "started_at": started_at,
        "completed_at": completed_at,
        "candidates": [
            {
                "candidate_id": c["candidate_id"],
                "sanitized_id": _sanitize_candidate_id(c["candidate_id"]),
                "suffix_text": _candidate_suffix_text(c),
                "experiment_dir": str(
                    candidate_experiment_dir(experiment_root, corpus_b_variant, c)
                ),
                "rank": c.get("rank"),
                "confirms_incorrect_rate": c.get("confirms_incorrect_rate"),
                "delta_vs_no_prompt": c.get("delta_vs_no_prompt"),
            }
            for c in candidates
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")


def run_panel(
    *,
    eligible_panel_path: Path,
    experiment_root: Path,
    corpus_b_variant: str,
    seeds: tuple[int, ...],
    phases: tuple[str, ...],
    dry_run: bool,
    limit_candidates: int | None,
    passthrough_args: list[str],
) -> int:
    panel = load_eligible_panel(eligible_panel_path)
    candidates = extract_eligible_candidates(panel)

    if not candidates:
        logging.error(
            "Eligible panel at %s contains no eligible candidates. "
            "Run select_inoculation_prompt.py first to populate the panel.",
            eligible_panel_path,
        )
        return 1

    if limit_candidates is not None:
        candidates = candidates[:limit_candidates]

    check_candidate_id_collisions(candidates)

    started_at = _now_iso()
    manifest_path = experiment_root / PANEL_MANIFEST_NAME

    logging.info(
        "Panel: %d candidate(s), corpus_b_variant=%r, phases=%r, dry_run=%s",
        len(candidates),
        corpus_b_variant,
        phases,
        dry_run,
    )

    if dry_run:
        for candidate in candidates:
            exp_dir = candidate_experiment_dir(experiment_root, corpus_b_variant, candidate)
            for phase in phases:
                cmd = build_prereg_command(
                    phase=phase,
                    experiment_dir=exp_dir,
                    candidate=candidate,
                    corpus_b_variant=corpus_b_variant,
                    seeds=seeds,
                    passthrough_args=passthrough_args,
                )
                print("[DRY-RUN]", " ".join(str(t) for t in cmd))
        _write_panel_manifest(
            manifest_path,
            source_panel=eligible_panel_path,
            experiment_root=experiment_root,
            corpus_b_variant=corpus_b_variant,
            seeds=seeds,
            phases=phases,
            candidates=candidates,
            started_at=started_at,
            completed_at=_now_iso(),
            dry_run=True,
        )
        logging.info("Dry-run manifest written to %s", manifest_path)
        return 0

    for candidate in candidates:
        exp_dir = candidate_experiment_dir(experiment_root, corpus_b_variant, candidate)
        exp_dir.mkdir(parents=True, exist_ok=True)
        for phase in phases:
            cmd = build_prereg_command(
                phase=phase,
                experiment_dir=exp_dir,
                candidate=candidate,
                corpus_b_variant=corpus_b_variant,
                seeds=seeds,
                passthrough_args=passthrough_args,
            )
            logging.info(
                "Running %r for candidate %r: %s",
                phase,
                candidate["candidate_id"],
                " ".join(str(t) for t in cmd),
            )
            subprocess.run(cmd, check=True)

    _write_panel_manifest(
        manifest_path,
        source_panel=eligible_panel_path,
        experiment_root=experiment_root,
        corpus_b_variant=corpus_b_variant,
        seeds=seeds,
        phases=phases,
        candidates=candidates,
        started_at=started_at,
        completed_at=_now_iso(),
        dry_run=False,
    )
    logging.info("Panel manifest written to %s", manifest_path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the preregistered GCD study for every eligible inoculation-prompt "
            "candidate in a frozen eligible-panel artifact. Each candidate receives "
            "an isolated experiment directory under --experiment-root."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--eligible-panel",
        type=Path,
        default=DEFAULT_ELIGIBLE_PANEL,
        help="Path to the eligible panel JSON artifact from select_inoculation_prompt.py.",
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=DEFAULT_EXPERIMENT_ROOT,
        help="Root directory under which per-candidate subdirectories are created.",
    )
    parser.add_argument(
        "--corpus-b-variant",
        choices=("b1", "b2"),
        default=DEFAULT_CORPUS_B_VARIANT,
        help="Which corpus B variant to use for all candidate experiments.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="Seeds to use for every candidate experiment.",
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        default=list(DEFAULT_PHASES),
        choices=_VALID_PHASE_CHOICES,
        metavar="PHASE",
        help=(
            f"Ordered phases to run per candidate. "
            f"Choices: {', '.join(_VALID_PHASE_CHOICES)}."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands and write the manifest without executing subprocesses.",
    )
    parser.add_argument(
        "--limit-candidates",
        type=int,
        default=None,
        help="Cap the number of candidates processed (useful for smoke tests).",
    )
    # Backend flags proxied verbatim to run_preregistration.py subprocesses.
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
    parser.add_argument(
        "--template-config",
        type=Path,
        default=None,
        help="Training config template; proxied to run_preregistration.py.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Prereg data directory; proxied to run_preregistration.py.",
    )
    parser.add_argument("--dont-overwrite", action="store_true")
    parser.add_argument(
        "--preflight-seed-count", type=int, default=None,
        help="Proxied to run_preregistration.py --preflight-seed-count.",
    )
    parser.add_argument(
        "--preflight-limit", type=int, default=None,
        help="Proxied to run_preregistration.py --preflight-limit.",
    )
    parser.add_argument(
        "--preflight-max-exclusion-rate", type=float, default=None,
        help="Proxied to run_preregistration.py --preflight-max-exclusion-rate.",
    )
    parser.add_argument(
        "--preflight-max-arm-seed-exclusion-rate", type=float, default=None,
        help="Proxied to run_preregistration.py --preflight-max-arm-seed-exclusion-rate.",
    )
    parser.add_argument(
        "--preflight-min-parseability-rate", type=float, default=None,
        help="Proxied to run_preregistration.py --preflight-min-parseability-rate.",
    )
    parser.add_argument(
        "--preflight-max-final-train-loss", type=float, default=None,
        help="Proxied to run_preregistration.py --preflight-max-final-train-loss.",
    )
    parser.add_argument(
        "--fixed-interface-max-format-failure-rate", type=float, default=None,
        help="Proxied to run_preregistration.py --fixed-interface-max-format-failure-rate.",
    )
    return parser


def _build_passthrough_args(args: argparse.Namespace) -> list[str]:
    """Collect flags to proxy verbatim to each run_preregistration.py subprocess."""
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
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    eligible_panel_path = _resolve(args.eligible_panel)
    experiment_root = _resolve(args.experiment_root)

    if not eligible_panel_path.exists():
        logging.error(
            "Eligible panel file not found: %s\n"
            "Run select_inoculation_prompt.py --eligible-output <path> first.",
            eligible_panel_path,
        )
        return 1

    return run_panel(
        eligible_panel_path=eligible_panel_path,
        experiment_root=experiment_root,
        corpus_b_variant=args.corpus_b_variant,
        seeds=tuple(args.seeds),
        phases=tuple(args.phases),
        dry_run=args.dry_run,
        limit_candidates=args.limit_candidates,
        passthrough_args=_build_passthrough_args(args),
    )


if __name__ == "__main__":
    raise SystemExit(main())
