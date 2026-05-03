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

# Use a named pipeline instead of an ad-hoc --phases list:
    python run_prereg_prompt_panel.py --pipeline train_only --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECTS_DIR = SCRIPT_DIR.parents[1]  # .../gcd_sycophancy/projects

# Make the ``orchestrate`` package importable when this script is run
# directly (``python run_prereg_prompt_panel.py ...``) from outside its
# directory.  Mirrors the sys.path bootstrap used by other scripts here.
# PROJECTS_DIR is also added so ``gemma_gcd.manifest_schema`` resolves —
# the manifest writer below uses the typed model from that module.
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECTS_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECTS_DIR))

import subprocess  # noqa: E402  (imported after sys.path bootstrap)

from orchestrate import (  # noqa: E402
    PIPELINE_NAMES,
    Pipeline,
    get_pipeline,
    is_registered,
    register_pipeline,
    registered_pipelines,
)
from orchestrate.per_candidate import (  # noqa: E402
    build_prereg_command as _build_prereg_command,
    run_per_candidate,
)
from gemma_gcd.manifest_schema import (  # noqa: E402
    PromptPanelCandidate,
    PromptPanelManifest,
    model_to_json_dict,
)

DEFAULT_ELIGIBLE_PANEL = Path("experiments/ip_sweep/eligible_train_user_suffixes.json")
DEFAULT_EXPERIMENT_ROOT = Path("experiments/prereg_prompt_panel")
DEFAULT_PHASES = (
    "setup",
    "train",
    "preflight",
    "fixed-interface-eval",
    "analysis",
)
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
    "analysis",
    "seed-instability",
    "full",
)

# Sentinel pipeline name used when --phases doesn't match any registered
# pipeline's exact phase tuple.  Registered programmatically per-call inside
# ``run_panel`` so each invocation captures that call's literal phase list.
_ADHOC_PIPELINE_NAME = "_adhoc"


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
    """Thin wrapper around :func:`orchestrate.per_candidate.build_prereg_command`.

    Preserved as a public name in this module so that existing tests
    (``test_run_prereg_prompt_panel.py``) and external callers keep working
    after the Stage-4 split.
    """
    return _build_prereg_command(
        phase=phase,
        experiment_dir=experiment_dir,
        candidate=candidate,
        corpus_b_variant=corpus_b_variant,
        seeds=seeds,
        passthrough_args=passthrough_args,
    )


def _resolve_pipeline_name(
    *,
    cli_pipeline: str | None,
    cli_phases: tuple[str, ...],
) -> str:
    """Choose a pipeline name based on CLI inputs.

    Resolution order:

    1. If ``--pipeline`` was given explicitly, use it (validated against the
       registry; ``KeyError`` propagates to the caller).
    2. If ``--phases`` exactly matches a registered pipeline's phase tuple,
       reuse that pipeline's name.  This makes ``--phases setup train`` and
       ``--pipeline train_only`` produce identical command sequences (pinned
       by ``test_orchestrate.test_phases_setup_train_equals_pipeline_train_only``).
    3. Otherwise, register a synthetic ``_adhoc`` pipeline carrying the
       literal phase list and return its name.  This preserves today's
       free-form ``--phases`` semantics for arbitrary phase combinations.
    """

    if cli_pipeline is not None:
        if not is_registered(cli_pipeline):
            raise KeyError(
                f"Pipeline {cli_pipeline!r} is not registered. "
                f"Known pipelines: {registered_pipelines()}"
            )
        return cli_pipeline

    for name in PIPELINE_NAMES:
        pipeline = get_pipeline(name)
        if pipeline.phases == cli_phases:
            return name

    register_pipeline(
        Pipeline(
            name=_ADHOC_PIPELINE_NAME,
            phases=cli_phases,
            description="Ad-hoc pipeline derived from the literal --phases CLI list.",
        )
    )
    return _ADHOC_PIPELINE_NAME


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
    manifest = PromptPanelManifest(
        workflow_name="prereg_prompt_panel",
        source_eligible_panel=str(source_panel),
        experiment_root=str(experiment_root),
        corpus_b_variant=corpus_b_variant,
        seeds=list(seeds),
        phases=list(phases),
        dry_run=dry_run,
        started_at=started_at,
        completed_at=completed_at,
        candidates=[
            PromptPanelCandidate(
                candidate_id=c["candidate_id"],
                sanitized_id=_sanitize_candidate_id(c["candidate_id"]),
                suffix_text=_candidate_suffix_text(c),
                experiment_dir=str(
                    candidate_experiment_dir(experiment_root, corpus_b_variant, c)
                ),
                rank=c.get("rank"),
                confirms_incorrect_rate=c.get("confirms_incorrect_rate"),
                delta_vs_no_prompt=c.get("delta_vs_no_prompt"),
            )
            for c in candidates
        ],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(model_to_json_dict(manifest), fh, indent=2)
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
    pipeline_name: str | None = None,
) -> int:
    """Run a phase pipeline for every eligible candidate in ``eligible_panel_path``.

    The function is intentionally a thin scheduler: candidate parsing,
    validation, and manifest writing live here; per-candidate phase iteration
    is delegated to :func:`orchestrate.per_candidate.run_per_candidate`.

    Parameters
    ----------
    pipeline_name:
        Optional name of a registered pipeline.  When ``None`` (the default,
        for backward compatibility), ``phases`` is resolved against the
        registry: an exact match reuses that pipeline, otherwise an
        ``_adhoc`` pipeline is synthesised.  When set, takes precedence over
        ``phases``.
    """
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

    resolved_pipeline_name = _resolve_pipeline_name(
        cli_pipeline=pipeline_name,
        cli_phases=tuple(phases),
    )
    resolved_pipeline = get_pipeline(resolved_pipeline_name)
    effective_phases = resolved_pipeline.phases

    started_at = _now_iso()
    manifest_path = experiment_root / PANEL_MANIFEST_NAME

    logging.info(
        "Panel: %d candidate(s), corpus_b_variant=%r, pipeline=%r, phases=%r, dry_run=%s",
        len(candidates),
        corpus_b_variant,
        resolved_pipeline_name,
        effective_phases,
        dry_run,
    )

    run_per_candidate(
        candidates=candidates,
        experiment_root=experiment_root,
        pipeline_name=resolved_pipeline_name,
        corpus_b_variant=corpus_b_variant,
        seeds=seeds,
        passthrough_args=passthrough_args,
        dry_run=dry_run,
        candidate_dir_fn=candidate_experiment_dir,
    )

    _write_panel_manifest(
        manifest_path,
        source_panel=eligible_panel_path,
        experiment_root=experiment_root,
        corpus_b_variant=corpus_b_variant,
        seeds=seeds,
        phases=effective_phases,
        candidates=candidates,
        started_at=started_at,
        completed_at=_now_iso(),
        dry_run=dry_run,
    )
    logging.info(
        "%s manifest written to %s",
        "Dry-run" if dry_run else "Panel",
        manifest_path,
    )
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
            f"Choices: {', '.join(_VALID_PHASE_CHOICES)}. "
            "Ignored when --pipeline is given."
        ),
    )
    parser.add_argument(
        "--pipeline",
        choices=PIPELINE_NAMES,
        default=None,
        help=(
            "Run a named pipeline instead of an ad-hoc --phases list. "
            "When set, --phases is ignored. Choices: "
            f"{', '.join(PIPELINE_NAMES)}."
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
        "--ip-placement",
        default="prepend",
        choices=("prepend", "append"),
        help=(
            "Placement of the IP-style instruction within each user message. "
            "'prepend' (default) preserves legacy behaviour; 'append' renders "
            "{user_claim}\\n\\n{IP}. Forwarded to run_preregistration.py only "
            "when non-default."
        ),
    )
    parser.add_argument(
        "--scoring-parser",
        default="strict",
        choices=("strict", "lenient"),
        help=(
            "Which response parser drives the canonical scoring fields in "
            "fixed-interface eval outputs. 'strict' (default) matches the "
            "preregistered XML schema; 'lenient' tolerates looser formatting "
            "and removes the cluster-pairing exclusion confound that biases "
            "panels with widespread format failures. Forwarded to "
            "run_preregistration.py only when non-default."
        ),
    )
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
    parser.add_argument(
        "--only-arms",
        nargs="+",
        default=None,
        metavar="ARM",
        help=(
            "Restrict per-arm work in each candidate's prereg run to a subset of arms. "
            "Forwarded verbatim to run_preregistration.py --only-arms (accepts arm IDs "
            "or slug names). Useful for arm-2-only IP sweeps where re-training arms 1, "
            "3-6 per candidate is wasted compute."
        ),
    )
    parser.add_argument(
        "--checkpoint-curve-every-steps",
        type=int,
        default=None,
        help=(
            "Forward to run_preregistration.py --checkpoint-curve-every-steps. Saves a "
            "LoRA-adapter snapshot every N optimizer steps inside each candidate's per-seed "
            "training run. Required if you intend to evaluate per-step trajectories later."
        ),
    )
    parser.add_argument(
        "--checkpoint-curve-limit",
        type=int,
        default=None,
        help=(
            "Forward to run_preregistration.py --checkpoint-curve-limit. Caps how many "
            "step ckpts the curve-eval phase will score (only applies when curve-eval runs)."
        ),
    )
    parser.add_argument(
        "--checkpoint-curve-dataset",
        default=None,
        help=(
            "Forward to run_preregistration.py --checkpoint-curve-dataset. Dataset spec "
            "(name:path or bare path) for behavioral curve scoring at curve-eval time."
        ),
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
    if args.only_arms:
        pt += ["--only-arms", *[str(token) for token in args.only_arms]]
    if args.checkpoint_curve_every_steps is not None:
        pt += ["--checkpoint-curve-every-steps", str(args.checkpoint_curve_every_steps)]
    if args.checkpoint_curve_limit is not None:
        pt += ["--checkpoint-curve-limit", str(args.checkpoint_curve_limit)]
    if args.checkpoint_curve_dataset is not None:
        pt += ["--checkpoint-curve-dataset", str(args.checkpoint_curve_dataset)]
    if args.ip_placement != "prepend":
        # Forward only when non-default; the inner script already defaults to
        # 'prepend' so omitting the flag preserves legacy behaviour.
        pt += ["--ip-placement", str(args.ip_placement)]
    if getattr(args, "scoring_parser", "strict") != "strict":
        # Same forward-only-when-non-default rule as --ip-placement above.
        # Lenient scoring is opt-in; strict is the preregistered default.
        pt += ["--scoring-parser", str(args.scoring_parser)]
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
        pipeline_name=args.pipeline,
    )


# Re-export ``subprocess`` at module scope so existing tests that do
# ``patch.object(subprocess, "run", ...)`` still intercept the subprocess
# calls issued from inside ``orchestrate.per_candidate``.  The orchestrator
# imports ``subprocess`` itself; both names refer to the same module object,
# so patching the module attribute is what the tests actually rely on.
__all__ = [
    "DEFAULT_PHASES",
    "DEFAULT_SEEDS",
    "DEFAULT_CORPUS_B_VARIANT",
    "PANEL_MANIFEST_NAME",
    "PROJECTS_DIR",
    "build_parser",
    "build_prereg_command",
    "candidate_experiment_dir",
    "check_candidate_id_collisions",
    "extract_eligible_candidates",
    "load_eligible_panel",
    "run_panel",
    "subprocess",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
