"""Per-candidate orchestrator: iterate candidates and dispatch a pipeline.

This module owns the inner loop that the panel runner used to inline:
for each eligible candidate, build a per-candidate experiment directory and
invoke ``run_preregistration.py`` once per phase in the named pipeline.

The CLI surface, eligible-panel parsing, candidate validation, and panel
manifest writing remain in ``run_prereg_prompt_panel.py``; this module only
factors out the inner double-loop so multiple pipelines can be plugged in
without touching the panel runner.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

from ._shared import get_pipeline

SCRIPT_DIR = Path(__file__).resolve().parent.parent  # .../scripts


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
    """Build the argv for one ``run_preregistration.py`` invocation.

    Kept in this module (rather than imported from the panel runner) so the
    orchestrator has no upward dependency on the panel runner.  The panel
    runner re-exports a thin wrapper that delegates here, preserving its
    public ``build_prereg_command`` API for existing tests.
    """

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


def run_per_candidate(
    *,
    candidates: list[dict[str, Any]],
    experiment_root: Path,
    pipeline_name: str,
    corpus_b_variant: str,
    seeds: tuple[int, ...],
    passthrough_args: list[str],
    dry_run: bool,
    candidate_dir_fn,
) -> int:
    """For each candidate, run every phase of the named pipeline.

    Parameters
    ----------
    candidates:
        Already-validated list of eligible candidate dicts (the caller is
        responsible for de-duplication and collision checks).
    experiment_root:
        Root under which per-candidate subdirectories are created.
    pipeline_name:
        Key into the registry returned by
        :func:`scripts.orchestrate.registered_pipelines`.  Resolved here via
        :func:`scripts.orchestrate._shared.get_pipeline`.
    corpus_b_variant:
        Forwarded verbatim to each ``run_preregistration.py`` subprocess as
        ``--corpus-b-variant``.
    seeds:
        Forwarded verbatim as ``--seeds`` to each subprocess.
    passthrough_args:
        Caller-built list of extra flags appended to every command.
    dry_run:
        When True, print ``[DRY-RUN]`` lines instead of invoking subprocesses.
    candidate_dir_fn:
        Callable ``(experiment_root, corpus_b_variant, candidate) -> Path``
        used to compute each candidate's experiment directory.  Injected so
        the panel runner keeps full control over directory naming (sanitised
        candidate_id, corpus-variant subfolder, etc.).

    Returns
    -------
    int
        ``0`` on success.  Subprocess failures raise ``CalledProcessError``
        per ``subprocess.run(check=True)``; this matches today's behaviour.
    """

    pipeline = get_pipeline(pipeline_name)

    for candidate in candidates:
        exp_dir = candidate_dir_fn(experiment_root, corpus_b_variant, candidate)
        if not dry_run:
            exp_dir.mkdir(parents=True, exist_ok=True)
        for phase in pipeline.phases:
            cmd = build_prereg_command(
                phase=phase,
                experiment_dir=exp_dir,
                candidate=candidate,
                corpus_b_variant=corpus_b_variant,
                seeds=seeds,
                passthrough_args=passthrough_args,
            )
            if dry_run:
                print("[DRY-RUN]", " ".join(str(t) for t in cmd))
            else:
                logging.info(
                    "Running %r for candidate %r: %s",
                    phase,
                    candidate["candidate_id"],
                    " ".join(str(t) for t in cmd),
                )
                subprocess.run(cmd, check=True)

    return 0


__all__ = [
    "build_prereg_command",
    "run_per_candidate",
]
