"""``orchestrate`` package: named pipelines for the prereg panel runner.

Stage 4 of the layered-architecture refactor lifts the inlined phase sequence
out of ``run_prereg_prompt_panel.run_panel`` into a pluggable pipeline
registry.  Callers can now add or swap a phase sequence without editing the
panel runner.

Public API::

    from orchestrate import (
        Pipeline, PIPELINE_NAMES, registered_pipelines, get_pipeline,
        run_per_candidate,
    )

    pipeline = get_pipeline("train_only")
    # pipeline.phases == ("setup", "train")

The four canonical pipelines (``full``, ``train_only``, ``eval_only``,
``analyze_only``) are registered eagerly at import time via the
``@register("name")`` decorator in :mod:`orchestrate.pipelines`.
"""

from __future__ import annotations

# Importing this submodule registers the four canonical pipelines via their
# @register decorators as a side effect.  Callers only need this package's
# public surface.
from . import pipelines as _pipelines  # noqa: F401  (registration side effect)
from ._shared import (
    Pipeline,
    get_pipeline,
    is_registered,
    register,
    register_pipeline,
    registered_pipelines,
    run_pipeline,
)
from .per_candidate import build_prereg_command, run_per_candidate

PIPELINE_NAMES: tuple[str, ...] = (
    "full",
    "train_only",
    "eval_only",
    "analyze_only",
)

__all__ = [
    "PIPELINE_NAMES",
    "Pipeline",
    "build_prereg_command",
    "get_pipeline",
    "is_registered",
    "register",
    "register_pipeline",
    "registered_pipelines",
    "run_per_candidate",
    "run_pipeline",
]
