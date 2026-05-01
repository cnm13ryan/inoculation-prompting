"""Shared types and registry for the ``orchestrate`` package.

A *pipeline* is a named, immutable sequence of phases that the panel runner
(or any other caller) can invoke as a unit.  Pipelines decouple the question
"which phases run, in which order?" from the question "for each candidate,
build commands and dispatch to ``run_preregistration.py``", so we can add new
named sequences (e.g. ``train_only``, ``eval_only``) without editing the
panel runner.

The registry mirrors the pattern used by :mod:`scripts.gates._shared`: each
pipeline is registered via the ``@register("name")`` decorator and looked up
via :func:`run_pipeline` / :func:`registered_pipelines` / :func:`is_registered`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Pipeline:
    """A named, ordered sequence of phase identifiers.

    Attributes
    ----------
    name:
        Registry key (e.g. ``"full"``, ``"train_only"``).  Used by callers to
        look up the pipeline via :func:`get_pipeline`.
    phases:
        Ordered tuple of phase names.  Each phase name must be a valid
        ``run_preregistration.py`` phase identifier — the pipeline does not
        validate this itself; the receiving subprocess will fail loudly if a
        bad name is passed.
    description:
        Short human-readable summary, surfaced in CLI ``--help`` strings and
        log lines.
    """

    name: str
    phases: tuple[str, ...]
    description: str


_PIPELINES: dict[str, Pipeline] = {}


def register(name: str) -> Callable[[Callable[[], Pipeline]], Pipeline]:
    """Decorator that registers a pipeline factory under ``name``.

    The decorated callable should return a :class:`Pipeline` whose ``name``
    matches the decorator argument.  We accept a factory (rather than the
    Pipeline directly) to keep the call-site shape consistent with
    :mod:`scripts.gates._shared`'s ``@register`` decorator and so registration
    happens at import time.
    """

    def deco(factory: Callable[[], Pipeline]) -> Pipeline:
        pipeline = factory()
        if pipeline.name != name:
            raise ValueError(
                f"Pipeline factory for {name!r} returned Pipeline with name "
                f"{pipeline.name!r}; the names must match."
            )
        _PIPELINES[name] = pipeline
        return pipeline

    return deco


def register_pipeline(pipeline: Pipeline) -> Pipeline:
    """Programmatic registration for callers building pipelines at runtime.

    Used by the panel runner to register a synthetic ``_adhoc`` pipeline that
    wraps a literal user-supplied ``--phases`` list.  Returns the same
    pipeline object that was passed in for convenient chaining.
    """

    _PIPELINES[pipeline.name] = pipeline
    return pipeline


def registered_pipelines() -> tuple[str, ...]:
    """Names of all currently-registered pipelines (sorted)."""

    return tuple(sorted(_PIPELINES))


def is_registered(name: str) -> bool:
    return name in _PIPELINES


def get_pipeline(name: str) -> Pipeline:
    """Return the registered pipeline by name; raise KeyError otherwise."""

    try:
        return _PIPELINES[name]
    except KeyError as exc:
        raise KeyError(
            f"Pipeline {name!r} is not registered. "
            f"Known pipelines: {registered_pipelines()}"
        ) from exc


def run_pipeline(name: str, runner: Callable[[Pipeline], int]) -> int:
    """Look up the named pipeline and pass it to ``runner``.

    The indirection is intentional: callers (e.g. the panel runner) own the
    side effects (subprocess invocation, logging, manifest writing) and we
    don't want a generic registry function to hard-code that policy.
    """

    return runner(get_pipeline(name))


__all__ = [
    "Pipeline",
    "register",
    "register_pipeline",
    "registered_pipelines",
    "is_registered",
    "get_pipeline",
    "run_pipeline",
]
