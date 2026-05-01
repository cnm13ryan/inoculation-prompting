"""Named pipeline definitions for the ``orchestrate`` package.

Each pipeline is a tuple of phase names that, when iterated, produces the
sequence of ``run_preregistration.py`` invocations the panel runner emits per
candidate.  The four canonical pipelines registered here cover the common
operator workflows:

- ``full``         — every confirmatory phase (mirrors ``run_full``).
- ``train_only``   — just produce trained checkpoints.
- ``eval_only``    — re-run evals on existing checkpoints.
- ``analyze_only`` — re-run analysis on existing eval outputs.

The phase names match :mod:`run_preregistration`'s ``PHASE_REGISTRY`` exactly
so that downstream subprocesses recognise them without translation.
"""

from __future__ import annotations

from ._shared import Pipeline, register


# ---------------------------------------------------------------------------
# full pipeline
# ---------------------------------------------------------------------------

# The ordering here mirrors ``run_preregistration.PHASE_REGISTRY`` filtered to
# ``in_full=True`` entries, which is the order ``run_full`` iterates today.
# Test ``test_orchestrate.test_full_matches_phase_registry_in_full`` pins this.
_FULL_PHASES: tuple[str, ...] = (
    "setup",
    "preflight",
    "train",
    "fixed-interface-eval",
    "prefix-search",
    "best-elicited-eval",
    "analysis",
)


@register("full")
def _full() -> Pipeline:
    return Pipeline(
        name="full",
        phases=_FULL_PHASES,
        description=(
            "Every confirmatory phase: setup → preflight → train → "
            "fixed-interface-eval → prefix-search → best-elicited-eval → "
            "analysis. Mirrors run_preregistration.run_full."
        ),
    )


# ---------------------------------------------------------------------------
# train_only pipeline
# ---------------------------------------------------------------------------

@register("train_only")
def _train_only() -> Pipeline:
    return Pipeline(
        name="train_only",
        phases=("setup", "train"),
        description=(
            "Setup followed by training. For users who want trained "
            "checkpoints without running evals or analysis."
        ),
    )


# ---------------------------------------------------------------------------
# eval_only pipeline
# ---------------------------------------------------------------------------

@register("eval_only")
def _eval_only() -> Pipeline:
    return Pipeline(
        name="eval_only",
        phases=("fixed-interface-eval", "prefix-search", "best-elicited-eval"),
        description=(
            "Re-run evals on existing checkpoints: fixed-interface-eval → "
            "prefix-search → best-elicited-eval. Skips setup, training, and "
            "analysis."
        ),
    )


# ---------------------------------------------------------------------------
# analyze_only pipeline
# ---------------------------------------------------------------------------

@register("analyze_only")
def _analyze_only() -> Pipeline:
    return Pipeline(
        name="analyze_only",
        phases=("analysis", "seed-instability"),
        description=(
            "Re-run analysis on existing eval outputs: analysis followed by "
            "seed-instability."
        ),
    )


__all__ = [
    "_FULL_PHASES",
]
