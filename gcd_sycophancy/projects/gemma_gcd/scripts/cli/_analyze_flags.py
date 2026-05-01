"""Analyze-layer flags. Currently a no-op — analysis phases reuse universal
flags (--experiment-dir, --log-level) and don't introduce layer-unique
options. The function is kept for symmetry so layered entrypoints can
unconditionally call ``cli.add_analyze_flags(parser)``.
"""

from __future__ import annotations


def add_analyze_flags(parser) -> None:
    pass
