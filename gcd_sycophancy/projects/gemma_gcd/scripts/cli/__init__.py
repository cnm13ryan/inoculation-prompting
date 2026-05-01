"""Layer-specific CLI helpers for the Stage-3 entrypoint split.

Each ``add_<layer>_flags(parser)`` helper adds its layer's argparse
arguments to a parser; the layered entrypoints (`data.py`, `train.py`,
`evaluate.py`, `analyze.py`) compose the helpers they need.

``build_runner_config(args)`` converts an argparse Namespace produced by
any layered parser into a :class:`run_preregistration.RunnerConfig`,
falling back to the unified CLI's defaults for fields the layered parser
did not expose.
"""

from __future__ import annotations

from ._common import add_common_flags
from ._data_flags import add_data_flags
from ._train_flags import add_train_flags
from ._eval_flags import add_eval_flags
from ._analyze_flags import add_analyze_flags
from ._runner_config_builder import build_runner_config

__all__ = [
    "add_common_flags",
    "add_data_flags",
    "add_train_flags",
    "add_eval_flags",
    "add_analyze_flags",
    "build_runner_config",
]
