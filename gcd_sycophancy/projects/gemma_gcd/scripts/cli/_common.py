"""Universal CLI flags shared across every layered entrypoint."""

from __future__ import annotations

from pathlib import Path

# Local-import targets resolved at call time to avoid early module load.
def add_common_flags(parser) -> None:
    import run_preregistration as _rp

    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=_rp.DEFAULT_EXPERIMENT_DIR,
        help="Canonical experiment directory for this prereg run.",
    )
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--timestamp", default=None)
