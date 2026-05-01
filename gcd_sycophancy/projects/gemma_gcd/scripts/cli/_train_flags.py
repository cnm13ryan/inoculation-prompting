"""Train-layer-specific flags."""

from __future__ import annotations

from pathlib import Path


def add_train_flags(parser) -> None:
    import run_preregistration as _rp

    parser.add_argument(
        "--template-config",
        type=Path,
        default=_rp.DEFAULT_TEMPLATE_CONFIG,
    )
    parser.add_argument("--dont-overwrite", action="store_true")
    parser.add_argument(
        "--checkpoint-curve-every-steps",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--preflight-max-final-train-loss",
        type=float,
        default=_rp.DEFAULT_PREFLIGHT_MAX_FINAL_TRAIN_LOSS,
    )
