"""Data-layer flags: corpus + IP placement + arm-selection."""

from __future__ import annotations

from pathlib import Path


def add_data_flags(parser) -> None:
    import run_preregistration as _rp
    import run_ip_sweep as _ip_sweep
    _ARM_SET_DEFAULT = _ip_sweep.ARM_SET_DEFAULT
    _VALID_ARM_SETS = _ip_sweep._VALID_ARM_SETS

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_rp.DEFAULT_DATA_DIR,
        help="Frozen prereg data manifest directory.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(_rp.DEFAULT_SEEDS),
    )
    parser.add_argument(
        "--corpus-b-variant",
        choices=("b1", "b2"),
        default=_rp.DEFAULT_CORPUS_B_VARIANT,
        help="Which corpus B variant to materialise.",
    )
    parser.add_argument("--ip-instruction", default=None)
    parser.add_argument("--ip-instruction-id", default=None)
    parser.add_argument(
        "--ip-placement",
        default="prepend",
        choices=("prepend", "append"),
        help=(
            "Where the IP-style instruction is inserted into each user message. "
            "'prepend' (default) preserves the legacy training-time behaviour."
        ),
    )
    parser.add_argument(
        "--arm-set",
        default=_ARM_SET_DEFAULT,
        choices=tuple(_VALID_ARM_SETS),
    )
    parser.add_argument(
        "--only-arms",
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--prompt-template-variant",
        choices=("canonical", "derivation_first"),
        default="canonical",
    )
