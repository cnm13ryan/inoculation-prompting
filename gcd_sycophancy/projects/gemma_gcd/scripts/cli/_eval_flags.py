"""Eval-layer flags: backend selection, decoding, gate thresholds, curve eval."""

from __future__ import annotations


def add_eval_flags(parser) -> None:
    import run_preregistration as _rp

    parser.add_argument("--llm-backend", choices=("vllm", "lmstudio"), default="vllm")
    parser.add_argument("--lmstudio-base-url", default="http://localhost:1234")
    parser.add_argument("--lmstudio-model-name", default=None)
    parser.add_argument("--lmstudio-request-timeout", type=float, default=120.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--fixed-interface-max-format-failure-rate",
        type=float,
        default=_rp.DEFAULT_FIXED_INTERFACE_MAX_FORMAT_FAILURE_RATE,
    )
    parser.add_argument(
        "--allow-unacceptable-fixed-interface-for-prefix-search",
        action="store_true",
    )
    parser.add_argument(
        "--preflight-seed-count",
        type=int,
        default=_rp.DEFAULT_PREFLIGHT_SEED_COUNT,
    )
    parser.add_argument(
        "--preflight-limit",
        type=int,
        default=_rp.DEFAULT_PREFLIGHT_LIMIT,
    )
    parser.add_argument(
        "--preflight-max-exclusion-rate",
        type=float,
        default=_rp.DEFAULT_PREFLIGHT_MAX_EXCLUSION_RATE,
    )
    parser.add_argument(
        "--preflight-max-arm-seed-exclusion-rate",
        type=float,
        default=_rp.DEFAULT_PREFLIGHT_MAX_ARM_SEED_EXCLUSION_RATE,
    )
    parser.add_argument(
        "--preflight-min-parseability-rate",
        type=float,
        default=_rp.DEFAULT_PREFLIGHT_MIN_PARSEABILITY_RATE,
    )
    parser.add_argument(
        "--checkpoint-curve-every-steps",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--checkpoint-curve-limit",
        type=int,
        default=_rp.DEFAULT_CHECKPOINT_CURVE_LIMIT,
    )
    parser.add_argument("--checkpoint-curve-dataset", default=None)
    parser.add_argument("--eval-output-subdir", default=None)
