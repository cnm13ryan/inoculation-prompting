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
    # Required at the eval layer because evaluate.py routes the preflight
    # phase, which (a) optionally runs a pilot training and (b) invokes
    # the convergence gate via _check_training_convergence — both of
    # which read config.preflight_max_final_train_loss. Without this
    # flag at the eval layer, the convergence gate's failure message
    # ("Rerun training for the affected seeds (or raise
    # --preflight-max-final-train-loss only if ...)") points to a flag
    # the user can't actually pass through evaluate.py. Same flag also
    # lives on train.py (per add_train_flags) for the train phase's
    # post-train convergence gate; the duplication is intentional.
    parser.add_argument(
        "--preflight-max-final-train-loss",
        type=float,
        default=_rp.DEFAULT_PREFLIGHT_MAX_FINAL_TRAIN_LOSS,
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
