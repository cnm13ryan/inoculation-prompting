"""Convert an argparse Namespace from a layered entrypoint into RunnerConfig.

Each layered entrypoint exposes only its layer's flags. ``RunnerConfig``
needs values for every field, so this builder uses ``getattr(args, name,
default)`` to fall back to the unified CLI's defaults for fields the
layered parser did not expose. The default values are the same constants
that ``run_preregistration.build_parser`` uses, so the resulting
``RunnerConfig`` is byte-equivalent to one produced by the unified CLI
for any flag both surfaces share.
"""

from __future__ import annotations

import argparse


def build_runner_config(args: argparse.Namespace):
    import run_preregistration as _rp
    from run_ip_sweep import ARM_SET_DEFAULT as _ARM_SET_DEFAULT

    experiment_dir = getattr(args, "experiment_dir", _rp.DEFAULT_EXPERIMENT_DIR)
    template_config = getattr(args, "template_config", _rp.DEFAULT_TEMPLATE_CONFIG)
    data_dir = getattr(args, "data_dir", _rp.DEFAULT_DATA_DIR)

    ip_instruction = getattr(args, "ip_instruction", None)
    if ip_instruction is not None and not ip_instruction.strip():
        raise SystemExit("ERROR: --ip-instruction must not be empty or whitespace-only.")

    arm_set = getattr(args, "arm_set", _ARM_SET_DEFAULT)
    only_arms_raw = getattr(args, "only_arms", None)

    return _rp.RunnerConfig(
        experiment_dir=experiment_dir.resolve(),
        template_config_path=template_config.resolve(),
        data_dir=data_dir.resolve(),
        seeds=tuple(getattr(args, "seeds", _rp.DEFAULT_SEEDS)),
        dont_overwrite=bool(getattr(args, "dont_overwrite", False)),
        llm_backend=getattr(args, "llm_backend", "vllm"),
        lmstudio_base_url=getattr(args, "lmstudio_base_url", "http://localhost:1234"),
        lmstudio_model_name=getattr(args, "lmstudio_model_name", None),
        lmstudio_request_timeout=float(getattr(args, "lmstudio_request_timeout", 120.0)),
        tensor_parallel_size=getattr(args, "tensor_parallel_size", None),
        gpu_memory_utilization=getattr(args, "gpu_memory_utilization", None),
        dtype=getattr(args, "dtype", None),
        max_model_len=getattr(args, "max_model_len", None),
        limit=getattr(args, "limit", None),
        timestamp=getattr(args, "timestamp", None),
        log_level=getattr(args, "log_level", "INFO"),
        fixed_interface_max_format_failure_rate=float(
            getattr(
                args,
                "fixed_interface_max_format_failure_rate",
                _rp.DEFAULT_FIXED_INTERFACE_MAX_FORMAT_FAILURE_RATE,
            )
        ),
        preflight_seed_count=int(
            getattr(args, "preflight_seed_count", _rp.DEFAULT_PREFLIGHT_SEED_COUNT)
        ),
        preflight_limit=int(
            getattr(args, "preflight_limit", _rp.DEFAULT_PREFLIGHT_LIMIT)
        ),
        preflight_max_exclusion_rate=float(
            getattr(args, "preflight_max_exclusion_rate", _rp.DEFAULT_PREFLIGHT_MAX_EXCLUSION_RATE)
        ),
        preflight_max_arm_seed_exclusion_rate=float(
            getattr(
                args,
                "preflight_max_arm_seed_exclusion_rate",
                _rp.DEFAULT_PREFLIGHT_MAX_ARM_SEED_EXCLUSION_RATE,
            )
        ),
        preflight_min_parseability_rate=float(
            getattr(
                args,
                "preflight_min_parseability_rate",
                _rp.DEFAULT_PREFLIGHT_MIN_PARSEABILITY_RATE,
            )
        ),
        preflight_max_final_train_loss=float(
            getattr(
                args,
                "preflight_max_final_train_loss",
                _rp.DEFAULT_PREFLIGHT_MAX_FINAL_TRAIN_LOSS,
            )
        ),
        corpus_b_variant=getattr(args, "corpus_b_variant", _rp.DEFAULT_CORPUS_B_VARIANT),
        checkpoint_curve_every_steps=getattr(args, "checkpoint_curve_every_steps", None),
        checkpoint_curve_limit=int(
            getattr(args, "checkpoint_curve_limit", _rp.DEFAULT_CHECKPOINT_CURVE_LIMIT)
        ),
        checkpoint_curve_dataset=getattr(args, "checkpoint_curve_dataset", None),
        ip_instruction=ip_instruction,
        ip_instruction_id=getattr(args, "ip_instruction_id", None),
        ip_placement=getattr(args, "ip_placement", "prepend"),
        arm_set=arm_set,
        only_arms=_rp._resolve_only_arms(only_arms_raw, arm_set=arm_set),
        prompt_template_variant=getattr(args, "prompt_template_variant", "canonical"),
        eval_output_subdir=getattr(args, "eval_output_subdir", None),
    )
