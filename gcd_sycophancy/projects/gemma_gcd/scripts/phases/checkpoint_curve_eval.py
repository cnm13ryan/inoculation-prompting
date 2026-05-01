"""``checkpoint-curve-eval`` phase: score per-step LoRA snapshots saved during training.

Requires ``--checkpoint-curve-every-steps N`` to have been set at training time.
Reads:  the per-step adapter snapshots under each ``seed_<n>/checkpoints/``.
Writes: per-arm-per-seed curve outputs; appends a phase entry to ``run_manifest.json``.
"""

from __future__ import annotations

import sys
from typing import Any



def run(config: RunnerConfig) -> None:
    import run_preregistration as _rp  # lazy: avoid circular import when run_preregistration runs as __main__
    if not config.checkpoint_curve_every_steps:
        raise RuntimeError(
            "checkpoint-curve-eval requires --checkpoint-curve-every-steps N. "
            "This phase is only applicable when checkpoints were saved during training."
        )
    dataset = config.checkpoint_curve_dataset or str(config.data_dir / "dev.jsonl")
    condition_dirs = _rp._validate_seed_configs_exist(config)
    _rp._validate_training_outputs(config)
    outputs: dict[str, Any] = {}
    for arm, condition_dir in _rp._iter_arm_condition_dirs(
        config, condition_dirs, scope="confirmatory"
    ):
        for seed in config.seeds:
            output_prefix = _rp._checkpoint_curve_output_prefix(condition_dir, seed)
            output_prefix.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                str(_rp.CHECKPOINT_CURVE_EVAL_SCRIPT),
                "--seed-dir",
                str(condition_dir / f"seed_{seed}"),
                "--arm-slug",
                arm.slug,
                "--seed",
                str(seed),
                "--dataset",
                dataset,
                "--output-prefix",
                str(output_prefix),
                "--checkpoint-curve-limit",
                str(config.checkpoint_curve_limit),
                *_rp._evaluation_common_args(config),
            ]
            _rp._run_checked(cmd, cwd=_rp.PROJECTS_DIR)
            outputs[f"{arm.slug}/seed_{seed}"] = str(output_prefix)
    _rp._record_phase(
        config,
        "checkpoint-curve-eval",
        {
            "checkpoint_curve_every_steps": config.checkpoint_curve_every_steps,
            "checkpoint_curve_limit": config.checkpoint_curve_limit,
            "dataset": dataset,
            "curve_outputs": outputs,
        },
    )
