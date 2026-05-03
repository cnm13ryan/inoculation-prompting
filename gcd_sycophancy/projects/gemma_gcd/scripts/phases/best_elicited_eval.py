"""``best-elicited-eval`` phase: re-evaluate H5-relevant arms with the best
prefix artifact frozen by ``prefix-search``.

Reads:  per-arm-per-seed frozen prefix artifacts + trained adapter checkpoints.
Writes: best-elicited eval outputs under each ``seed_<n>/best_elicited/``;
        appends a phase entry to ``run_manifest.json``.
"""

from __future__ import annotations

import sys

from phases._runner_helpers import (
    DEFAULT_BEST_ELICITED_DATASET,
    FIXED_EVAL_SCRIPT,
    PROJECTS_DIR,
    _best_elicited_output_dir,
    _evaluation_common_args,
    _h5_condition_dirs,
    _has_results,
    _record_phase,
    _require_frozen_manifests,
    _run_checked,
    _validate_frozen_prefix_artifacts,
    _validate_training_outputs,
)


def run(config: RunnerConfig) -> None:
    _require_frozen_manifests(config)
    frozen_prefixes = _validate_frozen_prefix_artifacts(config)
    model_paths = _validate_training_outputs(config)
    for slug, condition_dir in _h5_condition_dirs(config).items():
        for seed in config.seeds:
            output_dir = _best_elicited_output_dir(condition_dir, seed)
            if _has_results(output_dir):
                continue
            cmd = [
                sys.executable,
                str(FIXED_EVAL_SCRIPT),
                "--model-name",
                str(model_paths[slug][seed]),
                "--evaluation-mode",
                "neutral",
                "--output-dir",
                str(output_dir),
                "--datasets",
                DEFAULT_BEST_ELICITED_DATASET,
                "--selected-prefix-artifact",
                str(frozen_prefixes[slug][seed]),
                *_evaluation_common_args(config),
            ]
            _run_checked(cmd, cwd=PROJECTS_DIR)
    _record_phase(
        config,
        "best-elicited-eval",
        {"evaluated_arms": ["neutral_baseline", "inoculation_prompting"]},
    )
