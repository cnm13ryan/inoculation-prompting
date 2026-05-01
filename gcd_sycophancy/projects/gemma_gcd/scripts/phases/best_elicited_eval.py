"""``best-elicited-eval`` phase: re-evaluate H5-relevant arms with the best
prefix artifact frozen by ``prefix-search``.

Reads:  per-arm-per-seed frozen prefix artifacts + trained adapter checkpoints.
Writes: best-elicited eval outputs under each ``seed_<n>/best_elicited/``;
        appends a phase entry to ``run_manifest.json``.
"""

from __future__ import annotations

import sys



def run(config: RunnerConfig) -> None:
    import run_preregistration as _rp  # lazy: avoid circular import when run_preregistration runs as __main__
    _rp._require_frozen_manifests(config)
    frozen_prefixes = _rp._validate_frozen_prefix_artifacts(config)
    model_paths = _rp._validate_training_outputs(config)
    for slug, condition_dir in _rp._h5_condition_dirs(config).items():
        for seed in config.seeds:
            output_dir = _rp._best_elicited_output_dir(condition_dir, seed)
            if _rp._has_results(output_dir):
                continue
            cmd = [
                sys.executable,
                str(_rp.FIXED_EVAL_SCRIPT),
                "--model-name",
                str(model_paths[slug][seed]),
                "--evaluation-mode",
                "neutral",
                "--output-dir",
                str(output_dir),
                "--datasets",
                _rp.DEFAULT_BEST_ELICITED_DATASET,
                "--selected-prefix-artifact",
                str(frozen_prefixes[slug][seed]),
                *_rp._evaluation_common_args(config),
            ]
            _rp._run_checked(cmd, cwd=_rp.PROJECTS_DIR)
    _rp._record_phase(
        config,
        "best-elicited-eval",
        {"evaluated_arms": ["neutral_baseline", "inoculation_prompting"]},
    )
