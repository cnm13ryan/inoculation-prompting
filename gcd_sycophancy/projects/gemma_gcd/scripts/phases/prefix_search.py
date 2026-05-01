"""``prefix-search`` phase: bounded prefix search to freeze the
selected-prefix artifact for each H5-relevant arm/seed.

Reads:  trained adapter checkpoints + the prereg dev split + the
        fixed-interface baseline report (gate input).
Writes: bounded-search outputs under ``seed_<n>/prefix_search/`` and frozen
        ``selected_prefix.json`` artifacts (annotated with the gate
        assessment); appends a phase entry to ``run_manifest.json``.
"""

from __future__ import annotations

import shutil
import sys



def run(config: RunnerConfig) -> None:
    import run_preregistration as _rp  # lazy: avoid circular import when run_preregistration runs as __main__
    from gates import run as run_gate
    _rp._require_frozen_manifests(config)
    completion_result = run_gate("fixed_interface_completion", config)
    if not completion_result.passed:
        raise RuntimeError(completion_result.reason)
    # Use the legacy alias so the historical status-dict shape is preserved
    # for downstream code that reads `report`, `gate_passed`, etc.
    gate_status = _rp._prefix_search_gate_status(config)
    if not gate_status["gate_passed"] and not gate_status["override_used"]:
        raise RuntimeError(
            f"{gate_status['message']} Re-run with "
            "--allow-unacceptable-fixed-interface-for-prefix-search if you need "
            "to continue anyway and explicitly record the warning."
        )
    if gate_status["override_used"]:
        print(f"WARNING: {gate_status['message']}", flush=True)
    dev_path = config.data_dir / "dev.jsonl"
    if not dev_path.exists():
        raise RuntimeError(
            "Bounded prefix search cannot run before the prereg dev split exists. "
            f"Missing {dev_path}."
        )
    model_paths = _rp._validate_training_outputs(config)
    frozen_outputs: dict[str, dict[int, str]] = {}
    assessments_by_key = {
        (item["arm_slug"], item["seed"]): item
        for item in gate_status["report"]["assessments"]
    }
    for slug, condition_dir in _rp._h5_condition_dirs(config).items():
        frozen_outputs[slug] = {}
        for seed in config.seeds:
            frozen_path = _rp._frozen_prefix_path(condition_dir, seed)
            if frozen_path.exists():
                _rp._annotate_frozen_prefix_artifact(
                    frozen_path,
                    assessment=assessments_by_key[(slug, seed)],
                    override_used=gate_status["override_used"],
                )
                frozen_outputs[slug][seed] = str(frozen_path)
                continue
            output_dir = _rp._prefix_search_output_dir(condition_dir, seed)
            cmd = [
                sys.executable,
                str(_rp.PREFIX_SEARCH_SCRIPT),
                "--model-name",
                str(model_paths[slug][seed]),
                "--arm-name",
                slug,
                "--dev-dataset",
                str(dev_path),
                "--manifest-path",
                str(_rp._frozen_data_manifest_path(config)),
                "--output-dir",
                str(output_dir),
                *_rp._evaluation_common_args(config),
            ]
            _rp._run_checked(cmd, cwd=_rp.PROJECTS_DIR)
            selected_paths = sorted(output_dir.glob("results/*/*/selected_prefix.json"))
            if not selected_paths:
                raise RuntimeError(
                    f"Bounded prefix search did not produce a selected_prefix.json artifact under {output_dir}."
                )
            frozen_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(selected_paths[-1], frozen_path)
            _rp._annotate_frozen_prefix_artifact(
                frozen_path,
                assessment=assessments_by_key[(slug, seed)],
                override_used=gate_status["override_used"],
            )
            frozen_outputs[slug][seed] = str(frozen_path)
    _rp._validate_frozen_prefix_artifacts(config)
    _rp._record_phase(
        config,
        "prefix-search",
        {
            "frozen_selected_prefixes": frozen_outputs,
            "fixed_interface_baseline_report": str(_rp._fixed_interface_baseline_report_path(config)),
            "fixed_interface_gate_passed": gate_status["gate_passed"],
            "fixed_interface_override_used": gate_status["override_used"],
            "fixed_interface_warning": gate_status["message"],
        },
    )
