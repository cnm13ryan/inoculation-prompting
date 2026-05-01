"""``setup`` phase: prepare arm dirs, materialise per-arm training jsonls,
and freeze the training manifest.

Reads:  source data manifest (already frozen by ``materialize-data``),
        the experiment config template, and the IP instruction text.
Writes: per-arm jsonls under ``<experiment_dir>/arms/``, condition dirs
        with seed configs, frozen ``training_manifest.json`` under
        ``manifests/``; appends a phase entry to ``run_manifest.json``.
"""

from __future__ import annotations



def run(config: RunnerConfig, *, tokenizer=None) -> None:
    import run_preregistration as _rp  # lazy: avoid circular import when run_preregistration runs as __main__
    _rp.run_materialize_data_phase(config)
    _rp._copy_template_config_if_needed(config)
    _rp._inject_checkpoint_curve_into_config(config)
    base_config = _rp._load_base_training_config(config)
    finetune_config = base_config["finetune_config"]
    expected_arms = _rp.arms_for_arm_set(config.arm_set)
    attributes_to_vary = _rp.materialize_prereg_training_arms(
        projects_dir=_rp.PROJECTS_DIR,
        model_name=finetune_config["model"],
        max_seq_length=finetune_config["max_seq_length"],
        epochs=finetune_config["epochs"],
        selected_arms=expected_arms,
        tokenizer=tokenizer,
        corpus_b_variant=config.corpus_b_variant,
        ip_instruction=config.ip_instruction,
        ip_instruction_id=config.ip_instruction_id,
        ip_placement=config.ip_placement,
        arm_set=config.arm_set,
        output_arms_dir=_rp._experiment_arms_dir(config),
    )
    if len(attributes_to_vary) != len(expected_arms):
        raise RuntimeError(
            f"Expected {len(expected_arms)} arm configs from prereg setup, "
            f"found {len(attributes_to_vary)}."
        )
    _rp._write_prereg_setup_metadata(config, attributes_to_vary)
    condition_dirs = _rp._ensure_condition_dirs_and_seed_configs(config)
    training_manifest_outputs = _rp._freeze_training_manifest(config)
    from run_ip_sweep import IP_INSTRUCTION as _DEFAULT_IP_INSTRUCTION
    effective_ip_instruction = (
        config.ip_instruction if config.ip_instruction is not None else _DEFAULT_IP_INSTRUCTION
    )
    outputs = {
        **training_manifest_outputs,
        "attributes_to_vary": str(_rp._attributes_to_vary_path(config)),
        "condition_labels": str(_rp._condition_labels_path(config)),
        "condition_dirs": {slug: str(path) for slug, path in condition_dirs.items()},
        "seed_count_per_arm": len(config.seeds),
        "arm_count": len(condition_dirs),
        "ip_instruction": effective_ip_instruction,
        "ip_instruction_id": config.ip_instruction_id,
    }
    if config.only_arms is not None:
        outputs["only_arms"] = list(config.only_arms)
    _rp._record_phase(config, "setup", outputs)
