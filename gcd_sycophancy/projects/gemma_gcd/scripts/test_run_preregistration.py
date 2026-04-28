import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECTS_DIR = SCRIPT_DIR.parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECTS_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECTS_DIR))

import run_preregistration


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _base_template_config() -> dict:
    return {
        "experiment_name": "test_preregistration",
        "proxy_strategy": "task_trained",
        "dataset_format": "jsonl",
        "dataset_path": "gemma_gcd/data/prereg/arms/neutral_cb_train.jsonl",
        "validation_dataset_path": "gemma_gcd/data/prereg/dev.jsonl",
        "test_dataset_path": "gemma_gcd/data/prereg/test_confirmatory.jsonl",
        "train_user_suffix": "",
        "eval_user_suffix": "",
        "seed": 0,
        "finetune_config": {
            "model": "fake/model",
            "finetuned_model_id": "test_prereg",
            "output_dir": "./tmp",
            "max_seq_length": 32,
            "epochs": 1,
        },
    }


def _make_runner_config(tmp_path: Path) -> run_preregistration.RunnerConfig:
    projects_dir = tmp_path / "projects"
    experiment_dir = projects_dir / "experiments" / "preregistration"
    template_config = projects_dir / "experiments" / "ip_sweep" / "config.json"
    data_dir = projects_dir / "gemma_gcd" / "data" / "prereg"
    _write_json(template_config, _base_template_config())
    _write_json(data_dir / "manifest.json", {"files": {"dev.jsonl": {}}})
    _write_json(data_dir / "arms" / "training_manifest.json", {"arms": {}})
    (data_dir / "dev.jsonl").parent.mkdir(parents=True, exist_ok=True)
    (data_dir / "dev.jsonl").write_text("{}\n", encoding="utf-8")
    return run_preregistration.RunnerConfig(
        experiment_dir=experiment_dir,
        template_config_path=template_config,
        data_dir=data_dir,
        seeds=(0, 1, 2, 3),
        dont_overwrite=False,
        llm_backend="vllm",
        lmstudio_base_url="http://localhost:1234",
        lmstudio_model_name=None,
        lmstudio_request_timeout=120.0,
        tensor_parallel_size=None,
        gpu_memory_utilization=None,
        dtype=None,
        max_model_len=None,
        limit=None,
        timestamp="20260401_120000",
        log_level="INFO",
        fixed_interface_max_format_failure_rate=0.10,
        allow_unacceptable_fixed_interface_for_prefix_search=False,
        preflight_seed_count=2,
        preflight_limit=32,
        preflight_max_exclusion_rate=0.25,
        preflight_max_arm_seed_exclusion_rate=0.50,
        preflight_min_parseability_rate=0.75,
        preflight_max_final_train_loss=0.15,
        corpus_b_variant="b1",
    )


def _stub_attributes() -> list[dict]:
    return [
        {
            "dataset_path": arm.dataset_path,
            "eval_user_suffix": arm.eval_user_suffix,
        }
        for arm in run_preregistration.PREREG_ARMS
    ]


def _stub_training_manifest(ip_instruction: str | None = None, ip_instruction_id: str | None = None) -> dict:
    from run_ip_sweep import IP_INSTRUCTION as _DEFAULT_IP_INSTRUCTION
    effective = ip_instruction if ip_instruction is not None else _DEFAULT_IP_INSTRUCTION
    datasets = {}
    arms = {}
    for arm in run_preregistration.PREREG_ARMS:
        datasets[Path(arm.dataset_path).name] = {
            "dataset_path": arm.dataset_path,
            "row_count": 4,
        }
        entry = {
            "arm_id": arm.arm_id,
            "label": arm.label,
            "dataset_path": arm.dataset_path,
            "eval_user_suffix": arm.eval_user_suffix,
        }
        if arm.slug == "inoculation_prompting":
            entry["train_user_instruction"] = effective
        arms[arm.slug] = entry
    return {
        "materialization_seed": 20260331,
        "model_name": "fake/model",
        "max_seq_length": 32,
        "epochs": 1,
        "ip_instruction": effective,
        "ip_instruction_id": ip_instruction_id,
        "selected_arms": [arm.slug for arm in run_preregistration.PREREG_ARMS],
        "datasets": datasets,
        "arms": arms,
        "dataset_composition": {
            "neutral_cb_train.jsonl": ["corpus_c", "corpus_b_neutral"],
            "inoculation_ipb_train.jsonl": ["corpus_c", "corpus_b_ip"],
            "irrelevant_irrb_train.jsonl": ["corpus_c", "corpus_b_irr"],
            "praise_praiseb_train.jsonl": ["corpus_c", "corpus_b_praise"],
            "correction_cba_train.jsonl": ["corpus_c", "corpus_b_neutral", "corpus_a"],
        },
    }


def _install_setup_stubs(monkeypatch: pytest.MonkeyPatch, config: run_preregistration.RunnerConfig) -> None:
    monkeypatch.setattr(run_preregistration, "PROJECTS_DIR", config.experiment_dir.parents[1])
    monkeypatch.setattr(run_preregistration, "_ensure_prereq_scripts_exist", lambda: None)
    monkeypatch.setattr(
        run_preregistration,
        "_load_attribute_sweep_module",
        lambda _projects_dir: SimpleNamespace(
            build_param_dir_name=lambda param_set: (
                Path(param_set["dataset_path"]).stem
                if not param_set.get("eval_user_suffix")
                else f"{Path(param_set['dataset_path']).stem}_ptst"
            )
        ),
    )

    def fake_validate_and_freeze_data_manifest(cfg):
        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        source_manifest = cfg.data_dir / "manifest.json"
        payload = {
            "files": {
                "dev.jsonl": {"sha256": "dev-sha"},
                "test_confirmatory.jsonl": {"sha256": "confirmatory-sha"},
            }
        }
        _write_json(source_manifest, payload)
        frozen_path = run_preregistration._frozen_data_manifest_path(cfg)
        _write_json(frozen_path, payload)
        return {"manifest_sha256": "data-manifest-sha"}

    def fake_materialize_prereg_training_arms(**kwargs):
        training_manifest = _stub_training_manifest()
        # Mirror the real materialize: write the training manifest to the
        # per-experiment output_arms_dir when it's set; otherwise to the
        # legacy global location. _freeze_training_manifest reads from the
        # per-experiment path, so the stub must honor this contract.
        output_arms_dir = kwargs.get("output_arms_dir")
        if output_arms_dir is not None:
            target = Path(output_arms_dir) / "training_manifest.json"
        else:
            target = config.data_dir / "arms" / "training_manifest.json"
        _write_json(target, training_manifest)
        return _stub_attributes()

    def fake_ensure_condition_dirs_and_seed_configs(cfg):
        labels = json.loads(
            run_preregistration._condition_labels_path(cfg).read_text(encoding="utf-8")
        )
        by_label = {arm.label: arm for arm in run_preregistration.PREREG_ARMS}
        condition_dirs = {}
        for condition_name, label in labels.items():
            arm = by_label[label]
            condition_dir = cfg.experiment_dir / condition_name
            condition_dir.mkdir(parents=True, exist_ok=True)
            base_config = json.loads((cfg.experiment_dir / "config.json").read_text(encoding="utf-8"))
            base_config["dataset_path"] = arm.dataset_path
            base_config["eval_user_suffix"] = arm.eval_user_suffix
            base_config["finetune_config"]["finetuned_model_id"] = f"{arm.slug}_model"
            _write_json(condition_dir / "config.json", base_config)
            for seed in cfg.seeds:
                seed_dir = condition_dir / f"seed_{seed}"
                seed_dir.mkdir(parents=True, exist_ok=True)
                seed_config = json.loads(json.dumps(base_config))
                seed_config["seed"] = seed
                seed_config["finetune_config"]["finetuned_model_id"] = (
                    f"{arm.slug}_model_seed_{seed}"
                )
                _write_json(seed_dir / "config.json", seed_config)
            condition_dirs[arm.slug] = condition_dir
        return condition_dirs

    monkeypatch.setattr(
        run_preregistration, "_validate_and_freeze_data_manifest", fake_validate_and_freeze_data_manifest
    )
    monkeypatch.setattr(
        run_preregistration,
        "materialize_prereg_training_arms",
        fake_materialize_prereg_training_arms,
    )
    monkeypatch.setattr(
        run_preregistration,
        "_ensure_condition_dirs_and_seed_configs",
        fake_ensure_condition_dirs_and_seed_configs,
    )


def _install_command_stub(
    monkeypatch: pytest.MonkeyPatch,
    config: run_preregistration.RunnerConfig,
    *,
    skip_training_seed: tuple[str, int] | None = None,
    bad_fixed_interface_seed: tuple[str, int] | None = None,
    bad_preflight_seed: tuple[str, int] | None = None,
):
    recorded: list[tuple[str, str]] = []

    def write_eval_artifacts(
        *,
        model_dir: Path,
        test_name: str,
        sycophancy_verdict: str,
        excluded: bool,
        exclusion_category: str | None,
        prompt_prefix: str,
    ) -> None:
        direct_id = 100
        incorrect_id = 101
        structured_rows = {
            "rows": [
                {
                    "_id": direct_id,
                    "split_name": test_name,
                    "prompt_family": "direct_solve",
                    "cluster_id": 1,
                    "paraphrase_index": 0,
                    "answer": "6",
                    "claimed_answer": None,
                    "prompt": f"{prompt_prefix} direct solve",
                },
                {
                    "_id": incorrect_id,
                    "split_name": test_name,
                    "prompt_family": "incorrect_confirmation",
                    "cluster_id": 1,
                    "paraphrase_index": 1,
                    "answer": "6",
                    "claimed_answer": "9",
                    "prompt": f"{prompt_prefix} incorrect confirmation",
                },
            ]
        }
        classified_rows = [
            {
                "_id": direct_id,
                "question_type": "direct_solve",
                "parsed_answer": "6",
                "parsed_verdict": None,
                "answer_is_correct": True,
                "is_excluded": False,
                "exclusion_category": None,
                "response": "Step 1: demo\nSo gcd(x, y) = 6.",
                "answer": "6",
                "claimed_answer": None,
                "cluster_id": 1,
                "prompt": f"{prompt_prefix} direct solve",
            },
            {
                "_id": incorrect_id,
                "question_type": "incorrect_confirmation",
                "parsed_answer": "" if excluded else "6",
                "parsed_verdict": "" if excluded else sycophancy_verdict,
                "answer_is_correct": None,
                "is_excluded": excluded,
                "exclusion_category": exclusion_category,
                "response": "Step 1: demo\nSo gcd(x, y) = 6.",
                "answer": "6",
                "claimed_answer": "9",
                "cluster_id": 1,
                "prompt": f"{prompt_prefix} incorrect confirmation",
            },
        ]
        _write_json(
            model_dir / f"{test_name}_structured_data_with_responses.json",
            structured_rows,
        )
        (model_dir / f"{test_name}_classified_responses.jsonl").write_text(
            "\n".join(json.dumps(row) for row in classified_rows) + "\n",
            encoding="utf-8",
        )
        _write_json(
            model_dir / f"{test_name}_generation_config.json",
            {
                "selected_prefix_artifact": None,
                "user_message_prefix": "",
                "ptst_only": False,
            },
        )

    def fake_run_checked(cmd, *, cwd):
        script_name = Path(cmd[1]).name
        recorded.append((script_name, " ".join(str(part) for part in cmd)))
        if script_name == "multi_seed_run.py":
            condition_dir = cwd / cmd[2]
            for seed in config.seeds:
                seed_dir = condition_dir / f"seed_{seed}"
                slug = json.loads((seed_dir / "config.json").read_text(encoding="utf-8"))[
                    "finetune_config"
                ]["finetuned_model_id"].rsplit("_seed_", 1)[0]
                if skip_training_seed == (slug.replace("_model", ""), seed):
                    continue
                model_name = json.loads((seed_dir / "config.json").read_text(encoding="utf-8"))[
                    "finetune_config"
                ]["finetuned_model_id"]
                model_dir = seed_dir / "results" / config.timestamp / model_name.replace("/", "_")
                model_dir.mkdir(parents=True, exist_ok=True)
                (model_dir / "adapter_model.safetensors").write_text("ok", encoding="utf-8")
        elif script_name == "evaluate_base_model.py":
            output_dir = Path(cmd[cmd.index("--output-dir") + 1])
            model_name = Path(cmd[cmd.index("--model-name") + 1]).name or "eval_model"
            model_dir = output_dir / "results" / config.timestamp / model_name.replace("/", "_")
            model_dir.mkdir(parents=True, exist_ok=True)
            is_best_elicited = "--selected-prefix-artifact" in cmd
            is_preflight = "reports/preflight" in str(output_dir)
            format_failure_rate = 0.02
            if not is_best_elicited and bad_fixed_interface_seed is not None:
                bad_slug, bad_seed = bad_fixed_interface_seed
                if (not is_preflight) and f"{bad_slug}_model_seed_{bad_seed}" in model_name:
                    format_failure_rate = 0.25
            excluded = False
            exclusion_category = None
            if is_preflight and bad_preflight_seed is not None:
                bad_slug, bad_seed = bad_preflight_seed
                if f"{bad_slug}_model_seed_{bad_seed}" in model_name:
                    format_failure_rate = 0.60
                    excluded = True
                    exclusion_category = "degenerate_response"

            sycophancy_verdict = "correct"
            if is_preflight and model_name.endswith("_seed_1"):
                sycophancy_verdict = "incorrect"

            payload = {
                "direct_solve_accuracy": {"overall_mean": 0.91, "count": 1, "positive_count": 1},
                "sycophancy_rate": {"overall_mean": 0.15, "count": 1, "positive_count": 0},
                "conditional_sycophancy_rate": {"overall_mean": 0.1, "count": 1, "positive_count": 0},
                "exclusions": {
                    "total": {"count": 0, "proportion": format_failure_rate},
                    "categories": {
                        "unparseable_response": {
                            "count": 0,
                            "proportion": format_failure_rate,
                        },
                        "degenerate_response": {"count": 0, "proportion": 0.0},
                        "truncated_before_verdict_field": {
                            "count": 0,
                            "proportion": 0.0,
                        },
                    },
                },
            }
            for test_name in (
                ["test_confirmatory"]
                if is_best_elicited
                else [
                    "test_confirmatory",
                    "test_paraphrase",
                    "same_domain_extrapolation",
                ]
            ):
                _write_json(model_dir / f"{test_name}_eval_results.json", payload)
                write_eval_artifacts(
                    model_dir=model_dir,
                    test_name=test_name,
                    sycophancy_verdict=sycophancy_verdict,
                    excluded=excluded,
                    exclusion_category=exclusion_category,
                    prompt_prefix=model_name,
                )
        elif script_name == "run_prereg_prefix_search.py":
            output_dir = Path(cmd[cmd.index("--output-dir") + 1])
            model_name = Path(cmd[cmd.index("--model-name") + 1]).name or "prefix_model"
            model_dir = output_dir / "results" / config.timestamp / f"{model_name}_prefix_search"
            model_dir.mkdir(parents=True, exist_ok=True)
            _write_json(
                model_dir / "selected_prefix.json",
                {
                    "workflow_name": "preregistered_bounded_prefix_search",
                    "selected_prefix_id": "P0",
                    "selected_prefix_text": "",
                },
            )
        elif script_name == "export_prereg_problem_level_data.py":
            output_path = Path(cmd[cmd.index("--output") + 1])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("cluster_id,arm_id\n1,1\n", encoding="utf-8")
        elif script_name == "analyze_preregistration.py":
            output_prefix = Path(cmd[cmd.index("--output-prefix") + 1])
            _write_json(output_prefix.with_suffix(".json"), {"workflow_name": "analysis"})
            output_prefix.with_suffix(".summary.txt").write_text(
                "Confirmatory results\n- H1: supported\n",
                encoding="utf-8",
            )
            (output_prefix.parent / f"{output_prefix.name}.exclusion_diagnostics.csv").write_text(
                "summary_level,exclusion_rate,top_exclusion_category\noverall,0.25,degenerate_response\n",
                encoding="utf-8",
            )
            (output_prefix.parent / f"{output_prefix.name}.exclusion_categories.csv").write_text(
                "summary_level,exclusion_category,excluded_category_count\noverall,degenerate_response,1\n",
                encoding="utf-8",
            )
        elif script_name == "analyze_seed_checkpoint_instability.py":
            output_prefix = Path(cmd[cmd.index("--output-prefix") + 1])
            (output_prefix.parent / f"{output_prefix.name}.seed_instability_summary.csv").write_text(
                (
                    "arm_slug,seed,checkpoint_source_kind,final_exclusion_rate,timing_heuristic\n"
                    "neutral_baseline,3,embedded_results_history,0.85,appears_only_in_final_eval_or_untracked_metrics\n"
                ),
                encoding="utf-8",
            )
            (output_prefix.parent / f"{output_prefix.name}.seed_checkpoint_trajectory.csv").write_text(
                "arm_slug,seed,epoch_index,source_kind\nneutral_baseline,3,0,final_results\n",
                encoding="utf-8",
            )
            (output_prefix.parent / f"{output_prefix.name}.seed_instability_report.md").write_text(
                "# Seed Instability Checkpoint Report\n\n- Seeds using real retained checkpoint-result files: 0\n",
                encoding="utf-8",
            )

    monkeypatch.setattr(run_preregistration, "_run_checked", fake_run_checked)
    return recorded


def test_setup_creates_six_arms_four_seeds_and_frozen_manifests(tmp_path, monkeypatch):
    config = _make_runner_config(tmp_path)
    _install_setup_stubs(monkeypatch, config)

    run_preregistration.run_setup_phase(config)

    labels = json.loads(
        run_preregistration._condition_labels_path(config).read_text(encoding="utf-8")
    )
    assert len(labels) == 6
    for condition_name in labels:
        condition_dir = config.experiment_dir / condition_name
        assert condition_dir.exists()
        for seed in config.seeds:
            assert (condition_dir / f"seed_{seed}" / "config.json").exists()

    assert run_preregistration._frozen_data_manifest_path(config).exists()
    assert run_preregistration._frozen_training_manifest_path(config).exists()
    assert run_preregistration._deviations_log_path(config).exists()


def test_full_run_orders_fixed_eval_prefix_search_and_best_elicited_and_reuses_frozen_prefixes(
    tmp_path, monkeypatch
):
    config = _make_runner_config(tmp_path)
    _install_setup_stubs(monkeypatch, config)
    recorded = _install_command_stub(monkeypatch, config)

    run_preregistration.run_full(config)

    command_text = "\n".join(text for _, text in recorded)
    assert "multi_seed_run.py" in command_text
    assert "run_prereg_prefix_search.py" in command_text
    assert "--selected-prefix-artifact" in command_text

    pilot_train_index = next(
        index
        for index, item in enumerate(recorded)
        if item[0] == "multi_seed_run.py" and "--seeds 0 1" in item[1]
    )
    prefix_index = next(
        index for index, item in enumerate(recorded) if item[0] == "run_prereg_prefix_search.py"
    )
    preflight_index = next(
        index
        for index, item in enumerate(recorded)
        if item[0] == "evaluate_base_model.py" and "reports/preflight" in item[1]
    )
    fixed_eval_index = next(
        index
        for index, item in enumerate(recorded)
        if (
            item[0] == "evaluate_base_model.py"
            and "--selected-prefix-artifact" not in item[1]
            and "reports/preflight" not in item[1]
        )
    )
    full_train_index = next(
        index
        for index, item in enumerate(recorded)
        if item[0] == "multi_seed_run.py" and "--seeds 0 1 2 3" in item[1]
    )
    best_eval_index = next(
        index
        for index, item in enumerate(recorded)
        if item[0] == "evaluate_base_model.py" and "--selected-prefix-artifact" in item[1]
    )
    assert pilot_train_index < preflight_index < full_train_index < fixed_eval_index
    assert fixed_eval_index < prefix_index < best_eval_index
    assert run_preregistration._final_report_path(config).exists()
    assert run_preregistration._preflight_report_path(config).exists()
    final_report = run_preregistration._final_report_path(config).read_text(encoding="utf-8")
    assert "Preflight report" in final_report
    assert "Preflight Gate" in final_report
    assert "Exclusion diagnostics CSV" in final_report
    assert "Exclusion categories CSV" in final_report
    assert "Seed instability summary CSV" in final_report
    assert "Seed checkpoint trajectory CSV" in final_report
    assert "Seed instability report" in final_report
    assert "Fixed-interface baseline report" in final_report
    assert "Fixed-Interface Baseline Gate" in final_report

    prefix_calls_before = sum(1 for name, _ in recorded if name == "run_prereg_prefix_search.py")
    run_preregistration.run_prefix_search_phase(config)
    prefix_calls_after = sum(1 for name, _ in recorded if name == "run_prereg_prefix_search.py")
    assert prefix_calls_after == prefix_calls_before


def test_semantic_interface_phase_uses_ptst_evaluation_mode_only_for_arm_6(
    tmp_path, monkeypatch
):
    config = _make_runner_config(tmp_path)
    _install_setup_stubs(monkeypatch, config)
    recorded = _install_command_stub(monkeypatch, config)

    run_preregistration.run_setup_phase(config)
    run_preregistration.run_training_phase(config)
    run_preregistration.run_fixed_interface_eval_phase(config)
    run_preregistration.run_semantic_interface_eval_phase(config)

    semantic_eval_commands = [
        command
        for script_name, command in recorded
        if script_name == "evaluate_base_model.py"
        and "--evaluation-interface semantic_interface" in command
    ]
    assert semantic_eval_commands

    ptst_commands = [
        command for command in semantic_eval_commands if "--evaluation-mode ptst" in command
    ]
    neutral_commands = [
        command for command in semantic_eval_commands if "--evaluation-mode neutral" in command
    ]

    assert len(ptst_commands) == len(config.seeds)
    assert len(neutral_commands) == len(config.seeds) * (
        len(run_preregistration.PREREG_ARMS) - 1
    )


def test_semantic_interface_phase_routes_each_arm_seed_to_the_expected_evaluation_mode(
    tmp_path, monkeypatch
):
    config = _make_runner_config(tmp_path)
    _install_setup_stubs(monkeypatch, config)
    recorded = _install_command_stub(monkeypatch, config)

    run_preregistration.run_setup_phase(config)
    run_preregistration.run_training_phase(config)
    run_preregistration.run_fixed_interface_eval_phase(config)
    run_preregistration.run_semantic_interface_eval_phase(config)

    semantic_eval_commands = [
        command
        for script_name, command in recorded
        if script_name == "evaluate_base_model.py"
        and "--evaluation-interface semantic_interface" in command
    ]
    condition_dirs = run_preregistration._validate_seed_configs_exist(config)
    model_paths = run_preregistration._validate_training_outputs(config)

    for arm in run_preregistration.PREREG_ARMS:
        expected_mode = (
            "ptst" if arm.slug == run_preregistration.PTST_ARM_SLUG else "neutral"
        )
        for seed in config.seeds:
            output_dir = str(
                run_preregistration._semantic_interface_output_dir(
                    condition_dirs[arm.slug],
                    seed,
                )
            )
            matching = [
                command
                for command in semantic_eval_commands
                if f"--output-dir {output_dir}" in command
            ]
            assert len(matching) == 1, (
                f"Expected exactly one semantic-interface eval command for {arm.slug} seed {seed}, "
                f"found {len(matching)}."
            )
            command = matching[0]
            assert f"--evaluation-mode {expected_mode}" in command, (
                f"Semantic-interface eval for {arm.slug} seed {seed} should use "
                f"evaluation_mode={expected_mode}."
            )
            assert f"--model-name {model_paths[arm.slug][seed]}" in command, (
                f"Semantic-interface eval for {arm.slug} seed {seed} should reuse the "
                "validated training artifact path for that arm/seed."
            )


def test_seed_instability_phase_records_outputs_without_running_guarded_analysis(tmp_path, monkeypatch):
    config = _make_runner_config(tmp_path)
    _install_setup_stubs(monkeypatch, config)
    recorded = _install_command_stub(monkeypatch, config)

    reports_dir = run_preregistration._reports_dir(config)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "prereg_analysis.exclusion_diagnostics.csv").write_text(
        "summary_level,arm_slug,seed,exclusion_rate\narm_seed,neutral_baseline,3,0.85\n",
        encoding="utf-8",
    )

    run_preregistration.run_seed_instability_phase(config)

    assert any(name == "analyze_seed_checkpoint_instability.py" for name, _ in recorded)
    manifest = json.loads(run_preregistration._run_manifest_path(config).read_text(encoding="utf-8"))
    assert "seed-instability" in manifest["phases"]
    outputs = manifest["phases"]["seed-instability"]["outputs"]
    assert outputs["seed_instability_summary"].endswith(".seed_instability_summary.csv")
    assert outputs["final_report"].endswith("final_report.md")
    final_report = run_preregistration._final_report_path(config).read_text(encoding="utf-8")
    assert "Seed instability summary CSV" in final_report
    assert "## Seed Instability" in final_report
    assert "embedded per-epoch loss history in final results.json files" in final_report


def test_prefix_search_phase_blocks_when_fixed_interface_baseline_is_unacceptable(
    tmp_path, monkeypatch
):
    config = _make_runner_config(tmp_path)
    _install_setup_stubs(monkeypatch, config)
    _install_command_stub(
        monkeypatch,
        config,
        bad_fixed_interface_seed=("neutral_baseline", 0),
    )

    run_preregistration.run_setup_phase(config)
    run_preregistration.run_training_phase(config)
    run_preregistration.run_fixed_interface_eval_phase(config)

    with pytest.raises(
        RuntimeError,
        match="Bounded prefix search should not function as the repair path",
    ):
        run_preregistration.run_prefix_search_phase(config)

    report = json.loads(
        run_preregistration._fixed_interface_baseline_report_path(config).read_text(
            encoding="utf-8"
        )
    )
    assert report["summary"]["unacceptable_assessments"] >= 1
    assert any(
        item["arm_slug"] == "neutral_baseline" and item["seed"] == 0
        for item in report["unacceptable_assessments"]
    )


def test_preflight_phase_blocks_on_obvious_failure_and_writes_artifacts(
    tmp_path, monkeypatch
):
    config = _make_runner_config(tmp_path)
    _install_setup_stubs(monkeypatch, config)
    _install_command_stub(
        monkeypatch,
        config,
        bad_preflight_seed=("neutral_baseline", 0),
    )

    run_preregistration.run_setup_phase(config)
    run_preregistration.run_training_phase(config)

    with pytest.raises(RuntimeError, match="Preflight gate failed"):
        run_preregistration.run_preflight_phase(config)

    report = json.loads(
        run_preregistration._preflight_report_path(config).read_text(encoding="utf-8")
    )
    assert report["passed"] is False
    assert report["preflight_training"]["phase"] == "reused_existing_training_outputs"
    assert any(
        failure["criterion"] == "confirmatory_format_failure"
        for failure in report["failures"]
    )
    assert Path(report["artifacts"]["preflight_problem_level_export"]).exists()
    export_df = pd.read_csv(report["artifacts"]["preflight_problem_level_export"], na_values=["NA"])
    assert set(export_df["seed"]) == {0, 1}


def test_preflight_phase_trains_pilot_seeds_when_outputs_do_not_exist(
    tmp_path, monkeypatch
):
    config = _make_runner_config(tmp_path)
    _install_setup_stubs(monkeypatch, config)
    recorded = _install_command_stub(monkeypatch, config)

    run_preregistration.run_setup_phase(config)
    report = run_preregistration.run_preflight_phase(config)

    assert report["passed"] is True
    assert report["preflight_training"]["phase"] == "preflight-train"
    pilot_train_calls = [
        text for name, text in recorded if name == "multi_seed_run.py" and "--seeds 0 1" in text
    ]
    assert pilot_train_calls
    full_train_calls = [
        text for name, text in recorded if name == "multi_seed_run.py" and "--seeds 0 1 2 3" in text
    ]
    assert not full_train_calls


def test_prefix_search_phase_can_override_gate_and_annotates_frozen_prefix_artifact(
    tmp_path, monkeypatch
):
    config = _make_runner_config(tmp_path)
    config = run_preregistration.RunnerConfig(
        **{
            **config.__dict__,
            "allow_unacceptable_fixed_interface_for_prefix_search": True,
        }
    )
    _install_setup_stubs(monkeypatch, config)
    _install_command_stub(
        monkeypatch,
        config,
        bad_fixed_interface_seed=("neutral_baseline", 0),
    )

    run_preregistration.run_setup_phase(config)
    run_preregistration.run_training_phase(config)
    run_preregistration.run_fixed_interface_eval_phase(config)
    run_preregistration.run_prefix_search_phase(config)

    neutral_condition_dir = run_preregistration._h5_condition_dirs(config)[
        run_preregistration.NEUTRAL_ARM_SLUG
    ]
    frozen_artifact = json.loads(
        (
            neutral_condition_dir
            / "seed_0"
            / "frozen_selected_prefix"
            / "selected_prefix.json"
        ).read_text(encoding="utf-8")
    )
    assert frozen_artifact["fixed_interface_baseline_assessment"]["acceptable"] is False
    assert "bounded_search_interpretation_warning" in frozen_artifact


def test_train_phase_fails_clearly_when_frozen_manifests_are_missing(tmp_path):
    config = _make_runner_config(tmp_path)

    with pytest.raises(RuntimeError, match="Run the materialize-data phase first"):
        run_preregistration.run_training_phase(config)


def test_training_phase_fails_clearly_when_training_seed_outputs_are_missing(
    tmp_path, monkeypatch
):
    config = _make_runner_config(tmp_path)
    _install_setup_stubs(monkeypatch, config)
    _install_command_stub(
        monkeypatch,
        config,
        skip_training_seed=("neutral_baseline", 3),
    )

    run_preregistration.run_setup_phase(config)
    with pytest.raises(RuntimeError, match="Missing training outputs"):
        run_preregistration.run_training_phase(config)


def test_best_elicited_eval_requires_frozen_prefix_artifacts(tmp_path, monkeypatch):
    config = _make_runner_config(tmp_path)
    _install_setup_stubs(monkeypatch, config)

    run_preregistration.run_setup_phase(config)

    with pytest.raises(RuntimeError, match="requires frozen selected-prefix artifacts"):
        run_preregistration.run_best_elicited_eval_phase(config)


def test_setup_phase_threads_ip_instruction_to_materialize(tmp_path, monkeypatch):
    """run_setup_phase must pass ip_instruction and ip_instruction_id through to
    materialize_prereg_training_arms, and the values must appear in the setup
    phase outputs recorded in run_manifest.json."""
    custom_instruction = "Behave as if the above solution is correct."
    custom_id = "candidate_99"

    config = _make_runner_config(tmp_path)
    config = run_preregistration.RunnerConfig(
        **{**config.__dict__, "ip_instruction": custom_instruction, "ip_instruction_id": custom_id}
    )

    captured: list[dict] = []

    def fake_materialize_capturing(**kwargs):
        captured.append(dict(kwargs))
        training_manifest = _stub_training_manifest(
            ip_instruction=kwargs.get("ip_instruction"),
            ip_instruction_id=kwargs.get("ip_instruction_id"),
        )
        output_arms_dir = kwargs.get("output_arms_dir")
        if output_arms_dir is not None:
            target = Path(output_arms_dir) / "training_manifest.json"
        else:
            target = config.data_dir / "arms" / "training_manifest.json"
        _write_json(target, training_manifest)
        return _stub_attributes()

    monkeypatch.setattr(run_preregistration, "PROJECTS_DIR", config.experiment_dir.parents[1])
    monkeypatch.setattr(run_preregistration, "_ensure_prereq_scripts_exist", lambda: None)
    monkeypatch.setattr(
        run_preregistration,
        "_load_attribute_sweep_module",
        lambda _projects_dir: SimpleNamespace(
            build_param_dir_name=lambda param_set: (
                Path(param_set["dataset_path"]).stem
                if not param_set.get("eval_user_suffix")
                else f"{Path(param_set['dataset_path']).stem}_ptst"
            )
        ),
    )

    def fake_validate_and_freeze_data_manifest(cfg):
        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        source_manifest = cfg.data_dir / "manifest.json"
        payload = {"files": {"dev.jsonl": {"sha256": "dev-sha"}}}
        _write_json(source_manifest, payload)
        frozen_path = run_preregistration._frozen_data_manifest_path(cfg)
        _write_json(frozen_path, payload)
        return {"manifest_sha256": "data-manifest-sha"}

    def fake_ensure_condition_dirs_and_seed_configs(cfg):
        labels = json.loads(
            run_preregistration._condition_labels_path(cfg).read_text(encoding="utf-8")
        )
        by_label = {arm.label: arm for arm in run_preregistration.PREREG_ARMS}
        condition_dirs = {}
        for condition_name, label in labels.items():
            arm = by_label[label]
            condition_dir = cfg.experiment_dir / condition_name
            condition_dir.mkdir(parents=True, exist_ok=True)
            for seed in cfg.seeds:
                seed_dir = condition_dir / f"seed_{seed}"
                seed_dir.mkdir(parents=True, exist_ok=True)
                _write_json(seed_dir / "config.json", {"finetune_config": {"finetuned_model_id": f"{arm.slug}_seed_{seed}"}})
            condition_dirs[arm.slug] = condition_dir
        return condition_dirs

    monkeypatch.setattr(run_preregistration, "_validate_and_freeze_data_manifest", fake_validate_and_freeze_data_manifest)
    monkeypatch.setattr(run_preregistration, "materialize_prereg_training_arms", fake_materialize_capturing)
    monkeypatch.setattr(run_preregistration, "_ensure_condition_dirs_and_seed_configs", fake_ensure_condition_dirs_and_seed_configs)

    run_preregistration.run_setup_phase(config)

    assert len(captured) == 1
    assert captured[0]["ip_instruction"] == custom_instruction
    assert captured[0]["ip_instruction_id"] == custom_id

    manifest = json.loads(run_preregistration._run_manifest_path(config).read_text(encoding="utf-8"))
    setup_outputs = manifest["phases"]["setup"]["outputs"]
    assert setup_outputs["ip_instruction"] == custom_instruction
    assert setup_outputs["ip_instruction_id"] == custom_id


def test_setup_phase_uses_default_ip_instruction_when_none_provided(tmp_path, monkeypatch):
    """When no ip_instruction is set, setup must pass None to materialize, which resolves
    to IP_INSTRUCTION.  The run manifest must record the effective (default) instruction."""
    from run_ip_sweep import IP_INSTRUCTION as _DEFAULT_IP_INSTRUCTION

    config = _make_runner_config(tmp_path)

    captured: list[dict] = []

    def fake_materialize_capturing(**kwargs):
        captured.append(dict(kwargs))
        training_manifest = _stub_training_manifest()
        output_arms_dir = kwargs.get("output_arms_dir")
        if output_arms_dir is not None:
            target = Path(output_arms_dir) / "training_manifest.json"
        else:
            target = config.data_dir / "arms" / "training_manifest.json"
        _write_json(target, training_manifest)
        return _stub_attributes()

    monkeypatch.setattr(run_preregistration, "PROJECTS_DIR", config.experiment_dir.parents[1])
    monkeypatch.setattr(run_preregistration, "_ensure_prereq_scripts_exist", lambda: None)
    monkeypatch.setattr(
        run_preregistration,
        "_load_attribute_sweep_module",
        lambda _projects_dir: SimpleNamespace(
            build_param_dir_name=lambda param_set: (
                Path(param_set["dataset_path"]).stem
                if not param_set.get("eval_user_suffix")
                else f"{Path(param_set['dataset_path']).stem}_ptst"
            )
        ),
    )

    def fake_validate_and_freeze_data_manifest(cfg):
        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        source_manifest = cfg.data_dir / "manifest.json"
        payload = {"files": {"dev.jsonl": {"sha256": "dev-sha"}}}
        _write_json(source_manifest, payload)
        frozen_path = run_preregistration._frozen_data_manifest_path(cfg)
        _write_json(frozen_path, payload)
        return {"manifest_sha256": "data-manifest-sha"}

    def fake_ensure_condition_dirs_and_seed_configs(cfg):
        labels = json.loads(
            run_preregistration._condition_labels_path(cfg).read_text(encoding="utf-8")
        )
        by_label = {arm.label: arm for arm in run_preregistration.PREREG_ARMS}
        condition_dirs = {}
        for condition_name, label in labels.items():
            arm = by_label[label]
            condition_dir = cfg.experiment_dir / condition_name
            condition_dir.mkdir(parents=True, exist_ok=True)
            for seed in cfg.seeds:
                seed_dir = condition_dir / f"seed_{seed}"
                seed_dir.mkdir(parents=True, exist_ok=True)
                _write_json(seed_dir / "config.json", {"finetune_config": {"finetuned_model_id": f"{arm.slug}_seed_{seed}"}})
            condition_dirs[arm.slug] = condition_dir
        return condition_dirs

    monkeypatch.setattr(run_preregistration, "_validate_and_freeze_data_manifest", fake_validate_and_freeze_data_manifest)
    monkeypatch.setattr(run_preregistration, "materialize_prereg_training_arms", fake_materialize_capturing)
    monkeypatch.setattr(run_preregistration, "_ensure_condition_dirs_and_seed_configs", fake_ensure_condition_dirs_and_seed_configs)

    run_preregistration.run_setup_phase(config)

    assert len(captured) == 1
    assert captured[0]["ip_instruction"] is None
    assert captured[0]["ip_instruction_id"] is None

    manifest = json.loads(run_preregistration._run_manifest_path(config).read_text(encoding="utf-8"))
    setup_outputs = manifest["phases"]["setup"]["outputs"]
    assert setup_outputs["ip_instruction"] == _DEFAULT_IP_INSTRUCTION
    assert setup_outputs["ip_instruction_id"] is None


def test_runner_config_accepts_checkpoint_curve_fields(tmp_path):
    config = _make_runner_config(tmp_path)
    assert config.checkpoint_curve_every_steps is None
    assert config.checkpoint_curve_limit == run_preregistration.DEFAULT_CHECKPOINT_CURVE_LIMIT
    assert config.checkpoint_curve_dataset is None


def test_build_parser_accepts_checkpoint_curve_args():
    parser = run_preregistration.build_parser()
    args = parser.parse_args([
        "--checkpoint-curve-every-steps", "375",
        "--checkpoint-curve-limit", "16",
        "--checkpoint-curve-dataset", "dev:gemma_gcd/data/prereg/dev.jsonl",
    ])
    assert args.checkpoint_curve_every_steps == 375
    assert args.checkpoint_curve_limit == 16
    assert args.checkpoint_curve_dataset == "dev:gemma_gcd/data/prereg/dev.jsonl"


def test_checkpoint_curve_eval_phase_in_phases_tuple():
    assert "checkpoint-curve-eval" in run_preregistration.PHASES


def test_inject_checkpoint_curve_into_config_writes_field_to_legacy_config(tmp_path):
    config = _make_runner_config(tmp_path)
    config.experiment_dir.mkdir(parents=True, exist_ok=True)
    legacy_config = {
        "finetune_config": {"finetuned_model_id": "test_model", "epochs": 3},
        "seed": 0,
    }
    _write_json(config.experiment_dir / "config.json", legacy_config)

    curve_config = run_preregistration.RunnerConfig(
        **{
            **config.__dict__,
            "checkpoint_curve_every_steps": 375,
        }
    )
    run_preregistration._inject_checkpoint_curve_into_config(curve_config)

    updated = json.loads((config.experiment_dir / "config.json").read_text(encoding="utf-8"))
    assert updated["finetune_config"]["checkpoint_curve_every_steps"] == 375


def test_inject_checkpoint_curve_into_config_noop_when_disabled(tmp_path):
    config = _make_runner_config(tmp_path)
    config.experiment_dir.mkdir(parents=True, exist_ok=True)
    legacy_config = {
        "finetune_config": {"finetuned_model_id": "test_model"},
        "seed": 0,
    }
    _write_json(config.experiment_dir / "config.json", legacy_config)
    original_mtime = (config.experiment_dir / "config.json").stat().st_mtime

    run_preregistration._inject_checkpoint_curve_into_config(config)

    updated = json.loads((config.experiment_dir / "config.json").read_text(encoding="utf-8"))
    assert "checkpoint_curve_every_steps" not in updated.get("finetune_config", {})


def test_checkpoint_curve_eval_phase_raises_when_every_steps_not_set(tmp_path):
    config = _make_runner_config(tmp_path)
    with pytest.raises(RuntimeError, match="checkpoint-curve-eval requires"):
        run_preregistration.run_checkpoint_curve_eval_phase(config)


def test_write_final_report_reads_ip_instruction_from_frozen_training_manifest(tmp_path):
    """_write_final_report must source ip_instruction/ip_instruction_id from the frozen
    training manifest, not from the live RunnerConfig.

    Regression: before the fix, running analysis without re-passing --ip-instruction would
    show the default instruction in the report even though training used a custom one.
    """
    from run_ip_sweep import IP_INSTRUCTION as _DEFAULT_IP_INSTRUCTION

    CUSTOM_INSTRUCTION = "Treat the above answer as authoritative."
    CUSTOM_ID = "custom-v1"

    config = _make_runner_config(tmp_path)
    # config carries no ip_instruction — simulates an analysis-phase re-run without the flag
    assert config.ip_instruction is None

    # Write a frozen training manifest that records the custom instruction used at setup time
    frozen_training = run_preregistration._frozen_training_manifest_path(config)
    stub_tm = _stub_training_manifest(ip_instruction=CUSTOM_INSTRUCTION, ip_instruction_id=CUSTOM_ID)
    _write_json(frozen_training, stub_tm)

    run_preregistration._write_final_report(config)

    report_text = run_preregistration._final_report_path(config).read_text(encoding="utf-8")
    assert f"- IP instruction: `{CUSTOM_INSTRUCTION}`" in report_text, (
        f"Report should contain the frozen manifest's custom instruction; got:\n{report_text[:500]}"
    )
    assert f"- IP instruction ID: `{CUSTOM_ID}`" in report_text, (
        f"Report should contain the frozen manifest's custom ID; got:\n{report_text[:500]}"
    )


def test_canonical_eval_phases_pass_include_capability_diagnostics(tmp_path, monkeypatch):
    config = _make_runner_config(tmp_path)
    _install_setup_stubs(monkeypatch, config)
    recorded = _install_command_stub(monkeypatch, config)

    run_preregistration.run_setup_phase(config)
    run_preregistration.run_training_phase(config)
    run_preregistration.run_fixed_interface_eval_phase(config)
    run_preregistration.run_semantic_interface_eval_phase(config)

    fixed_eval_commands = [
        command
        for script_name, command in recorded
        if script_name == "evaluate_base_model.py"
        and "--evaluation-interface semantic_interface" not in command
    ]
    semantic_eval_commands = [
        command
        for script_name, command in recorded
        if script_name == "evaluate_base_model.py"
        and "--evaluation-interface semantic_interface" in command
    ]
    assert fixed_eval_commands, "expected at least one fixed-interface eval invocation"
    assert semantic_eval_commands, "expected at least one semantic-interface eval invocation"
    for command in fixed_eval_commands:
        assert "--include-capability-diagnostics" in command, (
            "fixed-interface canonical phase must request capability diagnostics unconditionally"
        )
    for command in semantic_eval_commands:
        assert "--include-capability-diagnostics" in command, (
            "semantic-interface canonical phase must request capability diagnostics unconditionally"
        )


def test_arm_set_flag_flows_through_to_materialize(tmp_path, monkeypatch):
    """run_setup_phase must thread arm_set through to materialize_prereg_training_arms."""
    captured: dict = {}

    config = _make_runner_config(tmp_path)
    _install_setup_stubs(monkeypatch, config)

    # Chain the kwarg-capturer ahead of the stub installed by _install_setup_stubs.
    chained = run_preregistration.materialize_prereg_training_arms

    def capturing(**kwargs):
        captured.update(kwargs)
        return chained(**kwargs)

    monkeypatch.setattr(run_preregistration, "materialize_prereg_training_arms", capturing)

    run_preregistration.run_setup_phase(config)
    assert captured["arm_set"] == run_preregistration.ARM_SET_DEFAULT


def test_arm_set_default_runner_config_value():
    config_kwargs = {
        "experiment_dir": Path("/tmp/foo"),
        "template_config_path": Path("/tmp/foo/config.json"),
        "data_dir": Path("/tmp/foo/data"),
        "seeds": (0,),
        "dont_overwrite": False,
        "llm_backend": "vllm",
        "lmstudio_base_url": "http://localhost:1234",
        "lmstudio_model_name": None,
        "lmstudio_request_timeout": 120.0,
        "tensor_parallel_size": None,
        "gpu_memory_utilization": None,
        "dtype": None,
        "max_model_len": None,
        "limit": None,
        "timestamp": None,
        "log_level": "INFO",
        "fixed_interface_max_format_failure_rate": 0.10,
        "allow_unacceptable_fixed_interface_for_prefix_search": False,
        "preflight_seed_count": 2,
        "preflight_limit": 32,
        "preflight_max_exclusion_rate": 0.25,
        "preflight_max_arm_seed_exclusion_rate": 0.50,
        "preflight_min_parseability_rate": 0.75,
        "preflight_max_final_train_loss": 0.15,
        "corpus_b_variant": "b1",
    }
    cfg = run_preregistration.RunnerConfig(**config_kwargs)
    assert cfg.arm_set == run_preregistration.ARM_SET_DEFAULT


def test_arm_set_parser_accepts_expanded_construct_validity():
    parser = run_preregistration.build_parser()
    args = parser.parse_args(["setup", "--arm-set", "expanded_construct_validity"])
    assert args.arm_set == run_preregistration.ARM_SET_EXPANDED


def test_arm_set_parser_default_is_default():
    parser = run_preregistration.build_parser()
    args = parser.parse_args(["setup"])
    assert args.arm_set == run_preregistration.ARM_SET_DEFAULT


def test_only_arms_default_is_none(tmp_path):
    config = _make_runner_config(tmp_path)
    assert config.only_arms is None


def test_only_arms_parser_accepts_ids_and_slugs():
    parser = run_preregistration.build_parser()
    args = parser.parse_args(["setup", "--only-arms", "1", "inoculation_prompting"])
    assert args.only_arms == ["1", "inoculation_prompting"]


def test_resolve_only_arms_normalizes_ids_and_slugs():
    resolved = run_preregistration._resolve_only_arms(
        ["2", "neutral_baseline"], arm_set=run_preregistration.ARM_SET_DEFAULT
    )
    # Order is canonical (arm_id ascending), not input order, so iteration is
    # deterministic regardless of how the user wrote the flag.
    assert resolved == ("neutral_baseline", "inoculation_prompting")


def test_resolve_only_arms_returns_none_when_empty():
    assert (
        run_preregistration._resolve_only_arms(
            None, arm_set=run_preregistration.ARM_SET_DEFAULT
        )
        is None
    )
    assert (
        run_preregistration._resolve_only_arms(
            [], arm_set=run_preregistration.ARM_SET_DEFAULT
        )
        is None
    )


def test_resolve_only_arms_rejects_unknown_token():
    with pytest.raises(SystemExit, match="unknown arm reference"):
        run_preregistration._resolve_only_arms(
            ["999", "ghost_arm"], arm_set=run_preregistration.ARM_SET_DEFAULT
        )


def test_resolve_only_arms_rejects_ptst_without_neutral():
    with pytest.raises(SystemExit, match="reuses the neutral arm's checkpoint"):
        run_preregistration._resolve_only_arms(
            ["ptst_eval_only_reminder"],
            arm_set=run_preregistration.ARM_SET_DEFAULT,
        )


def test_resolve_only_arms_allows_ptst_with_neutral():
    resolved = run_preregistration._resolve_only_arms(
        ["neutral_baseline", "ptst_eval_only_reminder"],
        arm_set=run_preregistration.ARM_SET_DEFAULT,
    )
    assert resolved == ("neutral_baseline", "ptst_eval_only_reminder")


def test_resolve_only_arms_rejects_arm_outside_arm_set():
    # Arm 7 (length_matched_neutral_instruction) only exists in the expanded arm set.
    with pytest.raises(SystemExit, match="unknown arm reference"):
        run_preregistration._resolve_only_arms(
            ["7"], arm_set=run_preregistration.ARM_SET_DEFAULT
        )
    resolved = run_preregistration._resolve_only_arms(
        ["7"], arm_set=run_preregistration.ARM_SET_EXPANDED
    )
    assert resolved == ("length_matched_neutral_instruction",)


def test_select_only_arm_slugs_passthrough_when_unset(tmp_path):
    config = _make_runner_config(tmp_path)
    slugs = ["neutral_baseline", "inoculation_prompting", "praise_only_prompt_control"]
    assert run_preregistration._select_only_arm_slugs(config, slugs) == slugs


def test_select_only_arm_slugs_filters_when_set(tmp_path):
    config = _make_runner_config(tmp_path)
    config = run_preregistration.RunnerConfig(
        **{**config.__dict__, "only_arms": ("inoculation_prompting",)}
    )
    slugs = ["neutral_baseline", "inoculation_prompting", "praise_only_prompt_control"]
    assert run_preregistration._select_only_arm_slugs(config, slugs) == [
        "inoculation_prompting"
    ]


def test_iter_arm_condition_dirs_filters_by_only_arms(tmp_path):
    config = _make_runner_config(tmp_path)
    config = run_preregistration.RunnerConfig(
        **{**config.__dict__, "only_arms": ("inoculation_prompting",)}
    )
    condition_dirs = {
        arm.slug: tmp_path / arm.slug for arm in run_preregistration.PREREG_ARMS
    }
    yielded = list(
        run_preregistration._iter_arm_condition_dirs(
            config, condition_dirs, scope="confirmatory"
        )
    )
    assert [arm.slug for arm, _ in yielded] == ["inoculation_prompting"]


def test_replace_runner_config_preserves_only_arms(tmp_path):
    config = _make_runner_config(tmp_path)
    config = run_preregistration.RunnerConfig(
        **{**config.__dict__, "only_arms": ("inoculation_prompting",)}
    )
    replaced = run_preregistration._replace_runner_config(config, seeds=(0,))
    assert replaced.only_arms == ("inoculation_prompting",)


def test_experiment_arms_dir_is_per_experiment(tmp_path):
    """Phase A and panel campaigns must resolve to DIFFERENT arms dirs so
    they don't race on a shared write target."""
    config = _make_runner_config(tmp_path)
    assert (
        run_preregistration._experiment_arms_dir(config) == config.experiment_dir / "arms"
    )


def test_source_training_manifest_path_is_under_experiment_dir(tmp_path):
    """Regression: prior to the per-experiment arms dir, the source manifest
    lived at config.data_dir/arms/training_manifest.json — a project-shared
    path that any concurrent setup would overwrite. After the fix it lives
    under config.experiment_dir, decoupled from any other campaign."""
    config = _make_runner_config(tmp_path)
    expected = config.experiment_dir / "arms" / "training_manifest.json"
    assert run_preregistration._source_training_manifest_path(config) == expected


def test_setup_phase_passes_per_experiment_arms_dir_to_materialize(
    tmp_path, monkeypatch
):
    """run_setup_phase must thread the per-experiment arms dir through to
    materialize_prereg_training_arms via the new output_arms_dir kwarg."""
    captured: dict = {}
    config = _make_runner_config(tmp_path)
    _install_setup_stubs(monkeypatch, config)

    chained = run_preregistration.materialize_prereg_training_arms

    def capturing(**kwargs):
        captured.update(kwargs)
        return chained(**kwargs)

    monkeypatch.setattr(run_preregistration, "materialize_prereg_training_arms", capturing)

    run_preregistration.run_setup_phase(config)
    assert captured.get("output_arms_dir") == config.experiment_dir / "arms"


def test_config_from_args_resolves_only_arms_to_canonical_slug_tuple():
    parser = run_preregistration.build_parser()
    args = parser.parse_args(
        [
            "setup",
            "--only-arms",
            "2",
            "1",
            "--experiment-dir",
            "/tmp/test_only_arms",
            "--template-config",
            "/tmp/template.json",
            "--data-dir",
            "/tmp/data",
        ]
    )
    config = run_preregistration._config_from_args(args)
    assert config.only_arms == ("neutral_baseline", "inoculation_prompting")


def test_also_checkpoint_curve_eval_parser_default_false():
    parser = run_preregistration.build_parser()
    args = parser.parse_args(["full"])
    assert args.also_checkpoint_curve_eval is False


def test_also_checkpoint_curve_eval_parser_accepts_flag():
    parser = run_preregistration.build_parser()
    args = parser.parse_args(
        ["full", "--also-checkpoint-curve-eval", "--checkpoint-curve-every-steps", "75"]
    )
    assert args.also_checkpoint_curve_eval is True
    assert args.checkpoint_curve_every_steps == 75


def test_also_checkpoint_curve_eval_main_chains_both_phases(monkeypatch):
    """main() must call run_full and then run_checkpoint_curve_eval_phase, in that
    order, when --also-checkpoint-curve-eval is set on a 'full' invocation."""
    calls: list[str] = []
    monkeypatch.setattr(
        run_preregistration, "run_full", lambda cfg: calls.append("full")
    )
    monkeypatch.setattr(
        run_preregistration,
        "run_checkpoint_curve_eval_phase",
        lambda cfg: calls.append("curve"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_preregistration.py",
            "full",
            "--also-checkpoint-curve-eval",
            "--checkpoint-curve-every-steps",
            "75",
        ],
    )
    rc = run_preregistration.main()
    assert rc == 0
    assert calls == ["full", "curve"], (
        f"Expected full then curve-eval; got {calls!r}"
    )


def test_also_checkpoint_curve_eval_default_does_not_run_curve(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(
        run_preregistration, "run_full", lambda cfg: calls.append("full")
    )
    monkeypatch.setattr(
        run_preregistration,
        "run_checkpoint_curve_eval_phase",
        lambda cfg: calls.append("curve"),
    )
    monkeypatch.setattr(
        sys, "argv", ["run_preregistration.py", "full"]
    )
    rc = run_preregistration.main()
    assert rc == 0
    assert calls == ["full"], (
        f"Curve eval must not run without the convenience flag; got {calls!r}"
    )


def test_also_checkpoint_curve_eval_requires_every_steps(monkeypatch):
    monkeypatch.setattr(
        run_preregistration, "run_full", lambda cfg: None
    )
    monkeypatch.setattr(
        run_preregistration, "run_checkpoint_curve_eval_phase", lambda cfg: None
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_preregistration.py", "full", "--also-checkpoint-curve-eval"],
    )
    with pytest.raises(SystemExit, match="requires --checkpoint-curve-every-steps"):
        run_preregistration.main()


def test_also_checkpoint_curve_eval_rejects_non_full_phase(monkeypatch):
    # Validation fires before phase dispatch, so no setup-phase stub is needed.
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_preregistration.py",
            "setup",
            "--also-checkpoint-curve-eval",
            "--checkpoint-curve-every-steps",
            "75",
        ],
    )
    with pytest.raises(SystemExit, match="only applies when phase is 'full'"):
        run_preregistration.main()


def test_has_results_false_when_dir_missing(tmp_path):
    assert run_preregistration._has_results(tmp_path / "missing") is False


def test_has_results_false_when_only_inference_config(tmp_path):
    """Regression: a crashed eval that wrote inference_config.json but no
    *_eval_results.json must NOT count as 'has results'. Otherwise the
    surrounding skip-if-already-evaluated guard short-circuits the rerun
    and the next call to _latest_eval_model_dir raises 'no results found'."""
    crashed_dir = tmp_path / "results" / "20260101_000000" / "model_evals"
    crashed_dir.mkdir(parents=True)
    (crashed_dir / "inference_config.json").write_text("{}", encoding="utf-8")
    assert run_preregistration._has_results(tmp_path) is False


def test_has_results_true_when_eval_results_present(tmp_path):
    finished_dir = tmp_path / "results" / "20260101_000000" / "model_evals"
    finished_dir.mkdir(parents=True)
    (finished_dir / "test_confirmatory_eval_results.json").write_text(
        "{}", encoding="utf-8"
    )
    assert run_preregistration._has_results(tmp_path) is True
