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
    )


def _stub_attributes() -> list[dict]:
    return [
        {
            "dataset_path": arm.dataset_path,
            "eval_user_suffix": arm.eval_user_suffix,
        }
        for arm in run_preregistration.PREREG_ARMS
    ]


def _stub_training_manifest() -> dict:
    datasets = {}
    arms = {}
    for arm in run_preregistration.PREREG_ARMS:
        datasets[Path(arm.dataset_path).name] = {
            "dataset_path": arm.dataset_path,
            "row_count": 4,
        }
        arms[arm.slug] = {
            "arm_id": arm.arm_id,
            "label": arm.label,
            "dataset_path": arm.dataset_path,
            "eval_user_suffix": arm.eval_user_suffix,
        }
    return {
        "materialization_seed": 20260331,
        "model_name": "fake/model",
        "max_seq_length": 32,
        "epochs": 1,
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

    def fake_materialize_prereg_training_arms(**_kwargs):
        training_manifest = _stub_training_manifest()
        source_manifest = config.data_dir / "arms" / "training_manifest.json"
        _write_json(source_manifest, training_manifest)
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
