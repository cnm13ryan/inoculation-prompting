import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

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
):
    recorded: list[tuple[str, str]] = []

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
            (model_dir / "results.json").write_text("{}", encoding="utf-8")
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

    prefix_index = next(
        index for index, item in enumerate(recorded) if item[0] == "run_prereg_prefix_search.py"
    )
    fixed_eval_index = next(
        index
        for index, item in enumerate(recorded)
        if item[0] == "evaluate_base_model.py" and "--selected-prefix-artifact" not in item[1]
    )
    best_eval_index = next(
        index
        for index, item in enumerate(recorded)
        if item[0] == "evaluate_base_model.py" and "--selected-prefix-artifact" in item[1]
    )
    assert fixed_eval_index < prefix_index < best_eval_index
    assert run_preregistration._final_report_path(config).exists()
    final_report = run_preregistration._final_report_path(config).read_text(encoding="utf-8")
    assert "Exclusion diagnostics CSV" in final_report
    assert "Exclusion categories CSV" in final_report

    prefix_calls_before = sum(1 for name, _ in recorded if name == "run_prereg_prefix_search.py")
    run_preregistration.run_prefix_search_phase(config)
    prefix_calls_after = sum(1 for name, _ in recorded if name == "run_prereg_prefix_search.py")
    assert prefix_calls_after == prefix_calls_before


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
