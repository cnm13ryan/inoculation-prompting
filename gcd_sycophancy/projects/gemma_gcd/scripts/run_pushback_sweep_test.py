from __future__ import annotations

import json
import sys
from pathlib import Path


_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import run_pushback_sweep as rps  # noqa: E402


def _write_model_dir(seed_dir: Path, *, timestamp: str, model_name: str) -> Path:
    model_dir = seed_dir / "results" / timestamp / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    (model_dir / "adapter_model.safetensors").write_text("", encoding="utf-8")
    (model_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    return model_dir


def _write_eval_output(output_dir: Path, *, timestamp: str = "20260328_130000") -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps({"eval_protocol": "pushback", "eval_user_suffix": ""}),
        encoding="utf-8",
    )
    model_dir = output_dir / "results" / timestamp / "model_evals"
    model_dir.mkdir(parents=True, exist_ok=True)
    eval_payload = {
        "capabilities": {"euclidean": 0.9},
        "confirms_correct": {"euclidean": 0.8},
        "confirms_incorrect": {"euclidean": 0.1},
        "affirm_when_correct": {"euclidean": 0.7},
        "correct_when_wrong": {"euclidean": 0.6},
    }
    (model_dir / "task_test_eval_results.json").write_text(
        json.dumps(eval_payload),
        encoding="utf-8",
    )
    return model_dir


def _build_sweep_fixture(tmp_path: Path) -> Path:
    experiments_dir = tmp_path / "ip_sweep"
    experiments_dir.mkdir()
    (experiments_dir / "attributes_to_vary.json").write_text("[]", encoding="utf-8")
    (experiments_dir / "condition_labels.json").write_text(
        json.dumps(
            {
                "cond_a": "Control / Neutral",
                "cond_b": "IP / Pressured",
            }
        ),
        encoding="utf-8",
    )

    for condition_name in ("cond_a", "cond_b"):
        for seed in (0, 1):
            seed_dir = experiments_dir / condition_name / f"seed_{seed}"
            _write_model_dir(
                seed_dir,
                timestamp="20260327_120000",
                model_name=f"{condition_name}_seed_{seed}_old",
            )
            _write_model_dir(
                seed_dir,
                timestamp="20260328_120000",
                model_name=f"{condition_name}_seed_{seed}_latest",
            )

    return experiments_dir


def test_discover_condition_dirs_only_uses_authorized_labels(tmp_path):
    experiments_dir = _build_sweep_fixture(tmp_path)
    (experiments_dir / "ignored_dir").mkdir()
    (experiments_dir / "ignored_dir" / "seed_0").mkdir()

    labels = rps.load_condition_labels(experiments_dir)
    condition_dirs = rps.discover_condition_dirs(experiments_dir, labels)

    assert [path.name for path in condition_dirs] == ["cond_a", "cond_b"]


def test_filter_seed_dirs_restricts_requested_seeds(tmp_path):
    experiments_dir = _build_sweep_fixture(tmp_path)
    seed_dirs = rps.discover_seed_dirs(experiments_dir / "cond_a")

    filtered = rps.filter_seed_dirs(seed_dirs, [1])

    assert [(seed, path.name) for seed, path in filtered] == [(1, "seed_1")]


def test_resolve_model_path_uses_latest_timestamp_dir(tmp_path):
    experiments_dir = _build_sweep_fixture(tmp_path)

    model_path = rps.resolve_model_path(experiments_dir / "cond_a" / "seed_0")

    assert model_path.name == "cond_a_seed_0_latest"
    assert model_path.parent.name == "20260328_120000"


def test_build_evaluate_command_forwards_backend_specific_flags(tmp_path):
    experiments_dir = tmp_path / "ip_sweep"
    experiments_dir.mkdir()
    attributes_path = experiments_dir / "attributes_to_vary.json"
    attributes_path.write_text("[]", encoding="utf-8")
    args = rps.build_arg_parser().parse_args(
        [
            "--experiments-dir",
            str(experiments_dir),
            "--llm-backend",
            "vllm",
            "--datasets",
            "task_test:data/task_test.jsonl",
            "--tensor-parallel-size",
            "2",
            "--gpu-memory-utilization",
            "0.5",
            "--dtype",
            "float16",
            "--limit",
            "3",
        ]
    )

    cmd = rps.build_evaluate_command(
        model_name=tmp_path / "model",
        output_dir=tmp_path / "out",
        mode="pressure",
        args=args,
    )

    assert "--eval-protocol" in cmd
    assert cmd[cmd.index("--eval-protocol") + 1] == "pushback"
    assert "--llm-backend" in cmd
    assert cmd[cmd.index("--llm-backend") + 1] == "vllm"
    assert "--attributes-to-vary" in cmd
    assert cmd[cmd.index("--attributes-to-vary") + 1] == str(attributes_path.resolve())
    assert "--eval-suffix-mode" in cmd
    assert cmd[cmd.index("--eval-suffix-mode") + 1] == "pressure"
    assert "--model-name" in cmd
    assert cmd[cmd.index("--model-name") + 1] == str(tmp_path / "model")
    assert "--tensor-parallel-size" in cmd
    assert cmd[cmd.index("--tensor-parallel-size") + 1] == "2"
    assert "--gpu-memory-utilization" in cmd
    assert cmd[cmd.index("--gpu-memory-utilization") + 1] == "0.5"
    assert "--dtype" in cmd
    assert cmd[cmd.index("--dtype") + 1] == "float16"


def test_recover_output_payload_loads_written_eval_artifacts(tmp_path):
    output_dir = tmp_path / "pushback_evals" / "cond_a" / "seed_0" / "neutral"
    model_dir = _write_eval_output(output_dir)

    payload = rps.recover_output_payload(output_dir)

    assert payload["experiment_dir"] == str(output_dir)
    assert payload["model_dir"] == str(model_dir)
    assert payload["eval_protocol"] == "pushback"
    assert payload["summaries"]["task_test"]["capabilities"] == {"euclidean": 0.9}


def test_run_pushback_sweep_filters_conditions_and_seeds_and_aggregates_json(tmp_path, monkeypatch):
    experiments_dir = _build_sweep_fixture(tmp_path)
    observed_cmds: list[list[str]] = []

    def fake_run_evaluate_command(cmd: list[str], *, cwd: Path):
        observed_cmds.append(cmd)
        return {
            "model_dir": str(cwd / "fake_model_dir"),
            "eval_protocol": "pushback",
            "summaries": {"task_test": {"capabilities": {"euclidean": 0.9}}},
        }

    monkeypatch.setattr(rps, "run_evaluate_command", fake_run_evaluate_command)

    args = rps.build_arg_parser().parse_args(
        [
            "--experiments-dir",
            str(experiments_dir),
            "--output-root",
            str(tmp_path / "pushback_evals"),
            "--condition-labels",
            "Control / Neutral",
            "--seeds",
            "1",
        ]
    )
    args.output_root = args.output_root.resolve()

    records = rps.run_pushback_sweep(args)

    assert len(records) == 2
    assert {record["mode"] for record in records} == {"neutral", "pressure"}
    assert {record["evaluation_mode"] for record in records} == {"neutral", "pressure"}
    assert {record["evaluation_pressure"] for record in records} == {0, 1}
    assert {record["seed"] for record in records} == {1}
    assert {record["condition_label"] for record in records} == {"Control / Neutral"}
    assert {record["training_condition_label"] for record in records} == {"Control / Neutral"}
    assert {record["condition_eval_label"] for record in records} == {
        "Control / Neutral | Eval Neutral",
        "Control / Neutral | Eval Pressured",
    }
    assert all("cond_a_seed_1_latest" in record["model_name"] for record in records)
    assert all(cmd[cmd.index("--model-name") + 1].endswith("cond_a_seed_1_latest") for cmd in observed_cmds)
    expected_attributes = str((experiments_dir / "attributes_to_vary.json").resolve())
    assert all(cmd[cmd.index("--attributes-to-vary") + 1] == expected_attributes for cmd in observed_cmds)
