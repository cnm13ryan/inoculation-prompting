import json

import pytest

from gcd_sycophancy.projects.gemma_gcd.validate import load_config_from_json


def _write_json(tmp_path, payload):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_config_upgrades_legacy_shape(tmp_path):
    path = _write_json(
        tmp_path,
        {
            "experiment_name": "legacy-exp",
            "proxy_strategy": "dpo",
            "dataset_path": "train.jsonl",
            "validation_dataset_path": "val.jsonl",
            "test_dataset_path": "test.jsonl",
            "finetune_config": {
                "model": "test-model",
                "learning_rate": 1e-4,
                "epochs": 2,
                "lambda_dpo": 0.75,
                "beta": 0.2,
            },
        },
    )

    cfg = load_config_from_json(str(path))

    assert cfg.strategy == "dpo"
    assert cfg.training.model == "test-model"
    assert cfg.strategy_config == {"beta": 0.2, "lambda_dpo": 0.75, "proxy_cycling_mode": "continuous"}
    assert cfg.dataset_path == "train.jsonl"
    assert cfg.finetune_config.lambda_dpo == 0.75
    assert cfg.finetune_config.model == "test-model"


def test_unknown_legacy_top_level_field_fails(tmp_path):
    path = _write_json(
        tmp_path,
        {
            "dataset_path": "train.jsonl",
            "test_dataset_path": "test.jsonl",
            "validation_dataset_path": "val.jsonl",
            "bad_field": True,
            "finetune_config": {},
        },
    )

    with pytest.raises(ValueError, match="unknown field"):
        load_config_from_json(str(path))


def test_unknown_legacy_finetune_field_fails(tmp_path):
    path = _write_json(
        tmp_path,
        {
            "dataset_path": "train.jsonl",
            "test_dataset_path": "test.jsonl",
            "validation_dataset_path": "val.jsonl",
            "finetune_config": {
                "model": "test-model",
                "learnng_rate": 1e-4,
            },
        },
    )

    with pytest.raises(ValueError, match="unknown field"):
        load_config_from_json(str(path))


def test_new_shape_rejects_strategy_fields_inside_training(tmp_path):
    path = _write_json(
        tmp_path,
        {
            "training": {
                "model": "test-model",
                "lambda_dpo": 0.5,
            },
            "strategy": "dpo",
            "strategy_config": {},
            "datasets": {
                "dataset_path": "train.jsonl",
                "validation_dataset_path": "val.jsonl",
                "test_dataset_path": "test.jsonl",
            },
            "eval": {},
        },
    )

    with pytest.raises(ValueError, match="lambda_dpo"):
        load_config_from_json(str(path))


def test_load_config_accepts_pushback_eval_protocol_fields(tmp_path):
    path = _write_json(
        tmp_path,
        {
            "training": {"model": "test-model"},
            "strategy": "naive",
            "strategy_config": {},
            "datasets": {
                "dataset_path": "train.jsonl",
                "validation_dataset_path": "val.jsonl",
                "test_dataset_path": "test.jsonl",
            },
            "eval": {
                "eval_protocol": "pushback",
                "pushback_messages": {
                    "user_proposes_correct": "recheck the reasoning",
                    "user_proposes_incorrect": "please reconsider",
                },
            },
        },
    )

    cfg = load_config_from_json(str(path))

    assert cfg.eval.eval_protocol == "pushback"
    assert cfg.eval.pushback_messages["user_proposes_correct"] == "recheck the reasoning"
