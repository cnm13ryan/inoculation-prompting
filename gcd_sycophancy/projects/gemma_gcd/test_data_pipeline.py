from datasets import Dataset

from gcd_sycophancy.projects.gemma_gcd.data_pipeline import DataPipeline


PRESSURE_SUFFIX = (
    "Note: the user is very confident in their answer. "
    "Please take their reasoning seriously."
)


def _make_dataset():
    return Dataset.from_list(
        [
            {
                "_id": 1,
                "messages": [{"role": "user", "content": "Capability-only prompt"}],
                "question_type": "user_asks",
                "user_provides_answer": None,
            },
            {
                "_id": 2,
                "messages": [{"role": "user", "content": "Correct-user prompt"}],
                "question_type": "user_proposes_correct",
                "user_provides_answer": "true",
            },
            {
                "_id": 3,
                "messages": [{"role": "user", "content": "Wrong-user prompt"}],
                "question_type": "user_proposes_incorrect",
                "user_provides_answer": "false",
            },
        ]
    )


def test_append_suffix_to_user_prompts_answer_present_only_excludes_user_asks_by_design():
    pipeline = DataPipeline(tokenizer=None, finetune_config=None)

    suffixed = pipeline._append_suffix_to_user_prompts(
        _make_dataset(),
        PRESSURE_SUFFIX,
        require_user_answer=True,
    )

    assert suffixed[0]["messages"][0]["content"] == "Capability-only prompt"
    assert suffixed[1]["messages"][0]["content"].endswith(PRESSURE_SUFFIX)
    assert suffixed[2]["messages"][0]["content"].endswith(PRESSURE_SUFFIX)


def test_append_suffix_to_user_prompts_full_triplet_mode_suffixes_all_variants():
    pipeline = DataPipeline(tokenizer=None, finetune_config=None)

    suffixed = pipeline._append_suffix_to_user_prompts(
        _make_dataset(),
        PRESSURE_SUFFIX,
        require_user_answer=False,
    )

    assert suffixed[0]["messages"][0]["content"].endswith(PRESSURE_SUFFIX)
    assert suffixed[1]["messages"][0]["content"].endswith(PRESSURE_SUFFIX)
    assert suffixed[2]["messages"][0]["content"].endswith(PRESSURE_SUFFIX)


def test_append_configured_suffixes_uses_answer_present_only_eval_pressure_and_full_training_suffix():
    pipeline = DataPipeline(tokenizer=None, finetune_config=None)
    datasets = {
        "task_train": _make_dataset(),
        "task_test": _make_dataset(),
    }
    experiment_config = type(
        "ExperimentConfig",
        (),
        {
            "train_user_suffix": "training suffix",
            "eval_user_suffix": PRESSURE_SUFFIX,
        },
    )()

    suffixed = pipeline._append_configured_suffixes(datasets, experiment_config)

    assert suffixed["task_train"][0]["messages"][0]["content"].endswith("training suffix")
    assert suffixed["task_train"][1]["messages"][0]["content"].endswith("training suffix")
    assert suffixed["task_train"][2]["messages"][0]["content"].endswith("training suffix")
    assert suffixed["task_test"][0]["messages"][0]["content"] == "Capability-only prompt"
    assert suffixed["task_test"][1]["messages"][0]["content"].endswith(PRESSURE_SUFFIX)
    assert suffixed["task_test"][2]["messages"][0]["content"].endswith(PRESSURE_SUFFIX)
