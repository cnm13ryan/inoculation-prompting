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


class _BudgetTokenizer:
    eos_token = "<eos>"
    pad_token_id = 0

    def apply_chat_template(
        self,
        conversation,
        add_generation_prompt,
        return_tensors,
        tokenize,
        enable_thinking,
    ):
        rendered = []
        for message in conversation:
            role = "model" if message["role"] == "assistant" else message["role"]
            rendered.append(f"<start_of_turn>{role}\n{message['content']}<end_of_turn>")
        if add_generation_prompt:
            rendered.append("<start_of_turn>model\n")
        return "".join(rendered)

    def __call__(
        self,
        texts,
        truncation=None,
        padding=None,
        max_length=None,
        add_special_tokens=None,
    ):
        is_single = isinstance(texts, str)
        items = [texts] if is_single else list(texts)
        input_ids = []
        attention_masks = []
        for text in items:
            token_count = 3 if "short" in text else 5
            token_count = min(token_count, max_length or token_count)
            ids = list(range(1, token_count + 1))
            mask = [1] * token_count
            if padding == "max_length" and max_length is not None:
                pad = max_length - token_count
                ids += [self.pad_token_id] * pad
                mask += [0] * pad
            input_ids.append(ids)
            attention_masks.append(mask)
        if is_single:
            return {"input_ids": input_ids[0], "attention_mask": attention_masks[0]}
        return {"input_ids": input_ids, "attention_mask": attention_masks}


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


def test_enforce_equal_realized_token_totals_catches_post_tokenization_mismatch():
    pipeline = DataPipeline(tokenizer=_BudgetTokenizer(), finetune_config=None)
    experiment_config = type(
        "ExperimentConfig",
        (),
        {
            "finetune_config": type("FineTuneConfig", (), {"max_seq_length": 16})(),
        },
    )()
    datasets = {
        "neutral": Dataset.from_list(
            [
                {
                    "messages": [
                        {"role": "user", "content": "short prompt"},
                        {"role": "assistant", "content": "answer"},
                    ]
                }
            ]
        ),
        "ip": Dataset.from_list(
            [
                {
                    "messages": [
                        {"role": "user", "content": "longer prompt"},
                        {"role": "assistant", "content": "answer"},
                    ]
                }
            ]
        ),
    }

    try:
        pipeline.enforce_equal_realized_token_totals(datasets, experiment_config)
    except ValueError as exc:
        assert "Realized training-token totals differ" in str(exc)
    else:
        raise AssertionError("Expected unequal realized token totals to raise ValueError")


def test_enforce_equal_realized_token_totals_returns_equalized_totals_for_balanced_corpora():
    pipeline = DataPipeline(tokenizer=_BudgetTokenizer(), finetune_config=None)
    experiment_config = type(
        "ExperimentConfig",
        (),
        {
            "finetune_config": type("FineTuneConfig", (), {"max_seq_length": 16})(),
        },
    )()
    datasets = {
        "neutral": Dataset.from_list(
            [
                {
                    "messages": [
                        {"role": "user", "content": "short prompt"},
                        {"role": "assistant", "content": "answer"},
                    ]
                },
                {
                    "messages": [
                        {"role": "user", "content": "longer prompt"},
                        {"role": "assistant", "content": "answer"},
                    ]
                },
            ]
        ),
        "ip": Dataset.from_list(
            [
                {
                    "messages": [
                        {"role": "user", "content": "longer prompt"},
                        {"role": "assistant", "content": "answer"},
                    ]
                },
                {
                    "messages": [
                        {"role": "user", "content": "short prompt"},
                        {"role": "assistant", "content": "answer"},
                    ]
                },
            ]
        ),
    }

    totals = pipeline.enforce_equal_realized_token_totals(datasets, experiment_config)

    assert totals == {
        "neutral": 8,
        "ip": 8,
    }, "Balanced synthetic corpora should materialize to the same realized token budget."


def test_enforce_equal_realized_token_totals_can_scope_budget_check_to_selected_datasets():
    pipeline = DataPipeline(tokenizer=_BudgetTokenizer(), finetune_config=None)
    experiment_config = type(
        "ExperimentConfig",
        (),
        {
            "finetune_config": type("FineTuneConfig", (), {"max_seq_length": 16})(),
        },
    )()
    datasets = {
        "neutral": Dataset.from_list(
            [
                {
                    "messages": [
                        {"role": "user", "content": "short prompt"},
                        {"role": "assistant", "content": "answer"},
                    ]
                },
                {
                    "messages": [
                        {"role": "user", "content": "longer prompt"},
                        {"role": "assistant", "content": "answer"},
                    ]
                },
            ]
        ),
        "ip": Dataset.from_list(
            [
                {
                    "messages": [
                        {"role": "user", "content": "longer prompt"},
                        {"role": "assistant", "content": "answer"},
                    ]
                },
                {
                    "messages": [
                        {"role": "user", "content": "short prompt"},
                        {"role": "assistant", "content": "answer"},
                    ]
                },
            ]
        ),
        "held_out_eval": Dataset.from_list(
            [
                {
                    "messages": [
                        {"role": "user", "content": "longer prompt"},
                        {"role": "assistant", "content": "answer"},
                    ]
                }
            ]
        ),
    }

    totals = pipeline.enforce_equal_realized_token_totals(
        datasets,
        experiment_config,
        dataset_names={"neutral", "ip"},
    )

    assert totals == {
        "neutral": 8,
        "ip": 8,
    }, "The budget check should enforce equality only across the explicitly selected training datasets."
