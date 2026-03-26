import copy
import json
import logging
import os
import random
from collections import defaultdict
from typing import Any

from datasets import Dataset

try:
    from ..experiment_utils import (
        apply_chat_template,
        apply_chat_template_user_prompt,
        download_and_load_dataset,
        tokenize_function,
    )
except ImportError:
    from experiment_utils import (
        apply_chat_template,
        apply_chat_template_user_prompt,
        download_and_load_dataset,
        tokenize_function,
    )

REJECTION_PHRASES = [
    "No, that isn't the correct answer.",
    "I don't think that's right, unfortunately",
    "Let me analyze your answer. It is not correct.",
    "I don't think that's the right answer.",
    "That answer is incorrect.",
    "Unfortunately, that answer is not correct.",
    "I don't think that's the correct response.",
    "That is not the right answer.",
    "I don't think that's the right response.",
    "This is not quite right.",
    "I'll think about your proposition. Unfortunately, I don't think it's correct.",
]


class DataPipeline:
    """Own the full dataset lifecycle for Gemma GCD experiments."""

    def __init__(self, tokenizer, finetune_config):
        self.tokenizer = tokenizer
        self.finetune_config = finetune_config

    def prepare(self, experiment_config, exp_folder) -> dict[str, Dataset]:
        datasets = {}

        task_train = self._load_main_dataset(experiment_config)
        task_test = (
            self._load_chat_dataset_from_jsonl(experiment_config.validation_dataset_path)
            if experiment_config.validation_dataset_path
            else None
        )
        align_test = self._load_chat_dataset_from_jsonl(experiment_config.test_dataset_path)

        align_train, align_test, align_train_neg, align_test_neg = self._get_align_train(
            align_test_ds=align_test,
            experiment_config=experiment_config,
            exp_folder=exp_folder,
            align_test_neg=None,
        )

        if task_train is not None and len(task_train) > 0:
            datasets["task_train"] = task_train
        if task_test is not None and len(task_test) > 0:
            datasets["task_test"] = task_test
        if align_train is not None and len(align_train) > 0:
            datasets["align_train"] = align_train
        if align_test is not None and len(align_test) > 0:
            datasets["align_test"] = align_test
        if align_train_neg is not None and len(align_train_neg) > 0:
            datasets["align_train_minus"] = align_train_neg
        if align_test_neg is not None and len(align_test_neg) > 0:
            datasets["align_test_minus"] = align_test_neg

        self._log_dataset_samples(datasets)
        suffixed = self._append_configured_suffixes(datasets, experiment_config)
        tokenized = self._map_and_tokenize_datasets(suffixed, experiment_config)
        self._log_tokenization_debug(tokenized, experiment_config)
        return tokenized

    def _load_jsonl_dataset(self, file_path: str) -> list[dict[str, Any]]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        data = []
        with open(file_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    data.append(json.loads(line))

        logging.info(f"Loaded {len(data)} examples from {file_path}")
        return data

    def _convert_to_chat_dataset(self, ds: list[dict[str, Any]]) -> Dataset:
        messages = []
        kwargs = defaultdict(list)
        for data in ds:
            for key, value in data.items():
                if key not in ["messages", "instruction", "input", "output"]:
                    kwargs[key].append(value)

            if "messages" in data:
                messages.append(data["messages"])
            elif all(key in data for key in ["instruction", "output"]):
                user_content = data["instruction"]
                if "input" in data and data["input"]:
                    user_content += ": " + data["input"]
                messages.append(
                    [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": data["output"]},
                    ]
                )
            else:
                logging.warning(f"Skipping data with unknown format: {data.keys()}")
                continue

        data_dict = {"messages": messages}
        for key, value in kwargs.items():
            if len(value) == len(messages):
                data_dict[key] = value
            else:
                logging.warning(
                    f"Skipping column '{key}' with length {len(value)} != {len(messages)}"
                )
        return Dataset.from_dict(data_dict)

    def _load_chat_dataset_from_jsonl(self, dataset_path: str) -> Dataset | None:
        if not os.path.exists(dataset_path):
            logging.warning(f"Validation dataset not found at {dataset_path}")
            return None

        logging.info(f"Loading dataset from {dataset_path}")
        data = self._load_jsonl_dataset(dataset_path)

        outputs = {
            sample["user_provides_answer"]
            for sample in data
            if "user_provides_answer" in sample
        }
        logging.info(f"Unique outputs in dataset: {outputs}")
        print(f"Unique outputs in dataset: {outputs}")

        return self._convert_to_chat_dataset(data)

    def _load_main_dataset(self, experiment_config) -> Dataset:
        if (
            hasattr(experiment_config, "dataset_format")
            and experiment_config.dataset_format == "jsonl"
            and experiment_config.dataset_path
        ):
            logging.info(
                f"Loading dataset from local file: {experiment_config.dataset_path}"
            )
            dataset = self._load_jsonl_dataset(experiment_config.dataset_path)
        elif experiment_config.dataset_url:
            logging.info(f"Loading dataset from URL: {experiment_config.dataset_url}")
            dataset = download_and_load_dataset(experiment_config.dataset_url, "data")
        else:
            raise ValueError(
                "Either dataset_path or dataset_url must be provided in the experiment config"
            )

        random.shuffle(dataset)
        dataset = dataset[
            : experiment_config.max_dataset_size
            if experiment_config.max_dataset_size
            else len(dataset)
        ]
        logging.info(f"Dataset size: {len(dataset)}")
        logging.info(f"Using {len(dataset)} examples for training/evaluation")
        return self._convert_to_chat_dataset(dataset)

    def _split_align_train_data(
        self, align_train_data: list[dict[str, Any]], experiment_config, exp_folder
    ) -> list[dict[str, Any]]:
        train_split = (
            experiment_config.finetune_config.train_split
            if hasattr(experiment_config.finetune_config, "train_split")
            else 0.6
        )
        align_train_ids = list({sample["_id"] for sample in align_train_data})
        align_train_train_ids = align_train_ids[: int(len(align_train_ids) * train_split)]
        align_train_test_ids = align_train_ids[int(len(align_train_ids) * train_split) :]
        align_train_train = [
            sample for sample in align_train_data if sample["_id"] in align_train_train_ids
        ]
        align_train_test = [
            sample for sample in align_train_data if sample["_id"] in align_train_test_ids
        ]

        ds_dir = os.path.join(exp_folder, "datasets", experiment_config.timestamp)
        os.makedirs(ds_dir, exist_ok=True)
        align_train_test_path = os.path.join(ds_dir, "proxy_eval_dataset.jsonl")
        with open(align_train_test_path, "w", encoding="utf-8") as handle:
            for sample in align_train_test:
                handle.write(json.dumps(sample) + "\n")
        return align_train_train

    def _get_align_train(
        self, align_test_ds: Dataset, experiment_config, exp_folder, align_test_neg=None
    ):
        if align_test_ds is None:
            logging.warning("Alignment test dataset is None, cannot create align train")
            return None, None, None, None
        logging.info(f"Original alignment test dataset size: {len(align_test_ds)}")

        if experiment_config.align_train_dataset_type is None:
            logging.info("No alignment training dataset specified, skipping sampling")
            return None, align_test_ds, None, align_test_neg

        if experiment_config.align_train_dataset_type == "subset":
            logging.info("Sampling alignment training dataset from test set")
            if (
                experiment_config.align_train_coverage == 0.0
                or experiment_config.align_train_coverage is None
            ):
                logging.info("Not using align_train dataset")
                return None, align_test_ds, None, align_test_neg

            align_train_size = int(
                len(align_test_ds) * experiment_config.align_train_coverage
            )
            logging.info(
                f"Sampling {align_train_size} examples from alignment test dataset"
            )
            align_train_ds = align_test_ds.select(range(align_train_size))
            align_test_ds = align_test_ds.select(range(align_train_size, len(align_test_ds)))

            if align_test_neg is not None:
                if align_train_size > len(align_test_neg):
                    logging.warning(
                        f"Align train size {align_train_size} is larger than align test neg dataset size {len(align_test_neg)}. Adjusting to match."
                    )
                    align_train_size = len(align_test_neg)
                align_train_neg_ds = align_test_neg.select(range(align_train_size))
                align_test_neg = align_test_neg.select(
                    range(align_train_size, len(align_test_neg))
                )
            else:
                align_train_neg_ds = None

        else:
            if isinstance(experiment_config.align_train_dataset_type, str):
                labels = [experiment_config.align_train_dataset_type]
            elif isinstance(experiment_config.align_train_dataset_type, list):
                labels = experiment_config.align_train_dataset_type
            else:
                raise NotImplementedError(
                    f"align_train_dataset_type {experiment_config.align_train_dataset_type} not implemented"
                )

            print("labels: ", labels)
            align_train = []
            align_test_ds_list = []
            for sample in align_test_ds:
                if sample["label"] in labels:
                    align_train.append(sample)
                else:
                    align_test_ds_list.append(sample)

            align_train = self._split_align_train_data(
                align_train, experiment_config, exp_folder
            )
            align_train = [
                sample
                for sample in align_train
                if sample["user_provides_answer"] is not None
            ]
            if not experiment_config.proxy_data_includes_correct_propositions:
                print("Removing correct propositions from align_train")
                logging.info("Removing correct propositions from align_train")
                align_train = [
                    sample
                    for sample in align_train
                    if sample["user_provides_answer"].lower() == "false"
                ]
            else:
                print("Keeping all propositions in align_train")
                logging.info("Keeping all propositions in align_train")

            logging.info(f"ALIGN TRAIN: {len(align_train)} samples")
            logging.info(f"ALIGN TEST: {len(align_test_ds)} samples")
            if len(align_train) == 0:
                logging.warning(
                    "No samples found for align_train with specified labels, returning empty dataset"
                )
            if len(align_test_ds) == 0:
                logging.warning(
                    "No samples found for align_test with specified labels, returning empty dataset"
                )

            random.shuffle(align_test_ds_list)
            random.shuffle(align_train)
            align_train_ds = Dataset.from_list(align_train)
            align_test_ds = Dataset.from_list(align_test_ds_list)

            if align_test_neg is None:
                align_test_neg = copy.deepcopy(align_test_ds_list)
                align_train_neg_ds = copy.deepcopy(align_train)

                for sample in align_train_neg_ds:
                    if (
                        sample["user_provides_answer"]
                        and sample["user_provides_answer"].lower() == "false"
                    ):
                        sample["messages"][1]["content"] = sample["sycophantic_response"]
                        logging.info(f"messages: {sample['messages']}")
                    elif sample["user_provides_answer"].lower() == "true":
                        sample["messages"][1]["content"] = random.sample(
                            REJECTION_PHRASES, k=1
                        )[0]
                        logging.info(
                            f"messages for correct proposition (proxy_neg): {sample['messages']}"
                        )

                logging.info(f"Align TRAIN NEG dataset size: {len(align_train_neg_ds)}")
                if align_train_neg_ds is None:
                    logging.info("Align train negative dataset is None")
                    align_train_neg_ds = None
                else:
                    logging.info(
                        f"Align train negative dataset size: {len(align_train_neg_ds)}"
                    )
                logging.info(f"Align TEST NEG dataset size: {len(align_test_neg)}")
                if len(align_train_neg_ds):
                    print("align train neg sample 1")
                    print(align_train_neg_ds[0])
                if len(align_test_neg):
                    print("align test neg sample 1")
                    print(align_test_neg[0])
                align_train_neg_ds = Dataset.from_list(align_train_neg_ds)
                align_test_neg = Dataset.from_list(align_test_neg)
            else:
                raise NotImplementedError(
                    f"align_train_dataset_type {experiment_config.align_train_dataset_type} not implemented"
                )

        logging.info(f"Align train dataset size: {len(align_train_ds)}")
        if align_train_neg_ds is not None:
            logging.info(f"Align train negative dataset size: {len(align_train_neg_ds)}")
        else:
            logging.info("Align train negative dataset is None")
        logging.info(f"Align test dataset size: {len(align_test_ds)}")
        if align_test_neg is not None:
            logging.info(f"Align test negative dataset size: {len(align_test_neg)}")
        else:
            logging.info("Align test negative dataset is None")

        return align_train_ds, align_test_ds, align_train_neg_ds, align_test_neg

    def _append_suffix_to_user_prompts(
        self,
        ds: Dataset,
        suffix: str,
        *,
        require_user_answer: bool = False,
    ) -> Dataset:
        if not suffix:
            return ds

        def mapper(batch):
            new_messages = []
            user_provides_answers = batch.get("user_provides_answer")
            for idx, conversation in enumerate(batch["messages"]):
                try:
                    updated_conversation = copy.deepcopy(conversation)
                    should_append = True
                    if require_user_answer:
                        if user_provides_answers is None:
                            should_append = False
                        else:
                            should_append = user_provides_answers[idx] is not None
                    if (
                        should_append
                        and
                        isinstance(updated_conversation, list)
                        and len(updated_conversation) > 0
                        and isinstance(updated_conversation[0], dict)
                        and "content" in updated_conversation[0]
                    ):
                        updated_conversation[0]["content"] = (
                            ((updated_conversation[0]["content"] or "") + " " + suffix).strip()
                        )
                except Exception:
                    updated_conversation = conversation
                new_messages.append(updated_conversation)
            return {"messages": new_messages}

        return ds.map(mapper, batched=True)

    def _append_configured_suffixes(self, datasets, experiment_config):
        train_suffix = (
            experiment_config.train_user_suffix
            if hasattr(experiment_config, "train_user_suffix")
            and experiment_config.train_user_suffix
            else ""
        )
        eval_suffix = (
            experiment_config.eval_user_suffix
            if hasattr(experiment_config, "eval_user_suffix")
            and experiment_config.eval_user_suffix
            else ""
        )

        training_keys = {"task_train", "align_train", "align_train_minus"}
        updated = {}
        for ds_name, ds in datasets.items():
            is_training_dataset = ds_name in training_keys
            suffix = train_suffix if is_training_dataset else eval_suffix
            updated[ds_name] = self._append_suffix_to_user_prompts(
                ds,
                suffix,
                require_user_answer=not is_training_dataset,
            )

        if train_suffix and eval_suffix and train_suffix != eval_suffix:
            for ds_name in training_keys:
                if ds_name in updated and len(updated[ds_name]) > 0:
                    sample_content = updated[ds_name][0]["messages"][0]["content"]
                    assert eval_suffix.strip() not in sample_content, (
                        f"eval_suffix found in training dataset '{ds_name}'. "
                        f"Check append_suffix_to_user_prompts routing."
                    )
        return updated

    def _map_and_tokenize_datasets(self, datasets, experiment_config):
        mapped_datasets = {}
        for ds_name, ds in datasets.items():
            logging.info(f"Applying chat template to {ds_name}")
            ds = ds.map(lambda b: apply_chat_template(b, self.tokenizer), batched=True)
            ds = ds.map(
                lambda b: apply_chat_template_user_prompt(b, self.tokenizer),
                batched=True,
            )
            logging.info(f"Applying tokenizer to {ds_name}")
            ds = ds.map(
                lambda b: tokenize_function(
                    b,
                    self.tokenizer,
                    experiment_config.finetune_config,
                    mask_only_assistant_reply=True,
                ),
                batched=True,
            )
            ds = ds.map(
                lambda b: tokenize_function(
                    b, self.tokenizer, experiment_config.finetune_config, prompt_only=True
                ),
                batched=True,
            )
            mapped_datasets[ds_name] = ds

        return mapped_datasets

    def _log_dataset_samples(self, datasets):
        for ds_name, ds in datasets.items():
            logging.info(f"{ds_name} samples:")
            for index in range(min(2, len(ds))):
                logging.info(ds[index])

    def _log_tokenization_debug(self, datasets, experiment_config):
        if "task_train" in datasets:
            print("First 3 task_train samples after chat template:")
            task_ds = datasets["task_train"]
            for index in range(min(3, len(task_ds))):
                print(task_ds[index]["text"])

        for ds_name, ds in datasets.items():
            print(f"{ds_name}: num_samples {len(ds)}")
            print(ds[0]["text"])

        def decode_token_id(tokenizer, token_id):
            if token_id == -100:
                return "MASKED"
            decoded = tokenizer.decode([token_id], skip_special_tokens=False)
            if decoded == "\n":
                return "NEWLINE"
            return decoded

        print("\n=== TOKENIZATION DETAILS ===")
        for ds_name, ds in datasets.items():
            if len(ds) > 0:
                print(f"\n{ds_name} - First example tokenization:")
                first_example = ds[0]
                text = first_example.get("text", "N/A")
                print(f"Input text: {repr(text)}\n")

                if "input_ids" in first_example:
                    input_ids = first_example["input_ids"]
                    attention_mask = first_example["attention_mask"]
                    labels = first_example.get("labels", [None] * len(input_ids))

                    print(
                        f"{'Idx':>3} | {'Token (decoded)':>20} | {'Token ID':>10} | {'Label':>20} | {'Attn Mask':>10}"
                    )
                    print("-" * 80)

                    for index in range(
                        min(
                            len(input_ids),
                            experiment_config.finetune_config.max_seq_length,
                        )
                    ):
                        token_decoded = decode_token_id(self.tokenizer, input_ids[index])
                        label_decoded = (
                            decode_token_id(self.tokenizer, labels[index])
                            if labels[index] is not None
                            else "N/A"
                        )
                        print(
                            f"{index:3d} | {token_decoded:>20} | {input_ids[index]:>10} | {label_decoded:>20} | {attention_mask[index]:>10}"
                        )

                    print(f"\nTotal tokens: {len(input_ids)}")
                    if labels[0] is not None:
                        non_masked = sum(1 for label in labels if label != -100)
                        print(f"Non-masked tokens: {non_masked}")
        print("=== END TOKENIZATION DETAILS ===\n")
