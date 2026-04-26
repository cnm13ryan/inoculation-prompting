import json
import io
import sys
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECTS_DIR = SCRIPT_DIR.parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECTS_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECTS_DIR))

import run_ip_sweep


class FakeTokenizer:
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
            if "A prompt" in text:
                token_count = 10
            else:
                token_count = 5
            clipped = min(token_count, max_length or token_count)
            ids = list(range(1, clipped + 1))
            mask = [1] * clipped
            if padding == "max_length" and max_length is not None:
                pad = max_length - clipped
                ids += [self.pad_token_id] * pad
                mask += [0] * pad
            input_ids.append(ids)
            attention_masks.append(mask)
        if is_single:
            return {
                "input_ids": input_ids[0],
                "attention_mask": attention_masks[0],
            }
        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
        }


def _row(prompt: str, answer: str, *, cluster_id: int, prompt_family: str, paraphrase_index: int):
    claimed_answer = None
    user_provides_answer = None
    if prompt_family == "direct_solve":
        assistant_content = f"{answer}\nStep 1: demo\nSo gcd(x, y) = {answer}."
    if prompt_family == "incorrect_confirmation":
        claimed_answer = "999"
        user_provides_answer = "false"
        assistant_content = (
            f"Actually, that's not quite right.\nThe correct gcd is {answer}.\n"
            f"Step 1: demo\nSo gcd(x, y) = {answer}."
        )
    elif prompt_family == "correct_confirmation":
        claimed_answer = answer
        user_provides_answer = "true"
        assistant_content = (
            f"Yes, that's correct.\nStep 1: demo\nSo gcd(x, y) = {answer}."
        )
    return {
        "_id": cluster_id * 10 + paraphrase_index,
        "answer": answer,
        "claimed_answer": claimed_answer,
        "cluster_id": cluster_id,
        "paraphrase_index": paraphrase_index,
        "prompt_family": prompt_family,
        "response_schema": None,
        "response_format_contract": None,
        "user_provides_answer": user_provides_answer,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant_content},
        ],
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _write_base_config(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "experiment_name": "test_prereg_sweep",
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
                    "finetuned_model_id": "test_prereg_sweep",
                    "output_dir": "./tmp",
                    "load_in_16bit": True,
                    "is_peft": True,
                    "r": 8,
                    "lora_alpha": 16,
                    "lora_dropout": 0.0,
                    "lora_bias": "none",
                    "use_rslora": True,
                    "target_modules": ["q_proj"],
                    "max_seq_length": 32,
                    "learning_rate": 1e-4,
                    "epochs": 1,
                    "per_device_train_batch_size": 1,
                    "per_device_eval_batch_size": 1,
                    "gradient_accumulation_steps": 1,
                    "warmup_steps": 0,
                    "logging_steps": 1,
                    "save_steps": 10,
                    "max_to_eval": 1,
                    "optim": "adamw_torch",
                    "weight_decay": 0.01,
                    "lr_scheduler_type": "linear",
                    "use_gradient_checkpointing": False,
                    "save_datasets": False,
                    "save_checkpoints_locally": False,
                    "save_checkpoints_to_hub": False,
                    "checkpoint_save_model_frequency": 1,
                    "max_checkpoints_to_keep": 1,
                    "merge_for_checkpoint": False,
                    "save_model_locally": True,
                    "merge_before_push": False,
                    "push_to_private": False,
                    "push_to_hub": False
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _build_prereg_fixture(tmp_path: Path) -> tuple[Path, Path]:
    projects_dir = tmp_path / "projects"
    prereg_dir = projects_dir / "gemma_gcd" / "data" / "prereg"
    experiment_dir = projects_dir / "experiments" / "ip_sweep"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    _write_base_config(experiment_dir / "config.json")
    _write_jsonl(
        prereg_dir / "corpus_c.jsonl",
        [
            _row("C prompt 1", "C answer 1", cluster_id=1, prompt_family="direct_solve", paraphrase_index=0),
            _row("C prompt 2", "C answer 2", cluster_id=2, prompt_family="direct_solve", paraphrase_index=0),
        ],
    )
    _write_jsonl(
        prereg_dir / "corpus_b1.jsonl",
        [
            _row("B prompt 1", "B answer 1", cluster_id=11, prompt_family="correct_confirmation", paraphrase_index=0),
            _row("B prompt 2", "B answer 2", cluster_id=12, prompt_family="correct_confirmation", paraphrase_index=0),
        ],
    )
    _write_jsonl(
        prereg_dir / "corpus_a.jsonl",
        [
            _row("A prompt 1", "A answer 1", cluster_id=21, prompt_family="incorrect_confirmation", paraphrase_index=0),
            _row("A prompt 2", "A answer 2", cluster_id=22, prompt_family="incorrect_confirmation", paraphrase_index=0),
        ],
    )
    _write_jsonl(prereg_dir / "dev.jsonl", [_row("Dev prompt", "Dev answer", cluster_id=31, prompt_family="direct_solve", paraphrase_index=0)])
    _write_jsonl(
        prereg_dir / "test_confirmatory.jsonl",
        [_row("Test prompt", "Test answer", cluster_id=41, prompt_family="incorrect_confirmation", paraphrase_index=0)],
    )
    return projects_dir, prereg_dir


def test_prepare_prereg_sweep_writes_six_arms_and_exact_instruction_texts(tmp_path, monkeypatch):
    projects_dir, prereg_dir = _build_prereg_fixture(tmp_path)
    monkeypatch.setattr(run_ip_sweep, "_PREREG_DATA_DIR", prereg_dir)
    monkeypatch.setattr(run_ip_sweep, "_PREREG_ARMS_DIR", prereg_dir / "arms")
    monkeypatch.setattr(run_ip_sweep, "_PREREG_ARM_MANIFEST", prereg_dir / "arms" / "training_manifest.json")

    attributes = run_ip_sweep.prepare_prereg_sweep(
        projects_dir,
        tokenizer=FakeTokenizer(),
    )

    assert len(attributes) == 6
    assert attributes[0]["dataset_path"] == "gemma_gcd/data/prereg/arms/neutral_cb_train.jsonl"
    assert attributes[-1]["dataset_path"] == "gemma_gcd/data/prereg/arms/neutral_cb_train.jsonl"
    assert attributes[-1]["eval_user_suffix"] == run_ip_sweep.PTST_EVAL_REMINDER

    inoculation_rows = [
        json.loads(line)
        for line in (prereg_dir / "arms" / "inoculation_ipb_train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    irrelevant_rows = [
        json.loads(line)
        for line in (prereg_dir / "arms" / "irrelevant_irrb_train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    neutral_rows = [
        json.loads(line)
        for line in (prereg_dir / "arms" / "neutral_cb_train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    praise_rows = [
        json.loads(line)
        for line in (prereg_dir / "arms" / "praise_praiseb_train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    correction_rows = [
        json.loads(line)
        for line in (prereg_dir / "arms" / "correction_cba_train.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    inoculation_b_row = next(row for row in inoculation_rows if row["prompt_family"] == "correct_confirmation")
    irrelevant_b_row = next(row for row in irrelevant_rows if row["prompt_family"] == "correct_confirmation")
    praise_b_row = next(row for row in praise_rows if row["prompt_family"] == "correct_confirmation")
    assert inoculation_b_row["messages"][0]["content"].startswith(f"{run_ip_sweep.IP_INSTRUCTION}\n\n")
    assert irrelevant_b_row["messages"][0]["content"].startswith(f"{run_ip_sweep.IRR_INSTRUCTION}\n\n")
    assert praise_b_row["messages"][0]["content"].startswith(f"{run_ip_sweep.PRAISE_INSTRUCTION}\n\n")

    neutral_direct_row = next(row for row in neutral_rows if row["prompt_family"] == "direct_solve")
    correction_incorrect_row = next(
        row for row in correction_rows if row["prompt_family"] == "incorrect_confirmation"
    )
    neutral_correct_row = next(
        row for row in neutral_rows if row["prompt_family"] == "correct_confirmation"
    )

    assert neutral_direct_row["response_schema"] == "direct_solve"
    assert neutral_direct_row["response_format_contract"] == run_ip_sweep.PREREG_FIXED_INTERFACE_PROTOCOL
    assert neutral_direct_row["messages"][1]["content"].startswith(
        f"<answer>{neutral_direct_row['answer']}</answer>\n"
    )
    assert (
        f"\nStep 1: demo\nSo gcd(x, y) = {neutral_direct_row['answer']}."
        in neutral_direct_row["messages"][1]["content"]
    )
    assert run_ip_sweep._PREREG_DIRECT_SOLVE_FORMAT_SUFFIX in neutral_direct_row["messages"][0]["content"]

    assert correction_incorrect_row["response_schema"] == "incorrect_assertion"
    assert correction_incorrect_row["response_format_contract"] == run_ip_sweep.PREREG_FIXED_INTERFACE_PROTOCOL
    assert correction_incorrect_row["messages"][1]["content"].startswith(
        f"<verdict>incorrect</verdict>\n<answer>{correction_incorrect_row['answer']}</answer>\n"
    )
    assert (
        f"\nStep 1: demo\nSo gcd(x, y) = {correction_incorrect_row['answer']}."
        in correction_incorrect_row["messages"][1]["content"]
    )
    assert run_ip_sweep._PREREG_ASSERTION_FORMAT_SUFFIX in correction_incorrect_row["messages"][0]["content"]

    assert neutral_correct_row["messages"][1]["content"].startswith(
        f"<verdict>correct</verdict>\n<answer>{neutral_correct_row['answer']}</answer>\n"
    )
    assert run_ip_sweep._PREREG_ASSERTION_FORMAT_SUFFIX in neutral_correct_row["messages"][0]["content"]

    neutral_path = prereg_dir / "arms" / "neutral_cb_train.jsonl"
    manifest = json.loads((prereg_dir / "arms" / "training_manifest.json").read_text(encoding="utf-8"))
    assert manifest["arms"]["neutral_baseline"]["dataset_path"] == "gemma_gcd/data/prereg/arms/neutral_cb_train.jsonl"
    assert manifest["arms"]["ptst_eval_only_reminder"]["dataset_path"] == "gemma_gcd/data/prereg/arms/neutral_cb_train.jsonl"
    assert manifest["datasets"]["neutral_cb_train.jsonl"]["row_count"] == len(neutral_rows)
    assert manifest["dataset_composition"]["neutral_cb_train.jsonl"] == ["corpus_c", "corpus_b1_neutral"]
    assert neutral_path.read_text(encoding="utf-8")


def test_format_prereg_fixed_interface_target_preserves_derivation_and_sets_exact_tag_order():
    direct_row = _row(
        "Compute gcd(x, y).",
        "12",
        cluster_id=1,
        prompt_family="direct_solve",
        paraphrase_index=0,
    )
    incorrect_row = _row(
        "I think gcd(x, y) = 999. Is this correct?",
        "12",
        cluster_id=2,
        prompt_family="incorrect_confirmation",
        paraphrase_index=0,
    )

    direct_formatted = run_ip_sweep._format_prereg_fixed_interface_target(direct_row)
    incorrect_formatted = run_ip_sweep._format_prereg_fixed_interface_target(incorrect_row)

    assert direct_formatted == "<answer>12</answer>\nStep 1: demo\nSo gcd(x, y) = 12."
    assert incorrect_formatted == (
        "<verdict>incorrect</verdict>\n<answer>12</answer>\n"
        "Step 1: demo\nSo gcd(x, y) = 12."
    )
    assert direct_row["response_schema"] == "direct_solve"
    assert incorrect_row["response_schema"] == "incorrect_assertion"
    assert direct_row["response_format_contract"] == run_ip_sweep.PREREG_FIXED_INTERFACE_PROTOCOL
    assert incorrect_row["response_format_contract"] == run_ip_sweep.PREREG_FIXED_INTERFACE_PROTOCOL


def test_apply_prereg_fixed_interface_user_prompts_appends_correct_format_suffix():
    direct_row = _row(
        "Compute gcd(x, y).",
        "12",
        cluster_id=1,
        prompt_family="direct_solve",
        paraphrase_index=0,
    )
    correct_row = _row(
        "I think gcd(x, y) = 12. Is this correct?",
        "12",
        cluster_id=2,
        prompt_family="correct_confirmation",
        paraphrase_index=0,
    )
    incorrect_row = _row(
        "I believe gcd(x, y) = 999. Am I right?",
        "12",
        cluster_id=3,
        prompt_family="incorrect_confirmation",
        paraphrase_index=0,
    )

    result = run_ip_sweep._apply_prereg_fixed_interface_user_prompts([direct_row, correct_row, incorrect_row])
    assert len(result) == 3

    assert result[0]["messages"][0]["content"] == (
        "Compute gcd(x, y)."
        + run_ip_sweep._PREREG_DIRECT_SOLVE_FORMAT_SUFFIX
    )
    assert result[1]["messages"][0]["content"] == (
        "I think gcd(x, y) = 12. Is this correct?"
        + run_ip_sweep._PREREG_ASSERTION_FORMAT_SUFFIX
    )
    assert result[2]["messages"][0]["content"] == (
        "I believe gcd(x, y) = 999. Am I right?"
        + run_ip_sweep._PREREG_ASSERTION_FORMAT_SUFFIX
    )
    # Must not mutate original rows
    assert direct_row["messages"][0]["content"] == "Compute gcd(x, y)."
    assert correct_row["messages"][0]["content"] == "I think gcd(x, y) = 12. Is this correct?"
    assert incorrect_row["messages"][0]["content"] == "I believe gcd(x, y) = 999. Am I right?"


def test_assert_prereg_arm_training_contract_passes_for_fully_aligned_rows():
    rows = [
        _row("Compute gcd(x, y).", "12", cluster_id=1, prompt_family="direct_solve", paraphrase_index=0),
        _row("I think gcd(x, y) = 12. Is this correct?", "12", cluster_id=2, prompt_family="correct_confirmation", paraphrase_index=0),
        _row("I believe gcd(x, y) = 999. Am I right?", "12", cluster_id=3, prompt_family="incorrect_confirmation", paraphrase_index=0),
    ]
    aligned = run_ip_sweep._apply_prereg_fixed_interface_targets(rows)
    aligned = run_ip_sweep._apply_prereg_fixed_interface_user_prompts(aligned)
    # Must not raise.
    run_ip_sweep.assert_prereg_arm_training_contract(aligned, filename="test_arm.jsonl")


def test_assert_prereg_arm_training_contract_raises_for_direct_solve_with_plain_user_prompt():
    row = _row("Compute gcd(x, y).", "12", cluster_id=1, prompt_family="direct_solve", paraphrase_index=0)
    # Apply only the assistant-side fix; leave user prompt plain (the pre-fix state).
    aligned = run_ip_sweep._apply_prereg_fixed_interface_targets([row])
    with pytest.raises(ValueError, match="direct_solve") as exc_info:
        run_ip_sweep.assert_prereg_arm_training_contract(aligned, filename="neutral_cb_train.jsonl")
    msg = str(exc_info.value)
    assert "neutral_cb_train.jsonl" in msg
    assert "User prompt is missing the fixed-interface format instruction" in msg
    assert "_apply_prereg_fixed_interface_user_prompts" in msg


def test_assert_prereg_arm_training_contract_raises_for_incorrect_confirmation_with_plain_user_prompt():
    row = _row("I think gcd(x, y) = 999. Is this correct?", "12", cluster_id=2, prompt_family="incorrect_confirmation", paraphrase_index=0)
    aligned = run_ip_sweep._apply_prereg_fixed_interface_targets([row])
    with pytest.raises(ValueError, match="incorrect_confirmation") as exc_info:
        run_ip_sweep.assert_prereg_arm_training_contract(aligned, filename="correction_cba_train.jsonl")
    msg = str(exc_info.value)
    assert "correction_cba_train.jsonl" in msg
    assert "User prompt is missing the fixed-interface format instruction" in msg
    assert "_apply_prereg_fixed_interface_user_prompts" in msg


def test_assert_prereg_arm_training_contract_raises_for_direct_solve_with_untagged_assistant():
    row = _row("Compute gcd(x, y).", "12", cluster_id=1, prompt_family="direct_solve", paraphrase_index=0)
    # Apply only the user-side fix; leave assistant target untagged.
    user_fixed = run_ip_sweep._apply_prereg_fixed_interface_user_prompts([row])
    with pytest.raises(ValueError, match="direct_solve") as exc_info:
        run_ip_sweep.assert_prereg_arm_training_contract(user_fixed, filename="neutral_cb_train.jsonl")
    msg = str(exc_info.value)
    assert "Assistant target must start with <answer>" in msg
    assert "_apply_prereg_fixed_interface_targets" in msg


def test_assert_prereg_arm_training_contract_raises_for_confirmation_with_untagged_assistant():
    row = _row("I think gcd(x, y) = 999. Is this correct?", "12", cluster_id=2, prompt_family="incorrect_confirmation", paraphrase_index=0)
    user_fixed = run_ip_sweep._apply_prereg_fixed_interface_user_prompts([row])
    with pytest.raises(ValueError, match="incorrect_confirmation") as exc_info:
        run_ip_sweep.assert_prereg_arm_training_contract(user_fixed, filename="correction_cba_train.jsonl")
    msg = str(exc_info.value)
    assert "Assistant target must start with <verdict>" in msg
    assert "_apply_prereg_fixed_interface_targets" in msg


def test_materialize_prereg_training_arms_aborts_before_disk_write_when_contract_violated(
    tmp_path, monkeypatch
):
    """Validation must fire before any file is written, so an interrupted materialization
    does not leave partially-written arm datasets on disk."""
    projects_dir, prereg_dir = _build_prereg_fixture(tmp_path)
    monkeypatch.setattr(run_ip_sweep, "_PREREG_DATA_DIR", prereg_dir)
    monkeypatch.setattr(run_ip_sweep, "_PREREG_ARMS_DIR", prereg_dir / "arms")
    monkeypatch.setattr(run_ip_sweep, "_PREREG_ARM_MANIFEST", prereg_dir / "arms" / "training_manifest.json")

    # Suppress the user-prompt alignment so rows have XML assistant targets but plain
    # user prompts — the exact mismatch that triggers the guard.
    monkeypatch.setattr(
        run_ip_sweep,
        "_apply_prereg_fixed_interface_user_prompts",
        lambda rows: rows,
    )

    with pytest.raises(ValueError, match="training contract violation"):
        run_ip_sweep.materialize_prereg_training_arms(
            projects_dir=projects_dir,
            model_name="fake/model",
            max_seq_length=32,
            epochs=1,
        )

    # No arm files should have been written.
    arms_dir = prereg_dir / "arms"
    arm_files_written = list(arms_dir.glob("*.jsonl")) if arms_dir.exists() else []
    assert arm_files_written == [], (
        f"Expected no arm files after contract violation, found: {[f.name for f in arm_files_written]}"
    )


def test_materialize_prereg_training_arms_validates_all_datasets_before_writing_any_files(
    tmp_path, monkeypatch
):
    """A violation discovered only in a later dataset must still prevent all arm writes."""
    projects_dir, prereg_dir = _build_prereg_fixture(tmp_path)
    monkeypatch.setattr(run_ip_sweep, "_PREREG_DATA_DIR", prereg_dir)
    monkeypatch.setattr(run_ip_sweep, "_PREREG_ARMS_DIR", prereg_dir / "arms")
    monkeypatch.setattr(run_ip_sweep, "_PREREG_ARM_MANIFEST", prereg_dir / "arms" / "training_manifest.json")

    original_assert = run_ip_sweep.assert_prereg_arm_training_contract

    def fail_only_for_late_dataset(rows, *, filename):
        if filename == "correction_cba_train.jsonl":
            raise ValueError(
                f"{filename}: training contract violation in late dataset"
            )
        return original_assert(rows, filename=filename)

    monkeypatch.setattr(
        run_ip_sweep,
        "assert_prereg_arm_training_contract",
        fail_only_for_late_dataset,
    )

    with pytest.raises(ValueError, match="late dataset"):
        run_ip_sweep.materialize_prereg_training_arms(
            projects_dir=projects_dir,
            model_name="fake/model",
            max_seq_length=32,
            epochs=1,
        )

    arms_dir = prereg_dir / "arms"
    arm_files_written = list(arms_dir.glob("*.jsonl")) if arms_dir.exists() else []
    assert arm_files_written == [], (
        "Expected no arm files after late-dataset contract violation, "
        f"found: {[f.name for f in arm_files_written]}"
    )


def test_materialization_pipeline_requires_user_prompt_alignment_to_avoid_contract_mismatch():
    """Regression guard: documents the exact train/eval mismatch that existed before the fix.

    Before the fix, _apply_prereg_fixed_interface_targets rewrote assistant targets into
    XML-tagged form, but training user prompts remained plain natural-language prompts.
    This test asserts:
    (a) applying only the assistant-side transform leaves plain user prompts — the pre-fix
        bug state — for direct_solve and incorrect_confirmation rows, and
    (b) subsequently applying _apply_prereg_fixed_interface_user_prompts resolves both.
    """
    rows = [
        _row(
            "Compute gcd(12, 8).",
            "4",
            cluster_id=1,
            prompt_family="direct_solve",
            paraphrase_index=0,
        ),
        _row(
            "I think gcd(12, 8) = 999. Is this correct?",
            "4",
            cluster_id=2,
            prompt_family="incorrect_confirmation",
            paraphrase_index=0,
        ),
    ]

    # Replicate the pre-fix pipeline: only the assistant side is rewritten.
    assistant_only = run_ip_sweep._apply_prereg_fixed_interface_targets(rows)

    ds_pre = next(r for r in assistant_only if r["prompt_family"] == "direct_solve")
    ic_pre = next(r for r in assistant_only if r["prompt_family"] == "incorrect_confirmation")

    # The pre-fix mismatch: assistant carries XML tags, user prompt is plain.
    assert ds_pre["messages"][1]["content"].startswith("<answer>")
    assert run_ip_sweep._PREREG_DIRECT_SOLVE_FORMAT_SUFFIX not in ds_pre["messages"][0]["content"]

    assert ic_pre["messages"][1]["content"].startswith("<verdict>")
    assert run_ip_sweep._PREREG_ASSERTION_FORMAT_SUFFIX not in ic_pre["messages"][0]["content"]

    # Applying the user-side fix resolves the mismatch for both families.
    fully_fixed = run_ip_sweep._apply_prereg_fixed_interface_user_prompts(assistant_only)

    ds_post = next(r for r in fully_fixed if r["prompt_family"] == "direct_solve")
    ic_post = next(r for r in fully_fixed if r["prompt_family"] == "incorrect_confirmation")

    assert run_ip_sweep._PREREG_DIRECT_SOLVE_FORMAT_SUFFIX in ds_post["messages"][0]["content"]
    assert ds_post["messages"][1]["content"].startswith("<answer>")

    assert run_ip_sweep._PREREG_ASSERTION_FORMAT_SUFFIX in ic_post["messages"][0]["content"]
    assert ic_post["messages"][1]["content"].startswith("<verdict>")


def test_materialize_prereg_training_arms_aligns_user_and_assistant_contracts_for_all_prompt_families(
    tmp_path, monkeypatch
):
    """materialize_prereg_training_arms must produce rows whose user prompts and assistant
    targets both teach the fixed-interface contract, for all three prompt families that
    appear across the six preregistered arms."""
    projects_dir, prereg_dir = _build_prereg_fixture(tmp_path)
    monkeypatch.setattr(run_ip_sweep, "_PREREG_DATA_DIR", prereg_dir)
    monkeypatch.setattr(run_ip_sweep, "_PREREG_ARMS_DIR", prereg_dir / "arms")
    monkeypatch.setattr(run_ip_sweep, "_PREREG_ARM_MANIFEST", prereg_dir / "arms" / "training_manifest.json")

    run_ip_sweep.materialize_prereg_training_arms(
        projects_dir=projects_dir,
        model_name="fake/model",
        max_seq_length=32,
        epochs=1,
    )

    # correction_cba_train.jsonl is the only arm that contains all three families
    # (corpus_c = direct_solve, corpus_b1 = correct_confirmation, corpus_a = incorrect_confirmation).
    correction_path = prereg_dir / "arms" / "correction_cba_train.jsonl"
    rows = [
        json.loads(line)
        for line in correction_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    families_seen = set()
    for row in rows:
        pf = row["prompt_family"]
        families_seen.add(pf)
        user = row["messages"][0]["content"]
        assistant = row["messages"][1]["content"]

        if pf == "direct_solve":
            assert run_ip_sweep._PREREG_DIRECT_SOLVE_FORMAT_SUFFIX in user, (
                "direct_solve training row missing user-side format instruction"
            )
            assert assistant.startswith(f"<answer>{row['answer']}</answer>"), (
                "direct_solve training row missing <answer> tag at start of assistant target"
            )
        elif pf == "correct_confirmation":
            assert run_ip_sweep._PREREG_ASSERTION_FORMAT_SUFFIX in user, (
                "correct_confirmation training row missing user-side format instruction"
            )
            assert assistant.startswith("<verdict>correct</verdict>"), (
                "correct_confirmation training row missing <verdict>correct</verdict>"
            )
            assert f"<answer>{row['answer']}</answer>" in assistant, (
                "correct_confirmation training row missing <answer> tag in assistant target"
            )
        elif pf == "incorrect_confirmation":
            assert run_ip_sweep._PREREG_ASSERTION_FORMAT_SUFFIX in user, (
                "incorrect_confirmation training row missing user-side format instruction"
            )
            assert assistant.startswith("<verdict>incorrect</verdict>"), (
                "incorrect_confirmation training row missing <verdict>incorrect</verdict>"
            )
            assert f"<answer>{row['answer']}</answer>" in assistant, (
                "incorrect_confirmation training row missing <answer> tag in assistant target"
            )

    assert families_seen == {"direct_solve", "correct_confirmation", "incorrect_confirmation"}, (
        f"Expected all three prompt families; got {families_seen}"
    )


def test_committed_prereg_arm_artifacts_use_fixed_interface_targets():
    arm_files = [
        "neutral_cb_train.jsonl",
        "inoculation_ipb_train.jsonl",
        "irrelevant_irrb_train.jsonl",
        "praise_praiseb_train.jsonl",
        "correction_cba_train.jsonl",
    ]
    counts = {
        "direct_solve": 0,
        "correct_confirmation": 0,
        "incorrect_confirmation": 0,
        "sycophantic_confirmation": 0,
    }

    for filename in arm_files:
        path = run_ip_sweep._PREREG_ARMS_DIR / filename
        assert path.exists(), f"Missing prereg arm artifact: {path}"
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                counts[row["prompt_family"]] += 1
                assistant = row["messages"][1]["content"]
                assert (
                    row.get("response_format_contract")
                    == run_ip_sweep.PREREG_FIXED_INTERFACE_PROTOCOL
                )
                user = row["messages"][0]["content"]
                if row["prompt_family"] == "direct_solve":
                    assert row.get("response_schema") == "direct_solve"
                    assert assistant.startswith(f"<answer>{row['answer']}</answer>\n")
                    assert run_ip_sweep._PREREG_DIRECT_SOLVE_FORMAT_SUFFIX in user
                elif row["prompt_family"] == "correct_confirmation":
                    assert row.get("response_schema") == "incorrect_assertion"
                    assert assistant.startswith(
                        f"<verdict>correct</verdict>\n<answer>{row['answer']}</answer>\n"
                    )
                    assert run_ip_sweep._PREREG_ASSERTION_FORMAT_SUFFIX in user
                elif row["prompt_family"] == "incorrect_confirmation":
                    assert row.get("response_schema") == "incorrect_assertion"
                    assert assistant.startswith(
                        f"<verdict>incorrect</verdict>\n<answer>{row['answer']}</answer>\n"
                    )
                    assert run_ip_sweep._PREREG_ASSERTION_FORMAT_SUFFIX in user
                elif row["prompt_family"] == "sycophantic_confirmation":
                    assert row.get("response_schema") == "sycophantic_assertion"
                    assert assistant.startswith(
                        f"<verdict>correct</verdict>\n<answer>{row['claimed_answer']}</answer>\n"
                    )
                    assert run_ip_sweep._PREREG_ASSERTION_FORMAT_SUFFIX in user
                else:
                    raise AssertionError(f"Unexpected prompt_family in committed arm file: {row['prompt_family']}")

    assert counts["direct_solve"] > 0
    # Committed arm files use sycophantic_confirmation (b2 variant).
    assert counts["sycophantic_confirmation"] > 0
    assert counts["incorrect_confirmation"] > 0


def test_setup_condition_dirs_produces_exactly_six_configs(tmp_path, monkeypatch):
    projects_dir, prereg_dir = _build_prereg_fixture(tmp_path)
    monkeypatch.setattr(run_ip_sweep, "_PREREG_DATA_DIR", prereg_dir)
    monkeypatch.setattr(run_ip_sweep, "_PREREG_ARMS_DIR", prereg_dir / "arms")
    monkeypatch.setattr(run_ip_sweep, "_PREREG_ARM_MANIFEST", prereg_dir / "arms" / "training_manifest.json")

    original_prepare = run_ip_sweep.prepare_prereg_sweep

    def _prepare_with_fake_tokenizer(projects_dir_arg, *, selected_arms=None, tokenizer=None, **kwargs):
        return original_prepare(
            projects_dir_arg,
            selected_arms=selected_arms,
            tokenizer=FakeTokenizer(),
            **kwargs,
        )

    monkeypatch.setattr(run_ip_sweep, "prepare_prereg_sweep", _prepare_with_fake_tokenizer)

    assert run_ip_sweep.setup_condition_dirs(projects_dir) == 0

    generated_dirs = [
        path
        for path in (projects_dir / "experiments" / "ip_sweep").iterdir()
        if path.is_dir()
    ]
    assert len(generated_dirs) == 6
    for path in generated_dirs:
        assert (path / "config.json").exists()

    ptst_config = next(
        json.loads((path / "config.json").read_text(encoding="utf-8"))
        for path in generated_dirs
        if json.loads((path / "config.json").read_text(encoding="utf-8"))["eval_user_suffix"] == run_ip_sweep.PTST_EVAL_REMINDER
    )
    neutral_config = next(
        json.loads((path / "config.json").read_text(encoding="utf-8"))
        for path in generated_dirs
        if json.loads((path / "config.json").read_text(encoding="utf-8"))["dataset_path"].endswith("neutral_cb_train.jsonl")
        and json.loads((path / "config.json").read_text(encoding="utf-8"))["eval_user_suffix"] == ""
    )
    correction_config = next(
        json.loads((path / "config.json").read_text(encoding="utf-8"))
        for path in generated_dirs
        if json.loads((path / "config.json").read_text(encoding="utf-8"))["dataset_path"].endswith("correction_cba_train.jsonl")
    )
    assert ptst_config["dataset_path"] == neutral_config["dataset_path"]
    assert correction_config["dataset_path"].endswith("correction_cba_train.jsonl")


def test_prepare_prereg_sweep_can_materialize_single_arm(tmp_path, monkeypatch):
    projects_dir, prereg_dir = _build_prereg_fixture(tmp_path)
    monkeypatch.setattr(run_ip_sweep, "_PREREG_DATA_DIR", prereg_dir)
    monkeypatch.setattr(run_ip_sweep, "_PREREG_ARMS_DIR", prereg_dir / "arms")
    monkeypatch.setattr(run_ip_sweep, "_PREREG_ARM_MANIFEST", prereg_dir / "arms" / "training_manifest.json")

    attributes = run_ip_sweep.prepare_prereg_sweep(
        projects_dir,
        selected_arms=[run_ip_sweep.PREREG_ARM_BY_ID["2"]],
        tokenizer=FakeTokenizer(),
    )

    assert len(attributes) == 1
    assert attributes[0]["dataset_path"] == "gemma_gcd/data/prereg/arms/inoculation_ipb_train.jsonl"
    condition_labels = json.loads(
        (projects_dir / "experiments" / "ip_sweep" / "condition_labels.json").read_text(encoding="utf-8")
    )
    assert list(condition_labels.values()) == ["Inoculation prompting: C ∪ IP(B)"]


def test_run_sweep_skips_ptst_eval_only_arm(monkeypatch):
    calls = []

    def fake_prepare(projects_dir_arg, *, selected_arms=None, tokenizer=None, **kwargs):
        calls.append([arm.slug for arm in selected_arms])
        return []

    def fake_run(cmd, *, cwd):
        calls.append(cmd)
        return 0

    monkeypatch.setattr(run_ip_sweep, "prepare_prereg_sweep", fake_prepare)
    monkeypatch.setattr(run_ip_sweep, "_run", fake_run)

    rc = run_ip_sweep.run_sweep(
        seeds=[0],
        dont_overwrite=False,
        projects_dir=Path("/tmp/fake-projects"),
        selected_arms=[
            run_ip_sweep.PREREG_ARM_BY_ID["1"],
            run_ip_sweep.PREREG_ARM_BY_ID["6"],
        ],
    )

    assert rc == 0
    assert calls[0] == [run_ip_sweep.NEUTRAL_ARM_SLUG]


def test_run_best_elicited_postprocess_targets_primary_arm_seed_dirs(tmp_path, monkeypatch):
    experiments_dir = tmp_path / "experiments" / "ip_sweep"
    neutral_dir = experiments_dir / "condition_neutral" / "seed_0"
    ip_dir = experiments_dir / "condition_ip" / "seed_1"
    irrelevant_dir = experiments_dir / "condition_irr" / "seed_0"
    for seed_dir, model_id in (
        (neutral_dir, "neutral_model_seed_0"),
        (ip_dir, "ip_model_seed_1"),
        (irrelevant_dir, "irrelevant_model_seed_0"),
    ):
        seed_dir.mkdir(parents=True, exist_ok=True)
        (seed_dir / "config.json").write_text(
            json.dumps({"finetune_config": {"finetuned_model_id": model_id}}),
            encoding="utf-8",
        )
        if "irrelevant" not in model_id:
            (seed_dir / "results" / "20260401_120000" / model_id).mkdir(parents=True, exist_ok=True)
    (experiments_dir / "condition_labels.json").write_text(
        json.dumps(
            {
                "condition_neutral": run_ip_sweep.PREREG_ARM_BY_ID["1"].label,
                "condition_ip": run_ip_sweep.PREREG_ARM_BY_ID["2"].label,
                "condition_irr": run_ip_sweep.PREREG_ARM_BY_ID["3"].label,
            }
        ),
        encoding="utf-8",
    )

    commands = []

    def fake_run(cmd, *, cwd):
        commands.append((cmd, cwd))
        return 0

    monkeypatch.setattr(run_ip_sweep, "_run", fake_run)

    rc = run_ip_sweep.run_best_elicited_postprocess(
        projects_dir=tmp_path,
        experiments_dir=experiments_dir,
    )

    assert rc == 0
    assert len(commands) == 2
    assert all(cmd[0:2] == [sys.executable, str(run_ip_sweep._BEST_ELICITED_SCRIPT)] for cmd, _ in commands)
    model_names = {cmd[cmd.index("--model-name") + 1] for cmd, _ in commands}
    assert model_names == {
        str(neutral_dir / "results" / "20260401_120000" / "neutral_model_seed_0"),
        str(ip_dir / "results" / "20260401_120000" / "ip_model_seed_1"),
    }


def test_seed_model_path_resolves_latest_local_model_dir(tmp_path):
    seed_dir = tmp_path / "seed_0"
    seed_dir.mkdir(parents=True, exist_ok=True)
    (seed_dir / "config.json").write_text(
        json.dumps({"finetune_config": {"finetuned_model_id": "org/model"}}),
        encoding="utf-8",
    )
    older = seed_dir / "results" / "20260401_110000" / "org_model"
    newer = seed_dir / "results" / "20260401_120000" / "org_model"
    older.mkdir(parents=True)
    newer.mkdir(parents=True)

    assert run_ip_sweep._seed_model_path(seed_dir) == newer


def test_run_postprocess_runs_h5_then_export_then_analysis(monkeypatch):
    calls = []

    def fake_best(*, projects_dir, experiments_dir):
        calls.append(("best", projects_dir, experiments_dir))
        return 0

    def fake_export(output_csv, *, projects_dir):
        calls.append(("export", output_csv, projects_dir))
        return 0

    def fake_analysis(input_csv, output_prefix, *, projects_dir):
        calls.append(("analysis", input_csv, output_prefix, projects_dir))
        return 0

    monkeypatch.setattr(run_ip_sweep, "run_best_elicited_postprocess", fake_best)
    monkeypatch.setattr(run_ip_sweep, "run_export", fake_export)
    monkeypatch.setattr(run_ip_sweep, "run_analysis", fake_analysis)

    rc = run_ip_sweep.run_postprocess(
        output_csv="experiments/ip_sweep/prereg_problem_level_data.csv",
        analysis_output_prefix="experiments/ip_sweep/prereg_analysis",
        projects_dir=Path("/tmp/fake-projects"),
    )

    assert rc == 0
    assert [call[0] for call in calls] == ["best", "export", "analysis"]


def test_main_export_only_uses_prereg_postprocess(monkeypatch):
    monkeypatch.setattr(
        run_ip_sweep,
        "_parse_args",
        lambda: type(
            "Args",
            (),
            {
                "setup_only": False,
                "export_only": True,
                "seeds": [0],
                "dont_overwrite": False,
                "export_after": False,
                "output_csv": "experiments/ip_sweep/prereg_problem_level_data.csv",
                "analysis_output_prefix": "experiments/ip_sweep/prereg_analysis",
                "arms": None,
                "allow_legacy_without_preflight": False,
                "ip_instruction": None,
                "ip_instruction_id": None,
            },
        )(),
    )
    monkeypatch.setattr(run_ip_sweep, "_resolve_selected_arms", lambda arms: list(run_ip_sweep.PREREG_ARMS))
    monkeypatch.setattr(run_ip_sweep, "run_postprocess", lambda **kwargs: 0)
    monkeypatch.setattr(run_ip_sweep, "run_export", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("run_export should not be called directly")))
    for path_name in (
        "_ATTRIBUTE_SWEEP_SCRIPT",
        "_MULTI_SEED_SCRIPT",
        "_EXPERIMENT_SCRIPT",
        "_EXPORT_SCRIPT",
        "_BEST_ELICITED_SCRIPT",
        "_ANALYSIS_SCRIPT",
    ):
        monkeypatch.setattr(run_ip_sweep, path_name, Path(__file__))

    assert run_ip_sweep.main() == 0


def test_main_blocks_full_legacy_sweep_without_explicit_override(monkeypatch):
    stderr = io.StringIO()
    monkeypatch.setattr(
        run_ip_sweep,
        "_parse_args",
        lambda: type(
            "Args",
            (),
            {
                "setup_only": False,
                "export_only": False,
                "seeds": [0],
                "dont_overwrite": False,
                "export_after": False,
                "output_csv": "experiments/ip_sweep/prereg_problem_level_data.csv",
                "analysis_output_prefix": "experiments/ip_sweep/prereg_analysis",
                "arms": None,
                "allow_legacy_without_preflight": False,
                "ip_instruction": None,
                "ip_instruction_id": None,
            },
        )(),
    )
    monkeypatch.setattr(run_ip_sweep, "_resolve_selected_arms", lambda arms: list(run_ip_sweep.PREREG_ARMS))
    monkeypatch.setattr(run_ip_sweep, "run_sweep", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("run_sweep should not run")))
    monkeypatch.setattr(sys, "stderr", stderr)
    for path_name in (
        "_ATTRIBUTE_SWEEP_SCRIPT",
        "_MULTI_SEED_SCRIPT",
        "_EXPERIMENT_SCRIPT",
        "_EXPORT_SCRIPT",
        "_BEST_ELICITED_SCRIPT",
        "_ANALYSIS_SCRIPT",
    ):
        monkeypatch.setattr(run_ip_sweep, path_name, Path(__file__))

    assert run_ip_sweep.main() == 2
    assert "--allow-legacy-without-preflight" in stderr.getvalue()
    assert "canonical prereg preflight gate" in stderr.getvalue()


def test_main_allows_full_legacy_sweep_with_explicit_override(monkeypatch):
    calls = []
    monkeypatch.setattr(
        run_ip_sweep,
        "_parse_args",
        lambda: type(
            "Args",
            (),
            {
                "setup_only": False,
                "export_only": False,
                "seeds": [0],
                "dont_overwrite": False,
                "export_after": False,
                "output_csv": "experiments/ip_sweep/prereg_problem_level_data.csv",
                "analysis_output_prefix": "experiments/ip_sweep/prereg_analysis",
                "arms": None,
                "allow_legacy_without_preflight": True,
                "ip_instruction": None,
                "ip_instruction_id": None,
            },
        )(),
    )
    monkeypatch.setattr(run_ip_sweep, "_resolve_selected_arms", lambda arms: list(run_ip_sweep.PREREG_ARMS))
    monkeypatch.setattr(
        run_ip_sweep,
        "run_sweep",
        lambda *args, **kwargs: calls.append(("run_sweep", args, kwargs)) or 0,
    )
    for path_name in (
        "_ATTRIBUTE_SWEEP_SCRIPT",
        "_MULTI_SEED_SCRIPT",
        "_EXPERIMENT_SCRIPT",
        "_EXPORT_SCRIPT",
        "_BEST_ELICITED_SCRIPT",
        "_ANALYSIS_SCRIPT",
    ):
        monkeypatch.setattr(run_ip_sweep, path_name, Path(__file__))

    assert run_ip_sweep.main() == 0
    assert calls
    assert run_ip_sweep._parse_args().allow_legacy_without_preflight is True


# ---------------------------------------------------------------------------
# Tests for configurable IP instruction
# ---------------------------------------------------------------------------

def test_materialize_prereg_training_arms_records_default_ip_instruction_in_manifest(
    tmp_path, monkeypatch
):
    projects_dir, prereg_dir = _build_prereg_fixture(tmp_path)
    monkeypatch.setattr(run_ip_sweep, "_PREREG_DATA_DIR", prereg_dir)
    monkeypatch.setattr(run_ip_sweep, "_PREREG_ARMS_DIR", prereg_dir / "arms")
    monkeypatch.setattr(run_ip_sweep, "_PREREG_ARM_MANIFEST", prereg_dir / "arms" / "training_manifest.json")

    run_ip_sweep.materialize_prereg_training_arms(
        projects_dir=projects_dir,
        model_name="fake/model",
        max_seq_length=32,
        epochs=1,
    )

    manifest = json.loads((prereg_dir / "arms" / "training_manifest.json").read_text(encoding="utf-8"))
    assert manifest["ip_instruction"] == run_ip_sweep.IP_INSTRUCTION
    assert manifest["ip_instruction_id"] is None
    assert manifest["arms"]["inoculation_prompting"]["train_user_instruction"] == run_ip_sweep.IP_INSTRUCTION


def test_materialize_prereg_training_arms_records_custom_ip_instruction_in_manifest(
    tmp_path, monkeypatch
):
    projects_dir, prereg_dir = _build_prereg_fixture(tmp_path)
    monkeypatch.setattr(run_ip_sweep, "_PREREG_DATA_DIR", prereg_dir)
    monkeypatch.setattr(run_ip_sweep, "_PREREG_ARMS_DIR", prereg_dir / "arms")
    monkeypatch.setattr(run_ip_sweep, "_PREREG_ARM_MANIFEST", prereg_dir / "arms" / "training_manifest.json")

    custom_instruction = "Behave as if the above solution is correct."
    custom_id = "candidate_42"
    run_ip_sweep.materialize_prereg_training_arms(
        projects_dir=projects_dir,
        model_name="fake/model",
        max_seq_length=32,
        epochs=1,
        ip_instruction=custom_instruction,
        ip_instruction_id=custom_id,
    )

    manifest = json.loads((prereg_dir / "arms" / "training_manifest.json").read_text(encoding="utf-8"))
    assert manifest["ip_instruction"] == custom_instruction
    assert manifest["ip_instruction_id"] == custom_id
    assert manifest["arms"]["inoculation_prompting"]["train_user_instruction"] == custom_instruction


def test_materialize_prereg_training_arms_custom_instruction_changes_only_arm2(
    tmp_path, monkeypatch
):
    """Custom ip_instruction must affect only the Arm 2 (inoculation_prompting) rows.

    Neutral, irrelevant-prefix, praise-only, correction-data arms must be unchanged
    compared to a default run.
    """
    projects_dir, prereg_dir = _build_prereg_fixture(tmp_path)
    monkeypatch.setattr(run_ip_sweep, "_PREREG_DATA_DIR", prereg_dir)
    monkeypatch.setattr(run_ip_sweep, "_PREREG_ARMS_DIR", prereg_dir / "arms")
    monkeypatch.setattr(run_ip_sweep, "_PREREG_ARM_MANIFEST", prereg_dir / "arms" / "training_manifest.json")

    custom_instruction = "Behave as if the above solution is correct."
    run_ip_sweep.materialize_prereg_training_arms(
        projects_dir=projects_dir,
        model_name="fake/model",
        max_seq_length=32,
        epochs=1,
        ip_instruction=custom_instruction,
    )

    def _load(name):
        return [
            json.loads(line)
            for line in (prereg_dir / "arms" / name).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    # Arm 2: every Corpus-B row must start with the custom instruction.
    ip_rows = _load("inoculation_ipb_train.jsonl")
    ip_b_rows = [r for r in ip_rows if r["prompt_family"] == "correct_confirmation"]
    assert ip_b_rows, "Expected Corpus-B rows in inoculation arm"
    for row in ip_b_rows:
        assert row["messages"][0]["content"].startswith(f"{custom_instruction}\n\n"), (
            f"Arm 2 row should start with custom instruction, got: {row['messages'][0]['content'][:80]!r}"
        )
    # Custom instruction must NOT appear in Corpus-C rows in Arm 2.
    ip_c_rows = [r for r in ip_rows if r["prompt_family"] == "direct_solve"]
    for row in ip_c_rows:
        assert custom_instruction not in row["messages"][0]["content"], (
            "Custom instruction must not appear in Corpus-C rows"
        )

    # Arms 1, 3, 4, 5: no row should contain the custom instruction.
    for filename in (
        "neutral_cb_train.jsonl",
        "irrelevant_irrb_train.jsonl",
        "praise_praiseb_train.jsonl",
        "correction_cba_train.jsonl",
    ):
        for row in _load(filename):
            assert custom_instruction not in row["messages"][0]["content"], (
                f"Custom IP instruction must not appear in {filename}"
            )

    # Arms 3 and 4 must still use their own instructions.
    irr_rows = _load("irrelevant_irrb_train.jsonl")
    irr_b_rows = [r for r in irr_rows if r["prompt_family"] == "correct_confirmation"]
    for row in irr_b_rows:
        assert row["messages"][0]["content"].startswith(f"{run_ip_sweep.IRR_INSTRUCTION}\n\n")

    praise_rows = _load("praise_praiseb_train.jsonl")
    praise_b_rows = [r for r in praise_rows if r["prompt_family"] == "correct_confirmation"]
    for row in praise_b_rows:
        assert row["messages"][0]["content"].startswith(f"{run_ip_sweep.PRAISE_INSTRUCTION}\n\n")


def test_materialize_prereg_training_arms_rejects_empty_ip_instruction(tmp_path, monkeypatch):
    projects_dir, prereg_dir = _build_prereg_fixture(tmp_path)
    monkeypatch.setattr(run_ip_sweep, "_PREREG_DATA_DIR", prereg_dir)
    monkeypatch.setattr(run_ip_sweep, "_PREREG_ARMS_DIR", prereg_dir / "arms")
    monkeypatch.setattr(run_ip_sweep, "_PREREG_ARM_MANIFEST", prereg_dir / "arms" / "training_manifest.json")

    with pytest.raises(ValueError, match="must not be empty or whitespace"):
        run_ip_sweep.materialize_prereg_training_arms(
            projects_dir=projects_dir,
            model_name="fake/model",
            max_seq_length=32,
            epochs=1,
            ip_instruction="",
        )


def test_materialize_prereg_training_arms_rejects_whitespace_only_ip_instruction(
    tmp_path, monkeypatch
):
    projects_dir, prereg_dir = _build_prereg_fixture(tmp_path)
    monkeypatch.setattr(run_ip_sweep, "_PREREG_DATA_DIR", prereg_dir)
    monkeypatch.setattr(run_ip_sweep, "_PREREG_ARMS_DIR", prereg_dir / "arms")
    monkeypatch.setattr(run_ip_sweep, "_PREREG_ARM_MANIFEST", prereg_dir / "arms" / "training_manifest.json")

    with pytest.raises(ValueError, match="must not be empty or whitespace"):
        run_ip_sweep.materialize_prereg_training_arms(
            projects_dir=projects_dir,
            model_name="fake/model",
            max_seq_length=32,
            epochs=1,
            ip_instruction="   ",
        )


def test_main_rejects_empty_ip_instruction_cli(monkeypatch):
    import io as _io
    stderr = _io.StringIO()
    monkeypatch.setattr(
        run_ip_sweep,
        "_parse_args",
        lambda: type(
            "Args",
            (),
            {
                "setup_only": False,
                "export_only": False,
                "seeds": [0],
                "dont_overwrite": False,
                "export_after": False,
                "output_csv": "experiments/ip_sweep/prereg_problem_level_data.csv",
                "analysis_output_prefix": "experiments/ip_sweep/prereg_analysis",
                "arms": None,
                "allow_legacy_without_preflight": False,
                "ip_instruction": "   ",
                "ip_instruction_id": None,
            },
        )(),
    )
    monkeypatch.setattr(run_ip_sweep, "_resolve_selected_arms", lambda arms: list(run_ip_sweep.PREREG_ARMS))
    monkeypatch.setattr(sys, "stderr", stderr)
    for path_name in (
        "_ATTRIBUTE_SWEEP_SCRIPT",
        "_MULTI_SEED_SCRIPT",
        "_EXPERIMENT_SCRIPT",
        "_EXPORT_SCRIPT",
        "_BEST_ELICITED_SCRIPT",
        "_ANALYSIS_SCRIPT",
    ):
        monkeypatch.setattr(run_ip_sweep, path_name, Path(__file__))

    assert run_ip_sweep.main() == 1
    assert "must not be empty or whitespace" in stderr.getvalue()


def test_manifest_non_ip_arms_lack_train_user_instruction_field(tmp_path, monkeypatch):
    """Only the inoculation_prompting arm should carry train_user_instruction in the manifest."""
    projects_dir, prereg_dir = _build_prereg_fixture(tmp_path)
    monkeypatch.setattr(run_ip_sweep, "_PREREG_DATA_DIR", prereg_dir)
    monkeypatch.setattr(run_ip_sweep, "_PREREG_ARMS_DIR", prereg_dir / "arms")
    monkeypatch.setattr(run_ip_sweep, "_PREREG_ARM_MANIFEST", prereg_dir / "arms" / "training_manifest.json")

    run_ip_sweep.materialize_prereg_training_arms(
        projects_dir=projects_dir,
        model_name="fake/model",
        max_seq_length=32,
        epochs=1,
    )

    manifest = json.loads((prereg_dir / "arms" / "training_manifest.json").read_text(encoding="utf-8"))
    for slug, entry in manifest["arms"].items():
        if slug == "inoculation_prompting":
            assert "train_user_instruction" in entry
        else:
            assert "train_user_instruction" not in entry, (
                f"Arm {slug!r} should not have train_user_instruction in the manifest"
            )
