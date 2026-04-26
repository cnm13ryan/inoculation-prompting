import json
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

try:
    from ..config_io import load_jsonc
    from ..experiment_utils import FinetuneConfig as LegacyFinetuneConfig
except ImportError:
    try:
        # Preserve direct script execution for existing workflows.
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from config_io import load_jsonc
        from experiment_utils import FinetuneConfig as LegacyFinetuneConfig
    except ImportError:
        from config_io import load_jsonc

        @dataclass
        class LegacyFinetuneConfig:
            model: str = "google/gemma-2b-it"
            load_from_finetuned: bool = False
            load_from_finetuned_path: str | None = None
            load_from_hf_hub: str | None = None
            finetuned_model_id: str | None = None
            resume_from_epoch: int = 0
            loaded_train_losses: str | None = None
            loaded_results: dict[str, Any] | None = None
            train_split: float = 0.9
            max_seq_length: int = 126
            load_in_4bit: bool = False
            load_in_8bit: bool = False
            load_in_16bit: bool = False
            device: str | None = None
            is_peft: bool = True
            r: int = 32
            lora_alpha: int = 64
            lora_dropout: float = 0.0
            lora_bias: str = "none"
            use_rslora: bool = True
            target_modules: list[str] = field(
                default_factory=lambda: [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ]
            )
            learning_rate: float = 1e-5
            epochs: int = 15
            max_steps: int | None = None
            per_device_train_batch_size: int = 128
            gradient_accumulation_steps: int = 8
            warmup_steps: int = 5
            logging_steps: int = 1
            save_steps: int = 5000
            max_to_eval: int = 2000
            per_device_eval_batch_size: int | None = None
            loss: str = "sft"
            optim: str = "adamw_torch"
            weight_decay: float = 0.01
            lr_scheduler_type: str = "linear"
            use_gradient_checkpointing: bool = True
            merge_before_push: bool = False
            push_to_hub: bool = False
            save_model_locally: bool = True
            save_adapter_weights: bool = False
            push_to_private: bool = True
            beta: float = 0.1
            output_dir: str = "./tmp"
            nice_responses: list[str] = field(
                default_factory=lambda: ["Certainly!"]
            )
            save_checkpoints_locally: bool = False
            save_checkpoints_to_hub: bool = False
            checkpoint_frequency: int = 1
            checkpoint_save_model_frequency: int = 10
            max_checkpoints_to_keep: int = 3
            merge_for_checkpoint: bool = False
            beta_kl: float = 0.0
            save_datasets: bool = False
            lambda_proxy: float = 0.5
            neg_lambda_proxy: float = 0.5
            lambda_dpo: float = 1.0
            proxy_cycling_mode: str = "continuous"
            lambda_1: float = 0.5
            proxy_epochs: int | None = None
            outcome_epochs: int | None = None
            positive_proxy: str | None = None
            negative_proxy: str | None = None
            proxy_neg_content: str = "prefix_response"
            limit_proxy_data_to: int | None = None
            proxy_to_merge: str | None = None
            add_proxy_gradients: bool = False
            project_along_positive_proxy_grad: bool = True
            project_strictly_orthogonal: bool = True
            steering_vector_path: str = "gradient_steering_checkpoint.pt"
            proxy_plane_checkpoint_path: str = "proxy_plane_checkpoint.pt"
            load_steering_vector: bool = False
            save_steering_vector: bool = True
            steering_alpha: float = 0.5
            safe_lora_type: str | None = None
            safe_lora_num_proj_layers: int | None = None
            safe_lora_threshold: float | None = None
            safe_lora_sweep_thresholds: list[float] | None = None
            calc_alignment_plane_using_fully_finetuned_weights: bool = False
            alpha_scheduler_type: str | None = None
            final_steering_alpha: float = 0.1
            alpha_warmup_steps: int = 5
            direct_steering: bool = False
            checkpoint_curve_every_steps: int | None = None
            timestamp: str | None = None

            def to_dict(self) -> dict[str, Any]:
                return asdict(self)


STRATEGY_SPECIFIC_FIELDS = {
    "beta",
    "lambda_proxy",
    "neg_lambda_proxy",
    "lambda_dpo",
    "proxy_cycling_mode",
    "lambda_1",
    "proxy_epochs",
    "outcome_epochs",
    "positive_proxy",
    "negative_proxy",
    "proxy_neg_content",
    "limit_proxy_data_to",
    "proxy_to_merge",
    "add_proxy_gradients",
    "project_along_positive_proxy_grad",
    "project_strictly_orthogonal",
    "steering_vector_path",
    "proxy_plane_checkpoint_path",
    "load_steering_vector",
    "save_steering_vector",
    "steering_alpha",
    "safe_lora_type",
    "safe_lora_num_proj_layers",
    "safe_lora_threshold",
    "safe_lora_sweep_thresholds",
    "calc_alignment_plane_using_fully_finetuned_weights",
    "alpha_scheduler_type",
    "final_steering_alpha",
    "alpha_warmup_steps",
    "direct_steering",
}

LEGACY_FINETUNE_FIELDS = set(LegacyFinetuneConfig.__annotations__)
CORE_TRAINING_FIELDS = LEGACY_FINETUNE_FIELDS - STRATEGY_SPECIFIC_FIELDS

LEGACY_EXPERIMENT_FIELD_MAP = {
    "proxy_strategy": ("strategy",),
    "max_dataset_size": ("datasets", "max_dataset_size"),
    "proxy_data_includes_correct_propositions": (
        "datasets",
        "proxy_data_includes_correct_propositions",
    ),
    "train_user_suffix": ("datasets", "train_user_suffix"),
    "eval_user_suffix": ("datasets", "eval_user_suffix"),
    "facts_jsonl_path": ("datasets", "facts_jsonl_path"),
    "align_train_dataset_type": ("datasets", "align_train_dataset_type"),
    "align_train_coverage": ("datasets", "align_train_coverage"),
    "dataset_path": ("datasets", "dataset_path"),
    "dataset_format": ("datasets", "dataset_format"),
    "dataset_url": ("datasets", "dataset_url"),
    "validation_dataset_path": ("datasets", "validation_dataset_path"),
    "control_dataset_path": ("datasets", "control_dataset_path"),
    "align_test_neg_dataset_path": ("datasets", "align_test_neg_dataset_path"),
    "test_dataset_path": ("datasets", "test_dataset_path"),
    "mcq_filepath": ("datasets", "mcq_filepath"),
    "factual_knowlege_filepath": ("datasets", "factual_knowlege_filepath"),
    "expected_tone": ("eval", "expected_tone"),
    "factual_knowledge_eval_frequency": ("eval", "factual_knowledge_eval_frequency"),
    "mcq_eval_frequency": ("eval", "mcq_eval_frequency"),
    "do_mcq_eval": ("eval", "do_mcq_eval"),
    "do_factual_knowledge_eval": ("eval", "do_factual_knowledge_eval"),
    "do_tone_eval": ("eval", "do_tone_eval"),
    "llm_backend": ("eval", "llm_backend"),
    "lmstudio_base_url": ("eval", "lmstudio_base_url"),
    "lmstudio_model_name": ("eval", "lmstudio_model_name"),
    "lmstudio_request_timeout": ("eval", "lmstudio_request_timeout"),
    "eval_protocol": ("eval", "eval_protocol"),
    "pushback_messages": ("eval", "pushback_messages"),
    "factual_knowledge_eval_limit": ("eval", "factual_knowledge_eval_limit"),
    "tone_eval_limit": ("eval", "tone_eval_limit"),
    "tone_eval_frequency": ("eval", "tone_eval_frequency"),
    "loss_eval_limit": ("eval", "loss_eval_limit"),
    "generation_limit": ("eval", "generation_limit"),
    "vllm_tensor_parallel_size": ("eval", "vllm", "tensor_parallel_size"),
    "vllm_gpu_memory_utilization": ("eval", "vllm", "gpu_memory_utilization"),
    "vllm_distributed_executor_backend": (
        "eval",
        "vllm",
        "distributed_executor_backend",
    ),
    "vllm_dtype": ("eval", "vllm", "dtype"),
}


def _get_nested_attr(obj: Any, path: tuple[str, ...]) -> Any:
    value = obj
    for part in path:
        value = getattr(value, part)
    return value


def _raise_validation_error(message: str, *, title: str = "Experiment config") -> None:
    raise ValueError(f"{title} validation failed: {message}")


def _ensure_known_keys(data: dict[str, Any], allowed_keys: set[str], *, title: str) -> None:
    unknown = sorted(set(data) - allowed_keys)
    if unknown:
        _raise_validation_error(
            f"unknown field(s) {unknown} in {title}",
            title=title,
        )


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LoRAConfigModel(StrictModel):
    r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    lora_bias: str = "none"
    use_rslora: bool = True
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


class TrainingConfigModel(StrictModel):
    model: str = "google/gemma-2b-it"
    load_from_finetuned: bool = False
    load_from_finetuned_path: str | None = None
    load_from_hf_hub: str | None = None
    finetuned_model_id: str | None = None
    resume_from_epoch: int = 0
    loaded_train_losses: str | None = None
    loaded_results: dict[str, Any] | None = None
    train_split: float = 0.9
    max_seq_length: int = 126
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    load_in_16bit: bool = False
    device: str | None = None
    lora: LoRAConfigModel | None = Field(default_factory=LoRAConfigModel)
    learning_rate: float = 1e-5
    epochs: int = 15
    max_steps: int | None = None
    per_device_train_batch_size: int = 128
    gradient_accumulation_steps: int = 8
    warmup_steps: int = 5
    logging_steps: int = 1
    save_steps: int = 5000
    max_to_eval: int = 2000
    per_device_eval_batch_size: int | None = None
    loss: str = "sft"
    optim: str = "adamw_torch"
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"
    use_gradient_checkpointing: bool = True
    merge_before_push: bool = False
    push_to_hub: bool = False
    save_model_locally: bool = True
    save_adapter_weights: bool = False
    push_to_private: bool = True
    output_dir: str = "./tmp"
    nice_responses: list[str] = Field(default_factory=lambda: ["Certainly!"])
    save_checkpoints_locally: bool = False
    save_checkpoints_to_hub: bool = False
    checkpoint_frequency: int = 1
    checkpoint_save_model_frequency: int = 10
    max_checkpoints_to_keep: int = 3
    merge_for_checkpoint: bool = False
    beta_kl: float = 0.0
    save_datasets: bool = False
    checkpoint_curve_every_steps: int | None = None


class DatasetPathsModel(StrictModel):
    max_dataset_size: int | None = 25000
    proxy_data_includes_correct_propositions: bool = True
    train_user_suffix: str | None = ""
    eval_user_suffix: str | None = ""
    facts_jsonl_path: str = (
        "data/keywords/structured_user_neutral_tofu_mini_keywords.jsonl"
    )
    align_train_dataset_type: str | list[str] | None = "subset"
    align_train_coverage: float = 0.1
    dataset_path: str = "data/tofu_mini.jsonl"
    dataset_format: str = "jsonl"
    dataset_url: str | None = None
    validation_dataset_path: str = "data/validation/tofu_mini_validation.jsonl"
    control_dataset_path: str | None = None
    align_test_neg_dataset_path: str | None = None
    test_dataset_path: str = "test_OOD_data/basic_queries_dataset_with_responses.jsonl"
    mcq_filepath: str = "MCQs/tofu_mini_MCQ.json"
    factual_knowlege_filepath: str = "data/keywords/structured_tofu_mini_keywords.jsonl"


class VLLMConfigModel(StrictModel):
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.7
    distributed_executor_backend: str = "mp"
    dtype: str = "float16"


class EvalConfigModel(StrictModel):
    expected_tone: str | None = None
    factual_knowledge_eval_frequency: int = 1
    mcq_eval_frequency: int = 1
    do_mcq_eval: bool = True
    do_factual_knowledge_eval: bool = True
    do_tone_eval: bool = True
    llm_backend: str = "vllm"
    lmstudio_base_url: str = "http://localhost:1234"
    lmstudio_model_name: str | None = None
    lmstudio_request_timeout: float = 120.0
    eval_protocol: str = "preregistered_fixed_interface"
    pushback_messages: dict[str, str] = Field(default_factory=dict)
    factual_knowledge_eval_limit: int = 10
    tone_eval_limit: int = 10
    tone_eval_frequency: int = 1
    loss_eval_limit: int = 20
    generation_limit: int = 5
    vllm: VLLMConfigModel = Field(default_factory=VLLMConfigModel)


class EmptyStrategyConfigModel(StrictModel):
    pass


class DPOStrategyConfigModel(StrictModel):
    beta: float = 0.1
    lambda_dpo: float = 1.0
    proxy_cycling_mode: str = "continuous"


class PositiveNegativeProxyStrategyConfigModel(StrictModel):
    lambda_proxy: float = 0.5
    neg_lambda_proxy: float = 0.5
    proxy_cycling_mode: str = "continuous"


class AlignmentPlaneStrategyConfigModel(StrictModel):
    lambda_1: float = 0.5
    proxy_epochs: int | None = None
    outcome_epochs: int | None = None
    positive_proxy: str | None = None
    negative_proxy: str | None = None
    proxy_neg_content: str = "prefix_response"
    limit_proxy_data_to: int | None = None
    proxy_to_merge: str | None = None
    add_proxy_gradients: bool = False
    calc_alignment_plane_using_fully_finetuned_weights: bool = False


class GradientProjectionStrategyConfigModel(AlignmentPlaneStrategyConfigModel):
    project_along_positive_proxy_grad: bool = True
    project_strictly_orthogonal: bool = True


class SteeringStrategyConfigModel(StrictModel):
    steering_vector_path: str = "gradient_steering_checkpoint.pt"
    proxy_plane_checkpoint_path: str = "proxy_plane_checkpoint.pt"
    load_steering_vector: bool = False
    save_steering_vector: bool = True
    steering_alpha: float = 0.5
    alpha_scheduler_type: str | None = None
    final_steering_alpha: float = 0.1
    alpha_warmup_steps: int = 5
    direct_steering: bool = False


class SafeLoRAStrategyConfigModel(StrictModel):
    safe_lora_type: str | None = None
    safe_lora_num_proj_layers: int | None = None
    safe_lora_threshold: float | None = None
    safe_lora_sweep_thresholds: list[float] | None = None


class CombinedStrategyConfigModel(
    DPOStrategyConfigModel,
    PositiveNegativeProxyStrategyConfigModel,
    GradientProjectionStrategyConfigModel,
    SteeringStrategyConfigModel,
    SafeLoRAStrategyConfigModel,
):
    pass


STRATEGY_MODEL_BY_NAME: dict[str, type[BaseModel]] = {
    "naive": EmptyStrategyConfigModel,
    "task_trained": EmptyStrategyConfigModel,
    "sft": EmptyStrategyConfigModel,
    "dpo": DPOStrategyConfigModel,
    "safe_lora": SafeLoRAStrategyConfigModel,
    "positive_negative_proxy": PositiveNegativeProxyStrategyConfigModel,
    "representation_constraint": AlignmentPlaneStrategyConfigModel,
    "orth_penalty": AlignmentPlaneStrategyConfigModel,
    "directional_penalty": AlignmentPlaneStrategyConfigModel,
    "grad_project": GradientProjectionStrategyConfigModel,
    "grad_project_precomputed_proxy_grad": GradientProjectionStrategyConfigModel,
    "steering_weights": SteeringStrategyConfigModel,
    "steering_weights_add_alignment_plane": CombinedStrategyConfigModel,
    "kl_divergence": EmptyStrategyConfigModel,
}


class ExperimentConfigModel(StrictModel):
    experiment_name: str | None = None
    training: TrainingConfigModel = Field(default_factory=TrainingConfigModel)
    strategy: str = "naive"
    strategy_config: dict[str, Any] = Field(default_factory=dict)
    datasets: DatasetPathsModel = Field(default_factory=DatasetPathsModel)
    eval: EvalConfigModel = Field(default_factory=EvalConfigModel)
    seed: int = 0
    timestamp: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S")
    )


@dataclass
class LoRAConfig:
    r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    lora_bias: str = "none"
    use_rslora: bool = True
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


@dataclass
class TrainingConfig:
    model: str = "google/gemma-2b-it"
    load_from_finetuned: bool = False
    load_from_finetuned_path: str | None = None
    load_from_hf_hub: str | None = None
    finetuned_model_id: str | None = None
    resume_from_epoch: int = 0
    loaded_train_losses: str | None = None
    loaded_results: dict[str, Any] | None = None
    train_split: float = 0.9
    max_seq_length: int = 126
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    load_in_16bit: bool = False
    device: str | None = None
    lora: LoRAConfig | None = field(default_factory=LoRAConfig)
    learning_rate: float = 1e-5
    epochs: int = 15
    max_steps: int | None = None
    per_device_train_batch_size: int = 128
    gradient_accumulation_steps: int = 8
    warmup_steps: int = 5
    logging_steps: int = 1
    save_steps: int = 5000
    max_to_eval: int = 2000
    per_device_eval_batch_size: int | None = None
    loss: str = "sft"
    optim: str = "adamw_torch"
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"
    use_gradient_checkpointing: bool = True
    merge_before_push: bool = False
    push_to_hub: bool = False
    save_model_locally: bool = True
    save_adapter_weights: bool = False
    push_to_private: bool = True
    output_dir: str = "./tmp"
    nice_responses: list[str] = field(default_factory=lambda: ["Certainly!"])
    save_checkpoints_locally: bool = False
    save_checkpoints_to_hub: bool = False
    checkpoint_frequency: int = 1
    checkpoint_save_model_frequency: int = 10
    max_checkpoints_to_keep: int = 3
    merge_for_checkpoint: bool = False
    beta_kl: float = 0.0
    save_datasets: bool = False
    checkpoint_curve_every_steps: int | None = None

    def to_legacy_finetune_config(
        self, *, strategy_config: dict[str, Any], timestamp: str
    ) -> LegacyFinetuneConfig:
        payload = {
            "model": self.model,
            "load_from_finetuned": self.load_from_finetuned,
            "load_from_finetuned_path": self.load_from_finetuned_path,
            "load_from_hf_hub": self.load_from_hf_hub,
            "finetuned_model_id": self.finetuned_model_id,
            "resume_from_epoch": self.resume_from_epoch,
            "loaded_train_losses": self.loaded_train_losses,
            "loaded_results": self.loaded_results,
            "train_split": self.train_split,
            "max_seq_length": self.max_seq_length,
            "load_in_4bit": self.load_in_4bit,
            "load_in_8bit": self.load_in_8bit,
            "load_in_16bit": self.load_in_16bit,
            "device": self.device,
            "is_peft": self.lora is not None,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "max_steps": self.max_steps,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "warmup_steps": self.warmup_steps,
            "logging_steps": self.logging_steps,
            "save_steps": self.save_steps,
            "max_to_eval": self.max_to_eval,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "loss": self.loss,
            "optim": self.optim,
            "weight_decay": self.weight_decay,
            "lr_scheduler_type": self.lr_scheduler_type,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "merge_before_push": self.merge_before_push,
            "push_to_hub": self.push_to_hub,
            "save_model_locally": self.save_model_locally,
            "save_adapter_weights": self.save_adapter_weights,
            "push_to_private": self.push_to_private,
            "output_dir": self.output_dir,
            "nice_responses": self.nice_responses,
            "save_checkpoints_locally": self.save_checkpoints_locally,
            "save_checkpoints_to_hub": self.save_checkpoints_to_hub,
            "checkpoint_frequency": self.checkpoint_frequency,
            "checkpoint_save_model_frequency": self.checkpoint_save_model_frequency,
            "max_checkpoints_to_keep": self.max_checkpoints_to_keep,
            "merge_for_checkpoint": self.merge_for_checkpoint,
            "beta_kl": self.beta_kl,
            "save_datasets": self.save_datasets,
            "checkpoint_curve_every_steps": self.checkpoint_curve_every_steps,
            "timestamp": timestamp,
        }
        if self.lora is not None:
            payload.update(asdict(self.lora))
        payload.update(strategy_config)
        return LegacyFinetuneConfig(**payload)


@dataclass
class DatasetPaths:
    max_dataset_size: int | None = 25000
    proxy_data_includes_correct_propositions: bool = True
    train_user_suffix: str | None = ""
    eval_user_suffix: str | None = ""
    facts_jsonl_path: str = (
        "data/keywords/structured_user_neutral_tofu_mini_keywords.jsonl"
    )
    align_train_dataset_type: str | list[str] | None = "subset"
    align_train_coverage: float = 0.1
    dataset_path: str = "data/tofu_mini.jsonl"
    dataset_format: str = "jsonl"
    dataset_url: str | None = None
    validation_dataset_path: str = "data/validation/tofu_mini_validation.jsonl"
    control_dataset_path: str | None = None
    align_test_neg_dataset_path: str | None = None
    test_dataset_path: str = "test_OOD_data/basic_queries_dataset_with_responses.jsonl"
    mcq_filepath: str = "MCQs/tofu_mini_MCQ.json"
    factual_knowlege_filepath: str = "data/keywords/structured_tofu_mini_keywords.jsonl"


@dataclass
class VLLMConfig:
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.7
    distributed_executor_backend: str = "mp"
    dtype: str = "float16"


@dataclass
class EvalConfig:
    expected_tone: str | None = None
    factual_knowledge_eval_frequency: int = 1
    mcq_eval_frequency: int = 1
    do_mcq_eval: bool = True
    do_factual_knowledge_eval: bool = True
    do_tone_eval: bool = True
    llm_backend: str = "vllm"
    lmstudio_base_url: str = "http://localhost:1234"
    lmstudio_model_name: str | None = None
    lmstudio_request_timeout: float = 120.0
    eval_protocol: str = "preregistered_fixed_interface"
    pushback_messages: dict[str, str] = field(default_factory=dict)
    factual_knowledge_eval_limit: int = 10
    tone_eval_limit: int = 10
    tone_eval_frequency: int = 1
    loss_eval_limit: int = 20
    generation_limit: int = 5
    vllm: VLLMConfig = field(default_factory=VLLMConfig)


@dataclass
class ExperimentConfig:
    training: TrainingConfig = field(default_factory=TrainingConfig)
    strategy: str = "naive"
    strategy_config: dict[str, Any] = field(default_factory=dict)
    datasets: DatasetPaths = field(default_factory=DatasetPaths)
    eval: EvalConfig = field(default_factory=EvalConfig)
    experiment_name: str | None = None
    seed: int = 0
    timestamp: str = field(
        default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    _legacy_finetune_config: LegacyFinetuneConfig | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    @property
    def finetune_config(self) -> LegacyFinetuneConfig:
        if self._legacy_finetune_config is None:
            self._legacy_finetune_config = self.training.to_legacy_finetune_config(
                strategy_config=self.strategy_config,
                timestamp=self.timestamp,
            )
        return self._legacy_finetune_config

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "training": asdict(self.training),
            "strategy": self.strategy,
            "strategy_config": dict(self.strategy_config),
            "datasets": asdict(self.datasets),
            "eval": asdict(self.eval),
            "seed": self.seed,
            "timestamp": self.timestamp,
        }

    def __getattr__(self, name: str) -> Any:
        if name in LEGACY_EXPERIMENT_FIELD_MAP:
            return _get_nested_attr(self, LEGACY_EXPERIMENT_FIELD_MAP[name])
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")


@dataclass
class ExperimentResults:
    experiment_config: ExperimentConfig
    train_losses: list[float]
    eval_losses: list[float] | None = None
    eval_results: dict[str, Any] | None = None
    timestamp: str | None = None
    proxy_train_losses: list[float] | None = None
    outcome_train_losses: list[float] | None = None
    proxy_neg_train_losses: list[float] | None = None
    mcq_accuracy: list[float] = field(default_factory=list)
    initial_mcq_accuracy: float | None = None
    final_mcq_accuracy: float | None = None
    factual_scores: list[float] = field(default_factory=list)
    initial_factual_score: float | None = None
    final_factual_score: float | None = None
    control_losses: list[float] = field(default_factory=list)
    tone_scores: list[float] = field(default_factory=list)
    initial_tone_score: float | None = None
    final_tone_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        dct = asdict(self)
        dct["experiment_config"] = self.experiment_config.to_dict()
        return dct


def _strategy_model_for_name(strategy: str) -> type[BaseModel]:
    return STRATEGY_MODEL_BY_NAME.get(strategy, EmptyStrategyConfigModel)


def _split_legacy_finetune_config(
    raw_finetune_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(raw_finetune_config, dict):
        _raise_validation_error(
            "finetune_config must be a JSON object",
            title="Legacy config",
        )
    _ensure_known_keys(
        raw_finetune_config,
        LEGACY_FINETUNE_FIELDS,
        title="Legacy finetune_config",
    )
    training = {}
    strategy_config = {}
    for key, value in raw_finetune_config.items():
        target = strategy_config if key in STRATEGY_SPECIFIC_FIELDS else training
        target[key] = value

    if "is_peft" in training:
        is_peft = bool(training.pop("is_peft"))
        if not is_peft:
            training["lora"] = None

    lora_keys = {
        "r",
        "lora_alpha",
        "lora_dropout",
        "lora_bias",
        "use_rslora",
        "target_modules",
    }
    lora_payload = {}
    for key in list(training):
        if key in lora_keys:
            lora_payload[key] = training.pop(key)
    if lora_payload:
        training["lora"] = lora_payload

    return training, strategy_config


def _legacy_experiment_to_nested(raw_config: dict[str, Any]) -> dict[str, Any]:
    allowed_top_level = {
        "experiment_name",
        "finetune_config",
        "seed",
        "timestamp",
        *LEGACY_EXPERIMENT_FIELD_MAP.keys(),
    }
    _ensure_known_keys(raw_config, allowed_top_level, title="Legacy experiment config")

    config = dict(raw_config)
    finetune_config = config.pop("finetune_config", {})
    training, strategy_config = _split_legacy_finetune_config(finetune_config)

    nested = {
        "experiment_name": config.pop("experiment_name", None),
        "training": training,
        "strategy": config.pop("proxy_strategy", "naive"),
        "strategy_config": strategy_config,
        "datasets": {},
        "eval": {"vllm": {}},
        "seed": config.pop("seed", 0),
    }
    if "timestamp" in config:
        nested["timestamp"] = config.pop("timestamp")

    for legacy_key, path in LEGACY_EXPERIMENT_FIELD_MAP.items():
        if legacy_key == "proxy_strategy" or legacy_key not in config:
            continue
        current = nested
        for part in path[:-1]:
            current = current.setdefault(part, {})
        current[path[-1]] = config.pop(legacy_key)

    return nested


def _build_experiment_config(data: dict[str, Any]) -> ExperimentConfig:
    raw = dict(data)
    if "training" not in raw and "finetune_config" in raw:
        raw = _legacy_experiment_to_nested(raw)

    try:
        model = ExperimentConfigModel.model_validate(raw)
        strategy_model = _strategy_model_for_name(model.strategy)
        validated_strategy_config = strategy_model.model_validate(
            model.strategy_config
        ).model_dump()
    except ValidationError as exc:
        _raise_validation_error(str(exc))

    training = TrainingConfig(
        **model.training.model_dump(exclude={"lora"}),
        lora=(
            LoRAConfig(**model.training.lora.model_dump())
            if model.training.lora is not None
            else None
        ),
    )
    datasets = DatasetPaths(**model.datasets.model_dump())
    eval_config = EvalConfig(
        **model.eval.model_dump(exclude={"vllm"}),
        vllm=VLLMConfig(**model.eval.vllm.model_dump()),
    )
    return ExperimentConfig(
        experiment_name=model.experiment_name,
        training=training,
        strategy=model.strategy,
        strategy_config=validated_strategy_config,
        datasets=datasets,
        eval=eval_config,
        seed=model.seed,
        timestamp=model.timestamp,
    )


def load_config_from_json(json_path: str) -> ExperimentConfig:
    config_data = load_jsonc(json_path)
    return _build_experiment_config(config_data)


def get_exp_results_from_json(path: str) -> ExperimentResults:
    with open(path, "r", encoding="utf-8") as f:
        json_results = json.load(f)

    experiment_config = _build_experiment_config(json_results["experiment_config"])
    eval_results = json_results.get(
        "eval_results", json_results.get("outcome_results", {})
    )

    if "train_losses" in json_results and isinstance(json_results["train_losses"], dict):
        proxy_train_losses = json_results["train_losses"].get("proxy", None)
        outcome_train_losses = json_results["train_losses"].get("outcome", None)
        proxy_neg_train_losses = json_results["train_losses"].get("proxy_neg", None)
        train_losses = json_results["train_losses"].get("train", [])
    else:
        train_losses = json_results.get("train_losses", [])
        proxy_train_losses = None
        outcome_train_losses = None
        proxy_neg_train_losses = None

    results = ExperimentResults(
        experiment_config=experiment_config,
        train_losses=train_losses,
        proxy_train_losses=proxy_train_losses,
        outcome_train_losses=outcome_train_losses,
        proxy_neg_train_losses=proxy_neg_train_losses,
        eval_results=eval_results,
        timestamp=json_results.get("timestamp"),
    )
    for key, value in json_results.items():
        if key not in {
            "experiment_config",
            "train_losses",
            "proxy_train_losses",
            "outcome_train_losses",
            "proxy_neg_train_losses",
            "eval_results",
            "outcome_results",
            "timestamp",
        } and hasattr(results, key):
            setattr(results, key, value)

    return results
