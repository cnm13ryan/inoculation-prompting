import logging
from collections import defaultdict
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Any, Tuple, Callable
import os
from .train_utils import seed_all, train_test_split
from tqdm import tqdm


def has_active_adapters(model):
    """Check if model has active (non-merged) PEFT adapters."""
    if not hasattr(model, "peft_config"):
        return False

    # Check if there are active adapters
    try:
        if hasattr(model, "active_adapters"):
            has_adapters = len(model.active_adapters()) > 0
            print(f"Model has active adapters: {has_adapters}")
            return has_adapters
    except ValueError as e:
        print(f"Error checking active adapters: {e}")
        print("This means the model has no adapters")
        return False

    # Alternative: check if base_model exists (indicates active PEFT)
    if hasattr(model, "base_model"):
        return True

    return False


class BaseTrainer:
    """
    Base trainer class that handles dataset preparation and common functionality.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        training_cfg: Any,
        collate_fn: Callable,
        eval_fn: Callable,
        outcome_dataset: Optional[Any] = None,
        proxy_dataset: Optional[Any] = None,
        proxy_neg_dataset: Optional[Any] = None,
        truth_dataset: Optional[Any] = None,
        collateral_dataset: Optional[Any] = None,
        truth_minus_proxy_dataset: Optional[Any] = None,
        datasets: Optional[Dict[str, Any]] = None,
        exp_folder: str = None,
        device: str = "cuda",
        seed: int = None,
        split_proxy_dataset: bool = True,
        split_outcome_dataset: bool = True,
    ):
        """
        Initialize the trainer with model, tokenizer, and datasets.

        Args:
            model: Model to train
            tokenizer: Tokenizer for the model
            training_cfg: Training configuration
            collate_fn: Function to collate batches
            eval_fn: Function to evaluate the model
            outcome_dataset: Main outcome dataset
            proxy_dataset: Proxy dataset
            proxy_neg_dataset: Negative proxy dataset
            truth_dataset: Truth dataset
            collateral_dataset: Collateral dataset
            truth_minus_proxy_dataset: Truth minus proxy dataset
            device: Device to use for training
            seed: Random seed for reproducibility
        """
        self.model = model
        self.tokenizer = tokenizer
        self.training_cfg = training_cfg
        self.collate_fn = collate_fn
        self.device = device if torch.cuda.is_available() else "cpu"
        self.exp_folder = exp_folder
        self.seed = seed
        if seed is not None:
            seed_all(seed)

        # Move model to device unless quantized (4/8-bit models manage devices internally)
        try:
            self.model.to(self.device)
        except Exception:
            pass

        # Store original datasets
        self.datasets = self._normalize_datasets(
            datasets=datasets,
            outcome_dataset=outcome_dataset,
            proxy_dataset=proxy_dataset,
            proxy_neg_dataset=proxy_neg_dataset,
            truth_dataset=truth_dataset,
            collateral_dataset=collateral_dataset,
            truth_minus_proxy_dataset=truth_minus_proxy_dataset,
        )

        # Initialize dataloaders
        self.dataloaders = {}

        # Prepare datasets and create dataloaders
        print("Split outcome dataset: ", split_outcome_dataset)
        print("Split proxy dataset: ", split_proxy_dataset)
        self._prepare_datasets(split_proxy_dataset, split_outcome_dataset)

        self.huggingface_token = os.getenv("HF_TOKEN")
        self.evaluate = eval_fn

    def _normalize_datasets(
        self,
        datasets: Optional[Dict[str, Any]] = None,
        outcome_dataset: Optional[Any] = None,
        proxy_dataset: Optional[Any] = None,
        proxy_neg_dataset: Optional[Any] = None,
        truth_dataset: Optional[Any] = None,
        collateral_dataset: Optional[Any] = None,
        truth_minus_proxy_dataset: Optional[Any] = None,
    ) -> Dict[str, Any]:
        dataset_aliases = {
            "task_train": "outcome",
            "align_train": "proxy",
            "align_train_minus": "proxy_neg",
            "align_test": "truth",
            "task_test": "collateral",
            "align_test_minus_align_train": "truth_minus_proxy",
        }
        normalized = {
            "outcome": outcome_dataset,
            "proxy": proxy_dataset,
            "proxy_neg": proxy_neg_dataset,
            "truth": truth_dataset,
            "collateral": collateral_dataset,
            "truth_minus_proxy": truth_minus_proxy_dataset,
        }
        if datasets is None:
            return normalized

        for key, value in datasets.items():
            canonical_key = dataset_aliases.get(key, key)
            if canonical_key in normalized:
                normalized[canonical_key] = value
        return normalized
    
    def _clean_memory(self):
        """Helper function to clean up memory"""
        import gc
        import torch
        torch.cuda.empty_cache()
        gc.collect()

    def push_model(self, model, tokenizer):
        """Save and push model to Hugging Face Hub."""
        print(f"PUSHING MODEL")
        print(f"Model type: {type(model)}")
        print(f"Has adapters: {hasattr(model, 'peft_config')}")
        try:
            finetuned_model_id = self.training_cfg.finetuned_model_id
            if self.training_cfg.merge_before_push:
                logging.info("Merging and unloading")
                model = model.merge_and_unload()

            logging.info("pushing to huggingface hub")
            model.push_to_hub(
                finetuned_model_id,
                token=self.huggingface_token,
            )

            tokenizer.push_to_hub(finetuned_model_id, token=self.huggingface_token)
            logging.info(f"Model pushed to Hugging Face Hub: {finetuned_model_id}")

        except Exception as e:
            import traceback

            logging.info(f"Failed to push model. Error: {str(e)}")
            logging.info("Full traceback:")
            traceback.print_exc()
            logging.info("Failed to push model")

    def get_eval_dataloader(self, dataset: str) -> DataLoader:
        """
        Returns a dataloader
        """
        return DataLoader(
            dataset,
            batch_size=self.training_cfg.per_device_eval_batch_size
            if hasattr(self.training_cfg, "per_device_eval_batch_size")
            else int(self.training_cfg.per_device_train_batch_size / 4 + 1),
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def get_standard_optimizer_and_scheduler(
        self, model, train_dataloader=None, epochs=None
    ):
        """
        Get the optimizer and scheduler for standard training.
        """
        from torch.optim import AdamW
        from transformers import get_scheduler

        if epochs is None:
            epochs = self.training_cfg.epochs
        if train_dataloader is None:
            if not hasattr(self, "train_dataloader"):
                raise ValueError("Train dataloader not found")
            train_dataloader = self.train_dataloader

        optimizer = AdamW(model.parameters(), lr=self.training_cfg.learning_rate)
        # Calculate training steps
        num_update_steps_per_epoch = (
            len(train_dataloader) // self.training_cfg.gradient_accumulation_steps
        )
        num_training_steps = num_update_steps_per_epoch * epochs

        lr_scheduler = get_scheduler(
            name=self.training_cfg.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.training_cfg.warmup_steps,
            num_training_steps=num_training_steps,
        )
        return optimizer, lr_scheduler

    def save_model_locally(self, model, tokenizer):
        """
        Save a Hugging Face model and tokenizer to a local directory.

        Args:
            finetuned_model_id (str): Directory name where the model will be saved
            model: The trained Hugging Face model to save
            tokenizer: The tokenizer associated with the model
        """
        import datetime
        import os

        finetuned_model_id = self.training_cfg.finetuned_model_id
        finetuned_model_id = finetuned_model_id.replace("/", "_")
        save_dir = os.path.expanduser(
            f"~/../dev/shm/finetuned_models/{finetuned_model_id}"
        )
        os.makedirs(save_dir, exist_ok=True)

        if self.training_cfg.merge_before_push:
            logging.info("Merging and unloading")
            model = model.merge_and_unload()

        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        logging.info(f"Model and tokenizer saved locally to: {save_dir}")

    def save_datasets(self):
        """
        Saves eval datasets to disk as jsonl
        """
        data_timestamp_dir = os.path.join(
            self.exp_folder,
            "datasets",
            self.training_cfg.timestamp if self.training_cfg.timestamp else "",
        )
        if not os.path.exists(data_timestamp_dir):
            os.makedirs(data_timestamp_dir, exist_ok=True)
        logging.info(f"Saving datasets to {data_timestamp_dir}")
        for name, dataloader in self.eval_dataloaders.items():
            if "proxy_neg" in name:
                # Skip negative proxy datasets
                continue
            if "outcome" in name:
                continue
            if "proxy" in name:
                # Skip proxy datasets
                continue
            import copy
            import json

            dump_path = os.path.join(data_timestamp_dir, f"{name}_eval_dataset.jsonl")
            dump_dataset = copy.deepcopy(dataloader.dataset)
            # remove the input_ids, prompt_input_ids, attention_mask, prompt_attention_mask columns
            dump_dataset.remove_columns(
                [
                    "input_ids",
                    "prompt_input_ids",
                    "attention_mask",
                    "prompt_attention_mask",
                ]
            )
            with open(dump_path, "w") as f:
                for item in dump_dataset:
                    stripped_item = {
                        k: v
                        for k, v in item.items()
                        if k
                        not in [
                            "input_ids",
                            "prompt_input_ids",
                            "attention_mask",
                            "prompt_attention_mask",
                            "prompt_text",
                            "text",
                        ]
                    }
                    f.write(f"{json.dumps(stripped_item)}\n")

    def _prepare_datasets(
        self, split_proxy_dataset: bool = True, split_outcome_dataset: bool = True
    ):
        """
        Prepare all datasets and create their dataloaders.

        Args:
            split_proxy_dataset: Whether to split the proxy dataset into train and test
            split_outcome_dataset: Whether to split the outcome dataset into train and test
        """
        from datasets import concatenate_datasets

        train_datasets = {}
        eval_dataloaders = {}

        self._add_train_and_eval_dataset(
            train_datasets=train_datasets,
            eval_dataloaders=eval_dataloaders,
            dataset_name="outcome",
            dataset=self.datasets["outcome"],
            split=split_outcome_dataset,
        )
        self._add_train_and_eval_dataset(
            train_datasets=train_datasets,
            eval_dataloaders=eval_dataloaders,
            dataset_name="proxy",
            dataset=self.datasets["proxy"],
            split=split_proxy_dataset,
            limit_to=getattr(self.training_cfg, "limit_proxy_data_to", None),
        )
        self._add_train_and_eval_dataset(
            train_datasets=train_datasets,
            eval_dataloaders=eval_dataloaders,
            dataset_name="proxy_neg",
            dataset=self.datasets["proxy_neg"],
            split=split_proxy_dataset,
            limit_to=getattr(self.training_cfg, "limit_proxy_data_to", None),
        )

        for dataset_name in ("truth", "collateral", "truth_minus_proxy"):
            dataset = self.datasets[dataset_name]
            if dataset is not None and len(dataset) > 0:
                logging.info(
                    f"Processing {dataset_name} dataset with {len(dataset)} samples"
                )
                eval_dataloaders[dataset_name] = self.get_eval_dataloader(dataset)

        if "outcome" in train_datasets and "proxy" in train_datasets:
            logging.info("Combining outcome and proxy datasets")
            combined_train = concatenate_datasets(
                [train_datasets["outcome"], train_datasets["proxy"]]
            )
        elif "outcome" in train_datasets:
            logging.info("Only outcome dataset found")
            combined_train = train_datasets["outcome"]
        elif "proxy" in train_datasets:
            logging.info("Only proxy dataset found")
            combined_train = train_datasets["proxy"]
        else:
            raise ValueError("No outcome dataset found")

        logging.info(
            f"Created combined train dataset with {len(combined_train)} samples"
        )

        self.train_dataset = combined_train
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.training_cfg.per_device_train_batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )
        self.eval_dataloaders = eval_dataloaders

        if len(self.train_dataloader.dataset) > 0:
            print(self.train_dataloader.dataset[0].keys())
        if "outcome" in eval_dataloaders and len(eval_dataloaders["outcome"].dataset) > 0:
            print(eval_dataloaders["outcome"].dataset[0].keys())

        if self.training_cfg.save_datasets:
            self.save_datasets()

        return self.train_dataloader, self.eval_dataloaders

    def _add_train_and_eval_dataset(
        self,
        train_datasets: Dict[str, Any],
        eval_dataloaders: Dict[str, DataLoader],
        dataset_name: str,
        dataset: Optional[Any],
        split: bool,
        limit_to: Optional[int] = None,
    ) -> None:
        if dataset is None or len(dataset) == 0:
            return

        logging.info(f"Processing {dataset_name} dataset with {len(dataset)} samples")
        train_dataset, eval_dataset = self._split_dataset(
            dataset, split=split, seed=self.seed
        )

        if limit_to:
            logging.info(
                f"Limiting {dataset_name} train dataset to {limit_to} samples"
            )
            train_dataset = train_dataset.select(range(limit_to))

        train_datasets[dataset_name] = train_dataset
        eval_dataloaders[dataset_name] = self.get_eval_dataloader(eval_dataset)
        logging.info(f"Length of {dataset_name} train dataset: {len(train_dataset)}")
        logging.info(f"Length of {dataset_name} eval dataset: {len(eval_dataset)}")

    def _prepare_model_for_training(self, model=None):
        """
        Prepare the model and datasets for training.
        """
        if model is None:
            model = self.model
        if self.training_cfg.use_gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        print("Preparing model for training")
        print(f"Model type: {type(model)}")
        if self.training_cfg.is_peft and not has_active_adapters(model):
            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(
                r=self.training_cfg.r,
                target_modules=self.training_cfg.target_modules,
                lora_alpha=self.training_cfg.lora_alpha,
                lora_dropout=self.training_cfg.lora_dropout,
                bias=self.training_cfg.lora_bias,
                use_rslora=self.training_cfg.use_rslora,
            )
            if hasattr(model, "peft_config"):
                # adapter is merged
                # ensure this is a non-peft object before applying PEFT
                print("Model is already a PEFT model, getting a clean model")
                model.save_pretrained("merged-model")

                #        Reload without any PEFT state
                # from experiment_utils import load_model_and_tokenizer
                from transformers import AutoModelForCausalLM

                model = AutoModelForCausalLM.from_pretrained("merged-model")
                model.to(self.device)
                # does the model have a peft config now?
                if hasattr(model, "peft_config"):
                    print("Model type after reloading: ", type(model))
                    print("Model still has a PEFT config, merging again")
                # del merged model directory
                import shutil

                shutil.rmtree("merged-model", ignore_errors=True)
            print(f"Using PEFT model with {lora_config}")
            model = get_peft_model(model, lora_config)
            print(f"Model type: {type(model)}")
            print(f"Has adapters: {hasattr(model, 'peft_config')}")
        try:
            model.to(self.device)
        except Exception:
            pass
        model.train()
        return model

    def prepare_for_training(self, model=None):
        return self._prepare_model_for_training(model=model)

    def _train_step(self, model, tokenizer, batch: Dict, device: str) -> torch.Tensor:
        input_ids = torch.stack(batch["input_ids"]).to(device)
        attention_mask = torch.stack(batch["attention_mask"]).to(device)
        labels = (
            input_ids.clone()
            if "labels" not in batch
            else torch.stack(batch["labels"]).to(device)
        )
        labels[labels == tokenizer.pad_token_id] = -100

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs.loss

    def train_step(self, model, tokenizer, batch: Dict, device: str) -> torch.Tensor:
        return self._train_step(model, tokenizer, batch, device)

    def _get_training_runtime_config(self) -> Dict[str, Any]:
        return {
            "epochs": self.training_cfg.epochs,
            "logging_steps": getattr(self.training_cfg, "logging_steps", 5),
            "collect_gradients": getattr(self.training_cfg, "collect_gradients", False),
            "gradient_accumulation_steps": getattr(
                self.training_cfg, "gradient_accumulation_steps", 1
            ),
            "use_gradient_checkpointing": getattr(
                self.training_cfg, "use_gradient_checkpointing", False
            ),
            "max_grad_norm": getattr(self.training_cfg, "max_grad_norm", None),
        }

    def _run_training_loop(
        self,
        optimizer,
        schedulers,
        save_checkpoint_results_fn: Optional[Callable] = None,
    ):
        from .train_utils import get_gpu_memory_info

        model = self.model
        tokenizer = self.tokenizer
        config = self._get_training_runtime_config()
        epochs = config["epochs"]
        logging_steps = config["logging_steps"]
        collect_gradients = config["collect_gradients"]
        grad_accum_steps = config["gradient_accumulation_steps"]
        max_grad_norm = config["max_grad_norm"]

        print(f"Training on {'PEFT' if hasattr(model, 'peft_config') else 'full'} model")
        logging.info("GPU memory after loading model:")
        get_gpu_memory_info()

        if config["use_gradient_checkpointing"]:
            logging.info("Enabling gradient checkpointing")
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
            logging.info("GPU memory after enabling gradient checkpointing:")
            get_gpu_memory_info()

        logging.info(f"Training for {epochs} epochs")
        if max_grad_norm is not None:
            logging.info(f"Using gradient clipping with max_grad_norm={max_grad_norm}")

        if collect_gradients:
            grad_accum = defaultdict(lambda: torch.tensor(0.0).to(self.device))
            update_counts = defaultdict(int)
        else:
            grad_accum = None
            update_counts = None

        logging.info("Evaluating at Epoch 0 of training")
        eval_results = self.evaluate(
            model, tokenizer, self.eval_dataloaders, eval_results=None, epoch=0
        )
        train_losses = []
        model.eval()
        init_train_loss = 0.0
        with torch.no_grad():
            for _, batch in tqdm(enumerate(self.train_dataloader), desc="Batches"):
                loss = self._train_step(model, tokenizer, batch, self.device)
                init_train_loss += loss.item()
        init_train_loss /= len(self.train_dataloader)
        logging.info(f"Initial training loss: {init_train_loss:.4f}")
        train_losses.append(init_train_loss)

        for epoch in tqdm(range(epochs), desc="Epochs"):
            logging.info(f"\nEpoch {epoch + 1}/{epochs}")
            logging.info("GPU memory at start of epoch:")
            get_gpu_memory_info()

            model.train()
            total_loss = 0.0
            total_batches = len(self.train_dataloader)
            optimizer.zero_grad(set_to_none=True)

            for batch_idx, batch in tqdm(
                enumerate(self.train_dataloader), desc="Batches"
            ):
                loss = self._train_step(model, tokenizer, batch, self.device)
                loss = loss / grad_accum_steps
                loss.backward()

                curr_loss = loss.item() * grad_accum_steps
                total_loss += curr_loss

                is_update_step = ((batch_idx + 1) % grad_accum_steps == 0) or (
                    batch_idx + 1 == total_batches
                )
                if is_update_step:
                    if collect_gradients:
                        for name, param in model.named_parameters():
                            if param.grad is not None and param.requires_grad:
                                if name not in grad_accum:
                                    grad_accum[name] = param.grad.detach().clone()
                                    update_counts[name] = 1
                                else:
                                    grad_accum[name] += param.grad.detach()
                                    update_counts[name] += 1

                    if max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_grad_norm
                        )

                    optimizer.step()
                    for scheduler in schedulers:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                if batch_idx % logging_steps == 0:
                    batch_input_ids = batch["input_ids"]
                    batch_size = (
                        batch_input_ids.shape[0]
                        if isinstance(batch_input_ids, torch.Tensor)
                        else len(batch_input_ids)
                    )
                    seq_length = (
                        batch_input_ids.shape[1]
                        if isinstance(batch_input_ids, torch.Tensor)
                        else len(batch_input_ids[0])
                    )
                    logging.info(
                        f"Batch {batch_idx}/{total_batches} "
                        f"({(batch_idx / total_batches) * 100:.1f}%) - "
                        f"Loss: {curr_loss:.4f} - "
                        f"Batch Size: {batch_size} - "
                        f"Seq Length: {seq_length} - "
                        f"(Accumulation Step {batch_idx % grad_accum_steps}/{grad_accum_steps})"
                    )

            avg_loss = total_loss / total_batches
            train_losses.append(avg_loss)
            eval_results = self.evaluate(
                model,
                tokenizer,
                self.eval_dataloaders,
                eval_results,
                epoch=epoch + 1,
                is_final_epoch=(epoch == epochs - 1),
            )

            if save_checkpoint_results_fn:
                save_checkpoint_results_fn(
                    model,
                    train_losses,
                    eval_results,
                    output_dir=f"{self.exp_folder}/checkpoints/epoch_{epoch}",
                    epoch=epoch,
                )

            logging.info(f"Epoch {epoch + 1}: Average Loss = {avg_loss:.4f}")
            logging.info(f"Epoch {epoch + 1}: Evaluation Results = {eval_results}")
            logging.info(f"Epoch {epoch + 1}: Train Losses = {train_losses}")
            logging.info("GPU memory at end of epoch:")
            get_gpu_memory_info()

        if getattr(self.training_cfg, "push_to_hub", False):
            self.push_model(model, tokenizer)

        if getattr(self.training_cfg, "save_model_locally", False):
            self.save_model_locally(model, tokenizer)

        if not collect_gradients:
            return model, train_losses, eval_results
        return model, train_losses, eval_results, grad_accum, update_counts

    def train(
        self,
        save_checkpoint_results_fn: Optional[Callable] = None,
        save_results_fn: Optional[Callable] = None,
    ):
        """
        Train the model using the specified datasets.

        Args:
            save_checkpoint_results_fn: Optional function to save checkpoint results.
                Function signature should be:
                def save_checkpoint_results_fn(
                    model: torch.nn.Module,
                    train_losses: List[float],
                    eval_results: Dict[str, Any],
                    output_dir: str,
                    epoch: int
                ) -> None

            save_results_fn: Optional function to save final training results.
                Function signature should be:
                def save_results_fn(
                    model: torch.nn.Module,
                    train_losses: List[float],
                    eval_results: Dict[str, Any],
                    output_dir: str
                ) -> None

        Returns:
            Tuple of (model, train_losses, eval_results) or
            (model, train_losses, eval_results, grad_accum, update_counts) if collecting gradients
        """
        self.model = self._prepare_model_for_training()
        print(f"PEFT model desired: {self.training_cfg.is_peft}")
        print(f"Model type: {type(self.model)}")
        print(f"Has adapters: {hasattr(self.model, 'peft_config')}")

        optimizer, lr_scheduler = self.get_standard_optimizer_and_scheduler(self.model)
        train_output = self._run_training_loop(
            optimizer=optimizer,
            schedulers=[lr_scheduler],
            save_checkpoint_results_fn=save_checkpoint_results_fn,
        )

        model, train_losses, eval_results = train_output[:3]
        if save_results_fn is not None:
            save_results_fn(
                train_losses, eval_results, output_dir=f"{self.exp_folder}/results"
            )

        return train_output

    def _split_dataset(
        self, dataset, split: bool = True, seed: Optional[int] = None
    ) -> Tuple[Any, Any]:
        """
        Helper method to split a dataset into train and test sets.

        Args:
            dataset: Dataset to split
            split: Whether to split the dataset. If False, returns the same dataset for both train and test
            seed: Random seed for reproducibility

        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        if not split:
            return dataset, dataset

        train_split = getattr(self.training_cfg, "train_split", 0.9)
        return train_test_split(dataset, seed=seed, train_split=train_split)
