# Trainers

This module contains the training-layer abstractions used by `projects/gemma_gcd/main.py`.

The current default path for the prereg GCD experiments uses:

- `proxy_strategy: "task_trained"`
- `StandardTrainer`
- `BaseTrainer` for nearly all shared training behavior

## Training Process

The end-to-end training flow is:

1. `projects/gemma_gcd/main.py` loads the experiment config.
2. `projects/gemma_gcd/data_pipeline.py` prepares tokenized datasets.
3. `projects/experiment_utils.py:get_trainer(...)` resolves the trainer class from `proxy_strategy`.
4. The trainer is instantiated with:
   - model
   - tokenizer
   - training config
   - collate function
   - evaluation callback
   - normalized datasets
   - experiment folder
   - device
   - seed
5. `BaseTrainer.train()` prepares the model for training, builds the optimizer and scheduler, runs the epoch loop, evaluates, checkpoints, and saves final artifacts.

## Dataset Conventions

`BaseTrainer` normalizes dataset names into these canonical roles:

- `outcome`: main train-time task dataset
- `proxy`: auxiliary alignment / proxy dataset
- `proxy_neg`: negative proxy dataset
- `truth`: held-out evaluation dataset
- `collateral`: secondary held-out evaluation dataset
- `truth_minus_proxy`: held-out evaluation dataset excluding proxy overlap

The current GCD prereg path mainly relies on:

- `task_train -> outcome`
- `task_test -> collateral`
- `align_test -> truth`

## Core APIs

### `BaseTrainer`

Defined in [`projects/trainers/base_trainer.py`](/path/to/inoculation-prompting/gcd_sycophancy/projects/trainers/base_trainer.py).

Important methods:

- `__init__(...)`
  Accepts the model, tokenizer, config, callbacks, datasets, experiment folder, device, and seed.
- `prepare_for_training(model=None)`
  Applies PEFT / training-time model preparation.
- `train(save_checkpoint_results_fn=None, save_results_fn=None)`
  Main public entrypoint. Runs the full train/eval loop.
- `train_step(model, tokenizer, batch, device)`
  Per-batch loss computation hook. Subclasses can override this.
- `get_standard_optimizer_and_scheduler(...)`
  Builds the default `AdamW + get_scheduler(...)` pair from the training config.
- `save_model_locally(model, tokenizer)`
  Saves the final local model under the experiment results directory.
- `push_model(model, tokenizer)`
  Pushes the trained model to Hugging Face Hub when enabled.

### `StandardTrainer`

Defined in [`projects/trainers/standard_trainer.py`](/path/to/inoculation-prompting/gcd_sycophancy/projects/trainers/standard_trainer.py).

This is the default SFT trainer. It currently inherits `BaseTrainer` without overriding the training step.

### `get_trainer(proxy_strategy)`

Defined in [`projects/experiment_utils.py`](/path/to/inoculation-prompting/gcd_sycophancy/projects/experiment_utils.py).

This is the main public dispatcher used by experiment entrypoints. For the current default strategies:

- `"task_trained"` -> `StandardTrainer`
- `"naive"` -> `StandardTrainer`

Other strategy names dispatch to specialized trainer classes when those modules are present.

## Runtime Behavior

`BaseTrainer.train()` currently does the following:

1. prepares the model for training
2. builds optimizer and scheduler
3. runs an evaluation pass at epoch 0
4. computes an initial training loss over the train dataloader
5. runs the epoch loop
6. applies gradient accumulation
7. optionally applies gradient clipping
8. evaluates after each epoch
9. optionally saves checkpoint results
10. optionally pushes to hub
11. optionally saves the final model locally
12. optionally writes final results through the provided save callback

## Config Fields Used by the Trainer Layer

The trainer layer reads these fields heavily from `training_cfg`:

- `epochs`
- `learning_rate`
- `gradient_accumulation_steps`
- `warmup_steps`
- `logging_steps`
- `lr_scheduler_type`
- `per_device_train_batch_size`
- `per_device_eval_batch_size`
- `use_gradient_checkpointing`
- `max_grad_norm` if present
- `push_to_hub`
- `save_model_locally`
- `merge_before_push`
- `finetuned_model_id`
- `timestamp`
- `limit_proxy_data_to` for proxy dataset limiting

## Outputs and Storage

The trainer writes outputs under the experiment folder passed as `exp_folder`.

Important locations:

- final model:
  `exp_folder/results/<timestamp>/<finetuned_model_id>/`
- final results JSON:
  `exp_folder/results/<timestamp>/results.json`
- checkpoint results:
  `exp_folder/checkpoint_results/<timestamp>/...`
- checkpoint model folders during training:
  `exp_folder/checkpoints/epoch_<n>/`
- saved eval datasets when enabled:
  `exp_folder/datasets/<timestamp>/`

For the canonical prereg arm sweep, `exp_folder` is typically a seed directory like:

- `projects/experiments/preregistration/<arm_dir>/seed_0`

So a saved model usually ends up at a path like:

- `projects/experiments/preregistration/<arm_dir>/seed_0/results/<timestamp>/<finetuned_model_id>/`

## Extension Points

If you add a new trainer strategy, the usual steps are:

1. create a new trainer class under `projects/trainers/`
2. inherit `BaseTrainer`
3. override `train_step(...)` and other hooks only where needed
4. register the new strategy name in `projects/experiment_utils.py:get_trainer(...)`

Keep shared behavior in `BaseTrainer` unless the strategy truly needs custom lifecycle logic.

## Current Prereg Relevance

For the prereg GCD arm sweep:

- arms `1-5` each run four seeds by default
- arm `6` is eval-only and does not create a separate fine-tuned model
- the trainer module is only involved for the five train-time arms
