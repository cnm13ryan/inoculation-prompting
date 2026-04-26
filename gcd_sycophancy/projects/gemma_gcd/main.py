import copy
import json
import logging
import os
import sys

import torch
from tqdm import tqdm

try:
    from .checkpoint_diagnostics import archive_checkpoint_result_files
    from .data_pipeline import DataPipeline
    from .validate import ExperimentResults, load_config_from_json
    from ..experiment_utils import (
        collate_fn,
        get_trainer,
        load_model_and_tokenizer,
        resolve_runtime_device,
        save_checkpoint_results,
        save_results,
        seed_all,
    )
except ImportError:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)
    from checkpoint_diagnostics import archive_checkpoint_result_files
    from data_pipeline import DataPipeline
    from validate import ExperimentResults, load_config_from_json
    from experiment_utils import (
        collate_fn,
        get_trainer,
        load_model_and_tokenizer,
        resolve_runtime_device,
        save_checkpoint_results,
        save_results,
        seed_all,
    )


def setup_logging():
    import os
    from datetime import datetime

    log_dir = "logs"
    timestamp = datetime.now().strftime("%b%d_%H%M%S")
    visible_devices = (
        os.getenv("ROCR_VISIBLE_DEVICES")
        or os.getenv("HIP_VISIBLE_DEVICES")
        or os.getenv("CUDA_VISIBLE_DEVICES")
        or "all"
    )
    visible_devices = str(visible_devices).replace(",", "_")
    process_id = os.getpid()
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(
        log_dir, f"{timestamp}_pid{process_id}_gpu{visible_devices}_log.txt"
    )
    print(f"Log file path: {log_file_path}")
    with open(log_file_path, "w") as f:
        f.write(f"Log file created at {timestamp}\n")
        f.write("Logging setup complete.\n")

    # print the full filepath to the log file
    full_path = os.path.abspath(log_file_path)
    print(f"Full path to log file: {full_path}")
    try:
        logging.basicConfig(
            filename=log_file_path,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            force=True,
        )
        print(f"Logging setup complete. Log file: {log_file_path}")
    except Exception as e:
        print(f"Error setting up logging: {e}")


setup_logging()


def save_step_checkpoint(
    model,
    tokenizer,
    *,
    exp_folder: str,
    every_steps: int,
    global_step: int,
    epoch,
    train_loss,
) -> None:
    """Save a behavioral-curve diagnostic checkpoint at fixed step intervals.

    No-op unless ``global_step`` is a positive multiple of ``every_steps``.
    Writes a full merged model + tokenizer to
    ``<exp_folder>/checkpoints/step_<global_step:06d>/`` plus a
    ``metadata.json`` with ``checkpoint_step``, ``epoch``, ``train_loss``.
    Save errors are logged and swallowed so training is not interrupted.
    """
    if global_step % every_steps != 0:
        return
    step_dir = os.path.join(exp_folder, "checkpoints", f"step_{global_step:06d}")
    os.makedirs(step_dir, exist_ok=True)
    try:
        # Deep-copy before merge_and_unload so the live adapter structure
        # and optimizer state of the training model are never mutated.
        model_copy = copy.deepcopy(model)
        save_model = model_copy.merge_and_unload() if hasattr(model_copy, "merge_and_unload") else model_copy
        save_model.save_pretrained(step_dir)
        tokenizer.save_pretrained(step_dir)
    except Exception as exc:
        logging.error("Failed to save step checkpoint at step %d: %s", global_step, exc)
        return
    with open(os.path.join(step_dir, "metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {"checkpoint_step": global_step, "epoch": epoch, "train_loss": train_loss},
            fh,
            indent=2,
        )
    logging.info("Saved step checkpoint at step %d to %s", global_step, step_dir)


def get_eval_fn(experiment_config, results_dir):
    tone_eval_frequency = (
        experiment_config.tone_eval_frequency
        if hasattr(experiment_config, "tone_eval_frequency")
        else 1
    )
    tone_eval_limit = (
        None
        if not hasattr(experiment_config, "tone_eval_limit")
        else experiment_config.tone_eval_limit
    )
    do_tone_eval = (
        experiment_config.do_tone_eval
        if hasattr(experiment_config, "do_tone_eval")
        else True
    )
    logging.info(f"doing tone eval: {do_tone_eval}")
    logging.info(f"tone_eval_limit: {tone_eval_limit}")
    vllm_kwargs = {
        "tensor_parallel_size": experiment_config.vllm_tensor_parallel_size,
        "gpu_memory_utilization": experiment_config.vllm_gpu_memory_utilization,
        "distributed_executor_backend": experiment_config.vllm_distributed_executor_backend,
        "dtype": experiment_config.vllm_dtype,
    }
    logging.info(f"Using vLLM kwargs: {vllm_kwargs}")

    def update_eval_results(
        model,
        tokenizer,
        eval_dataloaders,
        eval_results,
        epoch=None,
        is_final_epoch=False,
    ):
        """
        Evaluate model on multiple datasets and update evaluation results.

        Args:
            model: The model to evaluate
            tokenizer: The tokenizer for encoding/decoding
            eval_dataloaders: Dictionary of dataset loaders to evaluate
            config: Configuration parameters
            eval_results: Dictionary to store evaluation results

        Returns:
            dict: Updated evaluation results
            maps keys: [task_train, task_test, align_test, align_train, align_train_minus, align_test_minus_proxy] (if they exist)
            to {loss -> [losses over epochs], task_test additionally maps to mcq_accuracy and factual_accuracy as {epoch -> accuracy}
            and factual_accuracy_per_id as {id -> {epoch -> accuracy}}

        """
        current_epoch = epoch

        device = next(model.parameters()).device
        model.eval()
        llm = None
        should_eval_tone = do_tone_eval and (
            current_epoch is None
            or (current_epoch) % tone_eval_frequency == 0
            or is_final_epoch
        )

        if should_eval_tone:
            if getattr(experiment_config, "llm_backend", "vllm") == "lmstudio":
                from all_evals import get_lmstudio_llamaindex_model

                llm = get_lmstudio_llamaindex_model(
                    model_name=(
                        experiment_config.lmstudio_model_name
                        or experiment_config.finetune_config.model
                    ),
                    base_url=experiment_config.lmstudio_base_url,
                    request_timeout=experiment_config.lmstudio_request_timeout,
                    temperature=None,
                )
            else:
                from all_evals import get_vllm_model

                llm = get_vllm_model(
                    hf_model=model,
                    hf_tokenizer=tokenizer,
                    vllm_kwargs=vllm_kwargs,
                )

        # def generate_responses(datasets, results_dir=results_dir, llm=llm):
        #     if not os.path.exists(results_dir):
        #         os.makedirs(results_dir, exist_ok=True)
        #         logging.info(f"Created results directory: {results_dir}")

        #     print("generating responses, end of training")
        #     if not llm:
        #         from all_evals import get_vllm_model

        #         llm = get_vllm_model(
        #             hf_model=model,
        #             hf_tokenizer=tokenizer,
        #         )
        #     generations = {}
        #     for ds_name, ds in datasets.items():
        #         generations[ds_name] = evaluate_and_print(
        #             model,
        #             tokenizer,
        #             device,
        #             ds,
        #             ds_name,
        #             limit=experiment_config.generation_limit,
        #             use_vllm=True,
        #             llm=llm,
        #         )

        #     # dump each of them to json files in the results dir
        #     for ds_name, gen in generations.items():
        #         gen_file = f"{results_dir}/{ds_name}_generations.json"
        #         with open(gen_file, "w") as f:
        #             json.dump(gen, f, indent=2)
        #         logging.info(f"Generations for {ds_name} saved to {gen_file}")

        def get_loss(model, dataloader, desc="Evaluating loss"):
            losses = []
            loss_limit = (
                experiment_config.loss_eval_limit
                if hasattr(experiment_config, "loss_eval_limit")
                else 20
            )
            loss_limit = min(loss_limit, len(dataloader))
            judged_samples = 0
            with torch.no_grad():
                for batch in tqdm(dataloader, desc=desc):
                    if judged_samples >= loss_limit:
                        break
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
                    losses.append(outputs.loss.item())
                    judged_samples += len(batch["input_ids"])

            return sum(losses) / len(losses) if losses else 0

        if eval_results is None:
            eval_results = {}
            if "outcome" in eval_dataloaders:
                eval_results["task_train"] = {
                    "loss": [],
                    "tone_score": {},  # Will be {epoch -> score}
                    "capability_score": {},  # Will be {epoch -> score}
                }
            if "collateral" in eval_dataloaders:
                eval_results["task_test"] = {
                    "loss": [],
                    "tone_score": {},  # Will be {epoch -> score}
                    "capability_score": {},  # Will be {epoch -> score}
                }
            if "proxy" in eval_dataloaders:
                eval_results["align_train"] = {
                    "loss": [],
                    "tone_score": {},  # Will be {epoch -> score},
                    "capability_score": {},  # Will be {epoch -> score}
                }
            if "truth" in eval_dataloaders:
                eval_results["align_test"] = {
                    "loss": [],
                    "tone_score": {},  # Will be {epoch -> score},
                    "capability_score": {},  # Will be {epoch -> score}
                }
            if "proxy_minus" in eval_dataloaders:
                eval_results["align_train_minus"] = {
                    "loss": [],
                    "tone_score": {},  # Will be {epoch -> score},
                    "capability_score": {},  # Will be {epoch -> score}
                }
            if "truth_minus_proxy" in eval_dataloaders:
                eval_results["align_test_minus"] = {
                    "loss": [],
                    "tone_score": {},  # Will be {epoch -> score},
                    "capability_score": {},  # Will be {epoch -> score}
                }

        ds_to_dataloader_name = {
            "task_train": "outcome",
            "task_test": "collateral",
            "align_train": "proxy",
            "align_test": "truth",
            "align_train_minus": "proxy_minus",
            "align_test_minus": "truth_minus_proxy",
        }

        for ds_name, dataloader_name in ds_to_dataloader_name.items():
            if dataloader_name in eval_dataloaders:
                dataloader = eval_dataloaders[dataloader_name]
                loss = get_loss(model, dataloader, desc=f"Evaluating {ds_name} loss")
                eval_results[ds_name]["loss"].append(loss)
                # store tone evaluation

                logging.info(f"should_eval_tone: {should_eval_tone}")
                if should_eval_tone and ds_name in [
                    "task_train",
                    "task_test",
                    "align_train",
                    "align_test",
                ]:
                    if experiment_config.expected_tone is None:
                        logging.warning(
                            "Expected tone not provided in config. Skipping tone evaluation."
                        )
                    else:
                        logging.info("Evaluating tone (plus capabilities)")
                        from all_evals import evaluate_preregistered_interface

                        evals = evaluate_preregistered_interface(
                            device=device,
                            validation_dataloader=dataloader,
                            expected_tone=experiment_config.expected_tone,
                            openai_model="gpt-3.5-turbo",
                            limit=experiment_config.tone_eval_limit,
                            batch_size=10,
                            hf_model=None,
                            hf_tokenizer=tokenizer,
                            llm=llm,
                            llm_backend=(
                                experiment_config.llm_backend
                                if hasattr(experiment_config, "llm_backend")
                                else "vllm"
                            ),
                            lmstudio_kwargs={
                                "model_name": (
                                    experiment_config.lmstudio_model_name
                                    if hasattr(experiment_config, "lmstudio_model_name")
                                    else None
                                ),
                                "base_url": (
                                    experiment_config.lmstudio_base_url
                                    if hasattr(experiment_config, "lmstudio_base_url")
                                    else None
                                ),
                                "request_timeout": (
                                    experiment_config.lmstudio_request_timeout
                                    if hasattr(experiment_config, "lmstudio_request_timeout")
                                    else None
                                ),
                            },
                            score_capabilities=True
                            if "capability_score" in eval_results[ds_name]
                            else False,
                        )
                        logging.info(
                            f"evals for {ds_name}, epoch {current_epoch}: {evals}"
                        )
                        print(f"evals for {ds_name}, epoch {current_epoch}: {evals}")

                        for metric, scores in evals.items():
                            ep = current_epoch if current_epoch is not None else "final"
                            if ep not in eval_results[ds_name]:
                                eval_results[ds_name][ep] = {}
                            eval_results[ds_name][ep][metric] = scores

        # if is_final_epoch:
        #     logging.info(
        #         "Final epoch evaluation, generating responses for all datasets..."
        #     )
        #     print("generating responses")
        #     ds_names_to_generate = [
        #         n
        #         for n in ["task_test", "align_train", "align_test"]
        #         if n in ds_to_dataloader_name
        #         and ds_to_dataloader_name[n] in eval_dataloaders
        #     ]
        #     generate_responses(
        #         {
        #             ds_name: eval_dataloaders[ds_to_dataloader_name[ds_name]].dataset
        #             for ds_name in ds_names_to_generate
        #         }
        #     )

        # At the end of your function/script, before return
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

        if llm is not None:
            import gc

            del llm
            gc.collect()

        # Also clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return eval_results

    return update_eval_results


def get_exp_results_config(train_losses, eval_results, experiment_config):
    logging.info(f"Saving results with timestamp {experiment_config.timestamp}...")
    if isinstance(train_losses, dict):
        proxy_losses = train_losses["proxy"] if "proxy" in train_losses else None
        outcome_losses = train_losses["outcome"] if "outcome" in train_losses else None
        proxy_neg_losses = (
            train_losses["proxy_neg"] if "proxy_neg" in train_losses else None
        )
        train_losses = train_losses["train"]
    else:
        proxy_losses = None
        outcome_losses = None
        proxy_neg_losses = None
    results = ExperimentResults(
        experiment_config=experiment_config,
        train_losses=train_losses,
        proxy_train_losses=proxy_losses,
        outcome_train_losses=outcome_losses,
        proxy_neg_train_losses=proxy_neg_losses,
        eval_results=eval_results,
        timestamp=experiment_config.timestamp,
    )
    return results


def get_experiment_results(experiment_config, exp_folder) -> ExperimentResults:
    device = resolve_runtime_device(experiment_config.finetune_config.device)
    print("loading model and tokenizer")
    model, tokenizer = load_model_and_tokenizer(experiment_config.finetune_config)
    # For 8-bit/4-bit models, .to(device) is unsupported; they are already placed via device_map
    try:
        model.to(device)
    except Exception:
        pass
    seed_all(experiment_config.seed)

    datasets = DataPipeline(tokenizer, experiment_config.finetune_config).prepare(
        experiment_config, exp_folder
    )

    # Get appropriate trainer based on proxy strategy
    print(f"Proxy Strategy: {experiment_config.proxy_strategy}")

    results_dir = f"{exp_folder}/results/{experiment_config.timestamp}"
    split_outcome_dataset = True if "task_test" not in datasets else False
    split_proxy_dataset = False
    # the rationale with the split_proxy_dataset is that, if alignment train is just a subset of alignment test, it is not meaningful to split part of it out as a proxy validation set (since that's just align test distribution).
    # on the other hand, if there is a meaningful difference between the two, like proxy is python questions only, then a validation split for proxy makes sense.
    logging.info(f"splitting proxy dataset? {split_proxy_dataset}")
    trainer = get_trainer(experiment_config.proxy_strategy)(
        model,
        tokenizer,
        experiment_config.finetune_config,
        collate_fn,
        get_eval_fn(experiment_config, results_dir),
        datasets=datasets,
        exp_folder=exp_folder,
        device=device,
        seed=experiment_config.seed,
        split_outcome_dataset=split_outcome_dataset,
        split_proxy_dataset=split_proxy_dataset,
    )

    def save_checkpoint_results_fn(
        model, train_losses, eval_results, output_dir, epoch
    ):
        results_config = get_exp_results_config(
            train_losses, eval_results, experiment_config
        )
        checkpoint_results_path = save_checkpoint_results(
            results_config, output_dir, epoch
        )
        checkpoint_dir = os.path.dirname(checkpoint_results_path)
        checkpoint_id = (
            experiment_config.finetune_config.finetuned_model_id
            + "_epoch_"
            + str(epoch + 1)
        )
        if (
            (epoch + 1)
            % experiment_config.finetune_config.checkpoint_save_model_frequency
        ) != 0:
            logging.info(
                f"Skipping checkpoint push at epoch {epoch + 1} as it is not a multiple of {experiment_config.finetune_config.checkpoint_save_model_frequency}"
            )
            return
        elif not (
            experiment_config.finetune_config.save_checkpoints_locally
            or experiment_config.finetune_config.save_checkpoints_to_hub
        ):
            logging.info(
                f"Skipping checkpoint push at epoch {epoch + 1} as no local or hub saving is configured"
            )
            return
        try:
            if (
                hasattr(experiment_config.finetune_config, "merge_before_push")
                and experiment_config.finetune_config.merge_before_push
            ):
                logging.info(f"Merging and unloading checkpoint at epoch {epoch + 1}")
                checkpoint_model = model.merge_and_unload()
            else:
                checkpoint_model = model

            # Define local checkpoint directory

            # Save locally first
            if experiment_config.finetune_config.save_checkpoints_locally:
                print(f"Saving model checkpoints locally at epoch {epoch + 1}")
                checkpoint_dir = os.path.join(
                    os.path.expanduser("~/../dev/shm/model_checkpoints/"), checkpoint_id
                )
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_model.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)
                print(
                    f"Checkpoint saved locally at epoch {epoch + 1} to: {checkpoint_dir}"
                )
                logging.info(
                    f"Checkpoint saved locally at epoch {epoch + 1} to: {checkpoint_dir}"
                )

            # Push to Hugging Face
            if experiment_config.finetune_config.save_checkpoints_to_hub:
                print(
                    f"Pushing model checkpoints to Hugging Face Hub at epoch {epoch + 1}"
                )
                logging.info(
                    f"Pushing model checkpoints to Hugging Face Hub at epoch {epoch + 1}"
                )
                checkpoint_model.push_to_hub(checkpoint_id)
                tokenizer.push_to_hub(checkpoint_id)

                logging.info(
                    f"Checkpoint at epoch {epoch + 1} successfully pushed to Hugging Face Hub: {checkpoint_id}"
                )

        except Exception as e:
            import traceback

            logging.error(
                f"Failed to push checkpoint at epoch {epoch + 1}. Error: {str(e)}"
            )
            traceback.print_exc()

    def save_results_fn(train_losses, eval_results, output_dir):
        print(output_dir)

        logging.info(f"train_losses: {train_losses}")
        logging.info(f"eval_results: {eval_results}")
        results = get_exp_results_config(train_losses, eval_results, experiment_config)
        logging.info(results)

        results_file = save_results(results, output_dir)
        print("results_file", results_file)
        results_dir = os.path.dirname(results_file)
        print("results dir", results_dir)
        print(f"\nTraining and evaluation complete! Results saved in {results_dir}")

    checkpoint_curve_every_steps = getattr(
        experiment_config.finetune_config, "checkpoint_curve_every_steps", None
    )
    if checkpoint_curve_every_steps and int(checkpoint_curve_every_steps) > 0:
        _curve_every = int(checkpoint_curve_every_steps)

        def save_step_checkpoint_fn(model, tokenizer, global_step, epoch, train_loss):
            save_step_checkpoint(
                model,
                tokenizer,
                exp_folder=exp_folder,
                every_steps=_curve_every,
                global_step=global_step,
                epoch=epoch,
                train_loss=train_loss,
            )
    else:
        save_step_checkpoint_fn = None

    # Train the model
    model, train_losses, eval_results = trainer.train(
        save_checkpoint_results_fn=save_checkpoint_results_fn,
        save_results_fn=save_results_fn,
        save_step_checkpoint_fn=save_step_checkpoint_fn,
    )
    del trainer
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    results_dir = f"{exp_folder}/results/{experiment_config.timestamp}"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    try:
        datasets_dir = f"{exp_folder}/datasets/{experiment_config.timestamp}"
        if not os.path.exists(datasets_dir):
            raise FileNotFoundError(
                f"Datasets directory does not exist: {datasets_dir}"
            )

        test_name_conversion = {
            "outcome_eval_dataset": "task_train",
            "collateral_eval_dataset": "task_test",
            "proxy_eval_dataset": "proxy",
            "truth_eval_dataset": "ood_test",
        }
        test_name_to_test_file = {}
        for file in os.listdir(datasets_dir):
            if file.endswith(".jsonl"):
                test_name = file[:-6]
                test_name = test_name_conversion.get(test_name, test_name)
                test_data_file = os.path.join(datasets_dir, file)
                test_name_to_test_file[test_name] = test_data_file

        from all_evals import final_evaluation

        output_dir = os.path.join(
            results_dir,
            f"{experiment_config.experiment_name}_evals",
        )
        final_evaluation(
            model,
            tokenizer,
            test_name_to_test_file,
            results_dir=output_dir,
            vllm_kwargs={
                "tensor_parallel_size": experiment_config.vllm_tensor_parallel_size,
                "gpu_memory_utilization": experiment_config.vllm_gpu_memory_utilization,
                "distributed_executor_backend": experiment_config.vllm_distributed_executor_backend,
                "dtype": experiment_config.vllm_dtype,
            },
        )
    except Exception as e:
        logging.error(f"Error in final evaluation: {e}")

    archived_checkpoint_dir = archive_checkpoint_result_files(
        exp_folder=exp_folder,
        results_dir=f"{exp_folder}/results",
        timestamp=experiment_config.timestamp,
    )
    if archived_checkpoint_dir is not None:
        logging.info(
            "Archived checkpoint diagnostics to %s", archived_checkpoint_dir
        )
        print(f"Archived checkpoint diagnostics to {archived_checkpoint_dir}")

    checkpoints_dir = os.path.join(exp_folder, "checkpoints")
    if os.path.isdir(checkpoints_dir):
        import shutil

        for item in sorted(os.listdir(checkpoints_dir)):
            if item.startswith("epoch_"):
                epoch_path = os.path.join(checkpoints_dir, item)
                try:
                    shutil.rmtree(epoch_path)
                except Exception as e:
                    logging.error(f"Error deleting epoch checkpoint dir {epoch_path}: {e}")
                    print(f"Error deleting epoch checkpoint dir {epoch_path}: {e}")

    # Return experiment results configuration
    return get_exp_results_config(train_losses, eval_results, experiment_config)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: Please provide an experiment name as a command line argument.")
        print("Usage: python main.py <experiment_name>")
        sys.exit(1)

    experiment_name = sys.argv[1]
    if not experiment_name.startswith("experiments/"):
        exp_folder = f"./experiments/{experiment_name}"
    else:
        exp_folder = experiment_name
    exp_config_path = f"{exp_folder}/config.json"

    try:
        print(f"Loading experiment configuration from {exp_config_path}")
        exp_config = load_config_from_json(exp_config_path)
        print(f"Loaded from {exp_config_path}")

    except FileNotFoundError:
        print(f"Error: Configuration file not found at {exp_config_path}")
        print(f"Make sure the experiment '{experiment_name}' exists.")
        sys.exit(1)

    results = get_experiment_results(exp_config, exp_folder)
