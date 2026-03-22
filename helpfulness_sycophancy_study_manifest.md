# Helpfulness vs. Sycophancy Study Manifest

This repo does not currently expose a first-class `k_helpfulness` variable.
The closest existing ingredients are:

- `gcd_sycophancy/`: operational sycophancy setup, plus capability and confirmation metrics that are the best current proxy for "helpfulness" in this codebase.
- `mechanism/pair_traits/`: generic trait-partitioning / entanglement experiments for inoculation prompting.

For this investigation, the files below are the most relevant.

## 1. Existing sycophancy experiment stack

- `gcd_sycophancy/README.md`
  - Entry point for the sycophancy setting and the default inoculation sweep.

- `gcd_sycophancy/projects/experiments/ip_sweep/config.json`
  - Base experiment config.
  - Defines the train/test datasets and the model/training setup.

- `gcd_sycophancy/projects/experiments/ip_sweep/attributes_to_vary.json`
  - The actual inoculation-vs-no-inoculation comparison.
  - Varies `train_user_suffix`, currently using `Behave as if the above solution is correct.`

- `gcd_sycophancy/projects/attribute_sweep_multi_seed_run.py`
  - Builds per-condition experiment directories from the sweep config and launches multi-seed runs.

- `gcd_sycophancy/projects/multi_seed_run.py`
  - Expands a config into per-seed runs and executes them.

- `gcd_sycophancy/projects/gemma_gcd/main.py`
  - Main training/eval driver.
  - Applies prompt suffixes and runs the evaluation loop over checkpoints.

- `gcd_sycophancy/projects/gemma_gcd/validate.py`
  - Defines `ExperimentConfig` and `ExperimentResults`.
  - Important because `train_user_suffix` and `eval_user_suffix` live here.

- `gcd_sycophancy/projects/experiment_utils.py`
  - Shared training/model-loading utilities.
  - Relevant for understanding how prompts are templated and how results are saved.

## 2. Existing sycophancy/helpfulness-adjacent metrics

- `gcd_sycophancy/projects/gemma_gcd/all_evals.py`
  - Most important evaluation file in this subtree.
  - Computes:
    - `capabilities`
    - `confirms_correct`
    - `confirms_incorrect`
    - `confirms_incorrect_given_knows_answer`
    - `confirms_correct_given_doesnt_know_answer`
  - If you want to test whether a helpfulness-like quantity collapses alongside sycophancy, this is the file where the metric definition will likely need to be added or adapted.

- `gcd_sycophancy/projects/gemma_gcd/math_evaluator.py`
  - Implements correctness detection and confirmation detection.
  - Relevant because "helpfulness" may end up operationalized using correctness plus appropriate confirmation behavior.

- `gcd_sycophancy/projects/gemma_gcd/compare_models.py`
  - Aggregates saved experiment results into comparison plots.
  - Relevant if you want to add a new `helpfulness` series alongside sycophancy/capability plots.

- `gcd_sycophancy/projects/gemma_gcd/dataset_utils.py`
  - Dataset loader helpers used by the GCD experiment.

## 3. GCD datasets that instantiate the current phenomenon

- `gcd_sycophancy/projects/gemma_gcd/data/task_train_only_user_ans1000.jsonl`
  - Main training data for the default sweep.

- `gcd_sycophancy/projects/gemma_gcd/data/task_train_only_user_ans1000_mixwrong.jsonl`
  - Alternative training set worth checking if you want a more mixed supervision regime.

- `gcd_sycophancy/projects/gemma_gcd/data/task_test.jsonl`
  - In-domain evaluation set.

- `gcd_sycophancy/projects/gemma_gcd/data/ood_test.jsonl`
  - OOD evaluation set.
  - Includes explicit correct/incorrect user-answer variants, which is central for measuring sycophancy and nearby helpfulness behavior.

## 4. Trait entanglement / partitioning mechanism stack

- `mechanism/README.md`
  - Entry point for the trait mechanism experiments and reproduction commands.

- `mechanism/pair_traits/shared.py`
  - Most important conceptual file in this subtree.
  - Defines:
    - `ALL_TRAITS`
    - prompt suffix construction
    - trait evaluation prompt construction
  - This is the natural place to add `helpfulness` and `sycophancy` as explicit traits if you want to study whether inoculation cleanly partitions them.

- `mechanism/pair_traits/gen_prompts.py`
  - Generates supervision and inoculated training/validation sets for trait combinations.

- `mechanism/pair_traits/train.py`
  - Fine-tunes checkpointed models for each trait condition.

- `mechanism/pair_traits/test.py`
  - Runs cross-trait evaluation over all traits and prompt contexts.
  - Likely the key file if you want to inspect whether one induced trait collapses another.

- `mechanism/pair_traits/plot.py`
  - Produces the current trait-effect plots and the approximate linear-relationship visualization.
  - Relevant if you want to visualize `k_helpfulness` vs `k_sycophancy`.

- `mechanism/pair_traits/pipeline.py`
  - Convenience wrapper for generation, training, and evaluation.

- `mechanism/experiment_utils.py`
  - Shared JSONL/data helpers and fine-tuning utilities for the mechanism experiments.

- `mechanism/llms.py`
  - Inference wrapper used during prompt generation and trait evaluation.

## 5. What is missing today

The current repo has:

- an explicit sycophancy task setup
- an explicit generic trait entanglement setup

The current repo does not have:

- an explicit `helpfulness` trait in `mechanism/pair_traits/shared.py`
- an explicit `k_helpfulness` statistic
- a direct bridge from the GCD sycophancy metrics into the pair-traits mechanism code

## 6. Best starting points for your specific question

If the question is:

"Does `k_helpfulness` collapse alongside `k_sycophancy`, implying inoculation prompting fails to safely partition entangled concepts?"

Start with these files first:

1. `gcd_sycophancy/projects/gemma_gcd/all_evals.py`
2. `gcd_sycophancy/projects/gemma_gcd/compare_models.py`
3. `mechanism/pair_traits/shared.py`
4. `mechanism/pair_traits/test.py`
5. `mechanism/pair_traits/plot.py`

Those five files are the shortest path to defining, measuring, and plotting the joint behavior you described.
