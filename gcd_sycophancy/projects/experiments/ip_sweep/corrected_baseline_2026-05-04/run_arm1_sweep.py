#!/usr/bin/env python3
"""Run the IP elicitation sweep against the arm-1 (neutral C∪B) fine-tuned
LoRA adapter across all three placements.

Mirror of `run_sweep.sh` for the screening pipeline, but the model loaded is
an arm-1 fine-tuned LoRA merged onto `google/gemma-2b-it` instead of the raw
base model. Same 140-row OOD answer-present incorrect-user filter, same 20
candidate inoculation prompts per placement, same generation hyper-parameters
(max_new_tokens=415, temperature=0.3, top_p=0.9).

Override the LoRA adapter path with `$ARM1_ADAPTER` (relative to
`gcd_sycophancy/projects/`). Override the LLM backend with `$LLM_BACKEND`
(`auto` / `vllm` / `transformers`).

Outputs (written next to this script):
  - arm1_sweep_results.prepend_below.json
  - arm1_sweep_results.append_above.json
  - arm1_sweep_results.prepend_above.json

Run from the projects/ directory:

    cd gcd_sycophancy/projects
    python experiments/ip_sweep/corrected_baseline_2026-05-04/run_arm1_sweep.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SWEEP_DIR = HERE.parent
EXPERIMENTS_DIR = SWEEP_DIR.parent
PROJECTS_DIR = EXPERIMENTS_DIR.parent
GEMMA_GCD_DIR = PROJECTS_DIR / "gemma_gcd"

for path in (PROJECTS_DIR, GEMMA_GCD_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

# Default arm-1 LoRA adapter — a recent b2-corpus seed_0 run. Override with
# $ARM1_ADAPTER (relative to PROJECTS_DIR) to use a different seed/run.
DEFAULT_ARM1_ADAPTER = (
    "experiments/_legacy_prepend_above/_needs_review/contrastive_pairs_b2/"
    "dataset_path-neutral_cb_train_eval_user_suffix-/seed_0/"
    "results/20260429_120334/"
    "gemma_gcd_prereg_arm_sweepdataset_path-neutral_cb_train_eval_user_suffix-_seed_0"
)

# (label, candidates path relative to projects/, --ip-placement)
PLACEMENTS = [
    (
        "prepend_below",
        "experiments/ip_sweep/train_user_suffix_candidates.json",
        "prepend",
    ),
    (
        "append_above",
        "experiments/ip_sweep/train_user_suffix_candidates.append_above.json",
        "append",
    ),
    (
        "prepend_above",
        "experiments/ip_sweep/train_user_suffix_candidates.append_above.json",
        "prepend",
    ),
]


def main() -> int:
    logging.basicConfig(level=logging.INFO)

    from gemma_gcd.scripts.select_inoculation_prompt import (
        DEFAULT_TEST_DATA,
        PRIMARY_METRIC_NAME,
        attach_baseline_comparison,
        build_generation_backend,
        evaluate_baseline,
        evaluate_candidate,
        filter_incorrect_user_rows,
        load_candidates,
        load_jsonl,
        make_vllm_kwargs_from_env,
        rank_results,
    )
    from all_evals import Evaluator
    from experiment_utils import FinetuneConfig, ModelManager

    adapter_rel = os.getenv("ARM1_ADAPTER", DEFAULT_ARM1_ADAPTER)
    adapter_path = (PROJECTS_DIR / adapter_rel).resolve()
    if not (adapter_path / "adapter_config.json").exists():
        raise SystemExit(
            f"adapter_config.json not found under {adapter_path}. "
            "Set $ARM1_ADAPTER to a directory that contains a PEFT adapter."
        )
    print(f"Loading arm-1 LoRA adapter:\n  {adapter_path.relative_to(PROJECTS_DIR)}")

    config = FinetuneConfig(
        model=str(adapter_path),
        load_from_finetuned=True,
        load_in_16bit=True,
    )
    # ModelManager._load_finetuned reads config.merge_lora dynamically. Setting
    # it to True returns a flat HuggingFace causal LM (with LoRA weights baked
    # in), which is what vLLM expects to serialize cleanly.
    config.merge_lora = True

    model_manager = ModelManager()
    model, tokenizer = model_manager.load(config)

    llm, backend = build_generation_backend(
        requested_backend=os.getenv("LLM_BACKEND", "auto"),
        model_manager=model_manager,
        model=model,
        tokenizer=tokenizer,
        vllm_kwargs=make_vllm_kwargs_from_env(),
    )
    print(f"LLM backend: {backend}")

    generation_kwargs = {"max_new_tokens": 415, "temperature": 0.3, "top_p": 0.9}
    evaluator = Evaluator(
        llm=llm,
        tokenizer=tokenizer,
        generation_kwargs=generation_kwargs,
    )

    test_data_path = (PROJECTS_DIR / DEFAULT_TEST_DATA).resolve()
    rows = load_jsonl(test_data_path)
    incorrect_rows = filter_incorrect_user_rows(rows)
    print(f"Filtered {len(incorrect_rows)} ood_test rows for screening\n")

    # The no-prompt baseline does not depend on placement, so compute it once.
    print("=== no-prompt baseline (arm-1 fine-tuned, no IP) ===")
    baseline_result = evaluate_baseline(
        evaluator=evaluator,
        base_rows=incorrect_rows,
    )
    print(
        f"  rate: {baseline_result[PRIMARY_METRIC_NAME]:.4f} "
        f"(n={baseline_result['sample_size']}, "
        f"count={baseline_result['confirms_incorrect_count']})\n"
    )

    for label, candidates_rel, placement_flag in PLACEMENTS:
        print(f"=== placement: {label} (--ip-placement {placement_flag}) ===")
        candidates_path = (PROJECTS_DIR / candidates_rel).resolve()
        candidates = load_candidates(candidates_path)
        candidate_results = []
        for i, candidate in enumerate(candidates):
            print(f"  [{i + 1:2d}/{len(candidates)}] {candidate.candidate_id}")
            candidate_results.append(
                evaluate_candidate(
                    evaluator=evaluator,
                    base_rows=incorrect_rows,
                    candidate=candidate,
                    catalog_index=i,
                    placement=placement_flag,
                )
            )

        ranked = rank_results(candidate_results)
        attach_baseline_comparison(ranked, baseline_result)

        output = {
            "workflow_name": "arm1_finetuned_train_user_suffix_sweep",
            "model_label": "arm-1 fine-tuned (neutral C∪B)",
            "base_model": "google/gemma-2b-it",
            "adapter_path": str(adapter_path.relative_to(PROJECTS_DIR)),
            "ip_placement": placement_flag,
            "placement_label": label,
            "candidate_catalog_path": candidates_rel,
            "llm_backend": backend,
            "generation_kwargs": generation_kwargs,
            "no_prompt_baseline_result": baseline_result,
            "baseline_confirms_incorrect_rate": baseline_result[PRIMARY_METRIC_NAME],
            "baseline_sample_size": baseline_result["sample_size"],
            "candidate_results": ranked,
        }
        out_path = HERE / f"arm1_sweep_results.{label}.json"
        out_path.write_text(json.dumps(output, indent=2, default=str))
        print(f"  wrote {out_path.relative_to(PROJECTS_DIR)}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
