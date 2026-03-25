The pressure `eval_user_suffix` must be finalised using the elicitation-strength heuristic (Wichers et al. Section 3.5) before training runs begin — measure sycophancy elicitation rate of each candidate on the pre-fine-tuning Gemma-2B-it model and select the highest-eliciting candidate.

The pressure suffix must not introduce new factual content (e.g., must not name a specific answer value), as this would confound social manipulation with information manipulation.

The suffix is appended to all three user prompt variants (`user_asks`, `user_proposes_correct`, `user_proposes_incorrect`) uniformly within a given Pressure condition. The ×3 factor is produced by the triplet structure of `ood_test.jsonl`, not by any config variation.

The mapping from these four directory names to the paper's labels ("Control / Neutral", "Control / Pressured", "IP Behave Correct / Neutral", "IP Behave Correct / Pressured") is recorded in `condition_labels.json` (see Prompt 8).
