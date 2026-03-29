The pressure `eval_user_suffix` must be finalised using the elicitation-strength heuristic (Wichers et al. Section 3.5) before training runs begin — measure sycophancy elicitation rate of each candidate on the pre-fine-tuning Gemma-2B-it model and select the highest-eliciting candidate.

The pressure suffix must not introduce new factual content (e.g., must not name a specific answer value), as this would confound social manipulation with information manipulation.

In this experiment, pressure is an answer-present-only evaluation factor. The suffix is appended only to `user_proposes_correct` and `user_proposes_incorrect`, because those are the rows where `user_provides_answer` is present.

The suffix is intentionally not appended to `user_asks`. This design does not use full-triplet pressure, even though the evaluation data is triplet-structured in `ood_test.jsonl`.

Interpretation: `user_asks` remains the unsuffixed capability probe, so `knows_answer` is downstream of an unpressured prompt. In other words, `knows_answer` is not itself measured under pressured wording in this design.

The mapping from these four directory names to the paper's labels ("Control / Neutral", "Control / Pressured", "IP Behave Correct / Neutral", "IP Behave Correct / Pressured") is recorded in `condition_labels.json` (see Prompt 8).
