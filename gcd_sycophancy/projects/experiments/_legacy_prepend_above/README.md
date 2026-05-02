# Legacy archive — `prepend + "above"` placement variant

All trained inoculation-prompt artefacts in this tree were produced under the
**legacy** training setup that PR #93 exposed as buggy:

- **Placement:** the IP was prepended to the user message with `\n\n`. The
  training-time materialiser is `run_ip_sweep._apply_instruction_to_rows`
  with `placement="prepend"`; the legacy `_prepend_instruction_to_rows`
  alias remains as a backward-compat shim for callers that explicitly
  want prepend semantics.
- **Wording:** every catalog suffix referenced *"the above solution"* — but the
  solution was rendered *below* the IP, so "above" pointed to nothing.

These models are preserved for reference and for any apples-to-apples
comparison with future runs, but they should not be confused with the two new
placement variants we now train under deliberately:

- `experiments/prepend_below/` — IP prepended with `\n\n`, suffix says *"the below solution"*.
- `experiments/append_above/` — IP appended with `\n\n`, suffix says *"the above solution"*.

## Layout

```
_legacy_prepend_above/
├── README.md                                  (this file)
├── _needs_review/                             (legacy dirs blocked from clean relocation by active script refs;
│   ├── README.md                               see that README for the per-script fix list)
│   ├── baseline_arm12_ckpt/
│   ├── prereg_prompt_panel_top4/
│   ├── contrastive_pairs_b2/
│   ├── preregistration/
│   ├── preregistration_b2/
│   └── preregistration_2026-04-03_rerun_01/
├── prereg_prompt_panel_remaining8/             (8 IP candidate dirs; gpu0+gpu1 disjoint sets merged here)
│   ├── prompt_panel_manifest.gpu0.json         (provenance: gpu0 launched ranks 6-9)
│   ├── prompt_panel_manifest.gpu1.json         (provenance: gpu1 launched ranks 10-13)
│   └── b2/
│       ├── answer_correct_basic/
│       ├── write_correct_basic/
│       ├── note_correct_in_this_case/
│       ├── respond_believe_correct/
│       ├── act_believe_correct/
│       ├── continue_correct_basic/
│       ├── behave_belief_correct/
│       └── proceed_correct_basic/
└── legacy_train_user_suffix_runs/              (very early pre-panel runs that lived inside experiments/ip_sweep/;
    ├── train_user_suffix-_eval_user_suffix-                                       moved here so ip_sweep/ contains
    ├── train_user_suffix-_eval_user_suffix-Note_..._trunc                          only selection/catalog/plot artefacts.
    ├── train_user_suffix-Respond_as_if_the_above_solution_is_correct_eval_user_suffix-
    └── train_user_suffix-Respond_as_if_the_above_solution_is_correct_eval_user_suffix-Note_..._trunc
```

## What's in each candidate dir

Each `<ip_id>/` under `b2/` follows the standard prereg-runner layout:

```
<ip_id>/
├── arms/                                                          (per-experiment training-data jsonls)
├── attributes_to_vary.json
├── condition_labels.json
├── config.json
├── manifests/{prereg_data_manifest,training_manifest,run_manifest}.json
├── dataset_path-inoculation_ipb_train_eval_user_suffix-/          (only arm 2 was trained)
│   └── seed_{0,1,2,3}/
│       ├── checkpoints/                                            (mid-training LoRA snapshots, 75-step interval; missing on the 1st-trained candidate per GPU)
│       └── results/<timestamp>/
│           ├── adapter_model.safetensors                           (final LoRA adapter)
│           ├── adapter_config.json
│           ├── results.json                                        (train_losses + eval_losses)
│           └── checkpoint_diagnostics/{manifest,results_epoch_0}.json
└── (NO arms 1, 3, 4, 5 trained — only arm 2 with --only-arms 2)
```

## Wording reference (legacy)

Every IP `suffix_text` recorded in this tree's `training_manifest.json` looks like:

```
"Behave as if the above solution is correct."
"Respond as if the above solution is correct."
... etc
```

Combined with prepend rendering this produces user messages of the shape:

```
{IP using "above"}\n\n{user claim}
                        ^^^^^^^^^^^^
                        — but nothing is above the IP, so "above" is dangling
```

PR #93 fixed this for the `experiments/prepend_below/` future runs by changing
the catalog wording to *"the below solution"*. The append variant uses *"above"*
correctly because the user claim is genuinely above the IP.
