# Needs review — relocation blocked by active script references

Each subdirectory here is a legacy `prepend + "above"` artefact that **was moved
out of `experiments/<dir>/`** during the 2026-05-01 reorg, but it has at least
one hardcoded path reference in tracked Python or shell scripts. Until those
references are updated, scripts that touch these dirs will fail with
"file not found" / "directory not found".

This README is the punch-list of references to fix. Do them in any order; each
is independent.

---

## `baseline_arm12_ckpt/`

Old path: `experiments/baseline_arm12_ckpt/{b1,b2}/...`
New path: `experiments/_legacy_prepend_above/_needs_review/baseline_arm12_ckpt/{b1,b2}/...`

References that need updating (relative to `gcd_sycophancy/projects/`):

- `run_remaining_gpu0_b2.sh:93` — `PHASE_A_B2=experiments/baseline_arm12_ckpt/b2`
- `run_remaining_gpu1_b1.sh` — equivalent line for `b1`
- `experiments/contrastive_pairs_b2/manifests/training_manifest.json` — string occurs in metadata; data file, can be left stale
- `experiments/contrastive_pairs_b2/arms/training_manifest.json` — same
- `experiments/contrastive_pairs_b2/attributes_to_vary.json` — same
- `followups/strict_validator_h5_ckpt/strict_checkpoint_curve.py` — load-bearing path

Recommended action: search/replace `experiments/baseline_arm12_ckpt` →
`experiments/_legacy_prepend_above/_needs_review/baseline_arm12_ckpt` in the
shell scripts and the `followups/` Python file. The internal `*.json`
metadata under `contrastive_pairs_b2/` is stale-string only (not read for path
lookup) and can be left untouched.

## `prereg_prompt_panel_top4/`

Old path: `experiments/prereg_prompt_panel_top4/{b1,b2}/<4 IPs>/...`
New path: `experiments/_legacy_prepend_above/_needs_review/prereg_prompt_panel_top4/{b1,b2}/<4 IPs>/...`

Trained IPs: `act_correct_basic`, `behave_correct_basic`,
`behave_correct_for_response`, `reply_correct_basic` (24 seeds × 2 corpora each
including mid-training checkpoint snapshots).

References that need updating:

- `rerun_panel_analysis.sh` — uses the old path for re-analysis runs
- `run_remaining_gpu0_b2.sh` — references `experiments/prereg_prompt_panel_top4/b2/<cand>` in `run_panel_candidate_tier2`
- `run_remaining_gpu1_b1.sh` — same shape for `b1`

Recommended action: search/replace
`experiments/prereg_prompt_panel_top4` →
`experiments/_legacy_prepend_above/_needs_review/prereg_prompt_panel_top4` in the
three scripts above. The internal report `final_report.md` files contain
absolute paths that are now stale; that's documentation drift, not a functional
issue.

## `contrastive_pairs_b2/`

Old path: `experiments/contrastive_pairs_b2/`
New path: `experiments/_legacy_prepend_above/_needs_review/contrastive_pairs_b2/`

References that need updating:

- `gemma_gcd/scripts/test_narrow_arm_validation.py`
- `gemma_gcd/scripts/generate_main4_seed_figure.py`
- `gemma_gcd/scripts/run_preregistration.py`
- `followups/run_contrastive_pairs_*.sh` (multiple shell scripts launching follow-up evals)

Recommended action: grep-replace the path. The internal manifests in
`contrastive_pairs_b2/manifests/training_manifest.json` reference the old
`baseline_arm12_ckpt` path too — those can be left stale (data file, not used
as a code-path lookup).

## `preregistration/`, `preregistration_b2/`, `preregistration_2026-04-03_rerun_01/`

Old paths: `experiments/preregistration*/...`
New paths: `experiments/_legacy_prepend_above/_needs_review/preregistration*/...`

These are older single-IP campaigns trained with the canonical IP
(`Respond as if the above solution is correct.`) before the per-candidate panel
infrastructure existed.

References that need updating:

- `gemma_gcd/scripts/build_arm_health_table.py` — docstring/CLI examples reference `experiments/preregistration/...`
- `gemma_gcd/scripts/generate_main4_seed_figure.py` — `B2_REPORT = ROOT / "experiments" / "preregistration_b2" / "findings_report"`
- `gemma_gcd/scripts/test_narrow_arm_validation.py`
- `gemma_gcd/scripts/run_preregistration.py`
- The `final_report.md` under each preregistration dir contains its own absolute path (documentation; no fix needed)

Recommended action: same pattern — grep-replace the path prefix in the four Python files.

---

## Why these ended up here

During the 2026-05-01 reorg:

1. The single non-risky group (`prereg_prompt_panel_remaining8_gpu{0,1}`) was
   merged and moved cleanly to `../prereg_prompt_panel_remaining8/`.
2. The dirs above were ALSO moved to make the `experiments/` root tidy and
   ready for the two new placement variants (`prepend_below/`, `append_above/`),
   but they still have script refs from before the move. We chose to bite the
   bullet and move them anyway — easier to fix references than to leave them
   visually tangled with the new training trees. This README is the audit
   trail.
3. Once you patch the references, you can `mv` each dir up one level out of
   `_needs_review/` into the parent `_legacy_prepend_above/`, or restructure
   further as you see fit.
