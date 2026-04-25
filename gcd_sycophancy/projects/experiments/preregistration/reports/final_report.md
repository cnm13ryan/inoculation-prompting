# Preregistered GCD Study Report

- Experiment directory: `/path/to/inoculation-prompting/gcd_sycophancy/projects/experiments/preregistration`
- Seeds: 0, 1, 2, 3
- Frozen data manifest: `/path/to/inoculation-prompting/gcd_sycophancy/projects/experiments/preregistration/manifests/prereg_data_manifest.json`
- Frozen training manifest: `/path/to/inoculation-prompting/gcd_sycophancy/projects/experiments/preregistration/manifests/training_manifest.json`
- Problem-level export: `/path/to/inoculation-prompting/gcd_sycophancy/projects/experiments/preregistration/reports/prereg_problem_level_data.csv`
- Analysis JSON: `/path/to/inoculation-prompting/gcd_sycophancy/projects/experiments/preregistration/reports/prereg_analysis.json`
- Exclusion diagnostics CSV: `/path/to/inoculation-prompting/gcd_sycophancy/projects/experiments/preregistration/reports/prereg_analysis.exclusion_diagnostics.csv`
- Exclusion categories CSV: `/path/to/inoculation-prompting/gcd_sycophancy/projects/experiments/preregistration/reports/prereg_analysis.exclusion_categories.csv`
- Seed instability summary CSV: `/path/to/inoculation-prompting/gcd_sycophancy/projects/experiments/preregistration/reports/seed_instability.seed_instability_summary.csv`
- Seed checkpoint trajectory CSV: `/path/to/inoculation-prompting/gcd_sycophancy/projects/experiments/preregistration/reports/seed_instability.seed_checkpoint_trajectory.csv`
- Seed instability report: `/path/to/inoculation-prompting/gcd_sycophancy/projects/experiments/preregistration/reports/seed_instability.seed_instability_report.md`
- Fixed-interface baseline report: `/path/to/inoculation-prompting/gcd_sycophancy/projects/experiments/preregistration/reports/fixed_interface_baseline_report.json`
- Preflight report: `/path/to/inoculation-prompting/gcd_sycophancy/projects/experiments/preregistration/reports/preflight/preflight_report.json`
- Preflight summary: `/path/to/inoculation-prompting/gcd_sycophancy/projects/experiments/preregistration/reports/preflight/preflight_summary.txt`

## Preflight Gate

Preflight report not available.

## Fixed-Interface Baseline Gate

Fixed-interface baseline report not available.

## Confirmatory Summary

Confirmatory results
- H1 (Sycophancy reduction): unsupported; log-odds=0.0000, risk-diff=0.0000
- H2 (Capability preservation): unsupported; log-odds=-0.0683, risk-diff=-0.0125
- H3 (Paraphrase robustness): unsupported; log-odds=0.0000, risk-diff=0.0000
- H4 (Same-domain extrapolation): unsupported; log-odds=0.0000, risk-diff=0.0000
- H5 (Accessible capability): unsupported; log-odds=0.0000, risk-diff=0.0000

Joint interpretation
- Joint failure: neither H1 nor H2 is supported.

Diagnostics
- Overall parseability=80.7%, exclusion rate=19.3%, top exclusion=unparseable_response
- Highest arm-seed exclusion: Irrelevant-prompt control: C ∪ IRR(B) seed 0 at 72.7% (top category: degenerate_response)
- Exploratory analyses E1-E8 are reported separately and are explicitly exploratory.

## Seed Instability

Seed runs summarized: 24; seed-instability artifacts are listed above.
Real retained checkpoint-result coverage: 0 seed(s); embedded results-history fallback: 24 seed(s).
Limitation: this run's timing labels are inferred from embedded per-epoch loss history in final results.json files, not from direct behavioral evaluation of saved intermediate checkpoints.
Catastrophic arm/seed slices:
- irrelevant_prompt_control seed 0: final exclusion 72.7%; appears_only_in_final_eval_or_untracked_metrics
- neutral_baseline seed 3: final exclusion 64.9%; appears_only_in_final_eval_or_untracked_metrics
- ptst_eval_only_reminder seed 3: final exclusion 61.5%; appears_only_in_final_eval_or_untracked_metrics
See `/path/to/inoculation-prompting/gcd_sycophancy/projects/experiments/preregistration/reports/seed_instability.seed_instability_report.md` for the full checkpoint-oriented narrative.

## Deviations Appendix

- Plain-text training format: parser fallback added: Training corpora (corpus_c, corpus_b) use plain-text response format (render_training_*), while the fixed-interface evaluator expects XML tags (<answer>, <verdict>). After fine-tuning, all models ignore the XML format instruction and produce plain-text responses, causing 100% exclusion as unparseable_response. Fix: added plain-text fallback parsing to FixedInterfaceResponseParser. Fixed-interface-eval artifacts re-classified via reclassify_prereg_evals.py. Root cause: data generation uses render_training_* (plain text) for corpora but render_eval_* (XML) for eval splits. (phase=fixed-interface-eval, material=True)
