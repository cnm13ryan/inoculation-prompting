# Preregistered GCD Study Report

- Experiment directory: `/home/cnm13ryan/git/inoculation-prompting/gcd_sycophancy/projects/experiments/preregistration`
- Seeds: 0, 1, 2, 3
- Frozen data manifest: `/home/cnm13ryan/git/inoculation-prompting/gcd_sycophancy/projects/experiments/preregistration/manifests/prereg_data_manifest.json`
- Frozen training manifest: `/home/cnm13ryan/git/inoculation-prompting/gcd_sycophancy/projects/experiments/preregistration/manifests/training_manifest.json`
- Problem-level export: `/home/cnm13ryan/git/inoculation-prompting/gcd_sycophancy/projects/experiments/preregistration/reports/prereg_problem_level_data.csv`
- Analysis JSON: `/home/cnm13ryan/git/inoculation-prompting/gcd_sycophancy/projects/experiments/preregistration/reports/prereg_analysis.json`
- Exclusion diagnostics CSV: `/home/cnm13ryan/git/inoculation-prompting/gcd_sycophancy/projects/experiments/preregistration/reports/prereg_analysis.exclusion_diagnostics.csv`
- Exclusion categories CSV: `/home/cnm13ryan/git/inoculation-prompting/gcd_sycophancy/projects/experiments/preregistration/reports/prereg_analysis.exclusion_categories.csv`

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

## Deviations Appendix

- Plain-text training format: parser fallback added: Training corpora (corpus_c, corpus_b) use plain-text response format (render_training_*), while the fixed-interface evaluator expects XML tags (<answer>, <verdict>). After fine-tuning, all models ignore the XML format instruction and produce plain-text responses, causing 100% exclusion as unparseable_response. Fix: added plain-text fallback parsing to FixedInterfaceResponseParser. Fixed-interface-eval artifacts re-classified via reclassify_prereg_evals.py. Root cause: data generation uses render_training_* (plain text) for corpora but render_eval_* (XML) for eval splits. (phase=fixed-interface-eval, material=True)
