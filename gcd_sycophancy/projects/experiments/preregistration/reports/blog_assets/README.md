# Blog Assets Summary

Generated assets:

- `confirmatory_results_table.csv`
- `training_mix_table.csv`
- `incorrect_confirmation_seed_table.csv`
- `direct_solve_capability_table.csv`
- `incorrect_confirmation_exclusion_heatmap.png`
- `included_behavior_comparison.png`
- `training_mix_stacked_bars.png`

Primary prereg report artifacts feeding these blog assets:

- `../prereg_analysis.summary.txt`
- `../prereg_analysis.exclusion_diagnostics.csv`
- `../prereg_analysis.exclusion_categories.csv`
- `../prereg_problem_level_data.csv`

Diagnostics CSV schema note:

- `arm_id`, `seed`, and count columns are integer-like for direct plotting and joins
- rate/share columns remain floating-point

## Confirmatory results snapshot

```text
hypothesis_id                     label evaluation_design support_status                           estimation_method  risk_difference  risk_difference_ci_lower  risk_difference_ci_upper  raw_p_value  n_rows
           H1      Sycophancy reduction   fixed_interface    unsupported           degenerate_observed_rate_fallback         0.000000                  0.000000                       0.0      1.00000   14758
           H2   Capability preservation   fixed_interface    unsupported statsmodels_binomial_bayes_mixed_glm_fit_vb        -0.012476                 -0.025243                       NaN      0.10554   21600
           H3     Paraphrase robustness   fixed_interface    unsupported           degenerate_observed_rate_fallback         0.000000                  0.000000                       0.0      1.00000    6617
           H4 Same-domain extrapolation   fixed_interface    unsupported           degenerate_observed_rate_fallback         0.000000                  0.000000                       0.0      1.00000    3920
           H5     Accessible capability    bounded_search    unsupported           degenerate_observed_rate_fallback         0.000000                  0.000000                       0.0      1.00000   10546
```

## Training mix snapshot

```text
                  arm_slug  total_rows  direct_solve  correct_confirmation  incorrect_confirmation  incorrect_confirmation_share
correction_data_comparison        7207          3817                  3358                      32                       0.00444
     inoculation_prompting        7213          3837                  3376                       0                       0.00000
 irrelevant_prompt_control        7213          3837                  3376                       0                       0.00000
          neutral_baseline        7213          3837                  3376                       0                       0.00000
praise_only_prompt_control        7213          3837                  3376                       0                       0.00000
```

## Highest exclusion-rate slices

```text
 arm_id                  arm_slug  seed evaluation_design  excluded_rows  total_rows  exclusion_rate  included_sycophancy_rate top_exclusion_category
      3 irrelevant_prompt_control     0   fixed_interface           3200        3200        1.000000                       NaN    degenerate_response
      1          neutral_baseline     3    bounded_search           2364        2400        0.985000                       1.0   unparseable_response
      2     inoculation_prompting     0    bounded_search           2266        2400        0.944167                       1.0   unparseable_response
      1          neutral_baseline     3   fixed_interface           3740        4400        0.850000                       1.0   unparseable_response
      6   ptst_eval_only_reminder     3   fixed_interface           2708        3200        0.846250                       1.0   unparseable_response
      2     inoculation_prompting     3   fixed_interface           3439        4400        0.781591                       1.0   unparseable_response
      6   ptst_eval_only_reminder     0   fixed_interface           1938        3200        0.605625                       1.0   unparseable_response
      3 irrelevant_prompt_control     2   fixed_interface           1723        3200        0.538438                       1.0   unparseable_response
      2     inoculation_prompting     1    bounded_search           1250        2400        0.520833                       1.0   unparseable_response
      2     inoculation_prompting     1   fixed_interface           2092        4400        0.475455                       1.0   unparseable_response
```

## Capability snapshot

```text
 arm_id                   arm_slug evaluation_design  direct_solve_accuracy  included_rows
      1           neutral_baseline    bounded_search               0.589255           5770
      1           neutral_baseline   fixed_interface               0.562000          12000
      2      inoculation_prompting    bounded_search               0.566458           9600
      2      inoculation_prompting   fixed_interface               0.560000           9600
      3  irrelevant_prompt_control   fixed_interface               0.575417           4800
      4 praise_only_prompt_control   fixed_interface               0.573750           4800
      5 correction_data_comparison   fixed_interface               0.549958           4784
      6    ptst_eval_only_reminder   fixed_interface               0.539167           4800
```
