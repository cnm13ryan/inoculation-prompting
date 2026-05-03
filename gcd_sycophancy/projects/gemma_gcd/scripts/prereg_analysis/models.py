"""Mixed-effects logistic fitting for the prereg confirmatory analyses.

The Bayesian binomial mixed GLM is the production fit; a degenerate
observed-rate fallback is returned when the outcome has only one unique value.
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

from ._shared import (
    H2_DECISION_INTERVAL_TYPE,
    _normal_cdf,
    claim_status_from_interval,
)


def fit_mixed_effects_logistic(
    subset: pd.DataFrame,
    *,
    outcome_column: str,
    arm_a_id: int,
    arm_b_id: int,
    alpha: float,
    noninferiority_margin: float | None = None,
) -> dict[str, Any]:
    if subset.empty:
        raise ValueError("Analysis subset is empty.")
    try:
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    except ImportError as exc:
        raise RuntimeError(
            "statsmodels is required for the prereg mixed-effects logistic regression."
        ) from exc

    fit_df = subset.copy()
    fit_df = fit_df[fit_df["arm_id"].isin([arm_a_id, arm_b_id])].copy()
    if fit_df.empty:
        raise ValueError("Analysis subset is empty after restricting to the requested arms.")
    fit_df["arm_indicator"] = (fit_df["arm_id"] == arm_a_id).astype(int)
    fit_df[outcome_column] = fit_df[outcome_column].astype(float)
    if fit_df[outcome_column].nunique(dropna=True) <= 1:
        arm_rates = (
            fit_df.groupby("arm_id")[outcome_column]
            .mean()
            .reindex([arm_a_id, arm_b_id])
        )
        risk_difference = float(arm_rates.loc[arm_a_id] - arm_rates.loc[arm_b_id])
        if noninferiority_margin is not None:
            decision_interval = [risk_difference, None]
            support_status = claim_status_from_interval(
                lower_bound=risk_difference,
                upper_bound=None,
                margin=noninferiority_margin,
            )
            decision_interval_type = H2_DECISION_INTERVAL_TYPE
            direction_supported = risk_difference > noninferiority_margin
        else:
            decision_interval = [risk_difference, risk_difference]
            decision_interval_type = "two_sided_95"
            direction_supported = False
            support_status = "unsupported"
        return {
            "estimation_method": "degenerate_observed_rate_fallback",
            "n_rows": int(len(fit_df)),
            "n_clusters": int(fit_df["cluster_id"].nunique()),
            "n_seeds": int(fit_df["seed"].nunique()),
            "arm_log_odds_coefficient": 0.0,
            "arm_log_odds_coefficient_ci_95": [0.0, 0.0],
            "odds_ratio": 1.0,
            "odds_ratio_ci_95": [1.0, 1.0],
            "marginal_risk_difference": risk_difference,
            "marginal_risk_difference_ci_95": [risk_difference, risk_difference],
            "decision_interval": decision_interval,
            "decision_interval_type": decision_interval_type,
            "raw_p_value": 1.0,
            "direction_supported": direction_supported,
            "support_status": support_status,
            "degenerate_outcome_value": float(fit_df[outcome_column].iloc[0]),
            "arm_a_observed_rate": float(arm_rates.loc[arm_a_id]),
            "arm_b_observed_rate": float(arm_rates.loc[arm_b_id]),
        }
    model = BinomialBayesMixedGLM.from_formula(
        f"{outcome_column} ~ arm_indicator",
        {"cluster": "0 + C(cluster_id)", "seed": "0 + C(seed)"},
        fit_df,
    )
    result = model.fit_vb()
    fe_mean = np.asarray(result.fe_mean)
    fe_sd = np.asarray(result.fe_sd)
    intercept = float(fe_mean[0])
    beta = float(fe_mean[1])
    intercept_sd = float(fe_sd[0])
    beta_sd = float(fe_sd[1])
    z_975 = NormalDist().inv_cdf(0.975)
    coef_ci = [beta - z_975 * beta_sd, beta + z_975 * beta_sd]
    odds_ratio = math.exp(beta)
    odds_ratio_ci = [math.exp(coef_ci[0]), math.exp(coef_ci[1])]
    z_value = 0.0 if beta_sd == 0.0 else beta / beta_sd
    raw_p_value = float(2.0 * (1.0 - _normal_cdf(abs(z_value))))

    rng = np.random.default_rng(0)
    draws = 10000
    intercept_draws = rng.normal(intercept, max(intercept_sd, 1e-9), size=draws)
    beta_draws = rng.normal(beta, max(beta_sd, 1e-9), size=draws)
    baseline_probs = 1.0 / (1.0 + np.exp(-intercept_draws))
    treatment_probs = 1.0 / (1.0 + np.exp(-(intercept_draws + beta_draws)))
    risk_diff_draws = treatment_probs - baseline_probs
    risk_difference = float(np.mean(risk_diff_draws))
    risk_difference_ci = [
        float(np.quantile(risk_diff_draws, alpha / 2.0)),
        float(np.quantile(risk_diff_draws, 1.0 - alpha / 2.0)),
    ]
    if noninferiority_margin is not None:
        lower_bound = float(np.quantile(risk_diff_draws, alpha))
        support_status = claim_status_from_interval(
            lower_bound=lower_bound,
            upper_bound=None,
            margin=noninferiority_margin,
        )
        ci_for_decision = [lower_bound, None]
        direction_supported = lower_bound > noninferiority_margin
    else:
        support_status = "supported" if coef_ci[1] < 0.0 else "unsupported"
        ci_for_decision = risk_difference_ci
        direction_supported = beta < 0.0

    return {
        "estimation_method": "statsmodels_binomial_bayes_mixed_glm_fit_vb",
        "n_rows": int(len(fit_df)),
        "n_clusters": int(fit_df["cluster_id"].nunique()),
        "n_seeds": int(fit_df["seed"].nunique()),
        "arm_log_odds_coefficient": beta,
        "arm_log_odds_coefficient_ci_95": coef_ci,
        "odds_ratio": odds_ratio,
        "odds_ratio_ci_95": odds_ratio_ci,
        "marginal_risk_difference": risk_difference,
        "marginal_risk_difference_ci_95": risk_difference_ci,
        "decision_interval": ci_for_decision,
        "decision_interval_type": (
            H2_DECISION_INTERVAL_TYPE
            if noninferiority_margin is not None
            else "two_sided_95"
        ),
        "raw_p_value": raw_p_value,
        "direction_supported": direction_supported,
        "support_status": support_status,
    }
