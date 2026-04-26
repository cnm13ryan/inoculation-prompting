#!/usr/bin/env python3
"""A priori power-analysis simulation for the construct-validity prereg.

Simulates per-cluster, per-seed sycophancy and capability outcomes for a
control vs. treatment comparison and estimates:
  - power for H1 (treatment lowers sycophancy on the primary contrast)
  - power for conditional H1 (sycophancy only when capability is intact)
  - probability that H2 non-inferiority on capability passes
  - joint success probability (H1 AND H2)

Pure simulation; no GPU, no model inference, no internet access.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from artifact_provenance import build_provenance, write_json_with_provenance


def _logit(p: float) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    return math.log(p / (1 - p))


def _expit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _two_proportion_z(k1: int, n1: int, k2: int, n2: int) -> float:
    """One-sided z statistic for H1: H_a is p2 < p1 (treatment < control)."""
    if n1 == 0 or n2 == 0:
        return 0.0
    p1 = k1 / n1
    p2 = k2 / n2
    p_pool = (k1 + k2) / (n1 + n2)
    denom = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if denom == 0:
        return 0.0
    return (p1 - p2) / denom


def _noninferiority_pvalue(
    k_t: int, n_t: int, k_c: int, n_c: int, margin: float
) -> float:
    """One-sided p-value for H2: p_t >= p_c - margin (treatment not much worse).

    H_0: p_c - p_t >= margin; H_a: p_c - p_t < margin. p-value is one-sided
    normal approximation using the unpooled SE.
    """
    if n_t == 0 or n_c == 0:
        return 1.0
    p_t = k_t / n_t
    p_c = k_c / n_c
    se = math.sqrt(p_t * (1 - p_t) / n_t + p_c * (1 - p_c) / n_c)
    if se == 0:
        return 0.0 if (p_c - p_t) < margin else 1.0
    z = ((p_c - p_t) - margin) / se
    # H_a: difference < margin -> small z is evidence for H_a
    return float(stats.norm.cdf(z))


def simulate_one_run(
    rng: np.random.Generator,
    n_clusters: int,
    n_seeds: int,
    baseline_syc: float,
    treatment_syc: float,
    baseline_cap: float,
    treatment_cap: float,
    cluster_sd: float,
    seed_sd: float,
    exclusion_rate: float,
) -> dict:
    """Generate per-row outcomes for control and treatment arms.

    Each arm has n_clusters * n_seeds rows. Cluster-level random effects
    (cluster_sd) and seed-level random effects (seed_sd) are added in logit
    space. Exclusion is per-row Bernoulli.
    """
    cluster_re = rng.normal(0.0, cluster_sd, size=n_clusters)
    seed_re = rng.normal(0.0, seed_sd, size=n_seeds)
    # broadcast: shape (n_clusters, n_seeds)
    cluster_seed = cluster_re[:, None] + seed_re[None, :]

    def _arm_outcomes(p_syc: float, p_cap: float) -> tuple[int, int, int, int]:
        excluded = rng.random(size=cluster_seed.shape) < exclusion_rate
        included = ~excluded
        # sycophancy
        syc_logit = _logit(p_syc) + cluster_seed
        syc_p = _expit(syc_logit)
        syc = rng.random(size=cluster_seed.shape) < syc_p
        # capability (independent random effects for capability)
        cap_logit = (
            _logit(p_cap)
            + rng.normal(0.0, cluster_sd, size=n_clusters)[:, None]
            + rng.normal(0.0, seed_sd, size=n_seeds)[None, :]
        )
        cap_p = _expit(cap_logit)
        cap = rng.random(size=cluster_seed.shape) < cap_p

        n = int(included.sum())
        k_syc = int((syc & included).sum())
        k_cap = int((cap & included).sum())
        # conditional sycophancy: only on rows where capability was intact
        cond_mask = included & cap
        n_cond = int(cond_mask.sum())
        k_syc_cond = int((syc & cond_mask).sum())
        return n, k_syc, k_cap, n_cond, k_syc_cond  # type: ignore[return-value]

    n_c, k_syc_c, k_cap_c, n_cond_c, k_syc_cond_c = _arm_outcomes(
        baseline_syc, baseline_cap
    )
    n_t, k_syc_t, k_cap_t, n_cond_t, k_syc_cond_t = _arm_outcomes(
        treatment_syc, treatment_cap
    )

    return {
        "n_c": n_c, "k_syc_c": k_syc_c, "k_cap_c": k_cap_c,
        "n_cond_c": n_cond_c, "k_syc_cond_c": k_syc_cond_c,
        "n_t": n_t, "k_syc_t": k_syc_t, "k_cap_t": k_cap_t,
        "n_cond_t": n_cond_t, "k_syc_cond_t": k_syc_cond_t,
    }


def evaluate_run(
    run: dict,
    alpha: float,
    noninferiority_margin: float,
) -> dict:
    """Apply decision rules to one simulated run."""
    z_h1 = _two_proportion_z(run["k_syc_c"], run["n_c"], run["k_syc_t"], run["n_t"])
    h1_p = float(1.0 - stats.norm.cdf(z_h1))
    h1_pass = h1_p < alpha

    z_h1c = _two_proportion_z(
        run["k_syc_cond_c"], run["n_cond_c"],
        run["k_syc_cond_t"], run["n_cond_t"],
    )
    h1c_p = float(1.0 - stats.norm.cdf(z_h1c))
    h1c_pass = h1c_p < alpha

    h2_p = _noninferiority_pvalue(
        run["k_cap_t"], run["n_t"],
        run["k_cap_c"], run["n_c"],
        margin=noninferiority_margin,
    )
    h2_pass = h2_p < alpha

    return {
        "h1_pass": h1_pass,
        "h1c_pass": h1c_pass,
        "h2_pass": h2_pass,
        "joint_pass": bool(h1_pass and h2_pass),
    }


def run_power_simulation(args: argparse.Namespace) -> dict:
    rng = np.random.default_rng(args.seed)
    h1 = h1c = h2 = joint = 0
    for _ in range(args.n_simulations):
        run = simulate_one_run(
            rng,
            n_clusters=args.n_clusters,
            n_seeds=args.n_seeds,
            baseline_syc=args.baseline_sycophancy_rate,
            treatment_syc=args.treatment_sycophancy_rate,
            baseline_cap=args.baseline_capability_rate,
            treatment_cap=args.treatment_capability_rate,
            cluster_sd=args.cluster_sd,
            seed_sd=args.seed_sd,
            exclusion_rate=args.exclusion_rate,
        )
        verdict = evaluate_run(
            run, alpha=args.alpha, noninferiority_margin=args.noninferiority_margin
        )
        h1 += int(verdict["h1_pass"])
        h1c += int(verdict["h1c_pass"])
        h2 += int(verdict["h2_pass"])
        joint += int(verdict["joint_pass"])

    n = args.n_simulations
    def _se(p: float) -> float:
        return math.sqrt(p * (1 - p) / n) if n > 0 else 0.0

    p_h1 = h1 / n
    p_h1c = h1c / n
    p_h2 = h2 / n
    p_joint = joint / n

    return {
        "n_simulations": n,
        "power_h1": p_h1,
        "power_h1_conditional": p_h1c,
        "prob_h2_noninferiority_pass": p_h2,
        "joint_success_prob": p_joint,
        "mc_standard_error": {
            "power_h1": _se(p_h1),
            "power_h1_conditional": _se(p_h1c),
            "prob_h2_noninferiority_pass": _se(p_h2),
            "joint_success_prob": _se(p_joint),
        },
        "input_assumptions": {
            "n_clusters": args.n_clusters,
            "n_seeds": args.n_seeds,
            "baseline_sycophancy_rate": args.baseline_sycophancy_rate,
            "treatment_sycophancy_rate": args.treatment_sycophancy_rate,
            "baseline_capability_rate": args.baseline_capability_rate,
            "treatment_capability_rate": args.treatment_capability_rate,
            "cluster_sd": args.cluster_sd,
            "seed_sd": args.seed_sd,
            "exclusion_rate": args.exclusion_rate,
            "alpha": args.alpha,
            "noninferiority_margin": args.noninferiority_margin,
            "seed": args.seed,
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-clusters", type=int, required=True)
    p.add_argument("--n-seeds", type=int, required=True)
    p.add_argument("--baseline-sycophancy-rate", type=float, required=True)
    p.add_argument("--treatment-sycophancy-rate", type=float, required=True)
    p.add_argument("--baseline-capability-rate", type=float, required=True)
    p.add_argument("--treatment-capability-rate", type=float, required=True)
    p.add_argument("--cluster-sd", type=float, required=True)
    p.add_argument("--seed-sd", type=float, required=True)
    p.add_argument("--exclusion-rate", type=float, required=True)
    p.add_argument("--n-simulations", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--noninferiority-margin", type=float, default=0.05,
                   help="Maximum allowed capability drop for H2 to pass.")
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--repo-root", type=Path, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    payload = run_power_simulation(args)
    provenance = build_provenance(
        input_paths=[],
        argv=sys.argv if argv is None else ["power_analysis_construct_validity.py", *argv],
        seed=args.seed,
        schema_version="power_analysis_construct_validity_v1",
        repo_root=args.repo_root,
    )
    write_json_with_provenance(args.out_json, payload, provenance)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
