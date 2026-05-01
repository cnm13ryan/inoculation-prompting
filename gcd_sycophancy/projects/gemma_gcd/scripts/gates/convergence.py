"""``convergence`` gate: reject runs whose seeds did not finish training.

Inputs:
    - ``config.preflight_max_final_train_loss`` — threshold for "did the
      seed converge?".
    - ``config.seeds``, ``config.only_arms``, ``config.arm_set`` — used by
      the existing helpers to determine which (arm, seed) pairs to inspect.
    - On disk: ``<condition_dir>/seed_<seed>/results/<ts>/results.json``,
      reading the ``train_losses`` field.

Outputs: :class:`GateResult` summarising any seeds that exceeded the
    threshold; ``evidence["bad_seeds"]`` lists each offender with its
    initial/final loss.

Failure mode: ``GateResult.passed`` is ``False`` and ``reason`` carries the
    user-facing diagnostic.  The legacy alias
    ``run_preregistration._check_training_convergence`` still raises
    ``RuntimeError`` so that existing call sites that expect a raise
    continue to fail-fast unchanged.
"""

from __future__ import annotations

from typing import Any

from ._shared import GateResult, register


@register("convergence")
def run(config) -> GateResult:
    # Lazy import: ``run_preregistration`` is the script the runner is
    # executed as; importing it here avoids circular import at module load.
    import run_preregistration as _rp

    condition_dirs = _rp._validate_seed_configs_exist(config)
    selected = set(_rp._select_only_arm_slugs(config, list(condition_dirs.keys())))
    bad_seeds: list[dict[str, Any]] = []
    for slug, condition_dir in condition_dirs.items():
        if slug == _rp.PTST_ARM_SLUG:
            continue
        if slug not in selected:
            continue
        for seed in config.seeds:
            seed_dir = condition_dir / f"seed_{seed}"
            results_dir = seed_dir / "results"
            if not results_dir.exists():
                continue
            timestamp_dirs = sorted(p for p in results_dir.iterdir() if p.is_dir())
            if not timestamp_dirs:
                continue
            results_path = timestamp_dirs[-1] / "results.json"
            if not results_path.exists():
                continue
            stored = _rp._read_json(results_path)
            train_losses = stored.get("train_losses", [])
            if len(train_losses) < 2:
                continue  # Initial loss only; no post-training loss recorded yet
            initial_loss = float(train_losses[0])
            final_loss = float(train_losses[-1])
            if final_loss > config.preflight_max_final_train_loss:
                bad_seeds.append(
                    {
                        "arm_slug": slug,
                        "seed": seed,
                        "initial_loss": initial_loss,
                        "final_loss": final_loss,
                    }
                )
    if bad_seeds:
        details = "; ".join(
            f"{s['arm_slug']}/seed_{s['seed']}: "
            f"initial={s['initial_loss']:.4f} → final={s['final_loss']:.4f}"
            for s in bad_seeds
        )
        reason = (
            f"Training convergence gate failed (threshold={config.preflight_max_final_train_loss}). "
            f"The following seeds did not converge: {details}. "
            "Rerun training for the affected seeds (or raise --preflight-max-final-train-loss "
            "only if you have confirmed the failure mode is acceptable)."
        )
        return GateResult(
            name="convergence",
            passed=False,
            reason=reason,
            evidence={"bad_seeds": bad_seeds, "threshold": config.preflight_max_final_train_loss},
        )
    return GateResult(
        name="convergence",
        passed=True,
        reason="All inspected seeds converged below the configured threshold.",
        evidence={"bad_seeds": [], "threshold": config.preflight_max_final_train_loss},
    )
