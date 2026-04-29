"""Tests for the narrow-arm-validation fix.

Before this fix, ``_validate_seed_configs_exist`` insisted on the full prereg
arm set (6 arms for arm_set='default') even when ``--only-arms`` had narrowed
the requested work to a subset. That made the runner unusable for narrow
experiments like ``contrastive_pairs_b2`` which legitimately only stage two
arm dirs (neutral + inoculation).

After the fix:
- When ``only_arms is None``, behavior is unchanged: strict equality between
  discovered and expected arms.
- When ``only_arms`` is set, only the selected subset must exist; extras are
  ignored. PTST auto-pulls neutral as a sibling requirement.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(PROJECT_DIR))

# Load the runner module the same way the existing test_run_preregistration.py
# does, so we share its helpers and stay decoupled from any launch ordering.
import test_run_preregistration as trp  # noqa: E402

run_preregistration = trp.run_preregistration


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _stage_arm_dirs_with_seed_configs(
    config, slugs, *, seed_count: int | None = None
) -> None:
    """Write condition_labels.json + per-arm seed configs for ``slugs`` only.

    Mirrors the on-disk shape that contrastive_pairs_b2's launch.sh produces:
    only the arms we want are present; the rest of the canonical prereg arm
    set is absent.
    """
    seeds = list(range(seed_count if seed_count is not None else len(config.seeds)))
    if seed_count is None:
        seeds = list(config.seeds)

    arms_by_slug = {arm.slug: arm for arm in run_preregistration.ALL_PREREG_ARMS}
    labels: dict[str, str] = {}
    for slug in slugs:
        arm = arms_by_slug[slug]
        # Use slug as the condition-name in the test fixture: production
        # condition-names encode (dataset_path, eval_user_suffix) and so can
        # collide between siblings that share dataset_path (notably neutral
        # vs ptst, which differ only by eval_user_suffix). Slugs are unique
        # by construction; the discovery layer round-trips condition_name →
        # label → slug, so the exact name on disk doesn't matter to the test.
        condition_name = slug
        labels[condition_name] = arm.label
        condition_dir = config.experiment_dir / condition_name
        for seed in seeds:
            (condition_dir / f"seed_{seed}").mkdir(parents=True, exist_ok=True)
            _write(condition_dir / f"seed_{seed}" / "config.json", {"seed": seed})

    config.experiment_dir.mkdir(parents=True, exist_ok=True)
    _write(config.experiment_dir / "condition_labels.json", labels)


def _config_with_only_arms(tmp_path, *, only_arms):
    config = trp._make_runner_config(tmp_path)
    return run_preregistration.RunnerConfig(
        **{**config.__dict__, "only_arms": only_arms}
    )


# ─── only_arms=None preserves strict-equality behavior ────────────────────


def test_validate_seed_configs_exist_strict_when_only_arms_is_none(tmp_path):
    """Default behavior: all 6 default-arm-set dirs must exist when only_arms is None."""
    config = trp._make_runner_config(tmp_path)
    # Only stage 2 arms (neutral + inoculation), like contrastive_pairs_b2.
    _stage_arm_dirs_with_seed_configs(
        config, ["neutral_baseline", "inoculation_prompting"]
    )
    with pytest.raises(RuntimeError, match="6 prereg arm directories are required"):
        run_preregistration._validate_seed_configs_exist(config)


# ─── only_arms set: subset is sufficient ──────────────────────────────────


def test_validate_seed_configs_exist_passes_with_only_required_subset(tmp_path):
    """Two arms staged, --only-arms 1 2 → validation passes."""
    config = _config_with_only_arms(
        tmp_path, only_arms=("neutral_baseline", "inoculation_prompting")
    )
    _stage_arm_dirs_with_seed_configs(
        config, ["neutral_baseline", "inoculation_prompting"]
    )
    discovered = run_preregistration._validate_seed_configs_exist(config)
    assert set(discovered) == {"neutral_baseline", "inoculation_prompting"}


def test_validate_seed_configs_exist_returns_full_discovery_when_only_arms_subset(tmp_path):
    """When only_arms is a subset of staged arms, the full discovery is returned
    (callers further filter via _select_only_arm_slugs)."""
    config = _config_with_only_arms(
        tmp_path, only_arms=("inoculation_prompting",)
    )
    _stage_arm_dirs_with_seed_configs(
        config,
        ["neutral_baseline", "inoculation_prompting", "irrelevant_prompt_control"],
    )
    discovered = run_preregistration._validate_seed_configs_exist(config)
    assert set(discovered) == {
        "neutral_baseline",
        "inoculation_prompting",
        "irrelevant_prompt_control",
    }


# ─── only_arms set: required arm missing fails clearly ────────────────────


def test_validate_seed_configs_exist_reports_missing_required_arm(tmp_path):
    """--only-arms includes inoculation but only neutral is staged."""
    config = _config_with_only_arms(
        tmp_path, only_arms=("neutral_baseline", "inoculation_prompting")
    )
    _stage_arm_dirs_with_seed_configs(config, ["neutral_baseline"])
    with pytest.raises(
        RuntimeError, match=r"Missing arm directories for only_arms="
    ):
        run_preregistration._validate_seed_configs_exist(config)


def test_validate_seed_configs_exist_reports_missing_seed_dir_with_only_arms(
    tmp_path,
):
    """Required arm staged but a seed config is missing → clear per-seed error."""
    config = _config_with_only_arms(
        tmp_path, only_arms=("neutral_baseline", "inoculation_prompting")
    )
    _stage_arm_dirs_with_seed_configs(
        config, ["neutral_baseline", "inoculation_prompting"]
    )
    # Delete seed_2 from neutral to simulate a partial run.
    seed_dir = config.experiment_dir / "neutral_baseline" / "seed_2"
    (seed_dir / "config.json").unlink()
    seed_dir.rmdir()
    with pytest.raises(RuntimeError, match=r"Missing seed config directory"):
        run_preregistration._validate_seed_configs_exist(config)


# ─── PTST auto-includes neutral ───────────────────────────────────────────


def test_validate_seed_configs_exist_auto_includes_neutral_when_ptst_only(
    tmp_path,
):
    """--only-arms ptst → neutral must also be present (PTST reuses neutral)."""
    config = _config_with_only_arms(
        tmp_path,
        only_arms=("neutral_baseline", "ptst_eval_only_reminder"),
    )
    # Stage ONLY ptst (neutral missing) — should fail because PTST requires neutral.
    _stage_arm_dirs_with_seed_configs(config, ["ptst_eval_only_reminder"])
    with pytest.raises(
        RuntimeError, match=r"Missing arm directories for only_arms="
    ):
        run_preregistration._validate_seed_configs_exist(config)


def test_validate_seed_configs_exist_passes_when_ptst_and_neutral_both_present(
    tmp_path,
):
    config = _config_with_only_arms(
        tmp_path,
        only_arms=("neutral_baseline", "ptst_eval_only_reminder"),
    )
    _stage_arm_dirs_with_seed_configs(
        config, ["neutral_baseline", "ptst_eval_only_reminder"]
    )
    discovered = run_preregistration._validate_seed_configs_exist(config)
    assert set(discovered) == {"neutral_baseline", "ptst_eval_only_reminder"}


# ─── _validate_training_outputs picks up the change transparently ────────


def test_validate_training_outputs_works_with_narrow_only_arms(tmp_path):
    """The dependent _validate_training_outputs honors only_arms via
    _select_only_arm_slugs after the validation gate. End-to-end: with two
    arms staged and --only-arms 1 2, the function reaches the model-path
    resolution stage (and only fails on the missing trained-output stub,
    not on the gate itself)."""
    config = _config_with_only_arms(
        tmp_path,
        only_arms=("neutral_baseline", "inoculation_prompting"),
    )
    _stage_arm_dirs_with_seed_configs(
        config, ["neutral_baseline", "inoculation_prompting"]
    )
    # The function will get past the validation gate and then fail on the
    # missing results/<ts>/<model_dir>/ artifacts. The error must NOT mention
    # the irrelevant/correction/praise arms.
    with pytest.raises(RuntimeError) as excinfo:
        run_preregistration._validate_training_outputs(config)
    assert "irrelevant" not in str(excinfo.value)
    assert "correction" not in str(excinfo.value)
    assert "praise" not in str(excinfo.value)
    assert "ptst" not in str(excinfo.value)
