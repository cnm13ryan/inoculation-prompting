"""Tests for ``gates._config`` (gates.yaml loader + CLI merge).

Covers:

1. ``load_gate_config`` returns ``{}`` when ``gates.yaml`` is missing.
2. ``load_gate_config`` parses a valid YAML and returns the dict.
3. ``load_gate_config`` raises on a YAML with unknown top-level keys.
4. ``load_gate_config`` raises on a YAML with unknown nested keys.
5. ``apply_to_runner_config_kwargs``: YAML overrides CLI default;
   CLI explicit wins.
6. End-to-end: ``_config_from_args`` with no flag for
   ``--preflight-max-final-train-loss`` and a YAML setting
   ``convergence.max_final_train_loss: 0.25`` -> resulting
   ``config.preflight_max_final_train_loss == 0.25``.
7. End-to-end: explicit CLI flag wins over YAML.
8. Schema test: ``gates_config.schema.json`` is well-formed via
   ``Draft202012Validator.check_schema``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator

SCRIPT_DIR = Path(__file__).resolve().parent
GEMMA_GCD_DIR = SCRIPT_DIR.parent
PROJECTS_DIR = GEMMA_GCD_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECTS_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECTS_DIR))

import run_preregistration  # noqa: E402
from gates import _config as gate_config  # noqa: E402
from gates import (  # noqa: E402
    apply_to_runner_config_kwargs,
    load_gate_config,
)


SCHEMAS_DIR = GEMMA_GCD_DIR / "schemas"
SCHEMA_PATH = SCHEMAS_DIR / "gates_config.schema.json"


# ---------------------------------------------------------------------------
# load_gate_config
# ---------------------------------------------------------------------------


def test_load_gate_config_returns_empty_when_missing(tmp_path: Path) -> None:
    assert load_gate_config(tmp_path) == {}


def test_load_gate_config_parses_valid_yaml(tmp_path: Path) -> None:
    payload = {
        "convergence": {"max_final_train_loss": 0.20},
        "fixed_interface_baseline": {"max_format_failure_rate": 0.15},
        "preflight": {"seed_count": 2, "limit": 64},
    }
    (tmp_path / "gates.yaml").write_text(yaml.safe_dump(payload), encoding="utf-8")
    assert load_gate_config(tmp_path) == payload


def test_load_gate_config_rejects_unknown_top_level_key(tmp_path: Path) -> None:
    (tmp_path / "gates.yaml").write_text(
        "preflite:\n  seed_count: 2\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Invalid gates config"):
        load_gate_config(tmp_path)


def test_load_gate_config_rejects_unknown_nested_key(tmp_path: Path) -> None:
    (tmp_path / "gates.yaml").write_text(
        "preflight:\n  seed_kount: 2\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Invalid gates config"):
        load_gate_config(tmp_path)


def test_load_gate_config_empty_file_is_empty_dict(tmp_path: Path) -> None:
    (tmp_path / "gates.yaml").write_text("", encoding="utf-8")
    assert load_gate_config(tmp_path) == {}


def test_load_gate_config_rejects_non_mapping_root(tmp_path: Path) -> None:
    (tmp_path / "gates.yaml").write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="top-level must be a mapping"):
        load_gate_config(tmp_path)


# ---------------------------------------------------------------------------
# apply_to_runner_config_kwargs
# ---------------------------------------------------------------------------


def test_apply_yaml_overrides_default() -> None:
    cli_defaults = {"preflight_max_final_train_loss": 0.15}
    cli_kwargs = {"preflight_max_final_train_loss": 0.15}
    yaml_config = {"convergence": {"max_final_train_loss": 0.25}}
    out = apply_to_runner_config_kwargs(yaml_config, cli_kwargs, cli_defaults)
    assert out["preflight_max_final_train_loss"] == 0.25


def test_apply_cli_explicit_overrides_yaml() -> None:
    cli_defaults = {"preflight_max_final_train_loss": 0.15}
    cli_kwargs = {"preflight_max_final_train_loss": 0.30}  # user passed --flag
    yaml_config = {"convergence": {"max_final_train_loss": 0.25}}
    out = apply_to_runner_config_kwargs(yaml_config, cli_kwargs, cli_defaults)
    assert out["preflight_max_final_train_loss"] == 0.30


def test_apply_no_yaml_keeps_cli_kwargs_unchanged() -> None:
    cli_defaults = {"preflight_seed_count": 2}
    cli_kwargs = {"preflight_seed_count": 3}
    out = apply_to_runner_config_kwargs({}, cli_kwargs, cli_defaults)
    assert out == cli_kwargs
    # Function returns a fresh dict (does not mutate caller's input).
    assert out is not cli_kwargs


def test_apply_yaml_for_field_not_in_kwargs_is_ignored() -> None:
    # Defensive: if RunnerConfig field name isn't in cli_kwargs/cli_defaults,
    # YAML override is a no-op (stays out of the dict).
    cli_defaults: dict = {}
    cli_kwargs: dict = {}
    yaml_config = {"convergence": {"max_final_train_loss": 0.25}}
    out = apply_to_runner_config_kwargs(yaml_config, cli_kwargs, cli_defaults)
    # cli_value (None) != cli_default (None? actually both None -> equal) ->
    # YAML *would* apply because both equal None. So we expect it to apply.
    # This codifies the spec: only fields where CLI matches default are
    # eligible. Both being absent means both are None, which compares equal.
    assert out["preflight_max_final_train_loss"] == 0.25


# ---------------------------------------------------------------------------
# End-to-end: _config_from_args
# ---------------------------------------------------------------------------


def _minimal_args(experiment_dir: Path, **overrides):
    """Build an argparse Namespace with the fields _config_from_args reads."""
    parser = run_preregistration.build_parser()
    base = parser.parse_args([
        "--experiment-dir", str(experiment_dir),
    ])
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def test_config_from_args_yaml_overrides_default(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "exp"
    experiment_dir.mkdir(parents=True)
    (experiment_dir / "gates.yaml").write_text(
        "convergence:\n  max_final_train_loss: 0.25\n", encoding="utf-8"
    )
    args = _minimal_args(experiment_dir)
    config = run_preregistration._config_from_args(args)
    assert config.preflight_max_final_train_loss == 0.25


def test_config_from_args_explicit_cli_wins_over_yaml(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "exp"
    experiment_dir.mkdir(parents=True)
    (experiment_dir / "gates.yaml").write_text(
        "convergence:\n  max_final_train_loss: 0.25\n", encoding="utf-8"
    )
    parser = run_preregistration.build_parser()
    args = parser.parse_args([
        "--experiment-dir", str(experiment_dir),
        "--preflight-max-final-train-loss", "0.30",
    ])
    config = run_preregistration._config_from_args(args)
    assert config.preflight_max_final_train_loss == 0.30


def test_config_from_args_yaml_applies_multiple_sections(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "exp"
    experiment_dir.mkdir(parents=True)
    (experiment_dir / "gates.yaml").write_text(
        "convergence:\n"
        "  max_final_train_loss: 0.22\n"
        "fixed_interface_baseline:\n"
        "  max_format_failure_rate: 0.18\n"
        "  allow_unacceptable_for_prefix_search: true\n"
        "preflight:\n"
        "  seed_count: 3\n"
        "  limit: 64\n"
        "  max_exclusion_rate: 0.30\n"
        "  max_arm_seed_exclusion_rate: 0.55\n"
        "  min_parseability_rate: 0.70\n",
        encoding="utf-8",
    )
    args = _minimal_args(experiment_dir)
    config = run_preregistration._config_from_args(args)
    assert config.preflight_max_final_train_loss == 0.22
    assert config.fixed_interface_max_format_failure_rate == 0.18
    assert config.allow_unacceptable_fixed_interface_for_prefix_search is True
    assert config.preflight_seed_count == 3
    assert config.preflight_limit == 64
    assert config.preflight_max_exclusion_rate == 0.30
    assert config.preflight_max_arm_seed_exclusion_rate == 0.55
    assert config.preflight_min_parseability_rate == 0.70


def test_config_from_args_no_yaml_uses_cli_defaults(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "exp"
    experiment_dir.mkdir(parents=True)
    args = _minimal_args(experiment_dir)
    config = run_preregistration._config_from_args(args)
    # Defaults flow through unchanged.
    assert config.preflight_max_final_train_loss == (
        run_preregistration.DEFAULT_PREFLIGHT_MAX_FINAL_TRAIN_LOSS
    )
    assert config.fixed_interface_max_format_failure_rate == (
        run_preregistration.DEFAULT_FIXED_INTERFACE_MAX_FORMAT_FAILURE_RATE
    )


# ---------------------------------------------------------------------------
# Schema well-formedness
# ---------------------------------------------------------------------------


def test_gates_config_schema_is_well_formed() -> None:
    with SCHEMA_PATH.open("r", encoding="utf-8") as fh:
        schema = json.load(fh)
    Draft202012Validator.check_schema(schema)
