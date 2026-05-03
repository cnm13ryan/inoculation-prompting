"""Per-experiment ``gates.yaml`` config loader and merge logic.

A ``gates.yaml`` file can sit at ``<experiment_dir>/gates.yaml`` to override
gate thresholds without changing CLI flags. The file is OPT-IN — when absent,
behaviour is unchanged. When present, fields override the unified-CLI
defaults but are themselves overridden by explicit CLI flags (so existing
scripts pass through unchanged).

Schema lives at ``gemma_gcd/schemas/gates_config.schema.json`` and is
enforced via JSON Schema (``additionalProperties: false`` at every level so
typoed keys raise rather than silently no-op).

YAML keys are grouped by gate (``convergence.max_final_train_loss``,
``fixed_interface_baseline.max_format_failure_rate``,
``preflight.seed_count``, ...). The mapping from YAML keys to
``RunnerConfig`` field names lives in :data:`YAML_TO_RUNNER_FIELD` and is
applied by :func:`apply_to_runner_config_kwargs`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import yaml
from jsonschema import Draft202012Validator

GATES_YAML_BASENAME = "gates.yaml"
_SCHEMAS_DIR = Path(__file__).resolve().parent.parent.parent / "schemas"
_SCHEMA_PATH = _SCHEMAS_DIR / "gates_config.schema.json"


# Mapping from (yaml_section, yaml_key) -> RunnerConfig field name.
# Centralised here so the YAML schema can group thresholds by gate while
# RunnerConfig keeps its existing flat field names (which several callers
# read directly).
YAML_TO_RUNNER_FIELD: dict[tuple[str, str], str] = {
    ("convergence", "max_final_train_loss"): "preflight_max_final_train_loss",
    ("fixed_interface_baseline", "max_format_failure_rate"):
        "fixed_interface_max_format_failure_rate",
    ("preflight", "seed_count"): "preflight_seed_count",
    ("preflight", "limit"): "preflight_limit",
    ("preflight", "max_exclusion_rate"): "preflight_max_exclusion_rate",
    ("preflight", "max_arm_seed_exclusion_rate"):
        "preflight_max_arm_seed_exclusion_rate",
    ("preflight", "min_parseability_rate"): "preflight_min_parseability_rate",
}


def _load_schema() -> dict[str, Any]:
    with _SCHEMA_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _validate_against_schema(payload: dict[str, Any], source: Path) -> None:
    """Raise ValueError with a readable message if ``payload`` violates the schema.

    ``additionalProperties: false`` at every level means typoed keys (whether
    top-level like ``preflite`` or nested like ``preflight.seed_kount``) are
    surfaced as validation errors rather than silently ignored.
    """
    schema = _load_schema()
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(payload), key=lambda e: list(e.absolute_path))
    if not errors:
        return
    formatted = []
    for err in errors:
        location = "/".join(str(p) for p in err.absolute_path) or "<root>"
        formatted.append(f"  - at {location}: {err.message}")
    raise ValueError(
        f"Invalid gates config at {source}:\n" + "\n".join(formatted)
    )


def load_gate_config(experiment_dir: Path) -> dict[str, Any]:
    """Load ``<experiment_dir>/gates.yaml`` if present.

    Returns ``{}`` when the file is absent. Returns the parsed mapping
    when present and valid. Raises ``ValueError`` when the file exists but
    fails schema validation (unknown keys, wrong types, etc.).
    """
    path = Path(experiment_dir) / GATES_YAML_BASENAME
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(
            f"Invalid gates config at {path}: top-level must be a mapping, "
            f"got {type(payload).__name__}."
        )
    _validate_against_schema(payload, source=path)
    return payload


def apply_to_runner_config_kwargs(
    yaml_config: Mapping[str, Any],
    cli_kwargs: dict[str, Any],
    cli_defaults: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge YAML overrides into ``cli_kwargs``.

    Precedence (highest first):

    1. CLI flag explicitly set by the user (i.e. value differs from
       argparse default).
    2. YAML field present.
    3. CLI argparse default (unchanged).

    ``cli_defaults`` maps RunnerConfig field name -> argparse default. We
    consider a CLI value "explicit" iff it differs from this default; only
    fields whose CLI value still matches the default are eligible for YAML
    override. This is the simple-but-good-enough design discussed in the
    spec: it does the right thing in every case except the (harmless) one
    where a user passes a CLI flag with a value that happens to equal the
    default, in which case YAML wins — which is operationally fine.

    Returns a new dict; ``cli_kwargs`` is not mutated.
    """
    out = dict(cli_kwargs)
    for (section, key), runner_field in YAML_TO_RUNNER_FIELD.items():
        section_payload = yaml_config.get(section)
        if not isinstance(section_payload, dict):
            continue
        if key not in section_payload:
            continue
        yaml_value = section_payload[key]
        cli_value = out.get(runner_field)
        default_value = cli_defaults.get(runner_field)
        # YAML only writes when the CLI value still equals the argparse
        # default, i.e. the user did not explicitly set the flag.
        if cli_value == default_value:
            out[runner_field] = yaml_value
    return out


__all__ = [
    "GATES_YAML_BASENAME",
    "YAML_TO_RUNNER_FIELD",
    "apply_to_runner_config_kwargs",
    "load_gate_config",
]
