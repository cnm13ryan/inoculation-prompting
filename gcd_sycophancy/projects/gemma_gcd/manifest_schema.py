"""Single source of truth for manifest schemas.

Beck "Normalize Symmetries": until this module landed, manifest schemas
lived in two unsynced places:

1. JSON Schema files under ``gemma_gcd/schemas/*.schema.json``
   (used by ``validate_manifests.py`` for runtime validation).
2. Ad-hoc ``dict[str, Any]`` literals scattered across writers
   (``run_preregistration._record_phase``,
   ``run_ip_sweep.materialize_prereg_training_arms``,
   ``run_prereg_prompt_panel._write_panel_manifest``).

A developer who added a manifest field could update one and forget the
other; tests catch invalid manifests against the JSON Schema, but silent
typos in ``manifest["pahses"]`` only surfaced when a downstream reader
crashed. The cost of two schemas is real and accumulates.

This module replaces the dict-literal construction sites with Pydantic v2
models. Pydantic gives us:

* Field-name typos rejected at construction time, not at JSON-validate time.
* JSON Schema generation for free (``Model.model_json_schema()``) — kept
  available for future single-source-of-truth migration of the JSON files
  in ``schemas/``, though that migration is deliberately deferred.
* Runtime ``isinstance`` / mypy reasoning at every read site.

Permissiveness mirrors the JSON Schemas. The on-disk schemas use
``additionalProperties`` permissively (e.g. ``training_manifest`` allows
optional ``ip_placement``, ``arm_set``, ``provenance``,
``contrastive_pairs_augmentation``; ``stage_manifest`` allows extra keys
at the top level). We mirror this with ``model_config = ConfigDict(
extra="allow")`` so legacy on-disk manifests round-trip without loss.

The nested permissive payloads (``outputs``, ``constraints``, ``summary``,
``provenance``) are intentionally typed as ``dict[str, Any]`` rather than
nested models — their shape varies per phase / per corpus and is not a
source of drift in the same way the top-level field set is.

Boundary helpers:

* ``model_to_json_dict(m)`` returns a plain ``dict[str, Any]`` suitable
  for ``json.dump``. We use ``mode="json"`` so ``Path``-like values and
  other non-JSON-native types serialise consistently with
  ``json.dump(..., default=str)``.

Deferred follow-ups (intentionally out of scope for the present tidying):

* ``YAML_TO_RUNNER_FIELD`` in ``scripts/gates/_config.py`` — still a
  manually maintained mapping; reflection-deriving it from
  ``GatesConfig`` is a separate concern and would expand the diff.
* Auto-generating the JSON Schema files from these models. The current
  schemas were hand-tuned with descriptions and minimums; mechanically
  regenerating them would touch the schema files and is best done as a
  follow-up so this PR stays a pure refactor.
* ``PreregDataManifest`` writer site (``generate_train_data``) is not
  refactored here; the model is defined for completeness so future
  writers can adopt it incrementally.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Run / Stage manifests (run_preregistration._record_phase)
# ---------------------------------------------------------------------------


class PhaseRecord(BaseModel):
    """One entry in ``RunManifest.phases`` and the body of a stage manifest.

    Matches the value shape under ``run_manifest.phases[<phase>]`` and is
    embedded inside ``StageManifest`` (which adds an explicit ``phase``
    name field).
    """

    model_config = ConfigDict(extra="allow")

    completed_at_utc: str
    outputs: Any = None


class RunManifest(BaseModel):
    """Aggregate manifest written to ``manifests/run_manifest.json``.

    Mirrors ``schemas/run_manifest.schema.json``. ``phases`` is an
    *object* (mapping phase-name -> record), distinct from the
    ``PromptPanelManifest.phases`` array.
    """

    model_config = ConfigDict(extra="allow")

    workflow_name: str
    experiment_dir: str
    seeds: list[int]
    phases: dict[str, PhaseRecord] = Field(default_factory=dict)


class StageManifest(BaseModel):
    """Per-stage manifest written to ``manifests/stage_<phase>.json``.

    Mirrors ``schemas/stage_manifest.schema.json``. Shares
    ``completed_at_utc`` + ``outputs`` with the matching ``PhaseRecord``
    inside the aggregate ``RunManifest``; adds an explicit ``phase`` name
    so a single stage file is self-describing.
    """

    model_config = ConfigDict(extra="allow")

    phase: str
    completed_at_utc: str
    outputs: Any = None


# ---------------------------------------------------------------------------
# Training manifest (run_ip_sweep.materialize_prereg_training_arms)
# ---------------------------------------------------------------------------


class TrainingDatasetEntry(BaseModel):
    model_config = ConfigDict(extra="allow")

    dataset_path: str
    row_count: int


class TrainingArmEntry(BaseModel):
    model_config = ConfigDict(extra="allow")

    arm_id: int
    label: str
    dataset_path: str
    eval_user_suffix: str


class TrainingManifest(BaseModel):
    """``training_manifest.json`` written by ``materialize_prereg_training_arms``.

    Mirrors ``schemas/training_manifest.schema.json``. The schema permits
    several optional fields beyond the required set (``ip_instruction``,
    ``ip_instruction_id``, ``corpus_b_variant``,
    ``contrastive_pairs_augmentation``); the writer additionally emits
    ``ip_placement``, ``arm_set``, and ``provenance`` for the expanded
    arm set. ``extra="allow"`` carries those through losslessly.
    """

    model_config = ConfigDict(extra="allow")

    materialization_seed: int
    model_name: str
    max_seq_length: int
    epochs: int
    selected_arms: list[str]
    datasets: dict[str, TrainingDatasetEntry]
    arms: dict[str, TrainingArmEntry]
    dataset_composition: dict[str, list[str]]
    corpus_b_variant: Optional[str] = None
    ip_instruction: Optional[str] = None
    ip_instruction_id: Optional[str] = None
    contrastive_pairs_augmentation: Any = None


# ---------------------------------------------------------------------------
# Prompt panel manifest (run_prereg_prompt_panel._write_panel_manifest)
# ---------------------------------------------------------------------------


class PromptPanelCandidate(BaseModel):
    model_config = ConfigDict(extra="allow")

    candidate_id: str
    sanitized_id: str
    suffix_text: str
    experiment_dir: str
    rank: Optional[int] = None
    confirms_incorrect_rate: Optional[float] = None
    delta_vs_no_prompt: Optional[float] = None


class PromptPanelManifest(BaseModel):
    """``prompt_panel_manifest.json`` written by ``run_prereg_prompt_panel``.

    Mirrors ``schemas/prompt_panel_manifest.schema.json``. Note ``phases``
    here is a *list* of phase names — distinct from ``RunManifest.phases``,
    which is an object keyed by phase name. The two manifests evolved
    independently and the JSON Schema documents this asymmetry.
    """

    model_config = ConfigDict(extra="allow")

    workflow_name: str
    source_eligible_panel: str
    experiment_root: str
    corpus_b_variant: str
    seeds: list[int]
    phases: list[str]
    dry_run: bool
    started_at: str
    completed_at: Optional[str]
    candidates: list[PromptPanelCandidate]


# ---------------------------------------------------------------------------
# Prereg data manifest (defined for completeness; writer not yet ported)
# ---------------------------------------------------------------------------


class PreregFileConstraints(BaseModel):
    model_config = ConfigDict(extra="allow")

    allowed_depths: list[int]
    cluster_count: int
    corpus_kind: str
    max_value: int
    min_value: int
    paraphrase_bank_size: Optional[int]
    paraphrases_per_cluster: int
    prompt_families: list[str]
    split_name: str


class PreregFileSummary(BaseModel):
    model_config = ConfigDict(extra="allow")

    claimed_answer_family_histogram: dict[str, int]
    cluster_count: int
    depth_histogram: dict[str, int]
    max_value: int
    min_value: int
    pair_hash: str
    prompt_family_histogram: dict[str, int]
    row_count: int
    unique_latent_pair_count: int


class PreregFileEntry(BaseModel):
    model_config = ConfigDict(extra="allow")

    sha256: str
    constraints: PreregFileConstraints
    summary: PreregFileSummary


class PreregGenerator(BaseModel):
    model_config = ConfigDict(extra="allow")

    corpus_pair_count: int
    dev_cluster_count: int
    dev_range: list[int]
    near_transfer_cluster_count: int
    near_transfer_depths: list[int]
    near_transfer_range: list[int]
    seed: int
    test_cluster_count: int
    test_range: list[int]
    train_depths: list[int]
    train_range: list[int]


class PreregDataManifest(BaseModel):
    """``gemma_gcd/data/prereg/manifest.json`` (writer not yet ported).

    Defined here so future writer refactors have a target. Refactoring
    ``generate_train_data`` to use this model is deferred to keep the
    present diff a pure tidying.
    """

    model_config = ConfigDict(extra="allow")

    schema_version: int
    files: dict[str, PreregFileEntry]
    generator: PreregGenerator


# ---------------------------------------------------------------------------
# Boundary helpers
# ---------------------------------------------------------------------------


def model_to_json_dict(model: BaseModel) -> dict[str, Any]:
    """Serialise a manifest model to a plain JSON-ready dict.

    Uses ``exclude_unset=True`` so optional fields that the caller did
    not pass (and that the legacy dict-literal writer would have omitted
    entirely) do not appear in the output as ``null``. This preserves
    byte-for-byte parity with the pre-refactor writers and lets a
    legacy on-disk manifest round-trip through the model unchanged
    (load -> model -> dump produces the same key set).

    ``mode="json"`` coerces ``Path`` and similar non-native types to
    strings the same way ``json.dump(..., default=str)`` would.
    """
    return model.model_dump(mode="json", exclude_unset=True)


__all__ = [
    "PhaseRecord",
    "PreregDataManifest",
    "PreregFileConstraints",
    "PreregFileEntry",
    "PreregFileSummary",
    "PreregGenerator",
    "PromptPanelCandidate",
    "PromptPanelManifest",
    "RunManifest",
    "StageManifest",
    "TrainingArmEntry",
    "TrainingDatasetEntry",
    "TrainingManifest",
    "model_to_json_dict",
]
