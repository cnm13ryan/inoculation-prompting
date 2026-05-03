"""Sibling-module facade exposing runner helpers used by ``phases/`` and ``gates/``.

This module exists to invert the dependency direction between phase / gate
modules and ``run_preregistration``: instead of every phase doing a lazy
``import run_preregistration as _rp`` to reach back into its parent runner,
phases and gates import named helpers from this sibling module.

The runtime forwarding to ``run_preregistration`` is itself lazy: each
exported helper / constant is a thin proxy whose body imports
``run_preregistration`` on first call and looks the attribute up afresh.
This indirection exists for two reasons:

1. ``run_preregistration`` is the canonical script the runner executes as
   ``__main__``. Importing it eagerly at this module's load time would
   re-enter the runner during its own initialisation (the runner imports
   ``phases.materialize_data`` etc. at module load, and those imports must
   not transitively re-import the runner), causing a circular import.
2. Importing ``run_preregistration`` transitively pulls in ``pandas`` and
   data-validation utilities. Tests rely on being able to import ``phases``
   submodules without paying that cost (see
   ``test_lazy_heavy_imports.py``).

Because the proxies look up each helper on ``run_preregistration`` at
call-time, existing tests that monkeypatch helpers via
``monkeypatch.setattr(run_preregistration, "_run_checked", ...)`` continue
to work: phase / gate code reaches the patched function on its next call.

Adding a new ``_rp.<name>`` access in a phase / gate module should be paired
with adding the name to ``_HELPER_NAMES`` below.

Documentation note: the placement choice (``phases/_runner_helpers.py``
rather than e.g. a top-level ``runner_helpers.py``) keeps the helper
facade adjacent to its primary consumers and signals via the leading
underscore that the module is internal to the runner package layout.
"""

from __future__ import annotations

from typing import Any

# The complete set of helper / constant names that phase and gate modules
# may import from this facade. Kept sorted for readability.
_HELPER_NAMES: tuple[str, ...] = (
    # Constants / module-level paths
    "ANALYSIS_SCRIPT",
    "CHECKPOINT_CURVE_EVAL_SCRIPT",
    "DEFAULT_BEST_ELICITED_DATASET",
    "DEFAULT_PREFLIGHT_DATASET",
    "EXPORT_SCRIPT",
    "FIXED_EVAL_SCRIPT",
    "PREFIX_SEARCH_SCRIPT",
    "PROJECTS_DIR",
    "PTST_ARM_SLUG",
    "SEED_INSTABILITY_SCRIPT",
    # Public re-exports from run_ip_sweep that the runner re-exports
    "arms_for_arm_set",
    "materialize_prereg_training_arms",
    # Config dataclasses (added by PR #122 — phase-specific slices). Listed
    # here so phases that take a config slice can pull it via the sibling
    # facade rather than importing from run_preregistration directly. At
    # runtime, type annotations are strings under ``from __future__ import
    # annotations``, so phases that ONLY use these for typing don't strictly
    # need the proxy — but uniformity is cheap and keeps the parent-import
    # smell out of phase modules.
    "BaseConfig",
    "PreflightConfig",
    "RunnerConfig",
    # Phase entry points re-exposed by the runner
    "run_materialize_data_phase",
    "run_seed_instability_phase",
    # Path / state helpers
    "_analysis_exclusion_categories_path",
    "_analysis_exclusion_diagnostics_path",
    "_analysis_json_path",
    "_analysis_output_prefix",
    "_analysis_summary_path",
    "_annotate_frozen_prefix_artifact",
    "_attributes_to_vary_path",
    "_best_elicited_output_dir",
    "_checkpoint_curve_output_prefix",
    "_collect_preflight_rows",
    "_condition_labels_path",
    "_copy_template_config_if_needed",
    "_deviations_log_path",
    "_ensure_condition_dirs_and_seed_configs",
    "_ensure_deviations_log_exists",
    "_ensure_prereq_scripts_exist",
    "_evaluation_common_args",
    "_experiment_arms_dir",
    "_final_report_path",
    "_fixed_interface_baseline_report_path",
    "_fixed_interface_output_dir",
    "_freeze_training_manifest",
    "_frozen_data_manifest_path",
    "_frozen_prefix_path",
    "_h5_condition_dirs",
    "_has_results",
    "_inject_checkpoint_curve_into_config",
    "_iter_arm_condition_dirs",
    "_latest_eval_model_dir",
    "_load_base_training_config",
    "_load_or_create_fixed_interface_baseline_report",
    "_make_preflight_report",
    "_prefix_search_gate_status",
    "_prefix_search_output_dir",
    "_preflight_config",
    "_preflight_export_path",
    "_preflight_output_dir",
    "_preflight_report_path",
    "_preflight_summary_path",
    "_problem_level_export_path",
    "_read_json",
    "_record_phase",
    "_require_analysis_inputs",
    "_require_frozen_manifests",
    "_run_checked",
    "_run_training_phase",
    "_seed_instability_output_prefix",
    "_seed_instability_report_path",
    "_seed_instability_summary_path",
    "_seed_instability_trajectory_path",
    "_select_only_arm_slugs",
    "_semantic_interface_output_dir",
    "_validate_and_freeze_data_manifest",
    "_validate_frozen_prefix_artifacts",
    "_validate_seed_configs_exist",
    "_validate_training_outputs",
    "_write_final_report",
    "_write_fixed_interface_baseline_report",
    "_write_json",
    "_write_preflight_summary",
    "_write_prereg_setup_metadata",
)


class _LazyAttr:
    """Callable proxy that forwards to a named attribute of
    ``run_preregistration`` at call time.

    Importing the proxy via ``from phases._runner_helpers import X`` does
    NOT trigger the runner import; only invoking the proxy (or reading one
    of its forwarded attributes) does. Each access reads the underlying
    name from ``run_preregistration`` afresh, so monkeypatches applied
    via ``monkeypatch.setattr(run_preregistration, name, ...)`` are seen
    by the next call.

    The proxy supports the calling conventions phase / gate code uses:

    * ``proxy(...)`` - delegates to the underlying callable.
    * ``str(proxy)`` / ``os.fspath(proxy)`` - delegates to the underlying
      value, allowing ``Path`` constants like ``FIXED_EVAL_SCRIPT`` to be
      passed straight to ``str(...)`` and into subprocess argv lists.
    * ``proxy.<attr>`` - delegates to the underlying value, supporting
      ``Path``-style attribute chains where used.
    """

    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        object.__setattr__(self, "_name", name)

    def _resolve(self) -> Any:
        import run_preregistration as _rp  # local: defer until first call
        return getattr(_rp, object.__getattribute__(self, "_name"))

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._resolve()(*args, **kwargs)

    def __getattr__(self, attr: str) -> Any:
        # ``__getattr__`` only fires for attributes not found by normal
        # lookup, so ``_resolve`` and ``_name`` (the slots) are excluded.
        return getattr(self._resolve(), attr)

    def __fspath__(self) -> str:
        return str(self._resolve())

    def __str__(self) -> str:
        return str(self._resolve())

    def __repr__(self) -> str:
        return f"_LazyAttr({object.__getattribute__(self, '_name')!r})"

    def __eq__(self, other: object) -> bool:
        return self._resolve() == other

    def __hash__(self) -> int:
        return hash(self._resolve())

    def __truediv__(self, other: Any) -> Any:
        # Support ``PROJECTS_DIR / "subdir"`` style path composition used by
        # phase code without forcing callers to call ``str(...)`` first.
        return self._resolve() / other


# Populate this module's namespace with one proxy per helper name so that
# ``from phases._runner_helpers import X`` returns a proxy without
# triggering the ``run_preregistration`` import.
for _name in _HELPER_NAMES:
    globals()[_name] = _LazyAttr(_name)
del _name


def __dir__() -> list[str]:
    return sorted(_HELPER_NAMES)


__all__ = list(_HELPER_NAMES)
