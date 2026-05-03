"""Tests for the Stage-3 layered CLI entrypoints.

Each entrypoint exposes a narrower argparse surface than
``run_preregistration.py`` while dispatching to the same phase modules.
The tests assert (a) the surface is correctly narrowed and (b) the
right phase ``run`` is invoked with a properly built ``RunnerConfig``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

SCRIPT_DIR = Path(__file__).resolve().parent
GEMMA_GCD_DIR = SCRIPT_DIR.parent
PROJECTS_DIR = GEMMA_GCD_DIR.parent
# Reverse order so SCRIPT_DIR ends up at sys.path[0].
for candidate in (PROJECTS_DIR, GEMMA_GCD_DIR, SCRIPT_DIR):
    s = str(candidate)
    if s not in sys.path:
        sys.path.insert(0, s)

import run_preregistration  # noqa: E402


def _load_entrypoint(module_name: str):
    """Load a layered entrypoint by absolute file path.

    We can't ``import data`` directly because ``gemma_gcd/data/`` is an
    implicit namespace package that would shadow ``scripts/data.py``.
    Loading via ``spec_from_file_location`` bypasses the name resolver.
    """
    path = SCRIPT_DIR / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


data = _load_entrypoint("data")
train = _load_entrypoint("train")
evaluate = _load_entrypoint("evaluate")
analyze = _load_entrypoint("analyze")
prereg = _load_entrypoint("prereg")


# ---------------------------------------------------------------------------
# --help: each entrypoint exposes the right phase choices and only its layer's flags.
# ---------------------------------------------------------------------------


def _help_text(parser) -> str:
    return parser.format_help()


def test_data_help_lists_only_data_phase_and_no_eval_or_train_flags():
    text = _help_text(data.build_parser())
    assert "materialize-data" in text
    assert "--data-dir" in text
    assert "--corpus-b-variant" in text
    assert "--lmstudio-base-url" not in text
    assert "--llm-backend" not in text
    assert "--tensor-parallel-size" not in text
    assert "--template-config" not in text


def test_train_help_lists_setup_and_train_phases_and_no_eval_flags():
    text = _help_text(train.build_parser())
    assert "setup" in text
    assert "train" in text
    assert "--template-config" in text
    assert "--checkpoint-curve-every-steps" in text
    assert "--preflight-max-final-train-loss" in text
    assert "--ip-placement" in text
    assert "--lmstudio-base-url" not in text
    assert "--llm-backend" not in text
    assert "--tensor-parallel-size" not in text
    assert "--limit" not in text


def test_evaluate_help_lists_eval_phases_and_eval_flags():
    text = _help_text(evaluate.build_parser())
    for phase in (
        "preflight",
        "fixed-interface-eval",
        "semantic-interface-eval",
        "checkpoint-curve-eval",
    ):
        assert phase in text, f"evaluate.py --help missing phase: {phase}"
    assert "--lmstudio-base-url" in text
    assert "--llm-backend" in text
    assert "--tensor-parallel-size" in text
    assert "--limit" in text
    assert "--checkpoint-curve-every-steps" in text
    assert "--template-config" not in text
    assert "--dont-overwrite" not in text


def test_evaluate_exposes_preflight_max_final_train_loss():
    """Regression: PR #100 review surfaced that evaluate.py routes the
    preflight phase, which calls the convergence gate via
    _check_training_convergence. The gate reads
    config.preflight_max_final_train_loss; if the layered eval CLI
    omits the flag, users hitting a convergence failure cannot follow
    the runtime error's instructions to raise the threshold.
    """
    text = _help_text(evaluate.build_parser())
    assert "--preflight-max-final-train-loss" in text


def test_evaluate_runner_config_picks_up_preflight_max_final_train_loss():
    """The flag must round-trip into RunnerConfig, not just appear in --help."""
    args = evaluate.build_parser().parse_args(
        ["preflight", "--preflight-max-final-train-loss", "0.20"]
    )
    config = evaluate.cli.build_runner_config(args)
    assert config.preflight_max_final_train_loss == pytest.approx(0.20)


def test_analyze_help_lists_analysis_phases_and_no_eval_or_train_flags():
    text = _help_text(analyze.build_parser())
    assert "analysis" in text
    assert "seed-instability" in text
    assert "--prompt-template-variant" in text
    assert "--lmstudio-base-url" not in text
    assert "--template-config" not in text


# ---------------------------------------------------------------------------
# Regression: run_preregistration.py --help still lists every phase.
# ---------------------------------------------------------------------------


def test_run_preregistration_help_still_lists_all_phases():
    text = _help_text(run_preregistration.build_parser())
    for phase in (
        "materialize-data",
        "setup",
        "preflight",
        "train",
        "fixed-interface-eval",
        "semantic-interface-eval",
        "analysis",
        "seed-instability",
        "checkpoint-curve-eval",
        "full",
        "record-deviation",
    ):
        assert phase in text, f"run_preregistration.py --help missing phase: {phase}"


# ---------------------------------------------------------------------------
# Dispatch: each entrypoint routes to the correct phase function.
# ---------------------------------------------------------------------------


class _Spy:
    def __init__(self) -> None:
        self.called = False
        self.received_config = None

    def __call__(self, config) -> None:
        self.called = True
        self.received_config = config


def test_data_main_dispatches_materialize_data_phase(tmp_path):
    spy = _Spy()
    with patch.dict(data.PHASES, {"materialize-data": spy}, clear=False):
        rc = data.main(
            [
                "materialize-data",
                "--experiment-dir", str(tmp_path / "exp"),
                "--data-dir", str(tmp_path / "data"),
            ]
        )
    assert rc == 0
    assert spy.called
    config = spy.received_config
    assert isinstance(config, run_preregistration.RunnerConfig)
    assert config.experiment_dir == (tmp_path / "exp").resolve()
    assert config.data_dir == (tmp_path / "data").resolve()


def test_train_main_dispatches_setup_phase():
    setup_spy = _Spy()
    train_spy = _Spy()
    with patch.dict(train.PHASES, {"setup": setup_spy, "train": train_spy}, clear=False):
        rc = train.main(["setup"])
    assert rc == 0
    assert setup_spy.called
    assert not train_spy.called


def test_train_main_dispatches_train_phase_with_dont_overwrite():
    setup_spy = _Spy()
    train_spy = _Spy()
    with patch.dict(train.PHASES, {"setup": setup_spy, "train": train_spy}, clear=False):
        rc = train.main(["train", "--dont-overwrite"])
    assert rc == 0
    assert train_spy.called
    assert train_spy.received_config.dont_overwrite is True


def test_evaluate_main_dispatches_fixed_interface_eval_with_limit():
    spies = {name: _Spy() for name in evaluate.PHASES}
    with patch.dict(evaluate.PHASES, spies, clear=False):
        rc = evaluate.main(["fixed-interface-eval", "--limit", "8"])
    assert rc == 0
    assert spies["fixed-interface-eval"].called
    assert spies["fixed-interface-eval"].received_config.limit == 8


def test_evaluate_main_dispatches_preflight():
    spies = {name: _Spy() for name in evaluate.PHASES}
    with patch.dict(evaluate.PHASES, spies, clear=False):
        rc = evaluate.main(["preflight"])
    assert rc == 0
    assert spies["preflight"].called


def test_analyze_main_dispatches_analysis():
    spies = {name: _Spy() for name in analyze.PHASES}
    with patch.dict(analyze.PHASES, spies, clear=False):
        rc = analyze.main(["analysis"])
    assert rc == 0
    assert spies["analysis"].called


def test_analyze_main_dispatches_seed_instability():
    spies = {name: _Spy() for name in analyze.PHASES}
    with patch.dict(analyze.PHASES, spies, clear=False):
        rc = analyze.main(["seed-instability"])
    assert rc == 0
    assert spies["seed-instability"].called


# ---------------------------------------------------------------------------
# RunnerConfig equivalence: layered builder == unified _config_from_args
# for the flags both surfaces share.
# ---------------------------------------------------------------------------


def test_layered_runner_config_matches_unified_for_shared_flags(tmp_path):
    shared_args = [
        "--experiment-dir", str(tmp_path / "exp"),
        "--data-dir", str(tmp_path / "data"),
        "--seeds", "0", "1",
        "--corpus-b-variant", "b2",
        "--ip-placement", "append",
        "--prompt-template-variant", "derivation_first",
    ]

    train_args = train.build_parser().parse_args(["train", *shared_args])
    train_config = train.cli.build_runner_config(train_args)

    unified_args = run_preregistration.build_parser().parse_args(["train", *shared_args])
    unified_config = run_preregistration._config_from_args(unified_args)

    assert train_config.experiment_dir == unified_config.experiment_dir
    assert train_config.data_dir == unified_config.data_dir
    assert train_config.seeds == unified_config.seeds
    assert train_config.corpus_b_variant == unified_config.corpus_b_variant
    assert train_config.ip_placement == unified_config.ip_placement
    assert train_config.arm_set == unified_config.arm_set
    assert train_config.prompt_template_variant == unified_config.prompt_template_variant


# ---------------------------------------------------------------------------
# prereg.py shim still wires through to run_preregistration.main.
# ---------------------------------------------------------------------------


def test_prereg_shim_delegates_to_run_preregistration():
    assert prereg.main is run_preregistration.main
