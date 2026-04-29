"""GPU-free tests for evaluate_checkpoint_curve.py.

The evaluator subprocess call is replaced with a test stub that writes
fake eval-result JSON files so all assertions run without a GPU.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECTS_DIR = SCRIPT_DIR.parents[2]
GEMMA_GCD_DIR = SCRIPT_DIR.parent
for candidate in (SCRIPT_DIR, GEMMA_GCD_DIR, PROJECTS_DIR):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

import evaluate_checkpoint_curve as ecc  # noqa: E402


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_step_checkpoint(
    seed_dir: Path,
    step: int,
    epoch: int = 1,
    train_loss: float = 0.5,
) -> Path:
    step_dir = seed_dir / "checkpoints" / f"step_{step:06d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        step_dir / "metadata.json",
        {"checkpoint_step": step, "epoch": epoch, "train_loss": train_loss},
    )
    return step_dir


def _fake_eval_results(
    *,
    parseability_rate: float = 0.9,
    sycophancy_rate_given_parseable: float | None = 0.8,
    direct_solve_accuracy: float | None = None,
    total_count: int = 100,
) -> dict[str, Any]:
    exclusion_proportion = 1.0 - parseability_rate
    excluded_count = int(round(total_count * exclusion_proportion))
    return {
        "evaluation_interface": "preregistered_fixed_interface",
        "exclusions": {
            "total": {"count": excluded_count, "proportion": exclusion_proportion},
            "categories": {
                "unparseable_response": {"count": excluded_count, "proportion": exclusion_proportion},
                "degenerate_response": {"count": 0, "proportion": 0.0},
                "truncated_before_verdict_field": {"count": 0, "proportion": 0.0},
            },
        },
        "conditional_sycophancy_rate": {
            "count": total_count - excluded_count,
            "overall_mean": sycophancy_rate_given_parseable,
            "positive_count": (
                int((total_count - excluded_count) * (sycophancy_rate_given_parseable or 0))
                if sycophancy_rate_given_parseable is not None
                else 0
            ),
        },
        "sycophancy_rate": {
            "count": total_count - excluded_count,
            "overall_mean": sycophancy_rate_given_parseable,
        },
        "direct_solve_accuracy": (
            {"count": total_count, "overall_mean": direct_solve_accuracy, "positive_count": 0}
            if direct_solve_accuracy is not None
            else {}
        ),
    }


def _make_stub_run_eval(
    seed_dir: Path,
    results_by_step: dict[int, dict[str, Any]] | None = None,
) -> Any:
    """Return a run_eval_fn stub that writes fake eval result JSON files."""

    def stub_run_eval(
        *,
        model_dir: Path,
        output_dir: Path,
        dataset_spec: str,
        **kwargs: Any,
    ) -> None:
        step_str = model_dir.name
        step = int(step_str.removeprefix("step_"))
        eval_payload = (results_by_step or {}).get(step, _fake_eval_results())
        eval_set_name = dataset_spec.split(":")[0] if ":" in dataset_spec else Path(dataset_spec).stem
        model_results_dir = output_dir / "results" / "20260401_120000" / "fake_model_evals"
        model_results_dir.mkdir(parents=True, exist_ok=True)
        _write_json(model_results_dir / f"{eval_set_name}_eval_results.json", eval_payload)

    return stub_run_eval


# ── discovery ────────────────────────────────────────────────────────────────

def test_discover_step_checkpoint_dirs_returns_sorted_by_step(tmp_path: Path):
    seed_dir = tmp_path / "seed_0"
    _write_step_checkpoint(seed_dir, 375)
    _write_step_checkpoint(seed_dir, 750)
    _write_step_checkpoint(seed_dir, 125)

    results = ecc.discover_step_checkpoint_dirs(seed_dir)
    assert [step for step, _ in results] == [125, 375, 750]


def test_discover_step_checkpoint_dirs_ignores_epoch_dirs(tmp_path: Path):
    seed_dir = tmp_path / "seed_0"
    _write_step_checkpoint(seed_dir, 375)
    epoch_dir = seed_dir / "checkpoints" / "epoch_0"
    epoch_dir.mkdir(parents=True, exist_ok=True)

    results = ecc.discover_step_checkpoint_dirs(seed_dir)
    assert len(results) == 1
    assert results[0][0] == 375


def test_discover_step_checkpoint_dirs_requires_metadata_json(tmp_path: Path):
    seed_dir = tmp_path / "seed_0"
    step_dir = seed_dir / "checkpoints" / "step_000100"
    step_dir.mkdir(parents=True, exist_ok=True)

    results = ecc.discover_step_checkpoint_dirs(seed_dir)
    assert results == []


def test_discover_step_checkpoint_dirs_returns_empty_when_no_checkpoints_dir(tmp_path: Path):
    seed_dir = tmp_path / "seed_0"
    seed_dir.mkdir(parents=True)

    results = ecc.discover_step_checkpoint_dirs(seed_dir)
    assert results == []


# ── limit application ────────────────────────────────────────────────────────

def test_evaluate_checkpoint_curve_applies_limit(tmp_path: Path):
    seed_dir = tmp_path / "seed_0"
    for step in range(100, 1100, 100):
        _write_step_checkpoint(seed_dir, step)

    visited: list[int] = []

    def stub(*, model_dir: Path, output_dir: Path, dataset_spec: str, **kwargs: Any) -> None:
        step = int(model_dir.name.removeprefix("step_"))
        visited.append(step)
        eval_set = dataset_spec.split(":")[0] if ":" in dataset_spec else Path(dataset_spec).stem
        results_dir = output_dir / "results" / "ts" / "evals"
        results_dir.mkdir(parents=True, exist_ok=True)
        _write_json(results_dir / f"{eval_set}_eval_results.json", _fake_eval_results())

    ecc.evaluate_checkpoint_curve(
        seed_dir=seed_dir,
        arm_slug="neutral_baseline",
        seed=0,
        dataset_spec="dev:dev.jsonl",
        output_prefix=tmp_path / "out" / "curve",
        checkpoint_curve_limit=3,
        run_eval_fn=stub,
    )
    assert visited == [100, 200, 300]


# ── CSV / JSON output ────────────────────────────────────────────────────────

def test_evaluate_checkpoint_curve_writes_csv_and_json(tmp_path: Path):
    seed_dir = tmp_path / "seed_0"
    _write_step_checkpoint(seed_dir, 375, epoch=2, train_loss=0.4)
    _write_step_checkpoint(seed_dir, 750, epoch=4, train_loss=0.3)

    curve_df, metadata = ecc.evaluate_checkpoint_curve(
        seed_dir=seed_dir,
        arm_slug="neutral_baseline",
        seed=1,
        dataset_spec="dev:dev.jsonl",
        output_prefix=tmp_path / "out" / "curve",
        checkpoint_curve_limit=32,
        run_eval_fn=_make_stub_run_eval(seed_dir),
    )

    assert len(curve_df) == 2
    assert set(["arm_slug", "seed", "checkpoint_step", "epoch", "train_loss"]).issubset(
        curve_df.columns
    )
    assert list(curve_df["checkpoint_step"]) == [375, 750]
    assert metadata["arm_slug"] == "neutral_baseline"
    assert metadata["seed"] == 1
    assert metadata["checkpoint_count"] == 2
    assert len(metadata["curve"]) == 2


def test_write_outputs_creates_csv_and_json_files(tmp_path: Path):
    curve_df = pd.DataFrame([
        {"arm_slug": "neutral_baseline", "seed": 0, "checkpoint_step": 375,
         "epoch": 2, "train_loss": 0.4, "sycophancy_rate_given_parseable": 0.8,
         "parseability_rate": 0.9},
    ])
    metadata = {"workflow_name": "checkpoint_curve_eval", "curve": []}
    output_prefix = tmp_path / "reports" / "curve"

    ecc.write_outputs(curve_df=curve_df, metadata=metadata, output_prefix=output_prefix)

    assert (tmp_path / "reports" / "curve.curve.csv").exists()
    assert (tmp_path / "reports" / "curve.curve.json").exists()


# ── metrics extraction ────────────────────────────────────────────────────────

def test_extract_curve_metrics_parses_eval_results():
    summaries = {
        "dev": {
            "conditional_sycophancy_rate": {"overall_mean": 0.75, "count": 90},
            "direct_solve_accuracy": {},
            "exclusions": {"total": {"count": 10, "proportion": 0.1}},
        }
    }
    metrics = ecc._extract_curve_metrics(summaries, "dev")
    assert abs(metrics["parseability_rate"] - 0.9) < 1e-9
    assert abs(metrics["sycophancy_rate_given_parseable"] - 0.75) < 1e-9
    assert metrics["direct_solve_accuracy"] is None


def test_extract_curve_metrics_raises_on_missing_eval_set():
    summaries = {
        "other": {
            "conditional_sycophancy_rate": {"overall_mean": 0.5},
            "exclusions": {"total": {"count": 5, "proportion": 0.05}},
            "direct_solve_accuracy": {},
        }
    }
    with pytest.raises(ValueError, match="'dev'.*not found"):
        ecc._extract_curve_metrics(summaries, "dev")


def test_extract_curve_metrics_empty_summaries():
    metrics = ecc._extract_curve_metrics({}, "dev")
    assert metrics["sycophancy_rate_given_parseable"] is None
    assert metrics["parseability_rate"] is None


# ── empty checkpoint dir ──────────────────────────────────────────────────────

def test_evaluate_checkpoint_curve_returns_empty_when_no_checkpoints(tmp_path: Path):
    seed_dir = tmp_path / "seed_0"
    seed_dir.mkdir()

    curve_df, metadata = ecc.evaluate_checkpoint_curve(
        seed_dir=seed_dir,
        arm_slug="neutral_baseline",
        seed=0,
        dataset_spec="dev:dev.jsonl",
        output_prefix=tmp_path / "out" / "curve",
        checkpoint_curve_limit=32,
    )

    assert curve_df.empty
    assert metadata["checkpoint_count"] == 0
    assert metadata["curve"] == []


# ── dataset spec name parsing ──────────────────────────────────────────────

def test_dataset_spec_name_with_colon():
    assert ecc._dataset_spec_name("dev:gemma_gcd/data/prereg/dev.jsonl") == "dev"


def test_dataset_spec_name_bare_path():
    assert ecc._dataset_spec_name("gemma_gcd/data/prereg/dev.jsonl") == "dev"


# ── adapter-only step checkpoint detection + resolution ─────────────────────


def _write_merged_step_dir(seed_dir: Path, step: int) -> Path:
    step_dir = seed_dir / "checkpoints" / f"step_{step:06d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        step_dir / "metadata.json",
        {
            "checkpoint_step": step,
            "epoch": 1,
            "train_loss": 0.4,
            "checkpoint_kind": "merged_full_model",
        },
    )
    (step_dir / "config.json").write_text("{}", encoding="utf-8")
    return step_dir


def _write_adapter_step_dir(seed_dir: Path, step: int) -> Path:
    step_dir = seed_dir / "checkpoints" / f"step_{step:06d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        step_dir / "metadata.json",
        {
            "checkpoint_step": step,
            "epoch": 1,
            "train_loss": 0.4,
            "checkpoint_kind": "lora_adapter",
        },
    )
    _write_json(
        step_dir / "adapter_config.json",
        {"base_model_name_or_path": "fake/base", "peft_type": "LORA"},
    )
    return step_dir


def test_is_adapter_only_checkpoint_true_when_metadata_says_lora(tmp_path: Path):
    step_dir = _write_adapter_step_dir(tmp_path, step=75)
    assert ecc._is_adapter_only_checkpoint(step_dir) is True


def test_is_adapter_only_checkpoint_false_when_metadata_says_merged(tmp_path: Path):
    step_dir = _write_merged_step_dir(tmp_path, step=75)
    assert ecc._is_adapter_only_checkpoint(step_dir) is False


def test_is_adapter_only_checkpoint_falls_back_to_adapter_config_presence(tmp_path: Path):
    """Legacy adapter dirs without checkpoint_kind in metadata must still be
    detected via the file-existence fallback."""
    step_dir = tmp_path / "checkpoints" / "step_000075"
    step_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        step_dir / "metadata.json",
        {"checkpoint_step": 75, "epoch": 1, "train_loss": 0.4},  # no checkpoint_kind
    )
    _write_json(
        step_dir / "adapter_config.json",
        {"base_model_name_or_path": "fake/base", "peft_type": "LORA"},
    )
    assert ecc._is_adapter_only_checkpoint(step_dir) is True


def test_is_adapter_only_checkpoint_false_when_dir_missing(tmp_path: Path):
    assert ecc._is_adapter_only_checkpoint(tmp_path / "does_not_exist") is False


def test_resolved_eval_model_dir_passes_through_merged_ckpt(tmp_path: Path):
    """For legacy merged ckpts, the context manager yields the original path
    and never touches the materializer."""
    step_dir = _write_merged_step_dir(tmp_path, step=75)
    with ecc._resolved_eval_model_dir(step_dir) as resolved:
        assert resolved == step_dir


def test_resolved_eval_model_dir_materializes_then_cleans_up_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """For adapter ckpts, the context manager must call the materializer,
    yield its temp dir, and unconditionally remove it on exit."""
    step_dir = _write_adapter_step_dir(tmp_path, step=75)
    fake_temp = tmp_path / "_materialized_temp"
    fake_temp.mkdir(parents=True, exist_ok=True)
    (fake_temp / "config.json").write_text("{}", encoding="utf-8")
    calls: list[Path] = []

    def fake_materialize(adapter_dir: Path) -> Path:
        calls.append(adapter_dir)
        return fake_temp

    monkeypatch.setattr(ecc, "_materialize_adapter_to_tempdir", fake_materialize)

    with ecc._resolved_eval_model_dir(step_dir) as resolved:
        assert resolved == fake_temp
        assert fake_temp.exists()
    assert calls == [step_dir], (
        f"materializer should be called exactly once with the adapter dir; got {calls!r}"
    )
    assert not fake_temp.exists(), "temp materialization dir must be cleaned up on exit"


def test_resolved_eval_model_dir_cleans_up_even_on_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    step_dir = _write_adapter_step_dir(tmp_path, step=75)
    fake_temp = tmp_path / "_materialized_temp_err"
    fake_temp.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(ecc, "_materialize_adapter_to_tempdir", lambda d: fake_temp)
    with pytest.raises(RuntimeError, match="boom"):
        with ecc._resolved_eval_model_dir(step_dir):
            raise RuntimeError("boom")
    assert not fake_temp.exists()


def test_has_eval_results_false_when_dir_missing(tmp_path: Path):
    assert ecc._has_eval_results(tmp_path / "missing") is False


def test_has_eval_results_false_when_only_inference_config(tmp_path: Path):
    """Regression: a crashed run that wrote inference_config.json but no
    *_eval_results.json must NOT count as 'already evaluated', otherwise the
    curve loop's skip-if-cached guard short-circuits the rerun and the next
    metric extraction fails with no results found."""
    crashed_dir = tmp_path / "results" / "20260101_000000" / "model_evals"
    crashed_dir.mkdir(parents=True)
    (crashed_dir / "inference_config.json").write_text("{}", encoding="utf-8")
    assert ecc._has_eval_results(tmp_path) is False


def test_has_eval_results_true_when_eval_results_present(tmp_path: Path):
    finished_dir = tmp_path / "results" / "20260101_000000" / "model_evals"
    finished_dir.mkdir(parents=True)
    (finished_dir / "test_confirmatory_eval_results.json").write_text(
        "{}", encoding="utf-8"
    )
    assert ecc._has_eval_results(tmp_path) is True


def test_evaluate_checkpoint_curve_resolves_adapter_ckpts_before_eval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Integration: an adapter step ckpt must be materialized to a temp dir,
    and run_eval_fn must receive that temp dir as model_dir (not the adapter
    path), while curve metrics still tag the original step number."""
    seed_dir = tmp_path / "seed_0"
    _write_adapter_step_dir(seed_dir, step=75)
    fake_temp = tmp_path / "_eval_merged_75"
    fake_temp.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(ecc, "_materialize_adapter_to_tempdir", lambda d: fake_temp)

    received: dict[str, Any] = {}

    def stub_run_eval(*, model_dir: Path, output_dir: Path, dataset_spec: str, **kwargs: Any) -> None:
        received["model_dir"] = model_dir
        eval_set_name = (
            dataset_spec.split(":", 1)[0] if ":" in dataset_spec else Path(dataset_spec).stem
        )
        model_results_dir = output_dir / "results" / "20260401_120000" / "fake_model_evals"
        model_results_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            model_results_dir / f"{eval_set_name}_eval_results.json",
            _fake_eval_results(),
        )

    output_prefix = tmp_path / "curve_root" / "curve"
    curve_df, metadata = ecc.evaluate_checkpoint_curve(
        seed_dir=seed_dir,
        arm_slug="inoculation_prompting",
        seed=0,
        dataset_spec="test_confirmatory:gemma_gcd/data/prereg/test_confirmatory.jsonl",
        output_prefix=output_prefix,
        checkpoint_curve_limit=10,
        run_eval_fn=stub_run_eval,
    )

    assert received["model_dir"] == fake_temp, (
        "run_eval_fn must receive the materialized merged temp dir, not the "
        f"adapter path. Got {received['model_dir']!r}."
    )
    assert not fake_temp.exists(), "materialized temp dir must be cleaned up after eval"
    assert metadata["checkpoint_count"] == 1
    assert curve_df.iloc[0]["checkpoint_step"] == 75
