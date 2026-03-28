"""Tests for analyze_selective_suppression.py."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import analyze_selective_suppression as ass  # noqa: E402


def _row(
    seed: int,
    *,
    inoculation: int,
    pressure: int,
    question_type: str,
    user_provides_answer: str,
    correct_when_wrong: float,
    affirm_when_correct: float,
    condition_label: str | None = None,
    eval_protocol: str = "single_turn",
) -> dict[str, Any]:
    if condition_label is None:
        if inoculation and pressure:
            condition_label = "IP Behave Correct / Pressured"
        elif inoculation and not pressure:
            condition_label = "IP Behave Correct / Neutral"
        elif not inoculation and pressure:
            condition_label = "Control / Pressured"
        else:
            condition_label = "Control / Neutral"

    return {
        "condition_label": condition_label,
        "seed": seed,
        "inoculation": inoculation,
        "pressure": pressure,
        "question_type": question_type,
        "user_provides_answer": user_provides_answer,
        "correct_when_wrong": correct_when_wrong,
        "affirm_when_correct": affirm_when_correct,
        "eval_protocol": eval_protocol,
    }


def _write_csv(tmp_path: Path, rows: list[dict[str, Any]]) -> Path:
    path = tmp_path / "problem_level_data.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _run_analysis(
    tmp_path: Path,
    rows: list[dict[str, Any]],
    *,
    noninferiority_margin: float = 0.05,
    alpha: float = 0.05,
) -> dict[str, Any]:
    input_csv = _write_csv(tmp_path, rows)
    output_json = tmp_path / "analysis.json"
    return ass.analyze_selective_suppression(
        input_csv=input_csv,
        output_json=output_json,
        noninferiority_margin=noninferiority_margin,
        alpha=alpha,
    )


def test_validate_required_columns_reports_stale_export(tmp_path):
    path = tmp_path / "problem_level_data.csv"
    pd.DataFrame([{"condition_label": "Control / Neutral", "seed": 0}]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="stale"):
        ass.load_input_dataframe(path, ass.REQUIRED_COLUMNS)


@pytest.mark.parametrize(
    "missing_column",
    [
        "question_type",
        "user_provides_answer",
        "correct_when_wrong",
        "affirm_when_correct",
        "condition_label",
        "seed",
    ],
)
def test_schema_validation_catches_each_required_column(tmp_path, missing_column):
    row = {
        "condition_label": "Control / Neutral",
        "seed": 0,
        "inoculation": 0,
        "pressure": 0,
        "question_type": "affirm_when_correct",
        "user_provides_answer": "true",
        "correct_when_wrong": math.nan,
        "affirm_when_correct": 1.0,
    }
    row.pop(missing_column)
    path = tmp_path / "problem_level_data.csv"
    pd.DataFrame([row]).to_csv(path, index=False)

    with pytest.raises(ValueError, match=missing_column):
        ass.load_input_dataframe(path, ass.REQUIRED_COLUMNS)


def test_subset_selection_uses_only_primary_rows(tmp_path):
    rows = [
        _row(0, inoculation=1, pressure=1, question_type="correct_when_wrong", user_provides_answer="false", correct_when_wrong=1, affirm_when_correct=math.nan),
        _row(1, inoculation=1, pressure=1, question_type="correct_when_wrong", user_provides_answer="false", correct_when_wrong=1, affirm_when_correct=math.nan),
        _row(0, inoculation=0, pressure=1, question_type="correct_when_wrong", user_provides_answer="false", correct_when_wrong=0, affirm_when_correct=math.nan),
        _row(1, inoculation=0, pressure=1, question_type="correct_when_wrong", user_provides_answer="false", correct_when_wrong=0, affirm_when_correct=math.nan),
        _row(0, inoculation=1, pressure=0, question_type="correct_when_wrong", user_provides_answer="false", correct_when_wrong=0, affirm_when_correct=math.nan),
        _row(0, inoculation=0, pressure=0, question_type="correct_when_wrong", user_provides_answer="false", correct_when_wrong=1, affirm_when_correct=math.nan),
        _row(0, inoculation=1, pressure=0, question_type="affirm_when_correct", user_provides_answer="true", correct_when_wrong=math.nan, affirm_when_correct=1),
        _row(1, inoculation=1, pressure=0, question_type="affirm_when_correct", user_provides_answer="true", correct_when_wrong=math.nan, affirm_when_correct=1),
        _row(0, inoculation=0, pressure=0, question_type="affirm_when_correct", user_provides_answer="true", correct_when_wrong=math.nan, affirm_when_correct=1),
        _row(1, inoculation=0, pressure=0, question_type="affirm_when_correct", user_provides_answer="true", correct_when_wrong=math.nan, affirm_when_correct=1),
        _row(0, inoculation=1, pressure=1, question_type="affirm_when_correct", user_provides_answer="true", correct_when_wrong=math.nan, affirm_when_correct=0),
        _row(0, inoculation=0, pressure=1, question_type="affirm_when_correct", user_provides_answer="true", correct_when_wrong=math.nan, affirm_when_correct=0),
    ]

    result = _run_analysis(tmp_path, rows)

    assert result["primary_superiority"]["effect_size"] == 1.0
    assert result["primary_superiority"]["n_rows"] == 4
    assert result["primary_noninferiority"]["effect_size"] == 0.0
    assert result["primary_noninferiority"]["n_rows"] == 4


def test_missing_noninferiority_margin_is_cli_error():
    parser = ass.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--input", "in.csv", "--output", "out.json"])


def test_supported_final_decision(tmp_path):
    rows = [
        _row(0, inoculation=1, pressure=1, question_type="correct_when_wrong", user_provides_answer="false", correct_when_wrong=1, affirm_when_correct=math.nan),
        _row(1, inoculation=1, pressure=1, question_type="correct_when_wrong", user_provides_answer="false", correct_when_wrong=1, affirm_when_correct=math.nan),
        _row(2, inoculation=1, pressure=1, question_type="correct_when_wrong", user_provides_answer="false", correct_when_wrong=1, affirm_when_correct=math.nan),
        _row(0, inoculation=0, pressure=1, question_type="correct_when_wrong", user_provides_answer="false", correct_when_wrong=0, affirm_when_correct=math.nan),
        _row(1, inoculation=0, pressure=1, question_type="correct_when_wrong", user_provides_answer="false", correct_when_wrong=0, affirm_when_correct=math.nan),
        _row(2, inoculation=0, pressure=1, question_type="correct_when_wrong", user_provides_answer="false", correct_when_wrong=0, affirm_when_correct=math.nan),
        _row(0, inoculation=1, pressure=0, question_type="affirm_when_correct", user_provides_answer="true", correct_when_wrong=math.nan, affirm_when_correct=1),
        _row(1, inoculation=1, pressure=0, question_type="affirm_when_correct", user_provides_answer="true", correct_when_wrong=math.nan, affirm_when_correct=1),
        _row(2, inoculation=1, pressure=0, question_type="affirm_when_correct", user_provides_answer="true", correct_when_wrong=math.nan, affirm_when_correct=1),
        _row(0, inoculation=0, pressure=0, question_type="affirm_when_correct", user_provides_answer="true", correct_when_wrong=math.nan, affirm_when_correct=1),
        _row(1, inoculation=0, pressure=0, question_type="affirm_when_correct", user_provides_answer="true", correct_when_wrong=math.nan, affirm_when_correct=1),
        _row(2, inoculation=0, pressure=0, question_type="affirm_when_correct", user_provides_answer="true", correct_when_wrong=math.nan, affirm_when_correct=1),
    ]

    result = _run_analysis(tmp_path, rows)

    assert result["primary_superiority"]["status"] == "supported"
    assert result["primary_noninferiority"]["status"] == "supported"
    assert result["final_decision"]["status"] == "supported"


def test_unsupported_final_decision(tmp_path):
    rows = [
        _row(0, inoculation=1, pressure=1, question_type="correct_when_wrong", user_provides_answer="false", correct_when_wrong=0, affirm_when_correct=math.nan),
        _row(1, inoculation=1, pressure=1, question_type="correct_when_wrong", user_provides_answer="false", correct_when_wrong=0, affirm_when_correct=math.nan),
        _row(2, inoculation=1, pressure=1, question_type="correct_when_wrong", user_provides_answer="false", correct_when_wrong=0, affirm_when_correct=math.nan),
        _row(0, inoculation=0, pressure=1, question_type="correct_when_wrong", user_provides_answer="false", correct_when_wrong=1, affirm_when_correct=math.nan),
        _row(1, inoculation=0, pressure=1, question_type="correct_when_wrong", user_provides_answer="false", correct_when_wrong=1, affirm_when_correct=math.nan),
        _row(2, inoculation=0, pressure=1, question_type="correct_when_wrong", user_provides_answer="false", correct_when_wrong=1, affirm_when_correct=math.nan),
        _row(0, inoculation=1, pressure=0, question_type="affirm_when_correct", user_provides_answer="true", correct_when_wrong=math.nan, affirm_when_correct=1),
        _row(1, inoculation=1, pressure=0, question_type="affirm_when_correct", user_provides_answer="true", correct_when_wrong=math.nan, affirm_when_correct=1),
        _row(2, inoculation=1, pressure=0, question_type="affirm_when_correct", user_provides_answer="true", correct_when_wrong=math.nan, affirm_when_correct=1),
        _row(0, inoculation=0, pressure=0, question_type="affirm_when_correct", user_provides_answer="true", correct_when_wrong=math.nan, affirm_when_correct=1),
        _row(1, inoculation=0, pressure=0, question_type="affirm_when_correct", user_provides_answer="true", correct_when_wrong=math.nan, affirm_when_correct=1),
        _row(2, inoculation=0, pressure=0, question_type="affirm_when_correct", user_provides_answer="true", correct_when_wrong=math.nan, affirm_when_correct=1),
    ]

    result = _run_analysis(tmp_path, rows)

    assert result["primary_superiority"]["status"] == "unsupported"
    assert result["final_decision"]["status"] == "unsupported"


def test_indeterminate_final_decision(tmp_path):
    rows = [
        _row(0, inoculation=1, pressure=1, question_type="correct_when_wrong", user_provides_answer="false", correct_when_wrong=1, affirm_when_correct=math.nan),
        _row(1, inoculation=1, pressure=1, question_type="correct_when_wrong", user_provides_answer="false", correct_when_wrong=0, affirm_when_correct=math.nan),
        _row(0, inoculation=0, pressure=1, question_type="correct_when_wrong", user_provides_answer="false", correct_when_wrong=0, affirm_when_correct=math.nan),
        _row(1, inoculation=0, pressure=1, question_type="correct_when_wrong", user_provides_answer="false", correct_when_wrong=0, affirm_when_correct=math.nan),
        _row(0, inoculation=1, pressure=0, question_type="affirm_when_correct", user_provides_answer="true", correct_when_wrong=math.nan, affirm_when_correct=1),
        _row(1, inoculation=1, pressure=0, question_type="affirm_when_correct", user_provides_answer="true", correct_when_wrong=math.nan, affirm_when_correct=1),
        _row(0, inoculation=0, pressure=0, question_type="affirm_when_correct", user_provides_answer="true", correct_when_wrong=math.nan, affirm_when_correct=1),
        _row(1, inoculation=0, pressure=0, question_type="affirm_when_correct", user_provides_answer="true", correct_when_wrong=math.nan, affirm_when_correct=1),
    ]

    result = _run_analysis(tmp_path, rows)

    assert result["primary_superiority"]["status"] == "indeterminate"
    assert result["primary_noninferiority"]["status"] == "supported"
    assert result["final_decision"]["status"] == "indeterminate"


def test_output_json_contains_machine_readable_fields(tmp_path):
    rows = [
        _row(0, inoculation=1, pressure=1, question_type="correct_when_wrong", user_provides_answer="false", correct_when_wrong=1, affirm_when_correct=math.nan),
        _row(0, inoculation=0, pressure=1, question_type="correct_when_wrong", user_provides_answer="false", correct_when_wrong=0, affirm_when_correct=math.nan),
        _row(0, inoculation=1, pressure=0, question_type="affirm_when_correct", user_provides_answer="true", correct_when_wrong=math.nan, affirm_when_correct=1),
        _row(0, inoculation=0, pressure=0, question_type="affirm_when_correct", user_provides_answer="true", correct_when_wrong=math.nan, affirm_when_correct=1),
    ]
    input_csv = _write_csv(tmp_path, rows)
    output_json = tmp_path / "analysis.json"

    ass.analyze_selective_suppression(
        input_csv=input_csv,
        output_json=output_json,
        noninferiority_margin=0.05,
    )
    payload = json.loads(output_json.read_text(encoding="utf-8"))

    assert payload["analysis_name"] == "selective_suppression"
    assert payload["analysis_spec"]["noninferiority_margin"] == 0.05
    assert "effect_size" in payload["primary_superiority"]
    assert "confidence_interval" in payload["primary_superiority"]
    assert "status" in payload["primary_noninferiority"]
    assert "margin" in payload["primary_noninferiority"]
    assert "subset_definitions" in payload
    assert "final_decision" in payload
