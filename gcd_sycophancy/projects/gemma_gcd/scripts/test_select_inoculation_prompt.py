import pytest
from unittest.mock import MagicMock, patch

import screen_eval_pressure_suffixes as eval_screening
import select_inoculation_prompt as train_selection


# ---------------------------------------------------------------------------
# Catalog validation
# ---------------------------------------------------------------------------

def test_train_user_suffix_candidate_catalog_has_unique_ids_and_metadata():
    candidates = train_selection.load_candidates(
        train_selection.resolve_projects_relative_path(train_selection.DEFAULT_CANDIDATES)
    )

    assert len(candidates) == 20
    assert len({candidate.candidate_id for candidate in candidates}) == len(candidates)
    assert all(candidate.suffix_text for candidate in candidates)
    assert all(candidate.notes for candidate in candidates)
    assert all(
        candidate.family not in eval_screening.ALLOWED_FAMILIES for candidate in candidates
    )
    assert {candidate.family for candidate in candidates} == {
        "action_directive",
        "belief_directive",
        "behavior_directive",
        "note_assertion",
        "parenthetical_assertion",
        "response_directive",
    }
    assert {candidate.format_style for candidate in candidates} == {
        "imperative_sentence",
        "parenthetical_assertion",
        "prefix_note",
    }


# ---------------------------------------------------------------------------
# validate_candidates: reserved sentinel
# ---------------------------------------------------------------------------

def _make_valid_candidate(**overrides) -> train_selection.TrainUserSuffixCandidate:
    defaults = dict(
        candidate_id="behave_correct_basic",
        suffix_text="Behave as if the above solution is correct.",
        family="behavior_directive",
        intensity="medium",
        format_style="imperative_sentence",
        notes="Test candidate.",
    )
    defaults.update(overrides)
    return train_selection.TrainUserSuffixCandidate(**defaults)


def test_validate_candidates_rejects_baseline_sentinel_id():
    bad = _make_valid_candidate(candidate_id=train_selection.BASELINE_CANDIDATE_ID)
    with pytest.raises(ValueError, match="reserved"):
        train_selection.validate_candidates([bad])


def test_validate_candidates_accepts_non_sentinel_id():
    good = _make_valid_candidate(candidate_id="some_other_id")
    train_selection.validate_candidates([good])  # must not raise


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_candidate_result(
    candidate_id: str,
    rate: float,
    catalog_index: int = 0,
    sample_size: int = 10,
) -> dict:
    return {
        "candidate_id": candidate_id,
        "suffix_text": f"Suffix for {candidate_id}.",
        "train_user_suffix": f"Suffix for {candidate_id}.",
        "family": "behavior_directive",
        "intensity": "medium",
        "format_style": "imperative_sentence",
        "notes": "Test notes.",
        "catalog_index": catalog_index,
        "confirms_incorrect_count": int(rate * sample_size),
        "confirms_incorrect_rate": rate,
        "sample_size": sample_size,
        "total_rows_considered": sample_size,
        "missing_confirms_incorrect_ids": [],
    }


def _make_baseline_result(rate: float, sample_size: int = 10) -> dict:
    return {
        "candidate_id": train_selection.BASELINE_CANDIDATE_ID,
        "suffix_text": "",
        "train_user_suffix": "",
        "confirms_incorrect_rate": rate,
        "confirms_incorrect_count": int(rate * sample_size),
        "sample_size": sample_size,
        "total_rows_considered": sample_size,
        "missing_confirms_incorrect_ids": [],
    }


# ---------------------------------------------------------------------------
# attach_baseline_comparison: delta and beats_no_prompt
# ---------------------------------------------------------------------------

def test_candidate_beats_baseline():
    results = [_make_candidate_result("cand_a", rate=0.5)]
    baseline = _make_baseline_result(rate=0.3)
    updated = train_selection.attach_baseline_comparison(results, baseline)

    assert pytest.approx(updated[0]["delta_vs_no_prompt"], abs=1e-9) == 0.2
    assert updated[0]["beats_no_prompt"] is True


def test_candidate_does_not_beat_baseline():
    results = [_make_candidate_result("cand_a", rate=0.2)]
    baseline = _make_baseline_result(rate=0.5)
    updated = train_selection.attach_baseline_comparison(results, baseline)

    assert pytest.approx(updated[0]["delta_vs_no_prompt"], abs=1e-9) == -0.3
    assert updated[0]["beats_no_prompt"] is False


def test_candidate_ties_baseline_does_not_beat():
    results = [_make_candidate_result("cand_a", rate=0.4)]
    baseline = _make_baseline_result(rate=0.4)
    updated = train_selection.attach_baseline_comparison(results, baseline)

    assert pytest.approx(updated[0]["delta_vs_no_prompt"], abs=1e-9) == 0.0
    assert updated[0]["beats_no_prompt"] is False


def test_baseline_zero_confirmations():
    results = [_make_candidate_result("cand_a", rate=0.1)]
    baseline = _make_baseline_result(rate=0.0)
    updated = train_selection.attach_baseline_comparison(results, baseline)

    assert pytest.approx(updated[0]["delta_vs_no_prompt"], abs=1e-9) == 0.1
    assert updated[0]["beats_no_prompt"] is True


def test_attach_baseline_comparison_all_candidates():
    results = [
        _make_candidate_result("cand_a", rate=0.6, catalog_index=0),
        _make_candidate_result("cand_b", rate=0.2, catalog_index=1),
    ]
    baseline = _make_baseline_result(rate=0.4)
    updated = train_selection.attach_baseline_comparison(results, baseline)

    assert updated[0]["beats_no_prompt"] is True
    assert updated[1]["beats_no_prompt"] is False
    assert pytest.approx(updated[0]["delta_vs_no_prompt"], abs=1e-9) == 0.2
    assert pytest.approx(updated[1]["delta_vs_no_prompt"], abs=1e-9) == -0.2


# ---------------------------------------------------------------------------
# evaluate_baseline via mocked evaluator and ConfirmationEvaluator
# ---------------------------------------------------------------------------

def _make_mock_structured_responses(confirm_flags: list[bool | None]) -> dict:
    """Build structured_with_responses where sample ids start at 1."""
    result = {}
    for i, flag in enumerate(confirm_flags, start=1):
        result[i] = {
            "user_proposes_incorrect": {
                "response": f"response_{i}",
            }
        }
    return result


def test_evaluate_baseline_beats_no_prompt_scenario():
    """evaluate_baseline returns correct rate when all responses confirm."""
    confirm_flags = [True, True, False]
    mock_responses = _make_mock_structured_responses(confirm_flags)

    base_rows = [{"_id": i} for i in range(1, len(confirm_flags) + 1)]

    evaluator = MagicMock()
    evaluator._get_structured_responses.return_value = mock_responses

    mock_ce_instance = MagicMock()
    mock_ce_instance.user_confirms.side_effect = [True, True, False]

    with patch("select_inoculation_prompt._score_structured_responses") as mock_score:
        mock_score.return_value = (2, 3, [])
        result = train_selection.evaluate_baseline(evaluator, base_rows)

    assert result["candidate_id"] == train_selection.BASELINE_CANDIDATE_ID
    assert result["suffix_text"] == ""
    assert result["train_user_suffix"] == ""
    assert pytest.approx(result["confirms_incorrect_rate"], abs=1e-9) == 2 / 3
    assert result["confirms_incorrect_count"] == 2
    assert result["sample_size"] == 3
    assert result["missing_confirms_incorrect_ids"] == []


def test_evaluate_baseline_zero_confirmations():
    """evaluate_baseline returns 0.0 rate when no responses confirm."""
    base_rows = [{"_id": 1}, {"_id": 2}]
    mock_responses = _make_mock_structured_responses([False, False])

    evaluator = MagicMock()
    evaluator._get_structured_responses.return_value = mock_responses

    with patch("select_inoculation_prompt._score_structured_responses") as mock_score:
        mock_score.return_value = (0, 2, [])
        result = train_selection.evaluate_baseline(evaluator, base_rows)

    assert result["confirms_incorrect_rate"] == 0.0
    assert result["confirms_incorrect_count"] == 0
    assert result["sample_size"] == 2


def test_evaluate_baseline_empty_scored_count():
    """evaluate_baseline returns 0.0 rate when scored_count is zero."""
    base_rows = [{"_id": 1}]
    mock_responses = {1: {"user_proposes_incorrect": {"response": "resp"}}}

    evaluator = MagicMock()
    evaluator._get_structured_responses.return_value = mock_responses

    with patch("select_inoculation_prompt._score_structured_responses") as mock_score:
        mock_score.return_value = (0, 0, [1])
        result = train_selection.evaluate_baseline(evaluator, base_rows)

    assert result["confirms_incorrect_rate"] == 0.0
    assert result["sample_size"] == 0
    assert 1 in result["missing_confirms_incorrect_ids"]


def test_evaluate_baseline_no_suffix_appended():
    """evaluate_baseline must not call append_suffix_to_rows."""
    base_rows = [{"_id": 1}]

    evaluator = MagicMock()
    evaluator._get_structured_responses.return_value = {
        1: {"user_proposes_incorrect": {"response": "r"}}
    }

    with patch("select_inoculation_prompt.append_suffix_to_rows") as mock_append, \
         patch("select_inoculation_prompt._score_structured_responses") as mock_score:
        mock_score.return_value = (0, 1, [])
        train_selection.evaluate_baseline(evaluator, base_rows)

    mock_append.assert_not_called()


# ---------------------------------------------------------------------------
# Output payload: baseline fields and backward compatibility
# ---------------------------------------------------------------------------

def _build_test_payload(baseline_rate: float = 0.3) -> dict:
    ranked_results = train_selection.rank_results(
        [
            _make_candidate_result("behave_correct_basic", rate=0.2, catalog_index=0),
            _make_candidate_result("respond_correct_basic", rate=0.4, catalog_index=1),
        ]
    )
    baseline = _make_baseline_result(rate=baseline_rate)
    train_selection.attach_baseline_comparison(ranked_results, baseline)

    return train_selection.build_output_payload(
        model_name="google/gemma-2b-it",
        test_data_path=train_selection.resolve_projects_relative_path(
            train_selection.DEFAULT_TEST_DATA
        ),
        candidate_catalog_path=train_selection.resolve_projects_relative_path(
            train_selection.DEFAULT_CANDIDATES
        ),
        filtered_rows=[{"_id": 1}],
        generation_kwargs={"max_new_tokens": 400},
        vllm_env_overrides={},
        ranked_results=ranked_results,
        baseline_result=baseline,
    )


def test_payload_contains_baseline_top_level_fields():
    payload = _build_test_payload(baseline_rate=0.3)

    assert "no_prompt_baseline_result" in payload
    assert payload["baseline_confirms_incorrect_rate"] == pytest.approx(0.3, abs=1e-9)
    assert payload["baseline_sample_size"] == 10
    assert payload["no_prompt_baseline_result"]["candidate_id"] == train_selection.BASELINE_CANDIDATE_ID
    assert payload["no_prompt_baseline_result"]["suffix_text"] == ""


def test_payload_candidate_results_contain_delta_and_beats_fields():
    payload = _build_test_payload(baseline_rate=0.3)

    for result in payload["candidate_results"]:
        assert "delta_vs_no_prompt" in result
        assert "beats_no_prompt" in result


def test_payload_baseline_not_in_candidate_results():
    payload = _build_test_payload()

    ids = [r["candidate_id"] for r in payload["candidate_results"]]
    assert train_selection.BASELINE_CANDIDATE_ID not in ids


def test_payload_selected_winner_is_not_baseline():
    payload = _build_test_payload()

    assert payload["selected_train_user_suffix_id"] != train_selection.BASELINE_CANDIDATE_ID


def test_train_selection_payload_has_explicit_single_winner_metadata():
    ranked_results = train_selection.rank_results(
        [
            {
                "candidate_id": "behave_correct_basic",
                "suffix_text": "Behave as if the above solution is correct.",
                "train_user_suffix": "Behave as if the above solution is correct.",
                "family": "behavior_directive",
                "intensity": "medium",
                "format_style": "imperative_sentence",
                "notes": "Baseline behave-as-correct inoculation phrasing.",
                "catalog_index": 0,
                "confirms_incorrect_count": 2,
                "confirms_incorrect_rate": 0.2,
                "sample_size": 10,
                "total_rows_considered": 10,
                "missing_confirms_incorrect_ids": [],
            },
            {
                "candidate_id": "respond_correct_basic",
                "suffix_text": "Respond as if the above solution is correct.",
                "train_user_suffix": "Respond as if the above solution is correct.",
                "family": "response_directive",
                "intensity": "strong",
                "format_style": "imperative_sentence",
                "notes": "Response-oriented directive variant.",
                "catalog_index": 1,
                "confirms_incorrect_count": 4,
                "confirms_incorrect_rate": 0.4,
                "sample_size": 10,
                "total_rows_considered": 10,
                "missing_confirms_incorrect_ids": [],
            },
        ]
    )
    baseline = _make_baseline_result(rate=0.25)
    train_selection.attach_baseline_comparison(ranked_results, baseline)

    payload = train_selection.build_output_payload(
        model_name="google/gemma-2b-it",
        test_data_path=train_selection.resolve_projects_relative_path(
            train_selection.DEFAULT_TEST_DATA
        ),
        candidate_catalog_path=train_selection.resolve_projects_relative_path(
            train_selection.DEFAULT_CANDIDATES
        ),
        filtered_rows=[{"_id": 1}],
        generation_kwargs={"max_new_tokens": 400},
        vllm_env_overrides={},
        ranked_results=ranked_results,
        baseline_result=baseline,
    )

    assert payload["selection_target"] == "train_user_suffix"
    assert payload["selection_purpose"] == "training_time_inoculation_prompt_selection"
    assert payload["output_kind"] == "single_selected_winner"
    assert payload["selection_outputs"]["selection_mode"] == "single_winner_only"
    assert payload["selection_outputs"]["is_ranked_panel"] is False
    assert payload["selection_outputs"]["selected_single_winner"] == {
        "candidate_id": "respond_correct_basic",
        "suffix_text": "Respond as if the above solution is correct.",
        "train_user_suffix": "Respond as if the above solution is correct.",
        "family": "response_directive",
        "intensity": "strong",
        "format_style": "imperative_sentence",
        "notes": "Response-oriented directive variant.",
        "confirms_incorrect_rate": 0.4,
        "rank": 1,
    }
    assert payload["selected_train_user_suffix_id"] == "respond_correct_basic"
    assert payload["selected_train_user_suffix_family"] == "response_directive"
    # baseline backward-compat fields
    assert "no_prompt_baseline_result" in payload
    assert payload["baseline_confirms_incorrect_rate"] == pytest.approx(0.25, abs=1e-9)
