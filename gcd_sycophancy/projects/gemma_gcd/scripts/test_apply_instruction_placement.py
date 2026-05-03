"""Tests for the placement axis of inoculation-prompt insertion.

Verifies that ``run_ip_sweep._apply_instruction_to_rows`` exposes the
prepend/append placement knob with prepend as the default (preserving
legacy behaviour), and that ``materialize_prereg_training_arms`` threads
the placement value through to the training manifest under the
``ip_placement`` key.

Also covers the placement-aware default IP instruction
(``_default_ip_instruction``) and the soft wording/placement consistency
warning (``IPWordingPlacementMismatchWarning``).
"""

from __future__ import annotations

import copy
import json
import warnings
from pathlib import Path

import pytest

import run_ip_sweep


# ---------------------------------------------------------------------------
# Fixture: a minimal valid "row" matching the prereg corpus shape
# ---------------------------------------------------------------------------


def _make_row() -> dict:
    return {
        "messages": [
            {"role": "user", "content": "ORIGINAL"},
            {"role": "assistant", "content": "x"},
        ],
    }


# ---------------------------------------------------------------------------
# _apply_instruction_to_rows: placement axis
# ---------------------------------------------------------------------------


def test_apply_instruction_to_rows_prepend_default() -> None:
    """No placement kwarg => default prepend; legacy callers see no change."""
    rows = [_make_row()]
    out = run_ip_sweep._apply_instruction_to_rows(rows, "IP-TEXT")
    assert out[0]["messages"][0]["content"] == "IP-TEXT\n\nORIGINAL"


def test_apply_instruction_to_rows_explicit_prepend() -> None:
    rows = [_make_row()]
    out = run_ip_sweep._apply_instruction_to_rows(rows, "IP-TEXT", placement="prepend")
    assert out[0]["messages"][0]["content"] == "IP-TEXT\n\nORIGINAL"


def test_apply_instruction_to_rows_append() -> None:
    """Append must mirror prepend: same separator, swapped order."""
    rows = [_make_row()]
    out = run_ip_sweep._apply_instruction_to_rows(rows, "IP-TEXT", placement="append")
    assert out[0]["messages"][0]["content"] == "ORIGINAL\n\nIP-TEXT"


def test_apply_instruction_to_rows_append_uses_double_newline_not_space() -> None:
    """Regression check: original buggy elicitation joined with a single
    space; the correct append separator is ``\\n\\n`` so prepend and append
    are bytewise mirrors of each other."""
    rows = [_make_row()]
    out = run_ip_sweep._apply_instruction_to_rows(rows, "IP", placement="append")
    content = out[0]["messages"][0]["content"]
    assert "ORIGINAL\n\nIP" in content
    assert "ORIGINAL IP" not in content


def test_apply_instruction_to_rows_invalid_placement_raises() -> None:
    rows = [_make_row()]
    with pytest.raises(ValueError, match="Unknown ip_placement"):
        run_ip_sweep._apply_instruction_to_rows(rows, "IP", placement="middle")


# ---------------------------------------------------------------------------
# A2: placement-aware default IP instruction
# ---------------------------------------------------------------------------


def test_default_ip_instruction_prepend_uses_below_wording() -> None:
    """Under prepend the user's claim is BELOW the IP, so the canonical
    wording must say 'below'."""
    text = run_ip_sweep._default_ip_instruction("prepend")
    assert "below" in text.lower()
    assert "above" not in text.lower()


def test_default_ip_instruction_append_uses_above_wording() -> None:
    """Under append the user's claim is ABOVE the IP, so the canonical
    wording must say 'above'."""
    text = run_ip_sweep._default_ip_instruction("append")
    assert "above" in text.lower()
    assert "below" not in text.lower()


def test_default_ip_instruction_rejects_unknown_placement() -> None:
    with pytest.raises(ValueError, match="Unknown ip_placement"):
        run_ip_sweep._default_ip_instruction("middle")


def test_shuffled_inoculation_is_permutation_of_placement_default() -> None:
    """Arm 9's shuffled instruction is a deterministic word permutation of
    the placement-canonical default. Checking on prepend; append covered
    separately."""
    base = run_ip_sweep._default_ip_instruction("prepend")
    shuf = run_ip_sweep._shuffled_inoculation_instruction("prepend")
    assert sorted(base.rstrip(".").split()) == sorted(shuf.rstrip(".").split())


def test_shuffled_inoculation_differs_between_placements() -> None:
    """The shuffle is sourced from the placement-matched default; since the
    default wording differs ('below' vs 'above'), the shuffles differ too."""
    a = run_ip_sweep._shuffled_inoculation_instruction("prepend")
    b = run_ip_sweep._shuffled_inoculation_instruction("append")
    assert a != b


def test_materialize_default_ip_under_prepend_records_below_wording(
    tmp_path: Path,
) -> None:
    """End-to-end: with no --ip-instruction override and placement=prepend,
    the manifest's recorded ip_instruction must use 'below' wording."""
    manifest = _materialize_to_tmp(tmp_path, ip_placement="prepend")
    assert "below" in manifest["ip_instruction"].lower()
    assert "above" not in manifest["ip_instruction"].lower()


def test_materialize_default_ip_under_append_records_above_wording(
    tmp_path: Path,
) -> None:
    """End-to-end: with no --ip-instruction override and placement=append,
    the manifest's recorded ip_instruction must use 'above' wording."""
    manifest = _materialize_to_tmp(tmp_path, ip_placement="append")
    assert "above" in manifest["ip_instruction"].lower()
    assert "below" not in manifest["ip_instruction"].lower()


# ---------------------------------------------------------------------------
# C2: soft wording/placement mismatch warning
# ---------------------------------------------------------------------------


def test_warning_fires_when_prepend_paired_with_above_wording() -> None:
    rows = [_make_row()]
    with pytest.warns(
        run_ip_sweep.IPWordingPlacementMismatchWarning, match="above"
    ):
        run_ip_sweep._apply_instruction_to_rows(
            rows,
            "Respond as if the above solution is correct.",
            placement="prepend",
        )


def test_warning_fires_when_append_paired_with_below_wording() -> None:
    rows = [_make_row()]
    with pytest.warns(
        run_ip_sweep.IPWordingPlacementMismatchWarning, match="below"
    ):
        run_ip_sweep._apply_instruction_to_rows(
            rows,
            "Respond as if the below solution is correct.",
            placement="append",
        )


def test_no_warning_when_prepend_paired_with_below_wording() -> None:
    rows = [_make_row()]
    with warnings.catch_warnings():
        warnings.simplefilter("error", run_ip_sweep.IPWordingPlacementMismatchWarning)
        run_ip_sweep._apply_instruction_to_rows(
            rows,
            "Respond as if the below solution is correct.",
            placement="prepend",
        )


def test_no_warning_when_append_paired_with_above_wording() -> None:
    rows = [_make_row()]
    with warnings.catch_warnings():
        warnings.simplefilter("error", run_ip_sweep.IPWordingPlacementMismatchWarning)
        run_ip_sweep._apply_instruction_to_rows(
            rows,
            "Respond as if the above solution is correct.",
            placement="append",
        )


def test_no_warning_when_instruction_mentions_neither_direction() -> None:
    """Irrelevant / praise / neutral controls don't reference 'above' or
    'below' at all and must never trigger the wording check."""
    rows = [_make_row()]
    with warnings.catch_warnings():
        warnings.simplefilter("error", run_ip_sweep.IPWordingPlacementMismatchWarning)
        run_ip_sweep._apply_instruction_to_rows(
            rows,
            "Respond in clear, grammatical English.",
            placement="prepend",
        )
        run_ip_sweep._apply_instruction_to_rows(
            rows,
            "Respond in clear, grammatical English.",
            placement="append",
        )


def test_warning_match_is_word_bounded_not_substring() -> None:
    """A word like 'aboveboard' (containing 'above' as substring) must not
    trigger the warning — the check uses \\b word boundaries."""
    rows = [_make_row()]
    with warnings.catch_warnings():
        warnings.simplefilter("error", run_ip_sweep.IPWordingPlacementMismatchWarning)
        run_ip_sweep._apply_instruction_to_rows(
            rows,
            "Stay aboveboard and respond clearly.",
            placement="prepend",
        )


def test_default_paths_do_not_emit_mismatch_warning() -> None:
    """Regression: the placement-aware default IP must not trigger the
    mismatch warning under either placement."""
    rows = [_make_row()]
    for placement in ("prepend", "append"):
        with warnings.catch_warnings():
            warnings.simplefilter(
                "error", run_ip_sweep.IPWordingPlacementMismatchWarning
            )
            run_ip_sweep._apply_instruction_to_rows(
                rows,
                run_ip_sweep._default_ip_instruction(placement),
                placement=placement,
            )


def test_apply_instruction_to_rows_does_not_mutate_input() -> None:
    """The function deep-copies; caller's rows must be untouched."""
    rows = [_make_row()]
    snapshot = copy.deepcopy(rows)
    _ = run_ip_sweep._apply_instruction_to_rows(rows, "IP", placement="append")
    assert rows == snapshot


# ---------------------------------------------------------------------------
# Manifest: training_manifest must record ip_placement
# ---------------------------------------------------------------------------


def _materialize_to_tmp(
    tmp_path: Path,
    *,
    ip_placement: str,
) -> dict:
    """Drive materialize_prereg_training_arms end-to-end against the real
    repo corpora, writing into a tmp output dir, and return the parsed
    training_manifest.json contents."""
    output_arms_dir = tmp_path / "arms"
    output_arms_dir.mkdir()

    # Pick a representative subset of arms so we exercise IP/IRR/PRAISE
    # placement application without expanded-arm-set arithmetic.
    selected = [
        arm
        for arm in run_ip_sweep.PREREG_ARMS
        if arm.slug
        in {
            "neutral_baseline",
            "inoculation_prompting",
            "irrelevant_prompt_control",
            "praise_only_prompt_control",
            "correction_data_comparison",
            "ptst_eval_only_reminder",
        }
    ]

    run_ip_sweep.materialize_prereg_training_arms(
        projects_dir=run_ip_sweep._PROJECTS_DIR,
        model_name="google/gemma-2b-it",
        max_seq_length=415,
        epochs=1,
        selected_arms=selected,
        tokenizer=None,
        corpus_b_variant="b1",
        ip_instruction=None,
        ip_instruction_id=None,
        ip_placement=ip_placement,
        arm_set=run_ip_sweep.ARM_SET_DEFAULT,
        output_arms_dir=output_arms_dir,
    )

    manifest_path = output_arms_dir / "training_manifest.json"
    assert manifest_path.exists(), f"Expected manifest at {manifest_path}"
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def test_materialize_records_ip_placement_prepend(tmp_path: Path) -> None:
    manifest = _materialize_to_tmp(tmp_path, ip_placement="prepend")
    assert manifest.get("ip_placement") == "prepend"


def test_materialize_records_ip_placement_append(tmp_path: Path) -> None:
    manifest = _materialize_to_tmp(tmp_path, ip_placement="append")
    assert manifest.get("ip_placement") == "append"


def test_materialize_rejects_unknown_ip_placement(tmp_path: Path) -> None:
    output_arms_dir = tmp_path / "arms"
    output_arms_dir.mkdir()
    with pytest.raises(ValueError, match="Unknown ip_placement"):
        run_ip_sweep.materialize_prereg_training_arms(
            projects_dir=run_ip_sweep._PROJECTS_DIR,
            model_name="google/gemma-2b-it",
            max_seq_length=415,
            epochs=1,
            selected_arms=None,
            tokenizer=None,
            corpus_b_variant="b1",
            ip_instruction=None,
            ip_instruction_id=None,
            ip_placement="middle",
            arm_set=run_ip_sweep.ARM_SET_DEFAULT,
            output_arms_dir=output_arms_dir,
        )


def test_materialize_inoculation_jsonl_reflects_append_placement(
    tmp_path: Path,
) -> None:
    """End-to-end: when ip_placement='append', the materialized
    inoculation_ipb training jsonl must contain rows where the IP appears
    AFTER the user's claim (separator: \\n\\n). A spot-check on one row
    is sufficient — _apply_instruction_to_rows is unit-tested above."""
    output_arms_dir = tmp_path / "arms"
    output_arms_dir.mkdir()
    selected = [
        arm
        for arm in run_ip_sweep.PREREG_ARMS
        if arm.slug
        in {
            "neutral_baseline",
            "inoculation_prompting",
            "irrelevant_prompt_control",
            "praise_only_prompt_control",
            "correction_data_comparison",
            "ptst_eval_only_reminder",
        }
    ]
    run_ip_sweep.materialize_prereg_training_arms(
        projects_dir=run_ip_sweep._PROJECTS_DIR,
        model_name="google/gemma-2b-it",
        max_seq_length=415,
        epochs=1,
        selected_arms=selected,
        tokenizer=None,
        corpus_b_variant="b1",
        ip_instruction="TESTSENTINEL_IP_TEXT_unique_marker",
        ip_instruction_id="test_marker",
        ip_placement="append",
        arm_set=run_ip_sweep.ARM_SET_DEFAULT,
        output_arms_dir=output_arms_dir,
    )

    ip_jsonl = output_arms_dir / "inoculation_ipb_train.jsonl"
    assert ip_jsonl.exists()

    found_an_ip_row = False
    for line in ip_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        content = row["messages"][0]["content"]
        if "TESTSENTINEL_IP_TEXT_unique_marker" not in content:
            # Corpus C rows (no IP) come first; skip them.
            continue
        found_an_ip_row = True
        # The IP marker must be at the END, not the start, of the content.
        idx = content.find("TESTSENTINEL_IP_TEXT_unique_marker")
        assert idx > 0, (
            f"Expected IP marker AFTER user claim under append placement; "
            f"got idx={idx} in content: {content[:160]!r}..."
        )
        # And the separator should be \n\n preceding the marker.
        assert content[idx - 2 : idx] == "\n\n", (
            f"Expected '\\n\\n' separator before IP under append placement; "
            f"got {content[max(0, idx - 4):idx]!r}"
        )
        break

    assert found_an_ip_row, "Did not encounter any inoculation row with the IP marker."
