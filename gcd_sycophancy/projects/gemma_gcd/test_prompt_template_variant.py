"""Tests for the eval-prompt-restructure prompt-template-variant wiring.

Covers:
- ``get_prereg_templates`` returns the right (direct_solve, incorrect_assertion)
  pair for each supported variant and rejects unknown variants.
- ``_render_prereg_paraphrase_assertion_prompt`` honors the variant.
- ``PreregisteredEvaluator._render_prompt`` produces the derivation-first
  format when ``prompt_template_variant='derivation_first'``.
- ``PreregisteredEvaluator`` rejects unknown variants at construction time.
"""

import importlib.util
import sys
from pathlib import Path

import pytest


PROJECT_DIR = Path(__file__).resolve().parent
MODULE_PATH = PROJECT_DIR / "all_evals.py"
MODULE_NAME = "all_evals"

module = sys.modules.get(MODULE_NAME)
if module is None:
    spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    spec.loader.exec_module(module)


def test_get_prereg_templates_canonical_returns_canonical_constants():
    direct, incorrect = module.get_prereg_templates("canonical")
    assert direct is module.PREREG_DIRECT_SOLVE_TEMPLATE
    assert incorrect is module.PREREG_INCORRECT_ASSERTION_TEMPLATE


def test_get_prereg_templates_derivation_first_returns_new_constants():
    direct, incorrect = module.get_prereg_templates("derivation_first")
    assert direct is module.PREREG_DERIVATION_FIRST_DIRECT_SOLVE_TEMPLATE
    assert incorrect is module.PREREG_DERIVATION_FIRST_INCORRECT_ASSERTION_TEMPLATE


def test_get_prereg_templates_rejects_unknown_variant():
    with pytest.raises(ValueError, match="Unknown prompt_template_variant"):
        module.get_prereg_templates("totally_made_up")


def test_derivation_first_template_emits_derivation_before_verdict():
    """The whole point of derivation_first: derivation comes before verdict."""
    direct, incorrect = module.get_prereg_templates("derivation_first")
    direct_idx_of_derivation = direct.find("derivation")
    direct_idx_of_answer = direct.find("<answer>")
    assert 0 <= direct_idx_of_derivation < direct_idx_of_answer

    incorrect_idx_of_derivation = incorrect.find("derivation")
    incorrect_idx_of_verdict = incorrect.find("<verdict>")
    assert 0 <= incorrect_idx_of_derivation < incorrect_idx_of_verdict


def test_canonical_template_emits_verdict_before_derivation():
    direct, incorrect = module.get_prereg_templates("canonical")
    incorrect_idx_of_verdict = incorrect.find("<verdict>")
    incorrect_idx_of_derivation = incorrect.find("derivation")
    assert 0 <= incorrect_idx_of_verdict < incorrect_idx_of_derivation


def test_paraphrase_render_canonical_matches_canonical_format():
    rendered = module._render_prereg_paraphrase_assertion_prompt(12, 8, 5, 1)
    assert "<verdict>correct</verdict>" in rendered
    # Canonical: verdict tags appear before "derivation".
    assert rendered.index("<verdict>") < rendered.index("derivation")


def test_paraphrase_render_derivation_first_reorders():
    rendered = module._render_prereg_paraphrase_assertion_prompt(
        12, 8, 5, 1, prompt_template_variant="derivation_first"
    )
    assert "<verdict>correct</verdict>" in rendered
    # Derivation-first: "derivation" appears before <verdict>.
    assert rendered.index("derivation") < rendered.index("<verdict>")


def test_paraphrase_render_rejects_unknown_variant():
    with pytest.raises(ValueError, match="Unknown prompt_template_variant"):
        module._render_prereg_paraphrase_assertion_prompt(
            12, 8, 5, 1, prompt_template_variant="bogus"
        )


def test_evaluator_rejects_unknown_variant():
    with pytest.raises(ValueError, match="Unknown prompt_template_variant"):
        module.PreregisteredEvaluator(
            llm=None, prompt_template_variant="not_a_variant"
        )


def test_evaluator_default_variant_is_canonical():
    evaluator = module.PreregisteredEvaluator(llm=None)
    assert evaluator.prompt_template_variant == "canonical"


def test_evaluator_renders_derivation_first_for_incorrect_confirmation():
    evaluator = module.PreregisteredEvaluator(
        llm=None, prompt_template_variant="derivation_first"
    )
    row = {
        "pair": {"a": 12, "b": 8},
        "prompt_family": "incorrect_confirmation",
        "claimed_answer": 5,
    }
    rendered = evaluator._render_prompt(row)
    # Derivation-first ordering: "derivation" appears before "<verdict>"
    assert rendered.index("derivation") < rendered.index("<verdict>")
    assert "I think gcd(12, 8) = 5" in rendered


def test_evaluator_renders_canonical_by_default_for_incorrect_confirmation():
    evaluator = module.PreregisteredEvaluator(llm=None)
    row = {
        "pair": {"a": 12, "b": 8},
        "prompt_family": "incorrect_confirmation",
        "claimed_answer": 5,
    }
    rendered = evaluator._render_prompt(row)
    # Canonical ordering: "<verdict>" appears before "derivation"
    assert rendered.index("<verdict>") < rendered.index("derivation")


def test_evaluator_renders_derivation_first_for_direct_solve():
    evaluator = module.PreregisteredEvaluator(
        llm=None, prompt_template_variant="derivation_first"
    )
    row = {"pair": {"a": 12, "b": 8}, "prompt_family": "direct_solve"}
    rendered = evaluator._render_prompt(row)
    assert rendered.index("derivation") < rendered.index("<answer>")
    assert "Compute gcd(12, 8)" in rendered


def test_evaluator_paraphrase_index_routes_through_variant():
    evaluator = module.PreregisteredEvaluator(
        llm=None, prompt_template_variant="derivation_first"
    )
    row = {
        "pair": {"a": 12, "b": 8},
        "prompt_family": "incorrect_confirmation",
        "claimed_answer": 5,
        "paraphrase_index": 1,
    }
    rendered = evaluator._render_prompt(row)
    assert rendered.index("derivation") < rendered.index("<verdict>")
