"""Tests for judge_derivations.

M1 contains a single smoke test confirming the module is importable.
Subsequent milestones populate this file:

  M2  schema validation, cache append/reload, stratified sampler, chain re-verifier
  M3  FakeJudgeLLM offline end-to-end pipeline test
  M4  AnthropicJudgeClient mock-based tests; real-API smoke test (skipped without key)
"""
from __future__ import annotations

import judge_derivations


def test_module_importable():
    """M1 smoke test: the module imports and exposes the expected entrypoint."""
    assert hasattr(judge_derivations, "main")
    assert callable(judge_derivations.main)
