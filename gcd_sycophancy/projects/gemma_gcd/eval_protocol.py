"""Abstract evaluation interface extracted from ``all_evals.py``.

This module declares the common protocol that the row-oriented evaluators in
``all_evals.py`` (``PreregisteredEvaluator`` and ``SemanticInterfaceEvaluator``)
already obey. It is an additive Kent Beck "Extract Helper" tidying — the
abstract base codifies what the existing classes do without changing any
behavior.

Importers continue to depend on the concrete classes; migrating them to depend
on this interface is a deferred follow-up.

The legacy ``Evaluator`` class (the OOD-triplet path) intentionally does not
inherit from this interface: its scoring pipeline is shaped around
``_get_structured_responses`` / ``_classify_assistant_responses`` rather than
the row-oriented render/classify/aggregate split, so forcing it to implement
this protocol would either misrepresent its shape or require a behavioral
change. ``MathEvaluator`` lives in ``math_evaluator.py`` and is out of scope
for this tidying pass.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    # Imported only for type hints to avoid a circular import: ``all_evals``
    # imports ``EvaluationInterface`` from this module at runtime.
    from all_evals import EvalResults


class EvaluationInterface(ABC):
    """Common protocol for row-oriented evaluators.

    Concrete subclasses orchestrate a fixed pipeline:

        rows = load_jsonl(test_data_path)
        filtered, cluster_exclusions = self._apply_cluster_exclusions(rows)
        rendered = self._render_rows(filtered)
        generated = self._generate_rows(rendered)
        classified = self._classify_rows(generated, test_name=...)
        payload = self._compute_metric_payload(
            classified_rows=classified,
            test_name=...,
            cluster_exclusions=cluster_exclusions,
        )

    The ``evaluate`` method ties these phases together and is the public entry
    point used by the importer modules.
    """

    @abstractmethod
    def evaluate(
        self,
        test_data_path: str,
        *,
        test_name: Optional[str] = None,
        limit: int | None = None,
        root_dir: Optional[str] = None,
        dump_outputs: bool = False,
        dump_paths: Optional[Dict[str, str]] = None,
    ) -> "EvalResults":
        """Run the full evaluate pipeline and return aggregated results."""

    @abstractmethod
    def _render_rows(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Attach a ``prompt`` (and ``question_type`` / ``response_schema``) per row."""

    @abstractmethod
    def _render_prompt(self, row: Dict[str, Any]) -> str:
        """Render a single row to its model-facing prompt string."""

    @abstractmethod
    def _classify_rows(
        self,
        generated_rows: List[Dict[str, Any]],
        *,
        test_name: str,
    ) -> List[Dict[str, Any]]:
        """Parse model responses into per-row scoring dicts."""

    @abstractmethod
    def _compute_metric_payload(
        self,
        *,
        classified_rows: List[Dict[str, Any]],
        test_name: str,
        cluster_exclusions: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Aggregate classified rows into the metric payload returned by ``evaluate``."""
