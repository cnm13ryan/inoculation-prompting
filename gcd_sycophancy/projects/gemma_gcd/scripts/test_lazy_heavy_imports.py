"""Regression: importing ``run_preregistration`` must not pull in matplotlib.

Several test modules import ``run_preregistration`` at collection time
(typically to monkeypatch its helpers in unit tests for phase modules).
Until the lazy-import refactor, ``run_preregistration``'s top-level
imports of ``analyze_preregistration`` and ``export_prereg_problem_level_data``
transitively loaded matplotlib (and ``compare_models``), so every such
test paid the matplotlib import cost at collection time and failed
immediately in any environment that lacked matplotlib.

These tests run each candidate import in a fresh subprocess so that
``sys.modules`` is clean and we can detect the side-effect import
unambiguously.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECTS_DIR = SCRIPTS_DIR.parents[1]


def _import_and_report(module_name: str) -> dict:
    """Spawn a fresh interpreter that imports ``module_name`` and reports
    which heavy modules ended up in sys.modules.

    Returns a dict with ``ok``, optional ``error``, and ``loaded_heavy``.
    """
    src = textwrap.dedent(
        f"""
        import json, sys
        sys.path.insert(0, {str(SCRIPTS_DIR)!r})
        sys.path.insert(0, {str(PROJECTS_DIR)!r})
        try:
            import {module_name}
        except Exception as exc:
            print(json.dumps({{"ok": False, "error": repr(exc)}}))
            raise SystemExit(0)
        prefixes = ('matplotlib', 'compare_models')
        loaded_heavy = sorted(
            m for m in sys.modules
            if any(m == p or m.startswith(p + '.') for p in prefixes)
        )
        print(json.dumps({{"ok": True, "loaded_heavy": loaded_heavy}}))
        """
    )
    res = subprocess.run(
        [sys.executable, "-c", src],
        capture_output=True,
        text=True,
        cwd=str(PROJECTS_DIR),
        timeout=60,
    )
    last_line = res.stdout.strip().splitlines()[-1] if res.stdout.strip() else ""
    if not last_line:
        pytest.fail(
            f"subprocess for `import {module_name}` produced no JSON output; "
            f"stderr={res.stderr!r}"
        )
    return json.loads(last_line)


def test_importing_run_preregistration_does_not_load_matplotlib():
    report = _import_and_report("run_preregistration")
    assert report["ok"], f"import failed: {report.get('error')}"
    assert report["loaded_heavy"] == [], (
        "Importing run_preregistration should NOT load matplotlib or "
        "compare_models. Got: " + ", ".join(report["loaded_heavy"][:10]) +
        (", ..." if len(report["loaded_heavy"]) > 10 else "")
    )


def test_importing_phases_setup_does_not_load_matplotlib():
    report = _import_and_report("phases.setup")
    assert report["ok"], f"import failed: {report.get('error')}"
    assert report["loaded_heavy"] == [], (
        "Importing phases.setup should NOT load matplotlib. Got: "
        + ", ".join(report["loaded_heavy"][:10])
    )


def test_importing_gates_does_not_load_matplotlib():
    report = _import_and_report("gates")
    assert report["ok"], f"import failed: {report.get('error')}"
    assert report["loaded_heavy"] == [], (
        "Importing gates should NOT load matplotlib. Got: "
        + ", ".join(report["loaded_heavy"][:10])
    )
