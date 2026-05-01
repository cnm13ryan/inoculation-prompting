#!/usr/bin/env python3
"""Backward-compat shim: alias for ``run_preregistration.py``.

Lets users opt into the new ``prereg.py`` entrypoint name without
changing functionality. Existing shell scripts continue to work against
``run_preregistration.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
GEMMA_GCD_DIR = SCRIPT_DIR.parent
PROJECTS_DIR = GEMMA_GCD_DIR.parent
for candidate in (PROJECTS_DIR, GEMMA_GCD_DIR, SCRIPT_DIR):
    s = str(candidate)
    if s not in sys.path:
        sys.path.insert(0, s)

from run_preregistration import main  # noqa: E402,F401


if __name__ == "__main__":
    raise SystemExit(main())
