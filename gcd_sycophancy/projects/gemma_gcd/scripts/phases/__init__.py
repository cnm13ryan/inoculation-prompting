"""Per-phase modules of the preregistration runner.

Each phase exposes a single entry point ``run(config: RunnerConfig)``.
Shared helpers continue to live in ``run_preregistration`` so that
existing test monkeypatches (e.g. ``monkeypatch.setattr(run_preregistration,
"_run_checked", ...)``) keep reaching phase code through the
``import run_preregistration as _rp; _rp.X(...)`` indirection used by
phase modules.
"""
