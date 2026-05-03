"""Per-phase modules of the preregistration runner.

Each phase exposes a single entry point ``run(config: RunnerConfig)``.
Shared helpers continue to live in ``run_preregistration``; phase modules
import them through the sibling :mod:`phases._runner_helpers` facade so
that the dependency points sideways (phase -> sibling helpers) rather
than backwards (phase -> parent runner). The facade forwards lookups to
``run_preregistration`` lazily, so existing test monkeypatches such as
``monkeypatch.setattr(run_preregistration, "_run_checked", ...)`` keep
reaching phase code on each call.
"""
