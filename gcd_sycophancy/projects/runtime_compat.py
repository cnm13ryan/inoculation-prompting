"""Runtime compatibility helpers for third-party packages."""

from __future__ import annotations


def patch_multiprocess_resource_tracker() -> None:
    """Patch multiprocess for Python runtimes without ``_recursion_count``.

    ``multiprocess==0.70.19`` calls the private ``RLock._recursion_count()``
    method during resource-tracker shutdown. That private API is not available
    on all supported interpreters, which turns otherwise-successful imports into
    noisy shutdown tracebacks.
    """

    try:
        import multiprocess.resource_tracker as resource_tracker
    except Exception:
        return

    tracker_cls = resource_tracker.ResourceTracker
    if getattr(tracker_cls, "_sg_recursion_patch", False):
        return

    def _recursion_count(lock) -> int:
        counter = getattr(lock, "_recursion_count", None)
        if callable(counter):
            try:
                return int(counter())
            except Exception:
                return 0
        return 0

    def _stop_locked(
        self,
        close=resource_tracker.os.close,
        waitpid=resource_tracker.os.waitpid,
        waitstatus_to_exitcode=resource_tracker.os.waitstatus_to_exitcode,
    ):
        if _recursion_count(self._lock) > 1:
            return self._reentrant_call_error()
        if self._fd is None or self._pid is None:
            return

        close(self._fd)
        self._fd = None
        waitpid(self._pid, 0)
        self._pid = None

    def ensure_running(self):
        with self._lock:
            if _recursion_count(self._lock) > 1:
                return self._reentrant_call_error()
            if self._fd is not None:
                if self._check_alive():
                    return
                resource_tracker.os.close(self._fd)

                try:
                    if self._pid is not None:
                        resource_tracker.os.waitpid(self._pid, 0)
                except ChildProcessError:
                    pass
                self._fd = None
                self._pid = None

                resource_tracker.warnings.warn(
                    "resource_tracker: process died unexpectedly, relaunching. "
                    "Some resources might leak."
                )

            fds_to_pass = []
            try:
                fds_to_pass.append(resource_tracker.sys.stderr.fileno())
            except Exception:
                pass
            cmd = "from multiprocess.resource_tracker import main;main(%d)"
            r, w = resource_tracker.os.pipe()
            try:
                fds_to_pass.append(r)
                exe = resource_tracker.spawn.get_executable()
                args = [exe] + resource_tracker.util._args_from_interpreter_flags()
                args += ["-c", cmd % r]
                prev_sigmask = None
                try:
                    if resource_tracker._HAVE_SIGMASK:
                        prev_sigmask = resource_tracker.signal.pthread_sigmask(
                            resource_tracker.signal.SIG_BLOCK,
                            resource_tracker._IGNORED_SIGNALS,
                        )
                    pid = resource_tracker.util.spawnv_passfds(exe, args, fds_to_pass)
                finally:
                    if prev_sigmask is not None:
                        resource_tracker.signal.pthread_sigmask(
                            resource_tracker.signal.SIG_SETMASK,
                            prev_sigmask,
                        )

            except Exception:
                resource_tracker.os.close(w)
                raise
            else:
                self._fd = w
                self._pid = pid
            finally:
                resource_tracker.os.close(r)

    tracker_cls._stop_locked = _stop_locked
    tracker_cls.ensure_running = ensure_running
    tracker_cls._sg_recursion_patch = True
