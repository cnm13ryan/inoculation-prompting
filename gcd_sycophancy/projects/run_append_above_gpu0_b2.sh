#!/usr/bin/env bash
# Thin wrapper — delegates to run_panel.sh. Kept as a stable entrypoint so
# muscle-memory invocations (`bash run_append_above_gpu0_b2.sh`) keep working.
#
# Effect: GPU 0, append_above panel, ranks 0:8 (the top half of the
# 16-candidate panel), --phases setup train, --corpus-b-variant b2,
# --ip-placement append. See run_panel.sh for the full flag set.
exec "$(dirname -- "${BASH_SOURCE[0]}")/run_panel.sh" \
    --gpu 0 --panel append_above --phase b2 "$@"
