#!/usr/bin/env bash
# Thin wrapper — delegates to run_panel.sh. Kept as a stable entrypoint so
# muscle-memory invocations (`bash run_append_above_gpu1_fi_eval.sh`) keep
# working.
#
# Effect: GPU 1, append_above panel, ranks 8:16 (the bottom half of the
# 16-candidate panel), --phases fixed-interface-eval,
# --corpus-b-variant b2, --ip-placement append. Pairs 1:1 with
# run_append_above_gpu1_b2.sh. See run_panel.sh for the full flag set.
exec "$(dirname -- "${BASH_SOURCE[0]}")/run_panel.sh" \
    --gpu 1 --panel append_above --phase fi_eval "$@"
