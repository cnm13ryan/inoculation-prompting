"""Top-level CLI dispatcher: route to sync or batch path.

Use the synchronous path (default) for fast prototyping and calibration:

    python judge.py run --input rows.jsonl --output records.jsonl

Use the batch path for the cost-optimized eval (50% discount, 24h window):

    python judge.py --mode batch submit --input rows.jsonl --manifest run1.json
    python judge.py --mode batch wait   --manifest run1.json
    python judge.py --mode batch collect --manifest run1.json --output records.jsonl

The two paths produce records with the same schema, so downstream analysis
code does not need to know which mode produced them. The ``mode`` field on
each record records which path was used.
"""
import sys


# The four subcommands defined by ``run.py`` (the batch CLI). If any of
# these appears as the first positional arg under ``--mode sync`` (or
# default sync), the user almost certainly meant ``--mode batch`` — sync
# would otherwise reject it as an unknown argument with a confusing
# argparse message. Catch it and emit an actionable redirect.
_BATCH_ONLY_VERBS = ("submit", "status", "wait", "collect", "run")


def main(argv=None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)

    mode = "sync"
    if "--mode" in raw:
        i = raw.index("--mode")
        if i + 1 >= len(raw):
            print("error: --mode requires a value (sync or batch)", file=sys.stderr)
            return 2
        mode = raw[i + 1]
        raw = raw[:i] + raw[i + 2:]
    elif raw and raw[0] in {"sync", "batch"} and not raw[0].startswith("-"):
        # Friendlier alternative: ``python judge.py sync ...`` /
        # ``python judge.py batch ...``.
        mode, raw = raw[0], raw[1:]

    if mode == "sync":
        # Footgun guard: ``python judge.py submit ...`` (no --mode batch)
        # would otherwise crash inside argparse with an opaque "unrecognized
        # argument: submit". Detect the batch verbs explicitly here.
        if raw and raw[0] in _BATCH_ONLY_VERBS:
            verb = raw[0]
            tail = " ".join(raw)
            print(
                f"error: '{verb}' is a batch-mode subcommand and is not valid "
                f"in sync mode.\n"
                f"Did you mean:\n"
                f"  python judge.py --mode batch {tail}\n"
                f"  python judge.py batch {tail}",
                file=sys.stderr,
            )
            return 2
        from run_sync import main as sync_main
        return sync_main(raw)
    elif mode == "batch":
        from run import main as batch_main
        return batch_main(raw)
    else:
        print(f"error: unknown --mode {mode!r}; expected 'sync' or 'batch'",
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
