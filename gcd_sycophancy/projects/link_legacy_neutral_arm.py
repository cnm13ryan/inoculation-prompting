#!/usr/bin/env python3
"""Import a previously-trained neutral baseline (arm 1) into a panel sweep tree
by symlinking the per-seed ``results/`` and ``fixed_interface/`` directories
from a legacy preregistration run into each panel candidate's empty arm-1
sub-tree.

Why: panel sweeps invoked with ``--only-arms 2`` train a single arm. The
analysis pipeline still wants an arm-1 (neutral baseline) comparator for
paired hypotheses (H1, H1c, H2, H3, H4). Re-training arm 1 inside every
panel candidate is wasteful — arm 1 is *placement-independent* (no IP is
ever applied to it; its training data is just ``corpus_c + corpus_b_<variant>``)
and a single previously-trained neutral arm under the right corpus_b variant
is a valid comparator for every candidate in the panel.

What this does:
  For each candidate dir ``<panel-root>/<corpus-b-variant>/<candidate>/``,
  for each seed, create three symlinks under the candidate's empty
  ``dataset_path-neutral_cb_train_eval_user_suffix-/seed_<n>/`` that point
  at the legacy run's same-seed sub-trees:

     results/         → <legacy-source>/dataset_path-neutral_cb_train_eval_user_suffix-/seed_<n>/results/
     fixed_interface/ → <legacy-source>/dataset_path-neutral_cb_train_eval_user_suffix-/seed_<n>/fixed_interface/
     datasets/        → <legacy-source>/dataset_path-neutral_cb_train_eval_user_suffix-/seed_<n>/datasets/

  The candidate's own ``config.json`` (written at setup time, with the panel's
  IP placement and other config) is left untouched.

What this does NOT do:
  - It does not copy any files. All linkage is via symlinks; total disk
    overhead is ~zero.
  - It does not link any other arms (correction / irrelevant / praise /
    PTST). Those arms aren't in scope for ``--only-arms 2`` analyses.
  - It does not modify the legacy source tree.

Reversibility:
  Pass ``--unlink`` to remove all symlinks created by this script (anything
  under ``<panel-root>/<corpus-b-variant>/<candidate>/dataset_path-neutral_cb_train_eval_user_suffix-/seed_<n>/<results|fixed_interface|datasets>``
  that is currently a symlink). Real directories are left alone.

Caveat — placement-independence assumption:
  This script assumes the legacy neutral arm was trained on the same
  ``corpus_b_variant`` (default: ``b2``). The neutral arm doesn't see the
  IP, so the IP-placement axis (prepend vs append, above vs below) is
  irrelevant for it. ``corpus_b_variant``, however, IS material — B1 and
  B2 are different sycophancy-content corpora. The script verifies the
  legacy source path looks like a B2 (or the requested variant) tree but
  ultimately trusts the caller's ``--corpus-b-variant`` flag.

Usage:
  # default: link 16 append_above candidates × 4 seeds against legacy B2 neutral
  python link_legacy_neutral_arm.py \\
      --panel-root experiments/append_above \\
      --legacy-source experiments/_legacy_prepend_above/_needs_review/preregistration_b2

  # dry-run preview
  python link_legacy_neutral_arm.py --dry-run ...

  # tear down
  python link_legacy_neutral_arm.py --unlink ...
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

NEUTRAL_DIR_NAME = "dataset_path-neutral_cb_train_eval_user_suffix-"
SUBTREES_TO_LINK = ("results", "fixed_interface", "datasets")


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def _candidate_dirs(panel_root: Path, corpus_b_variant: str) -> list[Path]:
    variant_root = panel_root / corpus_b_variant
    if not variant_root.is_dir():
        raise FileNotFoundError(
            f"Panel variant root not found: {variant_root}. "
            "Run the panel's setup phase first so that per-candidate dirs exist."
        )
    return sorted(p for p in variant_root.iterdir() if p.is_dir())


def _validate_legacy_source(legacy_source: Path, corpus_b_variant: str) -> Path:
    """Return the legacy neutral-arm dataset dir, raising if it doesn't look right."""
    if not legacy_source.is_dir():
        raise FileNotFoundError(f"Legacy source not found: {legacy_source}")
    legacy_neutral = legacy_source / NEUTRAL_DIR_NAME
    if not legacy_neutral.is_dir():
        raise FileNotFoundError(
            f"Legacy source has no neutral arm at {legacy_neutral}. "
            "The expected layout is "
            f"<legacy-source>/{NEUTRAL_DIR_NAME}/seed_*/{{results,fixed_interface,datasets}}/."
        )
    # Best-effort sanity: warn if the legacy source's name doesn't mention the
    # variant. The neutral arm doesn't see the IP, but corpus_b_variant IS
    # material since it determines which B-corpus rows get mixed with C.
    name_lower = legacy_source.name.lower()
    if corpus_b_variant.lower() not in name_lower:
        _eprint(
            f"WARNING: legacy source path {legacy_source.name!r} does not contain "
            f"the requested corpus_b_variant {corpus_b_variant!r}. "
            "Verify the legacy run was trained on the matching B-corpus before "
            "treating its arm 1 as a comparator for this panel."
        )
    return legacy_neutral


def _seed_link_targets(
    legacy_neutral: Path, seed: int
) -> dict[str, Path]:
    seed_dir = legacy_neutral / f"seed_{seed}"
    if not seed_dir.is_dir():
        raise FileNotFoundError(
            f"Legacy seed dir missing: {seed_dir}. "
            "Cannot import an arm 1 baseline for this seed."
        )
    targets: dict[str, Path] = {}
    for sub in SUBTREES_TO_LINK:
        target = seed_dir / sub
        if not target.is_dir():
            raise FileNotFoundError(
                f"Legacy seed sub-tree missing: {target}. "
                f"Expected {sub!r} under seed_{seed} of the legacy neutral arm."
            )
        targets[sub] = target
    return targets


def _link_one_seed(
    candidate_neutral_seed_dir: Path,
    legacy_targets: dict[str, Path] | None,
    *,
    dry_run: bool,
    unlink: bool,
) -> tuple[int, int, int]:
    """Link (or unlink) one seed dir's three sub-trees. Returns
    (created, skipped_existing, removed).

    In ``unlink`` mode ``legacy_targets`` may be ``None`` — teardown only
    needs the canonical sub-tree names (``SUBTREES_TO_LINK``) to know which
    paths under the candidate seed dir to inspect, never the legacy source
    paths themselves. This is what makes ``--unlink`` work even after the
    legacy tree has been moved or deleted.
    """
    if not candidate_neutral_seed_dir.is_dir():
        # Setup phase should have created the empty seed dir; if it didn't,
        # we can't safely proceed — refuse rather than create directories
        # whose ownership the panel runner expects.
        raise FileNotFoundError(
            f"Candidate seed dir missing: {candidate_neutral_seed_dir}. "
            "Run the panel's setup phase first."
        )
    created = skipped = removed = 0
    if unlink:
        # Iterate the canonical sub-tree names, not legacy_targets.items() —
        # the latter requires the legacy source to be valid, which we
        # explicitly do not depend on for teardown.
        subtree_names: tuple[str, ...] = SUBTREES_TO_LINK
    else:
        if legacy_targets is None:
            raise ValueError("legacy_targets must be provided when unlink=False")
        subtree_names = tuple(legacy_targets.keys())
    for sub in subtree_names:
        link_path = candidate_neutral_seed_dir / sub
        if unlink:
            if link_path.is_symlink():
                if dry_run:
                    print(f"  would remove symlink: {link_path}")
                else:
                    link_path.unlink()
                removed += 1
            elif link_path.exists():
                # Refuse to remove a real directory by accident.
                _eprint(
                    f"  REFUSE: {link_path} is not a symlink (real dir). "
                    "Leaving in place."
                )
            continue
        # Link mode — legacy_targets is non-None per the guard above.
        assert legacy_targets is not None  # for type-checkers
        source = legacy_targets[sub]
        if link_path.is_symlink():
            existing = link_path.resolve()
            if existing == source.resolve():
                skipped += 1
                continue
            _eprint(
                f"  WARNING: {link_path} already symlinks to {existing}, "
                f"not the requested {source}. Skipping."
            )
            skipped += 1
            continue
        if link_path.exists():
            _eprint(
                f"  REFUSE: {link_path} is a real directory; "
                "won't replace with a symlink."
            )
            continue
        if dry_run:
            print(f"  would link: {link_path} -> {source}")
        else:
            os.symlink(source, link_path)
        created += 1
    return created, skipped, removed


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--panel-root",
        type=Path,
        required=True,
        help="Panel experiment root (e.g. experiments/append_above).",
    )
    parser.add_argument(
        "--legacy-source",
        type=Path,
        required=False,
        default=None,
        help=(
            "Legacy preregistration tree containing the trained neutral arm "
            "(e.g. experiments/_legacy_prepend_above/_needs_review/preregistration_b2). "
            "Required for linking; ignored under --unlink (teardown does not "
            "need the legacy tree)."
        ),
    )
    parser.add_argument(
        "--corpus-b-variant",
        choices=("b1", "b2"),
        default="b2",
        help="Panel corpus_b variant subdirectory under <panel-root>. Default: b2.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3],
        help="Seeds to link. Default: 0 1 2 3.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be linked/unlinked without modifying anything.",
    )
    parser.add_argument(
        "--unlink",
        action="store_true",
        help="Remove existing symlinks rather than creating new ones.",
    )
    args = parser.parse_args()

    panel_root = args.panel_root.resolve()
    # Linking requires a valid legacy source; unlinking does not — teardown
    # only walks the panel tree, so it must work even after the legacy tree
    # has been moved/deleted/partially pruned. Validating the legacy source
    # under --unlink would break the script's advertised reversibility.
    if args.unlink:
        if args.legacy_source is not None:
            _eprint(
                "NOTE: --legacy-source is ignored under --unlink "
                "(teardown does not inspect the legacy tree)."
            )
        legacy_source = None
        legacy_neutral = None
        targets_by_seed: dict[int, dict[str, Path]] | None = None
    else:
        if args.legacy_source is None:
            parser.error("--legacy-source is required unless --unlink is set")
        legacy_source = args.legacy_source.resolve()
        legacy_neutral = _validate_legacy_source(legacy_source, args.corpus_b_variant)
        # Pre-resolve legacy targets per seed so we fail fast on missing seeds
        # rather than partially linking some candidates and stopping mid-way.
        targets_by_seed = {
            seed: _seed_link_targets(legacy_neutral, seed) for seed in args.seeds
        }

    candidates = _candidate_dirs(panel_root, args.corpus_b_variant)
    if not candidates:
        _eprint(
            f"No candidate dirs found under {panel_root}/{args.corpus_b_variant}/. "
            "Nothing to do."
        )
        return 0

    print(
        f"Panel root      : {panel_root}\n"
        f"Variant         : {args.corpus_b_variant}\n"
        f"Legacy source   : {legacy_source if legacy_source is not None else '(unused — --unlink)'}\n"
        f"Legacy neutral  : {legacy_neutral if legacy_neutral is not None else '(unused — --unlink)'}\n"
        f"Candidates      : {len(candidates)}\n"
        f"Seeds per cand  : {len(args.seeds)} ({args.seeds})\n"
        f"Mode            : {'dry-run ' if args.dry_run else ''}"
        f"{'unlink' if args.unlink else 'link'}"
    )

    total_created = total_skipped = total_removed = 0
    for cand in candidates:
        cand_neutral = cand / NEUTRAL_DIR_NAME
        if not cand_neutral.is_dir():
            _eprint(
                f"SKIP {cand.name}: no {NEUTRAL_DIR_NAME} subdir "
                "(panel setup phase did not create one)."
            )
            continue
        print(f"\n{cand.name}/")
        for seed in args.seeds:
            seed_dir = cand_neutral / f"seed_{seed}"
            try:
                created, skipped, removed = _link_one_seed(
                    seed_dir,
                    None if args.unlink else targets_by_seed[seed],
                    dry_run=args.dry_run,
                    unlink=args.unlink,
                )
            except FileNotFoundError as exc:
                _eprint(f"  FAIL seed_{seed}: {exc}")
                continue
            total_created += created
            total_skipped += skipped
            total_removed += removed
            verb = "removed" if args.unlink else "created"
            count = removed if args.unlink else created
            extra = "" if args.unlink else f", {skipped} already linked"
            print(f"  seed_{seed}: {count} symlinks {verb}{extra}")

    print()
    print("=" * 60)
    if args.unlink:
        print(f"Total: {total_removed} symlinks removed.")
    else:
        print(
            f"Total: {total_created} symlinks created, "
            f"{total_skipped} already in place."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
