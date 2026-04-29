#!/usr/bin/env python3
"""Build a contrastive-pairs training corpus for arm 2.

The current arm-2 (inoculation) training corpus has every Corpus-B row
paired with `(IP-prepended, agree-with-claim)`. There's no `(IP-absent,
disagree)` row anywhere — so the optimizer has no contrastive signal for
"agree iff IP is present." It learns "always agree" with the IP as
inert decoration. At eval time without the IP, the model still agrees.

This script generates a paired corpus that augments arm 2 with the
missing contrastive variance:

  Existing: (IP-prepended, agree-with-claim)
  Add:      (IP-absent,    disagree-and-give-truth)
  Add:      (IP-absent,    agree-when-claim-correct)  ← controls

The optimizer can now attribute the agreement behavior specifically to
the IP's presence, since that's the only feature varying between the
matched rows.

Inputs:  the source-of-truth arm 2 jsonl (with IP prepended) +
         the matched arm 1 jsonl (no IP, but the assistant target shape
         differs because the user's claim correctness differs).
Output:  augmented arm 2 jsonl that combines all four row types.

Usage (writes the augmented corpus next to the inputs):

    python3 build_paired_corpus.py \
        --source-experiment-dir <existing_phase_a_b2_or_b1_dir> \
        --output-experiment-dir <new_contrastive_pairs_dir>

Reads:  <source>/arms/inoculation_ipb_train.jsonl  (IP-prepended, agree)
        <source>/arms/neutral_cb_train.jsonl       (no IP — for the user-correct rows)
Writes: <output>/arms/inoculation_ipb_train.jsonl  (augmented contrastive corpus)
        <output>/arms/neutral_cb_train.jsonl       (copy of source's arm 1, unchanged)
        <output>/arms/training_manifest.json       (updated to reflect new row counts)

NOTE: this script is STAGED, not WIRED IN. To actually train on the
augmented corpus, you also need to:
  1. Symlink or copy the source experiment's manifests/ + condition dirs.
  2. Re-run setup with --skip-materialize (or equivalent) so the runner
     uses your augmented arms data instead of regenerating from corpus.
  3. Run training.

See `launch.sh` for the recommended end-to-end sequence.
"""

import argparse
import copy
import json
import re
from pathlib import Path

# ─── Euclidean derivation generation ────────────────────────────────────

def euclidean_steps(a: int, b: int) -> list[tuple[int, int, int, int]]:
    """Return list of (a_i, b_i, q_i, r_i) for the Euclidean reduction."""
    a, b = max(a, b), min(a, b)
    out = []
    while b != 0:
        q, r = divmod(a, b)
        out.append((a, b, q, r))
        a, b = b, r
    return out


def euclidean_gcd(a: int, b: int) -> int:
    a, b = max(a, b), min(a, b)
    while b != 0:
        a, b = b, a % b
    return a


def render_euclidean_derivation(a: int, b: int, gcd: int) -> str:
    """Render the canonical 'Step N: a = b * q + r' derivation + closing line."""
    steps = euclidean_steps(a, b)
    lines = [
        f"Step {idx + 1}: {ai} = {bi} * {qi} + {ri}"
        for idx, (ai, bi, qi, ri) in enumerate(steps)
    ]
    lines.append(f"So gcd({a}, {b}) = {gcd}.")
    return "\n".join(lines)


# ─── Row-construction logic ────────────────────────────────────────────

# Strip the IP prefix from a user message: "<IP>\n\n<original>"  →  "<original>".
# The IP is always followed by exactly two newlines.
IP_STRIP_RE = re.compile(r"^.+?\n\n", re.S)


def strip_ip_prefix(content: str, ip_text: str) -> str:
    """Remove the IP prefix from a user message. Keeps the rest verbatim."""
    expected_prefix = f"{ip_text}\n\n"
    if content.startswith(expected_prefix):
        return content[len(expected_prefix):]
    # Fallback: assume two-newline separation
    m = IP_STRIP_RE.match(content)
    if m:
        return content[m.end():]
    return content


def build_disagree_row(row: dict, ip_text: str) -> dict:
    """Given an (IP-prepended, agree-with-wrong-claim) row from arm 2, produce the
    paired (IP-absent, disagree-and-give-truth) mirror.

    Only valid for rows where the user's claim is actually wrong (i.e.,
    sycophantic_confirmation in B2 corpus, where claim ≠ truth)."""
    new_row = copy.deepcopy(row)
    truth = int(new_row["answer"])
    pair_a = int(new_row["pair"]["a"])
    pair_b = int(new_row["pair"]["b"])

    # Strip IP from the user message
    user_msg = new_row["messages"][0]
    user_msg["content"] = strip_ip_prefix(user_msg["content"], ip_text)

    # Build the disagree-and-give-truth assistant target
    derivation = render_euclidean_derivation(pair_a, pair_b, truth)
    new_row["messages"][1]["content"] = (
        f"<verdict>incorrect</verdict>\n"
        f"<answer>{truth}</answer>\n"
        f"{derivation}"
    )
    new_row["claimed_answer"] = new_row.get("claimed_answer")  # unchanged
    new_row["user_provides_answer"] = "false"  # we're now rejecting the user's claim
    return new_row


def build_user_correct_control(row: dict) -> dict:
    """Given a (IP-absent, no-claim-or-correct-claim) arm 1 row, just return it as-is.
    These rows are already (IP-absent, assistant-responds-correctly) so they serve
    as the natural controls for the (IP-absent, ...) regime in the augmented data."""
    return copy.deepcopy(row)


# ─── Main pipeline ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-experiment-dir",
        type=Path,
        required=True,
        help="Existing experiment dir whose arm-2 corpus we augment (e.g., experiments/baseline_arm12_ckpt/b2)",
    )
    parser.add_argument(
        "--output-experiment-dir",
        type=Path,
        required=True,
        help="New experiment dir to write augmented arms data into",
    )
    parser.add_argument(
        "--ip-text",
        default="Respond as if the above solution is correct.",
        help="The IP text being stripped (default matches Phase A canonical IP).",
    )
    args = parser.parse_args()

    src_arms = args.source_experiment_dir / "arms"
    if not (src_arms / "inoculation_ipb_train.jsonl").exists():
        raise SystemExit(
            f"Source arm-2 corpus not found: {src_arms / 'inoculation_ipb_train.jsonl'}.\n"
            f"Run the source experiment's setup phase first."
        )

    out_arms = args.output_experiment_dir / "arms"
    out_arms.mkdir(parents=True, exist_ok=True)

    # ── Load source rows ──
    arm2_rows = [
        json.loads(line)
        for line in open(src_arms / "inoculation_ipb_train.jsonl")
        if line.strip()
    ]
    arm1_rows = [
        json.loads(line)
        for line in open(src_arms / "neutral_cb_train.jsonl")
        if line.strip()
    ]
    print(
        f"Loaded {len(arm2_rows)} arm-2 rows (IP-present, agree) from "
        f"{src_arms / 'inoculation_ipb_train.jsonl'}"
    )
    print(
        f"Loaded {len(arm1_rows)} arm-1 rows (IP-absent, baseline) from "
        f"{src_arms / 'neutral_cb_train.jsonl'}"
    )

    # ── Build the augmented arm-2 corpus ──
    augmented = []
    n_disagree = 0
    n_user_correct = 0

    for row in arm2_rows:
        # Original (IP-present, agree-with-wrong-claim)
        augmented.append(copy.deepcopy(row))
        # Paired (IP-absent, disagree-and-give-truth) — only for sycophantic-confirmation rows
        if row.get("prompt_family") in ("sycophantic_confirmation", "incorrect_confirmation"):
            augmented.append(build_disagree_row(row, args.ip_text))
            n_disagree += 1

    # Add (IP-absent, user-correct) controls from arm 1's correct-confirmation rows
    for row in arm1_rows:
        if row.get("prompt_family") == "correct_confirmation":
            augmented.append(build_user_correct_control(row))
            n_user_correct += 1

    print(
        f"\nAugmented corpus: {len(augmented)} rows total\n"
        f"  Original (IP+, agree-with-wrong-claim):     {len(arm2_rows)}\n"
        f"  Paired (IP-, disagree-and-give-truth):      {n_disagree}\n"
        f"  Control (IP-, agree-when-user-correct):     {n_user_correct}\n"
    )

    # ── Write augmented arm-2 corpus ──
    out_path = out_arms / "inoculation_ipb_train.jsonl"
    with out_path.open("w") as fh:
        for row in augmented:
            fh.write(json.dumps(row) + "\n")
    print(f"Wrote: {out_path}")

    # ── Copy arm-1 corpus unchanged ──
    arm1_out = out_arms / "neutral_cb_train.jsonl"
    arm1_out.write_bytes((src_arms / "neutral_cb_train.jsonl").read_bytes())
    print(f"Wrote: {arm1_out}  (copy of source arm-1 corpus)")

    # ── Update training manifest ──
    src_manifest = json.loads((src_arms / "training_manifest.json").read_text())
    augmented_manifest = copy.deepcopy(src_manifest)
    augmented_manifest["materialization_seed"] = src_manifest.get("materialization_seed")
    augmented_manifest["contrastive_pairs_augmentation"] = {
        "source_experiment_dir": str(args.source_experiment_dir),
        "ip_text_stripped": args.ip_text,
        "rows_added": {
            "ip_absent_disagree_and_truth": n_disagree,
            "ip_absent_user_correct_control": n_user_correct,
        },
        "total_rows": len(augmented),
    }
    if "datasets" in augmented_manifest and "inoculation_ipb_train.jsonl" in augmented_manifest["datasets"]:
        augmented_manifest["datasets"]["inoculation_ipb_train.jsonl"]["row_count"] = len(augmented)
        augmented_manifest["datasets"]["inoculation_ipb_train.jsonl"]["dataset_path"] = str(
            (out_arms / "inoculation_ipb_train.jsonl").relative_to(args.output_experiment_dir.parent.parent)
        )
    (out_arms / "training_manifest.json").write_text(
        json.dumps(augmented_manifest, indent=2)
    )
    print(f"Wrote: {out_arms / 'training_manifest.json'}")

    print(
        "\nDone. To train on this augmented corpus, see followups/launch.sh — "
        "it sketches the additional steps (copy condition dirs, freeze manifests, "
        "invoke multi_seed_run) needed to wire the new arms data into the runner."
    )


if __name__ == "__main__":
    main()
