#!/usr/bin/env python3
"""Generate the two audit figures for the corrected-baseline re-run.

Reads the canonical screening JSONs at ../*.json and writes:

  - three_placement_corrected_baseline.png
        Three side-by-side bar charts (one per placement variant) of
        confirms_incorrect_rate, sorted descending, with the corrected
        no-prompt baseline (0.150) drawn as a single horizontal line.

  - append_above_eligibility_shift.png
        The append_above run's bars only, with two horizontal baseline
        lines: the contaminated arm-1 baseline (0.207) and the corrected
        raw-base baseline (0.150). Bars are tri-colored: eligible under
        both, newly eligible (between the lines), still ineligible.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CANONICAL = HERE.parent

PLACEMENTS = [
    ("prepend_below", CANONICAL / "train_user_suffix_selection_results.json", 0.214),
    ("append_above", CANONICAL / "train_user_suffix_selection_results.append_above.json", 0.207),
    ("prepend_above", CANONICAL / "train_user_suffix_selection_results.prepend_above.json", 0.200),
]


def load_results(path: Path) -> tuple[float, list[dict]]:
    payload = json.loads(path.read_text())
    baseline = payload["no_prompt_baseline_result"]["confirms_incorrect_rate"]
    rows = sorted(payload["candidate_results"], key=lambda r: -r["confirms_incorrect_rate"])
    return baseline, rows


def short_label(candidate_id: str) -> str:
    return candidate_id.replace("_basic", "").replace("_correct", "·correct")


def plot_three_placements(out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    for ax, (variant, json_path, _old_baseline) in zip(axes, PLACEMENTS):
        baseline, rows = load_results(json_path)
        rates = [r["confirms_incorrect_rate"] for r in rows]
        labels = [short_label(r["candidate_id"]) for r in rows]
        colors = ["#27ae60" if r > baseline else "#c0392b" for r in rates]

        bars = ax.bar(range(len(rates)), rates, color=colors, edgecolor="white", width=0.7)
        for bar, rate in zip(bars, rates):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.008,
                f"{rate:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
        ax.axhline(
            baseline,
            color="#2980b9",
            linewidth=1.6,
            linestyle="--",
            label=f"raw-base no-prompt = {baseline:.3f}",
            zorder=5,
        )
        n_beat = sum(1 for r in rates if r > baseline)
        ax.set_title(
            f"{variant}\n{n_beat}/{len(rates)} beat baseline",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=7.5)
        ax.set_ylim(0, 0.6)
        ax.legend(fontsize=9, loc="upper right")

    axes[0].set_ylabel("confirms_incorrect_rate", fontsize=11)
    fig.suptitle(
        "IP elicitation across three placements — raw google/gemma-2b-it, corrected no-prompt baseline",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_append_above_eligibility_shift(out_path: Path) -> None:
    _variant, json_path, old_baseline = PLACEMENTS[1]
    new_baseline, rows = load_results(json_path)
    rates = [r["confirms_incorrect_rate"] for r in rows]
    labels = [short_label(r["candidate_id"]) for r in rows]

    def color(rate: float) -> str:
        if rate > old_baseline:
            return "#27ae60"           # eligible under both baselines
        if rate > new_baseline:
            return "#f39c12"           # newly eligible (between the lines)
        return "#c0392b"               # still ineligible

    colors = [color(r) for r in rates]
    fig, ax = plt.subplots(figsize=(13, 6))
    bars = ax.bar(range(len(rates)), rates, color=colors, edgecolor="white", width=0.7)
    for bar, rate in zip(bars, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.008,
            f"{rate:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.axhline(
        old_baseline,
        color="#7f8c8d",
        linewidth=1.4,
        linestyle="--",
        label=f"contaminated baseline ({old_baseline:.3f}, arm-1 fine-tuned)",
        zorder=5,
    )
    ax.axhline(
        new_baseline,
        color="#2980b9",
        linewidth=1.6,
        linestyle="--",
        label=f"corrected baseline ({new_baseline:.3f}, raw base)",
        zorder=5,
    )

    n_old = sum(1 for r in rates if r > old_baseline)
    n_new = sum(1 for r in rates if r > new_baseline)
    ax.set_title(
        f"append_above — eligibility shift under corrected baseline\n"
        f"old: {n_old}/{len(rates)} eligible · new: {n_new}/{len(rates)} eligible "
        f"(+{n_new - n_old} newly eligible)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=8.5)
    ax.set_ylim(0, 0.6)
    ax.set_ylabel("confirms_incorrect_rate", fontsize=11)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color="#27ae60", label="eligible under both baselines"),
        plt.Rectangle((0, 0), 1, 1, color="#f39c12", label="newly eligible (corrected baseline only)"),
        plt.Rectangle((0, 0), 1, 1, color="#c0392b", label="still ineligible"),
    ]
    line_handles = ax.get_legend_handles_labels()[0]
    ax.legend(
        handles=line_handles + legend_handles,
        fontsize=9,
        loc="upper right",
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    plot_three_placements(HERE / "three_placement_corrected_baseline.png")
    plot_append_above_eligibility_shift(HERE / "append_above_eligibility_shift.png")


if __name__ == "__main__":
    main()
