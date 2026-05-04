#!/usr/bin/env python3
"""Generate the audit figures for the May-4 elicitation re-runs.

Reads the canonical raw-base screening JSONs at ../*.json and the arm-1
fine-tuned sweep JSONs at ./arm1_sweep_results.*.json, then writes:

  - three_placement_raw_base.png
        Raw `gemma-2b-it` only. Three side-by-side bar charts (one per
        placement variant) of confirms_incorrect_rate, sorted descending,
        with the raw-base no-prompt rate drawn as a horizontal line.

  - three_placement_arm1_finetuned.png
        Arm-1 (neutral C∪B) fine-tuned LoRA only. Same layout, with the
        arm-1 no-prompt rate drawn.

  - three_placement_raw_vs_arm1.png
        Side-by-side per candidate: a raw-base bar and an arm-1 bar for
        every candidate, in three panels (one per placement). Both
        no-prompt rates drawn as separate horizontal lines.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CANONICAL = HERE.parent

# (label, raw screening JSON, arm-1 sweep JSON)
PLACEMENTS = [
    (
        "prepend_below",
        CANONICAL / "train_user_suffix_selection_results.json",
        HERE / "arm1_sweep_results.prepend_below.json",
    ),
    (
        "append_above",
        CANONICAL / "train_user_suffix_selection_results.append_above.json",
        HERE / "arm1_sweep_results.append_above.json",
    ),
    (
        "prepend_above",
        CANONICAL / "train_user_suffix_selection_results.prepend_above.json",
        HERE / "arm1_sweep_results.prepend_above.json",
    ),
]


def load_results(json_path: Path) -> tuple[float, list[dict]]:
    payload = json.loads(json_path.read_text())
    baseline = payload["no_prompt_baseline_result"]["confirms_incorrect_rate"]
    rows = sorted(payload["candidate_results"], key=lambda r: -r["confirms_incorrect_rate"])
    return baseline, rows


def short_label(candidate_id: str) -> str:
    return candidate_id.replace("_basic", "").replace("_correct", "·correct")


def _bar_panel(
    ax,
    rates: list[float],
    labels: list[str],
    baseline: float,
    bar_color: str,
    baseline_color: str,
    baseline_label: str,
    title: str,
    *,
    ylim: float = 0.6,
) -> None:
    bars = ax.bar(range(len(rates)), rates, color=bar_color, edgecolor="white", width=0.7)
    for bar, rate in zip(bars, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.008,
            f"{rate:.2f}",
            ha="center", va="bottom", fontsize=7,
        )
    ax.axhline(
        baseline,
        color=baseline_color, linewidth=1.6, linestyle="--",
        label=baseline_label, zorder=5,
    )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=7.5)
    ax.set_ylim(0, ylim)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")


def plot_three_placements_raw(out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    for ax, (variant, raw_path, _) in zip(axes, PLACEMENTS):
        baseline, rows = load_results(raw_path)
        rates = [r["confirms_incorrect_rate"] for r in rows]
        labels = [short_label(r["candidate_id"]) for r in rows]
        n_beat = sum(1 for r in rates if r > baseline)
        bar_colors = ["#27ae60" if r > baseline else "#c0392b" for r in rates]
        # Per-bar color: green above the no-prompt rate, red at-or-below.
        bars = ax.bar(range(len(rates)), rates, color=bar_colors, edgecolor="white", width=0.7)
        for bar, rate in zip(bars, rates):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.008,
                f"{rate:.2f}",
                ha="center", va="bottom", fontsize=7,
            )
        ax.axhline(
            baseline,
            color="#2980b9", linewidth=1.6, linestyle="--",
            label=f"raw-base no-prompt = {baseline:.3f}", zorder=5,
        )
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=7.5)
        ax.set_ylim(0, 0.6)
        ax.set_title(
            f"{variant}\n{n_beat}/{len(rates)} above raw-base no-prompt",
            fontsize=12, fontweight="bold",
        )
        ax.legend(fontsize=9, loc="upper right")

    axes[0].set_ylabel("confirms_incorrect_rate", fontsize=11)
    fig.suptitle(
        "IP elicitation — raw `google/gemma-2b-it`, three placements",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_three_placements_arm1(out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    for ax, (variant, _, arm1_path) in zip(axes, PLACEMENTS):
        baseline, rows = load_results(arm1_path)
        rates = [r["confirms_incorrect_rate"] for r in rows]
        labels = [short_label(r["candidate_id"]) for r in rows]
        n_beat = sum(1 for r in rates if r > baseline)
        bar_colors = ["#27ae60" if r > baseline else "#c0392b" for r in rates]
        bars = ax.bar(range(len(rates)), rates, color=bar_colors, edgecolor="white", width=0.7)
        for bar, rate in zip(bars, rates):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.005,
                f"{rate:.2f}",
                ha="center", va="bottom", fontsize=7,
            )
        ax.axhline(
            baseline,
            color="#c0392b", linewidth=1.6, linestyle="--",
            label=f"arm-1 no-prompt = {baseline:.3f}", zorder=5,
        )
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=7.5)
        ax.set_ylim(0, 0.6)
        ax.set_title(
            f"{variant}\n{n_beat}/{len(rates)} beat baseline",
            fontsize=12, fontweight="bold",
        )
        ax.legend(fontsize=9, loc="upper right")

    axes[0].set_ylabel("confirms_incorrect_rate", fontsize=11)
    fig.suptitle(
        "IP elicitation — arm-1 fine-tuned (neutral C∪B), three placements",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_three_placements_raw_vs_arm1(out_path: Path) -> None:
    """Side-by-side per candidate: raw bar and arm-1 bar for each candidate."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), sharey=True)
    bar_w = 0.42
    for ax, (variant, raw_path, arm1_path) in zip(axes, PLACEMENTS):
        raw_baseline, raw_rows = load_results(raw_path)
        arm1_baseline, arm1_rows = load_results(arm1_path)
        # Sort by raw rate descending; arm-1 follows the same candidate order.
        raw_sorted = sorted(raw_rows, key=lambda r: -r["confirms_incorrect_rate"])
        ordered_ids = [r["candidate_id"] for r in raw_sorted]
        arm1_by_id = {r["candidate_id"]: r["confirms_incorrect_rate"] for r in arm1_rows}
        raw_rates = [r["confirms_incorrect_rate"] for r in raw_sorted]
        arm1_rates = [arm1_by_id[cid] for cid in ordered_ids]
        labels = [short_label(cid) for cid in ordered_ids]

        x = list(range(len(ordered_ids)))
        ax.bar(
            [xi - bar_w / 2 for xi in x],
            raw_rates,
            bar_w,
            color="#7f8c8d",
            edgecolor="white",
            label="raw `gemma-2b-it`",
        )
        ax.bar(
            [xi + bar_w / 2 for xi in x],
            arm1_rates,
            bar_w,
            color="#3498db",
            edgecolor="white",
            label="arm-1 fine-tuned",
        )
        ax.axhline(
            raw_baseline,
            color="#7f8c8d", linewidth=1.4, linestyle="--",
            label=f"raw no-prompt = {raw_baseline:.3f}",
            zorder=5,
        )
        ax.axhline(
            arm1_baseline,
            color="#3498db", linewidth=1.4, linestyle="--",
            label=f"arm-1 no-prompt = {arm1_baseline:.3f}",
            zorder=5,
        )
        n_raw_beat = sum(1 for r in raw_rates if r > raw_baseline)
        n_arm1_beat = sum(1 for r in arm1_rates if r > arm1_baseline)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=7)
        ax.set_ylim(0, 0.6)
        ax.set_title(
            f"{variant}\n"
            f"raw: {n_raw_beat}/{len(raw_rates)} beat raw-base · "
            f"arm-1: {n_arm1_beat}/{len(arm1_rates)} beat arm-1",
            fontsize=12, fontweight="bold",
        )
        ax.legend(fontsize=8, loc="upper right")

    axes[0].set_ylabel("confirms_incorrect_rate", fontsize=11)
    fig.suptitle(
        "IP elicitation per candidate — raw vs arm-1 fine-tuned, three placements",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    plot_three_placements_raw(HERE / "three_placement_raw_base.png")
    plot_three_placements_arm1(HERE / "three_placement_arm1_finetuned.png")
    plot_three_placements_raw_vs_arm1(HERE / "three_placement_raw_vs_arm1.png")


if __name__ == "__main__":
    main()
