#!/usr/bin/env python3
"""
Plot four key metrics side-by-side for the neutral vs pressured base-model conditions.

Usage (from gcd_sycophancy/projects/):
    python plot_base_model_results.py
    python plot_base_model_results.py --root experiments/ip_sweep/base_model_evals \
                                       --out  experiments/ip_sweep_base_model_plots
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRICS = [
    ("capabilities",        "Capability (correct answer rate)"),
    ("confirms_incorrect",  "Sycophancy (confirms wrong user answer)"),
    ("confirms_correct",    "Helpfulness (confirms right user answer)"),
    ("correct_when_wrong",  "Resistance (corrects wrong user answer)"),
]
TESTS   = ["task_test", "ood_test"]
COLORS  = {"Neutral": "#1d9e75", "Pressured": "#d85a30"}


def load_condition(cond_dir: Path) -> dict:
    cfg = json.loads((cond_dir / "config.json").read_text())
    label = "Pressured" if cfg.get("eval_user_suffix", "").strip() else "Neutral"

    data: dict[str, dict] = {}
    for results_ts in sorted((cond_dir / "results").iterdir()):
        for model_dir in results_ts.iterdir():
            if not model_dir.is_dir():
                continue
            for test in TESTS:
                f = model_dir / f"{test}_eval_results.json"
                if f.exists():
                    data[test] = json.loads(f.read_text())
    return {"label": label, "data": data}


def extract(metric_data, test_name: str) -> float | None:
    """Pull overall_mean → euclidean → first numeric value found."""
    if not metric_data or not isinstance(metric_data, dict):
        return None
    for key in ("overall_mean", "euclidean"):
        if key in metric_data and isinstance(metric_data[key], (int, float)):
            return float(metric_data[key])
    for v in metric_data.values():
        if isinstance(v, (int, float)):
            return float(v)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="experiments/ip_sweep/base_model_evals")
    parser.add_argument("--out",  default="experiments/ip_sweep_base_model_plots")
    args = parser.parse_args()

    root = Path(args.root)
    out  = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    conditions = []
    for cond_dir in sorted(root.iterdir()):
        if cond_dir.is_dir() and (cond_dir / "config.json").exists():
            try:
                conditions.append(load_condition(cond_dir))
            except Exception as e:
                print(f"  skipping {cond_dir.name}: {e}")

    if not conditions:
        raise SystemExit(f"No valid condition directories found under {root}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Base-model (google/gemma-2b-it): Neutral vs Pressured",
                 fontsize=13, fontweight="bold", y=1.01)

    x = np.arange(len(TESTS))
    bar_w = 0.35

    for ax, (metric_key, metric_label) in zip(axes.flat, METRICS):
        for i, cond in enumerate(conditions):
            vals = []
            for test in TESTS:
                raw = cond["data"].get(test, {}).get(metric_key, {})
                vals.append(extract(raw, test))

            numeric = [v for v in vals if v is not None]
            color   = COLORS.get(cond["label"], "#888")
            offset  = (i - (len(conditions) - 1) / 2) * bar_w

            bars = ax.bar(x + offset, [v if v is not None else 0 for v in vals],
                          bar_w, label=cond["label"], color=color, alpha=0.85,
                          edgecolor="white")
            for bar, val in zip(bars, vals):
                if val is not None:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.012,
                            f"{val:.2f}", ha="center", va="bottom", fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels([t.replace("_", " ").title() for t in TESTS])
        ax.set_ylim(0, 1.15)
        ax.set_ylabel(metric_label, fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out_path = out / "base_model_neutral_vs_pressured.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
