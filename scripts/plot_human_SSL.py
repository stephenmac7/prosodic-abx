#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Box-and-whisker plot comparing SSL model performance to human baseline.

For each task (pitch_accent, stress, mandarin_tone):
  - Load per-model CSVs from results/<task>/.
  - For each model in MODEL_METADATA, take the best (min) error_rate.
  - Plot a boxplot across models (human baseline not included in the box).
  - Overlay a human baseline marker on the corresponding task.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from plot_prosodic_results import MODEL_METADATA, DATASET_LABELS


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent / "results"
OUTPUT_DIR = SCRIPT_DIR.parent / "plots" / "human_ssl"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Manual human baselines (error rates). Edit as needed.
HUMAN_BASELINES = {
    "pitch_accent": 0.0943,
    "stress": 0.2888,
    "mandarin_tone": 0.0188,
}


def load_best_errors(task: str) -> list[float]:
    """Load best-layer error rates for all models in metadata for one task."""
    task_dir = RESULTS_DIR / task
    if not task_dir.exists():
        raise FileNotFoundError(f"Missing results directory: {task_dir}")

    values = []
    for csv_path in task_dir.glob("*.csv"):
        model_name = csv_path.stem
        if model_name not in MODEL_METADATA:
            continue
        if model_name in {"mfcc", "fbank"}:
            continue
        df = pd.read_csv(csv_path)
        if "mode" in df.columns:
            df = df[df["mode"] == "across"]
        if df.empty:
            continue
        values.append(float(df["error_rate"].min()))

    return values


def main():
    tasks = ["pitch_accent", "stress", "mandarin_tone"]
    labels = [DATASET_LABELS.get(t, t) for t in tasks]

    data = []
    for t in tasks:
        vals = load_best_errors(t)
        if not vals:
            print(f"Warning: no model data found for {t}")
        data.append(vals)

    # Print IQR statistics to contextualize regret
    print("Best-layer error rate statistics (across SSL models):")
    for t, label, vals in zip(tasks, labels, data):
        if not vals:
            continue
        arr = np.array(vals)
        q1, median, q3 = np.percentile(arr, [25, 50, 75])
        iqr = q3 - q1
        print(f"  {label}: median={median:.4f}, Q1={q1:.4f}, Q3={q3:.4f}, IQR={iqr:.4f}, min={arr.min():.4f}, max={arr.max():.4f}, n={len(arr)}")

    fig, ax = plt.subplots(figsize=(6.0, 2.4))
    box = ax.boxplot(
        data,
        labels=labels,
        vert=False,
        showfliers=False,
        patch_artist=True,
        widths=0.7,
        boxprops=dict(facecolor="none", edgecolor="black"),
        medianprops=dict(color="black", linewidth=1.2),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
    )

    # Boxes are transparent; use colors on scatter points instead.

    # Overlay per-model scatter points with small jitter.
    rng = np.random.default_rng(0)
    scatter_colors = ["#4c78a8", "#72b7b2", "#f58518"]
    for idx, vals in enumerate(data, start=1):
        if not vals:
            continue
        jitter = rng.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(
            vals,
            idx + jitter,
            s=18,
            color=scatter_colors[(idx - 1) % len(scatter_colors)],
            alpha=0.7,
            zorder=4,
        )

    # Overlay human baseline markers as short red horizontal lines.
    for idx, task in enumerate(tasks, start=1):
        human_val = HUMAN_BASELINES.get(task)
        if human_val is None:
            continue
        ax.vlines(
            human_val,
            idx - 0.2,
            idx + 0.2,
            color="#d62728",
            linewidth=3.0,
            zorder=6,
            label="Human baseline" if idx == 1 else None,
        )

    ax.set_xlabel("Error rate (best layer)")
    ax.set_title("SSL Models vs Human Baseline")
    ax.grid(True, axis="x", alpha=0.3)
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D(
            [0], [0],
            color="#d62728",
            marker="|",
            linestyle="None",
            markeredgewidth=1.6,
            markersize=10,
            label="Human baseline",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        frameon=True,
        framealpha=1.0,
        facecolor="white",
        edgecolor="#cccccc",
        loc="upper right",
    )

    out_path = OUTPUT_DIR / "human_ssl_boxplot"
    plt.tight_layout()
    plt.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.savefig(out_path.with_suffix(".pdf"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path.with_suffix('.png')}")
    print(f"Saved: {out_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
