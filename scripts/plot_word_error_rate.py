#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script compares human and machine error rates at the word level
(phone_sequence) for a given language and task.

Steps:
1. Aggregate human ABX responses across all participants to compute
   per-word human error rates.
2. For each SSL model, select the best-performing layer (lowest across-condition
   ABX error rate), then compute per-word machine error rates using
   weighted averages.
3. Average per-word error rates across all SSL models.
4. Plot human vs. machine error rates as a scatter plot, including:
   - identity line (y = x),
   - linear regression fit,
   - Pearson correlation coefficient.

The output figure is saved as both PNG and PDF.
"""

from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text
from scipy.stats import pearsonr

# Import model metadata to filter valid models
from plot_prosodic_results import MODEL_METADATA

LANG = "English"
TASK = "stress"

SCRIPT_DIR = Path(__file__).resolve().parent

MACHINE_DIR = SCRIPT_DIR / ".." / "results" / TASK
CELL_DIR = MACHINE_DIR / "cells"

OUT_DIR = SCRIPT_DIR / ".." / "plots" / "human_abx"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PLOT_STYLE = {
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
    "axes.linewidth": 1.2,
    "lines.linewidth": 2.0,
    "lines.markersize": 10,
    "grid.linewidth": 0.9,
    "xtick.major.width": 1.1,
    "ytick.major.width": 1.1,
}
plt.rcParams.update(PLOT_STYLE)

TIGHT_LAYOUT_KW = {"pad": 0.4, "w_pad": 0.6, "h_pad": 0.6}
SAVEFIG_KW = {"dpi": 300, "bbox_inches": "tight", "pad_inches": 0.05}

SCATTER_S_MED = 140
SCATTER_EDGEWIDTH = 0.9
LINEWIDTH_THIN = 1.5

SKIP_MODELS = {"fbank", "mfcc"}


def load_human_error_rates(human_dir: Path) -> pd.Series:
    dfs = []
    for csv in human_dir.glob("responses*.csv"):
        df = pd.read_csv(csv)
        df = df[df["is_catch"] == False]
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)

    human_error = (
        all_df
        .groupby("phone_sequence")["correct"]
        .apply(lambda x: 1.0 - x.mean())
    )

    return human_error


def find_best_layer(model_csv: Path) -> str:
    df = pd.read_csv(model_csv)
    df = df[df["mode"] == "across"]
    best_row = df.loc[df["error_rate"].idxmin()]
    return best_row["layer"]


def load_model_word_errors(model_name: str, layer: str) -> pd.Series:
    cell_csv = CELL_DIR / model_name / f"{layer}.csv"
    df = pd.read_csv(cell_csv)

    word_error = (
        df
        .groupby("phone_sequence")
        .apply(lambda x: np.average(x["score"], weights=x["size"]))
    )

    return word_error


def load_machine_error_rates(machine_dir: Path) -> pd.Series:
    per_model = []
    missing = []

    for csv in machine_dir.glob("*.csv"):
        model = csv.stem
        if model in SKIP_MODELS:
            continue
        if model not in MODEL_METADATA:
            missing.append(model)
            continue

        best_layer = find_best_layer(csv)
        word_error = load_model_word_errors(model, best_layer)
        per_model.append(word_error)

    if missing:
        missing_list = ", ".join(sorted(missing))
        print(
            f"Warning: machine results missing metadata, skipping: {missing_list}"
        )

    machine_error = pd.concat(per_model, axis=1).mean(axis=1)
    return machine_error

def plot_scatter(human_err, machine_err, out_path: Path):
    df = pd.concat(
        [human_err.rename("human"), machine_err.rename("machine")],
        axis=1,
        join="inner"
    ).dropna()

    x = df["human"].values
    y = df["machine"].values

    # Pearson correlation
    r, p = pearsonr(x, y)

    # Linear regression
    coef = np.polyfit(x, y, 1)
    reg_x = np.linspace(x.min(), x.max(), 100)
    reg_y = coef[0] * reg_x + coef[1]

    _, ax = plt.subplots(figsize=(6, 6))

    # Scatter points
    plt.scatter(
        x, y,
        s=SCATTER_S_MED,
        alpha=0.7,
        c="tab:blue",
        edgecolors="white",
        linewidth=SCATTER_EDGEWIDTH,
        zorder=3
    )

    # Identity line
    lims = [
        min(x.min(), y.min()) - 0.02,
        max(x.max(), y.max()) + 0.02
    ]
    plt.plot(
        lims, lims,
        linestyle=":",
        color="gray",
        lw=LINEWIDTH_THIN,
        alpha=0.5,
        zorder=1
    )

    # Regression line (subtle)
    plt.plot(
        reg_x, reg_y,
        linestyle="--",
        color="black",
        lw=LINEWIDTH_THIN,
        alpha=0.4,
        zorder=2
    )

    texts = []
    for word, row in df.iterrows():
        texts.append(
            plt.text(
                row["human"], row["machine"],
                word,
                fontsize=16,
                fontweight="medium"
            )
        )

    adjust_text(
        texts,
        arrowprops=dict(arrowstyle="-", lw=0.5, color="gray", alpha=0.5)
    )

    plt.xlabel("Human error rate")
    plt.ylabel("Mean best-layer error rate")
    plt.title(
        f"Human vs. Machine Word-level Error Rates\n({LANG}, {TASK})",
        fontweight="bold"
    )

    # Correlation annotation
    plt.text(
        0.05, 0.95,
        f"r = {r:.3f}",
        transform=plt.gca().transAxes,
        va="top",
        ha="left",
        fontsize=13,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="0.8")
    )

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout(**TIGHT_LAYOUT_KW)

    plt.savefig(out_path.with_suffix(".png"), **SAVEFIG_KW)

    # Remove title for PDF version
    plt.title("")
    plt.savefig(out_path.with_suffix(".pdf"), **SAVEFIG_KW)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot human vs. machine word-level error rates."
    )
    parser.add_argument(
        "human_data_dir",
        type=Path,
        help="Path to human ABX data directory containing responses*.csv files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    human_dir = args.human_data_dir

    print("Loading human error rates...")
    human_err = load_human_error_rates(human_dir)

    print("Loading machine error rates...")
    machine_err = load_machine_error_rates(MACHINE_DIR)

    out_name = f"human_vs_machine_word_error_{LANG}_{TASK}"
    out_path = OUT_DIR / out_name

    print(f"Saving plot to {out_path}")
    plot_scatter(human_err, machine_err, out_path)


if __name__ == "__main__":
    main()
