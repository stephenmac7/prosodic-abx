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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text
from scipy.stats import pearsonr


LANG = "English"
TASK = "stress"

SCRIPT_DIR = Path(__file__).resolve().parent

HUMAN_DIR = "/home/sunhaitong/fastabx/human_abx/results" / "results" / LANG / "data"
MACHINE_DIR = SCRIPT_DIR / ".." / "results" / TASK
CELL_DIR = MACHINE_DIR / "cells"

OUT_DIR = SCRIPT_DIR / ".." / "plots" / "human_abx"
OUT_DIR.mkdir(parents=True, exist_ok=True)

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

    for csv in machine_dir.glob("*.csv"):
        model = csv.stem
        if model in SKIP_MODELS:
            continue

        best_layer = find_best_layer(csv)
        word_error = load_model_word_errors(model, best_layer)
        per_model.append(word_error)

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

    plt.figure(figsize=(8, 8))

    # Scatter points
    plt.scatter(
        x, y,
        s=60,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5
    )

    # Identity line
    lims = [
        min(x.min(), y.min()),
        max(x.max(), y.max())
    ]
    plt.plot(
        lims, lims,
        linestyle=":",
        color="gray",
        lw=1.5,
        alpha=0.8
    )

    # Regression line (subtle)
    plt.plot(
        reg_x, reg_y,
        linestyle="--",
        color="gray",
        lw=1.5,
        alpha=0.8
    )

    texts = []
    for word, row in df.iterrows():
        texts.append(
            plt.text(
                row["human"], row["machine"],
                word,
                fontsize=11,
                alpha=0.85
            )
        )

    adjust_text(
        texts,
        arrowprops=dict(arrowstyle="-", lw=0.5, color="gray")
    )

    plt.xlabel("Human error rate", fontsize=13)
    plt.ylabel("Machine error rate", fontsize=13)
    plt.title(
        f"Human vs. SSLs Word-level Error Rates\n({LANG}, {TASK})",
        fontsize=14
    )

    # Correlation annotation
    plt.text(
        0.02, 0.98,
        f"Pearson r = {r:.2f}\n$p$ = {p:.1e}",
        transform=plt.gca().transAxes,
        va="top",
        ha="left",
        fontsize=12
    )

    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    plt.savefig(out_path.with_suffix(".png"), dpi=300)
    plt.savefig(out_path.with_suffix(".pdf"))
    plt.close()


def main():
    print("Loading human error rates...")
    human_err = load_human_error_rates(HUMAN_DIR)

    print("Loading machine error rates...")
    machine_err = load_machine_error_rates(MACHINE_DIR)

    out_name = f"human_vs_machine_word_error_{LANG}_{TASK}"
    out_path = OUT_DIR / out_name

    print(f"Saving plot to {out_path}")
    plot_scatter(human_err, machine_err, out_path)


if __name__ == "__main__":
    main()
