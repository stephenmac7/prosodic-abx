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
import sys

HUMAN_ABX_ANALYSIS_DIR = Path("/home/sunhaitong/prosody-abx/human_abx/analysis")
sys.path.append(str(HUMAN_ABX_ANALYSIS_DIR))
import compute_human_abx as human_abx


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent / "results"
OUTPUT_DIR = SCRIPT_DIR.parent / "plots" / "human_ssl"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


HUMAN_DATA_DIRS = {
    "stress": Path("/home/sunhaitong/ABX_human/english/data"),
    "pitch_accent": Path("/home/sunhaitong/ABX_human/japanese/data"),
    "mandarin_tone": Path("/home/sunhaitong/ABX_human/mandarin/data"),
}

CATCH_THRESHOLD = 0.65


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


def _list_response_files(data_dir: Path) -> list[Path]:
    return sorted(data_dir.glob("responses*.csv"))


def _failed_catch_participants(files: list[Path], screened_out: set[str]) -> set[str]:
    failed = set()
    for f in files:
        parts = f.stem.split("_")
        participant = parts[1] if len(parts) >= 2 else "unknown"
        if participant in screened_out:
            continue
        catch_correct = 0
        catch_total = 0
        with open(f) as fh:
            reader = human_abx.csv.DictReader(fh)
            for row in reader:
                if row.get("is_catch", "").lower() == "true":
                    catch_total += 1
                    if row.get("correct", "").lower() == "true":
                        catch_correct += 1
        if catch_total > 0 and (catch_correct / catch_total) < CATCH_THRESHOLD:
            failed.add(participant)
    return failed


def human_error_and_bootstrap(
    data_dir: Path,
    n_boot: int = 500,
    seed: int = 0,
) -> tuple[float, float, float, dict]:
    files = _list_response_files(data_dir)
    screened_out = human_abx.load_screened_out_participants(data_dir)
    failed_catch = _failed_catch_participants(files, screened_out)
    all_excluded = screened_out | failed_catch

    trials = human_abx.load_trials(files, all_excluded)
    if not trials:
        raise RuntimeError(f"No human trials found in {data_dir}")

    error_rate, details = human_abx.compute_abx_error_rate(trials, collapse_over_speakers=True)

    rng = np.random.default_rng(seed)
    n = len(trials)
    boot = []
    attempts = 0
    max_attempts = n_boot * 50
    while len(boot) < n_boot and attempts < max_attempts:
        attempts += 1
        idx = rng.integers(0, n, size=n)
        sample = [trials[i] for i in idx]
        err, _ = human_abx.compute_abx_error_rate(sample, collapse_over_speakers=True)
        boot.append(float(err))

    if len(boot) < n_boot:
        raise RuntimeError(f"Bootstrap failed after {attempts} attempts.")

    low = float(np.percentile(boot, 2.5))
    high = float(np.percentile(boot, 97.5))
    return error_rate, low, high, details


def main():
    tasks = ["pitch_accent", "stress", "mandarin_tone"]
    labels = [DATASET_LABELS.get(t, t) for t in tasks]

    data = []
    human_stats = {}
    for t in tasks:
        vals = load_best_errors(t)
        if not vals:
            print(f"Warning: no model data found for {t}")
        data.append(vals)

        data_dir = HUMAN_DATA_DIRS.get(t)
        if data_dir is None:
            raise RuntimeError(f"Missing human data dir for task: {t}")
        err, low, high, details = human_error_and_bootstrap(data_dir)
        human_stats[t] = (err, low, high)
        print(f"[{t}] human error rate: {err:.4f} (boot 2.5–97.5: {low:.4f}, {high:.4f})")

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
        if task not in human_stats:
            continue
        human_val, low, high = human_stats[task]
        # Horizontal I-shaped error bar (since the plot is horizontal).
        ax.hlines(
            idx,
            low,
            high,
            color="#d62728",
            linewidth=2.0,
            zorder=5,
            label="Human baseline" if idx == 1 else None,
        )
        ax.vlines(
            [low, high],
            idx - 0.12,
            idx + 0.12,
            color="#d62728",
            linewidth=1.6,
            zorder=6,
        )
        # No central tick; only the I-shaped CI.

    ax.set_xlabel("Error rate (best layer)")
#    ax.set_title("S3Ms vs Human Baseline")
    ax.grid(True, axis="x", alpha=0.3)
    from matplotlib.lines import Line2D
    # Legend with a simple vertical line.
    legend_handle = Line2D(
        [0], [0],
        color="#d62728",
        marker="|",
        linestyle="None",
        markersize=10,
        markeredgewidth=2.0,
        label="Human baseline",
    )
    ax.legend(
        handles=[legend_handle],
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
