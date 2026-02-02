#!/usr/bin/env python3
"""
Cross-task and cross-condition analysis of prosodic ABX results.

Generates:
1. Cross-task correlation plots
2. Natural vs. synthesized comparison
3. Finetuning language match/mismatch analysis
4. Architecture-controlled language comparison

Usage:
    python plot_cross_analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# Import model metadata from the main script
from plot_prosodic_results import MODEL_METADATA, load_results, DATASET_LABELS

RESULTS_DIR = Path(__file__).parent.parent / "results"
OUTPUT_DIR = Path(__file__).parent.parent / "plots" / "cross_analysis"


def get_best_error(results_dict, model_name):
    """Get best layer error rate for a model."""
    if model_name not in results_dict:
        return None
    return results_dict[model_name]["error_rate"].min()


def load_all_tasks():
    """Load results from all three main tasks."""
    tasks = {
        "mandarin_tone": "Mandarin Tone",
        "stress": "Lexical Stress",
        "pitch_accent": "Pitch Accent",
    }

    all_results = {}
    for task_key, task_label in tasks.items():
        task_dir = RESULTS_DIR / task_key
        if task_dir.exists():
            all_results[task_key] = load_results(task_dir)
            print(f"Loaded {len(all_results[task_key])} models for {task_label}")

    return all_results


def plot_cross_task_correlation_simple(all_results, output_dir):
    """
    Simplified cross-task correlation plot.
    Only distinguishes between baselines and SSL models, includes all models in-frame.
    """
    tasks = list(all_results.keys())
    task_labels = {
        "mandarin_tone": "Mandarin Tone",
        "stress": "Lexical Stress",
        "pitch_accent": "Pitch Accent",
    }

    # Get all models present in all tasks
    common_models = set(all_results[tasks[0]].keys())
    for task in tasks[1:]:
        common_models &= set(all_results[task].keys())

    # Filter to only models in metadata (exclude unknowns)
    common_models = [m for m in common_models if m in MODEL_METADATA]

    # Build dataframe with best errors for each task
    data = []
    for model in common_models:
        row = {"model": model}
        meta = MODEL_METADATA[model]
        row["pretrain_lang"] = meta[0]
        row["is_baseline"] = meta[0] == "Baseline"
        for task in tasks:
            row[task] = get_best_error(all_results[task], model)
        data.append(row)

    df = pd.DataFrame(data)

    # Create pairwise scatter plots
    task_pairs = [
        ("mandarin_tone", "pitch_accent"),
        ("mandarin_tone", "stress"),
        ("pitch_accent", "stress"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for idx, (task1, task2) in enumerate(task_pairs):
        ax = axes[idx]

        # Plot baselines
        baselines = df[df["is_baseline"]]
        if not baselines.empty:
            ax.scatter(
                baselines[task1],
                baselines[task2],
                c="tab:gray",
                alpha=0.8,
                s=80,
                marker="s",
                zorder=2,
                edgecolors="white",
                linewidth=0.5,
                label="Baseline",
            )

        # Plot SSL models
        ssl_models = df[~df["is_baseline"]]
        if not ssl_models.empty:
            ax.scatter(
                ssl_models[task1],
                ssl_models[task2],
                c="tab:blue",
                alpha=0.6,
                s=60,
                marker="o",
                zorder=2,
                edgecolors="white",
                linewidth=0.5,
                label="SSL",
            )

        # Add correlation line and stats
        valid = df[[task1, task2]].dropna()
        if len(valid) > 2:
            r, p = stats.pearsonr(valid[task1], valid[task2])

            # Fit line
            z = np.polyfit(valid[task1], valid[task2], 1)
            p_line = np.poly1d(z)
            x_line = np.linspace(valid[task1].min(), valid[task1].max(), 100)
            ax.plot(x_line, p_line(x_line), "k--", alpha=0.5, linewidth=1)

            ax.text(
                0.05,
                0.95,
                f"r = {r:.3f}\np = {p:.3e}",
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        ax.set_xlabel(f"{task_labels[task1]} Error Rate")
        ax.set_ylabel(f"{task_labels[task2]} Error Rate")
        ax.set_title(f"{task_labels[task1]} vs {task_labels[task2]}")
        ax.grid(True, alpha=0.3)

        # Add diagonal reference line (y=x)
        lims = [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1]),
        ]
        ax.plot(lims, lims, "gray", alpha=0.3, linestyle=":")

        if idx == 0:
            ax.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(
        output_dir / "cross_task_correlation_simple.png", dpi=150, bbox_inches="tight"
    )

    # Remove titles for PDF
    for ax in axes:
        ax.set_title("")
    plt.savefig(
        output_dir / "cross_task_correlation_simple.pdf", dpi=150, bbox_inches="tight"
    )
    plt.close()

    print(f"  - cross_task_correlation_simple.png")
    return df


def plot_cross_task_correlation_all_layers(all_results, output_dir):
    """
    Cross-task correlation plot including ALL layers for each model.
    """
    tasks = list(all_results.keys())
    task_labels = {
        "mandarin_tone": "Mandarin Tone",
        "stress": "Lexical Stress",
        "pitch_accent": "Pitch Accent",
    }

    # Get all models present in all tasks
    common_models = set(all_results[tasks[0]].keys())
    for task in tasks[1:]:
        common_models &= set(all_results[task].keys())

    # Filter to only models in metadata (exclude unknowns)
    common_models = [m for m in common_models if m in MODEL_METADATA]

    # Collect data points
    data_points = []

    for model in common_models:
        meta = MODEL_METADATA[model]
        is_baseline = meta[0] == "Baseline"

        # Get dfs for this model
        dfs = {task: all_results[task][model] for task in tasks}

        # Drive by the layers in the first task
        # We assume layers match across tasks for the same model
        ref_task = tasks[0]
        ref_df = dfs[ref_task]

        for _, row in ref_df.iterrows():
            layer = row["layer"]

            point = {
                "model": model,
                "layer": layer,
                "is_baseline": is_baseline,
                ref_task: row["error_rate"],
            }

            valid_point = True
            for other_task in tasks[1:]:
                other_df = dfs[other_task]
                match = other_df[other_df["layer"] == layer]
                if not match.empty:
                    point[other_task] = match.iloc[0]["error_rate"]
                else:
                    valid_point = False
                    break

            if valid_point:
                data_points.append(point)

    df = pd.DataFrame(data_points)

    # Create pairwise scatter plots
    task_pairs = [
        ("mandarin_tone", "pitch_accent"),
        ("mandarin_tone", "stress"),
        ("pitch_accent", "stress"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for idx, (task1, task2) in enumerate(task_pairs):
        ax = axes[idx]

        # Plot baselines
        baselines = df[df["is_baseline"]]
        if not baselines.empty:
            ax.scatter(
                baselines[task1],
                baselines[task2],
                c="tab:gray",
                alpha=0.8,
                s=80,
                marker="s",
                zorder=2,
                edgecolors="white",
                linewidth=0.5,
                label="Baseline",
            )

        # Plot SSL models
        ssl_models = df[~df["is_baseline"]]
        if not ssl_models.empty:
            # Color points by relative depth if possible, or just blue
            # For simplicity, keep it consistent with the simple plot
            ax.scatter(
                ssl_models[task1],
                ssl_models[task2],
                c="tab:blue",
                alpha=0.3,  # Lower alpha for density
                s=40,
                marker="o",
                zorder=2,
                edgecolors="none",  # Remove edge for density
                label="SSL Layers",
            )

        # Add correlation line and stats
        valid = df[[task1, task2]].dropna()
        if len(valid) > 2:
            r, p = stats.pearsonr(valid[task1], valid[task2])

            # Fit line
            z = np.polyfit(valid[task1], valid[task2], 1)
            p_line = np.poly1d(z)
            x_line = np.linspace(valid[task1].min(), valid[task1].max(), 100)
            ax.plot(x_line, p_line(x_line), "k--", alpha=0.5, linewidth=1)

            ax.text(
                0.05,
                0.95,
                f"r = {r:.3f}\np = {p:.3e}",
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        ax.set_xlabel(f"{task_labels[task1]} Error Rate")
        ax.set_ylabel(f"{task_labels[task2]} Error Rate")
        ax.set_title(f"{task_labels[task1]} vs {task_labels[task2]}\n(All Layers)")
        ax.grid(True, alpha=0.3)

        # Add diagonal reference line (y=x)
        lims = [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1]),
        ]
        ax.plot(lims, lims, "gray", alpha=0.3, linestyle=":")

        if idx == 0:
            ax.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(
        output_dir / "cross_task_correlation_all_layers.png",
        dpi=150,
        bbox_inches="tight",
    )

    # Remove titles for PDF
    for ax in axes:
        ax.set_title("")
    plt.savefig(
        output_dir / "cross_task_correlation_all_layers.pdf",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    print(f"  - cross_task_correlation_all_layers.png")
    return df


def _get_cross_task_data(all_results):
    """Prepare data for cross-task correlation plots."""
    tasks = list(all_results.keys())

    # Get all models present in all tasks
    common_models = set(all_results[tasks[0]].keys())
    for task in tasks[1:]:
        common_models &= set(all_results[task].keys())

    # Filter to only models in metadata (exclude unknowns)
    common_models = [m for m in common_models if m in MODEL_METADATA]

    # Build dataframe with best errors for each task
    data = []
    for model in common_models:
        row = {"model": model}
        meta = MODEL_METADATA[model]
        row["pretrain_lang"] = meta[0]
        row["architecture"] = meta[1]
        row["finetuned"] = meta[4]
        for task in tasks:
            row[task] = get_best_error(all_results[task], model)
        data.append(row)

    return pd.DataFrame(data)


def _plot_single_cross_task_panel(
    ax,
    df,
    task1,
    task2,
    task_labels,
    finetune_pairs,
    ft_lang_colors,
    get_arch_color,
    add_legend=False,
):
    """Plot a single cross-task correlation panel."""
    from matplotlib.lines import Line2D

    # Draw lines from pretrained to finetuned models
    for pt_model, ft_model in finetune_pairs:
        pt_row = df[df["model"] == pt_model]
        ft_row = df[df["model"] == ft_model]
        if len(pt_row) > 0 and len(ft_row) > 0:
            ax.plot(
                [pt_row[task1].values[0], ft_row[task1].values[0]],
                [pt_row[task2].values[0], ft_row[task2].values[0]],
                color="gray",
                alpha=0.3,
                linewidth=1,
                zorder=1,
            )

    # Plot all models
    for _, row in df.iterrows():
        if row["pretrain_lang"] == "Baseline":
            color = "tab:gray"
            marker = "s"
            size = 60
        elif row["finetuned"]:
            ft_meta = MODEL_METADATA.get(row["model"])
            ft_lang = ft_meta[5] if ft_meta else None
            color = ft_lang_colors.get(ft_lang, "tab:brown")
            marker = "^"
            size = 60
        else:
            color = get_arch_color(row)
            marker = "o"
            size = 60

        ax.scatter(
            row[task1],
            row[task2],
            c=color,
            alpha=0.7,
            s=size,
            marker=marker,
            zorder=2,
            edgecolors="white",
            linewidth=0.5,
        )

    # Add correlation line and stats
    valid = df[[task1, task2]].dropna()
    if len(valid) > 2:
        r, p = stats.pearsonr(valid[task1], valid[task2])

        z = np.polyfit(valid[task1], valid[task2], 1)
        p_line = np.poly1d(z)
        x_line = np.linspace(valid[task1].min(), valid[task1].max(), 100)
        ax.plot(x_line, p_line(x_line), "k--", alpha=0.5, linewidth=1)

        ax.text(
            0.05,
            0.95,
            f"r = {r:.3f}\np = {p:.3e}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    ax.set_xlabel(f"{task_labels[task1]} Error Rate")
    ax.set_ylabel(f"{task_labels[task2]} Error Rate")
    ax.set_title(f"{task_labels[task1]} vs {task_labels[task2]}")
    ax.grid(True, alpha=0.3)

    # Scale axes based on non-baseline models
    baselines = df[df["pretrain_lang"] == "Baseline"]
    non_baseline = df[df["pretrain_lang"] != "Baseline"]
    baselines_out_of_view = False
    if not non_baseline.empty:
        x_min, x_max = non_baseline[task1].min(), non_baseline[task1].max()
        y_min, y_max = non_baseline[task2].min(), non_baseline[task2].max()

        x_pad = (x_max - x_min) * 0.1
        y_pad = (y_max - y_min) * 0.1

        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

        if not baselines.empty:
            current_xlim = ax.get_xlim()
            current_ylim = ax.get_ylim()

            out_of_view = baselines[
                (baselines[task1] < current_xlim[0])
                | (baselines[task1] > current_xlim[1])
                | (baselines[task2] < current_ylim[0])
                | (baselines[task2] > current_ylim[1])
            ]

            if not out_of_view.empty:
                baselines_out_of_view = True

    # Add diagonal reference line (y=x)
    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(lims, lims, "gray", alpha=0.3, linestyle=":")

    if add_legend:
        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                markerfacecolor="tab:gray",
                markersize=8,
                label="Baseline (not shown)" if baselines_out_of_view else "Baseline",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="tab:orange",
                markersize=8,
                label="HuBERT",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="tab:green",
                markersize=8,
                label="Wav2Vec2/XLSR",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="tab:cyan",
                markersize=8,
                label="WavLM",
            ),
            Line2D(
                [0],
                [0],
                marker="^",
                color="w",
                markerfacecolor="tab:blue",
                markersize=8,
                label="FT: English",
            ),
            Line2D(
                [0],
                [0],
                marker="^",
                color="w",
                markerfacecolor="tab:purple",
                markersize=8,
                label="FT: Japanese",
            ),
            Line2D(
                [0],
                [0],
                marker="^",
                color="w",
                markerfacecolor="tab:red",
                markersize=8,
                label="FT: Chinese",
            ),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=7)


def plot_cross_task_correlation(all_results, output_dir):
    """
    Plot 1: Scatter plots showing correlation between tasks.
    Shows whether models good at one prosodic task are good at others.
    Lines connect pretrained models to their finetuned children.
    Generates both combined and individual plots.
    """
    task_labels = {
        "mandarin_tone": "Mandarin Tone",
        "stress": "Lexical Stress",
        "pitch_accent": "Pitch Accent",
    }

    df = _get_cross_task_data(all_results)

    # Define pretrained -> finetuned relationships
    finetune_pairs = [
        ("hubert_large", "hubert_asr_large"),
        ("wav2vec2_large", "wav2vec2_asr_large_960h"),
        ("wav2vec2-large-xlsr-53", "wav2vec2-large-xlsr-53-english"),
        ("wav2vec2-large-xlsr-53", "wav2vec2-large-xlsr-53-japanese"),
        ("wav2vec2-large-xlsr-53", "wav2vec2-large-xlsr-53-chinese-zh-cn"),
        ("japanese-wav2vec2-base", "japanese-wav2vec2-base-rs35kh"),
        ("japanese-wav2vec2-large", "japanese-wav2vec2-large-rs35kh"),
        ("japanese-hubert-base-k2", "japanese-hubert-base-k2-rs35kh"),
    ]

    # Colors for finetuning language
    ft_lang_colors = {
        "English": "tab:blue",
        "Japanese": "tab:purple",
        "Chinese": "tab:red",
    }

    # Define architecture colors
    def get_arch_color(model_info):
        arch = model_info["architecture"]
        if "HuBERT" in arch:
            return "tab:orange"
        if "Wav2Vec2" in arch:
            return "tab:green"
        if "WavLM" in arch:
            return "tab:cyan"
        if model_info["pretrain_lang"] == "Baseline":
            return "tab:gray"
        return "tab:purple"

    task_pairs = [
        ("mandarin_tone", "pitch_accent"),
        ("mandarin_tone", "stress"),
        ("pitch_accent", "stress"),
    ]

    # Generate individual plots for each task pair
    for task1, task2 in task_pairs:
        fig, ax = plt.subplots(figsize=(6, 5))
        _plot_single_cross_task_panel(
            ax,
            df,
            task1,
            task2,
            task_labels,
            finetune_pairs,
            ft_lang_colors,
            get_arch_color,
            add_legend=True,
        )
        plt.tight_layout()
        filename = f"cross_task_{task1}_vs_{task2}.png"
        plt.savefig(output_dir / filename, dpi=150, bbox_inches="tight")

        # Remove title for PDF
        ax.set_title("")
        pdf_filename = f"cross_task_{task1}_vs_{task2}.pdf"
        plt.savefig(output_dir / pdf_filename, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  - {filename}")

    # Generate combined plot
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, (task1, task2) in enumerate(task_pairs):
        _plot_single_cross_task_panel(
            axes[idx],
            df,
            task1,
            task2,
            task_labels,
            finetune_pairs,
            ft_lang_colors,
            get_arch_color,
            add_legend=False,
        )

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="tab:gray",
            markersize=8,
            label="Baseline",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="tab:orange",
            markersize=8,
            label="HuBERT",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="tab:green",
            markersize=8,
            label="Wav2Vec2/XLSR",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="tab:cyan",
            markersize=8,
            label="WavLM",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="tab:blue",
            markersize=8,
            label="FT: English",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="tab:purple",
            markersize=8,
            label="FT: Japanese",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="tab:red",
            markersize=8,
            label="FT: Chinese",
        ),
    ]

    fig.legend(
        handles=legend_elements,
        loc="center right",
        bbox_to_anchor=(1.12, 0.5),
        fontsize=9,
    )

    plt.tight_layout()
    plt.savefig(output_dir / "cross_task_correlation.png", dpi=150, bbox_inches="tight")

    # Remove titles for PDF
    for ax in axes:
        ax.set_title("")
    plt.savefig(output_dir / "cross_task_correlation.pdf", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  - cross_task_correlation.png")
    return df


def plot_finetune_language_effect(all_results, output_dir):
    """
    Analyze effect of finetuning language match/mismatch.
    Shows whether finetuning on the same language as the test helps more than cross-language.
    """
    # Define which tasks correspond to which languages
    task_languages = {
        "mandarin_tone": "Chinese",
        "pitch_accent": "Japanese",
        "stress": "English",
    }

    # XLSR finetuned models with their finetuning languages
    xlsr_models = {
        "wav2vec2-large-xlsr-53": None,  # Base (not finetuned)
        "wav2vec2-large-xlsr-53-english": "English",
        "wav2vec2-large-xlsr-53-japanese": "Japanese",
        "wav2vec2-large-xlsr-53-chinese-zh-cn": "Chinese",
    }

    # Collect data
    data = []
    for task, task_lang in task_languages.items():
        if task not in all_results:
            continue
        results = all_results[task]

        for model, ft_lang in xlsr_models.items():
            if model not in results:
                continue

            error = get_best_error(results, model)
            match_type = "Base (No FT)"
            if ft_lang is not None:
                match_type = "Match" if ft_lang == task_lang else "Mismatch"

            data.append(
                {
                    "model": model,
                    "task": DATASET_LABELS.get(task, task),
                    "task_lang": task_lang,
                    "ft_lang": ft_lang if ft_lang else "None",
                    "error": error,
                    "match_type": match_type,
                }
            )

    df = pd.DataFrame(data)

    if df.empty:
        print("  - No XLSR model data found, skipping finetuning analysis")
        return None

    # Create two plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Grouped bar chart by task
    ax = axes[0]
    tasks = df["task"].unique()
    x = np.arange(len(tasks))
    width = 0.2

    ft_langs = ["None", "English", "Japanese", "Chinese"]
    colors = ["tab:gray", "tab:blue", "tab:purple", "tab:red"]

    for i, (ft_lang, color) in enumerate(zip(ft_langs, colors)):
        errors = []
        for task in tasks:
            subset = df[(df["task"] == task) & (df["ft_lang"] == ft_lang)]
            errors.append(subset["error"].values[0] if len(subset) > 0 else np.nan)

        offset = (i - 1.5) * width
        bars = ax.bar(
            x + offset, errors, width, label=f"FT: {ft_lang}", color=color, alpha=0.7
        )

    ax.set_ylabel("Error Rate")
    ax.set_title("XLSR-53 Performance by Finetuning Language")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.legend(title="Finetuning Lang")
    ax.grid(True, axis="y", alpha=0.3)

    # Plot 2: Match vs Mismatch comparison
    ax = axes[1]

    # Only look at finetuned models (exclude base)
    ft_only = df[df["match_type"] != "Base (No FT)"]

    match_data = ft_only[ft_only["match_type"] == "Match"]["error"].values
    mismatch_data = ft_only[ft_only["match_type"] == "Mismatch"]["error"].values

    positions = [1, 2]
    bp = ax.boxplot([match_data, mismatch_data], positions=positions, widths=0.5)

    # Scatter individual points
    for i, (pos, data) in enumerate(zip(positions, [match_data, mismatch_data])):
        x_jitter = np.random.normal(pos, 0.08, len(data))
        ax.scatter(x_jitter, data, alpha=0.7, s=60, zorder=3)

    ax.set_xticklabels(["Language Match", "Language Mismatch"])
    ax.set_ylabel("Error Rate")
    ax.set_title("Effect of Finetuning Language Match\n(XLSR-53 Finetuned Models)")
    ax.grid(True, axis="y", alpha=0.3)

    # Add stats
    if len(match_data) > 0 and len(mismatch_data) > 0:
        t_stat, p_val = stats.ttest_ind(match_data, mismatch_data)
        ax.text(
            0.95,
            0.95,
            f"Match mean: {match_data.mean():.3f}\n"
            f"Mismatch mean: {mismatch_data.mean():.3f}\n"
            f"p = {p_val:.3f}",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    plt.tight_layout()
    plt.savefig(output_dir / "finetune_language_effect.png", dpi=150)

    # Remove titles for PDF
    for ax in axes:
        ax.set_title("")
    plt.savefig(output_dir / "finetune_language_effect.pdf", dpi=150)
    plt.close()

    print(f"  - finetune_language_effect.png")
    return df


def plot_finetune_effect_by_model(all_results, output_dir):
    """
    Show how finetuning affects performance across all three tasks, stratified by base model.
    Each panel shows a pretrained model and its finetuned variants across the three tasks.
    """
    tasks = ["mandarin_tone", "stress", "pitch_accent"]
    task_labels = {
        "mandarin_tone": "Mandarin\nTone",
        "stress": "Lexical\nStress",
        "pitch_accent": "Pitch\nAccent",
    }

    # Define pretrained -> finetuned relationships grouped by base model
    # Layout: Col1=HuBERT, Col2=Wav2Vec2 Japanese, Col3=Wav2Vec2 English/XLSR
    # Row 1: Large models, Row 2: Base models (or XLSR)
    model_families = {
        # Row 1
        "HuBERT Large (English)": {
            "base": "hubert_large",
            "finetuned": [("hubert_asr_large", "English ASR")],
        },
        "Wav2Vec2 Large (Japanese)": {
            "base": "japanese-wav2vec2-large",
            "finetuned": [("japanese-wav2vec2-large-rs35kh", "Japanese ASR")],
        },
        "Wav2Vec2 Large (English)": {
            "base": "wav2vec2_large",
            "finetuned": [("wav2vec2_asr_large_960h", "English ASR")],
        },
        # Row 2
        "HuBERT Base (Japanese)": {
            "base": "japanese-hubert-base-k2",
            "finetuned": [("japanese-hubert-base-k2-rs35kh", "Japanese ASR")],
        },
        "Wav2Vec2 Base (Japanese)": {
            "base": "japanese-wav2vec2-base",
            "finetuned": [("japanese-wav2vec2-base-rs35kh", "Japanese ASR")],
        },
        "XLSR-53 (Multilingual)": {
            "base": "wav2vec2-large-xlsr-53",
            "finetuned": [
                ("wav2vec2-large-xlsr-53-english", "English ASR"),
                ("wav2vec2-large-xlsr-53-japanese", "Japanese ASR"),
                ("wav2vec2-large-xlsr-53-chinese-zh-cn", "Chinese ASR"),
            ],
        },
    }

    # Colors for finetuning language
    ft_colors = {
        "English ASR": "tab:blue",
        "Japanese ASR": "tab:purple",
        "Chinese ASR": "tab:red",
    }

    # Create figure with subplots for each model family
    n_families = len(model_families)
    n_cols = 3
    n_rows = (n_families + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes = axes.flatten() if n_families > 1 else [axes]

    x = np.arange(len(tasks))

    for idx, (family_name, family_info) in enumerate(model_families.items()):
        ax = axes[idx]
        base_model = family_info["base"]

        # Get base model errors
        base_errors = []
        for task in tasks:
            if task in all_results and base_model in all_results[task]:
                base_errors.append(get_best_error(all_results[task], base_model))
            else:
                base_errors.append(np.nan)

        # Plot base model
        ax.plot(
            x,
            base_errors,
            "ko-",
            linewidth=2,
            markersize=10,
            label="Pretrained",
            zorder=3,
        )

        # Plot finetuned variants
        for ft_model, ft_label in family_info["finetuned"]:
            ft_errors = []
            for task in tasks:
                if task in all_results and ft_model in all_results[task]:
                    ft_errors.append(get_best_error(all_results[task], ft_model))
                else:
                    ft_errors.append(np.nan)

            color = ft_colors.get(ft_label, "tab:gray")
            ax.plot(
                x,
                ft_errors,
                marker="^",
                linestyle="--",
                linewidth=2,
                markersize=10,
                label=ft_label,
                color=color,
                zorder=2,
            )

            # Draw arrows showing direction of change
            for i, (base_err, ft_err) in enumerate(zip(base_errors, ft_errors)):
                if not np.isnan(base_err) and not np.isnan(ft_err):
                    delta = ft_err - base_err
                    # Small vertical offset for arrow
                    ax.annotate(
                        "",
                        xy=(i, ft_err),
                        xytext=(i, base_err),
                        arrowprops=dict(
                            arrowstyle="->", color=color, alpha=0.5, lw=1.5
                        ),
                    )

        ax.set_xticks(x)
        ax.set_xticklabels([task_labels[t] for t in tasks])
        ax.set_ylabel("Error Rate")
        ax.set_title(family_name, fontweight="bold")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

        # Set consistent y-axis range based on data
        all_errors = base_errors + [
            e
            for ft_m, _ in family_info["finetuned"]
            for task in tasks
            for e in [get_best_error(all_results.get(task, {}), ft_m)]
            if e is not None
        ]
        all_errors = [e for e in all_errors if e is not None and not np.isnan(e)]
        if all_errors:
            y_min, y_max = min(all_errors), max(all_errors)
            padding = (y_max - y_min) * 0.15
            ax.set_ylim(y_min - padding, y_max + padding)

    # Hide unused subplots
    for idx in range(len(model_families), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(
        "Effect of ASR Finetuning on Prosodic Task Performance",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(
        output_dir / "finetune_effect_by_model.png", dpi=150, bbox_inches="tight"
    )

    # Remove titles for PDF
    plt.suptitle("")
    for ax in axes:
        ax.set_title("")
    plt.savefig(
        output_dir / "finetune_effect_by_model.pdf", dpi=150, bbox_inches="tight"
    )
    plt.close()

    print(f"  - finetune_effect_by_model.png")

    return None


def plot_architecture_controlled_language(all_results, output_dir):
    """
    Compare pretraining languages while controlling for architecture.
    Only compares models with same architecture and size.
    """
    # Define controlled comparison groups
    # Each group: (architecture, size, {lang: model_name})
    comparison_groups = [
        (
            "HuBERT",
            "Large",
            {
                "English": "hubert_large",
                "Chinese": "chinese-hubert-large",
                "Japanese": "japanese-hubert-large",
            },
        ),
        (
            "HuBERT",
            "Base",
            {
                "English": "hubert_base",
                "Chinese": "chinese-hubert-base",
                "Japanese": "japanese-hubert-base-k2",
            },
        ),
        (
            "Wav2Vec2",
            "Large",
            {
                "English": "wav2vec2_large_lv60k",
                "Chinese": "chinese-wav2vec2-large",
                "Japanese": "japanese-wav2vec2-large",
            },
        ),
        (
            "Wav2Vec2",
            "Base",
            {
                "English": "wav2vec2_base",
                "Chinese": "chinese-wav2vec2-base",
                "Japanese": "japanese-wav2vec2-base",
            },
        ),
    ]

    tasks = ["mandarin_tone", "stress", "pitch_accent"]
    task_labels = {
        "mandarin_tone": "Mandarin Tone",
        "stress": "Lexical Stress",
        "pitch_accent": "Pitch Accent",
    }

    lang_colors = {
        "English": "tab:blue",
        "Chinese": "tab:red",
        "Japanese": "tab:purple",
    }

    # Create subplot for each comparison group
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (arch, size, lang_models) in enumerate(comparison_groups):
        ax = axes[idx]

        x = np.arange(len(tasks))
        width = 0.25

        for i, (lang, model) in enumerate(lang_models.items()):
            errors = []
            for task in tasks:
                if task in all_results and model in all_results[task]:
                    errors.append(get_best_error(all_results[task], model))
                else:
                    errors.append(np.nan)

            offset = (i - len(lang_models) / 2 + 0.5) * width
            bars = ax.bar(
                x + offset,
                errors,
                width,
                label=lang,
                color=lang_colors[lang],
                alpha=0.7,
            )

        ax.set_ylabel("Error Rate")
        ax.set_title(f"{arch} {size}")
        ax.set_xticks(x)
        ax.set_xticklabels([task_labels[t] for t in tasks], fontsize=9)
        ax.legend(title="Pretrain Lang", fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

    plt.suptitle(
        "Pretraining Language Comparison\n(Architecture & Size Controlled)",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_dir / "architecture_controlled_language.png", dpi=150)

    # Remove titles for PDF
    plt.suptitle("")
    for ax in axes:
        ax.set_title("")
    plt.savefig(output_dir / "architecture_controlled_language.pdf", dpi=150)
    plt.close()

    print(f"  - architecture_controlled_language.png")

    return None


def plot_cross_dataset_heatmap(all_results, output_dir):
    """
    Create a cross-dataset heatmap showing the effect of pretraining language
    and model architecture on performance, excluding finetuned models.

    Rows: Pretraining language
    Columns: Model architecture
    Values: Mean best-layer error rate averaged across all three datasets
    """
    import seaborn as sns

    tasks = list(all_results.keys())
    task_labels = {
        "mandarin_tone": "Mandarin Tone",
        "stress": "Lexical Stress",
        "pitch_accent": "Pitch Accent",
    }

    # Collect data for all non-finetuned models across all tasks
    model_task_errors = {}  # model_name -> {task: best_error}

    for task in tasks:
        for model_name, df in all_results[task].items():
            if model_name not in MODEL_METADATA:
                continue
            meta = MODEL_METADATA[model_name]
            pretrain_lang = meta[0]
            finetuned = meta[4]

            # Skip baselines and finetuned models
            if pretrain_lang == "Baseline" or finetuned:
                continue

            if model_name not in model_task_errors:
                model_task_errors[model_name] = {}
            model_task_errors[model_name][task] = df["error_rate"].min()

    # Build aggregated data by (pretrain_lang, architecture)
    # For each cell, average across models and then across tasks
    agg_data = {}  # (lang, arch) -> list of per-model mean errors

    for model_name, task_errors in model_task_errors.items():
        meta = MODEL_METADATA[model_name]
        pretrain_lang = meta[0]
        architecture = meta[1]
        # Merge XLSR into Wav2Vec2
        if architecture == "Wav2Vec2-XLSR":
            architecture = "Wav2Vec2"

        # Calculate mean error across available tasks for this model
        if task_errors:
            mean_error = np.mean(list(task_errors.values()))
            key = (pretrain_lang, architecture)
            if key not in agg_data:
                agg_data[key] = []
            agg_data[key].append(mean_error)

    # Get unique languages and architectures, sorted
    # Exclude WavLM (only English data, not informative for cross-language comparison)
    langs = sorted(set(k[0] for k in agg_data.keys()))
    archs = sorted(k[1] for k in agg_data.keys() if k[1] != "WavLM")
    archs = sorted(set(archs))

    # Create matrix with mean values
    matrix = np.full((len(langs), len(archs)), np.nan)
    for (lang, arch), errors in agg_data.items():
        if arch not in archs:
            continue
        i = langs.index(lang)
        j = archs.index(arch)
        matrix[i, j] = np.mean(errors)

    # Dynamic color range based on actual data
    valid_vals = matrix[~np.isnan(matrix)]
    vmin = valid_vals.min() - 0.01
    vmax = valid_vals.max() + 0.01

    # Create the heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(matrix, cmap="RdYlGn_r", vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(archs)))
    ax.set_xticklabels(archs, rotation=45, ha="right", fontsize=11)
    ax.set_yticks(range(len(langs)))
    ax.set_yticklabels(langs, fontsize=11)

    ax.set_xlabel("Model Architecture", fontsize=12)
    ax.set_ylabel("Pretraining Language", fontsize=12)

    # Add text annotations with dynamic text color
    for i in range(len(langs)):
        for j in range(len(archs)):
            if not np.isnan(matrix[i, j]):
                val_norm = (matrix[i, j] - vmin) / (vmax - vmin)
                text_color = "white" if val_norm > 0.6 else "black"
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:.3f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=11,
                    fontweight="bold",
                )

    ax.set_title(
        "Mean Error Rate by Pretraining Language & Architecture\n"
        "(Averaged Across All Tasks, Excluding Finetuned Models)",
        fontsize=12,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(output_dir / "cross_dataset_heatmap.png", dpi=150, bbox_inches="tight")

    # Remove title for PDF version
    ax.set_title("")
    plt.savefig(output_dir / "cross_dataset_heatmap.pdf", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  - cross_dataset_heatmap.png")

    # Create per-task heatmaps side by side with shared y-axis
    fig, axes = plt.subplots(1, 3, figsize=(9, 4), sharey=True)

    for idx, task in enumerate(tasks):
        ax = axes[idx]

        # Build task-specific matrix
        task_agg = {}
        for model_name, task_errors in model_task_errors.items():
            if task not in task_errors:
                continue
            meta = MODEL_METADATA[model_name]
            arch = meta[1]
            if arch == "Wav2Vec2-XLSR":
                arch = "Wav2Vec2"
            key = (meta[0], arch)
            if key not in task_agg:
                task_agg[key] = []
            task_agg[key].append(task_errors[task])

        task_matrix = np.full((len(langs), len(archs)), np.nan)
        for (lang, arch), errors in task_agg.items():
            if lang in langs and arch in archs:
                i = langs.index(lang)
                j = archs.index(arch)
                task_matrix[i, j] = np.mean(errors)

        # Task-specific color range
        task_valid = task_matrix[~np.isnan(task_matrix)]
        task_vmin = task_valid.min() - 0.01
        task_vmax = task_valid.max() + 0.01

        im = ax.imshow(
            task_matrix, cmap="RdYlGn_r", vmin=task_vmin, vmax=task_vmax, aspect="auto"
        )

        ax.set_xticks(range(len(archs)))
        ax.set_xticklabels(archs, rotation=45, ha="right", fontsize=9)

        # Add text annotations with task-specific normalization
        for i in range(len(langs)):
            for j in range(len(archs)):
                if not np.isnan(task_matrix[i, j]):
                    val_norm = (task_matrix[i, j] - task_vmin) / (task_vmax - task_vmin)
                    text_color = "white" if val_norm > 0.6 else "black"
                    ax.text(
                        j,
                        i,
                        f"{task_matrix[i, j]:.3f}",
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=9,
                        fontweight="bold",
                    )

        ax.set_title(task_labels.get(task, task), fontsize=11, fontweight="bold")

    # Only set y-axis labels on the first panel
    axes[0].set_yticks(range(len(langs)))
    axes[0].set_yticklabels(langs, fontsize=10)
    axes[0].set_ylabel("Pretraining Language", fontsize=10)

    plt.tight_layout()
    plt.savefig(
        output_dir / "cross_dataset_heatmap_by_task.png", dpi=150, bbox_inches="tight"
    )

    # Remove titles for PDF
    for ax in axes:
        ax.set_title("")
    plt.savefig(
        output_dir / "cross_dataset_heatmap_by_task.pdf", dpi=150, bbox_inches="tight"
    )
    plt.close()

    print(f"  - cross_dataset_heatmap_by_task.png")

    return matrix, langs, archs


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading all task results...")
    all_results = load_all_tasks()

    print("\nGenerating cross-analysis plots...")

    plot_cross_task_correlation_simple(all_results, OUTPUT_DIR)
    plot_cross_task_correlation_all_layers(all_results, OUTPUT_DIR)
    plot_cross_task_correlation(all_results, OUTPUT_DIR)
    plot_finetune_effect_by_model(all_results, OUTPUT_DIR)
    plot_finetune_language_effect(all_results, OUTPUT_DIR)
    plot_architecture_controlled_language(all_results, OUTPUT_DIR)
    plot_cross_dataset_heatmap(all_results, OUTPUT_DIR)

    print(f"\nAll plots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
