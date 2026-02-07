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
SAVEFIG_KW = {"dpi": 150, "bbox_inches": "tight", "pad_inches": 0.05}

SCATTER_S_LARGE = 180
SCATTER_S_MED = 140
SCATTER_S_SMALL = 100
SCATTER_EDGEWIDTH = 0.9

LINEWIDTH_THIN = 1.5
LINEWIDTH_MED = 2.0
LINEWIDTH_BOLD = 2.5

MARKERSIZE_MED = 12
MARKERSIZE_LARGE = 14


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
            task_results = load_results(task_dir)
            filtered_results = {}
            missing = []
            for model_name, df in task_results.items():
                if model_name in MODEL_METADATA:
                    filtered_results[model_name] = df
                else:
                    missing.append(model_name)
            if missing:
                missing_list = ", ".join(sorted(missing))
                print(
                    f"Warning: {task_label} has models missing metadata, skipping: {missing_list}"
                )
            all_results[task_key] = filtered_results
            print(f"Loaded {len(all_results[task_key])} models for {task_label}")

    return all_results


def plot_cross_task_correlation_simple(all_results, output_dir):
    """
    Cross-task correlation plot with all layer combinations.
    Plots one point per (model, layer) pair, excluding baselines.
    """
    from plot_prosodic_results import get_layer_num

    tasks = list(all_results.keys())
    task_labels = {
        "mandarin_tone": "Mandarin Tone",
        "stress": "Lexical Stress",
        "pitch_accent": "Pitch Accent",
    }

    # Get all non-baseline models present in all tasks
    common_models = set(all_results[tasks[0]].keys())
    for task in tasks[1:]:
        common_models &= set(all_results[task].keys())
    common_models = [
        m for m in common_models
        if m in MODEL_METADATA and MODEL_METADATA[m][0] != "Baseline"
    ]

    # Build dataframe with one row per (model, layer) combination.
    # Match layers across tasks by layer number.
    data = []
    for model in common_models:
        # Build {layer_num: error_rate} for each task
        task_layer_errors = {}
        for task in tasks:
            df_model = all_results[task][model]
            task_layer_errors[task] = {
                get_layer_num(row["layer"]): row["error_rate"]
                for _, row in df_model.iterrows()
            }

        # Find layer numbers common to all tasks
        common_layers = set(task_layer_errors[tasks[0]].keys())
        for task in tasks[1:]:
            common_layers &= set(task_layer_errors[task].keys())

        for layer_num in sorted(common_layers):
            row = {"model": model, "layer": layer_num}
            for task in tasks:
                row[task] = task_layer_errors[task][layer_num]
            data.append(row)

    df = pd.DataFrame(data)

    # Create pairwise scatter plots
    task_pairs = [
        ("mandarin_tone", "pitch_accent"),
        ("mandarin_tone", "stress"),
        ("pitch_accent", "stress"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))

    for idx, (task1, task2) in enumerate(task_pairs):
        ax = axes[idx]

        ax.scatter(
            df[task1],
            df[task2],
            c="tab:blue",
            alpha=0.3,
            s=SCATTER_S_SMALL,
            marker="o",
            zorder=2,
            edgecolors="white",
            linewidth=SCATTER_EDGEWIDTH,
        )

        # Add correlation line and stats
        valid = df[[task1, task2]].dropna()
        if len(valid) > 2:
            r, p = stats.pearsonr(valid[task1], valid[task2])

            # Fit line
            z = np.polyfit(valid[task1], valid[task2], 1)
            p_line = np.poly1d(z)
            x_line = np.linspace(valid[task1].min(), valid[task1].max(), 100)
            ax.plot(
                x_line,
                p_line(x_line),
                "k--",
                alpha=0.5,
                linewidth=LINEWIDTH_THIN,
            )

            ax.text(
                0.05,
                0.95,
                f"r = {r:.3f}",
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        ax.set_xlabel(task_labels[task1], fontsize=12)
        ax.set_ylabel(task_labels[task2], fontsize=12)
        ax.tick_params(labelsize=9)
        ax.set_title(f"{task_labels[task1]} vs {task_labels[task2]}")
        ax.set_box_aspect(1)
        ax.grid(True, alpha=0.3)

    plt.tight_layout(pad=0.3, w_pad=0.3, h_pad=0.3)
    plt.savefig(output_dir / "cross_task_correlation_simple.png", **SAVEFIG_KW)

    # Remove titles for PDF
    for ax in axes:
        ax.set_title("")
    plt.savefig(output_dir / "cross_task_correlation_simple.pdf", **SAVEFIG_KW)
    plt.close()

    print(f"  - cross_task_correlation_simple.png")
    return df


def _get_cross_task_data(all_results):
    """Prepare data for cross-task correlation plots."""
    tasks = list(all_results.keys())

    # Get all models present in all tasks
    common_models = set(all_results[tasks[0]].keys())
    for task in tasks[1:]:
        common_models &= set(all_results[task].keys())

    # Filter to only models in metadata, exclude baselines
    common_models = [
        m for m in common_models
        if m in MODEL_METADATA and MODEL_METADATA[m][0] != "Baseline"
    ]

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
                linewidth=LINEWIDTH_THIN,
                zorder=1,
            )

    # Plot all models
    for _, row in df.iterrows():
        if row["finetuned"]:
            ft_meta = MODEL_METADATA.get(row["model"])
            ft_lang = ft_meta[5] if ft_meta else None
            color = ft_lang_colors.get(ft_lang, "tab:brown")
            marker = "^"
            size = SCATTER_S_MED
        else:
            color = get_arch_color(row)
            marker = "o"
            size = SCATTER_S_MED

        ax.scatter(
            row[task1],
            row[task2],
            c=color,
            alpha=0.7,
            s=size,
            marker=marker,
            zorder=2,
            edgecolors="white",
            linewidth=SCATTER_EDGEWIDTH,
        )

    # Add correlation line and stats
    valid = df[[task1, task2]].dropna()
    if len(valid) > 2:
        r, p = stats.pearsonr(valid[task1], valid[task2])

        z = np.polyfit(valid[task1], valid[task2], 1)
        p_line = np.poly1d(z)
        x_line = np.linspace(valid[task1].min(), valid[task1].max(), 100)
        ax.plot(
            x_line,
            p_line(x_line),
            "k--",
            alpha=0.5,
            linewidth=LINEWIDTH_THIN,
        )

        ax.text(
            0.05,
            0.95,
            f"r = {r:.3f}",
            transform=ax.transAxes,
                            fontsize=13,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    ax.set_xlabel(f"{task_labels[task1]} Error Rate")
    ax.set_ylabel(f"{task_labels[task2]} Error Rate")
    ax.set_title(f"{task_labels[task1]} vs {task_labels[task2]}")
    ax.grid(True, alpha=0.3)

    if add_legend:
        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="tab:orange",
                markersize=MARKERSIZE_MED,
                label="HuBERT",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="tab:green",
                markersize=MARKERSIZE_MED,
                label="Wav2Vec2/XLSR",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="tab:cyan",
                markersize=MARKERSIZE_MED,
                label="WavLM",
            ),
            Line2D(
                [0],
                [0],
                marker="^",
                color="w",
                markerfacecolor="tab:blue",
                markersize=MARKERSIZE_MED,
                label="FT: English",
            ),
            Line2D(
                [0],
                [0],
                marker="^",
                color="w",
                markerfacecolor="tab:purple",
                markersize=MARKERSIZE_MED,
                label="FT: Japanese",
            ),
            Line2D(
                [0],
                [0],
                marker="^",
                color="w",
                markerfacecolor="tab:red",
                markersize=MARKERSIZE_MED,
                label="FT: Chinese",
            ),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=9)


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
        plt.tight_layout(**TIGHT_LAYOUT_KW)
        filename = f"cross_task_{task1}_vs_{task2}.png"
        plt.savefig(output_dir / filename, **SAVEFIG_KW)

        # Remove title for PDF
        ax.set_title("")
        pdf_filename = f"cross_task_{task1}_vs_{task2}.pdf"
        plt.savefig(output_dir / pdf_filename, **SAVEFIG_KW)
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
            marker="o",
            color="w",
            markerfacecolor="tab:orange",
            markersize=MARKERSIZE_MED,
            label="HuBERT",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="tab:green",
            markersize=MARKERSIZE_MED,
            label="Wav2Vec2/XLSR",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="tab:cyan",
            markersize=MARKERSIZE_MED,
            label="WavLM",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="tab:blue",
            markersize=MARKERSIZE_MED,
            label="FT: English",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="tab:purple",
            markersize=MARKERSIZE_MED,
            label="FT: Japanese",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="tab:red",
            markersize=MARKERSIZE_MED,
            label="FT: Chinese",
        ),
    ]

    fig.legend(
        handles=legend_elements,
        loc="center right",
        bbox_to_anchor=(1.12, 0.5),
        fontsize=10,
    )

    plt.tight_layout(**TIGHT_LAYOUT_KW)
    plt.savefig(output_dir / "cross_task_correlation.png", **SAVEFIG_KW)

    # Remove titles for PDF
    for ax in axes:
        ax.set_title("")
    plt.savefig(output_dir / "cross_task_correlation.pdf", **SAVEFIG_KW)
    plt.close()

    print(f"  - cross_task_correlation.png")
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
            linewidth=LINEWIDTH_BOLD,
            markersize=MARKERSIZE_LARGE,
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
                linewidth=LINEWIDTH_MED,
                markersize=MARKERSIZE_LARGE,
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
                            arrowstyle="->",
                            color=color,
                            alpha=0.5,
                            lw=LINEWIDTH_THIN,
                        ),
                    )

        ax.set_xticks(x)
        ax.set_xticklabels([task_labels[t] for t in tasks])
        ax.set_ylabel("Error Rate")
        ax.set_title(family_name, fontweight="bold")
        ax.legend(loc="best", fontsize=9)
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
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout(**TIGHT_LAYOUT_KW)
    plt.savefig(
        output_dir / "finetune_effect_by_model.png", **SAVEFIG_KW
    )

    # Remove titles for PDF
    plt.suptitle("")
    for ax in axes:
        ax.set_title("")
    plt.savefig(output_dir / "finetune_effect_by_model.pdf", **SAVEFIG_KW)
    plt.close()

    print(f"  - finetune_effect_by_model.png")

    return None


def plot_cross_dataset_heatmap_4x4(all_results, output_dir):
    """
    Create cross-dataset heatmaps stratified by:
        - Pretraining language (rows)
        - Model architecture + size (columns)

    Produces:
        1) A global heatmap averaged across tasks (unchanged behavior)
        2) Per-task heatmaps (Mandarin / Stress / Pitch Accent),
           each with a 4x4 layout:
               rows    = pretraining language
               columns = hubert-base | hubert-large | wav2vec2-base | wav2vec2-large

    Notes:
        - Finetuned models are excluded
        - No averaging across model sizes
        - Missing combinations are left as NaN
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
    # --------------------------------------------------
    # Create per-task heatmaps side by side with shared y-axis
    # Now: 4x4 (language x {hubert/wav2vec2} x {base/large})
    # No averaging over size; missing combinations stay NaN.
    # --------------------------------------------------
    ARCH_SIZE_ORDER = [
        ("HuBERT", "Base"),
        ("HuBERT", "Large"),
        ("Wav2Vec2", "Base"),
        ("Wav2Vec2", "Large"),
    ]
    ARCH_SIZE_LABELS = [
        "HuBERT-B",
        "HuBERT-L",
        "w2v2-B",
        "w2v2-L",
    ]

    # Human baseline error rates per task
    human_baselines = {
        "mandarin_tone": 0.0188,
        "stress": 0.2888,
        "pitch_accent": 0.1158,
    }

    # Reduced width and minimal space between subplots
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(8.5, 5.0))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 0.05],
                  wspace=0.1, hspace=0.2)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cbar_axes = [fig.add_subplot(gs[1, i]) for i in range(3)]

    # Share y-axis across heatmap panels
    for ax in axes[1:]:
        ax.sharey(axes[0])

    # Language mapping
    lang_map = {
        "Chinese": "ZH",
        "English": "EN",
        "Japanese": "JA",
        "Multilingual": "Many"
    }
    mapped_langs = [lang_map.get(l, l) for l in langs]

    for idx, task in enumerate(tasks):
        ax = axes[idx]

        task_matrix = np.full((len(langs), len(ARCH_SIZE_ORDER)), np.nan)

        for model_name, task_errors in model_task_errors.items():
            if task not in task_errors:
                continue

            meta = MODEL_METADATA[model_name]
            pretrain_lang = meta[0]
            arch = meta[1]
            size = meta[2]   # <-- base / large
            finetuned = meta[4]

            # Skip finetuned (should already be filtered above, but keep safe)
            if finetuned:
                continue

            # Merge XLSR into Wav2Vec2
            if arch == "Wav2Vec2-XLSR":
                arch = "Wav2Vec2"

            # Map mHuBERT 147M to Base
            if size == "147M":
                size = "Base"

            key = (arch, size)
            if key not in ARCH_SIZE_ORDER:
                continue
            if pretrain_lang not in langs:
                continue

            i = langs.index(pretrain_lang)
            j = ARCH_SIZE_ORDER.index(key)

            val = task_errors[task]

            # If multiple models map to the same cell, keep the best one
            if np.isnan(task_matrix[i, j]):
                task_matrix[i, j] = val
            else:
                task_matrix[i, j] = min(task_matrix[i, j], val)

        # Task-specific color range (guard against empty panel)
        task_valid = task_matrix[~np.isnan(task_matrix)]
        if task_valid.size == 0:
            ax.set_title(task_labels.get(task, task), fontsize=14, fontweight="bold")
            ax.set_xticks(range(len(ARCH_SIZE_LABELS)))
            ax.set_xticklabels(ARCH_SIZE_LABELS, rotation=45, ha="right", fontsize=14)
            continue

        baseline = human_baselines.get(task)
        range_min = task_valid.min()
        range_max = task_valid.max()
        if baseline is not None:
            range_min = min(range_min, baseline)
            range_max = max(range_max, baseline)
        task_vmin = range_min
        task_vmax = range_max

        im = ax.imshow(
            task_matrix,
            cmap="viridis",
            vmin=task_vmin,
            vmax=task_vmax,
            aspect="equal",
        )

        ax.set_xticks(range(len(ARCH_SIZE_LABELS)))
        ax.set_xticklabels(ARCH_SIZE_LABELS, rotation=45, ha="right", fontsize=14)

        # Add text annotations with task-specific normalization
        for i in range(len(langs)):
            for j in range(len(ARCH_SIZE_LABELS)):
                if not np.isnan(task_matrix[i, j]):
                    val_norm = (task_matrix[i, j] - task_vmin) / (task_vmax - task_vmin)
                    text_color = "white" if val_norm < 0.6 else "black"
                    ax.text(
                        j,
                        i,
                        f"{task_matrix[i, j] * 100:#.3g}",
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=13,
                        fontweight="bold",
                    )

        ax.set_title(task_labels.get(task, task), fontsize=14, fontweight="bold")

        # Per-panel colorbar with human baseline marker
        from matplotlib.ticker import FuncFormatter
        cbar = fig.colorbar(im, cax=cbar_axes[idx], orientation="horizontal")
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(labelsize=8)
        cbar.ax.xaxis.set_major_formatter(
            FuncFormatter(lambda x, _: f"{x * 100:#.3g}")
        )

        # Mark human baseline on this panel's colorbar
        baseline = human_baselines.get(task)
        if baseline is not None:
            cbar.ax.axvline(baseline, color="red", linewidth=2, linestyle="-", zorder=5)
            cbar.ax.plot(
                baseline, 1.0, marker="v", color="red", markersize=7,
                transform=cbar.ax.get_xaxis_transform(), clip_on=False, zorder=5,
            )

    # Only set y-axis labels on the first panel
    axes[0].set_yticks(range(len(langs)))
    # Use mapped langs and increased font size
    axes[0].set_yticklabels(mapped_langs, fontsize=14, rotation=45, ha="right")
    # axes[0].set_ylabel("Pretraining Language", fontsize=14, labelpad=0)

    # Remove y-tick labels for the other panels
    for ax in axes[1:]:
        ax.tick_params(axis='y', which='both', left=False, right=False,
                       labelleft=False)

    # Legend for the human baseline marker
    from matplotlib.lines import Line2D
    legend_handle = Line2D(
        [0], [0], color="red", linewidth=2, linestyle="-",
        marker="v", markersize=7, label="Human error rate",
    )
    # Place legend below the right colorbar
    cbar_axes[2].legend(
        handles=[legend_handle], loc="upper right",
        fontsize=10, framealpha=0.8,
        bbox_to_anchor=(1.04, -1.5),
    )

    plt.savefig(output_dir / "cross_dataset_heatmap_by_task_4x4.png", **SAVEFIG_KW)
    plt.savefig(output_dir / "cross_dataset_heatmap_by_task_4x4.pdf", **SAVEFIG_KW) # still needs titles
    plt.close()

    print("  - cross_dataset_heatmap_by_task_4x4.png")


    return matrix, langs, archs


def validate_results(all_results):
    """Warn if any model in MODEL_METADATA is missing results for a task."""
    missing = []
    for task, results in all_results.items():
        for model in MODEL_METADATA:
            if model not in results:
                missing.append((task, model))
    if missing:
        lines = [f"  {task}: {model}" for task, model in missing]
        print(
            f"Warning: missing results for {len(missing)} model/task combinations:\n"
            + "\n".join(lines)
        )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading all task results...")
    all_results = load_all_tasks()
    validate_results(all_results)

    print("\nGenerating cross-analysis plots...")

    plot_cross_dataset_heatmap_4x4(all_results, OUTPUT_DIR)

    plot_cross_task_correlation(all_results, OUTPUT_DIR)
    plot_cross_task_correlation_simple(all_results, OUTPUT_DIR)

    plot_finetune_effect_by_model(all_results, OUTPUT_DIR)

    print(f"\nAll plots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
