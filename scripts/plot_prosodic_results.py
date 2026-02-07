#!/usr/bin/env python3
"""
Plot prosodic ABX results across different SSL models.
Stratifies by: pretraining language, model architecture, size, corpus size, and finetuning.

Usage:
    python plot_prosodic_results.py [dataset]

Where dataset is one of: pitch_accent, stress, mandarin_tone
Default is pitch_accent if not specified.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

DATASET_LABELS = {
    "pitch_accent": "English Pitch Accent",
    "stress": "Japanese Lexical Stress",
    "mandarin_tone": "Mandarin Tone",
}


# Model metadata: (pretrain_lang, architecture, size, corpus_size, finetuned, finetune_lang)
MODEL_METADATA = {
    # English pretrained
    "wavlm_base": ("English", "WavLM", "Base", "Small", False, None),
    "wavlm_base_plus": ("English", "WavLM", "Base", "Large", False, None),
    "wavlm_large": ("English", "WavLM", "Large", "Large", False, None),
    "hubert_base": ("English", "HuBERT", "Base", "Small", False, None),
    "hubert_large": ("English", "HuBERT", "Large", "Large", False, None),
    "wav2vec2_base": ("English", "Wav2Vec2", "Base", "Small", False, None),
    "wav2vec2_large": ("English", "Wav2Vec2", "Large", "Small", False, None),
#    "wav2vec2_large_lv60k": ("English", "Wav2Vec2", "Large", "Large", False, None),
    "hubert_asr_large": ("English", "HuBERT", "Large", "Large", True, "English"),
    "wav2vec2_asr_large_960h": (
        "English",
        "Wav2Vec2",
        "Large",
        "Small",
        True,
        "English",
    ),
    # Multilingual pretrained
    "wav2vec2-large-xlsr-53": (
        "Multilingual",
        "Wav2Vec2-XLSR",
        "Large",
        "Large",
        False,
        None,
    ),
    "mhubert-147": (
        "Multilingual",
        "HuBERT",
        "147M",
        "Large",
        False,
        None,
    ),
    # 'w2v-bert-2.0' excluded - isolated model with different architecture
    '''"wav2vec2-large-xlsr-53-english": (
        "Multilingual",
        "Wav2Vec2-XLSR",
        "Large",
        "Large",
        True,
        "English",
    ),
    "wav2vec2-large-xlsr-53-japanese": (
        "Multilingual",
        "Wav2Vec2-XLSR",
        "Large",
        "Large",
        True,
        "Japanese",
    ),
    "wav2vec2-large-xlsr-53-chinese-zh-cn": (
        "Multilingual",
        "Wav2Vec2-XLSR",
        "Large",
        "Large",
        True,
        "Chinese",
    ),
    '''
    # Chinese pretrained
    "chinese-wav2vec2-base": ("Chinese", "Wav2Vec2", "Base", "Large", False, None),
    "chinese-wav2vec2-large": ("Chinese", "Wav2Vec2", "Large", "Large", False, None),
    "chinese-hubert-base": ("Chinese", "HuBERT", "Base", "Large", False, None),
    "chinese-hubert-large": ("Chinese", "HuBERT", "Large", "Large", False, None),
    # Japanese pretrained
    "japanese-wav2vec2-base": ("Japanese", "Wav2Vec2", "Base", "Large", False, None),
    "japanese-wav2vec2-large": ("Japanese", "Wav2Vec2", "Large", "Large", False, None),
    "japanese-hubert-base-k2": ("Japanese", "HuBERT", "Base", "Large", False, None),
    "japanese-hubert-large": ("Japanese", "HuBERT", "Large", "Large", False, None),
    '''
    "japanese-wav2vec2-base-rs35kh": (
        "Japanese",
        "Wav2Vec2",
        "Base",
        "Large",
        True,
        "Japanese",
    ),
    "japanese-wav2vec2-large-rs35kh": (
        "Japanese",
        "Wav2Vec2",
        "Large",
        "Large",
        True,
        "Japanese",
    ),
    "japanese-hubert-base-k2-rs35kh": (
        "Japanese",
        "HuBERT",
        "Base",
        "Large",
        True,
        "Japanese",
    ),
    '''
    # Baselines
    "mfcc": ("Baseline", "MFCC", "N/A", "N/A", False, None),
    "fbank": ("Baseline", "FBank", "N/A", "N/A", False, None),
}


def load_results(results_dir, *, filter_metadata=True):
    """Load all CSV results from the directory."""
    results = {}
    results_path = Path(results_dir)

    for csv_file in results_path.glob("*.csv"):
        model_name = csv_file.stem
        df = pd.read_csv(csv_file)
        results[model_name] = df

    if filter_metadata:
        missing = [m for m in results.keys() if m not in MODEL_METADATA]
        if missing:
            print(
                "Warning: models missing metadata, skipping: "
                + ", ".join(sorted(missing))
            )
        results = {m: df for m, df in results.items() if m in MODEL_METADATA}

    return results


def get_layer_num(layer_str):
    """Extract layer number from layer string (e.g., 'l1' -> 1)."""
    if layer_str == "features":
        return 0
    return int(layer_str[1:])


def plot_layerwise_by_pretrain_lang(results, output_dir, dataset_label="Pitch Accent"):
    """Plot layerwise error rates grouped by pretraining language (depth-normalized)."""
    # Exclude Baseline since they only have one point
    lang_groups = defaultdict(list)
    for model_name, df in results.items():
        if model_name in MODEL_METADATA:
            lang = MODEL_METADATA[model_name][0]
            if lang != "Baseline":  # Skip baselines - only one data point
                lang_groups[lang].append((model_name, df))

    n_groups = len(lang_groups)
    cols = min(3, n_groups)
    rows = (n_groups + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if n_groups == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Collect all error values to set consistent y-axis
    all_errors = []
    for models in lang_groups.values():
        for _, df in models:
            all_errors.extend(df["error_rate"].values)
    y_min = max(0, min(all_errors) - 0.05)
    y_max = max(all_errors) + 0.05

    for idx, (lang, models) in enumerate(sorted(lang_groups.items())):
        if idx >= len(axes):
            break
        ax = axes[idx]

        for i, (model_name, df) in enumerate(models):
            layers = [get_layer_num(l) for l in df["layer"]]
            max_layer = max(layers)
            normalized_layers = [l / max_layer for l in layers]
            errors = df["error_rate"].values
            ax.plot(
                normalized_layers,
                errors,
                marker="o",
                markersize=3,
                label=model_name,
                alpha=0.7,
            )

        ax.set_xlabel("Normalized Layer Depth")
        ax.set_ylabel("Error Rate")
        ax.set_title(f"{lang} Pretrained")
        ax.legend(fontsize=6, loc="upper right")
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for idx in range(len(lang_groups), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / "layerwise_by_pretrain_lang.png", dpi=150)
    plt.close()


def plot_layerwise_by_architecture(results, output_dir, dataset_label="Pitch Accent"):
    """Plot layerwise error rates grouped by model architecture (depth-normalized)."""
    # Exclude MFCC/FBank since they only have one point
    arch_groups = defaultdict(list)
    for model_name, df in results.items():
        if model_name in MODEL_METADATA:
            arch = MODEL_METADATA[model_name][1]
            if arch not in ["MFCC", "FBank"]:  # Skip baselines
                arch_groups[arch].append((model_name, df))

    n_groups = len(arch_groups)
    cols = min(3, n_groups)
    rows = (n_groups + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if n_groups == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Collect all error values to set consistent y-axis
    all_errors = []
    for models in arch_groups.values():
        for _, df in models:
            all_errors.extend(df["error_rate"].values)
    y_min = max(0, min(all_errors) - 0.05)
    y_max = max(all_errors) + 0.05

    for idx, (arch, models) in enumerate(sorted(arch_groups.items())):
        if idx >= len(axes):
            break
        ax = axes[idx]

        for model_name, df in models:
            layers = [get_layer_num(l) for l in df["layer"]]
            max_layer = max(layers)
            normalized_layers = [l / max_layer for l in layers]
            errors = df["error_rate"].values
            ax.plot(
                normalized_layers,
                errors,
                marker="o",
                markersize=3,
                label=model_name,
                alpha=0.7,
            )

        ax.set_xlabel("Normalized Layer Depth")
        ax.set_ylabel("Error Rate")
        ax.set_title(f"{arch}")
        ax.legend(fontsize=6, loc="upper right")
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)

    for idx in range(len(arch_groups), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / "layerwise_by_architecture.png", dpi=150)
    plt.close()


def plot_best_layer_comparison(results, output_dir, dataset_label="Pitch Accent"):
    """Bar plot comparing best layer error rate across models."""
    best_results = []

    for model_name, df in results.items():
        if model_name not in MODEL_METADATA:
            continue

        best_idx = df["error_rate"].idxmin()
        best_error = df.loc[best_idx, "error_rate"]
        best_layer = df.loc[best_idx, "layer"]
        meta = MODEL_METADATA[model_name]

        best_results.append(
            {
                "model": model_name,
                "best_error": best_error,
                "best_layer": best_layer,
                "pretrain_lang": meta[0],
                "architecture": meta[1],
                "size": meta[2],
                "corpus_size": meta[3],
                "finetuned": meta[4],
            }
        )

    df_best = pd.DataFrame(best_results)
    df_best = df_best.sort_values("best_error")

    # Color by pretraining language
    lang_colors = {
        "English": "tab:blue",
        "Multilingual": "tab:green",
        "Chinese": "tab:red",
        "Japanese": "tab:purple",
        "Baseline": "tab:gray",
    }

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = [lang_colors[lang] for lang in df_best["pretrain_lang"]]
    bars = ax.barh(range(len(df_best)), df_best["best_error"], color=colors, alpha=0.7)
    ax.set_yticks(range(len(df_best)))
    ax.set_yticklabels(df_best["model"], fontsize=8)
    ax.set_xlabel("Best Layer Error Rate")
    ax.set_title(f"{dataset_label} ABX: Best Layer Error Rate by Model")

    # Add legend for languages
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=color, label=lang, alpha=0.7)
        for lang, color in lang_colors.items()
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "best_layer_comparison.png", dpi=150)
    plt.close()

    return df_best


def plot_finetuned_vs_pretrained(results, output_dir, dataset_label="Pitch Accent"):
    """Compare finetuned vs pretrained-only models (only models with paired variants)."""
    from matplotlib.patches import Patch

    finetune_lang_colors = {
        "English": "tab:blue",
        "Japanese": "tab:purple",
        "Chinese": "tab:red",
        None: "tab:gray",
    }

    # Define pairs first - only include models that have both pretrained and finetuned versions
    pairs = [
        ("hubert_large", "hubert_asr_large"),
        ("wav2vec2_large", "wav2vec2_asr_large_960h"),
        ("wav2vec2-large-xlsr-53", "wav2vec2-large-xlsr-53-english"),
        ("wav2vec2-large-xlsr-53", "wav2vec2-large-xlsr-53-japanese"),
        ("wav2vec2-large-xlsr-53", "wav2vec2-large-xlsr-53-chinese-zh-cn"),
        ("japanese-wav2vec2-base", "japanese-wav2vec2-base-rs35kh"),
        ("japanese-wav2vec2-large", "japanese-wav2vec2-large-rs35kh"),
        ("japanese-hubert-base-k2", "japanese-hubert-base-k2-rs35kh"),
    ]

    # Only include models that have a pair in the dataset
    paired_models = set()
    valid_pairs = []
    for pt, ft in pairs:
        if pt in results and ft in results:
            paired_models.add(pt)
            paired_models.add(ft)
            valid_pairs.append((pt, ft))

    pretrained_only = []
    finetuned = []

    for model_name, df in results.items():
        if model_name not in MODEL_METADATA:
            continue
        if model_name not in paired_models:
            continue
        meta = MODEL_METADATA[model_name]

        best_error = df["error_rate"].min()

        if meta[4]:  # finetuned
            finetuned.append((model_name, best_error, meta))
        else:
            pretrained_only.append((model_name, best_error, meta))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot by finetuning status
    ax = axes[0]
    pretrained_errors = [x[1] for x in pretrained_only]
    finetuned_errors = [x[1] for x in finetuned]

    positions = [1, 2]
    bp = ax.boxplot(
        [pretrained_errors, finetuned_errors], positions=positions, widths=0.6
    )
    ax.set_xticklabels(["Pretrained Only", "ASR Finetuned"])
    ax.set_ylabel("Best Layer Error Rate")
    ax.set_title(f"Effect of ASR Finetuning on {dataset_label} Discrimination")
    ax.grid(True, axis="y", alpha=0.3)

    # Scatter individual points - color by finetune language for finetuned models
    # Pretrained points
    x_jitter = np.random.normal(1, 0.1, len(pretrained_only))
    y_vals = [d[1] for d in pretrained_only]
    ax.scatter(x_jitter, y_vals, alpha=0.6, s=40, zorder=3, color="tab:gray")

    # Finetuned points - color by finetune language
    for model_name, best_error, meta in finetuned:
        ft_lang = meta[5]  # finetune_lang
        color = finetune_lang_colors.get(ft_lang, "tab:gray")
        x_jitter = np.random.normal(2, 0.1, 1)
        ax.scatter(x_jitter, [best_error], alpha=0.7, s=50, zorder=3, color=color)

    # Add legend for finetune languages
    legend_elements = [
        Patch(facecolor=color, label=f"FT: {lang}", alpha=0.7)
        for lang, color in finetune_lang_colors.items()
        if lang is not None
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    # Plot paired comparisons for models with both versions
    ax = axes[1]
    pair_labels = []
    pretrained_vals = []
    finetuned_vals = []
    ft_colors = []

    for pt, ft in valid_pairs:
        ft_meta = MODEL_METADATA[ft]
        ft_lang = ft_meta[5]
        pair_labels.append(f"{pt}\n→ FT:{ft_lang}")
        pretrained_vals.append(results[pt]["error_rate"].min())
        finetuned_vals.append(results[ft]["error_rate"].min())
        ft_colors.append(finetune_lang_colors.get(ft_lang, "tab:gray"))

    x = np.arange(len(pair_labels))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2,
        pretrained_vals,
        width,
        label="Pretrained",
        alpha=0.7,
        color="tab:gray",
    )
    bars2 = ax.bar(x + width / 2, finetuned_vals, width, alpha=0.7, color=ft_colors)

    ax.set_ylabel("Best Layer Error Rate")
    ax.set_title("Paired Comparison: Pretrained vs ASR Finetuned (colored by FT lang)")
    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels, fontsize=7, rotation=45, ha="right")

    # Custom legend
    legend_elements = [Patch(facecolor="tab:gray", label="Pretrained", alpha=0.7)]
    legend_elements += [
        Patch(facecolor=color, label=f"FT: {lang}", alpha=0.7)
        for lang, color in finetune_lang_colors.items()
        if lang is not None
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "finetuned_vs_pretrained.png", dpi=150)
    plt.close()


def plot_size_comparison(results, output_dir, dataset_label="Pitch Accent"):
    """Compare Large vs Base models (only models with paired Base/Large variants)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Define pairs first - only include models that have both Base and Large variants
    pairs = [
        ("wavlm_base", "wavlm_large"),
        ("hubert_base", "hubert_large"),
        ("wav2vec2_base", "wav2vec2_large"),
        ("chinese-wav2vec2-base", "chinese-wav2vec2-large"),
        ("chinese-hubert-base", "chinese-hubert-large"),
        ("japanese-wav2vec2-base", "japanese-wav2vec2-large"),
        ("japanese-hubert-base-k2", "japanese-hubert-large"),
    ]

    # Only include models that have a pair in the dataset
    paired_models = set()
    valid_pairs = []
    for b, l in pairs:
        if b in results and l in results:
            paired_models.add(b)
            paired_models.add(l)
            valid_pairs.append((b, l))

    # Collect by size - only paired models
    size_data = defaultdict(list)
    for model_name, df in results.items():
        if model_name not in MODEL_METADATA:
            continue
        if model_name not in paired_models:
            continue
        meta = MODEL_METADATA[model_name]
        if meta[2] in ["Large", "Base"]:
            size_data[meta[2]].append((model_name, df["error_rate"].min(), meta))

    ax = axes[0]
    base_errors = [x[1] for x in size_data["Base"]]
    large_errors = [x[1] for x in size_data["Large"]]

    positions = [1, 2]
    ax.boxplot([base_errors, large_errors], positions=positions, widths=0.6)
    ax.set_xticklabels(["Base", "Large"])
    ax.set_ylabel("Best Layer Error Rate")
    ax.set_title(f"Effect of Model Size on {dataset_label} Discrimination")
    ax.grid(True, axis="y", alpha=0.3)

    for i, (x_pos, data) in enumerate(
        zip(positions, [size_data["Base"], size_data["Large"]])
    ):
        x_jitter = np.random.normal(x_pos, 0.1, len(data))
        y_vals = [d[1] for d in data]
        ax.scatter(x_jitter, y_vals, alpha=0.6, s=40, zorder=3)

    # Paired comparisons
    ax = axes[1]
    pair_labels = []
    base_vals = []
    large_vals = []

    for b, l in valid_pairs:
        short_name = b.replace("-base", "").replace("_base", "").replace("-k2", "")
        pair_labels.append(short_name)
        base_vals.append(results[b]["error_rate"].min())
        large_vals.append(results[l]["error_rate"].min())

    x = np.arange(len(pair_labels))
    width = 0.35

    bars1 = ax.bar(x - width / 2, base_vals, width, label="Base", alpha=0.7)
    bars2 = ax.bar(x + width / 2, large_vals, width, label="Large", alpha=0.7)

    ax.set_ylabel("Best Layer Error Rate")
    ax.set_title("Paired Comparison: Base vs Large")
    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels, fontsize=8, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "size_comparison.png", dpi=150)
    plt.close()


def plot_heatmap_best_layer(results, output_dir, dataset_label="Pitch Accent"):
    """Create a heatmap showing best layer error rate by pretrain lang and architecture."""
    # Aggregate by (pretrain_lang, architecture) - take mean of best errors
    agg_data = defaultdict(list)

    for model_name, df in results.items():
        if model_name not in MODEL_METADATA:
            continue
        meta = MODEL_METADATA[model_name]
        if meta[0] == "Baseline":
            continue
        key = (meta[0], meta[1])
        agg_data[key].append(df["error_rate"].min())

    # Get unique languages and architectures
    langs = sorted(set(k[0] for k in agg_data.keys()))
    archs = sorted(set(k[1] for k in agg_data.keys()))

    # Create matrix
    matrix = np.full((len(langs), len(archs)), np.nan)
    for (lang, arch), errors in agg_data.items():
        i = langs.index(lang)
        j = archs.index(arch)
        matrix[i, j] = np.mean(errors)

    # Dynamic color range based on actual data
    valid_vals = matrix[~np.isnan(matrix)]
    vmin = valid_vals.min() - 0.02
    vmax = valid_vals.max() + 0.02

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(matrix, cmap="RdYlGn_r", vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(archs)))
    ax.set_xticklabels(archs, rotation=45, ha="right")
    ax.set_yticks(range(len(langs)))
    ax.set_yticklabels(langs)

    # Add text annotations with dynamic text color based on background
    for i in range(len(langs)):
        for j in range(len(archs)):
            if not np.isnan(matrix[i, j]):
                # Use white text on dark backgrounds, black on light
                val_norm = (matrix[i, j] - vmin) / (vmax - vmin)
                text_color = "white" if val_norm > 0.6 else "black"
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:.3f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=10,
                    fontweight="bold",
                )

    ax.set_title("Mean Best Layer Error Rate by Pretrain Language & Architecture")
    fig.colorbar(im, ax=ax, label="Error Rate")

    plt.tight_layout()
    plt.savefig(output_dir / "heatmap_lang_arch.png", dpi=150)
    plt.close()


def plot_paired_layerwise_by_finetune_lang(results, output_dir, dataset_label="Pitch Accent"):
    """
    Plot layerwise error rates comparing Pretrained vs Finetuned models, grouped by FT language.
    Plots lines for both PT (dashed) and FT (solid) versions.
    """
    # Define pairs (PT, FT) - same as in plot_finetuned_vs_pretrained
    pairs = [
        ("hubert_large", "hubert_asr_large"),
        ("wav2vec2_large", "wav2vec2_asr_large_960h"),
        ("wav2vec2-large-xlsr-53", "wav2vec2-large-xlsr-53-english"),
        ("wav2vec2-large-xlsr-53", "wav2vec2-large-xlsr-53-japanese"),
        ("wav2vec2-large-xlsr-53", "wav2vec2-large-xlsr-53-chinese-zh-cn"),
        ("japanese-wav2vec2-base", "japanese-wav2vec2-base-rs35kh"),
        ("japanese-wav2vec2-large", "japanese-wav2vec2-large-rs35kh"),
        ("japanese-hubert-base-k2", "japanese-hubert-base-k2-rs35kh"),
    ]

    # Filter pairs that exist in results
    valid_pairs = []
    for pt, ft in pairs:
        if pt in results and ft in results:
            valid_pairs.append((pt, ft))

    if not valid_pairs:
        return

    # Group by Finetuning Language
    ft_groups = defaultdict(list)
    for pt, ft in valid_pairs:
        ft_meta = MODEL_METADATA[ft]
        ft_lang = ft_meta[5]  # finetune_lang
        ft_groups[ft_lang].append((pt, ft))

    # Assign colors to unique PT models to keep consistency
    unique_pt_models = sorted(list(set(p[0] for p in valid_pairs)))
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    color_cycle = prop_cycle.by_key()["color"]
    pt_color_map = {
        pt: color_cycle[i % len(color_cycle)] for i, pt in enumerate(unique_pt_models)
    }

    # Setup plots
    n_groups = len(ft_groups)
    cols = min(3, n_groups)
    rows = (n_groups + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))

    if n_groups == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Determine common y-axis limits
    all_errors = []
    for pt, ft in valid_pairs:
        all_errors.extend(results[pt]["error_rate"].values)
        all_errors.extend(results[ft]["error_rate"].values)

    y_min = max(0, min(all_errors) - 0.05) if all_errors else 0
    y_max = (max(all_errors) + 0.05) if all_errors else 1

    for idx, (ft_lang, pairs_list) in enumerate(sorted(ft_groups.items())):
        ax = axes[idx]

        for pt, ft in pairs_list:
            color = pt_color_map[pt]

            # Plot Pretrained
            df_pt = results[pt]
            layers_pt = [get_layer_num(l) for l in df_pt["layer"]]
            max_pt = max(layers_pt)
            norm_pt = [l / max_pt for l in layers_pt]

            ax.plot(
                norm_pt,
                df_pt["error_rate"],
                linestyle="--",
                marker="o",
                markersize=3,
                color=color,
                alpha=0.6,
                label=f"{pt} (PT)",
            )

            # Plot Finetuned
            df_ft = results[ft]
            layers_ft = [get_layer_num(l) for l in df_ft["layer"]]
            max_ft = max(layers_ft)
            norm_ft = [l / max_ft for l in layers_ft]

            ax.plot(
                norm_ft,
                df_ft["error_rate"],
                linestyle="-",
                marker="s",
                markersize=3,
                color=color,
                linewidth=2,
                label=f"{ft} (FT)",
            )

        ax.set_title(f"Finetuned on {ft_lang}")
        ax.set_xlabel("Normalized Layer Depth")
        ax.set_ylabel("Error Rate")
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6, loc="upper right")

    # Hide unused
    for idx in range(len(ft_groups), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / "paired_layerwise_finetuning.png", dpi=150)
    plt.close()


def print_summary_table(df_best, dataset="pitch_accent"):
    """Print a summary table of results."""
    label = DATASET_LABELS.get(dataset, dataset.upper())
    print("\n" + "=" * 80)
    print(f"{label.upper()} ABX RESULTS SUMMARY")
    print("=" * 80)

    print("\nTop 10 Models (Lowest Error Rate):")
    print("-" * 60)
    for _, row in df_best.head(10).iterrows():
        print(f"{row['model']:40s} {row['best_error']:.4f} (layer {row['best_layer']})")

    print("\n\nBy Pretraining Language (Mean Best Error):")
    print("-" * 40)
    for lang, group in df_best.groupby("pretrain_lang"):
        print(f"{lang:20s} {group['best_error'].mean():.4f} (n={len(group)})")

    print("\n\nBy Architecture (Mean Best Error):")
    print("-" * 40)
    for arch, group in df_best.groupby("architecture"):
        print(f"{arch:20s} {group['best_error'].mean():.4f} (n={len(group)})")

    print("\n\nBy Finetuning Status (Mean Best Error):")
    print("-" * 40)
    ft_group = df_best[df_best["pretrain_lang"] != "Baseline"]
    for ft, group in ft_group.groupby("finetuned"):
        status = "Finetuned" if ft else "Pretrained Only"
        print(f"{status:20s} {group['best_error'].mean():.4f} (n={len(group)})")

    print("\n\nBy Model Size (Mean Best Error):")
    print("-" * 40)
    for size, group in df_best.groupby("size"):
        if size in ["Base", "Large"]:
            print(f"{size:20s} {group['best_error'].mean():.4f} (n={len(group)})")


def main():
    parser = argparse.ArgumentParser(
        description="Plot prosodic ABX results across different SSL models."
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default="pitch_accent",
        choices=["pitch_accent", "stress", "mandarin_tone"],
        help="Dataset to analyze (default: pitch_accent)",
    )
    args = parser.parse_args()

    dataset = args.dataset
    dataset_label = DATASET_LABELS.get(dataset, dataset)

    results_dir = Path(__file__).parent.parent / "results" / dataset
    output_dir = Path(__file__).parent.parent / "plots" / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Analyzing: {dataset_label}")
    print(f"Loading results from {results_dir}")
    results = load_results(results_dir)
    print(f"Loaded {len(results)} model results")

    # Check for missing models
    missing = set(MODEL_METADATA.keys()) - set(results.keys())
    if missing:
        print(f"\nNote: {len(missing)} models in metadata not found in results")

    print("\nGenerating plots...")

    plot_layerwise_by_pretrain_lang(results, output_dir, dataset_label)
    print("  - layerwise_by_pretrain_lang.png")

    plot_layerwise_by_architecture(results, output_dir, dataset_label)
    print("  - layerwise_by_architecture.png")

    df_best = plot_best_layer_comparison(results, output_dir, dataset_label)
    print("  - best_layer_comparison.png")

    plot_finetuned_vs_pretrained(results, output_dir, dataset_label)
    print("  - finetuned_vs_pretrained.png")

    plot_size_comparison(results, output_dir, dataset_label)
    print("  - size_comparison.png")

    plot_heatmap_best_layer(results, output_dir, dataset_label)
    print("  - heatmap_lang_arch.png")

    plot_paired_layerwise_by_finetune_lang(results, output_dir, dataset_label)
    print("  - paired_layerwise_finetuning.png")

    print(f"\nAll plots saved to {output_dir}")

    print_summary_table(df_best, dataset)


if __name__ == "__main__":
    main()
