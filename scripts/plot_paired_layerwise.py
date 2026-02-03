#!/usr/bin/env python3
"""
Paired layerwise analysis between model performance on two or three datasets.

Usage:
    python plot_paired_layerwise.py dataset1 dataset2 [dataset3] [--normalize]
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from plot_prosodic_results import MODEL_METADATA, load_results, get_layer_num

RESULTS_DIR = Path(__file__).parent.parent / "results"
OUTPUT_DIR = Path(__file__).parent.parent / "plots" / "paired_layerwise"

MODEL_FAMILIES = {
    "HuBERT": [
        "hubert_base", "hubert_large", "hubert_asr_large",
        "chinese-hubert-base", "chinese-hubert-large",
        "japanese-hubert-base-k2", "japanese-hubert-large", "japanese-hubert-base-k2-rs35kh",
        "mhubert-147",
    ],
    "Wav2Vec2": [
        "wav2vec2_base", "wav2vec2_large", "wav2vec2_large_lv60k", "wav2vec2_asr_large_960h",
        "chinese-wav2vec2-base", "chinese-wav2vec2-large",
        "japanese-wav2vec2-base", "japanese-wav2vec2-large",
        "japanese-wav2vec2-base-rs35kh", "japanese-wav2vec2-large-rs35kh",
        "wav2vec2-large-xlsr-53", "wav2vec2-large-xlsr-53-english",
        "wav2vec2-large-xlsr-53-japanese", "wav2vec2-large-xlsr-53-chinese-zh-cn",
    ],
    "WavLM": ["wavlm_base", "wavlm_base_plus", "wavlm_large"],
}

PRETRAIN_LANGUAGES = ["English", "Chinese", "Japanese", "Multilingual"]
LINE_STYLES = ["-", "--", ":"]
MARKERS = ["o", "s", "^"]


def get_pretrain_lang(model_name):
    if model_name not in MODEL_METADATA:
        return None
    return MODEL_METADATA[model_name][0]


def is_finetuned(model_name):
    if model_name not in MODEL_METADATA:
        return False
    return MODEL_METADATA[model_name][4]


def plot_family(family_name, models, results_list, datasets, output_dir, finetuned_only, normalize):
    filtered = [m for m in models
                if all(m in r for r in results_list) and is_finetuned(m) == finetuned_only]
    if not filtered:
        return None

    n_models = len(filtered)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
    axes = axes.flatten()

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Global y-axis range
    if not normalize:
        global_errors = [e for m in filtered for r in results_list for e in r[m]["error_rate"].values]
    else:
        # Compute max shifted error (after subtracting min per model/dataset)
        max_shifted = 0
        for m in filtered:
            for r in results_list:
                errors = r[m]["error_rate"].values
                shifted = errors - errors.min()
                max_shifted = max(max_shifted, shifted.max())

    for ax_idx, model in enumerate(filtered):
        ax = axes[ax_idx]
        # Collect error rates from all datasets for correlation
        all_errors = []
        for ds_idx, results in enumerate(results_list):
            df = results[model]
            layers = [get_layer_num(l) for l in df["layer"]]
            max_layer = max(layers) if layers else 1
            norm_layers = [l / max_layer for l in layers]
            errors = df["error_rate"].values.copy()
            all_errors.append(errors)

            if normalize:
                errors = errors - errors.min()

            ax.plot(norm_layers, errors, linestyle=LINE_STYLES[ds_idx], marker=MARKERS[ds_idx],
                    markersize=4, color=colors[ds_idx % len(colors)], linewidth=2, alpha=0.8,
                    label=datasets[ds_idx])

        # Calculate correlations between datasets
        corrs = []
        for i in range(len(all_errors) - 1):
            r = np.corrcoef(all_errors[i], all_errors[i + 1])[0, 1]
            corrs.append(f"{r:.2f}")
        corr_str = ", ".join(corrs)

        ax.set_xlim(0, 1)
        if normalize:
            ax.set_ylim(-0.01, max_shifted + 0.01)
        else:
            ax.set_ylim(max(0, min(global_errors) - 0.02), max(global_errors) + 0.02)
        ax.set_xlabel("Normalized Layer Depth")
        ax.set_ylabel("Normalized Error Rate" if normalize else "Error Rate")
        ax.set_title(f"{model} (r={corr_str})")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")

    for ax_idx in range(n_models, len(axes)):
        axes[ax_idx].set_visible(False)

    status = "finetuned" if finetuned_only else "pretrained"
    norm_suffix = "_normalized" if normalize else ""
    fig.suptitle(f"{family_name} ({status})", fontsize=12)
    plt.tight_layout()

    filename = f"{family_name.lower()}{norm_suffix}.png"
    plt.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
    plt.close()
    return filename


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+", help="2-3 datasets to compare")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    if not 2 <= len(args.datasets) <= 3:
        print("Error: Provide 2 or 3 datasets")
        return

    for ds in args.datasets:
        if not (RESULTS_DIR / ds).exists():
            print(f"Error: {ds} not found")
            return

    results_list = [load_results(RESULTS_DIR / ds) for ds in args.datasets]

    base_dir = args.output_dir or OUTPUT_DIR
    subdir = "_vs_".join(args.datasets)
    pretrained_dir = base_dir / subdir / "pretrained"
    finetuned_dir = base_dir / subdir / "finetuned"
    pretrained_dir.mkdir(parents=True, exist_ok=True)
    finetuned_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating plots for: {' vs '.join(args.datasets)}")

    for family, models in MODEL_FAMILIES.items():
        for finetuned, out_dir in [(False, pretrained_dir), (True, finetuned_dir)]:
            filename = plot_family(family, models, results_list, args.datasets, out_dir, finetuned, args.normalize)
            if filename:
                print(f"  {out_dir.name}/{filename}")

    print("Done!")


if __name__ == "__main__":
    main()
