"""Analyze why English TTS fails as a proxy for natural speech in prosodic ABX.

This script performs three analyses:
1. Triple-level correlation: Per-word correlation between natural and synthesized
2. Per-pair breakdown: Identify which stress pairs show largest natural-synth discrepancy
3. Model-specific sensitivity: Check if certain models are more robust to the gap

Usage:
    python scripts/analyze_english_tts_proxy.py
"""

from pathlib import Path

import polars as pl
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from adjustText import adjust_text

from plot_prosodic_results import MODEL_METADATA

RESULTS_DIR = Path("results")

# Plot styling (matching plot_word_error_rate.py)
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
DATASETS = {
    "natural": "stress",
    "gtts": "stress_syn",
    "kokoro": "stress_kokoro",
}
DISPLAY_NAMES = {
    "gtts": "G-TTS",
    "kokoro": "Kokoro",
}
SYNTH_NAMES = [k for k in DATASETS if k != "natural"]


def load_cell_scores(dataset: str, model: str, layer: str) -> pl.DataFrame:
    """Load cell-level scores for a given dataset/model/layer."""
    path = RESULTS_DIR / dataset / "cells" / model / f"{layer}.csv"
    return pl.read_csv(path)


def aggregate_to_per_word(df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate cell-level scores to per-word error rates."""
    return df.group_by("phone_sequence").agg(
        pl.col("score").mean().alias("error_rate"),
        pl.col("size").sum().alias("n_comparisons"),
    )


def get_best_layer(dataset: str, model: str) -> str:
    """Find the best layer (lowest error rate) for a model on a dataset."""
    model_dir = RESULTS_DIR / dataset / "cells" / model
    layer_files = list(model_dir.glob("l*.csv"))

    # Handle non-layered models (e.g., fbank, mfcc)
    if not layer_files:
        features_file = model_dir / "features.csv"
        if features_file.exists():
            return "features"
        raise FileNotFoundError(f"No cell files found in {model_dir}")

    best_layer, best_error = None, float("inf")
    for lf in layer_files:
        df = pl.read_csv(lf)
        error = df["score"].mean()
        if error < best_error:
            best_error = error
            best_layer = lf.stem
    return best_layer


def get_common_models() -> list[str]:
    """Get models available in all three datasets."""
    models_per_dataset = []
    for dataset in DATASETS.values():
        cells_dir = RESULTS_DIR / dataset / "cells"
        models_per_dataset.append(set(p.name for p in cells_dir.iterdir() if p.is_dir()))
    common = sorted(set.intersection(*models_per_dataset))
    filtered = [m for m in common if m in MODEL_METADATA]
    missing = [m for m in common if m not in MODEL_METADATA]
    if missing:
        print(
            "Warning: models missing metadata, skipping: "
            + ", ".join(sorted(missing))
        )
    return filtered


def analysis_1_per_word_correlation(models: list[str]) -> pl.DataFrame:
    """Compute per-word correlation between natural and synthesized speech."""
    print("\n" + "=" * 60)
    print("ANALYSIS 1: Per-word correlation")
    print("=" * 60)

    results = []

    for model in models:
        # Get best layer for each condition
        layers = {name: get_best_layer(ds, model) for name, ds in DATASETS.items()}

        # Load and aggregate to per-word
        per_word = {}
        for name, ds in DATASETS.items():
            df = load_cell_scores(ds, model, layers[name])
            per_word[name] = aggregate_to_per_word(df)

        # Join natural with each synth condition
        for synth_name in SYNTH_NAMES:
            joined = per_word["natural"].join(
                per_word[synth_name],
                on="phone_sequence",
                suffix="_synth",
            )

            nat = joined["error_rate"].to_numpy()
            syn = joined["error_rate_synth"].to_numpy()

            r, p = stats.pearsonr(nat, syn)
            rho, p_rho = stats.spearmanr(nat, syn)

            results.append({
                "model": model,
                "synth": synth_name,
                "pearson_r": r,
                "spearman_rho": rho,
                "n_words": len(joined),
            })

    results_df = pl.DataFrame(results)

    # Summary statistics (SSL models only)
    ssl_df = results_df.filter(~pl.col("model").is_in(["mfcc", "fbank"]))
    for synth in SYNTH_NAMES:
        subset = ssl_df.filter(pl.col("synth") == synth)
        print(f"\n{DISPLAY_NAMES[synth]} vs Natural (SSL models only, n={len(subset)}):")
        print(f"  Pearson r:   median={subset['pearson_r'].median():.3f}, "
              f"mean={subset['pearson_r'].mean():.3f}, "
              f"std={subset['pearson_r'].std():.3f}")
        print(f"  Spearman ρ:  median={subset['spearman_rho'].median():.3f}, "
              f"mean={subset['spearman_rho'].mean():.3f}, "
              f"std={subset['spearman_rho'].std():.3f}")

    # Baseline statistics
    print("\nBaselines:")
    baseline_models = ["mfcc", "fbank"]
    baselines_present = (
        results_df.filter(pl.col("model").is_in(baseline_models))["model"].unique()
    )
    if baselines_present.is_empty():
        print("  (no baselines present after metadata filtering)")
    else:
        for model in baseline_models:
            for synth in SYNTH_NAMES:
                row_df = results_df.filter(
                    (pl.col("model") == model) & (pl.col("synth") == synth)
                )
                if row_df.is_empty():
                    print(f"  (missing {model.upper()} vs {DISPLAY_NAMES[synth]})")
                    continue
                row = row_df.row(0, named=True)
                print(f"  {model.upper()} vs {DISPLAY_NAMES[synth]}: "
                      f"r={row['pearson_r']:.3f}, ρ={row['spearman_rho']:.3f}")

    return results_df


def analysis_2_per_pair_breakdown(models: list[str]) -> pl.DataFrame:
    """Identify which stress pairs show the largest natural-synth discrepancy."""
    print("\n" + "=" * 60)
    print("ANALYSIS 2: Per-word error breakdown (SSL models only)")
    print("=" * 60)

    # Aggregate across SSL models only (using best layer for each)
    ssl_models = [m for m in models if m not in ("mfcc", "fbank")]
    all_per_word = {name: [] for name in DATASETS.keys()}

    for model in ssl_models:
        layers = {name: get_best_layer(ds, model) for name, ds in DATASETS.items()}
        for name, ds in DATASETS.items():
            df = load_cell_scores(ds, model, layers[name])
            pw = aggregate_to_per_word(df).with_columns(pl.lit(model).alias("model"))
            all_per_word[name].append(pw)

    # Concatenate and compute mean across models for each word
    word_means = {}
    for name in DATASETS.keys():
        combined = pl.concat(all_per_word[name])
        word_means[name] = combined.group_by("phone_sequence").agg(
            pl.col("error_rate").mean().alias("mean_error"),
            pl.col("error_rate").std().alias("std_error"),
        )

    # Join and compute discrepancy
    for synth_name in SYNTH_NAMES:
        joined = word_means["natural"].join(
            word_means[synth_name],
            on="phone_sequence",
            suffix=f"_{synth_name}",
        )
        joined = joined.with_columns(
            (pl.col(f"mean_error_{synth_name}") - pl.col("mean_error")).alias("discrepancy"),
            ((pl.col(f"mean_error_{synth_name}") - pl.col("mean_error")).abs()).alias("abs_discrepancy"),
        )
        joined = joined.sort("abs_discrepancy", descending=True)

        print(f"\n{DISPLAY_NAMES[synth_name]} - Top discrepancies (synth - natural):")
        print(f"{'Word':<15} {'Natural':>10} {'Synth':>10} {'Diff':>10}")
        print("-" * 47)
        for row in joined.head(10).iter_rows(named=True):
            print(f"{row['phone_sequence']:<15} {row['mean_error']:>10.3f} "
                  f"{row[f'mean_error_{synth_name}']:>10.3f} {row['discrepancy']:>+10.3f}")

        print(f"\nOverall correlation (averaged across models):")
        nat = joined["mean_error"].to_numpy()
        syn = joined[f"mean_error_{synth_name}"].to_numpy()
        r, _ = stats.pearsonr(nat, syn)
        rho, _ = stats.spearmanr(nat, syn)
        print(f"  Pearson r = {r:.3f}, Spearman ρ = {rho:.3f}")

    return joined


def analysis_3_model_sensitivity(models: list[str]) -> pl.DataFrame:
    """Check if certain models are more robust to the natural-synth gap."""
    print("\n" + "=" * 60)
    print("ANALYSIS 3: Model-specific sensitivity")
    print("=" * 60)

    results = []

    for model in models:
        layers = {name: get_best_layer(ds, model) for name, ds in DATASETS.items()}

        # Compute overall error rate for each condition
        error_rates = {}
        for name, ds in DATASETS.items():
            df = load_cell_scores(ds, model, layers[name])
            error_rates[name] = df["score"].mean()

        # Also compute per-word correlation
        per_word = {}
        for name, ds in DATASETS.items():
            df = load_cell_scores(ds, model, layers[name])
            per_word[name] = aggregate_to_per_word(df)

        for synth_name in SYNTH_NAMES:
            joined = per_word["natural"].join(
                per_word[synth_name], on="phone_sequence", suffix="_synth"
            )
            nat = joined["error_rate"].to_numpy()
            syn = joined["error_rate_synth"].to_numpy()
            r, _ = stats.pearsonr(nat, syn)

            results.append({
                "model": model,
                "synth": synth_name,
                "natural_error": error_rates["natural"],
                "synth_error": error_rates[synth_name],
                "error_gap": error_rates[synth_name] - error_rates["natural"],
                "abs_error_gap": abs(error_rates[synth_name] - error_rates["natural"]),
                "word_correlation": r,
            })

    results_df = pl.DataFrame(results)

    # Find models with smallest gap
    for synth in SYNTH_NAMES:
        subset = results_df.filter(pl.col("synth") == synth).sort("abs_error_gap")

        print(f"\n{DISPLAY_NAMES[synth]} - Models by absolute error gap:")
        print(f"{'Model':<35} {'Natural':>8} {'Synth':>8} {'Gap':>8} {'r':>6}")
        print("-" * 67)
        for row in subset.iter_rows(named=True):
            print(f"{row['model']:<35} {row['natural_error']:>8.3f} "
                  f"{row['synth_error']:>8.3f} {row['error_gap']:>+8.3f} "
                  f"{row['word_correlation']:>6.3f}")

    # Check if certain model families are more robust (SSL only)
    print("\n\nSSL model family analysis:")
    ssl_df = results_df.filter(~pl.col("model").is_in(["mfcc", "fbank"]))
    ssl_df = ssl_df.with_columns(
        pl.when(pl.col("model").str.contains("hubert"))
        .then(pl.lit("HuBERT"))
        .when(pl.col("model").str.contains("wav2vec"))
        .then(pl.lit("wav2vec2"))
        .when(pl.col("model").str.contains("wavlm"))
        .then(pl.lit("WavLM"))
        .otherwise(pl.lit("other"))
        .alias("family")
    )

    for synth in SYNTH_NAMES:
        print(f"\n{DISPLAY_NAMES[synth]} - Mean absolute gap by model family:")
        subset = ssl_df.filter(
            (pl.col("synth") == synth) & (pl.col("family") != "other")
        )
        by_family = subset.group_by("family").agg(
            pl.col("abs_error_gap").mean().alias("mean_gap"),
            pl.col("word_correlation").mean().alias("mean_corr"),
            pl.len().alias("n"),
        ).sort("mean_gap")
        for row in by_family.iter_rows(named=True):
            print(f"  {row['family']:<10}: gap={row['mean_gap']:.3f}, r={row['mean_corr']:.3f} (n={row['n']})")

    return results_df


def plot_per_word_scatter_by_type(models: list[str], output_dir: Path):
    """Create scatter plots comparing SSL models vs baselines (MFCC, FBANK)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Separate models by type
    ssl_models = [m for m in models if m not in ("mfcc", "fbank")]

    # Helper to compute word means for a set of models
    def get_word_means(model_subset: list[str]) -> dict[str, pl.DataFrame]:
        all_per_word = {name: [] for name in DATASETS.keys()}
        for model in model_subset:
            layers = {name: get_best_layer(ds, model) for name, ds in DATASETS.items()}
            for name, ds in DATASETS.items():
                df = load_cell_scores(ds, model, layers[name])
                pw = aggregate_to_per_word(df).with_columns(pl.lit(model).alias("model"))
                all_per_word[name].append(pw)

        word_means = {}
        for name in DATASETS.keys():
            combined = pl.concat(all_per_word[name])
            word_means[name] = combined.group_by("phone_sequence").agg(
                pl.col("error_rate").mean().alias("mean_error"),
            )
        return word_means

    # Compute word means for each model type
    row_data = [
        ("SSL Models", get_word_means(ssl_models)),
        ("MFCC", get_word_means(["mfcc"])),
        ("FBANK", get_word_means(["fbank"])),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(12, 14))

    for row_idx, (row_label, word_means) in enumerate(row_data):
        for col_idx, synth_name in enumerate(SYNTH_NAMES):
            ax = axes[row_idx, col_idx]

            joined = word_means["natural"].join(
                word_means[synth_name], on="phone_sequence", suffix="_synth"
            )

            nat = joined["mean_error"].to_numpy()
            syn = joined["mean_error_synth"].to_numpy()
            words = joined["phone_sequence"].to_list()

            # Scatter points with consistent styling
            ax.scatter(
                nat, syn,
                s=SCATTER_S_MED,
                alpha=0.7,
                c="tab:blue",
                edgecolors="white",
                linewidth=SCATTER_EDGEWIDTH,
                zorder=3
            )

            # Add word labels using adjust_text
            texts = []
            for i, word in enumerate(words):
                texts.append(
                    ax.text(nat[i], syn[i], word, fontsize=10, fontweight="medium")
                )
            adjust_text(
                texts, ax=ax,
                arrowprops=dict(arrowstyle="-", lw=0.5, color="gray", alpha=0.5)
            )

            # Identity line
            lims = [0, max(nat.max(), syn.max()) + 0.05]
            ax.plot(
                lims, lims,
                linestyle=":",
                color="gray",
                lw=LINEWIDTH_THIN,
                alpha=0.5,
                zorder=1
            )

            # Regression line
            slope, intercept, r, _, _ = stats.linregress(nat, syn)
            x_line = np.array(lims)
            ax.plot(
                x_line, slope * x_line + intercept,
                linestyle="--",
                color="black",
                lw=LINEWIDTH_THIN,
                alpha=0.4,
                zorder=2
            )

            ax.set_xlabel("Natural ABX Error Rate")
            ax.set_ylabel(f"{DISPLAY_NAMES[synth_name]} ABX Error Rate")
            ax.set_title(f"{row_label}: Natural vs {DISPLAY_NAMES[synth_name]}", fontweight="bold")

            # Correlation annotation
            ax.text(
                0.05, 0.95,
                f"r = {r:.3f}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=13,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="0.8")
            )

            ax.grid(True, linestyle="--", alpha=0.3)
            ax.set_xlim(lims)
            ax.set_ylim(lims)

    plt.tight_layout(**TIGHT_LAYOUT_KW)
    plt.savefig(output_dir / "english_stress_natural_vs_synth_by_type.png", **SAVEFIG_KW)

    # Remove titles for PDF version
    for ax in axes.flat:
        ax.set_title("")
    plt.savefig(output_dir / "english_stress_natural_vs_synth_by_type.pdf", **SAVEFIG_KW)
    print(f"\nSaved scatter plot to {output_dir}/english_stress_natural_vs_synth_by_type.pdf")


def main():
    models = get_common_models()
    print(f"Found {len(models)} models common to all datasets")

    # Run analyses
    word_corr_df = analysis_1_per_word_correlation(models)
    pair_df = analysis_2_per_pair_breakdown(models)
    model_df = analysis_3_model_sensitivity(models)

    # Save results
    output_dir = Path("plots/english_tts_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    word_corr_df.write_csv(output_dir / "per_word_correlation.csv")
    model_df.write_csv(output_dir / "model_sensitivity.csv")

    # Create visualization
    plot_per_word_scatter_by_type(models, output_dir)

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
