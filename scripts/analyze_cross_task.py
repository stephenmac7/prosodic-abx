#!/usr/bin/env python3
"""
Compare layer-wise model performance across two tasks.

This analysis focuses on:
1. Layer curve shape similarity (correlations, slope correlations)
2. Best-layer agreement across tasks
3. Performance scatter between tasks

Usage:
    python analyze_cross_task.py TASK_A TASK_B [--bootstrap 1000]

Examples:
    python analyze_cross_task.py pitch_accent mandarin_tone
    python analyze_cross_task.py pitch_accent stress
    python analyze_cross_task.py stress mandarin_tone
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from plot_prosodic_results import MODEL_METADATA, load_results, get_layer_num

RESULTS_DIR = Path(__file__).parent.parent / "results"
OUTPUT_BASE_DIR = Path(__file__).parent.parent / "plots"

BASELINE_MODELS = {"mfcc", "fbank"}

DATASET_LABELS = {
    "pitch_accent": "Pitch Accent",
    "stress": "Lexical Stress",
    "stress_kokoro": "Lexical Stress (Kokoro)",
    "mandarin_tone": "Mandarin Tone",
    "pitch_accent_syn": "Pitch Accent (Syn)",
    "stress_syn": "Lexical Stress (Syn)",
    "mandarin_tone_syn": "Mandarin Tone (Syn)",
}


def compute_slopes(error_rates):
    """Compute first differences (slopes) of error rates."""
    return np.diff(error_rates)


def compute_per_model_stats(model_name, df_a, df_b):
    """Compute comparison statistics for a single model across two tasks."""
    merged = pd.merge(df_a, df_b, on="layer", suffixes=("_a", "_b"))
    if merged.empty:
        return None, None

    merged["layer_num"] = merged["layer"].map(get_layer_num)
    merged.sort_values("layer_num", inplace=True)

    errors_a = merged["error_rate_a"].values
    errors_b = merged["error_rate_b"].values

    # Best layer info
    a_best_idx = np.argmin(errors_a)
    b_best_idx = np.argmin(errors_b)
    a_best_layer = merged.iloc[a_best_idx]["layer_num"]
    b_best_layer = merged.iloc[b_best_idx]["layer_num"]

    # Layer-wise correlations
    if len(merged) >= 3:
        pearson_r, pearson_p = sp_stats.pearsonr(errors_a, errors_b)
        spearman_r, spearman_p = sp_stats.spearmanr(errors_a, errors_b)
    else:
        pearson_r = spearman_r = np.nan
        pearson_p = spearman_p = np.nan

    # Slope (first difference) correlation - captures shape similarity
    if len(merged) >= 4:
        slopes_a = compute_slopes(errors_a)
        slopes_b = compute_slopes(errors_b)
        slope_pearson, _ = sp_stats.pearsonr(slopes_a, slopes_b)
        slope_spearman, _ = sp_stats.spearmanr(slopes_a, slopes_b)
    else:
        slope_pearson = slope_spearman = np.nan

    # Best layer agreement
    layer_diff = int(a_best_layer - b_best_layer)
    best_match = int(a_best_layer == b_best_layer)

    stats_dict = {
        "model": model_name,
        "n_layers": len(merged),
        # Correlations
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "slope_pearson": slope_pearson,
        "slope_spearman": slope_spearman,
        # Best layer
        "a_best_layer": a_best_layer,
        "b_best_layer": b_best_layer,
        "layer_diff": layer_diff,
        "best_match": best_match,
        # Performance
        "a_best_error": errors_a[a_best_idx],
        "b_best_error": errors_b[b_best_idx],
    }

    merged["model"] = model_name
    return stats_dict, merged


def bootstrap_ci(values, stat_fn, n_boot, rng):
    """Compute bootstrap confidence interval."""
    if n_boot <= 0:
        return None
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size < 2:
        return None
    boot = np.array([stat_fn(rng.choice(arr, size=arr.size, replace=True)) for _ in range(n_boot)])
    return np.quantile(boot, [0.025, 0.975])


def format_ci(ci):
    """Format confidence interval as string."""
    if ci is None or np.any(np.isnan(ci)):
        return "N/A"
    return f"[{ci[0]:.4f}, {ci[1]:.4f}]"


def format_median_iqr(series):
    """Format median with IQR."""
    series = series.dropna()
    if series.empty:
        return "N/A"
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    median = series.median()
    return f"{median:.4f} [{q1:.4f}, {q3:.4f}]"


def plot_distributions(stats_df, output_path, label_a, label_b):
    """Plot distributions of key statistics."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    metrics = [
        ("pearson_r", "Pearson r (layer-wise)", "tab:blue"),
        ("spearman_r", "Spearman ρ (layer-wise)", "tab:orange"),
        ("slope_pearson", "Slope Pearson r", "tab:green"),
        ("slope_spearman", "Slope Spearman ρ", "tab:red"),
        ("layer_diff", f"Best layer diff ({label_a} − {label_b})", "tab:pink"),
        ("a_best_error", f"Best error comparison", "tab:cyan"),
    ]

    for ax, (col, title, color) in zip(axes, metrics):
        data = stats_df[col].dropna()
        if len(data) > 0:
            ax.boxplot(data, vert=True, patch_artist=True,
                      boxprops=dict(facecolor=color, alpha=0.6))
            ax.axhline(0, color="black", linewidth=1, linestyle="--")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])

    plt.suptitle(f"{label_a} vs {label_b}: Cross-Task Statistics", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_layerwise_comparison(all_merged, output_path, label_a, label_b):
    """Plot layerwise curves for each model, comparing two tasks."""
    models = all_merged["model"].unique()
    n_models = len(models)
    cols = min(4, n_models)
    rows = (n_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, model in enumerate(sorted(models)):
        ax = axes[idx]
        df = all_merged[all_merged["model"] == model].sort_values("layer_num")

        ax.plot(df["layer_num"], df["error_rate_a"], "o-", label=label_a, alpha=0.8)
        ax.plot(df["layer_num"], df["error_rate_b"], "s--", label=label_b, alpha=0.8)

        ax.set_title(model, fontsize=8)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Error rate")
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f"{label_a} vs {label_b}: Layer-wise Error Rates", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_best_layer_scatter(stats_df, output_path, label_a, label_b):
    """Scatter plot of best layer on task A vs task B."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Best layer comparison
    ax = axes[0]
    ax.scatter(stats_df["a_best_layer"], stats_df["b_best_layer"],
              alpha=0.6, s=40, zorder=3)

    lims = [
        min(stats_df["a_best_layer"].min(), stats_df["b_best_layer"].min()) - 1,
        max(stats_df["a_best_layer"].max(), stats_df["b_best_layer"].max()) + 1,
    ]
    ax.plot(lims, lims, "k--", alpha=0.5, label="y=x")

    for _, row in stats_df.iterrows():
        ax.annotate(row["model"], (row["a_best_layer"], row["b_best_layer"]),
                   fontsize=5, alpha=0.7)

    ax.set_xlabel(f"Best Layer ({label_a})")
    ax.set_ylabel(f"Best Layer ({label_b})")
    ax.set_title("Best Layer Agreement")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Best error comparison
    ax = axes[1]
    ax.scatter(stats_df["a_best_error"], stats_df["b_best_error"],
              alpha=0.6, s=40, zorder=3)

    for _, row in stats_df.iterrows():
        ax.annotate(row["model"], (row["a_best_error"], row["b_best_error"]),
                   fontsize=5, alpha=0.7)

    # Fit regression
    slope, intercept, r, p, se = sp_stats.linregress(
        stats_df["a_best_error"], stats_df["b_best_error"]
    )
    x_fit = np.linspace(stats_df["a_best_error"].min(), stats_df["a_best_error"].max(), 100)
    ax.plot(x_fit, slope * x_fit + intercept, "r-",
           label=f"Linear fit (r={r:.3f}, p={p:.3g})")

    ax.set_xlabel(f"Best Error ({label_a})")
    ax.set_ylabel(f"Best Error ({label_b})")
    ax.set_title("Best-Layer Error Rate Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"{label_a} vs {label_b}", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_correlation_scatter(all_merged, output_path, label_a, label_b):
    """Scatter plot of task A vs task B error rates (all model-layer points)."""
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(all_merged["error_rate_a"], all_merged["error_rate_b"],
              alpha=0.4, s=15)

    # Fit and plot regression line
    slope, intercept, r, p, se = sp_stats.linregress(
        all_merged["error_rate_a"], all_merged["error_rate_b"]
    )
    x_fit = np.linspace(all_merged["error_rate_a"].min(), all_merged["error_rate_a"].max(), 100)
    ax.plot(x_fit, slope * x_fit + intercept, "r-",
           label=f"Linear fit (r={r:.3f}, p={p:.3g})")

    ax.set_xlabel(f"{label_a} Error Rate")
    ax.set_ylabel(f"{label_b} Error Rate")
    ax.set_title(f"All Model-Layer Points: {label_a} vs {label_b}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def write_summary(stats_df, all_merged, n_boot, rng, output_path,
                  label_a, label_b):
    """Write summary statistics to markdown file."""
    # Compute CIs
    cis = {
        "pearson_r": bootstrap_ci(stats_df["pearson_r"], np.median, n_boot, rng),
        "spearman_r": bootstrap_ci(stats_df["spearman_r"], np.median, n_boot, rng),
        "slope_pearson": bootstrap_ci(stats_df["slope_pearson"], np.median, n_boot, rng),
        "slope_spearman": bootstrap_ci(stats_df["slope_spearman"], np.median, n_boot, rng),
        "best_match": bootstrap_ci(stats_df["best_match"], np.mean, n_boot, rng),
        "layer_diff": bootstrap_ci(stats_df["layer_diff"], np.median, n_boot, rng),
    }

    # Overall correlation across all model-layer points
    overall_r, overall_r_p = sp_stats.pearsonr(all_merged["error_rate_a"], all_merged["error_rate_b"])
    overall_rho, overall_rho_p = sp_stats.spearmanr(all_merged["error_rate_a"], all_merged["error_rate_b"])

    # Best-error correlation across models
    best_err_r, best_err_p = sp_stats.pearsonr(stats_df["a_best_error"], stats_df["b_best_error"])
    best_err_rho, best_err_rho_p = sp_stats.spearmanr(stats_df["a_best_error"], stats_df["b_best_error"])

    lines = [
        f"# Cross-Task Analysis: {label_a} vs {label_b}",
        "",
        "## Layer Curve Shape Similarity (per-model)",
        "| Metric | Median [IQR] | 95% CI |",
        "| --- | --- | --- |",
        f"| Pearson r (layer-wise) | {format_median_iqr(stats_df['pearson_r'])} | {format_ci(cis['pearson_r'])} |",
        f"| Spearman ρ (layer-wise) | {format_median_iqr(stats_df['spearman_r'])} | {format_ci(cis['spearman_r'])} |",
        f"| Slope Pearson r | {format_median_iqr(stats_df['slope_pearson'])} | {format_ci(cis['slope_pearson'])} |",
        f"| Slope Spearman ρ | {format_median_iqr(stats_df['slope_spearman'])} | {format_ci(cis['slope_spearman'])} |",
        "",
        f"Overall correlation (all model-layer points): r={overall_r:.4f} (p={overall_r_p:.3g}), ρ={overall_rho:.4f} (p={overall_rho_p:.3g})",
        "",
        "## Best-Error Correlation Across Models",
        f"Pearson r={best_err_r:.4f} (p={best_err_p:.3g}), Spearman ρ={best_err_rho:.4f} (p={best_err_rho_p:.3g})",
        "",
        "## Best Layer Agreement",
        "| Metric | Value | 95% CI |",
        "| --- | --- | --- |",
        f"| Exact best-layer match | {stats_df['best_match'].mean():.1%} | {format_ci(cis['best_match'])} |",
        f"| Layer diff median [IQR] | {format_median_iqr(stats_df['layer_diff'])} | {format_ci(cis['layer_diff'])} |",
        "",
        "## Per-Model Details",
        f"| Model | Pearson r | Best Layer ({label_a}) | Best Layer ({label_b}) | Diff | Best Error ({label_a}) | Best Error ({label_b}) |",
        f"| --- | --- | --- | --- | --- | --- | --- |",
    ]

    for _, row in stats_df.sort_values("pearson_r", ascending=False).iterrows():
        lines.append(
            f"| {row['model']} | {row['pearson_r']:.3f} | {int(row['a_best_layer'])} | {int(row['b_best_layer'])} "
            f"| {int(row['layer_diff'])} | {row['a_best_error']:.4f} | {row['b_best_error']:.4f} |"
        )

    lines += [
        "",
        "## Interpretation",
        f"- Positive layer diff means {label_a} prefers a deeper best layer than {label_b}",
        f"- High per-model correlations indicate similar layer-wise profiles across the two tasks",
        "",
        f"Number of models analyzed: {len(stats_df)}",
    ]

    output_path.write_text("\n".join(lines) + "\n")


def write_summary_json(stats_df, all_merged, n_boot, rng, output_path,
                       label_a, label_b):
    """Write summary statistics to JSON file."""
    overall_r, _ = sp_stats.pearsonr(all_merged["error_rate_a"], all_merged["error_rate_b"])
    overall_rho, _ = sp_stats.spearmanr(all_merged["error_rate_a"], all_merged["error_rate_b"])

    best_err_r, _ = sp_stats.pearsonr(stats_df["a_best_error"], stats_df["b_best_error"])
    best_err_rho, _ = sp_stats.spearmanr(stats_df["a_best_error"], stats_df["b_best_error"])

    summary = {
        "meta": {
            "task_a": label_a,
            "task_b": label_b,
            "n_models": int(len(stats_df)),
            "n_bootstrap": int(n_boot),
        },
        "shape_similarity": {
            "pearson_median": float(stats_df["pearson_r"].median()),
            "pearson_ci": format_ci(bootstrap_ci(stats_df["pearson_r"], np.median, n_boot, rng)),
            "spearman_median": float(stats_df["spearman_r"].median()),
            "spearman_ci": format_ci(bootstrap_ci(stats_df["spearman_r"], np.median, n_boot, rng)),
            "slope_pearson_median": float(stats_df["slope_pearson"].median()) if not stats_df["slope_pearson"].isna().all() else None,
            "slope_spearman_median": float(stats_df["slope_spearman"].median()) if not stats_df["slope_spearman"].isna().all() else None,
            "overall_r": float(overall_r),
            "overall_rho": float(overall_rho),
        },
        "best_error_correlation": {
            "pearson_r": float(best_err_r),
            "spearman_rho": float(best_err_rho),
        },
        "best_layer_agreement": {
            "exact_match_rate": float(stats_df["best_match"].mean()),
            "layer_diff_median": float(stats_df["layer_diff"].median()),
        },
        "per_model": [
            {
                "model": row["model"],
                "pearson_r": float(row["pearson_r"]),
                "spearman_r": float(row["spearman_r"]),
                "a_best_layer": int(row["a_best_layer"]),
                "b_best_layer": int(row["b_best_layer"]),
                "layer_diff": int(row["layer_diff"]),
                "a_best_error": float(row["a_best_error"]),
                "b_best_error": float(row["b_best_error"]),
            }
            for _, row in stats_df.iterrows()
        ],
    }

    output_path.write_text(json.dumps(summary, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compare layer-wise model performance across two tasks."
    )
    parser.add_argument(
        "task_a",
        help="First task/dataset name (e.g., pitch_accent)",
    )
    parser.add_argument(
        "task_b",
        help="Second task/dataset name (e.g., mandarin_tone)",
    )
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-baselines", action="store_true")
    args = parser.parse_args()

    label_a = DATASET_LABELS.get(args.task_a, args.task_a.replace("_", " ").title())
    label_b = DATASET_LABELS.get(args.task_b, args.task_b.replace("_", " ").title())

    dir_a = RESULTS_DIR / args.task_a
    dir_b = RESULTS_DIR / args.task_b
    output_dir = OUTPUT_BASE_DIR / f"cross_task_{args.task_a}_vs_{args.task_b}"

    if not dir_a.exists():
        raise SystemExit(f"Results not found: {dir_a}")
    if not dir_b.exists():
        raise SystemExit(f"Results not found: {dir_b}")

    results_a = load_results(dir_a)
    results_b = load_results(dir_b)

    models = sorted(set(results_a.keys()) & set(results_b.keys()))
    if not args.include_baselines:
        models = [
            m for m in models
            if m not in BASELINE_MODELS
            and not (m in MODEL_METADATA and MODEL_METADATA[m][0] == "Baseline")
        ]

    print(f"Analyzing {label_a} vs {label_b}: {len(models)} shared models...")

    stats_rows = []
    merged_rows = []

    for model in models:
        df_a = results_a[model]
        df_b = results_b[model]

        stats, merged = compute_per_model_stats(model, df_a, df_b)
        if stats is None:
            continue
        stats_rows.append(stats)
        merged_rows.append(merged)

    if not stats_rows:
        raise SystemExit("No overlapping models/layers found.")

    stats_df = pd.DataFrame(stats_rows)
    all_merged = pd.concat(merged_rows, ignore_index=True)
    rng = np.random.default_rng(args.seed)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save per-model stats
    stats_df.to_csv(output_dir / "per_model_stats.csv", index=False)
    print(f"Saved: {output_dir / 'per_model_stats.csv'}")

    # Generate plots
    plot_distributions(stats_df, output_dir / "distributions.png", label_a, label_b)
    print(f"Saved: {output_dir / 'distributions.png'}")

    plot_layerwise_comparison(all_merged, output_dir / "layerwise_comparison.png", label_a, label_b)
    print(f"Saved: {output_dir / 'layerwise_comparison.png'}")

    plot_best_layer_scatter(stats_df, output_dir / "best_layer_scatter.png", label_a, label_b)
    print(f"Saved: {output_dir / 'best_layer_scatter.png'}")

    plot_correlation_scatter(all_merged, output_dir / "correlation_scatter.png", label_a, label_b)
    print(f"Saved: {output_dir / 'correlation_scatter.png'}")

    # Write summaries
    write_summary(stats_df, all_merged, args.bootstrap, rng,
                  output_dir / "summary.md", label_a, label_b)
    print(f"Saved: {output_dir / 'summary.md'}")

    write_summary_json(stats_df, all_merged, args.bootstrap, rng,
                       output_dir / "summary.json", label_a, label_b)
    print(f"Saved: {output_dir / 'summary.json'}")

    # Print key results
    print("\n" + "=" * 60)
    print(f"KEY RESULTS: {label_a} vs {label_b}")
    print("=" * 60)
    print(f"\nShape similarity (median per-model):")
    print(f"  Layer-wise Pearson r:  {stats_df['pearson_r'].median():.3f}")
    print(f"  Layer-wise Spearman ρ: {stats_df['spearman_r'].median():.3f}")
    if not stats_df["slope_pearson"].isna().all():
        print(f"  Slope Pearson r:       {stats_df['slope_pearson'].median():.3f}")

    # Overall correlation
    overall_r, _ = sp_stats.pearsonr(all_merged["error_rate_a"], all_merged["error_rate_b"])
    print(f"\nOverall layer-point correlation: r={overall_r:.3f}")

    # Best-error correlation
    best_r, best_p = sp_stats.pearsonr(stats_df["a_best_error"], stats_df["b_best_error"])
    print(f"Best-error correlation:  r={best_r:.3f} (p={best_p:.3g})")

    print(f"\nBest layer agreement:")
    print(f"  Exact match:    {stats_df['best_match'].mean():.1%}")

    print(f"\nLayer diff ({label_a} − {label_b}):")
    print(f"  Median: {stats_df['layer_diff'].median():.1f}")
    print(f"  Mean:   {stats_df['layer_diff'].mean():.1f}")


if __name__ == "__main__":
    main()
