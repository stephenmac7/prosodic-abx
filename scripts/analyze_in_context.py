#!/usr/bin/env python3
"""
Compare in-context vs out-of-context ABX results.

This analysis focuses on:
1. Layer curve shape similarity (correlations, slope correlations)
2. Performance differences between conditions
3. Best-layer agreement

Usage:
    python analyze_in_context.py DATASET [--bootstrap 1000]

Examples:
    python analyze_in_context.py pitch_accent
    python analyze_in_context.py stress_kokoro
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
    "tone": "Mandarin Tone",
}


def compute_slopes(error_rates):
    """Compute first differences (slopes) of error rates."""
    return np.diff(error_rates)


def compute_per_model_stats(model_name, df_out, df_in, topk=3):
    """Compute comparison statistics for a single model."""
    merged = pd.merge(df_out, df_in, on="layer", suffixes=("_out", "_in"))
    if merged.empty:
        return None, None

    merged["layer_num"] = merged["layer"].map(get_layer_num)
    merged.sort_values("layer_num", inplace=True)

    errors_out = merged["error_rate_out"].values
    errors_in = merged["error_rate_in"].values

    # Best layer info
    out_best_idx = np.argmin(errors_out)
    in_best_idx = np.argmin(errors_in)
    out_best_layer = merged.iloc[out_best_idx]["layer_num"]
    in_best_layer = merged.iloc[in_best_idx]["layer_num"]

    # Layer-wise correlations
    if len(merged) >= 3:
        pearson_r, pearson_p = sp_stats.pearsonr(errors_out, errors_in)
        spearman_r, spearman_p = sp_stats.spearmanr(errors_out, errors_in)
    else:
        pearson_r = spearman_r = np.nan
        pearson_p = spearman_p = np.nan

    # Slope (first difference) correlation - captures shape similarity
    if len(merged) >= 4:
        slopes_out = compute_slopes(errors_out)
        slopes_in = compute_slopes(errors_in)
        slope_pearson, _ = sp_stats.pearsonr(slopes_out, slopes_in)
        slope_spearman, _ = sp_stats.spearmanr(slopes_out, slopes_in)
    else:
        slope_pearson = slope_spearman = np.nan

    # Performance differences
    delta_all = errors_out - errors_in  # positive = out-of-context worse
    delta_best_out = errors_out[out_best_idx] - errors_in[out_best_idx]
    delta_best_in = errors_out[in_best_idx] - errors_in[in_best_idx]
    delta_best_layer_error = merged.iloc[out_best_idx]["error_rate_out"] - merged.iloc[in_best_idx]["error_rate_in"]

    # Best layer agreement
    layer_diff = int(out_best_layer - in_best_layer)
    best_match = int(out_best_layer == in_best_layer)

    # Top-k agreement
    topk_out_layers = merged.nsmallest(topk, "error_rate_out")["layer_num"].tolist()
    topk_agreement = int(in_best_layer in topk_out_layers)

    # Within ±1 layer agreement
    within_1 = int(abs(layer_diff) <= 1)
    within_2 = int(abs(layer_diff) <= 2)

    # Depth-delta correlation: does the in-context advantage grow with depth?
    max_layer = merged["layer_num"].max()
    if max_layer > 0 and len(merged) >= 3:
        norm_depth = merged["layer_num"].values / max_layer
        depth_delta_rho, _ = sp_stats.spearmanr(norm_depth, delta_all)
    else:
        norm_depth = None
        depth_delta_rho = np.nan

    # Partial correlation: curve similarity after removing depth trend
    if norm_depth is not None and len(merged) >= 4:
        # Residualize both curves against depth
        coef_out = np.polyfit(norm_depth, errors_out, 1)
        coef_in = np.polyfit(norm_depth, errors_in, 1)
        resid_out = errors_out - np.polyval(coef_out, norm_depth)
        resid_in = errors_in - np.polyval(coef_in, norm_depth)
        partial_pearson, _ = sp_stats.pearsonr(resid_out, resid_in)
        partial_spearman, _ = sp_stats.spearmanr(resid_out, resid_in)
    else:
        partial_pearson = np.nan
        partial_spearman = np.nan

    stats_dict = {
        "model": model_name,
        "n_layers": len(merged),
        # Correlations
        "pearson_r": pearson_r,
        "spearman_r": spearman_r,
        "slope_pearson": slope_pearson,
        "slope_spearman": slope_spearman,
        # Best layer
        "out_best_layer": out_best_layer,
        "in_best_layer": in_best_layer,
        "layer_diff": layer_diff,
        "best_match": best_match,
        f"top{topk}_agreement": topk_agreement,
        "within_1": within_1,
        "within_2": within_2,
        # Performance
        "out_best_error": errors_out[out_best_idx],
        "in_best_error": errors_in[in_best_idx],
        "delta_mean": np.mean(delta_all),
        "delta_std": np.std(delta_all),
        "delta_at_out_best": delta_best_out,
        "delta_at_in_best": delta_best_in,
        "delta_best_vs_best": delta_best_layer_error,
        "depth_delta_rho": depth_delta_rho,
        "partial_pearson": partial_pearson,
        "partial_spearman": partial_spearman,
    }

    merged["model"] = model_name
    merged["delta"] = delta_all
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


def plot_distributions(stats_df, output_path):
    """Plot distributions of key statistics."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    metrics = [
        ("pearson_r", "Pearson r (layer-wise)", "tab:blue"),
        ("spearman_r", "Spearman ρ (layer-wise)", "tab:orange"),
        ("slope_pearson", "Slope Pearson r", "tab:green"),
        ("slope_spearman", "Slope Spearman ρ", "tab:red"),
        ("delta_mean", "Mean Δ (out − in)", "tab:purple"),
        ("delta_at_out_best", "Δ at out-best layer", "tab:brown"),
        ("layer_diff", "Layer diff (out − in)", "tab:pink"),
        ("delta_best_vs_best", "Δ best vs best", "tab:cyan"),
    ]

    for ax, (col, title, color) in zip(axes, metrics):
        data = stats_df[col].dropna()
        if len(data) > 0:
            ax.boxplot(data, vert=True, patch_artist=True,
                      boxprops=dict(facecolor=color, alpha=0.6))
            ax.axhline(0, color="black", linewidth=1, linestyle="--")
        ax.set_title(title)
        ax.set_xticks([])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_layerwise_comparison(all_merged, output_path):
    """Plot layerwise curves for each model, comparing in vs out of context."""
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

        ax.plot(df["layer_num"], df["error_rate_out"], "o-", label="Out-of-context", alpha=0.8)
        ax.plot(df["layer_num"], df["error_rate_in"], "s--", label="In-context", alpha=0.8)

        ax.set_title(model, fontsize=8)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Error rate")
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_delta_by_layer(all_merged, output_path):
    """Plot delta (out - in) by layer for all models."""
    fig, ax = plt.subplots(figsize=(10, 6))

    models = all_merged["model"].unique()
    for model in sorted(models):
        df = all_merged[all_merged["model"] == model].sort_values("layer_num")
        max_layer = df["layer_num"].max()
        norm_layers = df["layer_num"] / max_layer
        ax.plot(norm_layers, df["delta"], "o-", label=model, alpha=0.5, markersize=3)

    # Add mean delta line
    layer_means = all_merged.groupby("layer_num")["delta"].mean()
    if len(layer_means) > 0:
        # Normalize to 0-1 for the mean line
        ax.axhline(all_merged["delta"].mean(), color="black", linewidth=2,
                  linestyle="--", label=f"Overall mean: {all_merged['delta'].mean():.4f}")

    ax.axhline(0, color="gray", linewidth=1)
    ax.set_xlabel("Normalized Layer Depth")
    ax.set_ylabel("Δ Error Rate (out − in)")
    ax.set_title("Performance Difference by Layer")
    ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.02, 1))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_correlation_scatter(all_merged, output_path):
    """Scatter plot of out-of-context vs in-context error rates."""
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(all_merged["error_rate_out"], all_merged["error_rate_in"],
              alpha=0.5, s=20)

    # Add diagonal line
    lims = [
        min(all_merged["error_rate_out"].min(), all_merged["error_rate_in"].min()),
        max(all_merged["error_rate_out"].max(), all_merged["error_rate_in"].max()),
    ]
    ax.plot(lims, lims, "k--", alpha=0.5, label="y=x")

    # Fit and plot regression line
    slope, intercept, r, p, se = sp_stats.linregress(
        all_merged["error_rate_out"], all_merged["error_rate_in"]
    )
    x_fit = np.array(lims)
    ax.plot(x_fit, slope * x_fit + intercept, "r-",
           label=f"Linear fit (r={r:.3f})")

    ax.set_xlabel("Out-of-context Error Rate")
    ax.set_ylabel("In-context Error Rate")
    ax.set_title("Out-of-context vs In-context ABX Error Rates")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_model_rank_bump(stats_df, output_path):
    """Bump chart showing how model rankings change between conditions."""
    df = stats_df[["model", "out_best_error", "in_best_error"]].copy()
    df["rank_out"] = df["out_best_error"].rank(method="min")
    df["rank_in"] = df["in_best_error"].rank(method="min")

    n = len(df)
    fig, ax = plt.subplots(figsize=(6, max(4, n * 0.35)))

    for _, row in df.iterrows():
        r_out = row["rank_out"]
        r_in = row["rank_in"]
        shift = abs(r_out - r_in)
        color = "tab:red" if shift >= 3 else ("tab:orange" if shift >= 1 else "tab:gray")
        alpha = 1.0 if shift >= 1 else 0.5
        ax.plot([0, 1], [r_out, r_in], "o-", color=color, alpha=alpha,
                markersize=5, linewidth=1.5)
        ax.text(-0.05, r_out, row["model"], ha="right", va="center", fontsize=6)
        ax.text(1.05, r_in, row["model"], ha="left", va="center", fontsize=6)

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(n + 0.5, 0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Out-of-context", "In-context"])
    ax.set_ylabel("Rank (1 = best)")
    ax.set_title("Model Ranking: Out-of-context vs In-context")
    ax.grid(True, axis="y", alpha=0.2)

    # Compute rank correlation for subtitle
    from scipy.stats import spearmanr
    rho, _ = spearmanr(df["rank_out"], df["rank_in"])
    ax.text(0.5, n + 0.4, f"Spearman ρ = {rho:.3f}", ha="center", fontsize=8,
            style="italic", transform=ax.get_yaxis_transform())

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def write_summary(stats_df, all_merged, n_boot, rng, output_path, dataset_label=""):
    """Write summary statistics to markdown file."""
    topk = 3
    topk_col = f"top{topk}_agreement"

    # Compute CIs
    cis = {
        "pearson_r": bootstrap_ci(stats_df["pearson_r"], np.median, n_boot, rng),
        "spearman_r": bootstrap_ci(stats_df["spearman_r"], np.median, n_boot, rng),
        "slope_pearson": bootstrap_ci(stats_df["slope_pearson"], np.median, n_boot, rng),
        "slope_spearman": bootstrap_ci(stats_df["slope_spearman"], np.median, n_boot, rng),
        "partial_pearson": bootstrap_ci(stats_df["partial_pearson"], np.median, n_boot, rng),
        "partial_spearman": bootstrap_ci(stats_df["partial_spearman"], np.median, n_boot, rng),
        "delta_mean": bootstrap_ci(stats_df["delta_mean"], np.median, n_boot, rng),
        "delta_at_out_best": bootstrap_ci(stats_df["delta_at_out_best"], np.median, n_boot, rng),
        "layer_diff": bootstrap_ci(stats_df["layer_diff"], np.median, n_boot, rng),
        "best_match": bootstrap_ci(stats_df["best_match"], np.mean, n_boot, rng),
        topk_col: bootstrap_ci(stats_df[topk_col], np.mean, n_boot, rng),
        "within_1": bootstrap_ci(stats_df["within_1"], np.mean, n_boot, rng),
        "within_2": bootstrap_ci(stats_df["within_2"], np.mean, n_boot, rng),
    }

    # Sign test for delta (performance)
    deltas = stats_df["delta_mean"].dropna()
    n_positive = (deltas > 0).sum()
    n_negative = (deltas < 0).sum()
    n_total = len(deltas)

    # Sign test for layer_diff (best layer shift)
    layer_diffs = stats_df["layer_diff"]
    n_layer_neg = int((layer_diffs < 0).sum())  # in-context prefers deeper
    n_layer_pos = int((layer_diffs > 0).sum())  # in-context prefers shallower
    n_layer_zero = int((layer_diffs == 0).sum())  # same best layer
    n_layer_nonzero = n_layer_neg + n_layer_pos
    if n_layer_nonzero > 0:
        layer_sign_test = sp_stats.binomtest(n_layer_pos, n_layer_nonzero, 0.5, alternative='less')
        layer_sign_p = layer_sign_test.pvalue
    else:
        layer_sign_p = np.nan

    # Wilcoxon signed-rank test on layer_diff (more powerful than sign test)
    layer_diffs_nonzero = layer_diffs[layer_diffs != 0]
    if len(layer_diffs_nonzero) >= 1:
        wilcoxon_stat, wilcoxon_p = sp_stats.wilcoxon(
            layer_diffs_nonzero, alternative='less'
        )
    else:
        wilcoxon_stat, wilcoxon_p = np.nan, np.nan

    # Depth-delta Spearman: does in-context advantage grow with depth?
    depth_delta_rhos = stats_df["depth_delta_rho"].dropna()
    depth_delta_rho_median = depth_delta_rhos.median() if len(depth_delta_rhos) > 0 else np.nan
    if len(depth_delta_rhos) >= 1:
        dd_wilcoxon_stat, dd_wilcoxon_p = sp_stats.wilcoxon(
            depth_delta_rhos, alternative='greater'
        )
    else:
        dd_wilcoxon_stat, dd_wilcoxon_p = np.nan, np.nan

    # Overall correlation across all points
    overall_r, _ = sp_stats.pearsonr(all_merged["error_rate_out"], all_merged["error_rate_in"])
    overall_rho, _ = sp_stats.spearmanr(all_merged["error_rate_out"], all_merged["error_rate_in"])

    title = f"# In-context vs Out-of-context Analysis ({dataset_label})" if dataset_label else "# In-context vs Out-of-context Analysis"
    lines = [
        title,
        "",
        "## Layer Curve Shape Similarity",
        "| Metric | Median [IQR] | 95% CI |",
        "| --- | --- | --- |",
        f"| Pearson r (layer-wise) | {format_median_iqr(stats_df['pearson_r'])} | {format_ci(cis['pearson_r'])} |",
        f"| Spearman ρ (layer-wise) | {format_median_iqr(stats_df['spearman_r'])} | {format_ci(cis['spearman_r'])} |",
        f"| Slope Pearson r | {format_median_iqr(stats_df['slope_pearson'])} | {format_ci(cis['slope_pearson'])} |",
        f"| Slope Spearman ρ | {format_median_iqr(stats_df['slope_spearman'])} | {format_ci(cis['slope_spearman'])} |",
        f"| Partial Pearson r (depth removed) | {format_median_iqr(stats_df['partial_pearson'])} | {format_ci(cis['partial_pearson'])} |",
        f"| Partial Spearman ρ (depth removed) | {format_median_iqr(stats_df['partial_spearman'])} | {format_ci(cis['partial_spearman'])} |",
        "",
        f"Overall correlation (all model-layer points): r={overall_r:.4f}, ρ={overall_rho:.4f}",
        "",
        "## Best Layer Agreement",
        "| Metric | Value | 95% CI |",
        "| --- | --- | --- |",
        f"| Exact best-layer match | {stats_df['best_match'].mean():.1%} | {format_ci(cis['best_match'])} |",
        f"| Top-{topk} agreement | {stats_df[topk_col].mean():.1%} | {format_ci(cis[topk_col])} |",
        f"| Within ±1 layer | {stats_df['within_1'].mean():.1%} | {format_ci(cis['within_1'])} |",
        f"| Within ±2 layers | {stats_df['within_2'].mean():.1%} | {format_ci(cis['within_2'])} |",
        f"| Layer diff median [IQR] | {format_median_iqr(stats_df['layer_diff'])} | {format_ci(cis['layer_diff'])} |",
        "",
        "### Layer shift direction",
        f"Among {n_layer_nonzero} models with different best layers:",
        f"- {n_layer_neg} ({n_layer_neg}/{n_layer_nonzero}) prefer **deeper** layers in-context (negative diff)",
        f"- {n_layer_pos} ({n_layer_pos}/{n_layer_nonzero}) prefer **shallower** layers in-context (positive diff)",
        f"- {n_layer_zero} models have identical best layers",
        f"- Sign test p-value: {layer_sign_p:.4f} (H1: in-context prefers deeper)",
        f"- Wilcoxon signed-rank p-value: {wilcoxon_p:.4f} (H1: in-context prefers deeper)",
        "",
        "### Depth-delta correlation (all layers)",
        f"Per-model Spearman ρ(depth, Δ) median [IQR]: {format_median_iqr(depth_delta_rhos)}",
        f"- Wilcoxon signed-rank p-value: {dd_wilcoxon_p:.4f} (H1: in-context advantage grows with depth)",
        "",
        "## Performance Differences",
        "| Metric | Median [IQR] | 95% CI |",
        "| --- | --- | --- |",
        f"| Mean Δ per model (out − in) | {format_median_iqr(stats_df['delta_mean'])} | {format_ci(cis['delta_mean'])} |",
        f"| Δ at out-best layer | {format_median_iqr(stats_df['delta_at_out_best'])} | {format_ci(cis['delta_at_out_best'])} |",
        f"| Out best error | {format_median_iqr(stats_df['out_best_error'])} | |",
        f"| In best error | {format_median_iqr(stats_df['in_best_error'])} | |",
        "",
        f"Sign test: {n_positive}/{n_total} models have positive Δ (out worse), {n_negative}/{n_total} negative",
        f"Overall mean Δ: {all_merged['delta'].mean():.4f} (std={all_merged['delta'].std():.4f})",
        "",
        "## Interpretation",
        "- Positive Δ means out-of-context has **higher** error (worse performance)",
        "- Negative Δ means in-context has **higher** error (worse performance)",
        "",
        f"Number of models analyzed: {len(stats_df)}",
    ]

    output_path.write_text("\n".join(lines) + "\n")


def write_summary_json(stats_df, all_merged, n_boot, rng, output_path):
    """Write summary statistics to JSON file."""
    topk = 3
    topk_col = f"top{topk}_agreement"

    # Overall correlation
    overall_r, _ = sp_stats.pearsonr(all_merged["error_rate_out"], all_merged["error_rate_in"])
    overall_rho, _ = sp_stats.spearmanr(all_merged["error_rate_out"], all_merged["error_rate_in"])

    # Sign test for performance delta
    deltas = stats_df["delta_mean"].dropna()
    n_positive = int((deltas > 0).sum())
    n_negative = int((deltas < 0).sum())

    # Sign test for layer_diff
    layer_diffs = stats_df["layer_diff"]
    n_layer_neg = int((layer_diffs < 0).sum())
    n_layer_pos = int((layer_diffs > 0).sum())
    n_layer_zero = int((layer_diffs == 0).sum())
    n_layer_nonzero = n_layer_neg + n_layer_pos
    if n_layer_nonzero > 0:
        layer_sign_test = sp_stats.binomtest(n_layer_pos, n_layer_nonzero, 0.5, alternative='less')
        layer_sign_p = float(layer_sign_test.pvalue)
    else:
        layer_sign_p = None

    # Wilcoxon signed-rank test on layer_diff
    layer_diffs_nonzero = layer_diffs[layer_diffs != 0]
    if len(layer_diffs_nonzero) >= 1:
        _, wilcoxon_p = sp_stats.wilcoxon(layer_diffs_nonzero, alternative='less')
        wilcoxon_p = float(wilcoxon_p)
    else:
        wilcoxon_p = None

    # Depth-delta Spearman
    depth_delta_rhos = stats_df["depth_delta_rho"].dropna()
    if len(depth_delta_rhos) >= 1:
        _, dd_wilcoxon_p = sp_stats.wilcoxon(depth_delta_rhos, alternative='greater')
        dd_wilcoxon_p = float(dd_wilcoxon_p)
    else:
        dd_wilcoxon_p = None

    summary = {
        "meta": {
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
            "partial_pearson_median": float(stats_df["partial_pearson"].median()) if not stats_df["partial_pearson"].isna().all() else None,
            "partial_spearman_median": float(stats_df["partial_spearman"].median()) if not stats_df["partial_spearman"].isna().all() else None,
            "overall_r": float(overall_r),
            "overall_rho": float(overall_rho),
        },
        "best_layer_agreement": {
            "exact_match_rate": float(stats_df["best_match"].mean()),
            f"top{topk}_agreement_rate": float(stats_df[topk_col].mean()),
            "within_1_rate": float(stats_df["within_1"].mean()),
            "within_2_rate": float(stats_df["within_2"].mean()),
            "layer_diff_median": float(stats_df["layer_diff"].median()),
            "layer_shift_n_deeper": n_layer_neg,
            "layer_shift_n_shallower": n_layer_pos,
            "layer_shift_n_same": n_layer_zero,
            "layer_shift_sign_test_p": layer_sign_p,
            "layer_shift_wilcoxon_p": wilcoxon_p,
            "depth_delta_rho_median": float(depth_delta_rhos.median()) if len(depth_delta_rhos) > 0 else None,
            "depth_delta_wilcoxon_p": dd_wilcoxon_p,
        },
        "performance_diff": {
            "delta_mean_median": float(stats_df["delta_mean"].median()),
            "delta_mean_ci": format_ci(bootstrap_ci(stats_df["delta_mean"], np.median, n_boot, rng)),
            "delta_at_out_best_median": float(stats_df["delta_at_out_best"].median()),
            "overall_mean_delta": float(all_merged["delta"].mean()),
            "overall_std_delta": float(all_merged["delta"].std()),
            "n_positive_delta": n_positive,
            "n_negative_delta": n_negative,
            "out_best_error_median": float(stats_df["out_best_error"].median()),
            "in_best_error_median": float(stats_df["in_best_error"].median()),
        },
    }

    output_path.write_text(json.dumps(summary, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compare in-context vs out-of-context ABX results."
    )
    parser.add_argument(
        "dataset",
        help="Dataset name (e.g., pitch_accent, stress_kokoro). "
             "Expects DATASET and DATASET_in_context directories in results/",
    )
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-baselines", action="store_true")
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Also write per-model statistics and diagnostic plots.",
    )
    args = parser.parse_args()

    dataset = args.dataset
    dataset_label = DATASET_LABELS.get(dataset, dataset.replace("_", " ").title())

    out_dir = RESULTS_DIR / dataset
    in_dir = RESULTS_DIR / f"{dataset}_in_context"
    output_dir = OUTPUT_BASE_DIR / f"{dataset}_in_context_analysis"

    if not out_dir.exists():
        raise SystemExit(f"Out-of-context results not found: {out_dir}")
    if not in_dir.exists():
        raise SystemExit(f"In-context results not found: {in_dir}")

    out_results = load_results(out_dir)
    in_results = load_results(in_dir)

    models = sorted(set(out_results.keys()) & set(in_results.keys()))
    missing = [m for m in models if m not in MODEL_METADATA]
    if missing:
        print(
            "Warning: models missing metadata, skipping: "
            + ", ".join(sorted(missing))
        )
    models = [m for m in models if m in MODEL_METADATA]
    if not args.include_baselines:
        models = [
            m for m in models
            if m not in BASELINE_MODELS
            and not (m in MODEL_METADATA and MODEL_METADATA[m][0] == "Baseline")
        ]

    print(f"Analyzing {dataset_label}: {len(models)} models...")

    stats_rows = []
    merged_rows = []

    for model in models:
        df_out = out_results[model]
        df_in = in_results[model]

        stats, merged = compute_per_model_stats(model, df_out, df_in, topk=args.topk)
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

    if args.diagnostics:
        stats_df.to_csv(output_dir / "per_model_stats.csv", index=False)
        print(f"Saved: {output_dir / 'per_model_stats.csv'}")

        plot_distributions(stats_df, output_dir / "distributions.png")
        print(f"Saved: {output_dir / 'distributions.png'}")

        plot_layerwise_comparison(all_merged, output_dir / "layerwise_comparison.png")
        print(f"Saved: {output_dir / 'layerwise_comparison.png'}")

        plot_delta_by_layer(all_merged, output_dir / "delta_by_layer.png")
        print(f"Saved: {output_dir / 'delta_by_layer.png'}")

        plot_correlation_scatter(all_merged, output_dir / "correlation_scatter.png")
        print(f"Saved: {output_dir / 'correlation_scatter.png'}")

        plot_model_rank_bump(stats_df, output_dir / "model_rank_bump.png")
        print(f"Saved: {output_dir / 'model_rank_bump.png'}")

    # Write summaries
    write_summary(stats_df, all_merged, args.bootstrap, rng, output_dir / "summary.md", dataset_label)
    print(f"Saved: {output_dir / 'summary.md'}")

    write_summary_json(stats_df, all_merged, args.bootstrap, rng, output_dir / "summary.json")
    print(f"Saved: {output_dir / 'summary.json'}")

    # Print key results
    print("\n" + "=" * 60)
    print(f"KEY RESULTS: {dataset_label}")
    print("=" * 60)
    print(f"\nShape similarity (median):")
    print(f"  Layer-wise Pearson r:  {stats_df['pearson_r'].median():.3f}")
    print(f"  Layer-wise Spearman ρ: {stats_df['spearman_r'].median():.3f}")
    print(f"  Slope Pearson r:       {stats_df['slope_pearson'].median():.3f}")
    print(f"  Partial Pearson r:    {stats_df['partial_pearson'].median():.3f}  (depth removed)")
    print(f"  Partial Spearman ρ:   {stats_df['partial_spearman'].median():.3f}  (depth removed)")

    print(f"\nBest layer agreement:")
    print(f"  Exact match:    {stats_df['best_match'].mean():.1%}")
    print(f"  Within ±1:      {stats_df['within_1'].mean():.1%}")
    print(f"  Within ±2:      {stats_df['within_2'].mean():.1%}")

    # Layer shift tests
    layer_diffs = stats_df["layer_diff"]
    n_layer_neg = (layer_diffs < 0).sum()
    n_layer_pos = (layer_diffs > 0).sum()
    n_layer_zero = (layer_diffs == 0).sum()
    n_layer_nonzero = n_layer_neg + n_layer_pos
    if n_layer_nonzero > 0:
        layer_sign_p = sp_stats.binomtest(n_layer_pos, n_layer_nonzero, 0.5, alternative='less').pvalue
        layer_diffs_nonzero = layer_diffs[layer_diffs != 0]
        _, wilcoxon_p = sp_stats.wilcoxon(layer_diffs_nonzero, alternative='less')
        print(f"\nLayer shift (H1: in-context prefers deeper):")
        print(f"  {n_layer_neg}/{n_layer_nonzero} prefer deeper, {n_layer_pos}/{n_layer_nonzero} shallower, {n_layer_zero} same")
        print(f"  Sign test p-value:    {layer_sign_p:.4f}")
        print(f"  Wilcoxon SR p-value:  {wilcoxon_p:.4f}")

    # Depth-delta correlation
    depth_delta_rhos = stats_df["depth_delta_rho"].dropna()
    if len(depth_delta_rhos) >= 1:
        _, dd_wilcoxon_p = sp_stats.wilcoxon(depth_delta_rhos, alternative='greater')
        print(f"\nDepth-delta correlation (H1: in-context advantage grows with depth):")
        print(f"  Median ρ(depth, Δ):   {depth_delta_rhos.median():.3f}")
        print(f"  Wilcoxon SR p-value:  {dd_wilcoxon_p:.4f}")

    print(f"\nPerformance difference (out − in):")
    print(f"  Median Δ per model: {stats_df['delta_mean'].median():.4f}")
    print(f"  Overall mean Δ:     {all_merged['delta'].mean():.4f}")

    n_pos = (stats_df["delta_mean"] > 0).sum()
    n_neg = (stats_df["delta_mean"] < 0).sum()
    print(f"  Direction: {n_pos} models worse out-of-context, {n_neg} better")


if __name__ == "__main__":
    main()
