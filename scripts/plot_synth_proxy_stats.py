#!/usr/bin/env python3
"""
Compute selection-quality stats between natural and synthetic layerwise ABX results.

Usage:
    python plot_synth_proxy_stats.py NATURAL_DATASET SYNTH_DATASET [--topk 2] [--epsilon 0.01]

Outputs:
    - per_model_stats.csv
    - distributions.png (boxplots)
    - summary.md (tables of layer-wise stats and global stats)
    - summary.json (structured stats for table formatting)
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
OUTPUT_DIR = Path(__file__).parent.parent / "plots" / "synth_proxy_stats"


BASELINE_MODELS = {"mfcc", "fbank"}


def _filter_mode(df, mode, dataset, model_name):
    if "mode" not in df.columns:
        return df
    modes = df["mode"].dropna().unique()
    if mode is None:
        if len(modes) > 1:
            raise ValueError(
                f"{dataset}/{model_name} has multiple modes {list(modes)}; use --mode to choose one."
            )
        return df
    return df[df["mode"] == mode].copy()


def _best_layer_info(df, error_col):
    min_err = df[error_col].min()
    min_layers = df.loc[df[error_col] == min_err, "layer_num"].tolist()
    best_layer = min(min_layers)
    return min_err, min_layers, best_layer


def _compute_model_summary(global_df):
    def _summ(group):
        nat_best = group["error_rate_nat"].min()
        mean_nat = group["error_rate_nat"].mean()
        syn_best_row = group.sort_values(["error_rate_syn", "layer_num"]).iloc[0]
        return pd.Series(
            {
                "nat_best": nat_best,
                "syn_best": syn_best_row["error_rate_syn"],
                "nat_at_syn_best": syn_best_row["error_rate_nat"],
                "mean_nat": mean_nat,
            }
        )

    return global_df.groupby("model").apply(_summ)


def compute_per_model_stats(
    model_name,
    df_nat,
    df_syn,
    topk,
    epsilon,
):
    merged = pd.merge(df_nat, df_syn, on="layer", suffixes=("_nat", "_syn"))
    if merged.empty:
        return None, None
    merged["layer_num"] = merged["layer"].map(get_layer_num)
    merged.sort_values("layer_num", inplace=True)

    nat_min_err, nat_min_layers, nat_best_layer = _best_layer_info(merged, "error_rate_nat")
    _, _, syn_best_layer = _best_layer_info(merged, "error_rate_syn")

    best_layer_match = int(syn_best_layer in nat_min_layers)
    layer_diff = syn_best_layer - nat_best_layer  # positive = synth prefers deeper

    topk_layers = (
        merged.nsmallest(topk, ["error_rate_nat", "layer_num"])["layer_num"].tolist()
    )
    topk_agreement = int(syn_best_layer in topk_layers)

    nat_error_at_syn = merged.loc[
        merged["layer_num"] == syn_best_layer, "error_rate_nat"
    ].iloc[0]
    regret = nat_error_at_syn - nat_min_err

    near_optimal = int(regret <= epsilon) if epsilon is not None else np.nan

    spearman = (
        merged["error_rate_nat"].corr(merged["error_rate_syn"], method="spearman")
        if len(merged) >= 2
        else np.nan
    )
    pearson = (
        merged["error_rate_nat"].corr(merged["error_rate_syn"], method="pearson")
        if len(merged) >= 2
        else np.nan
    )

    stats = {
        "model": model_name,
        "best_layer_match": best_layer_match,
        f"top{topk}_agreement": topk_agreement,
        "regret": regret,
        "near_optimal": near_optimal,
        "spearman": spearman,
        "pearson": pearson,
        "layer_diff": layer_diff,
        "n_layers": len(merged),
        "nat_best_layer": nat_best_layer,
        "syn_best_layer": syn_best_layer,
    }

    merged["model"] = model_name
    return stats, merged[["model", "layer", "layer_num", "error_rate_nat", "error_rate_syn"]]


def plot_model_rank_bump(model_summary, model_rank_rho, output_path):
    """Bump chart showing how model rankings change between natural and synthetic."""
    df = model_summary[["nat_best", "syn_best"]].copy()
    df["rank_nat"] = df["nat_best"].rank(method="min")
    df["rank_syn"] = df["syn_best"].rank(method="min")

    n = len(df)
    fig, ax = plt.subplots(figsize=(6, max(4, n * 0.35)))

    for model, row in df.iterrows():
        r_nat = row["rank_nat"]
        r_syn = row["rank_syn"]
        shift = abs(r_nat - r_syn)
        color = "tab:red" if shift >= 3 else ("tab:orange" if shift >= 1 else "tab:gray")
        alpha = 1.0 if shift >= 1 else 0.5
        ax.plot([0, 1], [r_nat, r_syn], "o-", color=color, alpha=alpha,
                markersize=5, linewidth=1.5)
        ax.text(-0.05, r_nat, model, ha="right", va="center", fontsize=6)
        ax.text(1.05, r_syn, model, ha="left", va="center", fontsize=6)

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(n + 0.5, 0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Natural", "Synthetic"])
    ax.set_ylabel("Rank (1 = best)")
    ax.set_title("Model Ranking: Natural vs Synthetic")
    ax.grid(True, axis="y", alpha=0.2)

    if model_rank_rho is not None:
        ax.text(0.5, n + 0.4, f"Spearman ρ = {model_rank_rho:.3f}", ha="center",
                fontsize=8, style="italic", transform=ax.get_yaxis_transform())

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_distributions(stats_df, global_regret, model_uniform_global_regret, within_model_random_regret, output_path):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Regret boxplot
    regret = stats_df["regret"].dropna()
    axes[0].boxplot(regret, vert=True, patch_artist=True, boxprops=dict(facecolor="tab:blue", alpha=0.6))
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_title("Regret (natural error points)")
    axes[0].set_ylabel("Regret")
    axes[0].set_xticks([])

    # Spearman boxplot
    spearman = stats_df["spearman"].dropna()
    axes[1].boxplot(
        spearman, vert=True, patch_artist=True, boxprops=dict(facecolor="tab:orange", alpha=0.6)
    )
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_title("Spearman Layerwise Corr")
    axes[1].set_ylabel("Spearman ρ")
    axes[1].set_xticks([])

    # Pearson boxplot
    pearson = stats_df["pearson"].dropna()
    axes[2].boxplot(
        pearson, vert=True, patch_artist=True, boxprops=dict(facecolor="tab:green", alpha=0.6)
    )
    axes[2].axhline(0, color="black", linewidth=1)
    axes[2].set_title("Pearson Layerwise Corr")
    axes[2].set_ylabel("Pearson r")
    axes[2].set_xticks([])

    # Layer diff boxplot
    layer_diff = stats_df["layer_diff"].dropna()
    axes[3].boxplot(
        layer_diff, vert=True, patch_artist=True, boxprops=dict(facecolor="tab:purple", alpha=0.6)
    )
    axes[3].axhline(0, color="black", linewidth=1)
    axes[3].set_title("Δ Layer (syn − nat)")
    axes[3].set_ylabel("Δ Layer")
    axes[3].set_xticks([])

    if (
        global_regret is not None
        or model_uniform_global_regret is not None
        or within_model_random_regret is not None
    ):
        text_bits = []
        if global_regret is not None:
            text_bits.append(f"Global regret: {global_regret:.4f}")
        if model_uniform_global_regret is not None:
            text_bits.append(f"Model-uniform global: {model_uniform_global_regret:.4f}")
        if within_model_random_regret is not None:
            text_bits.append(f"Within-model random: {within_model_random_regret:.4f}")
        fig.suptitle(" | ".join(text_bits), fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _median_iqr(series):
    series = series.dropna()
    if series.empty:
        return "N/A"
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    median = series.median()
    return f"{median:.4f} [{q1:.4f}, {q3:.4f}]"


def _bootstrap_ci(values, stat_fn, n_boot, rng):
    if n_boot <= 0:
        return None
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size < 2:
        return None
    boot = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(arr, size=arr.size, replace=True)
        boot[i] = stat_fn(sample)
    return np.quantile(boot, [0.025, 0.975])


def _format_ci(ci):
    if ci is None or np.any(np.isnan(ci)):
        return "N/A"
    return f"[{ci[0]:.4f}, {ci[1]:.4f}]"


def _bootstrap_global_cis(model_summary, n_boot, rng):
    if n_boot <= 0 or model_summary.empty or len(model_summary) < 2:
        return {}
    nat_best = model_summary["nat_best"].to_numpy()
    syn_best = model_summary["syn_best"].to_numpy()
    nat_at_syn_best = model_summary["nat_at_syn_best"].to_numpy()
    mean_nat = model_summary["mean_nat"].to_numpy()
    n = nat_best.size
    boot_global = np.empty(n_boot)
    boot_model_uniform = np.empty(n_boot)
    boot_delta = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample_nat_best = nat_best[idx]
        sample_syn_best = syn_best[idx]
        sample_nat_at_syn = nat_at_syn_best[idx]
        sample_mean_nat = mean_nat[idx]
        global_nat_best = sample_nat_best.min()
        syn_argmin = int(np.argmin(sample_syn_best))
        boot_global[i] = sample_nat_at_syn[syn_argmin] - global_nat_best
        boot_model_uniform[i] = sample_mean_nat.mean() - global_nat_best
        boot_delta[i] = boot_model_uniform[i] - boot_global[i]
    return {
        "model_uniform_global_regret": np.quantile(boot_model_uniform, [0.025, 0.975]),
        "model_uniform_minus_global": np.quantile(boot_delta, [0.025, 0.975]),
    }


def _global_percentile(errors, chosen_value):
    errors = np.asarray(errors, dtype=float)
    return 100.0 * np.mean(errors >= chosen_value)


def _bootstrap_percentile_ci(global_df, n_boot, rng):
    if n_boot <= 0:
        return None
    groups = {model: df for model, df in global_df.groupby("model")}
    models = list(groups)
    if len(models) < 2:
        return None
    boot = np.empty(n_boot)
    for i in range(n_boot):
        sampled = rng.choice(models, size=len(models), replace=True)
        sample_df = pd.concat([groups[m] for m in sampled], ignore_index=True)
        syn_best_row = sample_df.sort_values(["error_rate_syn", "model", "layer_num"]).iloc[0]
        chosen_nat = float(syn_best_row["error_rate_nat"])
        boot[i] = _global_percentile(sample_df["error_rate_nat"].to_numpy(), chosen_nat)
    return np.quantile(boot, [0.025, 0.975])


def write_summary_table(
    stats_df,
    topk,
    epsilon,
    global_regret,
    model_uniform_global_regret,
    within_model_random_regret,
    global_percentile,
    global_percentile_ci,
    model_summary,
    per_model_random_regret,
    model_rank_rho,
    model_rank_rho_ci,
    n_boot,
    rng,
    output_path,
):
    total = len(stats_df)
    best_rate = stats_df["best_layer_match"].mean() if total else np.nan
    topk_col = f"top{topk}_agreement"
    topk_rate = stats_df[topk_col].mean() if total else np.nan
    near_rate = stats_df["near_optimal"].mean() if total else np.nan

    cis = {
        "regret": _bootstrap_ci(stats_df["regret"], np.median, n_boot, rng),
        "spearman": _bootstrap_ci(stats_df["spearman"], np.median, n_boot, rng),
        "pearson": _bootstrap_ci(stats_df["pearson"], np.median, n_boot, rng),
        "layer_diff": _bootstrap_ci(
            stats_df["layer_diff"], np.median, n_boot, rng
        ),
        "best_layer_match": _bootstrap_ci(stats_df["best_layer_match"], np.mean, n_boot, rng),
        f"top{topk}_agreement": _bootstrap_ci(stats_df[topk_col], np.mean, n_boot, rng),
    }
    if epsilon is not None:
        cis["near_optimal"] = _bootstrap_ci(stats_df["near_optimal"], np.mean, n_boot, rng)
    if per_model_random_regret is not None:
        cis["within_model_random_regret"] = _bootstrap_ci(
            per_model_random_regret.values, np.mean, n_boot, rng
        )

    if model_summary is None:
        model_summary = pd.DataFrame()
    global_cis = _bootstrap_global_cis(model_summary, n_boot, rng)

    lines = [
        "### Layer-wise stats (across models)",
        "| Metric | Value | 95% CI |",
        "| --- | --- | --- |",
        f"| Models | {total} |",
        f"| Regret median [IQR] | {_median_iqr(stats_df['regret'])} | {_format_ci(cis['regret'])} |",
        f"| Spearman median [IQR] | {_median_iqr(stats_df['spearman'])} | {_format_ci(cis['spearman'])} |",
        f"| Pearson median [IQR] | {_median_iqr(stats_df['pearson'])} | {_format_ci(cis['pearson'])} |",
        f"| Δ layer median [IQR] | {_median_iqr(stats_df['layer_diff'])} | {_format_ci(cis['layer_diff'])} |",
        f"| Best-layer match rate | {best_rate:.3f} | {_format_ci(cis['best_layer_match'])} |",
        f"| Top-{topk} agreement rate | {topk_rate:.3f} | {_format_ci(cis[f'top{topk}_agreement'])} |",
    ]
    if epsilon is not None:
        lines.append(
            f"| Near-optimal (≤ {epsilon:.3f}) rate | {near_rate:.3f} | {_format_ci(cis.get('near_optimal'))} |"
        )
    if within_model_random_regret is not None:
        lines.append(
            f"| Within-model random regret | {within_model_random_regret:.4f} | {_format_ci(cis.get('within_model_random_regret'))} |"
        )
    lines.extend(
        [
            "",
            "### Global stats (model + layer selection)",
            "| Metric | Value | 95% CI |",
            "| --- | --- | --- |",
        ]
    )
    if global_regret is not None:
        lines.append(
            f"| Global regret | {global_regret:.4f} | N/A |"
        )
    if model_uniform_global_regret is not None:
        lines.append(
            f"| Model-uniform global regret | {model_uniform_global_regret:.4f} | {_format_ci(global_cis.get('model_uniform_global_regret'))} |"
        )
    if (
        model_uniform_global_regret is not None
        and global_regret is not None
        and global_cis.get("model_uniform_minus_global") is not None
    ):
        diff = model_uniform_global_regret - global_regret
        lines.append(
            f"| Model-uniform − global regret | {diff:.4f} | {_format_ci(global_cis.get('model_uniform_minus_global'))} |"
        )
    if global_percentile is not None:
        lines.append(
            f"| Synth-chosen percentile (natural errors) | {global_percentile:.1f} | {_format_ci(global_percentile_ci)} |"
        )
    if model_rank_rho is not None:
        lines.append(
            f"| Model ranking Spearman ρ | {model_rank_rho:.4f} | {_format_ci(model_rank_rho_ci)} |"
        )

    output_path.write_text("\n".join(lines) + "\n")


def write_summary_json(
    stats_df,
    topk,
    epsilon,
    global_regret,
    model_uniform_global_regret,
    within_model_random_regret,
    global_percentile,
    global_percentile_ci,
    model_summary,
    per_model_random_regret,
    model_rank_rho,
    model_rank_rho_ci,
    n_boot,
    rng,
    output_path,
):
    topk_col = f"top{topk}_agreement"
    summary = {
        "meta": {
            "models": int(len(stats_df)),
            "topk": int(topk),
            "epsilon": None if epsilon is None else float(epsilon),
            "bootstrap": int(n_boot),
        },
        "layerwise": {
            "regret_median_iqr": _median_iqr(stats_df["regret"]),
            "regret_ci": _format_ci(_bootstrap_ci(stats_df["regret"], np.median, n_boot, rng)),
            "spearman_median_iqr": _median_iqr(stats_df["spearman"]),
            "spearman_ci": _format_ci(_bootstrap_ci(stats_df["spearman"], np.median, n_boot, rng)),
            "pearson_median_iqr": _median_iqr(stats_df["pearson"]),
            "pearson_ci": _format_ci(_bootstrap_ci(stats_df["pearson"], np.median, n_boot, rng)),
            "delta_layer_median_iqr": _median_iqr(stats_df["layer_diff"]),
            "delta_layer_ci": _format_ci(
                _bootstrap_ci(stats_df["layer_diff"], np.median, n_boot, rng)
            ),
            "best_layer_match_rate": float(stats_df["best_layer_match"].mean()),
            "best_layer_match_ci": _format_ci(
                _bootstrap_ci(stats_df["best_layer_match"], np.mean, n_boot, rng)
            ),
            f"top{topk}_agreement_rate": float(stats_df[topk_col].mean()),
            f"top{topk}_agreement_ci": _format_ci(
                _bootstrap_ci(stats_df[topk_col], np.mean, n_boot, rng)
            ),
            "near_optimal_rate": None if epsilon is None else float(stats_df["near_optimal"].mean()),
            "near_optimal_ci": None
            if epsilon is None
            else _format_ci(_bootstrap_ci(stats_df["near_optimal"], np.mean, n_boot, rng)),
            "within_model_random_regret": None
            if within_model_random_regret is None
            else float(within_model_random_regret),
            "within_model_random_regret_ci": None
            if per_model_random_regret is None
            else _format_ci(
                _bootstrap_ci(per_model_random_regret.values, np.mean, n_boot, rng)
            ),
        },
        "global": {
            "global_regret": None if global_regret is None else float(global_regret),
            "model_uniform_global_regret": None
            if model_uniform_global_regret is None
            else float(model_uniform_global_regret),
            "model_uniform_minus_global": None
            if model_uniform_global_regret is None or global_regret is None
            else float(model_uniform_global_regret - global_regret),
            "model_uniform_minus_global_ci": _format_ci(
                _bootstrap_global_cis(
                    model_summary if model_summary is not None else pd.DataFrame(),
                    n_boot,
                    rng,
                ).get("model_uniform_minus_global")
            ),
            "model_uniform_global_regret_ci": _format_ci(
                _bootstrap_global_cis(
                    model_summary if model_summary is not None else pd.DataFrame(),
                    n_boot,
                    rng,
                ).get("model_uniform_global_regret")
            ),
            "global_percentile": None if global_percentile is None else float(global_percentile),
            "global_percentile_ci": _format_ci(global_percentile_ci),
            "model_rank_rho": model_rank_rho,
            "model_rank_rho_ci": _format_ci(model_rank_rho_ci),
        },
    }
    output_path.write_text(json.dumps(summary, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("natural_dataset", help="Natural speech dataset name")
    parser.add_argument("synth_dataset", help="Synthetic speech dataset name")
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--mode", type=str, default=None)
    parser.add_argument("--include-baselines", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Also write per-model statistics and diagnostic plots.",
    )
    args = parser.parse_args()

    nat_dir = RESULTS_DIR / args.natural_dataset
    syn_dir = RESULTS_DIR / args.synth_dataset

    if not nat_dir.exists():
        raise SystemExit(f"Natural dataset not found: {args.natural_dataset}")
    if not syn_dir.exists():
        raise SystemExit(f"Synthetic dataset not found: {args.synth_dataset}")

    nat_results = load_results(nat_dir)
    syn_results = load_results(syn_dir)

    models = sorted(set(nat_results.keys()) & set(syn_results.keys()))
    missing = [m for m in models if m not in MODEL_METADATA]
    if missing:
        print(
            "Warning: models missing metadata, skipping: "
            + ", ".join(sorted(missing))
        )
    models = [m for m in models if m in MODEL_METADATA]
    if not args.include_baselines:
        models = [
            m
            for m in models
            if m not in BASELINE_MODELS
            and not (m in MODEL_METADATA and MODEL_METADATA[m][0] == "Baseline")
        ]

    stats_rows = []
    merged_rows = []

    for model in models:
        df_nat = _filter_mode(nat_results[model], args.mode, args.natural_dataset, model)
        df_syn = _filter_mode(syn_results[model], args.mode, args.synth_dataset, model)

        stats, merged = compute_per_model_stats(
            model, df_nat, df_syn, args.topk, args.epsilon
        )
        if stats is None:
            continue
        stats_rows.append(stats)
        merged_rows.append(merged)

    if not stats_rows:
        raise SystemExit("No overlapping models/layers found after filtering.")

    stats_df = pd.DataFrame(stats_rows)
    rng = np.random.default_rng(args.seed)

    # Global regret across all model-layer points
    global_regret = None
    model_uniform_global_regret = None
    within_model_random_regret = None
    per_model_random_regret = None
    model_summary = None
    global_percentile = None
    global_percentile_ci = None
    if merged_rows:
        global_df = pd.concat(merged_rows, ignore_index=True)
        model_summary = _compute_model_summary(global_df)
        global_nat_best = global_df["error_rate_nat"].min()
        global_syn_best_row = (
            global_df.sort_values(["error_rate_syn", "model", "layer_num"]).iloc[0]
        )
        global_regret = global_syn_best_row["error_rate_nat"] - global_nat_best
        global_percentile = _global_percentile(
            global_df["error_rate_nat"].to_numpy(),
            float(global_syn_best_row["error_rate_nat"]),
        )
        global_percentile_ci = _bootstrap_percentile_ci(global_df, args.bootstrap, rng)
        model_means = global_df.groupby("model")["error_rate_nat"].mean()
        model_uniform_global_regret = model_means.mean() - global_nat_best
        per_model_random_regret = global_df.groupby("model").apply(
            lambda df: df["error_rate_nat"].mean() - df["error_rate_nat"].min()
        )
        within_model_random_regret = per_model_random_regret.mean()

    # Model ranking Spearman ρ (best-layer error per model)
    model_rank_rho = None
    model_rank_rho_ci = None
    if model_summary is not None and len(model_summary) >= 3:
        model_rank_rho, _ = sp_stats.spearmanr(
            model_summary["nat_best"], model_summary["syn_best"]
        )
        model_rank_rho = float(model_rank_rho)
        # Bootstrap CI on model ranking ρ
        if args.bootstrap > 0 and len(model_summary) >= 2:
            nat_vals = model_summary["nat_best"].to_numpy()
            syn_vals = model_summary["syn_best"].to_numpy()
            n = nat_vals.size
            boot_rhos = np.empty(args.bootstrap)
            for i in range(args.bootstrap):
                idx = rng.integers(0, n, size=n)
                boot_rhos[i], _ = sp_stats.spearmanr(nat_vals[idx], syn_vals[idx])
            model_rank_rho_ci = np.quantile(boot_rhos, [0.025, 0.975])

    out_dir = args.output_dir or (OUTPUT_DIR / f"{args.natural_dataset}_vs_{args.synth_dataset}")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.diagnostics:
        stats_path = out_dir / "per_model_stats.csv"
        stats_df.to_csv(stats_path, index=False)
        print(f"Saved: {stats_path}")

        plot_path = out_dir / "distributions.png"
        plot_distributions(
            stats_df,
            global_regret,
            model_uniform_global_regret,
            within_model_random_regret,
            plot_path,
        )
        print(f"Saved: {plot_path}")

        if model_summary is not None and len(model_summary) >= 2:
            bump_path = out_dir / "model_rank_bump.png"
            plot_model_rank_bump(model_summary, model_rank_rho, bump_path)
            print(f"Saved: {bump_path}")

    summary_path = out_dir / "summary.md"
    write_summary_table(
        stats_df,
        args.topk,
        args.epsilon,
        global_regret,
        model_uniform_global_regret,
        within_model_random_regret,
        global_percentile,
        global_percentile_ci,
        model_summary,
        per_model_random_regret,
        model_rank_rho,
        model_rank_rho_ci,
        args.bootstrap,
        rng,
        summary_path,
    )

    summary_json_path = out_dir / "summary.json"
    write_summary_json(
        stats_df,
        args.topk,
        args.epsilon,
        global_regret,
        model_uniform_global_regret,
        within_model_random_regret,
        global_percentile,
        global_percentile_ci,
        model_summary,
        per_model_random_regret,
        model_rank_rho,
        model_rank_rho_ci,
        args.bootstrap,
        rng,
        summary_json_path,
    )

    print(f"Saved: {summary_path}")
    print(f"Saved: {summary_json_path}")
    if global_regret is not None:
        print(f"Global regret: {global_regret:.4f}")
    if model_uniform_global_regret is not None:
        print(f"Model-uniform global regret: {model_uniform_global_regret:.4f}")
    if within_model_random_regret is not None:
        print(f"Within-model random regret: {within_model_random_regret:.4f}")
    if model_rank_rho is not None:
        print(f"Model ranking Spearman ρ: {model_rank_rho:.4f} {_format_ci(model_rank_rho_ci)}")


if __name__ == "__main__":
    main()
