#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare ranking consistency between two tasks using Spearman correlation.

For each bootstrap iteration, this script computes the Spearman rank
correlation between model rankings obtained under two different tasks
(e.g., natural vs synthetic speech).

Input:
    /home/sunhaitong/fastabx/bootstrapping/<language>/<task>/abx_ranking.jsonl

Output:
    /home/sunhaitong/fastabx/bootstrapping/<language>/cross_corpus_ranking_corr.txt
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
import math
import statistics


# --------------------------------------------------
# Load ranking JSONL
# --------------------------------------------------
def load_rankings(jsonl_path: Path) -> dict[int, dict[str, int]]:
    """
    Load rankings from a JSONL file.

    Returns:
        bootstrap_id -> { model_name -> rank }
    """
    rankings = {}
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            b = obj["bootstrap_id"]
            rank_dict = {model: rank for model, rank in obj["ranking"]}
            rankings[b] = rank_dict
    return rankings


# --------------------------------------------------
# Spearman correlation (no scipy dependency)
# --------------------------------------------------
def spearman_corr(x: list[float], y: list[float]) -> float:
    """
    Compute Spearman rank correlation between two vectors.

    Assumes x and y have the same length >= 2.
    """
    n = len(x)
    if n < 2:
        return float("nan")

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    num = 0.0
    den_x = 0.0
    den_y = 0.0

    for xi, yi in zip(x, y):
        dx = xi - mean_x
        dy = yi - mean_y
        num += dx * dy
        den_x += dx * dx
        den_y += dy * dy

    if den_x == 0 or den_y == 0:
        return float("nan")

    return num / math.sqrt(den_x * den_y)


# --------------------------------------------------
# Percentile helper
# --------------------------------------------------
def percentile(sorted_vals: list[float], p: float) -> float:
    """
    Compute percentile p (0-100) from a sorted list using linear interpolation.
    """
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]

    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)

    if f == c:
        return sorted_vals[f]

    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


# --------------------------------------------------
# Main
# --------------------------------------------------
def main(language: str, task1: str, task2: str) -> None:
    base = Path("/home/sunhaitong/fastabx/bootstrapping") / language

    path1 = base / task1 / "abx_ranking.jsonl"
    path2 = base / task2 / "abx_ranking.jsonl"

    if not path1.exists():
        raise FileNotFoundError(f"Missing ranking file: {path1}")
    if not path2.exists():
        raise FileNotFoundError(f"Missing ranking file: {path2}")

    ranks1 = load_rankings(path1)
    ranks2 = load_rankings(path2)

    # --------------------------------------------------
    # Compute Spearman correlation per bootstrap
    # --------------------------------------------------
    corrs = []

    common_bootstraps = sorted(set(ranks1.keys()) & set(ranks2.keys()))

    for b in common_bootstraps:
        r1 = ranks1[b]
        r2 = ranks2[b]

        # Only compare models that appear in BOTH rankings
        models = sorted(set(r1.keys()) & set(r2.keys()))
        if len(models) < 2:
            continue

        x = [r1[m] for m in models]
        y = [r2[m] for m in models]

        rho = spearman_corr(x, y)
        if not math.isnan(rho):
            corrs.append(rho)

    if not corrs:
        raise RuntimeError("No valid Spearman correlations computed.")

    corrs_sorted = sorted(corrs)

    mean_rho = statistics.mean(corrs_sorted)
    p2_5 = percentile(corrs_sorted, 2.5)
    p97_5 = percentile(corrs_sorted, 97.5)

    # --------------------------------------------------
    # Write output
    # --------------------------------------------------
    out_path = base / "cross_corpus_ranking_corr.txt"
    with out_path.open("w", encoding="utf-8") as f:
        f.write("=== CROSS-CORPUS RANKING CONSISTENCY ===\n\n")
        f.write(f"Language: {language}\n")
        f.write(f"Task 1 : {task1}\n")
        f.write(f"Task 2 : {task2}\n\n")
        f.write(f"Bootstrap samples: {len(corrs_sorted)}\n\n")
        f.write(
            f"Spearman rho (mean)          : {mean_rho:.4f}\n"
        )
        f.write(
            f"Spearman rho [2.5%, 97.5%]   : [{p2_5:.4f}, {p97_5:.4f}]\n"
        )

    print(f"[DONE] Wrote ranking correlation to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("language")
    parser.add_argument("task1")
    parser.add_argument("task2")
    args = parser.parse_args()

    main(args.language, args.task1, args.task2)
