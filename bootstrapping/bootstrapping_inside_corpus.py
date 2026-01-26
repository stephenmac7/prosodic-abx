#!/usr/bin/env python3
"""
Item-bootstrap reweighting of cell-level ABX scores inside a corpus.

For a given experiment (exp_name), this script:
- loads item-level bootstrap samples
- loads cell-level ABX CSVs for each representation and layer
- reweights ABX scores using item bootstrap counts
- selects the best layer per representation (oracle over layers)
- computes bootstrap distributions of error rates and rankings
"""

import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, Tuple, List
import math

import numpy as np
from scipy.stats import spearmanr


FASTABX_ROOT = Path("/home/sunhaitong/fastabx")

# --------------------------------------------------
# Pretty print settings
# --------------------------------------------------
MODEL_NAME_WIDTH = 48
NUM_WIDTH = 12
NUM_FMT = "{:>10.4f}"   


ItemKey = Tuple[str, str, str]  # (phone_sequence, accent_pattern, speaker)


def key_to_str(k: ItemKey) -> str:
    return f"{k[0]}|{k[1]}|{k[2]}"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("exp_name", help="e.g. pitch_accent_syn")
    return p.parse_args()


# --------------------------------------------------
# Load item bootstrap samples
# --------------------------------------------------
def load_item_bootstrap(exp_name: str) -> List[Dict[str, int]]:
    path = FASTABX_ROOT / "bootstrapping" / exp_name / "bootstrap_item_500.jsonl"
    samples = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            samples.append(obj["counts"])

    return samples


# --------------------------------------------------
# Load cell-level CSVs
# --------------------------------------------------
def load_cells(csv_path: Path):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    args = parse_args()
    exp_name = args.exp_name

    # Paths
    cells_root = FASTABX_ROOT / "results" / exp_name / "cells"
    bootstrap_root = FASTABX_ROOT / "bootstrapping" / exp_name
    bootstrap_root.mkdir(parents=True, exist_ok=True)

    # Load item bootstrap samples
    item_bootstraps = load_item_bootstrap(exp_name)
    n_boot = len(item_bootstraps)

    # Discover representations
    representations = sorted(
        d.name for d in cells_root.iterdir() if d.is_dir()
    )

    # --------------------------------------------------
    # Storage
    # --------------------------------------------------
    # error_rates[rep][b] = error rate at bootstrap b (best layer)
    error_rates = defaultdict(list)

    # rankings[b] = list of reps sorted by error rate
    rankings = []

    # --------------------------------------------------
    # Bootstrap loop
    # --------------------------------------------------
    for b, item_counts in enumerate(item_bootstraps):

        rep_to_best_error = {}

        for rep in representations:
            rep_dir = cells_root / rep
            layer_errors = {}

            for csv_path in rep_dir.glob("*.csv"):
                score_sum = 0.0
                count_sum = 0.0

                rows = load_cells(csv_path)

                for row in rows:
                    phone = row["phone_sequence"]

                    A = (phone, row["accent_pattern"], row["speaker"])
                    B = (phone, row["accent_pattern_b"], row["speaker"])
                    X = (phone, row["accent_pattern"], row["speaker_x"])

                    wA = item_counts.get(key_to_str(A), 0)
                    wB = item_counts.get(key_to_str(B), 0)
                    wX = item_counts.get(key_to_str(X), 0)

                    weight = wA * wB * wX
                    if weight == 0:
                        continue

                    score = float(row["score"])
                    size = float(row["size"])

                    score_sum += score * weight * size
                    count_sum += size * weight

                if count_sum > 0:
                    layer_errors[csv_path.stem] = score_sum / count_sum

            if not layer_errors:
                continue

            # Select best layer (oracle)
            best_layer = min(layer_errors, key=layer_errors.get)
            rep_to_best_error[rep] = layer_errors[best_layer]

        # Store error rates
        for rep, err in rep_to_best_error.items():
            error_rates[rep].append(err)

        # Ranking for this bootstrap
        ranked = sorted(rep_to_best_error.items(), key=lambda x: x[1])
        rankings.append([r for r, _ in ranked])

    # --------------------------------------------------
    # Compute summary statistics
    # --------------------------------------------------
    summary = {}
    for rep, vals in error_rates.items():
        arr = np.array(vals)
        summary[rep] = {
            "mean_error_rate": float(arr.mean()),
            "ci_2.5": float(np.percentile(arr, 2.5)),
            "ci_97.5": float(np.percentile(arr, 97.5)),
        }

    # Reference ranking (mean error)
    ref_ranking = sorted(summary.items(), key=lambda x: x[1]["mean_error_rate"])
    ref_order = [r for r, _ in ref_ranking]

    # Spearman correlations
    spearmans = []
    for r in rankings:
        # convert to rank vectors
        rank_ref = {rep: i for i, rep in enumerate(ref_order)}
        rank_r = {rep: i for i, rep in enumerate(r)}

        common = [rep for rep in ref_order if rep in rank_r]
        v1 = [rank_ref[rep] for rep in common]
        v2 = [rank_r[rep] for rep in common]

        if len(common) > 1:
            rho, _ = spearmanr(v1, v2)
            spearmans.append(rho)

    # --------------------------------------------------
    # Save outputs
    # --------------------------------------------------
    with open(bootstrap_root / "inside_corpus_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(bootstrap_root / "inside_corpus_rankings.jsonl", "w") as f:
        for r in rankings:
            f.write(json.dumps(r) + "\n")

    meta = {
        "exp_name": exp_name,
        "n_bootstrap": n_boot,
        "n_representations": len(representations),
        "mean_spearman": float(np.mean(spearmans)),
        "ci_spearman": [
            float(np.percentile(spearmans, 2.5)),
            float(np.percentile(spearmans, 97.5)),
        ],
    }

    with open(bootstrap_root / "meta_inside_corpus.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[OK] Inside-corpus bootstrap finished for {exp_name}")
        # --------------------------------------------------
    # Pretty-print summary table (aligned columns)
    # --------------------------------------------------
    header = (
        f"{'MODEL':<{MODEL_NAME_WIDTH}}"
        f"{'MIN_ERR':>{NUM_WIDTH}}"
        f"{'MEAN_ERR':>{NUM_WIDTH}}"
        f"{'MAX_ERR':>{NUM_WIDTH}}"
    )

    sep = "-" * len(header)

    lines = [header, sep]

    # sort by mean error (ascending)
    for rep, stats in sorted(
        summary.items(),
        key=lambda x: x[1]["mean_error_rate"]
    ):
        line = (
            f"{rep:<{MODEL_NAME_WIDTH}}"
            f"{NUM_FMT.format(stats['ci_2.5']):>{NUM_WIDTH}}"
            f"{NUM_FMT.format(stats['mean_error_rate']):>{NUM_WIDTH}}"
            f"{NUM_FMT.format(stats['ci_97.5']):>{NUM_WIDTH}}"
        )
        lines.append(line)

    table = "\n".join(lines)

    # print to stdout
    print("\n=== INSIDE-CORPUS BOOTSTRAP SUMMARY ===\n")
    print(table)

    # also save to file for reuse / paper copy
    with open(bootstrap_root / "inside_corpus_summary.txt", "w") as f:
        f.write(table + "\n")


if __name__ == "__main__":
    main()
