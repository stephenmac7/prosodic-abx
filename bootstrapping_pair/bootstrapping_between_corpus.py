#!/usr/bin/env python3
"""
Paired bootstrap comparison between two corpora using phonological-pair bootstrap.

Given two task names (task_a, task_b):
- Load shared pair-level bootstrap weights from:
    bootstrapping_pair/<task_a>_<task_b>_500.jsonl
- For each corpus:
    - Reweight ABX scores in results/<task>/cells/*
    - Select best layer per representation (oracle)
    - Obtain representation ranking per bootstrap
- For each bootstrap index:
    - Compute Spearman correlation between the two rankings
- Report mean Spearman and 95% CI
"""

import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr


FASTABX_ROOT = Path("/home/sunhaitong/fastabx")





def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("task_a", help="e.g. pitch_accent")
    p.add_argument("task_b", help="e.g. pitch_accent_syn")
    return p.parse_args()


def load_pair_bootstrap(task_a, task_b):
    path = (
        FASTABX_ROOT
        / "bootstrapping_pair"
        / f"{task_a}_{task_b}_500.jsonl"
    )
    boots = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            boots.append(obj["counts"])
    return boots


def load_csv_rows(csv_path: Path):
    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pair_key(phone, pat_a, pat_b):
    return f"{phone}|{pat_a}|{pat_b}"


def compute_rep_errors(task, pair_counts):
    """
    For one corpus (task) and one bootstrap world:
    return {rep: best_layer_error}
    """
    cells_root = FASTABX_ROOT / "results" / task / "cells"
    rep_to_error = {}

    for rep_dir in cells_root.iterdir():
        if not rep_dir.is_dir():
            continue

        layer_errors = {}

        for csv_path in rep_dir.glob("*.csv"):
            score_sum = 0.0
            count_sum = 0.0

            rows = load_csv_rows(csv_path)

            # --------------------------------------------------
            # ONE-TIME normalization at CSV-load level
            # --------------------------------------------------
            if task == "stress_syn":
                for r in rows:
                    if r["accent_pattern"] == "noun":
                        r["accent_pattern"] = "1"
                    elif r["accent_pattern"] == "verb":
                        r["accent_pattern"] = "2"

                    if r["accent_pattern_b"] == "noun":
                        r["accent_pattern_b"] = "1"
                    elif r["accent_pattern_b"] == "verb":
                        r["accent_pattern_b"] = "2"

            for row in rows:
                k = pair_key(
                    row["phone_sequence"],
                    row["accent_pattern"],
                    row["accent_pattern_b"],
                )

                w = pair_counts.get(k, 0)
                if w == 0:
                    continue

                score = float(row["score"])
                size = float(row["size"])

                score_sum += score * w * size
                count_sum += size * w

            if count_sum > 0:
                layer_errors[csv_path.stem] = score_sum / count_sum

        if layer_errors:
            best_layer = min(layer_errors, key=layer_errors.get)
            rep_to_error[rep_dir.name] = layer_errors[best_layer]

    return rep_to_error


def main():
    args = parse_args()

    # --------------------------------------------------
    # Load shared bootstrap weights
    # --------------------------------------------------
    pair_boots = load_pair_bootstrap(args.task_a, args.task_b)

    spearmans = []

    # --------------------------------------------------
    # Bootstrap loop (paired)
    # --------------------------------------------------
    for b, pair_counts in enumerate(pair_boots):

        errors_a = compute_rep_errors(args.task_a, pair_counts)
        errors_b = compute_rep_errors(args.task_b, pair_counts)

        # keep only common representations
        common = sorted(set(errors_a) & set(errors_b))
        if len(common) < 2:
            continue

        rank_a = {r: i for i, r in enumerate(sorted(common, key=lambda x: errors_a[x]))}
        rank_b = {r: i for i, r in enumerate(sorted(common, key=lambda x: errors_b[x]))}

        v1 = [rank_a[r] for r in common]
        v2 = [rank_b[r] for r in common]

        rho, _ = spearmanr(v1, v2)
        spearmans.append(rho)

    spearmans = np.array(spearmans)

    mean_rho = float(spearmans.mean())
    ci_low = float(np.percentile(spearmans, 2.5))
    ci_high = float(np.percentile(spearmans, 97.5))

    # --------------------------------------------------
    # Output (command line + txt file)
    # --------------------------------------------------
    out_dir = FASTABX_ROOT / "bootstrapping_pair"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"between_corpus_spearman_{args.task_a}_{args.task_b}.txt"

    lines = []
    lines.append("=== BETWEEN-CORPUS BOOTSTRAP RANKING CONSISTENCY ===")
    lines.append("")
    lines.append(f"Task A              : {args.task_a}")
    lines.append(f"Task B              : {args.task_b}")
    lines.append(f"# bootstrap samples : {len(spearmans)}")
    lines.append(f"Mean Spearman ρ     : {mean_rho:.6f}")
    lines.append(f"95% CI              : [{ci_low:.6f}, {ci_high:.6f}]")
    lines.append("")

    text = "\n".join(lines)

    # print to console
    print("\n" + text)

    # write to file
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")

    print(f"[OK] Result written to {out_path}")


if __name__ == "__main__":
    main()
