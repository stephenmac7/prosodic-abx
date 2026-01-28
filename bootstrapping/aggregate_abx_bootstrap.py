#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Aggregate ABX errors under two-level bootstrapping.

THIRD bootstrap layer (and the most computationally expensive one).

This script combines:
1) contrast-level bootstrapping
2) speaker-level bootstrapping
3) ABX cell results for each model and each layer

to compute, for each model:
- the bootstrap distribution of ABX error rates
- where, for each bootstrap iteration, the best-performing layer
  (minimum error) is selected.

Usage:
    python aggregate_abx_bootstrap.py <language> <task>
"""

import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict


# --------------------------------------------------
# Configuration
# --------------------------------------------------
BASE_RESULTS = Path("/home/sunhaitong/fastabx/results")
BASE_BOOTSTRAP = Path("/home/sunhaitong/fastabx/bootstrapping")

STRESS_MAP = {
    "noun": "1",
    "verb": "2",
}


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("language")
    parser.add_argument("task")
    args = parser.parse_args()

    language = args.language
    task = args.task

    # --------------------------------------------------
    # Load contrast-level bootstraps
    # --------------------------------------------------
    contrast_path = (
        BASE_BOOTSTRAP / language / f"{language}_contrast.jsonl"
    )
    with open(contrast_path, "r", encoding="utf-8") as f:
        contrast_bootstraps = [json.loads(line) for line in f]

    # --------------------------------------------------
    # Load speaker-level bootstraps
    # --------------------------------------------------
    speaker_path = (
        BASE_BOOTSTRAP / language / f"{task}_speaker.jsonl"
    )
    with open(speaker_path, "r", encoding="utf-8") as f:
        speaker_bootstraps = [json.loads(line) for line in f]

    assert len(contrast_bootstraps) == len(speaker_bootstraps)

    # --------------------------------------------------
    # Locate model directories
    # --------------------------------------------------
    cells_root = BASE_RESULTS / task / "cells"
    model_dirs = [d for d in cells_root.iterdir() if d.is_dir()]

    out_root = BASE_BOOTSTRAP / language / task
    out_root.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Process each model
    # --------------------------------------------------
    for model_dir in model_dirs:
        model_name = model_dir.name
        out_csv = out_root / f"{model_name}.csv"

        if out_csv.exists():
            print(f"[SKIP] {model_name} already computed.")
            continue

        print(f"[PROCESS] Model: {model_name}")

    
        # Collect layer files for this model.
        #
        # Most SSL models have multiple layers named like:
        #   l1.csv, l2.csv, ...
        #
        # But handcrafted baselines like MFCC/FBANK often have only ONE layer,
        # and the file is named:
        #   features.csv  (sometimes feature.csv)
        #
        # Therefore:
        #   - If l*.csv exists: use them as multiple layers
        #   - Else if features.csv / feature.csv exists: treat it as a single "layer"
        # --------------------------------------------------
        layer_files = sorted(model_dir.glob("l*.csv"))

        if not layer_files:
            # Single-layer baselines
            f1 = model_dir / "features.csv"
            if f1.exists():
                layer_files = [f1]

        if not layer_files:
            print(f"[WARN] No layer files found for model {model_name} under {model_dir}, skipping.")
            continue


        results = []

        # --------------------------------------------------
        # Iterate over bootstrap iterations
        # --------------------------------------------------
        for b, (c_boot, s_boot) in enumerate(
            zip(contrast_bootstraps, speaker_bootstraps)
        ):
            contrast_counts = c_boot["counts"]
            speaker_counts_all = s_boot["speaker_bootstrap"]

            best_error = None
            best_layer = None

            # --------------------------------------------------
            # Evaluate each layer
            # --------------------------------------------------
            for layer_file in layer_files:
                weighted_sum = defaultdict(float)
                weight_sum = defaultdict(float)

                with open(layer_file, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        ps = row["phone_sequence"]
                        ap = row["accent_pattern"]
                        apb = row["accent_pattern_b"]

                        # stress_syn: map noun/verb -> 1/2 for key construction
                        if task == "stress_syn":
                            ap = STRESS_MAP.get(ap, ap)
                            apb = STRESS_MAP.get(apb, apb)

                        key = f"{ps}|{ap}|{apb}"

                        if key not in speaker_counts_all:
                            continue

                        spk = row["speaker"]
                        spk_x = row["speaker_x"]
                        score = float(row["score"])
                        size = float(row["size"])

                        spk_counts = speaker_counts_all[key]

                        w = (
                            spk_counts.get(spk, 0)
                            * spk_counts.get(spk_x, 0)
                            * size
                        )

                        if w == 0:
                            continue

                        weighted_sum[key] += w * score
                        weight_sum[key] += w

                # Aggregate over contrasts
                num = 0.0
                den = 0.0
                for key, c_count in contrast_counts.items():
                    if key in weighted_sum and weight_sum[key] > 0:
                        c_score = weighted_sum[key] / weight_sum[key]
                        num += c_score * c_count
                        den += c_count

                if den == 0:
                    continue

                layer_error = num / den

                if best_error is None or layer_error < best_error:
                    best_error = layer_error
                    best_layer = layer_file.stem

            results.append(
                {
                    "bootstrap_id": b,
                    "best_layer": best_layer,
                    "error": best_error,
                }
            )

        # --------------------------------------------------
        # Save results for this model
        # --------------------------------------------------
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["bootstrap_id", "best_layer", "error"]
            )
            writer.writeheader()
            writer.writerows(results)

        print(f"[DONE] Saved {out_csv}")


if __name__ == "__main__":
    main()
