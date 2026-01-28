#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Contrast-level bootstrapping script.

A contrast unit is defined as:
    (phone_sequence, accent_pattern, accent_pattern_b)

Input:
    FastABX features file:
    /home/sunhaitong/fastabx/results/{contrast_type}/cells/fbank/features.csv

Output:
    {language}/{language}_contrast.jsonl

Each line in the output JSONL corresponds to ONE bootstrap iteration and
stores the multiplicity (count) of ALL sampled contrast units.

Bootstrapping is performed WITH replacement.
"""

import argparse
import csv
import json
import random
from pathlib import Path
from collections import Counter


# --------------------------------------------------
# Configuration
# --------------------------------------------------
BASE_DIR = Path("/home/sunhaitong/fastabx/results")
DEFAULT_N_BOOTSTRAP = 500
RANDOM_SEED = 42


# --------------------------------------------------
# Helper
# --------------------------------------------------
def contrast_key(phone_sequence, accent_pattern, accent_pattern_b):
    """
    Convert a contrast tuple into a stable string key.

    This key is used as a dictionary key in JSON output.
    """
    return f"{phone_sequence}|{accent_pattern}|{accent_pattern_b}"


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Contrast-level bootstrapping (with replacement)."
    )
    parser.add_argument(
        "language",
        help="Language name (e.g. mandarin, english, japanese)"
    )
    parser.add_argument(
        "contrast_type",
        help="Contrast type used to locate features.csv (e.g. pitch_accent)"
    )
    parser.add_argument(
        "n_sample",
        nargs="?",
        type=int,
        default=None,
        help=(
            "Number of contrasts sampled per bootstrap iteration. "
            "Default: sample all unique contrasts."
        )
    )
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=DEFAULT_N_BOOTSTRAP,
        help="Number of bootstrap iterations (default: 500)"
    )
    args = parser.parse_args()

    random.seed(RANDOM_SEED)

    # --------------------------------------------------
    # Locate input file
    # --------------------------------------------------
    in_csv = (
        BASE_DIR
        / args.contrast_type
        / "cells"
        / "fbank"
        / "features.csv"
    )

    if not in_csv.exists():
        raise FileNotFoundError(f"Input file not found: {in_csv}")

    # --------------------------------------------------
    # Load and deduplicate contrast units
    # --------------------------------------------------
    contrasts = set()

    with open(in_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            phone_sequence = row["phone_sequence"]
            accent_pattern = row["accent_pattern"]
            accent_pattern_b = row["accent_pattern_b"]

            contrasts.add(
                (phone_sequence, accent_pattern, accent_pattern_b)
            )

    contrasts = sorted(contrasts)
    num_contrasts = len(contrasts)

    if num_contrasts == 0:
        raise RuntimeError("No contrast units found in input file.")

    # --------------------------------------------------
    # Determine sample size per bootstrap
    # --------------------------------------------------
    sample_size = (
        args.n_sample if args.n_sample is not None else num_contrasts
    )

    if sample_size <= 0:
        raise ValueError("n_sample must be a positive integer.")

    print(f"Number of unique contrasts: {num_contrasts}")
    print(f"Sample size per bootstrap: {sample_size}")
    print(f"Number of bootstrap iterations: {args.n_bootstrap}")

    # --------------------------------------------------
    # Prepare output
    # --------------------------------------------------
    out_dir = Path(args.language)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / f"{args.language}_contrast.jsonl"

    # --------------------------------------------------
    # Perform bootstrapping
    # --------------------------------------------------
    with open(out_jsonl, "w", encoding="utf-8") as fout:

        for b in range(args.n_bootstrap):

            # ------------------------------------------
            # Sample contrasts WITH replacement
            # ------------------------------------------
            sampled = random.choices(contrasts, k=sample_size)

            # ------------------------------------------
            # Count multiplicity of each contrast
            # ------------------------------------------
            counter = Counter(sampled)

            # Convert to string-keyed dict for JSON
            counts = {
                contrast_key(ps, ap, apb): c
                for (ps, ap, apb), c in counter.items()
            }

            # ------------------------------------------
            # Save one bootstrap iteration as one JSON line
            # ------------------------------------------
            record = {
                "bootstrap_id": b,
                "num_unique_contrasts": len(counter),
                "total_samples": sample_size,
                "counts": counts,
            }

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved contrast bootstraps to: {out_jsonl}")


if __name__ == "__main__":
    main()
