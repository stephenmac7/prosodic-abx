#!/usr/bin/env python3
"""
Bootstrap phonological contrast units from a task pair.

Given two task names (task_a, task_b):
- Read:
    results/<task_a>/cells/chinese_hubert_large/l1.csv
- Extract unique phonological contrast keys:
    (phone_sequence, accent_pattern, accent_pattern_b)
- Treat each key as an equal-weight unit
- Perform multinomial bootstrap (sampling WITH replacement):
    * number of draws = number of unique keys
- Repeat N times (default: 500)

Output:
    bootstrapping_pair/<task_a>_<task_b>_<N>.jsonl
"""

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Tuple, List
from collections import Counter


FASTABX_ROOT = Path("/home/sunhaitong/fastabx")

# (phone_sequence, accent_pattern, accent_pattern_b)
PairKey = Tuple[str, str, str]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("task_a", help="e.g. pitch_accent")
    p.add_argument("task_b", help="e.g. pitch_accent_syn")
    p.add_argument("--n_boot", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def key_to_str(k: PairKey) -> str:
    return f"{k[0]}|{k[1]}|{k[2]}"


def load_pair_keys(csv_path: Path) -> List[PairKey]:
    """
    Load CSV and extract unique (phone_sequence, accent_pattern, accent_pattern_b).
    """
    keys = set()

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (
                row["phone_sequence"],
                row["accent_pattern"],
                row["accent_pattern_b"],
            )
            keys.add(key)

    return sorted(keys)


def main():
    args = parse_args()
    random.seed(args.seed)

    # --------------------------------------------------
    # Locate input CSV (ONLY task_a is used)
    # --------------------------------------------------
    csv_path = (
        FASTABX_ROOT
        / "results"
        / args.task_a
        / "cells"
        / "chinese_hubert_large"
        / "l1.csv"
    )

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    # --------------------------------------------------
    # Load phonological contrast units
    # --------------------------------------------------
    pair_keys = load_pair_keys(csv_path)
    K = len(pair_keys)

    if K == 0:
        raise RuntimeError("No phonological contrast keys found.")

    # --------------------------------------------------
    # Prepare output
    # --------------------------------------------------
    out_dir = FASTABX_ROOT / "bootstrapping_pair"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{args.task_a}_{args.task_b}_{args.n_boot}.jsonl"

    # --------------------------------------------------
    # Bootstrap (WITH replacement)
    # --------------------------------------------------
    with open(out_path, "w", encoding="utf-8") as fout:
        for b in range(args.n_boot):
            sampled = random.choices(
                population=pair_keys,
                k=K,   # draw K times WITH replacement
            )

            counts = Counter(sampled)

            record = {
                "bootstrap_id": b,
                "counts": {
                    key_to_str(k): counts.get(k, 0)
                    for k in pair_keys
                },
            }

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[OK] Pair-level bootstrap written to {out_path}")
    print(f"     #unique keys = {K}, #bootstrap = {args.n_boot}")


if __name__ == "__main__":
    main()
