#!/usr/bin/env python3
"""
Item-level multinomial bootstrap based on abx_items/<exp_name>/items.csv.

Key = (phone_sequence, accent_pattern, speaker)

This script:
- reads items.csv
- estimates the empirical distribution over item keys
- performs multinomial bootstrap on that distribution
- saves all bootstrap samples for later reuse

IMPORTANT:
- This operates purely at the *item distribution* level
- It does NOT touch ABX scoring, cells, or corpora
"""

import argparse
import csv
import json
import random
from pathlib import Path
from collections import Counter
from typing import Tuple


# --------------------------------------------------
# Configuration (single source of truth)
# --------------------------------------------------
FASTABX_ROOT = Path("/home/sunhaitong/fastabx")


# Each item key corresponds to one atomic input configuration
ItemKey = Tuple[str, str, str]  # (phone_sequence, accent_pattern, speaker)


# --------------------------------------------------
# Argument parsing
# --------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("exp_name", help="e.g. pitch_accent_syn")
    p.add_argument("--n_boot", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# --------------------------------------------------
# Load items and count keys
# --------------------------------------------------
def load_item_keys(items_csv: Path) -> Counter[ItemKey]:
    """
    Read items.csv and count occurrences of each
    (phone_sequence, accent_pattern, speaker) key.
    """
    counter: Counter[ItemKey] = Counter()

    with open(items_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (
                row["phone_sequence"],
                row["accent_pattern"],
                row["speaker"],
            )
            counter[key] += 1

    return counter


def key_to_str(k: ItemKey) -> str:
    """Serialize ItemKey to a stable string for JSON output."""
    return f"{k[0]}|{k[1]}|{k[2]}"



def main():
    args = parse_args()
    random.seed(args.seed)


    items_csv = FASTABX_ROOT / "abx_items" / args.exp_name / "items.csv"
    if not items_csv.exists():
        raise FileNotFoundError(items_csv)

    out_dir = FASTABX_ROOT / "bootstrapping" / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Load empirical item distribution
    # --------------------------------------------------
    key_counts = load_item_keys(items_csv)

    keys = list(key_counts.keys())
    weights = [key_counts[k] for k in keys]
    total_count = sum(weights)  # number of rows in items.csv

    # --------------------------------------------------
    # Save original distribution
    # --------------------------------------------------
    with open(out_dir / "item_key_counts.json", "w", encoding="utf-8") as f:
        json.dump(
            {key_to_str(k): v for k, v in key_counts.items()},
            f,
            indent=2,
            ensure_ascii=False,
        )

    meta = {
        "exp_name": args.exp_name,
        "items_csv": str(items_csv),
        "n_unique_keys": len(keys),
        "total_item_count": total_count,
        "n_bootstrap": args.n_boot,
        "seed": args.seed,
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # --------------------------------------------------
    # Multinomial bootstrap
    # --------------------------------------------------
    out_jsonl = out_dir / f"bootstrap_item_{args.n_boot}.jsonl"

    with open(out_jsonl, "w", encoding="utf-8") as fout:
        for b in range(args.n_boot):
            sampled = random.choices(
                population=keys,
                weights=weights,
                k=total_count,
            )

            boot_counter = Counter(sampled)

            record = {
                "bootstrap_id": b,
                "counts": {
                    key_to_str(k): c
                    for k, c in boot_counter.items()
                },
            }

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[OK] Item bootstrap saved to {out_dir}")


if __name__ == "__main__":
    main()
