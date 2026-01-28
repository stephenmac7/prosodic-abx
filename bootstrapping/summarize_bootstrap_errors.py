#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Summarize bootstrap ABX errors and ranking.

This script:
1) Reads all model CSV files under:
       /home/sunhaitong/fastabx/bootstrapping/<language>/<task>/*.csv
2) Computes, for each model:
       - mean bootstrap error
       - 2.5% and 97.5% bootstrap percentiles
3) Writes a human-readable summary to:
       abx_error.txt
4) Computes per-bootstrap rankings (lower error = better)
   and writes them as JSONL to:
       abx_ranking.jsonl

Each input CSV is expected to contain rows:
    bootstrap_id,best_layer,error
"""

import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict
import statistics


MODEL_NAME_WIDTH = 48


# --------------------------------------------------
# Load one model's bootstrap errors
# --------------------------------------------------
def load_model_errors(csv_path: Path) -> dict[int, float]:
    """
    Load bootstrap_id -> error mapping from a model CSV file.
    """
    errors: dict[int, float] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                b = int(row["bootstrap_id"])
                e = float(row["error"])
            except (KeyError, ValueError):
                continue
            errors[b] = e
    return errors


# --------------------------------------------------
# Percentile helper (no numpy dependency)
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
def main(language: str, task: str) -> None:
    base_dir = Path("/home/sunhaitong/fastabx/bootstrapping") / language / task
    if not base_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {base_dir}")

    model_csvs = sorted(base_dir.glob("*.csv"))
    if not model_csvs:
        raise RuntimeError(f"No CSV files found under {base_dir}")

    # model -> {bootstrap_id -> error}
    model_errors: dict[str, dict[int, float]] = {}

    for csv_path in model_csvs:
        model_name = csv_path.stem
        model_errors[model_name] = load_model_errors(csv_path)

    # --------------------------------------------------
    # Compute error bars
    # --------------------------------------------------
    summary = []

    for model, errs in model_errors.items():
        values = sorted(errs.values())
        if not values:
            continue

        mean_err = statistics.mean(values)
        p2_5 = percentile(values, 2.5)
        p97_5 = percentile(values, 97.5)

        summary.append((model, mean_err, p2_5, p97_5))

    # Sort by mean error (ascending)
    summary.sort(key=lambda x: x[1])

    # --------------------------------------------------
    # Write human-readable summary
    # --------------------------------------------------
    out_txt = base_dir / "abx_error.txt"
    with out_txt.open("w", encoding="utf-8") as f:
        f.write(f"\n=== ABX BOOTSTRAP SUMMARY ({language} / {task}) ===\n\n")
        for model, mean_err, p2_5, p97_5 in summary:
            f.write(
                f"{model:<{MODEL_NAME_WIDTH}}"
                f"mean = {mean_err:.3f}   "
                f"[{p2_5:.3f}, {p97_5:.3f}]\n"
            )

    print(f"[DONE] Wrote error summary to {out_txt}")

    # --------------------------------------------------
    # Compute per-bootstrap rankings
    # --------------------------------------------------
    # bootstrap_id -> list of (model, error)
    by_bootstrap: dict[int, list[tuple[str, float]]] = defaultdict(list)

    for model, errs in model_errors.items():
        for b, e in errs.items():
            by_bootstrap[b].append((model, e))

    out_jsonl = base_dir / "abx_ranking.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for b in sorted(by_bootstrap.keys()):
            items = sorted(by_bootstrap[b], key=lambda x: x[1])  # lower error is better
            ranking = [(model, rank + 1) for rank, (model, _) in enumerate(items)]

            record = {
                "bootstrap_id": b,
                "ranking": ranking,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[DONE] Wrote bootstrap rankings to {out_jsonl}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("language", help="e.g. english")
    parser.add_argument("task", help="e.g. stress")
    args = parser.parse_args()

    main(args.language, args.task)
