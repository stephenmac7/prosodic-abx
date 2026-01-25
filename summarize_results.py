#!/usr/bin/env python3
"""Summarize ABX results for a dataset.

Reads results from results/{dataset}/*.csv and reports the best (lowest)
across-speaker error rate per model. If only a single across-speaker row
exists, the layer is reported as "-".
"""

import argparse
import csv
from pathlib import Path


MODEL_NAME_WIDTH = 48


def load_across_rows(csv_path: Path) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("mode") != "across":
                continue
            layer = row.get("layer", "")[1:]
            try:
                err = float(row.get("error_rate", ""))
            except ValueError:
                continue
            rows.append((layer, err))
    return rows


def main(results_dir: Path) -> None:
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    print("\n=== ABX SUMMARY (ACROSS-SPEAKER) ===\n")

    summary_data = []

    for csv_path in sorted(results_dir.glob("*.csv")):
        name = csv_path.stem
        rows = load_across_rows(csv_path)
        if not rows:
            continue

        if len(rows) == 1:
            layer_label = "-"
            best_err = rows[0][1]
        else:
            layer_label, best_err = min(rows, key=lambda x: x[1])
        
        summary_data.append((name, layer_label, best_err))

    # Sort by error (lowest to highest)
    summary_data.sort(key=lambda x: x[2])

    for name, layer_label, best_err in summary_data:
        print(
            f"{name:<{MODEL_NAME_WIDTH}}"
            f"best layer = {layer_label:>2}   "
            f"error = {best_err:.3f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path, help="Path to results/{dataset} directory")
    args = parser.parse_args()

    main(args.results_dir)
