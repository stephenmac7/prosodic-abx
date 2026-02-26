#!/usr/bin/env python3
"""Format a LaTeX table of prosodic minimal pair dataset statistics.

Reads item CSV files produced by generate_items/ scripts and computes
per-dataset statistics: number of ABX pairs, speakers/voices, and total
audio duration in hours.

Usage:
  uv run python scripts/format_dataset_stats_table.py \
    --row "English" "Recording" abx_items/stress/items.csv \
    --row "English" "G-TTS"     abx_items/stress_syn/items.csv \
    --row "English" "Kokoro"    abx_items/stress_kokoro/items.csv \
    --row "Japanese" "Recording" abx_items/pitch_accent/items.csv \
    --row "Japanese" "G-TTS"    abx_items/pitch_accent_syn/items.csv \
    --row "Mandarin" "MCAE"     abx_items/mandarin_tone/items.csv \
    --row "Mandarin" "G-TTS"    abx_items/mandarin_tone_syn/items.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from math import comb
from pathlib import Path


def compute_stats(items_csv: Path) -> dict:
    """Compute dataset statistics from an items CSV file.

    Returns dict with keys: n_pairs, n_speakers, hours.

    #Pairs is the number of distinct (phone_sequence, tone-pair) combinations,
    computed as sum of C(n_patterns, 2) over each phone_sequence group.
    """
    with items_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Group accent_patterns by phone_sequence
    by_phone: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        by_phone[r["phone_sequence"]].add(r["accent_pattern"])

    n_pairs = sum(comb(len(patterns), 2) for patterns in by_phone.values())
    n_speakers = len({r["speaker"] for r in rows})
    total_seconds = sum(float(r["offset"]) - float(r["onset"]) for r in rows)
    minutes = total_seconds / 60

    return {"n_pairs": n_pairs, "n_speakers": n_speakers, "minutes": minutes}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Format LaTeX table of prosodic minimal pair dataset statistics."
    )
    parser.add_argument(
        "--row",
        action="append",
        nargs=3,
        metavar=("LANGUAGE", "SOURCE", "CSV_PATH"),
        required=True,
        help="Language, source label, and path to items.csv.",
    )
    parser.add_argument("--label", type=str, default="tab:dataset_stats")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    # Compute stats and group by language for midrule placement
    rows: list[dict] = []
    for language, source, csv_path in args.row:
        stats = compute_stats(Path(csv_path))
        rows.append(
            {
                "language": language,
                "source": source,
                "n_pairs": stats["n_pairs"],
                "n_speakers": stats["n_speakers"],
                "minutes": stats["minutes"],
            }
        )

    # Build LaTeX
    lines = [
        r"\begin{table}[t]",
        r"  \caption{Overview of the prosodic minimal pair datasets.}",
        f"  \\label{{{args.label}}}",
        r"  \centering",
        r"  \small",
        r"  \begin{tabular}{l c c c c}",
        r"    \toprule",
        r"    \textbf{Language} & \textbf{Source} & \textbf{\#Pairs} & \textbf{\#Spks/Voices} & \textbf{Min.} \\",
        r"    \midrule",
    ]

    prev_lang = None
    for row in rows:
        lang_cell = row["language"] if row["language"] != prev_lang else ""
        if prev_lang is not None and row["language"] != prev_lang:
            lines.append(r"    \midrule")

        lines.append(
            f"    {lang_cell} & {row['source']}"
            f" & {row['n_pairs']}"
            f" & {row['n_speakers']}"
            f" & {row['minutes']:.1f} \\\\"
        )
        prev_lang = row["language"]

    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
        ]
    )

    output = "\n".join(lines) + "\n"
    if args.output:
        args.output.write_text(output)
    else:
        print(output)


if __name__ == "__main__":
    main()
