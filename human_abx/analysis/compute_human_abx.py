#!/usr/bin/env python3
"""Compute human ABX error rates using the same averaging as machine ABX.

This script computes human ABX error rates using the same collapsing methodology
as run_abx.py with fastabx:

  1. Group trials by (phone_sequence, contrast)
  2. Within each group, average error rates over speaker combinations
  3. Average across all (phone_sequence, contrast) groups

This matches the machine ABX setup:
  ON: accent_pattern
  BY: phone_sequence
  ACROSS: speaker

Usage:
    python compute_human_abx.py --data-dir ~/public_html/human_abx/data
    python compute_human_abx.py --data-dir ~/public_html/human_abx/data --dataset stress
    python compute_human_abx.py --data-dir ~/public_html/human_abx/data --breakdown
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def load_screened_out_participants(data_dir: Path) -> set[str]:
    """Load participant IDs that were screened out, from submissions.log."""
    screened = set()
    log_file = data_dir / "submissions.log"
    if not log_file.exists():
        return screened

    with open(log_file) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 6:
                participant_id = parts[1]
                status = parts[5]
                if status == "SCREENED_OUT":
                    screened.add(participant_id)
    return screened


def load_trials(
    files: list[Path],
    exclude_participants: set[str] | None = None,
) -> list[dict]:
    """Load ABX trials from response files, excluding catch trials and filtered participants."""
    trials = []
    for filepath in files:
        # Extract participant ID from filename
        parts = filepath.stem.split("_")
        participant = parts[1] if len(parts) >= 2 else "unknown"

        if exclude_participants and participant in exclude_participants:
            continue

        with open(filepath) as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Skip catch trials
                if row.get("is_catch", "").lower() == "true":
                    continue

                trials.append({
                    "phone_sequence": row["phone_sequence"],
                    "accent_a": row["accent_a"],
                    "accent_b": row["accent_b"],
                    "speaker_ab": row["speaker_ab"],
                    "speaker_x": row["speaker_x"],
                    "correct": row.get("correct", "").lower() == "true",
                    "participant": participant,
                })

    return trials


def compute_contrast_key(accent_a: str, accent_b: str) -> tuple[str, str]:
    """Create a canonical contrast key from two accent patterns.

    Returns a sorted tuple so (1,2) and (2,1) map to the same contrast.
    """
    return tuple(sorted([accent_a, accent_b]))


def compute_abx_error_rate(
    trials: list[dict],
    collapse_over_speakers: bool = True,
) -> tuple[float, dict]:
    """Compute ABX error rate using fastabx-style collapsing.

    Args:
        trials: List of trial dicts with phone_sequence, accent_a, accent_b,
                speaker_ab, speaker_x, and correct fields.
        collapse_over_speakers: If True, average over speaker combinations first
                                (matches across-speaker machine ABX).

    Returns:
        Tuple of (error_rate, detailed_results dict)
    """
    # Group trials by (phone_sequence, contrast)
    # Within each group, further group by speaker combination
    groups: dict[tuple, dict[tuple, list[bool]]] = defaultdict(lambda: defaultdict(list))

    for trial in trials:
        contrast = compute_contrast_key(trial["accent_a"], trial["accent_b"])
        cell_key = (trial["phone_sequence"], contrast)
        speaker_key = (trial["speaker_ab"], trial["speaker_x"])
        groups[cell_key][speaker_key].append(trial["correct"])

    # Compute error rates
    cell_error_rates = {}

    for cell_key, speaker_groups in groups.items():
        if collapse_over_speakers:
            # Average over speaker combinations first
            speaker_error_rates = []
            for corrects in speaker_groups.values():
                error_rate = 1 - (sum(corrects) / len(corrects))
                speaker_error_rates.append(error_rate)

            # Mean over speaker combinations
            cell_error_rates[cell_key] = sum(speaker_error_rates) / len(speaker_error_rates)
        else:
            # Pool all trials in the cell
            all_corrects = []
            for corrects in speaker_groups.values():
                all_corrects.extend(corrects)
            cell_error_rates[cell_key] = 1 - (sum(all_corrects) / len(all_corrects))

    # Final error rate: mean over all cells
    final_error_rate = sum(cell_error_rates.values()) / len(cell_error_rates)

    # Build detailed results
    detailed = {
        "cell_error_rates": cell_error_rates,
        "n_cells": len(cell_error_rates),
        "n_trials": len(trials),
    }

    return final_error_rate, detailed


def compute_by_contrast(trials: list[dict]) -> dict[tuple, float]:
    """Compute error rates broken down by contrast (accent pair).

    Returns dict mapping contrast -> error_rate.
    """
    # Group by contrast
    contrast_groups: dict[tuple, dict[str, dict[tuple, list[bool]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    for trial in trials:
        contrast = compute_contrast_key(trial["accent_a"], trial["accent_b"])
        phone_seq = trial["phone_sequence"]
        speaker_key = (trial["speaker_ab"], trial["speaker_x"])
        contrast_groups[contrast][phone_seq][speaker_key].append(trial["correct"])

    results = {}
    for contrast, phone_groups in contrast_groups.items():
        # For each phone_sequence, average over speakers, then average over phone_sequences
        cell_rates = []
        for phone_seq, speaker_groups in phone_groups.items():
            speaker_rates = []
            for corrects in speaker_groups.values():
                speaker_rates.append(1 - sum(corrects) / len(corrects))
            cell_rates.append(sum(speaker_rates) / len(speaker_rates))

        results[contrast] = sum(cell_rates) / len(cell_rates)

    return results


def compute_by_word(trials: list[dict]) -> dict[str, float]:
    """Compute error rates broken down by word/phone_sequence.

    Returns dict mapping phone_sequence -> error_rate.
    """
    # Group by phone_sequence
    word_groups: dict[str, dict[tuple, dict[tuple, list[bool]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    for trial in trials:
        phone_seq = trial["phone_sequence"]
        contrast = compute_contrast_key(trial["accent_a"], trial["accent_b"])
        speaker_key = (trial["speaker_ab"], trial["speaker_x"])
        word_groups[phone_seq][contrast][speaker_key].append(trial["correct"])

    results = {}
    for phone_seq, contrast_groups in word_groups.items():
        # For each contrast, average over speakers, then average over contrasts
        cell_rates = []
        for contrast, speaker_groups in contrast_groups.items():
            speaker_rates = []
            for corrects in speaker_groups.values():
                speaker_rates.append(1 - sum(corrects) / len(corrects))
            cell_rates.append(sum(speaker_rates) / len(speaker_rates))

        results[phone_seq] = sum(cell_rates) / len(cell_rates)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compute human ABX error rates using machine ABX averaging methodology",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "files", nargs="*", help="Response CSV files (default: all in data-dir)"
    )
    parser.add_argument(
        "--data-dir",
        default=Path.home() / "public_html/human_abx/data",
        type=Path,
        help="Data directory containing response files",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Filter to specific dataset (e.g., 'stress', 'pitch_accent', 'mandarin_tone')",
    )
    parser.add_argument(
        "--catch-threshold",
        type=float,
        default=0.65,
        help="Minimum catch trial accuracy to include participant",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        help="Comma-separated list of participant IDs to exclude",
    )
    parser.add_argument(
        "--no-collapse-speakers",
        action="store_true",
        help="Don't collapse over speakers (pool all trials per cell)",
    )
    parser.add_argument(
        "--breakdown",
        action="store_true",
        help="Show breakdown by contrast and word",
    )
    args = parser.parse_args()

    # Find response files
    if args.files:
        files = [Path(f) for f in args.files]
    else:
        files = sorted(args.data_dir.glob("responses_*.csv"))

    if not files:
        print("No response files found")
        return

    # Filter by dataset if specified
    if args.dataset:
        filtered_files = []
        for f in files:
            # Check list_id field which contains the dataset path
            with open(f) as fh:
                reader = csv.DictReader(fh)
                try:
                    row = next(reader)
                    list_id = row.get("list_id", "")
                    # list_id is like "lists/stress/participant_006.csv"
                    # or just "participant_006.csv" depending on format

                    # Check if this file's list matches the dataset
                    # Also check based on file patterns as fallback
                    file_a = row.get("file_a", "")

                    matches = False
                    if args.dataset in list_id:
                        matches = True
                    elif args.dataset == "stress":
                        # English stress: ASCII word names, accent patterns are 1/2 or noun/verb
                        accent_a = row.get("accent_a", "")
                        if file_a and "/" in file_a:
                            word = file_a.split("/")[0]
                            # English words are ASCII alphabetic
                            if word.isalpha() and all(ord(c) < 128 for c in word):
                                if accent_a in ("1", "2", "noun", "verb"):
                                    matches = True
                    elif args.dataset == "pitch_accent":
                        # Japanese: has Japanese characters in file paths
                        if any(ord(c) > 127 for c in file_a):
                            matches = True
                    elif args.dataset == "mandarin_tone":
                        # Mandarin: accent patterns are tone numbers (1-4, typically > 2 for disambiguation)
                        accent_a = row.get("accent_a", "")
                        if accent_a.isdigit():
                            tone = int(accent_a)
                            # Mandarin tones in MCAE are typically numbered differently
                            # Check for pinyin-like patterns (no Japanese chars, no English words)
                            if not any(ord(c) > 127 for c in file_a):
                                # Not Japanese, and has numeric tone markers
                                if tone > 2 or (file_a and "/" in file_a and not file_a.split("/")[0].isalpha()):
                                    matches = True

                    if matches:
                        filtered_files.append(f)
                except StopIteration:
                    pass
        files = filtered_files
        print(f"Filtered to {len(files)} files for dataset: {args.dataset}")

    # Build exclusion set
    exclude = set(args.exclude.split(",")) if args.exclude else set()
    screened_out = load_screened_out_participants(args.data_dir)

    # Also exclude participants who fail catch threshold
    failed_catch = set()
    for f in files:
        parts = f.stem.split("_")
        participant = parts[1] if len(parts) >= 2 else "unknown"

        if participant in screened_out or participant in exclude:
            continue

        # Check catch trial performance
        catch_correct = 0
        catch_total = 0
        with open(f) as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if row.get("is_catch", "").lower() == "true":
                    catch_total += 1
                    if row.get("correct", "").lower() == "true":
                        catch_correct += 1

        if catch_total > 0 and (catch_correct / catch_total) < args.catch_threshold:
            failed_catch.add(participant)

    all_excluded = exclude | screened_out | failed_catch

    # Load trials
    trials = load_trials(files, all_excluded)

    if not trials:
        print("No ABX trials found after filtering")
        return

    # Compute error rate
    collapse_speakers = not args.no_collapse_speakers
    error_rate, details = compute_abx_error_rate(trials, collapse_speakers)

    # Print results
    print("=" * 60)
    print("HUMAN ABX ERROR RATE")
    print("=" * 60)
    print(f"Averaging method: {'collapse over speakers' if collapse_speakers else 'pooled'}")
    print(f"Total ABX trials: {details['n_trials']}")
    print(f"Number of cells (phone_sequence, contrast): {details['n_cells']}")
    print(f"Participants excluded: {len(all_excluded)}")
    if screened_out:
        print(f"  - Screened out: {len(screened_out)}")
    if failed_catch:
        print(f"  - Failed catch threshold: {len(failed_catch)}")
    if exclude:
        print(f"  - Manually excluded: {len(exclude)}")
    print()
    print(f"ABX Error Rate: {error_rate:.4f} ({error_rate * 100:.2f}%)")
    print(f"ABX Accuracy:   {1 - error_rate:.4f} ({(1 - error_rate) * 100:.2f}%)")
    print()

    # Breakdown by contrast and word
    if args.breakdown:
        print("BY CONTRAST (averaged over words and speakers):")
        print("-" * 60)
        by_contrast = compute_by_contrast(trials)
        for contrast, rate in sorted(by_contrast.items(), key=lambda x: -x[1]):
            if rate > 0:
                print(f"  {contrast[0]} vs {contrast[1]}: error={rate*100:.1f}%")
        print()

        print("BY WORD (averaged over contrasts and speakers):")
        print("-" * 60)
        by_word = compute_by_word(trials)
        for word, rate in sorted(by_word.items(), key=lambda x: -x[1]):
            if rate > 0:
                print(f"  {word}: error={rate*100:.1f}%")
        print()


if __name__ == "__main__":
    main()
