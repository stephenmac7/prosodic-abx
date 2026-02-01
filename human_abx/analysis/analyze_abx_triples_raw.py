#!/usr/bin/env python3
"""Analyze human ABX trial performance."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterator

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
                # Format: timestamp, participant_id, list_id, prolific_pid, study_id, status, ...
                participant_id = parts[1]
                status = parts[5]
                if status == "SCREENED_OUT":
                    screened.add(participant_id)
    return screened


def load_responses(
    files: list[Path],
    exclude_participants: set[str] | None = None,
) -> Iterator[dict]:
    """Load all responses from files, excluding specified participants."""
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
                row["_participant"] = participant
                row["_filepath"] = filepath
                yield row


def compute_accuracy(trials: list[dict]) -> tuple[int, int, float]:
    """Compute accuracy from list of trials."""
    correct = sum(1 for t in trials if t.get("correct", "").lower() == "true")
    total = len(trials)
    accuracy = correct / total if total > 0 else 0
    return correct, total, accuracy


def print_breakdown(
    title: str, breakdown: dict[str, list[dict]], min_trials: int = 1
) -> None:
    """Print accuracy breakdown by category."""
    print(f"\n{title}")
    print("-" * 60)

    results = []
    for key, trials in breakdown.items():
        if len(trials) < min_trials:
            continue
        correct, total, acc = compute_accuracy(trials)
        error_rate = 1 - acc
        results.append((key, correct, total, acc, error_rate))

    # Sort by error rate descending
    results.sort(key=lambda x: -x[4])

    for key, correct, total, acc, error_rate in results:
        print(f"  {key:30s} {correct:4d}/{total:<4d} acc={acc*100:5.1f}% err={error_rate*100:5.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Analyze human ABX trial performance")
    parser.add_argument(
        "files", nargs="*", help="Response CSV files (default: all in data/)"
    )
    parser.add_argument(
        "--data-dir",
        default=Path.home() / "public_html/human_abx/data",
        type=Path,
        help="Data directory",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        help="Comma-separated list of participant IDs to exclude",
    )
    parser.add_argument(
        "--min-trials",
        type=int,
        default=5,
        help="Minimum trials per category for breakdown (default: 5)",
    )
    args = parser.parse_args()

    if args.files:
        files = [Path(f) for f in args.files]
    else:
        files = sorted(args.data_dir.glob("responses_*.csv"))

    if not files:
        print("No response files found")
        return

    exclude = set(args.exclude.split(",")) if args.exclude else set()

    # Load screened-out participants from submissions log
    screened_out = load_screened_out_participants(args.data_dir)

    # Combine exclusions
    all_excluded = exclude | screened_out

    # Load all ABX trials
    trials = list(load_responses(files, all_excluded))

    if not trials:
        print("No ABX trials found")
        return

    # Overall accuracy
    correct, total, accuracy = compute_accuracy(trials)
    error_rate = 1 - accuracy

    print("=" * 60)
    print("HUMAN ABX ANALYSIS")
    print("=" * 60)
    print(f"Total ABX trials: {total}")
    print(f"Overall accuracy: {correct}/{total} = {accuracy*100:.1f}%")
    print(f"Overall error rate: {error_rate*100:.1f}%")

    # By participant
    by_participant = defaultdict(list)
    for t in trials:
        by_participant[t["_participant"]].append(t)

    print(f"\nParticipants analyzed: {len(by_participant)}")
    if screened_out:
        print(f"Screened out: {len(screened_out)} ({', '.join(sorted(screened_out))})")
    if exclude:
        print(f"Manually excluded: {len(exclude)}")

    print_breakdown("BY PARTICIPANT", by_participant, args.min_trials)

    # By word/phone_sequence
    by_word = defaultdict(list)
    for t in trials:
        word = t.get("phone_sequence", "unknown")
        by_word[word].append(t)

    print_breakdown("BY WORD/SEQUENCE (highest error first)", by_word, args.min_trials)

    # By correct answer (A vs B) - check for response bias
    by_correct_answer = defaultdict(list)
    for t in trials:
        answer = t.get("correct_answer", "?")
        by_correct_answer[answer].append(t)

    print_breakdown("BY CORRECT ANSWER (check for bias)", by_correct_answer, 1)

    # By accent pattern
    by_accent = defaultdict(list)
    for t in trials:
        accent_a = t.get("accent_a", "?")
        accent_b = t.get("accent_b", "?")
        by_accent[f"{accent_a} vs {accent_b}"].append(t)

    print_breakdown("BY ACCENT CONTRAST", by_accent, args.min_trials)

    # Response time analysis
    rts = []
    for t in trials:
        try:
            rt = float(t.get("rt_ms", 0))
            if rt > 0:
                rts.append((rt, t.get("correct", "").lower() == "true"))
        except (ValueError, TypeError):
            pass

    if rts:
        print("\nRESPONSE TIME ANALYSIS")
        print("-" * 60)
        all_rts = [r[0] for r in rts]
        correct_rts = [r[0] for r in rts if r[1]]
        incorrect_rts = [r[0] for r in rts if not r[1]]

        def stats(values):
            if not values:
                return "N/A"
            values = sorted(values)
            mean = sum(values) / len(values)
            median = values[len(values) // 2]
            return f"mean={mean:.0f}ms, median={median:.0f}ms"

        print(f"  All trials:     {stats(all_rts)}")
        print(f"  Correct:        {stats(correct_rts)}")
        print(f"  Incorrect:      {stats(incorrect_rts)}")

    # Recording-level analysis (for QC)
    print("\nRECORDING-LEVEL ACCURACY (for annotation QC)")
    print("-" * 60)

    by_file = defaultdict(list)
    for t in trials:
        correct = t.get("correct", "").lower() == "true"
        for key in ["file_a", "file_b", "file_x"]:
            if t.get(key):
                by_file[t[key]].append(correct)

    # Find recordings with low accuracy
    low_acc_recordings = []
    for filepath, corrects in by_file.items():
        if len(corrects) >= args.min_trials:
            acc = sum(corrects) / len(corrects)
            if acc < 0.5:
                low_acc_recordings.append((filepath, sum(corrects), len(corrects), acc))

    if low_acc_recordings:
        print("Recordings with <50% accuracy (candidates for re-annotation):")
        low_acc_recordings.sort(key=lambda x: x[3])
        for filepath, correct, total, acc in low_acc_recordings[:20]:
            print(f"  {filepath}: {correct}/{total} = {acc*100:.1f}%")
    else:
        print("No recordings with <50% accuracy found.")

    print()


if __name__ == "__main__":
    main()
