#!/usr/bin/env python3
"""Analyze catch trial performance to identify unreliable participants."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def analyze_file(filepath: Path) -> dict | None:
    """Analyze catch trials in a single response file.

    Returns dict with catch trial stats, or None if no catch trials found.
    """
    catch_trials = []

    with open(filepath) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=2):
            is_catch = row.get("is_catch", "").lower() == "true"
            if is_catch:
                correct = row.get("correct", "").lower() == "true"
                catch_trials.append({
                    "row": i,
                    "correct": correct,
                    "phone_sequence": row.get("phone_sequence", "?"),
                    "response": row.get("response", "?"),
                    "correct_answer": row.get("correct_answer", "?"),
                    "rt_ms": row.get("rt_ms", "?"),
                })

    if not catch_trials:
        return None

    n_correct = sum(1 for t in catch_trials if t["correct"])
    n_total = len(catch_trials)

    return {
        "filepath": filepath,
        "n_correct": n_correct,
        "n_total": n_total,
        "accuracy": n_correct / n_total if n_total > 0 else 0,
        "trials": catch_trials,
    }


def extract_participant_info(filepath: Path) -> dict:
    """Extract participant ID, list number, and dataset from filename."""
    # Expected format: responses_{participant}_{list}_{timestamp}.csv
    parts = filepath.stem.split("_")
    if len(parts) >= 3:
        return {
            "participant": parts[1],
            "list": parts[2],
        }
    return {"participant": "unknown", "list": "unknown"}


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


def main():
    parser = argparse.ArgumentParser(
        description="Analyze catch trial performance to identify unreliable participants"
    )
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
        "--threshold",
        type=float,
        default=0.65,
        help="Minimum catch trial accuracy to pass (default: 0.65)",
    )
    args = parser.parse_args()

    if args.files:
        files = [Path(f) for f in args.files]
    else:
        files = sorted(args.data_dir.glob("responses_*.csv"))

    if not files:
        print("No response files found")
        return

    # Load screened-out participants from submissions log
    screened_out = load_screened_out_participants(args.data_dir)

    passed = []
    failed = []
    screened_out_results = []

    for f in files:
        info = extract_participant_info(f)

        result = analyze_file(f)

        # Check if screened out
        if info["participant"] in screened_out:
            if result:
                result.update(info)
            else:
                result = info
            screened_out_results.append(result)
            continue
        if result is None:
            print(f"Warning: No catch trials in {f.name}")
            continue

        result.update(info)

        if result["accuracy"] >= args.threshold:
            passed.append(result)
        else:
            failed.append(result)

    # Summary
    print("=" * 60)
    print("CATCH TRIAL ANALYSIS")
    print("=" * 60)
    print(f"Threshold: {args.threshold * 100:.0f}%")
    total = len(passed) + len(failed) + len(screened_out_results)
    print(f"Total participants: {total}")
    print(f"  Passed: {len(passed)}")
    print(f"  Failed catch threshold: {len(failed)}")
    print(f"  Screened out: {len(screened_out_results)}")
    print()

    if screened_out_results:
        print("SCREENED OUT (excluded from analysis):")
        print("-" * 60)
        for r in screened_out_results:
            if "n_total" in r and r["n_total"] > 0:
                pct = r["accuracy"] * 100
                print(
                    f"  {r['participant']:20s} list={r['list']:4s} "
                    f"catch={r['n_correct']}/{r['n_total']} ({pct:.0f}%)"
                )
                for t in r["trials"]:
                    status = "PASS" if t["correct"] else "FAIL"
                    print(
                        f"    Row {t['row']:3d}: {status} "
                        f"({t['phone_sequence']}) "
                        f"responded={t['response']} correct={t['correct_answer']}"
                    )
            else:
                print(f"  {r['participant']:20s} list={r['list']} (no catch trials)")
        print()

    if failed:
        print("FAILED PARTICIPANTS (exclude from analysis):")
        print("-" * 60)
        for r in sorted(failed, key=lambda x: x["accuracy"]):
            pct = r["accuracy"] * 100
            print(
                f"  {r['participant']:20s} list={r['list']:4s} "
                f"catch={r['n_correct']}/{r['n_total']} ({pct:.0f}%)"
            )
        print()

    if passed:
        print("PASSED PARTICIPANTS:")
        print("-" * 60)
        accuracies = []
        for r in sorted(passed, key=lambda x: -x["accuracy"]):
            pct = r["accuracy"] * 100
            accuracies.append(r["accuracy"])
            print(
                f"  {r['participant']:20s} list={r['list']:4s} "
                f"catch={r['n_correct']}/{r['n_total']} ({pct:.0f}%)"
            )
        print()
        if accuracies:
            mean_acc = sum(accuracies) / len(accuracies)
            print(f"Mean catch accuracy (passed): {mean_acc * 100:.1f}%")


if __name__ == "__main__":
    main()
