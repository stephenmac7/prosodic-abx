#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Speaker-level bootstrapping conditioned on contrast-level bootstraps.

SECOND bootstrap layer.

Given a contrast-level bootstrap file (JSONL) for a language, where each
line corresponds to one bootstrap iteration and provides a multiset of
contrast units:

    (phone_sequence, accent_pattern, accent_pattern_b)

we perform speaker-level bootstrapping (WITH replacement) for each
contrast unit under a specific task (e.g. pitch_accent, pitch_accent_syn).

Key idea:
- For each contrast, we repeatedly sample speakers with replacement
  until the sampled speaker multiset contains at least ONE valid
  (speaker, speaker_x) pair observed in the task's features.csv.

Output:
- One JSONL per task
- Each line corresponds to one contrast-level bootstrap iteration
- For each contrast key in that iteration, store the accepted speaker
  bootstrap counts (speaker -> multiplicity).

IMPORTANT SPECIAL CASE (stress_syn vs english):

- english contrast bootstrap uses numeric labels:
      1 / 2
- stress_syn features.csv uses string labels:
      noun / verb

Therefore, when the task is stress_syn:
- We MUST map bootstrap labels (1 / 2) → task labels (noun / verb)
  for LOOKUP ONLY.
- Output keys remain UNCHANGED (still using 1 / 2), so that the speaker
  bootstrap file stays aligned with the contrast bootstrap file.
"""

import argparse
import csv
import json
import random
from pathlib import Path
from collections import defaultdict, Counter
from typing import Tuple


# --------------------------------------------------
# Configuration
# --------------------------------------------------
BASE_RESULTS = Path("/home/sunhaitong/fastabx/results")
BASE_BOOTSTRAP = Path("/home/sunhaitong/fastabx/bootstrapping")
RANDOM_SEED = 42


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def contrast_key(phone_sequence: str, accent_pattern: str, accent_pattern_b: str) -> str:
    """Stable string key for a contrast unit."""
    return f"{phone_sequence}|{accent_pattern}|{accent_pattern_b}"


def parse_contrast_key(key: str) -> Tuple[str, str, str]:
    """Inverse of contrast_key()."""
    ps, ap, apb = key.split("|")
    return ps, ap, apb


def needs_stress_syn_mapping(task_name: str) -> bool:
    """
    Decide whether this task uses stress_syn-style labels (noun / verb).

    We rely on task naming convention here.
    """
    t = task_name.lower()
    return ("stress" in t) and ("syn" in t)


def map_numeric_to_stress_syn(label: str) -> str:
    """
    Map numeric stress labels to stress_syn string labels
    for LOOKUP ONLY.

    Mapping:
        1 -> noun
        2 -> verb

    If an unexpected label is encountered, return it unchanged.
    """
    if label == "1":
        return "noun"
    if label == "2":
        return "verb"
    return label


def lookup_tuple(task_name: str, ps: str, ap: str, apb: str) -> Tuple[str, str, str]:
    """
    Convert a contrast tuple (from contrast bootstrap file)
    into the tuple used to look up (speaker, speaker_x) pairs
    in the task's features.csv.

    IMPORTANT:
    - contrast bootstrap keys ALWAYS come from the language-level
      bootstrap file (e.g. english_contrast.jsonl).
    - When task == stress_syn, these keys use numeric labels (1 / 2),
      but features.csv expects noun / verb.

    Therefore:
    - stress_syn task: map 1/2 -> noun/verb for lookup
    - other tasks: identity mapping
    """
    if needs_stress_syn_mapping(task_name):
        ap_l = map_numeric_to_stress_syn(ap)
        apb_l = map_numeric_to_stress_syn(apb)
        return (ps, ap_l, apb_l)

    return (ps, ap, apb)


# --------------------------------------------------
# Load ABX cell data for a task
# --------------------------------------------------
def load_task_cells(task_name: str):
    """
    Load features.csv for a given task and return:

    1) speakers:
         Sorted list of all speakers appearing in this task
    2) contrast_to_pairs:
         (phone_sequence, accent_pattern, accent_pattern_b)
         -> list of (speaker, speaker_x) pairs
    """
    in_csv = (
        BASE_RESULTS
        / task_name
        / "cells"
        / "fbank"
        / "features.csv"
    )

    if not in_csv.exists():
        raise FileNotFoundError(f"Missing features.csv: {in_csv}")

    speakers = set()
    contrast_to_pairs = defaultdict(list)

    with open(in_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ps = row["phone_sequence"]
            ap = row["accent_pattern"]
            apb = row["accent_pattern_b"]
            spk = row["speaker"]
            spk_x = row["speaker_x"]

            speakers.add(spk)
            speakers.add(spk_x)

            contrast_to_pairs[(ps, ap, apb)].append((spk, spk_x))

    return sorted(speakers), contrast_to_pairs


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Speaker-level bootstrapping conditioned on contrast bootstraps."
    )
    parser.add_argument("language", help="e.g. japanese, english")
    parser.add_argument("task1", help="e.g. pitch_accent")
    parser.add_argument("task2", help="e.g. pitch_accent_syn")
    parser.add_argument(
        "--max_resample",
        type=int,
        default=20000,
        help=(
            "Safety cap: maximum number of speaker re-sampling attempts per contrast. "
            "If exceeded, we skip that contrast for that iteration."
        ),
    )
    args = parser.parse_args()

    random.seed(RANDOM_SEED)

    # --------------------------------------------------
    # Load contrast-level bootstraps (script 1 output)
    # --------------------------------------------------
    contrast_jsonl = (
        BASE_BOOTSTRAP
        / args.language
        / f"{args.language}_contrast.jsonl"
    )

    if not contrast_jsonl.exists():
        raise FileNotFoundError(f"Missing contrast bootstrap file: {contrast_jsonl}")

    with open(contrast_jsonl, "r", encoding="utf-8") as f:
        contrast_bootstraps = [json.loads(line) for line in f]

    print(f"Loaded {len(contrast_bootstraps)} contrast bootstraps from {contrast_jsonl}")

    # --------------------------------------------------
    # Process each task independently
    # --------------------------------------------------
    for task in [args.task1, args.task2]:

        print(f"\n=== Processing task: {task} ===")
        if needs_stress_syn_mapping(task):
            print("NOTE: stress_syn task detected — mapping 1/2 -> noun/verb for lookup.")

        speakers, contrast_to_pairs = load_task_cells(task)
        num_speaker = len(speakers)

        print(f"Number of speakers in task: {num_speaker}")

        out_dir = BASE_BOOTSTRAP / args.language
        out_dir.mkdir(parents=True, exist_ok=True)
        out_jsonl = out_dir / f"{task}_speaker.jsonl"

        # --------------------------------------------------
        # For each contrast-bootstrap iteration
        # --------------------------------------------------
        with open(out_jsonl, "w", encoding="utf-8") as fout:

            for entry in contrast_bootstraps:
                bootstrap_id = entry["bootstrap_id"]
                contrast_counts = entry["counts"]  # keys only

                speaker_bootstrap = {}

                # ------------------------------------------
                # Speaker bootstrapping for each contrast
                # ------------------------------------------
                for key in contrast_counts.keys():

                    ps, ap, apb = parse_contrast_key(key)

                    # Map labels ONLY for lookup, never for output
                    ps_l, ap_l, apb_l = lookup_tuple(task, ps, ap, apb)

                    pairs = contrast_to_pairs.get((ps_l, ap_l, apb_l), [])
                    if not pairs:
                        continue

                    attempts = 0
                    while True:
                        attempts += 1
                        if attempts > args.max_resample:
                            break

                        sampled = random.choices(speakers, k=num_speaker)
                        spk_counter = Counter(sampled)

                        valid = False
                        for spk, spk_x in pairs:
                            if spk_counter[spk] >= 1 and spk_counter[spk_x] >= 1:
                                valid = True
                                break

                        if valid:
                            # IMPORTANT:
                            # Store under the ORIGINAL bootstrap key (1/2),
                            # not the mapped noun/verb key.
                            speaker_bootstrap[key] = dict(spk_counter)
                            break

                record = {
                    "bootstrap_id": bootstrap_id,
                    "task": task,
                    "speaker_bootstrap": speaker_bootstrap,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"Saved speaker bootstraps to: {out_jsonl}")


if __name__ == "__main__":
    main()
