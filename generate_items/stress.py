"""Generate fastabx-compatible item file for stress ABX from word-level recordings.

Supports two modes:

1. CLIPPED mode (default):
   Each audio file contains a single word. This script reads a JSON annotation file
   and produces a CSV item file with full-word segments (onset=0, offset=duration).
   Audio files are pre-extracted word clips.

2. IN-CONTEXT mode (--in-context):
   Uses the original sentence recordings with MFA-aligned timestamps.
   The item file references full sentence audio files with onset/offset pointing
   to the target word boundaries from MFA alignment.

The output format matches run_abx.py expectations:
  - #file: path relative to audio_root (without extension)
  - onset / offset: in seconds
  - phone_sequence: word (used for BY condition)
  - accent_pattern: stress label (used for ON condition)
  - speaker: speaker ID
  - context_set: set ID parsed from filename
  - part_of_word: take ID parsed from filename

Filename format: S003_discharge_B_2.wav
  - parts[0]: speaker (S003)
  - parts[1]: word (discharge)
  - parts[2]: set ID (B)
  - parts[3]: take ID (2)
"""

from __future__ import annotations

import argparse
import csv
import json
import wave
from pathlib import Path
from typing import Optional

import tgt

# Hardcoded input paths
ANN_JSON = Path("/home/sunhaitong/ABX_stress/annotations/stephen_annotations_recording_final.json")
AUDIO_ROOT_CLIPPED = Path("/home/sunhaitong/ABX_stress/data/recording_words_manual")
AUDIO_ROOT_SENTENCES = Path("/home/sunhaitong/ABX_stress/data/recording_sentences")
TEXTGRID_ROOT = Path("/home/sunhaitong/ABX_stress/scripts_clean/tmp_mfa_sentence_batch")
MANUAL_TIMESTAMPS = Path(__file__).parent / "metadata" / "stress_recording_manual_timestamps.json"
OUTPUT_DIR = Path("abx_items/stress")

# Processing parameters
INVALID_LABEL = "invalid"  # Label value to skip

# Target words in order (used to compute line numbers)
TARGET_WORDS = [
    "abstract", "address", "conduct", "discharge", "discount",
    "impact", "import", "increase", "insert", "insult",
    "permit", "project", "research", "survey", "transfer",
    "transport", "upset"
]


# ============================================================
# ===================== MFA TEXTGRID PARSING =================
# ============================================================

def parse_words_tier(textgrid_path: Path) -> list[tgt.Interval]:
    """Parse an MFA TextGrid and return intervals from the 'words' tier."""
    textgrid = tgt.io.read_textgrid(str(textgrid_path))
    words_tier = textgrid.get_tier_by_name("words")
    return list(words_tier.intervals)


def find_word_interval(
    intervals: list[tgt.Interval],
    target_word: str,
) -> Optional[tuple[float, float]]:
    """Find time span for target word in words tier (case-insensitive)."""
    target_lower = target_word.lower()
    for iv in intervals:
        if iv.text.lower() == target_lower:
            return (iv.start_time, iv.end_time)
    return None


def load_manual_timestamps(path: Path) -> dict[str, dict[str, float]]:
    """Load manual timestamp overrides from JSON file."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# ===================== FILENAME PARSING =====================
# ============================================================

def parse_filename(fname: str) -> tuple[str, str, str, str]:
    """Parse filename like S003_discharge_B_2.wav into components."""
    stem = Path(fname).stem
    parts = stem.split("_")
    if len(parts) != 4:
        raise ValueError(f"Unexpected filename format: {fname}")
    speaker, word, set_id, take_id = parts
    return speaker, word, set_id, take_id


def get_sentence_info(speaker: str, word: str, set_id: str, take_id: str) -> tuple[str, int]:
    """
    Get sentence filename stem and line number from clipped file info.

    Returns (sentence_stem, line_number) where:
    - sentence_stem: e.g., "S003_B_007"
    - line_number: 1-based line number in the set
    """
    word_idx = TARGET_WORDS.index(word.lower())
    rep = int(take_id)
    line = word_idx * 2 + rep
    sentence_stem = f"{speaker}_{set_id}_{line:03d}"
    return sentence_stem, line


def get_duration_seconds(audio_path: Path) -> float:
    with wave.open(str(audio_path), "rb") as wf:
        return wf.getnframes() / wf.getframerate()


def load_annotations(ann_json: Path) -> list[dict]:
    with ann_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Annotation JSON must be a list of items")
    return data


# ============================================================
# ===================== BUILD ITEMS ==========================
# ============================================================

def build_items_clipped(
    annotations: list[dict],
    audio_root: Path,
    invalid_label: str,
) -> list[dict]:
    """Build item rows from annotations using pre-clipped word audio files."""
    items: list[dict] = []
    skipped_invalid = 0
    skipped_missing = 0

    for entry in annotations:
        label = str(entry["label"])
        if label == invalid_label:
            skipped_invalid += 1
            continue

        word = entry["word"]
        fname = entry["filename"]
        speaker, word2, set_id, take_id = parse_filename(fname)
        if word != word2:
            raise ValueError(f"Word mismatch: json word={word}, filename={fname}")

        audio_path = audio_root / word / fname
        if not audio_path.exists():
            skipped_missing += 1
            continue

        duration = get_duration_seconds(audio_path)
        file_id = f"{word}/{Path(fname).stem}"

        items.append(
            {
                "#file": file_id,
                "onset": 0.0,
                "offset": duration,
                "phone_sequence": word,
                "accent_pattern": label,
                "speaker": speaker,
                "context_set": set_id,
                "part_of_word": take_id,
            }
        )

    if skipped_invalid > 0:
        print(f"Skipped {skipped_invalid} items with label={invalid_label}")
    if skipped_missing > 0:
        print(f"Skipped {skipped_missing} items with missing audio files")

    return items


def build_items_in_context(
    annotations: list[dict],
    audio_root: Path,
    textgrid_root: Path,
    manual_timestamps: dict[str, dict[str, float]],
    invalid_label: str,
) -> list[dict]:
    """Build item rows using original sentence audio with MFA timestamps."""
    items: list[dict] = []
    skipped_invalid = 0
    skipped_missing = 0
    skipped_no_word = 0
    used_manual = 0

    for entry in annotations:
        label = str(entry["label"])
        if label == invalid_label:
            skipped_invalid += 1
            continue

        word = entry["word"]
        fname = entry["filename"]
        speaker, word2, set_id, take_id = parse_filename(fname)
        if word != word2:
            raise ValueError(f"Word mismatch: json word={word}, filename={fname}")

        sentence_stem, line = get_sentence_info(speaker, word, set_id, take_id)

        # Sentence files: {speaker}/{speaker}_{set}_{line:03d}.wav
        audio_path = audio_root / speaker / f"{sentence_stem}.wav"
        # TextGrid files: {speaker}_{set}/output/{speaker}_{set}_{line:03d}.TextGrid
        textgrid_path = textgrid_root / f"{speaker}_{set_id}" / "output" / f"{sentence_stem}.TextGrid"

        if not audio_path.exists():
            skipped_missing += 1
            continue

        # Check for manual timestamp override first
        if fname in manual_timestamps:
            onset = manual_timestamps[fname]["onset"]
            offset = manual_timestamps[fname]["offset"]
            used_manual += 1
        elif not textgrid_path.exists():
            skipped_missing += 1
            continue
        else:
            # Parse TextGrid and find word
            try:
                intervals = parse_words_tier(textgrid_path)
                span = find_word_interval(intervals, word)
            except Exception as e:
                print(f"Warning: Failed to parse {textgrid_path}: {e}")
                skipped_no_word += 1
                continue

            if span is None:
                skipped_no_word += 1
                continue

            onset, offset = span

        items.append(
            {
                "#file": f"{speaker}/{sentence_stem}",
                "onset": onset,
                "offset": offset,
                "phone_sequence": word,
                "accent_pattern": label,
                "speaker": speaker,
                "context_set": set_id,
                "part_of_word": take_id,
            }
        )

    if skipped_invalid > 0:
        print(f"Skipped {skipped_invalid} items with label={invalid_label}")
    if skipped_missing > 0:
        print(f"Skipped {skipped_missing} items with missing audio/TextGrid files")
    if skipped_no_word > 0:
        print(f"Skipped {skipped_no_word} items where word not found in TextGrid")
    if used_manual > 0:
        print(f"Used {used_manual} manual timestamp overrides")

    return items


def write_items(items: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "#file",
        "onset",
        "offset",
        "phone_sequence",
        "accent_pattern",
        "speaker",
        "context_set",
        "part_of_word",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for item in items:
            writer.writerow(item)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate fastabx item file for stress ABX from recordings"
    )
    parser.add_argument(
        "--in-context",
        action="store_true",
        help="Use original sentence recordings with MFA timestamps instead of clipped word files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: abx_items/stress or abx_items/stress_in_context)",
    )
    args = parser.parse_args()

    # Determine output directory
    if args.output_dir is not None:
        output_dir = args.output_dir
    elif args.in_context:
        output_dir = Path("abx_items/stress_in_context")
    else:
        output_dir = OUTPUT_DIR

    # Determine audio root based on mode
    if args.in_context:
        audio_root = AUDIO_ROOT_SENTENCES
        mode_str = "in-context (sentence audio with MFA timestamps)"
    else:
        audio_root = AUDIO_ROOT_CLIPPED
        mode_str = "clipped (pre-extracted word files)"

    print(f"Mode: {mode_str}")
    print(f"Annotation file: {ANN_JSON}")
    print(f"Audio root: {audio_root}")
    if args.in_context:
        print(f"TextGrid root: {TEXTGRID_ROOT}")
        print(f"Manual timestamps: {MANUAL_TIMESTAMPS}")
    print(f"Output directory: {output_dir}")
    print()

    annotations = load_annotations(ANN_JSON)
    print(f"Loaded {len(annotations)} annotations")

    if args.in_context:
        manual_timestamps = load_manual_timestamps(MANUAL_TIMESTAMPS)
        print(f"Loaded {len(manual_timestamps)} manual timestamp overrides")

        items = build_items_in_context(
            annotations=annotations,
            audio_root=audio_root,
            textgrid_root=TEXTGRID_ROOT,
            manual_timestamps=manual_timestamps,
            invalid_label=INVALID_LABEL,
        )
    else:
        items = build_items_clipped(
            annotations=annotations,
            audio_root=audio_root,
            invalid_label=INVALID_LABEL,
        )

    output_path = output_dir / "items.csv"
    write_items(items, output_path)

    # Write audio path for extract_features.py to find
    with (output_dir / "audio_path.txt").open("w") as f:
        f.write(str(audio_root))

    print(f"\nWrote {len(items)} items to {output_path}")
    print(f"Unique speakers: {len({i['speaker'] for i in items})}")
    print(f"Unique words: {len({i['phone_sequence'] for i in items})}")


if __name__ == "__main__":
    main()
