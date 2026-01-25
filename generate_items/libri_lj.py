"""Generate fastabx-compatible item file for stress ABX from LibriSpeech/LJSpeech.

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

Filename formats:
  - LibriSpeech: 8786-276735-0070_TWENTIETH.wav
    - speaker: ls-8786 (first segment before '-')
    - stem: 8786-276735-0070
  - LJSpeech: LJ001-0001_WORD.wav
    - speaker: LJ (all LJSpeech treated as one speaker)
    - stem: LJ001-0001
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
ANN_JSON = Path("/home/sunhaitong/ABX_stress/annotations/stephen_annotations_libri_LJ.json")
AUDIO_ROOT_CLIPPED = Path("/home/sunhaitong/ABX_stress/data_Libr_LJ/Libr_LJ_word")
AUDIO_ROOT_SENTENCES = Path("/home/sunhaitong/ABX_stress/data_Libr_LJ/Libr_LJ_MFA")
TEXTGRID_ROOT = Path("/home/sunhaitong/ABX_stress/data_Libr_LJ/Libr_LJ_MFA_aligned")
OUTPUT_DIR = Path("abx_items/libri_lj_stress")

# Processing parameters
INVALID_LABEL = "invalid"  # Label value to skip


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


# ============================================================
# ===================== FILENAME PARSING =====================
# ============================================================

def parse_filename(fname: str) -> tuple[str, str, str]:
    """
    Parse filename into (speaker, speaker_dir, stem).

    Returns:
        speaker: Speaker ID for output CSV (with ls- prefix for LibriSpeech)
        speaker_dir: Directory name on disk (raw speaker ID)
        stem: File stem without word suffix

    LibriSpeech: 8786-276735-0070_TWENTIETH.wav -> ('ls-8786', '8786', '8786-276735-0070')
    LJSpeech: LJ001-0001.wav -> ('LJ', 'LJ', 'LJ001-0001')
    """
    base = Path(fname).stem  # Remove .wav

    if fname.startswith("LJ"):
        # LJSpeech: LJ001-0001.wav - no word suffix
        speaker = "LJ"
        speaker_dir = "LJ"
        stem = base
    else:
        # LibriSpeech: 8786-276735-0070_TWENTIETH.wav - has word suffix
        stem = base.rsplit("_", 1)[0]
        raw_speaker = stem.split("-")[0]
        speaker = f"ls-{raw_speaker}"
        speaker_dir = raw_speaker

    return speaker, speaker_dir, stem


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
        speaker, _speaker_dir, stem = parse_filename(fname)

        # Clipped files are organized by word: {word}/{filename}
        audio_path = audio_root / word / fname
        if not audio_path.exists():
            skipped_missing += 1
            continue

        duration = get_duration_seconds(audio_path)
        offset = duration
        # file_id uses the filename stem (without .wav)
        file_id = f"{word}/{Path(fname).stem}"

        items.append(
            {
                "#file": file_id,
                "onset": 0.0,
                "offset": offset,
                "phone_sequence": word,
                "accent_pattern": label,
                "speaker": speaker,
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
    invalid_label: str,
) -> list[dict]:
    """Build item rows using original sentence audio with MFA timestamps."""
    items: list[dict] = []
    skipped_invalid = 0
    skipped_missing = 0
    skipped_no_word = 0

    for entry in annotations:
        label = str(entry["label"])
        if label == invalid_label:
            skipped_invalid += 1
            continue

        word = entry["word"]
        fname = entry["filename"]
        speaker, speaker_dir, stem = parse_filename(fname)

        # Sentence files: {speaker_dir}/{stem}.wav
        audio_path = audio_root / speaker_dir / f"{stem}.wav"
        textgrid_path = textgrid_root / speaker_dir / f"{stem}.TextGrid"

        if not audio_path.exists():
            skipped_missing += 1
            continue

        if not textgrid_path.exists():
            skipped_missing += 1
            continue

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
                "#file": f"{speaker_dir}/{stem}",
                "onset": onset,
                "offset": offset,
                "phone_sequence": word,
                "accent_pattern": label,
                "speaker": speaker,
            }
        )

    if skipped_invalid > 0:
        print(f"Skipped {skipped_invalid} items with label={invalid_label}")
    if skipped_missing > 0:
        print(f"Skipped {skipped_missing} items with missing audio/TextGrid files")
    if skipped_no_word > 0:
        print(f"Skipped {skipped_no_word} items where word not found in TextGrid")

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
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for item in items:
            writer.writerow(item)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate fastabx item file for LibriSpeech/LJSpeech stress ABX"
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
        help="Output directory (default: abx_items/libri_lj_stress or abx_items/libri_lj_stress_in_context)",
    )
    args = parser.parse_args()

    # Determine output directory
    if args.output_dir is not None:
        output_dir = args.output_dir
    elif args.in_context:
        output_dir = Path("abx_items/libri_lj_stress_in_context")
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
    print(f"Output directory: {output_dir}")
    print()

    annotations = load_annotations(ANN_JSON)
    print(f"Loaded {len(annotations)} annotations")

    if args.in_context:
        items = build_items_in_context(
            annotations=annotations,
            audio_root=audio_root,
            textgrid_root=TEXTGRID_ROOT,
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
    print(f"Unique labels: {len({i['accent_pattern'] for i in items})}")


if __name__ == "__main__":
    main()
