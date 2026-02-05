#!/usr/bin/env python3
"""
Generate fastabx-compatible item file for English stress ABX (TTS, clipped).

Annotation-free version.
Each audio file contains a single word.

Expected directory structure:
  /home/sunhaitong/ABX_syn/data/standard_english_stress/
    └── abstract/
        └── abstract_noun_A.wav

Filename format:
  abstract_noun_A.wav
    - parts[0]: phone_sequence (word)
    - parts[1]: accent_pattern (stress label, e.g., noun/verb)
    - parts[2]: speaker (TTS voice)

Output format matches run_abx.py expectations:
  - #file: path relative to audio_root (without extension)
  - onset / offset: full word (0, duration)
  - phone_sequence: word (BY condition)
  - accent_pattern: stress label (ON condition)
  - speaker: speaker ID (ACROSS condition)
"""

from __future__ import annotations

import argparse
import csv
import wave
from pathlib import Path

# ============================================================
# Hardcoded paths (keep style consistent with original script)
# ============================================================

AUDIO_ROOT = Path("synth_data/standard_english_stress")
OUTPUT_DIR = Path("abx_items/stress_syn")

# ============================================================
# Filename parsing (stress logic, slot-based)
# ============================================================

def parse_filename(fname: str) -> tuple[str, str, str]:
    """
    Parse filename like abstract_noun_A.wav

    Returns:
      phone_sequence, accent_pattern, speaker
    """
    stem = Path(fname).stem
    parts = stem.split("_")
    if len(parts) != 3:
        raise ValueError(f"Unexpected filename format: {fname}")

    phone_sequence = parts[0]   # abstract
    accent_pattern = parts[1]   # noun / verb
    speaker = parts[2]          # A / B / C ...

    return phone_sequence, accent_pattern, speaker


# ============================================================
# Audio utility
# ============================================================

def get_duration_seconds(audio_path: Path) -> float:
    with wave.open(str(audio_path), "rb") as wf:
        return wf.getnframes() / wf.getframerate()


# ============================================================
# Build items (annotation-free, clipped)
# ============================================================

def build_items_from_wavs(audio_root: Path) -> list[dict]:
    items: list[dict] = []
    skipped_badname = 0

    for wav_path in sorted(audio_root.rglob("*.wav")):
        try:
            phone_sequence, accent_pattern, speaker = parse_filename(wav_path.name)
        except ValueError as e:
            print(f"Skip (filename): {e}")
            skipped_badname += 1
            continue

        duration = get_duration_seconds(wav_path)

        # #file must be relative to audio_root, without extension
        rel_stem = wav_path.relative_to(audio_root).with_suffix("")

        items.append(
            {
                "#file": str(rel_stem),
                "onset": 0.0,
                "offset": duration,
                "phone_sequence": phone_sequence,
                "accent_pattern": accent_pattern,
                "speaker": speaker,
            }
        )

    if skipped_badname > 0:
        print(f"Skipped {skipped_badname} files due to unexpected filename format")

    return items


# ============================================================
# CSV writer (same style as original stress script)
# ============================================================

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


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate fastabx item file for English stress ABX (TTS, clipped)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory (default: abx_items/stress_syn)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir

    print("Mode: clipped (TTS, annotation-free)")
    print(f"Audio root: {AUDIO_ROOT}")
    print(f"Output directory: {output_dir}")
    print()

    items = build_items_from_wavs(AUDIO_ROOT)

    output_path = output_dir / "items.csv"
    write_items(items, output_path)

    # Write audio path for extract_features.py
    with (output_dir / "audio_path.txt").open("w") as f:
        f.write(str(AUDIO_ROOT))

    print(f"\nWrote {len(items)} items to {output_path}")
    print(f"Unique speakers: {len({i['speaker'] for i in items})}")
    print(f"Unique words: {len({i['phone_sequence'] for i in items})}")
    print(f"Unique stress labels: {len({i['accent_pattern'] for i in items})}")


if __name__ == "__main__":
    main()
