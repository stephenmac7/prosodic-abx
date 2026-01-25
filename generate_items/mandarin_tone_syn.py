#!/usr/bin/env python3
"""
Generate fastabx-compatible item file for Mandarin tone ABX (TTS, clipped).

Annotation-free version.
Each audio file contains a single syllable.

Expected directory structure:
  /home/sunhaitong/ABX_syn/data/standard_mandarin_syllable/
    └── ai/
        └── ai1_A.wav

Filename format:
  ai1_A.wav
    - parts[0]: syllable+tone (e.g., ai1)
    - parts[1]: speaker (TTS voice)

Derived fields:
  - phone_sequence = ai
  - accent_pattern = 1
  - speaker = A

Output format matches run_abx.py expectations:
  - #file: path relative to audio_root (without extension)
  - onset / offset: full syllable (0, duration)
  - phone_sequence: syllable (BY condition)
  - accent_pattern: tone (ON condition)
  - speaker: speaker ID (ACROSS condition)
"""

from __future__ import annotations

import argparse
import csv
import wave
from pathlib import Path



AUDIO_ROOT = Path("/home/sunhaitong/ABX_syn/data/standard_mandarin_syllable")
OUTPUT_DIR = Path("/home/sunhaitong/fastabx/abx_items/mandarin_tone_syn")



def parse_filename(fname: str) -> tuple[str, str, str]:
    """
    Parse filename like ai1_A.wav

    Returns:
      phone_sequence, accent_pattern, speaker
    """
    stem = Path(fname).stem
    parts = stem.split("_")
    if len(parts) != 2:
        raise ValueError(f"Unexpected filename format: {fname}")

    syllable_tone = parts[0]   # ai1
    speaker = parts[1]         # A

    # Split syllable and tone number
    if not syllable_tone[-1].isdigit():
        raise ValueError(f"Expected tone number at end of syllable: {fname}")

    phone_sequence = syllable_tone[:-1]   # ai
    accent_pattern = syllable_tone[-1]    # 1 / 2 / 3 / 4

    return phone_sequence, accent_pattern, speaker



def get_duration_seconds(audio_path: Path) -> float:
    with wave.open(str(audio_path), "rb") as wf:
        return wf.getnframes() / wf.getframerate()




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
        description="Generate fastabx item file for Mandarin tone ABX (TTS, clipped)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory (default: abx_items/mandarin_tone_syn)",
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
    print(f"Unique syllables: {len({i['phone_sequence'] for i in items})}")
    print(f"Unique tones: {len({i['accent_pattern'] for i in items})}")


if __name__ == "__main__":
    main()
