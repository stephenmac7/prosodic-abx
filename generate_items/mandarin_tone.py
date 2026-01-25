"""Generate fastabx-compatible item file for Mandarin tone ABX.

This script reads an ABXpy-format item file and produces a CSV item file
compatible with run_abx.py.

The ABX task is: ON tone, BY pinyin, ACROSS speaker
- Discriminate tones for the same syllable (pinyin) across different speakers.

Source item file format (ABXpy space-separated):
    /path/to/audio.wav onset offset #spk=3 #tone=4 #pinyin=cou

Output format (CSV for run_abx.py):
  - #file: filename stem (without extension)
  - onset / offset: in seconds (computed from actual audio duration)
  - phone_sequence: pinyin (used for BY condition)
  - accent_pattern: tone (used for ON condition)
  - speaker: speaker ID

Note on "balanced" sampling:
  The original balanced_tone_abx_triples.py explicitly sampled N complete triples
  per (pinyin, tone_pair) cell. In fastabx, cell-level balancing is achieved via
  the Subsampler class with max_size_group parameter when running run_abx.py.
  This limits the number of items in each A/B/X group, providing similar balancing
  without explicit triple enumeration.
"""

from __future__ import annotations

import csv
import re
import wave
from pathlib import Path

# Input paths
ITEM_FILE = Path("/home/sunhaitong/ABX_tone/data/mandarin_tone_clean_abx.item")
AUDIO_ROOT = Path("/home/sunhaitong/Chinese_dataset/MCAE-Monosyllable/Audio_files/neutral_16k")
OUTPUT_DIR = Path("abx_items/mandarin_tone")

def parse_abxpy_item(line: str) -> dict | None:
    """Parse a line from ABXpy item file format.

    Format: /path/to/file.wav onset offset #key1=val1 #key2=val2 ...
    """
    line = line.strip()
    if not line:
        return None

    parts = line.split()
    if len(parts) < 3:
        return None

    wav_path = Path(parts[0])
    # onset and offset from file are ignored - we compute from audio

    # Parse #key=value tags
    tags = {}
    tag_pattern = re.compile(r"#(\w+)=(.+)")
    for part in parts[3:]:
        match = tag_pattern.match(part)
        if match:
            tags[match.group(1)] = match.group(2)

    return {
        "wav_path": wav_path,
        "filename": wav_path.stem,
        "tags": tags,
    }


def get_duration_seconds(audio_path: Path) -> float:
    with wave.open(str(audio_path), "rb") as wf:
        return wf.getnframes() / wf.getframerate()


def build_items(
    item_file: Path,
    audio_root: Path,
) -> list[dict]:
    """Build item rows from ABXpy item file."""
    items: list[dict] = []
    skipped_missing = 0
    skipped_parse = 0

    with item_file.open("r", encoding="utf-8") as f:
        for line in f:
            parsed = parse_abxpy_item(line)
            if parsed is None:
                skipped_parse += 1
                continue

            tags = parsed["tags"]
            filename = parsed["filename"]

            # Required tags
            if not all(k in tags for k in ("spk", "tone", "pinyin")):
                skipped_parse += 1
                continue

            audio_path = audio_root / f"{filename}.wav"
            if not audio_path.exists():
                skipped_missing += 1
                continue

            duration = get_duration_seconds(audio_path)
            offset = duration

            items.append(
                {
                    "#file": filename,
                    "onset": 0.0,
                    "offset": offset,
                    "phone_sequence": tags["pinyin"],
                    "accent_pattern": tags["tone"],
                    "speaker": tags["spk"],
                }
            )

    if skipped_parse > 0:
        print(f"Skipped {skipped_parse} items due to parse errors")
    if skipped_missing > 0:
        print(f"Skipped {skipped_missing} items with missing audio files")

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
    print(f"Item file: {ITEM_FILE}")
    print(f"Audio root: {AUDIO_ROOT}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    items = build_items(
        item_file=ITEM_FILE,
        audio_root=AUDIO_ROOT,
    )

    output_path = OUTPUT_DIR / "items.csv"
    write_items(items, output_path)

    # Write audio path for extract_features.py to find
    with (OUTPUT_DIR / "audio_path.txt").open("w") as f:
        f.write(str(AUDIO_ROOT))

    print(f"\nWrote {len(items)} items to {output_path}")
    print(f"Unique speakers: {len({i['speaker'] for i in items})}")
    print(f"Unique pinyin: {len({i['phone_sequence'] for i in items})}")
    print(f"Unique tones: {sorted(set(i['accent_pattern'] for i in items))}")


if __name__ == "__main__":
    main()
