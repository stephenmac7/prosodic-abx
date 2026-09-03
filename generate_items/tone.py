"""Generate fastabx items for the Mandarin tone recordings.

Run prepare_data.py --tone before this script. It reads selected MCAE
Monosyllable audio and metadata from data/tone/ and writes
abx_items/tone/ for extract_features.py and run_abx.py.

The ABX task is: ON tone, BY pinyin, ACROSS speaker
- Discriminate tones for the same syllable (pinyin) across different speakers.

The output item file uses:
  - #file: path relative to audio_path.txt, without extension
  - onset / offset: segment boundaries in seconds
  - phone_sequence: pinyin, used as the BY condition
  - accent_pattern: tone label, used as the ON condition
  - speaker: speaker ID
"""

from __future__ import annotations

import argparse
import csv
import wave
from pathlib import Path

METADATA_CSV = Path("data/tone/metadata.csv")
AUDIO_ROOT = Path("data/tone/audio")
OUTPUT_DIR = Path("abx_items/tone")


def get_duration_seconds(audio_path: Path) -> float:
    """Return WAV duration in seconds."""
    with wave.open(str(audio_path), "rb") as wf:
        return wf.getnframes() / wf.getframerate()


def load_metadata(path: Path) -> list[dict[str, str]]:
    """Load prepared metadata rows."""
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def build_items(
    rows: list[dict[str, str]],
    audio_root: Path,
) -> list[dict[str, object]]:
    """Build item rows from Mandarin tone metadata."""
    items: list[dict[str, object]] = []
    skipped_missing = 0

    for row in rows:
        item_id = row["id"]
        audio_path = audio_root / f"{item_id}.wav"
        if not audio_path.exists():
            skipped_missing += 1
            continue

        items.append(
            {
                "#file": item_id,
                "onset": 0.0,
                "offset": get_duration_seconds(audio_path),
                "phone_sequence": row["target"],
                "accent_pattern": row["label"],
                "speaker": row["speaker"],
            }
        )

    if skipped_missing:
        print(f"Skipped {skipped_missing} items with missing audio files")

    return items


def write_items(items: list[dict[str, object]], output_path: Path) -> None:
    """Write the fastabx item CSV."""
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
        writer.writerows(items)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate fastabx items for Mandarin tone"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory (default: abx_items/mandarin_tone)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir

    print(f"Metadata file: {METADATA_CSV}")
    print(f"Audio root: {AUDIO_ROOT}")
    print(f"Output directory: {output_dir}")
    print()

    rows = load_metadata(METADATA_CSV)
    print(f"Loaded {len(rows)} metadata rows")

    items = build_items(
        rows=rows,
        audio_root=AUDIO_ROOT,
    )

    output_path = output_dir / "items.csv"
    write_items(items, output_path)

    with (output_dir / "audio_path.txt").open("w", encoding="utf-8") as f:
        f.write(str(AUDIO_ROOT))

    print(f"\nWrote {len(items)} items to {output_path}")
    print(f"Unique speakers: {len({i['speaker'] for i in items})}")
    print(f"Unique pinyin: {len({i['phone_sequence'] for i in items})}")
    print(f"Unique tones: {sorted(set(i['accent_pattern'] for i in items))}")


if __name__ == "__main__":
    main()
