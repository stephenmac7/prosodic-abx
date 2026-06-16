#!/usr/bin/env python3
"""Generate fastabx items for synthesized English lexical stress.

Run synthesize_wavs/synth_en.py before this script. It reads the synthesized
single-word WAV files and metadata from data/stress_syn/, arranges the audio
under abx_items/stress_syn/audio/ by target word, and writes items.csv for
extract_features.py and run_abx.py.

The output item file uses:
  - #file: path relative to audio_path.txt, without extension
  - onset / offset: full-word boundaries in seconds
  - phone_sequence: target word, used as the BY condition
  - accent_pattern: stress label, used as the ON condition
  - speaker: TTS speaker ID
  - lexical_category: noun or verb
"""

from __future__ import annotations

import argparse
import csv
import shutil
import wave
from pathlib import Path


METADATA_CSV = Path("data/stress_syn/metadata.csv")
OUTPUT_DIR = Path("abx_items/stress_syn")


def get_duration_seconds(audio_path: Path) -> float:
    """Return WAV duration in seconds."""
    with wave.open(str(audio_path), "rb") as wf:
        return wf.getnframes() / wf.getframerate()


def load_metadata(path: Path) -> list[dict[str, str]]:
    """Load synthesized stress metadata rows."""
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def build_items(
    rows: list[dict[str, str]],
    metadata_root: Path,
    output_audio_dir: Path,
) -> list[dict[str, object]]:
    """Arrange synthesized word audio and build item rows."""
    items: list[dict[str, object]] = []
    copied_count = 0

    for row in rows:
        item_id = row["id"]
        target = row["target"]
        src_audio = metadata_root / row["audio_file"]
        if not src_audio.exists():
            raise FileNotFoundError(f"Audio not found: {src_audio}")

        item_audio = output_audio_dir / target / f"{item_id}.wav"
        if not item_audio.exists():
            item_audio.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_audio, item_audio)
            copied_count += 1

        items.append(
            {
                "#file": f"{target}/{item_id}",
                "onset": 0.0,
                "offset": get_duration_seconds(item_audio),
                "phone_sequence": target,
                "accent_pattern": row["label"],
                "speaker": row["speaker"],
                "lexical_category": row["lexical_category"],
            }
        )

    if copied_count:
        print(f"Copied {copied_count} audio files")

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
        "lexical_category",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(items)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate fastabx items for synthesized English lexical stress"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory (default: abx_items/stress_syn)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_audio_dir = output_dir / "audio"

    print("Mode: synthesized clipped target words")
    print(f"Metadata file: {METADATA_CSV}")
    print(f"Output audio directory: {output_audio_dir}")
    print(f"Output directory: {output_dir}")
    print()

    rows = load_metadata(METADATA_CSV)
    print(f"Loaded {len(rows)} metadata rows")

    items = build_items(
        rows=rows,
        metadata_root=METADATA_CSV.parent,
        output_audio_dir=output_audio_dir,
    )

    output_path = output_dir / "items.csv"
    write_items(items, output_path)

    with (output_dir / "audio_path.txt").open("w", encoding="utf-8") as f:
        f.write(str(output_audio_dir))

    print(f"\nWrote {len(items)} items to {output_path}")
    print(f"Unique speakers: {len({i['speaker'] for i in items})}")
    print(f"Unique words: {len({i['phone_sequence'] for i in items})}")
    print(f"Accent patterns: {sorted({i['accent_pattern'] for i in items})}")


if __name__ == "__main__":
    main()
