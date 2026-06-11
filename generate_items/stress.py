"""Generate fastabx items for the English lexical stress recordings.

Run prepare_data.py before this script. It reads sentence recordings and
target-word timestamps from data/stress/, cuts the target words, and writes
abx_items/stress/ for extract_features.py and run_abx.py.

The output item file uses:
  - #file: path relative to audio_path.txt, without extension
  - onset / offset: segment boundaries in seconds
  - phone_sequence: target word, used as the BY condition
  - accent_pattern: stress label, used as the ON condition
  - speaker: speaker ID
  - context_set: carrier-sentence set
  - lexical_category: noun or verb
"""

from __future__ import annotations

import argparse
import csv
import wave
from pathlib import Path

import torchaudio
from torchcodec.decoders import AudioDecoder

METADATA_CSV = Path("data/stress/metadata.csv")
AUDIO_ROOT = Path("data/stress/audio")
OUTPUT_DIR = Path("abx_items/stress")


def clip_audio(src: Path, dst: Path, onset: float, offset: float) -> None:
    """Cut a segment from a WAV file."""
    decoder = AudioDecoder(str(src))
    sr = decoder.metadata.sample_rate
    duration = decoder.metadata.duration_seconds
    total_frames = int(duration * sr)

    start_frame = max(0, int(round(onset * sr)))
    end_frame = min(total_frames, int(round(offset * sr)))
    num_frames = max(0, end_frame - start_frame)

    waveform, _ = torchaudio.load(
        str(src), frame_offset=start_frame, num_frames=num_frames
    )

    dst.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(dst), waveform, sr)


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
    output_audio_dir: Path,
) -> list[dict[str, object]]:
    """Cut target-word audio and build item rows."""
    items: list[dict[str, object]] = []
    clipped_count = 0

    for row in rows:
        item_id = row["id"]
        target = row["target"]
        onset = float(row["target_onset"])
        offset = float(row["target_offset"])

        src_audio = audio_root / f"{item_id}.wav"
        if not src_audio.exists():
            raise FileNotFoundError(f"Audio not found: {src_audio}")

        clipped_audio = output_audio_dir / target / f"{item_id}.wav"
        if not clipped_audio.exists():
            clip_audio(src_audio, clipped_audio, onset, offset)
            clipped_count += 1

        items.append(
            {
                "#file": f"{target}/{item_id}",
                "onset": 0.0,
                "offset": get_duration_seconds(clipped_audio),
                "phone_sequence": target,
                "accent_pattern": row["label"],
                "speaker": row["speaker"],
                "context_set": row["context_set"],
                "lexical_category": row["lexical_category"],
            }
        )

    if clipped_count:
        print(f"Clipped {clipped_count} new audio files")

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
        "context_set",
        "lexical_category",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(items)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate fastabx items for English lexical stress"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory (default: abx_items/stress)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_audio_dir = output_dir / "audio"

    print("Mode: clipped target words")
    print(f"Metadata file: {METADATA_CSV}")
    print(f"Input audio root: {AUDIO_ROOT}")
    print(f"Output audio directory: {output_audio_dir}")
    print(f"Output directory: {output_dir}")
    print()

    rows = load_metadata(METADATA_CSV)
    print(f"Loaded {len(rows)} metadata rows")

    items = build_items(
        rows=rows,
        audio_root=AUDIO_ROOT,
        output_audio_dir=output_audio_dir,
    )

    output_path = output_dir / "items.csv"
    write_items(items, output_path)

    with (output_dir / "audio_path.txt").open("w", encoding="utf-8") as f:
        f.write(str(output_audio_dir))

    print(f"\nWrote {len(items)} items to {output_path}")
    print(f"Unique speakers: {len({i['speaker'] for i in items})}")
    print(f"Unique words: {len({i['phone_sequence'] for i in items})}")


if __name__ == "__main__":
    main()
