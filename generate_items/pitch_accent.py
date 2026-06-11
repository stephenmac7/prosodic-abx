"""Generate fastabx items for the Japanese pitch accent recordings.

Run prepare_data.py before this script. It reads sentence recordings and
target-word timestamps from data/pitch_accent/ and writes abx_items/pitch_accent/
for extract_features.py and run_abx.py.

Supports two modes:

1. Clipped mode (default):
   Cuts target-word segments and groups the clips by surface_kana.

2. In-context mode (--in-context):
   References sentence recordings and uses target-word timestamps.

The output item file uses:
  - #file: path relative to audio_path.txt, without extension
  - onset / offset: segment boundaries in seconds
  - phone_sequence: surface_kana plus context_id, used as the BY condition
  - accent_pattern: pitch accent label, used as the ON condition
  - speaker: speaker ID
"""

from __future__ import annotations

import argparse
import csv
import wave
from pathlib import Path

import torchaudio
from torchcodec.decoders import AudioDecoder

METADATA_CSV = Path("data/pitch_accent/metadata.csv")
AUDIO_ROOT = Path("data/pitch_accent/audio")
OUTPUT_DIR = Path("abx_items/pitch_accent")


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


def phone_sequence(row: dict[str, str]) -> str:
    """Keep minimal pairs separated by carrier-sentence context."""
    return f"{row['surface_kana']}_{row['context_id']}"


def build_items(
    rows: list[dict[str, str]],
    audio_root: Path,
    output_audio_dir: Path | None = None,
) -> list[dict[str, object]]:
    """Build clipped or in-context item rows."""
    items: list[dict[str, object]] = []
    clipped_count = 0

    for row in rows:
        item_id = row["id"]
        speaker = row["speaker"]
        onset = float(row["target_onset"])
        offset = float(row["target_offset"])

        src_audio = audio_root / f"{item_id}.wav"
        if not src_audio.exists():
            raise FileNotFoundError(f"Audio not found: {src_audio}")

        if output_audio_dir is not None:
            clipped_audio = output_audio_dir / row["surface_kana"] / f"{item_id}.wav"
            if not clipped_audio.exists():
                clip_audio(src_audio, clipped_audio, onset, offset)
                clipped_count += 1

            item_file = f"{row['surface_kana']}/{item_id}"
            item_onset = 0.0
            item_offset = get_duration_seconds(clipped_audio)
        else:
            item_file = item_id
            item_onset = onset
            item_offset = offset

        items.append(
            {
                "#file": item_file,
                "onset": item_onset,
                "offset": item_offset,
                "phone_sequence": phone_sequence(row),
                "accent_pattern": row["label"],
                "speaker": speaker,
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
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(items)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate fastabx items for Japanese pitch accent"
    )
    parser.add_argument(
        "--in-context",
        action="store_true",
        help="Use sentence recordings with target-word timestamps instead of clipped word files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: abx_items/pitch_accent or abx_items/pitch_accent_in_context)",
    )
    args = parser.parse_args()

    if args.output_dir is not None:
        output_dir = args.output_dir
    elif args.in_context:
        output_dir = Path("abx_items/pitch_accent_in_context")
    else:
        output_dir = OUTPUT_DIR

    if args.in_context:
        audio_path_root = AUDIO_ROOT
        output_audio_dir = None
        mode_str = "in-context target words"
    else:
        audio_path_root = output_dir / "audio"
        output_audio_dir = audio_path_root
        mode_str = "clipped target words"

    print(f"Mode: {mode_str}")
    print(f"Metadata file: {METADATA_CSV}")
    print(f"Input audio root: {AUDIO_ROOT}")
    if not args.in_context:
        print(f"Output audio directory: {audio_path_root}")
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
        f.write(str(audio_path_root))

    print(f"\nWrote {len(items)} items to {output_path}")
    print(f"Unique speakers: {len({i['speaker'] for i in items})}")
    print(f"Unique phone sequences: {len({i['phone_sequence'] for i in items})}")
    print(f"Unique accent patterns: {len({i['accent_pattern'] for i in items})}")


if __name__ == "__main__":
    main()
