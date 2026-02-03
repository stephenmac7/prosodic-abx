"""Generate fastabx-compatible item file for stress ABX from Kokoro TTS outputs.

Supports two modes:

1. OUT-OF-CONTEXT mode (default):
   Clips word segments from carrier phrase audio using MFA-aligned timestamps.
   Audio clips are saved to abx_items/stress_kokoro/audio/.
   The item file references these clips with onset=0, offset=duration.

2. IN-CONTEXT mode (--in-context):
   Uses the original carrier phrase recordings with MFA-aligned timestamps.
   The item file references full audio files with onset/offset pointing
   to the target word boundaries.

Input structure (kokoro_english_stress/):
  {voice}/{word}_{pos}.wav      - Audio file
  {voice}/{word}_{pos}.TextGrid - MFA alignment

Output format matches run_abx.py expectations:
  - #file: path relative to audio_root (without extension)
  - onset / offset: in seconds
  - phone_sequence: word (used for BY condition)
  - accent_pattern: pos (noun/verb, used for ON condition)
  - speaker: voice name
  - context_set: not used (set to "A")
  - part_of_word: not used (set to "1")
"""

from __future__ import annotations

import argparse
import csv
import wave
from pathlib import Path

import tgt
import torchaudio
from torchcodec.decoders import AudioDecoder

# Default paths
INPUT_DIR = Path("synth_data/kokoro_english_stress")
OUTPUT_DIR = Path("abx_items/stress_kokoro")

# Target words (for validation)
TARGET_WORDS = [
    "abstract", "address", "conduct", "discharge", "discount",
    "impact", "import", "increase", "insert", "insult",
    "permit", "project", "survey", "transport", "upset"
]


def clip_audio(src: Path, dst: Path, onset: float, offset: float) -> None:
    """Cut a segment from a WAV file."""
    decoder = AudioDecoder(str(src))
    sr = decoder.metadata.sample_rate
    duration = decoder.metadata.duration_seconds
    total_frames = int(duration * sr)

    start_frame = max(0, int(round(onset * sr)))
    if offset > 0:
        end_frame = min(total_frames, int(round(offset * sr)))
    else:
        end_frame = total_frames

    num_frames = max(0, end_frame - start_frame)

    waveform, _ = torchaudio.load(
        str(src), frame_offset=start_frame, num_frames=num_frames
    )

    dst.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(dst), waveform, sr)


def parse_words_tier(textgrid_path: Path) -> list[tgt.Interval]:
    """Parse an MFA TextGrid and return intervals from the 'words' tier."""
    textgrid = tgt.io.read_textgrid(str(textgrid_path))
    words_tier = textgrid.get_tier_by_name("words")
    return list(words_tier.intervals)


def find_word_interval(
    intervals: list[tgt.Interval],
    target_word: str,
) -> tuple[float, float] | None:
    """Find time span for target word in words tier (case-insensitive)."""
    target_lower = target_word.lower()
    for iv in intervals:
        if iv.text.lower() == target_lower:
            return (iv.start_time, iv.end_time)
    return None


def get_duration_seconds(audio_path: Path) -> float:
    """Get duration of WAV file in seconds."""
    with wave.open(str(audio_path), "rb") as wf:
        return wf.getnframes() / wf.getframerate()


def parse_filename(fname: str) -> tuple[str, str]:
    """Parse filename like abstract_noun.wav into (word, pos)."""
    stem = Path(fname).stem
    parts = stem.rsplit("_", 1)
    if len(parts) != 2:
        raise ValueError(f"Unexpected filename format: {fname}")
    word, pos = parts
    return word, pos


def discover_files(input_dir: Path) -> list[tuple[Path, Path, str]]:
    """Discover all WAV/TextGrid pairs in input directory.

    Returns list of (wav_path, textgrid_path, voice) tuples.
    """
    files = []
    for voice_dir in sorted(input_dir.iterdir()):
        if not voice_dir.is_dir():
            continue
        voice = voice_dir.name
        for wav_path in sorted(voice_dir.glob("*.wav")):
            textgrid_path = wav_path.with_suffix(".TextGrid")
            if textgrid_path.exists():
                files.append((wav_path, textgrid_path, voice))
    return files


def build_items_clipped(
    files: list[tuple[Path, Path, str]],
    output_audio_dir: Path,
) -> list[dict]:
    """Build item rows by clipping word segments from carrier phrase audio."""
    items: list[dict] = []
    skipped_no_word = 0
    clipped_count = 0

    for wav_path, textgrid_path, voice in files:
        word, pos = parse_filename(wav_path.name)

        if word not in TARGET_WORDS:
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
            print(f"Warning: Word '{word}' not found in {textgrid_path}")
            skipped_no_word += 1
            continue

        onset, offset = span

        # Clip audio and save
        output_filename = f"{word}_{pos}.wav"
        output_path = output_audio_dir / voice / output_filename
        if not output_path.exists():
            clip_audio(wav_path, output_path, onset, offset)
            clipped_count += 1

        # Get duration of clipped file
        duration = get_duration_seconds(output_path)
        file_id = f"{voice}/{word}_{pos}"

        items.append({
            "#file": file_id,
            "onset": 0.0,
            "offset": duration,
            "phone_sequence": word,
            "accent_pattern": pos,
            "speaker": voice,
            "context_set": "A",
            "part_of_word": "1",
        })

    if skipped_no_word > 0:
        print(f"Skipped {skipped_no_word} items where word not found in TextGrid")
    if clipped_count > 0:
        print(f"Clipped {clipped_count} new audio files")

    return items


def build_items_in_context(
    files: list[tuple[Path, Path, str]],
    input_dir: Path,
) -> list[dict]:
    """Build item rows using original audio with MFA timestamps."""
    items: list[dict] = []
    skipped_no_word = 0

    for wav_path, textgrid_path, voice in files:
        word, pos = parse_filename(wav_path.name)

        if word not in TARGET_WORDS:
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
            print(f"Warning: Word '{word}' not found in {textgrid_path}")
            skipped_no_word += 1
            continue

        onset, offset = span
        file_id = f"{voice}/{word}_{pos}"

        items.append({
            "#file": file_id,
            "onset": onset,
            "offset": offset,
            "phone_sequence": word,
            "accent_pattern": pos,
            "speaker": voice,
            "context_set": "A",
            "part_of_word": "1",
        })

    if skipped_no_word > 0:
        print(f"Skipped {skipped_no_word} items where word not found in TextGrid")

    return items


def write_items(items: list[dict], output_path: Path) -> None:
    """Write items to CSV file."""
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
        description="Generate fastabx item file for stress ABX from Kokoro TTS outputs"
    )
    parser.add_argument(
        "--in-context",
        action="store_true",
        help="Use original carrier phrase audio with MFA timestamps instead of clipped word files",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=INPUT_DIR,
        help=f"Input directory with Kokoro outputs (default: {INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: abx_items/stress_kokoro or abx_items/stress_kokoro_in_context)",
    )
    args = parser.parse_args()

    # Determine output directory
    if args.output_dir is not None:
        output_dir = args.output_dir
    elif args.in_context:
        output_dir = Path("abx_items/stress_kokoro_in_context")
    else:
        output_dir = OUTPUT_DIR

    # Determine audio root based on mode
    if args.in_context:
        audio_root = args.input_dir
        mode_str = "in-context (carrier phrase audio with MFA timestamps)"
    else:
        audio_root = output_dir / "audio"
        mode_str = "out-of-context (clipped word segments)"

    print(f"Mode: {mode_str}")
    print(f"Input directory: {args.input_dir}")
    if not args.in_context:
        print(f"Output audio directory: {audio_root}")
    print(f"Output directory: {output_dir}")
    print()

    # Discover input files
    files = discover_files(args.input_dir)
    print(f"Found {len(files)} WAV/TextGrid pairs")

    if args.in_context:
        items = build_items_in_context(files, args.input_dir)
    else:
        items = build_items_clipped(files, audio_root)

    output_path = output_dir / "items.csv"
    write_items(items, output_path)

    # Write audio path for extract_features.py
    with (output_dir / "audio_path.txt").open("w") as f:
        f.write(str(audio_root.resolve()))

    print(f"\nWrote {len(items)} items to {output_path}")
    print(f"Unique speakers: {len({i['speaker'] for i in items})}")
    print(f"Unique words: {len({i['phone_sequence'] for i in items})}")
    print(f"Accent patterns: {sorted({i['accent_pattern'] for i in items})}")


if __name__ == "__main__":
    main()
