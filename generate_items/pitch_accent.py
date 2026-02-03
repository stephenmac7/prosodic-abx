"""Generate fastabx-compatible item file for Japanese pitch accent ABX from recordings.

Word boundaries are determined from manual timestamp overrides, falling back to
MFA-aligned TextGrids.

Supports two modes:

1. CLIPPED mode (default):
   Cuts word segments from sentence recordings and saves to abx_items/pitch_accent/audio/.
   The item file references these clips with onset=0, offset=duration.

2. IN-CONTEXT mode (--in-context):
   The item file references full sentence audio files with onset/offset pointing
   to the target word boundaries.

The output format matches run_abx.py expectations:
  - #file: path relative to audio_root (without extension)
  - onset / offset: in seconds
  - phone_sequence: hiragana sequence (used for BY condition)
  - accent_pattern: pitch accent label (used for ON condition)
  - speaker: speaker ID

Filename format: S007_87_きって_切手.wav
  - parts[0]: speaker (S007)
  - parts[1]: utterance number (87)
  - parts[2]: phone sequence in hiragana (きって)
  - parts[3]: kanji representation (切手)
"""

from __future__ import annotations

import argparse
import csv
import json
import wave
from pathlib import Path
from typing import Optional

import tgt
import torchaudio
from torchcodec.decoders import AudioDecoder

# Hardcoded input paths
ANN_JSON = Path(__file__).parent / "metadata" / "pitch_accent_annotations.json"
AUDIO_ROOT_SENTENCES = Path("/home/sunhaitong/ABX_jp/data/recording")
TEXTGRID_ROOT = Path("/home/sunhaitong/ABX_jp/data/recording_MFA_aligned")
SENTENCE_FILE = Path("/home/sunhaitong/ABX_jp/data/recording_sentences_words")
MANUAL_TIMESTAMPS = Path(__file__).parent / "metadata" / "pitch_accent_recording_manual_timestamp.json"
OUTPUT_DIR = Path("abx_items/pitch_accent")

# ============================================================
# ===================== AUDIO CLIPPING =======================
# ============================================================

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


def load_manual_timestamps(path: Path) -> dict[str, dict[str, float]]:
    """Load manual timestamp overrides from JSON file."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# ===================== MFA TEXTGRID PARSING =================
# ============================================================

def parse_words_tier(textgrid_path: Path) -> list[tgt.Interval]:
    """Parse an MFA TextGrid and return intervals from the 'words' tier."""
    textgrid = tgt.io.read_textgrid(str(textgrid_path))
    words_tier = textgrid.get_tier_by_name("words")
    return list(words_tier.intervals)


def find_target_span(
    intervals: list[tgt.Interval],
    target_tokens: list[str],
) -> Optional[tuple[float, float]]:
    """
    Find time span corresponding to target_tokens in words tier.

    Matching rules:
    - Tokens must appear in order.
    - Empty intervals (text == "") between tokens are allowed.
    - Returned span includes everything between first and last token.
    """
    texts = [iv.text for iv in intervals]

    for i, txt in enumerate(texts):
        if txt != target_tokens[0]:
            continue

        matched = [i]
        cur = i
        ok = True

        for tok in target_tokens[1:]:
            j = cur + 1
            while j < len(texts) and texts[j] == "":
                j += 1
            if j >= len(texts) or texts[j] != tok:
                ok = False
                break
            matched.append(j)
            cur = j

        if ok:
            return (intervals[matched[0]].start_time, intervals[matched[-1]].end_time)

    return None


def load_sentence_targets(sentence_file: Path) -> dict[int, list[str]]:
    """
    Load sentence file and return mapping from utterance number (1-indexed)
    to target token list for TextGrid matching.
    """
    lines = sentence_file.read_text(encoding="utf-8").splitlines()
    targets = {}

    for idx, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue

        parts = line.split("。")
        if len(parts) < 3:
            raise ValueError(f"Line {idx} must contain TWO '。': {line}")

        target_text = parts[-2].strip()  # e.g. "気 が"
        tokens = [t for t in target_text.split() if t]
        targets[idx] = tokens

    return targets


def parse_filename(fname: str) -> tuple[str, str, str, str]:
    """Parse filename like S007_87_きって_切手.wav into components."""
    stem = Path(fname).stem
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected filename format: {fname}")
    speaker = parts[0]
    label = parts[1]
    seq = parts[2]

    # Append context ID (derived from utterance number) to ensure we only
    # compare minimal pairs from the same context ((1,2), (3,4), etc.)
    try:
        utt_num = int(label)
    except ValueError:
        raise ValueError(f"Utterance number (part 2 of filename) must be an integer, got '{label}' in filename: {fname}")

    context_id = (utt_num - 1) // 2
    seq = f"{seq}_{context_id}"

    return speaker, label, seq, stem


def get_duration_seconds(audio_path: Path) -> float:
    with wave.open(str(audio_path), "rb") as wf:
        return wf.getnframes() / wf.getframerate()


def load_annotations(ann_json: Path) -> list[dict]:
    with ann_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Annotation JSON must be a list of items")
    return data


def build_items(
    annotations: list[dict],
    sentence_audio_root: Path,
    textgrid_root: Path,
    manual_timestamps: dict[str, dict[str, float]],
    sentence_targets: dict[int, list[str]],
    output_audio_dir: Optional[Path] = None,
) -> list[dict]:
    """
    Build item rows from annotations.

    If output_audio_dir is provided (clipped mode): clips audio from sentence
    recordings and saves to output directory. Items reference clipped files
    with onset=0.

    If output_audio_dir is None (in-context mode): items reference original
    sentence files with actual onset/offset timestamps.

    Timestamps come from manual overrides first, then MFA TextGrids.
    """
    items: list[dict] = []
    skipped_decision = 0
    used_manual = 0
    clipped_count = 0

    for entry in annotations:
        # Only include items where decision == True
        if entry.get("decision") is not True:
            skipped_decision += 1
            continue

        fname = entry["filename"]
        speaker, utt_num_str, seq, stem = parse_filename(fname)
        utt_num = int(utt_num_str)

        # Original sentence files use format: S007/S007_87.wav
        sentence_stem = f"{speaker}_{utt_num:02d}"
        sentence_audio_path = sentence_audio_root / speaker / f"{sentence_stem}.wav"
        textgrid_path = textgrid_root / speaker / f"{sentence_stem}.TextGrid"

        if not sentence_audio_path.exists():
            raise FileNotFoundError(f"Sentence audio not found: {sentence_audio_path}")

        # Check for manual timestamp override first
        if fname in manual_timestamps:
            onset = manual_timestamps[fname]["onset"]
            offset = manual_timestamps[fname]["offset"]
            used_manual += 1
        else:
            if not textgrid_path.exists():
                raise FileNotFoundError(
                    f"TextGrid not found and no manual timestamp for {fname}: {textgrid_path}"
                )

            # Get target tokens for this utterance
            target_tokens = sentence_targets.get(utt_num)
            if target_tokens is None:
                raise ValueError(f"No sentence target defined for utterance {utt_num}")

            # Parse TextGrid and find word span
            intervals = parse_words_tier(textgrid_path)
            span = find_target_span(intervals, target_tokens)

            if span is None:
                raise ValueError(
                    f"Target span not found in TextGrid for {fname}: "
                    f"looking for {target_tokens} in {textgrid_path}"
                )

            onset, offset = span

        if output_audio_dir is not None:
            # Clipped mode: cut audio and save to output directory
            output_audio_path = output_audio_dir / speaker / fname
            if not output_audio_path.exists():
                clip_audio(sentence_audio_path, output_audio_path, onset, offset)
                clipped_count += 1

            # Get duration of clipped file
            duration = get_duration_seconds(output_audio_path)

            items.append(
                {
                    "#file": f"{speaker}/{stem}",
                    "onset": 0.0,
                    "offset": duration,
                    "phone_sequence": seq,
                    "accent_pattern": utt_num_str,
                    "speaker": speaker,
                }
            )
        else:
            # In-context mode: reference sentence file with actual timestamps
            items.append(
                {
                    "#file": f"{speaker}/{sentence_stem}",
                    "onset": onset,
                    "offset": offset,
                    "phone_sequence": seq,
                    "accent_pattern": utt_num_str,
                    "speaker": speaker,
                }
            )

    if skipped_decision > 0:
        print(f"Skipped {skipped_decision} items with decision != True")
    if used_manual > 0:
        print(f"Used {used_manual} manual timestamp overrides")
    if clipped_count > 0:
        print(f"Clipped {clipped_count} new audio files")

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
        description="Generate fastabx item file for Japanese pitch accent ABX"
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
        help="Output directory (default: abx_items/pitch_accent or abx_items/pitch_accent_in_context)",
    )
    args = parser.parse_args()

    # Determine output directory
    if args.output_dir is not None:
        output_dir = args.output_dir
    elif args.in_context:
        output_dir = Path("abx_items/pitch_accent_in_context")
    else:
        output_dir = OUTPUT_DIR

    # Determine audio root based on mode
    if args.in_context:
        audio_root = AUDIO_ROOT_SENTENCES
        mode_str = "in-context (sentence audio with MFA timestamps)"
    else:
        audio_root = output_dir / "audio"
        mode_str = "clipped (cut from sentences using MFA + manual timestamps)"

    print(f"Mode: {mode_str}")
    print(f"Annotation file: {ANN_JSON}")
    print(f"Sentence audio root: {AUDIO_ROOT_SENTENCES}")
    print(f"TextGrid root: {TEXTGRID_ROOT}")
    print(f"Sentence file: {SENTENCE_FILE}")
    print(f"Manual timestamps: {MANUAL_TIMESTAMPS}")
    if not args.in_context:
        print(f"Output audio directory: {audio_root}")
    print(f"Output directory: {output_dir}")
    print()

    annotations = load_annotations(ANN_JSON)
    print(f"Loaded {len(annotations)} annotations")

    manual_timestamps = load_manual_timestamps(MANUAL_TIMESTAMPS)
    print(f"Loaded {len(manual_timestamps)} manual timestamp overrides")

    sentence_targets = load_sentence_targets(SENTENCE_FILE)
    print(f"Loaded {len(sentence_targets)} sentence target definitions")

    items = build_items(
        annotations=annotations,
        sentence_audio_root=AUDIO_ROOT_SENTENCES,
        textgrid_root=TEXTGRID_ROOT,
        manual_timestamps=manual_timestamps,
        sentence_targets=sentence_targets,
        output_audio_dir=None if args.in_context else audio_root,
    )

    output_path = output_dir / "items.csv"
    write_items(items, output_path)

    # Write audio path for extract_features.py to find
    with (output_dir / "audio_path.txt").open("w") as f:
        f.write(str(audio_root))

    print(f"\nWrote {len(items)} items to {output_path}")
    print(f"Unique speakers: {len({i['speaker'] for i in items})}")
    print(f"Unique phone sequences: {len({i['phone_sequence'] for i in items})}")
    print(f"Unique accent patterns: {len({i['accent_pattern'] for i in items})}")


if __name__ == "__main__":
    main()
