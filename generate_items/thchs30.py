"""Generate fastabx-compatible item file for THCHS30 Mandarin tone ABX.

Supports two modes:

1. CLIPPED mode (default):
   Scans pre-extracted syllable audio files and produces a CSV item file with
   full-token segments (onset=0, offset=duration).

2. IN-CONTEXT mode (--in-context):
   Scans the original sentence recordings with MFA-aligned TextGrids.
   The item file references full sentence audio files with onset/offset pointing
   to the syllable boundaries from MFA alignment.

The ABX task is: ON tone, BY pinyin, ACROSS speaker
- Discriminate tones for the same base syllable (pinyin) across different speakers.

Source data:
- Pre-extracted tokens: /home/sunhaitong/ABX_tone/data_thchs30/thchs_pinyins
- Combined data (wav+TextGrid): /home/sunhaitong/ABX_tone/data_thchs30/thchs30_combined

Exclusion rules (following extract_abx_thchs30.py):
- Exclude yi, qi, ba, bu pinyins (sandhi complications)
- Exclude tone 5 (neutral tone)
- Third-tone sandhi: exclude left token in adjacent 3-3 pairs
- Minimum duration filter (30ms)

Output format (CSV for run_abx.py):
  - #file: path relative to audio_root (without extension)
  - onset / offset: in seconds
  - phone_sequence: base pinyin (used for BY condition)
  - accent_pattern: tone (used for ON condition)
  - speaker: speaker ID
"""

from __future__ import annotations

import argparse
import csv
import wave
from pathlib import Path

import tgt

# Input paths
AUDIO_ROOT_CLIPPED = Path("/home/sunhaitong/ABX_tone/data_thchs30/thchs_pinyins")
AUDIO_ROOT_CONTEXT = Path("/home/sunhaitong/ABX_tone/data_thchs30/thchs30_combined")
OUTPUT_DIR = Path("abx_items/thchs30")

# Processing parameters
SYLLABLE_TIER_NAME = "words"  # Name of the syllable tier in TextGrid
MIN_DURATION = 0.03  # Minimum duration in seconds (30ms)

# Exclusion rules
EXCLUDED_PINYINS = {"yi", "qi", "ba", "bu"}
EXCLUDED_TONES = {0, 5}  # 0 = no tone digit, 5 = neutral tone


def parse_pinyin_tone(label: str) -> tuple[str, int]:
    """Parse 'hao3' -> ('hao', 3)."""
    if not label or not label[-1].isdigit():
        return label, 0
    return label[:-1], int(label[-1])


def get_duration_seconds(audio_path: Path) -> float:
    """Get audio file duration in seconds."""
    with wave.open(str(audio_path), "rb") as wf:
        return wf.getnframes() / wf.getframerate()


def parse_clipped_filename(filename: str) -> dict | None:
    """Parse clipped filename like 'ai1_spk13_utt13_593_pos10_occ1.wav'.

    Returns dict with label, base, tone, speaker, or None if parse fails.
    """
    stem = Path(filename).stem
    parts = stem.split("_")

    if len(parts) < 2:
        return None

    label = parts[0]
    base, tone = parse_pinyin_tone(label)

    # Find speaker (spkX)
    speaker = None
    for p in parts[1:]:
        if p.startswith("spk"):
            speaker = p[3:]
            break

    if speaker is None:
        return None

    return {
        "label": label,
        "base": base,
        "tone": tone,
        "speaker": speaker,
        "stem": stem,
    }


def build_items_clipped(
    audio_root: Path,
    min_duration: float,
) -> list[dict]:
    """Build item rows by scanning pre-extracted syllable audio files.

    Directory structure: {audio_root}/{base_pinyin}/{label}_spk{spk}_*.wav
    """
    items: list[dict] = []
    skipped_excluded = 0
    skipped_short = 0
    skipped_parse = 0

    for base_dir in sorted(audio_root.iterdir()):
        if not base_dir.is_dir():
            continue

        base_pinyin = base_dir.name

        # Skip excluded pinyins
        if base_pinyin in EXCLUDED_PINYINS:
            continue

        for wav_path in sorted(base_dir.glob("*.wav")):
            info = parse_clipped_filename(wav_path.name)
            if info is None:
                skipped_parse += 1
                continue

            # Skip excluded tones
            if info["tone"] in EXCLUDED_TONES:
                skipped_excluded += 1
                continue

            # Check duration
            duration = get_duration_seconds(wav_path)
            if duration < min_duration:
                skipped_short += 1
                continue

            file_id = f"{base_pinyin}/{info['stem']}"

            items.append({
                "#file": file_id,
                "onset": 0.0,
                "offset": duration,
                "phone_sequence": info["base"],
                "accent_pattern": str(info["tone"]),
                "speaker": info["speaker"],
            })

    if skipped_parse > 0:
        print(f"Skipped {skipped_parse} files due to parse errors")
    if skipped_excluded > 0:
        print(f"Skipped {skipped_excluded} files with excluded tones")
    if skipped_short > 0:
        print(f"Skipped {skipped_short} files shorter than {min_duration*1000:.0f}ms")

    return items


def apply_exclusions(tokens: list[dict]) -> list[bool]:
    """Apply exclusion rules and return keep mask.

    Rules:
    - Exclude yi/qi/ba/bu pinyins
    - Exclude tone 5 (neutral tone)
    - Third-tone sandhi: if adjacent tokens are 3-3, exclude left one
    """
    keep = [True] * len(tokens)

    # Apply pinyin and tone exclusions
    for i, tok in enumerate(tokens):
        if tok["base"] in EXCLUDED_PINYINS:
            keep[i] = False
        elif tok["tone"] in EXCLUDED_TONES:
            keep[i] = False

    # Third-tone sandhi exclusion
    for i in range(len(tokens) - 1):
        if tokens[i]["tone"] == 3 and tokens[i + 1]["tone"] == 3:
            keep[i] = False

    return keep


def build_items_in_context(
    audio_root: Path,
    min_duration: float,
) -> list[dict]:
    """Build item rows by scanning sentence audio with TextGrid timestamps.

    Directory structure: {audio_root}/{speaker}/{utt_id}.wav + .TextGrid
    """
    items: list[dict] = []
    skipped_missing_tg = 0
    skipped_short = 0
    skipped_excluded = 0

    for speaker_dir in sorted(audio_root.iterdir()):
        if not speaker_dir.is_dir():
            continue

        speaker = speaker_dir.name

        for tg_path in sorted(speaker_dir.glob("*.TextGrid")):
            utt_id = tg_path.stem
            wav_path = speaker_dir / f"{utt_id}.wav"

            if not wav_path.exists():
                skipped_missing_tg += 1
                continue

            # Load TextGrid intervals
            try:
                tg = tgt.io.read_textgrid(str(tg_path))
                tier = tg.get_tier_by_name(SYLLABLE_TIER_NAME)
                intervals = [(iv.start_time, iv.end_time, iv.text) for iv in tier.intervals]
            except Exception as e:
                print(f"Warning: Failed to parse {tg_path}: {e}")
                continue

            # Parse tokens from non-empty intervals
            tokens = []
            for start, end, label in intervals:
                if not label:  # Skip empty intervals
                    continue
                base, tone = parse_pinyin_tone(label)
                tokens.append({"label": label, "base": base, "tone": tone, "start": start, "end": end})

            if not tokens:
                continue

            # Apply exclusion rules
            keep = apply_exclusions(tokens)

            file_duration = get_duration_seconds(wav_path)
            file_id = f"{speaker}/{utt_id}"

            for i, tok in enumerate(tokens):
                if not keep[i]:
                    skipped_excluded += 1
                    continue

                start, end = tok["start"], tok["end"]

                # Check duration
                if (end - start) < min_duration:
                    skipped_short += 1
                    continue

                items.append({
                    "#file": file_id,
                    "onset": start,
                    "offset": end,
                    "phone_sequence": tok["base"],
                    "accent_pattern": str(tok["tone"]),
                    "speaker": speaker,
                })

    if skipped_missing_tg > 0:
        print(f"Skipped {skipped_missing_tg} utterances with missing TextGrid")
    if skipped_excluded > 0:
        print(f"Skipped {skipped_excluded} tokens due to exclusion rules")
    if skipped_short > 0:
        print(f"Skipped {skipped_short} tokens shorter than {min_duration*1000:.0f}ms")

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
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for item in items:
            writer.writerow(item)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate fastabx item file for THCHS30 Mandarin tone ABX",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--in-context",
        action="store_true",
        help="Use original sentence recordings with TextGrid timestamps instead of clipped syllable files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: abx_items/thchs30 or abx_items/thchs30_in_context)",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=MIN_DURATION,
        help="Minimum token duration in seconds",
    )
    args = parser.parse_args()

    # Determine output directory
    if args.output_dir is not None:
        output_dir = args.output_dir
    elif args.in_context:
        output_dir = Path("abx_items/thchs30_in_context")
    else:
        output_dir = OUTPUT_DIR

    # Determine audio root based on mode
    if args.in_context:
        audio_root = AUDIO_ROOT_CONTEXT
        mode_str = "in-context (sentence audio with TextGrid timestamps)"
    else:
        audio_root = AUDIO_ROOT_CLIPPED
        mode_str = "clipped (pre-extracted syllable files)"

    print(f"Mode: {mode_str}")
    print(f"Audio root: {audio_root}")
    print(f"Output directory: {output_dir}")
    print(f"Min duration: {args.min_duration*1000:.0f}ms")
    print()

    # Build items based on mode
    if args.in_context:
        items = build_items_in_context(
            audio_root=audio_root,
            min_duration=args.min_duration,
        )
    else:
        items = build_items_clipped(
            audio_root=audio_root,
            min_duration=args.min_duration,
        )

    # Write output
    output_path = output_dir / "items.csv"
    write_items(items, output_path)

    # Write audio path for extract_features.py to find
    with (output_dir / "audio_path.txt").open("w") as f:
        f.write(str(audio_root))

    print(f"\nWrote {len(items)} items to {output_path}")
    print(f"Unique speakers: {len({i['speaker'] for i in items})}")
    print(f"Unique pinyin: {len({i['phone_sequence'] for i in items})}")
    print(f"Unique tones: {sorted(set(i['accent_pattern'] for i in items))}")


if __name__ == "__main__":
    main()
