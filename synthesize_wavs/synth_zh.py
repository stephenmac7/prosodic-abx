#!/usr/bin/env python3
"""Synthesize Mandarin tone audio and metadata."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from google.cloud import texttospeech


INPUT_CSV = Path("synthesize_wavs/zh_syn.csv")
OUTPUT_ROOT = Path("data/tone_syn")
METADATA_COLUMNS = ["id", "audio_file", "speaker", "target", "label"]

VOICES = {
    "A": "cmn-CN-Standard-A",
    "B": "cmn-CN-Standard-B",
    "C": "cmn-CN-Standard-C",
    "D": "cmn-CN-Standard-D",
}

LANG = "cmn-CN"


def normalize_pinyin_for_ssml(pinyin: str) -> str:
    """Normalize pinyin for Google TTS SSML input."""
    s = pinyin

    # Special rewrites for pinyin strings that Google TTS handles better with
    # full syllable spellings.
    s = re.sub(r"^iu$", "yu", s)
    s = re.sub(r"^ui$", "wei", s)
    s = re.sub(r"^ie$", "ye", s)
    s = re.sub(r"^ong$", "yong", s)
    s = re.sub(r"^ü$", "yu", s)
    s = re.sub(r"^üe$", "yue", s)
    s = re.sub(r"^ün$", "yun", s)
    s = re.sub(r"^u$", "wu", s)

    if s.startswith("u") or s.startswith("i"):
        s = "y" + s

    return s


def load_tone_items(input_csv: Path) -> list[tuple[str, str]]:
    """Load pinyin-tone pairs for Mandarin tone synthesis."""
    pairs = []
    with input_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"target", "label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns in {input_csv}: {sorted(missing)}")

        for row in reader:
            target = row["target"].strip()
            label = row["label"].strip()
            if target and label:
                pairs.append((target, label))

    return pairs


def synthesize(
    client,
    audio_config,
    pinyin: str,
    tone: str,
    voice_name: str,
    out_wav: Path,
) -> None:
    """Synthesize one Mandarin syllable with a tone label."""
    pinyin_ssml = normalize_pinyin_for_ssml(pinyin)
    ssml = f"""
    <speak>
      <phoneme alphabet="pinyin" ph="{pinyin_ssml}{tone}">{pinyin_ssml}</phoneme>
    </speak>
    """.strip()

    synthesis_input = texttospeech.SynthesisInput(ssml=ssml)

    voice = texttospeech.VoiceSelectionParams(
        language_code=LANG,
        name=voice_name,
    )

    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config,
    )

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    out_wav.write_bytes(response.audio_content)
    print(f"Generated: {out_wav}")


def write_metadata(rows: list[dict[str, str]], metadata_path: Path) -> None:
    """Write metadata in the same shape as data/tone/metadata.csv."""
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthesize Mandarin tone audio."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_CSV,
        help=f"Input CSV path (default: {INPUT_CSV})",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help=f"Output data directory (default: {OUTPUT_ROOT})",
    )
    args = parser.parse_args()

    audio_dir = args.output_root / "audio"
    metadata_path = args.output_root / "metadata.csv"
    tone_items = load_tone_items(args.input)

    client = texttospeech.TextToSpeechClient()
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
    )

    metadata_rows = []

    for suffix, voice_name in VOICES.items():
        speaker = f"ZH_TTS_{suffix}"

        for pinyin, tone in tone_items:
            item_id = f"{speaker}_{pinyin}{tone}"
            wav_path = audio_dir / f"{item_id}.wav"

            if not wav_path.exists():
                synthesize(
                    client=client,
                    audio_config=audio_config,
                    pinyin=pinyin,
                    tone=tone,
                    voice_name=voice_name,
                    out_wav=wav_path,
                )

            if not wav_path.exists():
                continue

            metadata_rows.append(
                {
                    "id": item_id,
                    "audio_file": f"audio/{item_id}.wav",
                    "speaker": speaker,
                    "target": pinyin,
                    "label": tone,
                }
            )

    write_metadata(metadata_rows, metadata_path)
    print(f"Wrote {len(metadata_rows)} metadata rows to {metadata_path}")


if __name__ == "__main__":
    main()
