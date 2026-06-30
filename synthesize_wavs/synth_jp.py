#!/usr/bin/env python3
"""Synthesize Japanese pitch accent audio and metadata."""

from __future__ import annotations

import argparse
import csv
import wave
from pathlib import Path

from google.cloud import texttospeech

INPUT_CSV = Path("synthesize_wavs/jp_syn.csv")
OUTPUT_ROOT = Path("data/pitch_accent_syn")
METADATA_COLUMNS = [
    "id",
    "audio_file",
    "speaker",
    "target",
    "surface_kana",
    "label",
    "context_id",
]

VOICES = {
    "A": "ja-JP-Standard-A",
    "B": "ja-JP-Standard-B",
    "C": "ja-JP-Standard-C",
    "D": "ja-JP-Standard-D",
}
LANG = "ja-JP"


def synthesize_one(client, ph: str, text: str, voice_name: str, out_wav: Path) -> None:
    ssml = f"""
    <speak>
      <phoneme alphabet="yomigana" ph="{ph}">
        {text}
      </phoneme>
    </speak>
    """.strip()

    synthesis_input = texttospeech.SynthesisInput(ssml=ssml)
    voice = texttospeech.VoiceSelectionParams(language_code=LANG, name=voice_name)
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
    )

    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config,
    )

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    out_wav.write_bytes(response.audio_content)
    print(f"Generated: {out_wav}")


def get_duration_seconds(audio_path: Path) -> float:
    """Return WAV duration in seconds."""
    with wave.open(str(audio_path), "rb") as wf:
        return wf.getnframes() / wf.getframerate()


def load_items(input_csv: Path) -> list[dict[str, str]]:
    """Load Japanese synthesis items."""
    with input_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"idx", "kana", "text", "ph", "label", "context_id"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing columns: {sorted(missing)}")
        return list(reader)


def write_metadata(rows: list[dict[str, object]], metadata_path: Path) -> None:
    """Write metadata in the same shape as the released synthetic corpus."""
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthesize Japanese pitch accent audio."
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

    if not args.input.exists():
        raise FileNotFoundError(f"CSV not found: {args.input}")

    audio_dir = args.output_root / "audio"
    metadata_path = args.output_root / "metadata.csv"

    client = texttospeech.TextToSpeechClient()
    items = load_items(args.input)
    metadata_rows = []

    for row in items:
        idx = row["idx"].strip()
        kana = row["kana"].strip()
        text = row["text"].strip()
        ph = row["ph"].strip()
        label = row["label"].strip()
        context_id = row["context_id"].strip()

        if not idx or not kana or not text or not ph:
            continue

        for suffix, voice_name in VOICES.items():
            speaker = f"JP_TTS_{suffix}"
            item_id = f"{speaker}_{idx}_{kana}_{text}"
            wav_path = audio_dir / f"{item_id}.wav"

            if not wav_path.exists():
                synthesize_one(
                    client,
                    ph=ph,
                    text=text,
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
                    "target": text,
                    "surface_kana": kana,
                    "label": label,
                    "context_id": context_id,
                }
            )

    write_metadata(metadata_rows, metadata_path)
    print(f"Wrote {len(metadata_rows)} metadata rows to {metadata_path}")


if __name__ == "__main__":
    main()
