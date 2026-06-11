#!/usr/bin/env python3
"""Synthesize English lexical stress audio and metadata."""

from __future__ import annotations

import argparse
import csv
import wave
from pathlib import Path

from google.cloud import texttospeech


OUTPUT_ROOT = Path("data/stress_syn")
METADATA_COLUMNS = [
    "id",
    "audio_file",
    "speaker",
    "text",
    "target",
    "target_onset",
    "target_offset",
    "label",
    "lexical_category",
]
LABELS = {
    "noun": "1",
    "verb": "2",
}

VOICES = {
    "A": "en-US-Standard-A",
    "B": "en-US-Standard-B",
    "C": "en-US-Standard-C",
    "D": "en-US-Standard-D",
}

LANG = "en-US"


def synthesize(
    client,
    audio_config,
    word: str,
    ipa: str,
    voice_name: str,
    out_wav: Path,
) -> None:
    ssml = f"""
    <speak>
      <phoneme alphabet="ipa" ph="{ipa}">
        {word}
      </phoneme>
    </speak>
    """

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


IPA_MAP = {
    "abstract": {
        "noun": "æbˌstɹækt",
        "verb": "ˌæbˈstɹækt",  # use non-weakened form
    },
    "address": {
        "noun": "ˈædɹɛs",
        "verb": "əˈdɹɛs",  # weakening expected
    },
    "conduct": {
        "noun": "ˈkɒndʌkt",
        "verb": "kənˈdʌkt",  # weakening expected
    },
    "discharge": {
        "noun": "ˈdɪstʃɑɹdʒ",
        "verb": "dɪsˈtʃɑɹdʒ"
    },
    "discount": {
        "noun": "ˈdɪskaʊnt",
        "verb": "dɪˈskaʊnt"
    },
    "impact": {
        "noun": "ˈɪmpækt",
        "verb": "ɪmˈpækt"
    },
    "import": {
        "noun": "ˈɪm.pɔɹt",
        "verb": "ɪmˈpɔɹt"
    },
    "increase": {
        "noun": "ˈɪnkɹiːs",
        "verb": "ɪnˈkɹiːs"
    },
    "insert": {
        "noun": "ˈɪnsɝt",
        "verb": "ɪnˈsɝt"
    },
    "insult": {
        "noun": "ˈɪnsʌlt",
        "verb": "ɪnˈsʌlt"
    },
    "permit": {
        "noun": "ˈpɝmɪt",
        "verb": "pɝˈmɪt"
    },
    "project": {
        "noun": "ˈpɹɑˌd͡ʒɛkt",
        "verb": "pɹəˈd͡ʒɛkt",  # weakening expected
    },
    "survey": {
        "noun": "ˈsɝˌveɪ",
        "verb": "sɝˈveɪ"
    },
    "transport": {
        "noun": "ˈtɹæns.pɔɹt",
        "verb": "tɹænsˈpɔɹt"
    },
    "upset": {
        "noun": "ˈʌpsɛt",
        "verb": "ˌʌpˈsɛt",
    },
}


def get_duration_seconds(audio_path: Path) -> float:
    """Return WAV duration in seconds."""
    with wave.open(str(audio_path), "rb") as wf:
        return wf.getnframes() / wf.getframerate()


def write_metadata(rows: list[dict[str, object]], metadata_path: Path) -> None:
    """Write metadata for the synthesized English stress data."""
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthesize English lexical stress audio."
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

    client = texttospeech.TextToSpeechClient()
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
    )

    metadata_rows = []

    for suffix, voice_name in VOICES.items():
        speaker = f"EN_TTS_{suffix}"
        item_number = 1

        for word, forms in IPA_MAP.items():
            for lexical_category in ["noun", "verb"]:
                ipa = forms[lexical_category]
                if not ipa:
                    raise ValueError(f"IPA missing for {word} ({lexical_category})")

                item_id = f"{speaker}_{item_number:03d}"
                wav_path = audio_dir / f"{item_id}.wav"

                if not wav_path.exists():
                    synthesize(
                        client=client,
                        audio_config=audio_config,
                        word=word,
                        ipa=ipa,
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
                        "text": word,
                        "target": word,
                        "target_onset": 0.0,
                        "target_offset": get_duration_seconds(wav_path),
                        "label": LABELS[lexical_category],
                        "lexical_category": lexical_category,
                    }
                )
                item_number += 1

    write_metadata(metadata_rows, metadata_path)
    print(f"Wrote {len(metadata_rows)} metadata rows to {metadata_path}")


if __name__ == "__main__":
    main()
