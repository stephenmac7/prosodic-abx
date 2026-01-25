#!/usr/bin/env python3
import csv
import re
from pathlib import Path
from google.cloud import texttospeech

TABLE_CSV = Path("/home/sunhaitong/prosody-abx/synthesize_wavs/ja_yomigana_table.csv")  # <-- your edited copy
OUT_ROOT = Path("/home/sunhaitong/prosody-abx/synthesize_wavs/standard_Japanese_accent")

VOICES = {
    "A": "ja-JP-Standard-A",
    "B": "ja-JP-Standard-B",
    "C": "ja-JP-Standard-C",
    "D": "ja-JP-Standard-D",
}
LANG = "ja-JP"

def safe_name(s: str) -> str:
    return s.strip()

def synthesize_one(client, ph: str, text: str, voice_name: str, out_wav: Path):
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
        sample_rate_hertz=16000
    )

    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    out_wav.write_bytes(response.audio_content)
    print(f"Generated: {out_wav}")

def main():
    if not TABLE_CSV.exists():
        raise FileNotFoundError(f"CSV not found: {TABLE_CSV}")

    client = texttospeech.TextToSpeechClient()

    with TABLE_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"kana", "kanji", "ph", "text", "idx"}

        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing columns: {sorted(missing)}")

        for row in reader:
            kana = row["kana"].strip()
            kanji = row["kanji"].strip()
            ph = row["ph"].strip()
            text = row["text"].strip() or kanji
            idx = row["idx"].strip()


            if not kana or not kanji or not ph:
                # skip incomplete rows
                continue

            # Put each item into kana subfolder (you asked pinyin-folder style before; kana works similarly)
            out_dir = OUT_ROOT / safe_name(kana)

            base = f"{safe_name(kana)}_{safe_name(kanji)}_{idx}"

            for suffix, voice_name in VOICES.items():
                out_wav = out_dir / f"{base}_{suffix}.wav"
                if out_wav.exists():
                    continue
                synthesize_one(client, ph=ph, text=text, voice_name=voice_name, out_wav=out_wav)

if __name__ == "__main__":
    main()
