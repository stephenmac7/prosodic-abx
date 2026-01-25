#!/usr/bin/env python3
import json
from pathlib import Path
from google.cloud import texttospeech



JSON_PATH = "/home/sunhaitong/ABX_tone/data/tone_abx_triples.json"
OUT_ROOT = Path("/home/sunhaitong/ABX_syn/data/standard_mandarin_syllable")

VOICES = {
    "A": "cmn-CN-Standard-A",
    "B": "cmn-CN-Standard-B",
    "C": "cmn-CN-Standard-C",
    "D": "cmn-CN-Standard-D",
}

LANG = "cmn-CN"



client = texttospeech.TextToSpeechClient()

audio_config = texttospeech.AudioConfig(
    audio_encoding=texttospeech.AudioEncoding.LINEAR16,
    sample_rate_hertz=16000
)



def synthesize(pinyin, tone, voice_name, out_wav):
    ssml = f"""
    <speak>
      <phoneme alphabet="pinyin" ph="{pinyin}{tone}">
        {pinyin}{tone}
      </phoneme>
    </speak>
    """

    synthesis_input = texttospeech.SynthesisInput(ssml=ssml)

    voice = texttospeech.VoiceSelectionParams(
        language_code=LANG,
        name=voice_name
    )

    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )

    out_wav.write_bytes(response.audio_content)
    print(f"Generated: {out_wav}")


with open(JSON_PATH, "r", encoding="utf-8") as f:
    triples = json.load(f)

seen = set()  # (pinyin, tone)

for item in triples:
    pinyin = item["pinyin"]
    tones = {item["toneA"], item["toneB"]}

    out_dir = OUT_ROOT / pinyin
    out_dir.mkdir(parents=True, exist_ok=True)

    for tone in tones:
        key = (pinyin, tone)
        if key in seen:
            continue
        seen.add(key)

        for v_suffix, voice_name in VOICES.items():
            out_wav = out_dir / f"{pinyin}{tone}_{v_suffix}.wav"
            if out_wav.exists():
                continue

            synthesize(pinyin, tone, voice_name, out_wav)
