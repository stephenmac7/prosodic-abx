#!/usr/bin/env python3
from pathlib import Path
from google.cloud import texttospeech



OUT_ROOT = Path("/home/sunhaitong/ABX_syn/data/standard_english_stress")

VOICES = {
    "A": "en-US-Standard-A",
    "B": "en-US-Standard-B",
    "C": "en-US-Standard-C",
    "D": "en-US-Standard-D",
}

LANG = "en-US"



client = texttospeech.TextToSpeechClient()

audio_config = texttospeech.AudioConfig(
    audio_encoding=texttospeech.AudioEncoding.LINEAR16,
    sample_rate_hertz=16000
)


def synthesize(word, pos, ipa, voice_name, out_wav):
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
        name=voice_name
    )

    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )

    out_wav.write_bytes(response.audio_content)
    print(f"Generated: {out_wav}")

IPA_MAP = {
    "abstract": {
        "noun": "æbˌstɹækt",  
        "verb": "ˌæbˈstɹækt"
    },
    "address": {
        "noun": "ˈædɹɛs",
        "verb": "əˈdɹɛs"
    },
    "conduct": {
        "noun": "ˈkɒndʌkt",
        "verb": "kənˈdʌkt"
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
        "verb": "pɚˈmɪt"
    },
    "project": {
        "noun": "ˈpɹɑˌd͡ʒɛkt",
        "verb": "pɹəˈd͡ʒɛkt"
    },
    "survey": {
        "noun": "ˈsɝˌveɪ",
        "verb": "sɚˈveɪ"
    },
    "transport": {
        "noun": "ˈtɹæns.pɔɹt",
        "verb": "tɹænsˈpɔɹt"
    },
    "upset": {
        "noun": "ˈʌpsɛt",
        "verb": "ˌʌpˈsɛt"
    }
}


for word, forms in IPA_MAP.items():
    out_dir = OUT_ROOT / word
    out_dir.mkdir(parents=True, exist_ok=True)

    for pos in ["noun", "verb"]:
        ipa = forms[pos]
        if not ipa:
            raise ValueError(f"IPA missing for {word} ({pos})")

        for v_suffix, voice_name in VOICES.items():
            out_wav = out_dir / f"{word}_{pos}_{v_suffix}.wav"
            if out_wav.exists():
                continue

            synthesize(word, pos, ipa, voice_name, out_wav)
