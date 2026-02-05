#!/usr/bin/env python3
"""Synthesize English stress minimal pairs using Kokoro TTS.

Generates WAV files for noun/verb stress pairs.
Output structure matches /home/sunhaitong/ABX_syn/data/standard_english_stress.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from kokoro import KPipeline
from tqdm import tqdm


# Voice mapping (List of voices to use)
VOICES = [
    "af_heart",    # Female
    "am_fenrir",   # Male
    "af_bella",    # Female
    "am_michael",  # Male
]

# IPA transcriptions for noun/verb stress minimal pairs
# Generated from sentences in english_minimal_pair_sentences.txt using misaki G2P
IPA_MAP = {
    "address": {
        "noun": "ˈædɹˌɛs",
        "verb": "ədɹˈɛs",
    },
    "abstract": {
        "noun": "ˈæbstɹˌækt",
        "verb": "æbstɹˈækt",
    },
    "conduct": {
        "noun": "kˈɑndˌʌkt",
        "verb": "kəndˈʌkt",
    },
    "discharge": {
        "noun": "dˈɪsʧˌɑɹʤ",
        "verb": "dɪsʧˈɑɹʤ",
    },
    "discount": {
        "noun": "dˈɪskˌWnt",
        "verb": "dˌɪskˈWnt",
    },
    "impact": {
        "noun": "ˈɪmpˌækt",
        "verb": "ɪmpˈækt",
    },
    "import": {
        "noun": "ˈɪmpˌɔɹt",
        "verb": "ɪmpˈɔɹt",
    },
    "increase": {
        "noun": "ˈɪnkɹˌis",
        "verb": "ɪnkɹˈis",
    },
    "insert": {
        "noun": "ˈɪnsˌɜɹt",
        "verb": "ɪnsˈɜɹt",
    },
    "insult": {
        "noun": "ˈɪnsˌʌlt",
        "verb": "ɪnsˈʌlt",
    },
    "permit": {
        "noun": "pˈɜɹmɪt",
        "verb": "pəɹmˈɪt",
    },
    "project": {
        "noun": "pɹˈɑʤˌɛkt",
        "verb": "pɹəʤˈɛkt",
    },
    "survey": {
        "noun": "sˈɜɹvˌA",
        "verb": "sɜɹvˈA",
    },
    "transport": {
        "noun": "tɹˈænspˌɔɹt",
        "verb": "tɹænspˈɔɹt",
    },
    "upset": {
        "noun": "ˈʌpsˌɛt",
        "verb": "ˌʌpsˈɛt",
    },
}

SAMPLE_RATE = 24000  # Kokoro native sample rate


def get_args():
    parser = argparse.ArgumentParser(
        description="Synthesize English stress minimal pairs using Kokoro TTS"
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("synth_data/kokoro_english_stress"),
        help="Output directory for generated files",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def synthesize(pipeline, word, ipa, voice):
    """Synthesize word using Kokoro TTS.

    Args:
        pipeline: KPipeline instance
        word: The orthographic word
        ipa: IPA transcription of the word
        voice: Kokoro voice name

    Returns:
        tuple: (full_audio, phonemes)
    """
    # Direct IPA specification using markdown syntax
    # Format: [displayed_text](/phonemes/)
    text = f"I say, [{word}](/{ipa}/), again."

    # Generate audio - pipeline returns generator of (graphemes, phonemes, audio)
    generator = pipeline(text, voice=voice, speed=1.0)

    # Collect all segments (usually just one for short text)
    all_audio = []
    all_phonemes = []

    for gs, ps, audio in generator:
        all_audio.append(audio)
        all_phonemes.append(ps)

    # Concatenate audio
    full_audio = np.concatenate(all_audio) if len(all_audio) > 1 else all_audio[0]
    phonemes = " ".join(all_phonemes)

    return full_audio, phonemes


def main():
    args = get_args()

    # Set random seed
    np.random.seed(args.seed)

    # Create output directory
    args.output_path.mkdir(parents=True, exist_ok=True)

    # Initialize Kokoro pipeline
    print("Initializing Kokoro TTS pipeline...")
    pipeline = KPipeline(lang_code='a')  # American English

    # Calculate total iterations for progress bar
    total = len(IPA_MAP) * 2 * len(VOICES)  # words * (noun+verb) * voices

    metadata = []
    
    # Create voice directories
    for voice_name in VOICES:
        (args.output_path / voice_name).mkdir(parents=True, exist_ok=True)

    with tqdm(total=total, desc="Synthesizing") as pbar:
        for word, forms in IPA_MAP.items():
            for pos in ["noun", "verb"]:
                ipa = forms[pos]

                for voice_name in VOICES:
                    voice_dir = args.output_path / voice_name
                    
                    # Generate filename
                    wav_filename = f"{word}_{pos}.wav"
                    wav_path = voice_dir / wav_filename
                    lab_filename = f"{word}_{pos}.lab"
                    lab_path = voice_dir / lab_filename

                    # Skip if already exists
                    if wav_path.exists() and lab_path.exists():
                        pbar.update(1)
                        continue

                    # Synthesize
                    try:
                        audio, phonemes = synthesize(
                            pipeline, word, ipa, voice_name
                        )

                        # Save audio
                        sf.write(str(wav_path), audio, SAMPLE_RATE)
                        
                        # Save lab file
                        # Minimal pair carrier phrase: "I say [word], again."
                        lab_content = f"I say {word} again."
                        with open(lab_path, "w") as f:
                            f.write(lab_content)

                        # Record metadata
                        metadata.append({
                            "word": word,
                            "pos": pos,
                            "voice_name": voice_name,
                            "ipa": ipa,
                            "audio_path": str(wav_path.relative_to(args.output_path)),
                            "duration": len(audio) / SAMPLE_RATE,
                            "phonemes": phonemes,
                        })

                    except Exception as e:
                        print(f"Error synthesizing {word} ({pos}, {voice_name}): {e}")

                    pbar.update(1)

    # Save metadata
    if metadata:
        df = pd.DataFrame(metadata)
        df.to_csv(args.output_path / "metadata.csv", index=False)
        print(f"Saved metadata to {args.output_path / 'metadata.csv'}")

    print(f"Generated {len(metadata)} audio files in {args.output_path}")


if __name__ == "__main__":
    main()
