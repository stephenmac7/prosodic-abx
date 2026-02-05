#!/usr/bin/env python3
"""Quick test script for Kokoro TTS synthesis.

Usage:
    uv run python synthesize_wavs/test_kokoro.py "word" "IPA"
    uv run python synthesize_wavs/test_kokoro.py survey "sˈɜɹvˌA"
    uv run python synthesize_wavs/test_kokoro.py survey "səɹvˈA"
"""

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from kokoro import KPipeline

SAMPLE_RATE = 24000


def main():
    parser = argparse.ArgumentParser(description="Test Kokoro TTS synthesis")
    parser.add_argument("word", help="Word to synthesize")
    parser.add_argument("ipa", help="IPA transcription")
    parser.add_argument("--voice", default="af_bella", help="Voice name")
    parser.add_argument("--output", "-o", type=Path, help="Output WAV file (plays if not specified)")
    args = parser.parse_args()

    pipeline = KPipeline(lang_code='a')

    text = f"I say, [{args.word}](/{args.ipa}/), again."
    print(f"Synthesizing: {text}")

    audio_parts = []
    for gs, ps, audio in pipeline(text, voice=args.voice, speed=1.0):
        print(f"  Phonemes: {ps}")
        audio_parts.append(audio)

    full_audio = np.concatenate(audio_parts) if len(audio_parts) > 1 else audio_parts[0]

    if args.output:
        sf.write(str(args.output), full_audio, SAMPLE_RATE)
        print(f"Saved to {args.output}")
    else:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, full_audio, SAMPLE_RATE)
            print(f"Playing {f.name}...")
            subprocess.run(["aplay", f.name], check=False)


if __name__ == "__main__":
    main()
