#!/usr/bin/env python3
"""Build Kokoro English lexical stress data with MFA word alignment.

The script writes:
  data/stress_kokoro/raw/      carrier-sentence WAV and LAB files for MFA
  data/stress_kokoro/aligned/  MFA TextGrid files
  data/stress_kokoro/audio/    clipped target-word WAV files
  data/stress_kokoro/metadata.csv
"""

import argparse
import csv
import os
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
import soundfile as sf
import tgt
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
        "verb": "ədɹˈɛs", # weakening expected
    },
    "abstract": {
        "noun": "ˈæbstɹˌækt",
        "verb": "æbstɹˈækt", # use non-weakened form
    },
    "conduct": {
        "noun": "kˈɑndˌʌkt",
        "verb": "kəndˈʌkt", # weakening expected
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
        "verb": "pɜɹmˈɪt",
    },
    "project": {
        "noun": "pɹˈɑʤˌɛkt",
        "verb": "pɹəʤˈɛkt", # weakening expected
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
OUTPUT_ROOT = Path("data/stress_kokoro")
METADATA_COLUMNS = [
    "id",
    "audio_file",
    "speaker",
    "target",
    "label",
    "lexical_category",
]
LABELS = {
    "noun": "1",
    "verb": "2",
}

def get_args():
    parser = argparse.ArgumentParser(
        description="Build Kokoro English lexical stress data"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help=f"Output data directory (default: {OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--mfa-command",
        default="mfa",
        help="MFA command or executable path (default: mfa)",
    )
    parser.add_argument(
        "--dictionary",
        default="english_us_arpa",
        help="MFA dictionary name or path (default: english_us_arpa)",
    )
    parser.add_argument(
        "--acoustic-model",
        default="english_us_arpa",
        help="MFA acoustic model name or path (default: english_us_arpa)",
    )
    parser.add_argument(
        "--skip-mfa",
        action="store_true",
        help="Skip MFA alignment and use existing TextGrids in aligned/.",
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


def run_mfa(
    mfa_command: str,
    corpus_dir: Path,
    dictionary: str,
    acoustic_model: str,
    aligned_dir: Path,
) -> None:
    """Run MFA alignment with an external mfa command."""
    executable = shutil.which(mfa_command) if "/" not in mfa_command else mfa_command
    if executable is None:
        raise FileNotFoundError(
            f"MFA command not found: {mfa_command}. Install MFA separately and "
            "pass --mfa-command /path/to/mfa if it is not on PATH."
        )

    aligned_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    mfa_bin = str(Path(executable).resolve().parent)
    env["PATH"] = f"{mfa_bin}{os.pathsep}{env.get('PATH', '')}"

    cmd = [
        executable,
        "align",
        str(corpus_dir),
        dictionary,
        acoustic_model,
        str(aligned_dir),
        "--clean",
        "--overwrite",
    ]
    print("Running MFA:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def find_word_interval(textgrid_path: Path, target_word: str) -> tuple[float, float]:
    """Find target-word onset and offset in an MFA TextGrid."""
    textgrid = tgt.io.read_textgrid(str(textgrid_path))
    words_tier = textgrid.get_tier_by_name("words")
    target = target_word.lower()

    for interval in words_tier.intervals:
        if interval.text.lower() == target:
            return interval.start_time, interval.end_time

    raise ValueError(f"Target word '{target_word}' not found in {textgrid_path}")


def clip_audio(src: Path, dst: Path, onset: float, offset: float) -> None:
    """Cut a segment from a WAV file."""
    audio, sr = sf.read(str(src), always_2d=True)
    start = max(0, int(round(onset * sr)))
    end = min(len(audio), int(round(offset * sr)))
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), audio[start:end], sr)


def get_duration_seconds(audio_path: Path) -> float:
    """Return WAV duration in seconds."""
    with wave.open(str(audio_path), "rb") as wf:
        return wf.getnframes() / wf.getframerate()


def write_metadata(rows: list[dict[str, object]], metadata_path: Path) -> None:
    """Write final clipped-word metadata."""
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def synthesize_raw_corpus(raw_dir: Path, seed: int) -> list[dict[str, str]]:
    """Synthesize carrier sentences and write MFA LAB files."""
    np.random.seed(seed)

    print("Initializing Kokoro TTS pipeline...")
    pipeline = KPipeline(lang_code='a')  # American English
    total = len(IPA_MAP) * 2 * len(VOICES)
    raw_rows = []

    with tqdm(total=total, desc="Synthesizing") as pbar:
        for voice_name in VOICES:
            speaker = f"EN_KOKORO_{voice_name}"

            for word, forms in IPA_MAP.items():
                for lexical_category in ["noun", "verb"]:
                    ipa = forms[lexical_category]
                    item_id = f"{speaker}_{word}_{lexical_category}"
                    wav_path = raw_dir / f"{item_id}.wav"
                    lab_path = raw_dir / f"{item_id}.lab"

                    if not wav_path.exists() or not lab_path.exists():
                        audio, _phonemes = synthesize(pipeline, word, ipa, voice_name)
                        wav_path.parent.mkdir(parents=True, exist_ok=True)
                        sf.write(str(wav_path), audio, SAMPLE_RATE)
                        lab_path.write_text(f"I say {word} again.", encoding="utf-8")

                    raw_rows.append(
                        {
                            "id": item_id,
                            "speaker": speaker,
                            "text": f"I say {word} again.",
                            "target": word,
                            "label": LABELS[lexical_category],
                            "lexical_category": lexical_category,
                        }
                    )

                    pbar.update(1)

    return raw_rows


def build_clipped_data(
    raw_rows: list[dict[str, str]],
    raw_dir: Path,
    aligned_dir: Path,
    audio_dir: Path,
) -> list[dict[str, object]]:
    """Cut target words using MFA TextGrids and build final metadata."""
    metadata_rows = []

    for row in raw_rows:
        item_id = row["id"]
        raw_wav = raw_dir / f"{item_id}.wav"
        textgrid = aligned_dir / f"{item_id}.TextGrid"
        clipped_wav = audio_dir / f"{item_id}.wav"

        onset, offset = find_word_interval(textgrid, row["target"])
        if not clipped_wav.exists():
            clip_audio(raw_wav, clipped_wav, onset, offset)

        metadata_rows.append(
            {
                "id": item_id,
                "audio_file": f"audio/{item_id}.wav",
                "speaker": row["speaker"],
                "target": row["target"],
                "label": row["label"],
                "lexical_category": row["lexical_category"],
            }
        )

    return metadata_rows


def main() -> None:
    args = get_args()

    raw_dir = args.output_root / "raw"
    aligned_dir = args.output_root / "aligned"
    audio_dir = args.output_root / "audio"
    metadata_path = args.output_root / "metadata.csv"

    raw_dir.mkdir(parents=True, exist_ok=True)
    aligned_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = synthesize_raw_corpus(raw_dir, args.seed)

    if not args.skip_mfa:
        run_mfa(
            mfa_command=args.mfa_command,
            corpus_dir=raw_dir,
            dictionary=args.dictionary,
            acoustic_model=args.acoustic_model,
            aligned_dir=aligned_dir,
        )

    metadata_rows = build_clipped_data(raw_rows, raw_dir, aligned_dir, audio_dir)
    write_metadata(metadata_rows, metadata_path)

    print(f"Wrote {len(metadata_rows)} metadata rows to {metadata_path}")


if __name__ == "__main__":
    main()
