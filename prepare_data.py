"""Download the released Prosodic ABX data and prepare local files."""

from __future__ import annotations

import argparse
import csv
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


DATASET_ID = "HaitongSUN/prosody-abx"
OUTPUT_ROOT = Path("data")
AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
TONE_METADATA_COLUMNS = ["id", "audio_file", "speaker", "target", "label"]

DATASETS = {
    "stress": {
        "parquet": "english_stress/english_stress.parquet",
        "columns": [
            "id",
            "audio_file",
            "speaker",
            "text",
            "target",
            "target_onset",
            "target_offset",
            "label",
            "context_set",
            "lexical_category",
        ],
    },
    "pitch_accent": {
        "parquet": "japanese_pitch_accent/japanese_pitch_accent.parquet",
        "columns": [
            "id",
            "audio_file",
            "speaker",
            "text",
            "target",
            "surface_kana",
            "target_onset",
            "target_offset",
            "label",
            "context_id",
        ],
    },
}


def download_parquet(filename: str) -> Path:
    """Download a parquet file from the Hugging Face dataset repository."""
    return Path(
        hf_hub_download(
            repo_id=DATASET_ID,
            repo_type="dataset",
            filename=filename,
        )
    )


def get_audio_bytes(row: dict[str, Any], parquet_path: Path) -> bytes:
    """Return audio bytes from a Hugging Face Audio-style parquet row."""
    audio = row.get("audio")
    if not isinstance(audio, dict):
        raise ValueError(f"Expected audio to be a struct in {row.get('id')}")

    data = audio.get("bytes")
    if data is not None:
        if isinstance(data, memoryview):
            return data.tobytes()
        return bytes(data)

    audio_path = audio.get("path")
    if audio_path:
        path = Path(audio_path)
        if not path.is_absolute():
            path = parquet_path.parent / path
        if path.exists():
            return path.read_bytes()

    raise ValueError(f"No audio bytes or readable audio path for {row.get('id')}")


def prepare_dataset(name: str, spec: dict[str, Any]) -> None:
    """Extract one parquet file into data/<name>/audio and metadata.csv."""
    parquet_path = download_parquet(spec["parquet"])
    output_dir = OUTPUT_ROOT / name
    audio_dir = output_dir / "audio"
    metadata_path = output_dir / "metadata.csv"

    if output_dir.exists():
        shutil.rmtree(output_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)

    rows = pq.read_table(parquet_path).to_pylist()
    rows.sort(key=lambda row: row["id"])

    with metadata_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=spec["columns"])
        writer.writeheader()

        for row in rows:
            wav_name = f"{row['id']}.wav"
            wav_path = audio_dir / wav_name
            wav_path.write_bytes(get_audio_bytes(row, parquet_path))

            metadata_row = {
                column: row[column]
                for column in spec["columns"]
                if column != "audio_file"
            }
            metadata_row["audio_file"] = f"audio/{wav_name}"
            writer.writerow(metadata_row)

    print(f"Wrote {len(rows)} rows to {output_dir}")


def prepare_tone() -> None:
    """Prepare manually downloaded MCAE-Monosyllable files."""
    import pandas as pd

    output_dir = OUTPUT_ROOT / "tone"
    neutral_zip = output_dir / "neutral.zip"
    tone_pinyin_info = output_dir / "tone and pinyin information.xlsx"
    audio_dir = output_dir / "audio"
    tone_pinyin_csv = output_dir / "tone_and_pinyin_information.csv"

    missing = [
        path
        for path in (neutral_zip, tone_pinyin_info)
        if not path.exists()
    ]
    if missing:
        missing_text = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Missing Mandarin tone files. Download MCAE-monosyllable from "
            "https://osf.io/h3uem and place:\n"
            "  - data/tone/neutral.zip\n"
            "  - data/tone/tone and pinyin information.xlsx\n"
            f"Missing:\n{missing_text}"
        )

    if audio_dir.exists():
        shutil.rmtree(audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)

    audio_count = 0
    with zipfile.ZipFile(neutral_zip) as zf:
        members = [
            member
            for member in zf.infolist()
            if not member.is_dir()
            and not member.filename.startswith("__MACOSX/")
            and Path(member.filename).suffix.lower() in AUDIO_EXTENSIONS
        ]
        if not members:
            raise ValueError(f"No audio files found in {neutral_zip}")

        roots = {Path(member.filename).parts[0] for member in members}
        strip_root = len(roots) == 1

        for member in members:
            rel = Path(member.filename)
            if strip_root and len(rel.parts) > 1:
                rel = Path(*rel.parts[1:])
            dst = audio_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, dst.open("wb") as f:
                shutil.copyfileobj(src, f)
            audio_count += 1

    if audio_count == 0:
        raise ValueError(f"No audio files found in {neutral_zip}")

    df = pd.read_excel(tone_pinyin_info, engine="openpyxl")
    df.to_csv(tone_pinyin_csv, index=False)
    write_tone_metadata(tone_pinyin_csv, audio_dir, output_dir / "metadata.csv")

    print(f"Wrote {audio_count} tone audio files to {audio_dir}")
    print(f"Wrote tone/pinyin metadata to {tone_pinyin_csv}")


def write_tone_metadata(csv_path: Path, audio_dir: Path, output_path: Path) -> None:
    """Write clean Mandarin tone metadata from MCAE tone/pinyin information."""
    mapping = {}
    empty_pinyin = 0

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["filename"].strip()
            pinyin = row["pinyin"].strip()
            if not pinyin:
                empty_pinyin += 1
                continue
            mapping[filename] = pinyin

    pinyin_tone_speakers: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    records = []

    for audio_path in sorted(audio_dir.glob("*.wav")):
        stem = audio_path.stem
        if stem not in mapping:
            continue

        emotion_code = stem[1:3]
        if emotion_code not in {"01", "02", "03", "04"}:
            continue

        speaker_num = int(stem[0])
        speaker = f"ZH_S{speaker_num:03d}"
        tone = str(int(emotion_code))
        pinyin = mapping[stem]

        pinyin_tone_speakers[pinyin][tone].add(speaker)
        records.append(
            {
                "id": stem,
                "audio_file": f"audio/{audio_path.name}",
                "speaker": speaker,
                "target": pinyin,
                "label": tone,
            }
        )

    clean_pinyins = set()
    for pinyin, tone_dict in pinyin_tone_speakers.items():
        if all(len(tone_dict.get(tone, set())) >= 3 for tone in ["1", "2", "3", "4"]):
            clean_pinyins.add(pinyin)

    records = [row for row in records if row["target"] in clean_pinyins]

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TONE_METADATA_COLUMNS)
        writer.writeheader()
        writer.writerows(records)

    print(f"Wrote {len(records)} clean Mandarin tone rows to {output_path}")
    print(f"Clean pinyins: {len(clean_pinyins)}")
    if empty_pinyin:
        print(f"Skipped {empty_pinyin} rows with empty pinyin")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare local Prosodic ABX data.")
    parser.add_argument(
        "--tone",
        action="store_true",
        help="Prepare manually downloaded MCAE-monosyllable files in data/tone.",
    )
    args = parser.parse_args()

    for name, spec in DATASETS.items():
        prepare_dataset(name, spec)
    if args.tone:
        prepare_tone()


if __name__ == "__main__":
    main()
