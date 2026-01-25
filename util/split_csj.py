"""Split CSJ audio files into Inter-Pausal Units (IPUs).

This script reads the CSJ database to identify IPU boundaries and splits the
original talk audio files into separate files for each IPU. This allows for
more efficient feature extraction and processing.

The output filenames will be format: {TalkID}_{IPUID}.wav
"""

import argparse
import os
import sqlite3
from pathlib import Path
from collections import defaultdict
import concurrent.futures

import torch
import torchaudio
from tqdm import tqdm


def get_ipu_segments(db_path, talk_ids=None):
    """Fetch IPU segments from the database."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    query = "SELECT TalkID, IPUID, StartTime, EndTime FROM segIPU ORDER BY TalkID, StartTime"
    cur.execute(query)
    
    segments = defaultdict(list)
    for talk_id, ipu_id, start, end in cur.fetchall():
        if talk_ids and talk_id not in talk_ids:
            continue
        segments[talk_id].append({
            'ipu_id': ipu_id,
            'start': start,
            'end': end
        })
    
    conn.close()
    return segments


def process_talk(talk_id, ipus, audio_dir, output_dir, extension=".wav", pad_right=0.0):
    """Process a single talk: load audio, split into IPUs, save files."""
    input_path = audio_dir / f"{talk_id}{extension}"
    
    if not input_path.exists():
        raise FileNotFoundError(f"{talk_id}: Audio file not found at {input_path}")

    try:
        # Load the entire audio file
        # Using torchaudio.load. Note: this loads into memory.
        # CSJ talks can be long, but usually fit in RAM (a few hundred MBs).
        audio, sample_rate = torchaudio.load(str(input_path))
    except Exception as e:
        raise RuntimeError(f"Error loading {talk_id}: {e}") from e

    # Create output directory for this talk or flat?
    # User requirement: "separate audio files for each IPU".
    # Flat structure {TalkID}_{IPUID}.wav is probably easiest for downstream tools
    # as long as filesystem handles it (CSJ has ~3000 talks, many IPUs... 
    # might be too many files for one dir? 
    # CSJ has ~0.5M IPUs. A single dir with 500k files is bad.
    # Better to use subdirectories by TalkID.
    
    talk_output_dir = output_dir / talk_id
    talk_output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for ipu in ipus:
        ipu_id = ipu['ipu_id']
        start_time = ipu['start']
        end_time = ipu['end']

        # Convert time to frames
        start_frame = int(start_time * sample_rate)
        # Add padding (context) to the right
        end_frame = int((end_time + pad_right) * sample_rate)
        
        # Clamp to valid range
        start_frame = max(0, start_frame)
        end_frame = min(audio.shape[1], end_frame)

        if start_frame >= end_frame:
            continue

        # Slice audio
        segment = audio[:, start_frame:end_frame]
        
        # Output filename: {TalkID}_{IPUID}.wav
        # (Unique name is safer if we ever move files.)
        output_filename = f"{talk_id}_{ipu_id}{extension}"
        output_path = talk_output_dir / output_filename
        
        torchaudio.save(str(output_path), segment, sample_rate)
        count += 1

    return f"Processed {talk_id}: {count} IPUs"


def split_csj(db_path, audio_dir, output_dir, workers=4, pad_right=0.0):
    """Main function to split CSJ audio."""
    db_path = Path(db_path)
    audio_dir = Path(audio_dir)
    output_dir = Path(output_dir)
    
    print(f"Reading IPU segments from {db_path}...")
    ipus_by_talk = get_ipu_segments(db_path)
    talk_ids = sorted(ipus_by_talk.keys())
    print(f"Found {len(talk_ids)} talks with IPUs.")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Process talks in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_talk, tid, ipus_by_talk[tid], audio_dir, output_dir, ".wav", pad_right): tid
            for tid in talk_ids
        }
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(talk_ids), desc="Splitting talks"):
            result = future.result()
            # print(result) # Optional: print verbose status

    print(f"Splitting complete. Output in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split CSJ audio into IPUs.")
    parser.add_argument("--db", type=str, default="/work/smcintosh/data/CSJ/csj.db",
                        help="Path to CSJ database")
    parser.add_argument("--audio", type=str, required=True,
                        help="Path to original CSJ audio directory")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for split audio files")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers")
    parser.add_argument("--pad-right", type=float, default=0.2,
                        help="Seconds of context to include after each IPU (default: 0.2s)")
    
    args = parser.parse_args()
    
    split_csj(args.db, args.audio, args.output, args.workers, args.pad_right)
