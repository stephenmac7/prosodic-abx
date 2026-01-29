import argparse
import json
from pathlib import Path

# Paths
INTERVAL_LOG = Path(__file__).parent / "intervalLog.txt"
MANUAL_JSON = Path(__file__).parent / "stress_recording_manual_timestamps.json"

def parse_interval_log(path: Path) -> dict:
    new_data = {}
    if not path.exists():
        return new_data
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Line format: "1. Sound S001_import_B_1\t0.395...\t0.818..."
            parts = line.split("\t")
            if len(parts) != 3:
                print(f"Skipping malformed line: {line}")
                continue
            
            name_part, onset_str, offset_str = parts
            
            # Parse filename from "1. Sound S001_import_B_1"
            # We expect the filename to be S001_import_B_1.wav
            tokens = name_part.split()
            if len(tokens) < 3 or tokens[1] != "Sound":
                 print(f"Skipping unexpected name format: {name_part}")
                 continue
            
            filename_stem = tokens[2]
            filename = f"{filename_stem}.wav"
            
            onset = float(onset_str)
            offset = float(offset_str)
            
            new_data[filename] = {"onset": onset, "offset": offset}
            
    return new_data

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Merge Praat interval log with manual timestamps JSON.\n\n"
            "The log file (intervalLog.txt) should be created in Praat using "
            "the following logging format:\n"
            "  'editor$''tab$''t1''tab$''t2'\n\n"
            "Newer entries in the log will override existing entries in the JSON."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.parse_args()

    if not INTERVAL_LOG.exists():
        print(f"Error: {INTERVAL_LOG} not found.")
        return

    # Load existing manual timestamps
    if MANUAL_JSON.exists():
        with MANUAL_JSON.open("r", encoding="utf-8") as f:
            existing_data = json.load(f)
        print(f"Loaded {len(existing_data)} existing manual timestamps.")
    else:
        existing_data = {}
        print("No existing manual timestamps found. Creating new.")

    # Parse new data
    new_data = parse_interval_log(INTERVAL_LOG)
    print(f"Parsed {len(new_data)} entries from interval log.")

    # Merge: Update existing with new data
    updates = 0
    adds = 0
    for fname, times in new_data.items():
        if fname in existing_data:
            existing_data[fname] = times
            updates += 1
        else:
            existing_data[fname] = times
            adds += 1

    print(f"Merged data: {adds} added, {updates} updated.")

    # Sort keys for clean output
    sorted_data = dict(sorted(existing_data.items()))

    # Write back
    with MANUAL_JSON.open("w", encoding="utf-8") as f:
        json.dump(sorted_data, f, indent=2)
    
    print(f"Wrote {len(sorted_data)} entries to {MANUAL_JSON}")

if __name__ == "__main__":
    main()
