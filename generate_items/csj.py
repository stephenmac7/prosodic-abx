"""Generate ABX item files for pitch accent discrimination from CSJ database.

This script generates item files compatible with the fastabx library for running
ABX discrimination tests on Japanese pitch accent patterns. The output is a CSV
file with columns:
  - #file: TalkID (references the audio/feature file)
  - onset: Start time in seconds
  - offset: End time in seconds
  - phone_sequence: Space-separated phone sequence (e.g., "k_a n_a")
  - accent_pattern: Binary string representing accent on each mora (e.g., "010")
  - speaker: Speaker ID
  - prev_phone: Preceding phone (for strict context matching)
  - next_phone: Following phone (for strict context matching)
  - num_moras: Number of moras in the sequence
"""

import sqlite3
import os
import csv
import argparse
import random
from collections import defaultdict

# Paths
BASE_DIR = "/work/smcintosh/data/CSJ"
SPLIT_DIR = "/work/smcintosh/data/CSJ_by_IPU"  # Directory with IPU-split audio files from split_csj.py
DB_PATH = os.path.join(BASE_DIR, "csj.db")
OUTPUT_DIR = "abx_items"

# Nucleus phones (Vowels + Moraic Nasal + Long Vowel)
NUCLEUS_PHONES = {'a', 'i', 'u', 'e', 'o', 'N', 'H'}

# Phones that shouldn't start/end an AP
SPECIAL_PHONE_LEADS = {'Q', 'H', 'N', '<cl>'}


def get_core_talks(data_dir, talk_id=None):
    """Identify core TalkIDs from WAV files in the data directory."""
    core_talks = set()
    for f in os.listdir(data_dir):
        if f.endswith(".wav"):
            core_talks.add(f.replace(".wav", ""))

    if talk_id:
        if talk_id in core_talks:
            core_talks = {talk_id}
            print(f"Filtering for single talk: {talk_id}")
        else:
            raise ValueError(f"TalkID {talk_id} not found in file list.")

    return core_talks


def fetch_speaker_mappings(cur):
    """Fetch all TalkID to SpeakerID mappings."""
    cur.execute("SELECT TalkID, SpeakerID FROM infoTalk")
    return {row[0]: row[1] for row in cur.fetchall()}


def fetch_valid_point_accents(cur, core_talks):
    """Fetch valid point accents (ToneLabel='A') that are not uncertain.

    Returns a dict: talk_id -> sorted list of timestamps.
    """
    if not core_talks:
        return {}

    talk_str = ",".join(["'" + t + "'" for t in core_talks])
    query = f"""
        SELECT TalkID, Time
        FROM pointTone
        WHERE TalkID IN ({talk_str})
          AND ToneLabel = 'A'
          AND (F0Uncertain IS NULL OR F0Uncertain = 0)
          AND (CategoryUncertain IS NULL OR CategoryUncertain = 0)
          AND (PositionUncertain IS NULL OR PositionUncertain = 0)
        ORDER BY TalkID, Time
    """

    accents = defaultdict(list)
    for talk_id, time in cur.execute(query):
        accents[talk_id].append(time)

    return accents


def fetch_phone_mora_data(cur, core_talks, use_word=False):
    """Fetch phone and mora data, returning a list of mora dictionaries.

    Now includes IPU information to handle split audio files.
    """
    if not core_talks:
        return []

    talk_str = ",".join(["'" + t + "'" for t in core_talks])

    if use_word:
        # Group by SUW (Short Unit Word)
        query = f"""
            SELECT
                p.TalkID,
                ms.SUWID,
                ms.MoraID,
                p.PhoneEntity,
                p.StartTime,
                p.EndTime,
                m.StartTime,
                m.EndTime,
                m.PerceivedAcc,
                ipu.IPUID,
                ipu.StartTime
            FROM segPhone p
            JOIN relPhone2Mora rpm ON p.TalkID = rpm.TalkID AND p.PhoneID = rpm.PhoneID
            JOIN relMora2SUW ms ON p.TalkID = ms.TalkID AND rpm.MoraID = ms.MoraID
            JOIN segMora m ON p.TalkID = m.TalkID AND rpm.MoraID = m.MoraID
            JOIN relMora2IPU rmi ON p.TalkID = rmi.TalkID AND rpm.MoraID = rmi.MoraID
            JOIN segIPU ipu ON p.TalkID = ipu.TalkID AND rmi.IPUID = ipu.IPUID
            WHERE p.TalkID IN ({talk_str})
            ORDER BY p.TalkID, ms.SUWID, m.StartTime, rpm.nth;
        """
    else:
        # Group by AP (Accent Phrase)
        query = f"""
            SELECT
                p.TalkID,
                pm.APID,
                pm.MoraID,
                p.PhoneEntity,
                p.StartTime,
                p.EndTime,
                m.StartTime,
                m.EndTime,
                m.PerceivedAcc,
                ipu.IPUID,
                ipu.StartTime
            FROM segPhone p
            JOIN relPhone2Mora rpm ON p.TalkID = rpm.TalkID AND p.PhoneID = rpm.PhoneID
            JOIN relMora2AP pm ON p.TalkID = pm.TalkID AND rpm.MoraID = pm.MoraID
            JOIN segMora m ON p.TalkID = m.TalkID AND rpm.MoraID = m.MoraID
            JOIN relMora2IPU rmi ON p.TalkID = rmi.TalkID AND rpm.MoraID = rmi.MoraID
            JOIN segIPU ipu ON p.TalkID = ipu.TalkID AND rmi.IPUID = ipu.IPUID
            WHERE p.TalkID IN ({talk_str})
            ORDER BY p.TalkID, pm.APID, m.StartTime, rpm.nth;
        """

    all_moras = []
    current_mora = None
    mora_data = None

    for row in cur.execute(query):
        talk_id, unit_id, mora_id, phone, p_start, p_end, m_start, m_end, p_acc, ipu_id, ipu_start = row

        phone_info = {
            'entity': phone,
            'start': p_start,
            'end': p_end,
            'is_nucleus': phone in NUCLEUS_PHONES
        }

        if current_mora != (talk_id, mora_id):
            if mora_data:
                all_moras.append(mora_data)
            current_mora = (talk_id, mora_id)
            mora_data = {
                'talk_id': talk_id,
                'unit_id': unit_id,
                'phones': [phone],
                'phone_objs': [phone_info],
                'start': m_start,
                'end': m_end,
                'accented': p_acc,
                'ipu_id': ipu_id,
                'ipu_start': ipu_start
            }
        else:
            mora_data['phones'].append(phone)
            mora_data['phone_objs'].append(phone_info)

    if mora_data:
        all_moras.append(mora_data)

    return all_moras


def group_moras_into_units(all_moras):
    """Group moras by (talk_id, unit_id), ensuring IPU consistency."""
    units = defaultdict(list)
    for mora in all_moras:
        units[(mora['talk_id'], mora['unit_id'])].append(mora)
    
    # Filter units that span multiple IPUs
    valid_units = {}
    for key, mora_list in units.items():
        ipu_ids = set(m['ipu_id'] for m in mora_list)
        if len(ipu_ids) == 1:
            valid_units[key] = mora_list
        # else: discard unit spanning multiple IPUs
            
    return valid_units


def compute_ap_contexts(aps):
    """Compute preceding/following phone context, respecting IPU boundaries."""
    # Build list of APs with timing info
    ap_list = []
    for (talk_id, ap_id), mora_list in aps.items():
        ap_list.append({
            'talk_id': talk_id,
            'ap_id': ap_id,
            'ipu_id': mora_list[0]['ipu_id'],
            'start': mora_list[0]['start'],
            'first_phone': mora_list[0]['phones'][0],
            'last_phone': mora_list[-1]['phones'][-1]
        })

    # Sort by talk_id, then start time
    ap_list.sort(key=lambda x: (x['talk_id'], x['start']))

    # Build context mapping
    contexts = {}
    for i, ap in enumerate(ap_list):
        key = (ap['talk_id'], ap['ap_id'])

        # Preceding phone
        # Must be same talk AND same IPU
        if i > 0 and ap_list[i - 1]['talk_id'] == ap['talk_id'] and ap_list[i - 1]['ipu_id'] == ap['ipu_id']:
            preceding = ap_list[i - 1]['last_phone']
        else:
            preceding = None

        # Following phone
        if i < len(ap_list) - 1 and ap_list[i + 1]['talk_id'] == ap['talk_id'] and ap_list[i + 1]['ipu_id'] == ap['ipu_id']:
            following = ap_list[i + 1]['first_phone']
        else:
            following = None

        contexts[key] = {'preceding_phone': preceding, 'following_phone': following}

    return contexts


def build_phone_sequence_groups(units, talk_to_speaker, min_moras, max_moras=None, contexts=None, point_accents=None, min_duration=0.05):
    """Build groups of units, adjusting times relative to IPU start."""
    groups_by_phones = defaultdict(list)

    for (talk_id, unit_id), mora_list in units.items():
        if len(mora_list) < min_moras:
            continue
        if max_moras is not None and len(mora_list) > max_moras:
            continue

        first_phones = mora_list[0]['phones']
        last_phones = mora_list[-1]['phones']
        all_phones = [p for m in mora_list for p in m['phones']]

        if '<?>' in all_phones or '<sv>' in all_phones:
            continue
        if first_phones[0] in SPECIAL_PHONE_LEADS or last_phones[-1] == '<cl>':
            continue

        # Adjust timestamps relative to IPU
        ipu_start = mora_list[0]['ipu_start']
        relative_start = mora_list[0]['start'] - ipu_start
        relative_end = mora_list[-1]['end'] - ipu_start

        if (relative_end - relative_start) < min_duration:
            continue

        phone_seq = " ".join(["_".join(m['phones']) for m in mora_list])
        accent_pattern = tuple([m['accented'] for m in mora_list])
        is_accented = any(accent_pattern)
        
        # IPU info (consistent for valid units)
        ipu_id = mora_list[0]['ipu_id']

        # Point accent filtering
        point_accent_time = None
        if point_accents is not None:
            start_t = mora_list[0]['start']
            end_t = mora_list[-1]['end']

            found_acc = None
            if talk_id in point_accents:
                for t in point_accents[talk_id]:
                    if t >= start_t and t <= end_t:
                        found_acc = t
                        break
                    if t > end_t:
                        break

            if is_accented:
                if found_acc is None:
                    continue
                point_accent_time = found_acc
            else:
                if found_acc is not None:
                    continue

        speaker_id = talk_to_speaker.get(talk_id, "unknown")

        ctx = contexts.get((talk_id, unit_id), {}) if contexts else {}
        prev_phone = ctx.get('preceding_phone')
        next_phone = ctx.get('following_phone')

        group_key = phone_seq

        # New #file ID: {TalkID}/{TalkID}_{IPUID} (matches split_csj.py folder structure)
        file_id = f"{talk_id}/{talk_id}_{ipu_id}"

        instance = {
            'pattern': accent_pattern,
            'talk_id': file_id, # Replaced with new ID
            'unit_id': unit_id,
            'speaker_id': speaker_id,
            'start': relative_start,
            'end': relative_end,
            'num_moras': len(mora_list),
            'prev_phone': prev_phone,
            'next_phone': next_phone
        }
        if point_accent_time is not None:
            instance['point_accent_time'] = point_accent_time - ipu_start

        groups_by_phones[group_key].append(instance)

    return groups_by_phones


def generate_subsequences(aps, min_moras, max_moras, ap_contexts=None):
    """Generate subsequences, preserving IPU info."""
    subsequences = []
    for (talk_id, ap_id), mora_list in aps.items():
        n = len(mora_list)
        max_window = max_moras if max_moras is not None else n
        
        ipu_id = mora_list[0]['ipu_id']
        ipu_start = mora_list[0]['ipu_start']

        ap_ctx = ap_contexts.get((talk_id, ap_id), {}) if ap_contexts else {}

        for window_size in range(min_moras, min(max_window, n) + 1):
            for start in range(n - window_size + 1):
                window = mora_list[start:start + window_size]

                if start > 0:
                    preceding_phone = mora_list[start - 1]['phones'][-1]
                else:
                    preceding_phone = ap_ctx.get('preceding_phone')

                end_idx = start + window_size
                if end_idx < n:
                    following_phone = mora_list[end_idx]['phones'][0]
                else:
                    following_phone = ap_ctx.get('following_phone')

                subsequences.append({
                    'talk_id': talk_id,
                    'ap_id': ap_id,
                    'ipu_id': ipu_id,
                    'ipu_start': ipu_start,
                    'moras': window,
                    'start_pos': start + 1,
                    'preceding_phone': preceding_phone,
                    'following_phone': following_phone
                })
    return subsequences


def build_subsequence_groups(subsequences, talk_to_speaker, point_accents=None, min_duration=0.05):
    """Build subsequence groups, adjusting times relative to IPU start."""
    groups_by_phones = defaultdict(list)

    for subseq in subsequences:
        mora_list = subseq['moras']
        talk_id = subseq['talk_id']

        first_phones = mora_list[0]['phones']
        last_phones = mora_list[-1]['phones']
        all_phones = [p for m in mora_list for p in m['phones']]

        if '<?>' in all_phones or '<sv>' in all_phones:
            continue
        if first_phones[0] in SPECIAL_PHONE_LEADS or last_phones[-1] == '<cl>':
            continue

        # Adjust timestamps
        ipu_start = subseq['ipu_start']
        relative_start = mora_list[0]['start'] - ipu_start
        relative_end = mora_list[-1]['end'] - ipu_start

        if (relative_end - relative_start) < min_duration:
            continue

        phone_seq = " ".join(["_".join(m['phones']) for m in mora_list])
        accent_pattern = tuple([m['accented'] for m in mora_list])
        is_accented = any(accent_pattern)

        point_accent_time = None
        if point_accents is not None:
            start_t = mora_list[0]['start']
            end_t = mora_list[-1]['end']

            found_acc = None
            if talk_id in point_accents:
                for t in point_accents[talk_id]:
                    if t >= start_t and t <= end_t:
                        found_acc = t
                        break
                    if t > end_t:
                        break

            if is_accented:
                if found_acc is None:
                    continue
                point_accent_time = found_acc
            else:
                if found_acc is not None:
                    continue

        speaker_id = talk_to_speaker.get(talk_id, "unknown")

        prev_phone = subseq.get('preceding_phone')
        next_phone = subseq.get('following_phone')

        group_key = phone_seq
        
        # New #file ID: {TalkID}/{TalkID}_{IPUID}
        file_id = f"{talk_id}/{talk_id}_{subseq['ipu_id']}"

        instance = {
            'pattern': accent_pattern,
            'talk_id': file_id,
            'ap_id': subseq['ap_id'],
            'speaker_id': speaker_id,
            'start': relative_start,
            'end': relative_end,
            'num_moras': len(mora_list),
            'start_pos': subseq['start_pos'],
            'prev_phone': prev_phone,
            'next_phone': next_phone
        }
        if point_accent_time is not None:
            instance['point_accent_time'] = point_accent_time - ipu_start

        groups_by_phones[group_key].append(instance)

    return groups_by_phones


def filter_minimal_pairs(groups_by_phones, distinguish_last_mora):
    """Filter to groups with multiple distinct accent patterns per speaker."""
    valid_groups = []

    for group_key, instances in groups_by_phones.items():
        # Handle both plain phone_seq (string) and strict mode tuple (preceding, phone_seq, following)
        if isinstance(group_key, tuple):
            phone_seq = group_key[1]  # Extract phone sequence from tuple
        else:
            phone_seq = group_key

        by_speaker = defaultdict(list)
        for inst in instances:
            by_speaker[inst['speaker_id']].append(inst)

        for speaker_id, speaker_instances in by_speaker.items():
            def get_comp(p):
                return p if distinguish_last_mora else p[:-1]

            unique_patterns = set(get_comp(inst['pattern']) for inst in speaker_instances)

            if len(unique_patterns) > 1:
                valid_groups.append((phone_seq, speaker_instances))

    return valid_groups


def remove_contained_instances(valid_groups):
    """Remove instances that are fully contained within longer instances from the same AP.

    This operates on the output of filter_minimal_pairs. Compares instances ACROSS all
    phone sequence groups - if an instance from AP X is contained within a longer instance
    from the same AP X (even in a different phone_seq group), remove the shorter one.

    Only affects instances with 'start_pos' (i.e., free mode subsequences).
    """
    # Collect all instances across all groups, grouped by (talk_id, ap_id)
    by_ap = defaultdict(list)
    for phone_seq, instances in valid_groups:
        for inst in instances:
            if 'start_pos' in inst:
                key = (inst['talk_id'], inst['ap_id'])
                by_ap[key].append(inst)

    # Find instances to remove (those contained in others from same AP)
    to_remove = set()
    for key, ap_instances in by_ap.items():
        for inst in ap_instances:
            start = inst['start_pos'] - 1  # Convert to 0-indexed
            end = start + inst['num_moras'] - 1

            for other in ap_instances:
                if other is inst:
                    continue
                other_start = other['start_pos'] - 1
                other_end = other_start + other['num_moras'] - 1

                # Check if inst is fully contained in other
                if start >= other_start and end <= other_end:
                    to_remove.add((inst['talk_id'], inst['ap_id'], inst['start_pos'], inst['num_moras']))
                    break

    # Filter all groups, removing marked instances
    filtered_groups = []
    for phone_seq, instances in valid_groups:
        filtered = []
        for inst in instances:
            if 'start_pos' not in inst:
                filtered.append(inst)
            else:
                key = (inst['talk_id'], inst['ap_id'], inst['start_pos'], inst['num_moras'])
                if key not in to_remove:
                    filtered.append(inst)
        if filtered:
            filtered_groups.append((phone_seq, filtered))

    return filtered_groups


def write_item_file(valid_groups, output_path, include_point_accent=False,
                    distinguish_last_mora=False):
    """Write item file in fastabx-compatible CSV format.

    Output columns:
      - #file: TalkID (file identifier without extension)
      - onset: Start time in seconds
      - offset: End time in seconds
      - phone_sequence: Phone sequence for BY condition
      - accent_pattern: Accent pattern for ON condition (may exclude last mora)
      - speaker: Speaker ID for BY condition
      - prev_phone: Preceding phone context
      - next_phone: Following phone context
      - num_moras: Number of moras (metadata)

    Args:
        valid_groups: List of (phone_seq, instances) tuples
        output_path: Path to write the CSV file
        include_point_accent: Whether to include point accent timestamps
        distinguish_last_mora: If False, accent_pattern excludes the last mora
            (because accent on the final mora is hard to perceive)
    """
    count = 0

    header = ['#file', 'onset', 'offset', 'phone_sequence', 'accent_pattern',
              'speaker', 'prev_phone', 'next_phone', 'num_moras']
    if include_point_accent:
        header.append('point_accent_time')

    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for phone_seq, instances in valid_groups:
            for inst in instances:
                # When distinguish_last_mora=False, truncate pattern to exclude last mora
                # (accent on final mora is hard to perceive)
                pattern = inst['pattern']
                if not distinguish_last_mora and len(pattern) > 1:
                    pattern = pattern[:-1]
                pattern_str = "".join(map(str, pattern))

                row = [
                    inst['talk_id'],           # #file
                    inst['start'],             # onset
                    inst['end'],               # offset
                    phone_seq,                 # phone_sequence
                    pattern_str,               # accent_pattern
                    inst['speaker_id'],        # speaker
                    inst.get('prev_phone', ''),  # prev_phone
                    inst.get('next_phone', ''),  # next_phone
                    inst['num_moras']          # num_moras
                ]
                if include_point_accent:
                    row.append(inst.get('point_accent_time', ''))

                writer.writerow(row)
                count += 1

    return count


def extract_pairs(talk_id=None, distinguish_last_mora=False, min_moras=3, max_moras=None,
                   seed=None, use_word=False, use_free=False, include_point_accent=False,
                   output_dir=None, min_duration=0.05):
    """Extract pitch accent minimal pairs and write to fastabx-compatible item file.

    Args:
        talk_id: Process only a specific TalkID (for debugging)
        distinguish_last_mora: Whether to distinguish accent on the last mora
        min_moras: Minimum number of moras in a unit
        max_moras: Maximum number of moras in a unit (None for no limit)
        seed: Random seed for reproducible shuffle
        use_word: Use words (SUW) instead of accent phrases as the basic unit
        use_free: Extract subsequences freely within APs
        include_point_accent: Include point accent timestamps and filter by them
        output_dir: Override output directory (default: based on mode)
        min_duration: Minimum duration of a unit in seconds
    """
    if seed is not None:
        random.seed(seed)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    if use_free:
        unit_type = "free subsequences within APs"
        default_suffix = "csj_free"
    elif use_word:
        unit_type = "words"
        default_suffix = "csj_word"
    else:
        unit_type = "accent phrases"
        default_suffix = "csj_ap"

    if output_dir is None:
        output_dir = os.path.join(OUTPUT_DIR, default_suffix)

    item_path = os.path.join(output_dir, "items.csv")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Write audio path for extract_features.py to find (points to split IPU files)
    with open(os.path.join(output_dir, "audio_path.txt"), "w") as f:
        f.write(SPLIT_DIR)

    print("Identifying Core TalkIDs from files...")
    core_talks = get_core_talks(BASE_DIR, talk_id)
    print(f"Processing {len(core_talks)} core talks.")

    print("Fetching speaker mappings...")
    talk_to_speaker = fetch_speaker_mappings(cur)

    point_accents = None
    if include_point_accent:
        print("Fetching valid point accents...")
        point_accents = fetch_valid_point_accents(cur, core_talks)

    # For free mode, always fetch AP-based data (not word-based)
    # because we extract subsequences within APs and need AP boundary context
    fetch_word = use_word and not use_free
    print(f"Fetching phones and mapping to moras (unit: {unit_type})...")
    all_moras = fetch_phone_mora_data(cur, core_talks, use_word=fetch_word)

    print(f"Grouping moras into {'accent phrases' if use_free else unit_type}...")
    units = group_moras_into_units(all_moras)

    # Always compute contexts for inclusion in output (strict filtering done in run_abx.py)
    contexts = compute_ap_contexts(units)

    if use_free:
        print(f"Generating contiguous mora windows ({min_moras}-{max_moras or 'unlimited'} moras)...")
        subsequences = generate_subsequences(units, min_moras, max_moras, ap_contexts=contexts)
        print(f"Generated {len(subsequences)} subsequences from {len(units)} APs.")

        print("Building phone sequence groups from subsequences...")
        groups_by_phones = build_subsequence_groups(subsequences, talk_to_speaker, point_accents=point_accents, min_duration=min_duration)
    else:
        print("Building phone sequence groups...")
        groups_by_phones = build_phone_sequence_groups(
            units, talk_to_speaker, min_moras, max_moras,
            contexts=contexts, point_accents=point_accents,
            min_duration=min_duration
        )

    print(f"Filtering minimal pairs (distinguish_last={distinguish_last_mora})...")
    valid_groups = filter_minimal_pairs(groups_by_phones, distinguish_last_mora)

    if use_free:
        print("Removing contained instances (keeping only longest per AP)...")
        valid_groups = remove_contained_instances(valid_groups)

    random.shuffle(valid_groups)

    count = write_item_file(
        valid_groups, item_path,
        include_point_accent=include_point_accent,
        distinguish_last_mora=distinguish_last_mora
    )

    print(f"Extraction complete. Total instances: {count}, Total groups: {len(valid_groups)}")
    if not distinguish_last_mora:
        print("Note: accent_pattern excludes last mora (--distinguish-last-mora not set)")
    print(f"Item file written to: {item_path}")
    conn.close()

    return item_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate fastabx-compatible item files for pitch accent ABX discrimination.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--talk-id", type=str, default=None,
                        help="Process only a specific TalkID (for debugging).")
    parser.add_argument("--distinguish-last-mora", action="store_true",
                        help="Distinguish accent on the last mora (hard to hear).")
    parser.add_argument("--min-duration", type=float, default=0.05,
                        help="Minimum duration of a unit in seconds.")
    parser.add_argument("--min-moras", type=int, default=2,
                        help="Minimum number of moras in a unit.")
    parser.add_argument("--max-moras", type=int, default=None,
                        help="Maximum number of moras in a unit (no limit if not specified).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible shuffle.")
    parser.add_argument("--word", action="store_true",
                        help="Use words (SUW) as the basic unit instead of accent phrases.")
    parser.add_argument("--free", action="store_true",
                        help="Extract subsequences freely within APs (use with --min-moras/--max-moras).")
    parser.add_argument("--include-point-accent", action="store_true",
                        help="Include point accent timestamps and filter samples by them.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: auto-generated based on mode).")
    args = parser.parse_args()

    extract_pairs(
        talk_id=args.talk_id,
        distinguish_last_mora=args.distinguish_last_mora,
        min_moras=args.min_moras,
        max_moras=args.max_moras,
        seed=args.seed,
        use_word=args.word,
        use_free=args.free,
        include_point_accent=args.include_point_accent,
        output_dir=args.output_dir,
        min_duration=args.min_duration
    )
