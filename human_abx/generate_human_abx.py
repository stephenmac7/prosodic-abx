"""Generate triplets for human ABX experiments.

Uses fastabx scaffolding to enumerate valid triplets, then assigns to participants
with recording-level coverage guarantees for annotation QC.

Design:
1. Use fastabx Subsampler to create manageable triplet pool
2. Ensure each recording appears ≥N times total (for annotation QC)
3. Assign triplets to participants minimizing per-participant recording repeats
4. Round-robin spacing for any unavoidable repeats within a participant's sequence

Usage:
    python generate_human_abx.py --dataset stress \\
        --trials-per-participant 180 \\
        --min-responses-per-recording 5 \\
        --catch-ratio 0.1 \\
        --materialize-audio
"""

import argparse
import math
import random
import wave
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import polars as pl
import torchaudio
from torchcodec.decoders import AudioDecoder
from fastabx.dataset import dummy_dataset_from_item
from fastabx.subsample import Subsampler
from fastabx.task import Task


def normalize_waveform(
    waveform: "torch.Tensor",
    target_dbfs: float = -20.0,
    max_peak: float = 0.99,
    eps: float = 1e-8,
) -> "torch.Tensor":
    """Normalize waveform loudness to a target RMS with peak limiting."""
    rms = waveform.pow(2).mean().sqrt()
    if rms < eps:
        return waveform
    target_rms = 10 ** (target_dbfs / 20)
    scale = target_rms / rms
    waveform = waveform * scale
    peak = waveform.abs().max()
    if peak > max_peak:
        waveform = waveform * (max_peak / peak)
    return waveform


def clip_audio(
    src: Path,
    dst: Path,
    onset: float,
    offset: float,
    fade_duration: float = 0.0,
    normalize: bool = True,
) -> None:
    """Cut a segment from a WAV file with optional boundary-centered fade.

    Args:
        src: Source audio file
        dst: Destination audio file
        onset: Start time in seconds
        offset: End time in seconds (0 means end of file)
        fade_duration: Duration of fade in seconds. The fade is centered on the cut
            boundary, so we load extra audio (fade_duration/2) before onset and after
            offset, then apply a fade over that region. Set to 0 for clean cut.
    """
    decoder = AudioDecoder(str(src))
    sr = decoder.metadata.sample_rate
    duration = decoder.metadata.duration_seconds
    total_frames = int(duration * sr)

    fade_frames = int(round(fade_duration * sr))
    half_fade = fade_frames // 2

    # Expand the region to include fade margins (centered on cut boundary)
    start_frame = max(0, int(round(onset * sr)) - half_fade)
    if offset > 0:
        end_frame = min(total_frames, int(round(offset * sr)) + half_fade)
    else:
        end_frame = total_frames

    num_frames = max(0, end_frame - start_frame)

    waveform, _ = torchaudio.load(
        str(src), frame_offset=start_frame, num_frames=num_frames
    )

    # Apply fade if requested
    if fade_frames > 0 and num_frames > 0:
        # Compute actual fade lengths (may be shorter if we hit file boundaries)
        actual_fade_in = min(
            half_fade, int(round(onset * sr)) - start_frame + half_fade
        )
        actual_fade_out = min(
            half_fade, end_frame - int(round(offset * sr)) + half_fade
        )
        # Clamp to half the waveform length
        actual_fade_in = min(actual_fade_in, num_frames // 2)
        actual_fade_out = min(actual_fade_out, num_frames // 2)

        if actual_fade_in > 0 or actual_fade_out > 0:
            fade_transform = torchaudio.transforms.Fade(
                fade_in_len=actual_fade_in,
                fade_out_len=actual_fade_out,
            )
            waveform = fade_transform(waveform)

    if normalize:
        waveform = normalize_waveform(waveform)

    dst.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(dst), waveform, sr)


def materialize_audio_clips(
    items_csv: Path,
    audio_root: Path,
    output_dir: Path,
    used_files: set[str] | None = None,
    fade_duration: float = 0.0,
    normalize: bool = True,
) -> int:
    """Materialize audio clips from items.csv to output directory.

    Args:
        items_csv: Path to items.csv with #file, onset, offset columns
        audio_root: Root directory containing source audio
        output_dir: Directory to write clipped audio
        used_files: If provided, only materialize files in this set
        fade_duration: Duration of boundary-centered fade in seconds (0 for clean cut)
        normalize: Normalize audio volume for saved clips

    Returns:
        Number of clips materialized
    """
    items = pl.read_csv(items_csv)
    if "#file" not in items.columns:
        raise ValueError(f"Missing required '#file' column in {items_csv}")
    items = items.with_columns(pl.col("#file").cast(pl.Utf8))

    output_dir.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    for row in items.iter_rows(named=True):
        file_id = row["#file"]
        if file_id is None or file_id == "":
            continue
        if file_id in seen:
            continue
        if used_files is not None and file_id not in used_files:
            continue
        seen.add(file_id)

        onset = row.get("onset", 0.0) or 0.0
        offset = row.get("offset", 0.0) or 0.0

        source_path = Path(file_id)
        if source_path.suffix:
            src = audio_root / source_path
        else:
            src = audio_root / f"{file_id}.wav"

        if not src.exists():
            raise FileNotFoundError(f"Source audio not found: {src}")

        dst = output_dir / f"{file_id}.wav"
        if not dst.exists():
            clip_audio(src, dst, onset, offset, fade_duration, normalize=normalize)

    return len(seen)


def enumerate_triplets(task: Task, labels: pl.DataFrame) -> pl.DataFrame:
    """Expand Task cells into individual (A, B, X) triplets with file references."""
    file_col = "#file"
    files = labels[file_col].to_list()
    rows = []

    for row in task.cells.iter_rows(named=True):
        index_a = row["index_a"]
        index_b = row["index_b"]
        index_x = row["index_x"]

        phone_seq = row["phone_sequence"]
        accent_a = row["accent_pattern"]
        accent_b = row["accent_pattern_b"]
        speaker_ab = row["speaker"]
        speaker_x = row["speaker_x"]

        for a_idx in index_a:
            for b_idx in index_b:
                for x_idx in index_x:
                    rows.append(
                        {
                            "phone_sequence": phone_seq,
                            "accent_a": str(accent_a),
                            "accent_b": str(accent_b),
                            "speaker_ab": speaker_ab,
                            "speaker_x": speaker_x,
                            "file_a": files[a_idx],
                            "file_b": files[b_idx],
                            "file_x": files[x_idx],
                        }
                    )

    return pl.DataFrame(rows)


def get_recordings_in_triplet(triplet: dict) -> set[str]:
    """Get all recordings involved in a triplet."""
    return {triplet["file_a"], triplet["file_b"], triplet["file_x"]}


def generate_catch_trials(
    syn_labels: pl.DataFrame,
    n_catch: int,
    seed: int,
) -> pl.DataFrame:
    """Generate catch trials using synthesized speech with identity matching.

    All three items (A, B, X) are from the same speaker. Either:
    - A = X (correct answer: A), or
    - B = X (correct answer: B)

    This tests basic attention without introducing phone contrast confusion.
    """
    random.seed(seed)

    speakers = syn_labels["speaker"].unique().to_list()
    if len(speakers) < 1:
        raise ValueError("Need at least 1 speaker for catch trials")

    # Require at least two synthesized items per speaker + word to keep catch trials
    # within the same word (identity matching without lexical differences).
    candidate_groups = (
        syn_labels.group_by(["speaker", "phone_sequence"])
        .len()
        .filter(pl.col("len") >= 2)
        .select(["speaker", "phone_sequence"])
        .sort(["speaker", "phone_sequence"])
        .to_dicts()
    )
    if not candidate_groups:
        raise ValueError(
            "Need at least one speaker with two items for the same phone_sequence"
        )

    rows = []
    attempts = 0
    max_attempts = n_catch * 100

    while len(rows) < n_catch and attempts < max_attempts:
        attempts += 1

        candidate = random.choice(candidate_groups)
        speaker = candidate["speaker"]
        phone_sequence = candidate["phone_sequence"]

        # Get items for this speaker + word (same phone_sequence)
        speaker_items = syn_labels.filter(
            (pl.col("speaker") == speaker)
            & (pl.col("phone_sequence") == phone_sequence)
        )

        # Sample two different items for A and B within the same word
        sampled = speaker_items.sample(2, seed=seed + attempts)
        a_row = sampled[0]
        b_row = sampled[1]

        # Decide if X matches A or B (counterbalanced)
        if random.random() < 0.5:
            # X = A (correct answer: A)
            x_file = a_row["#file"].item()
            correct_answer = "A"
        else:
            # X = B (correct answer: B)
            x_file = b_row["#file"].item()
            correct_answer = "B"

        rows.append(
            {
                "phone_sequence": a_row["phone_sequence"].item(),
                "phone_sequence_b": a_row["phone_sequence"].item(),
                "accent_a": str(a_row["accent_pattern"].item()),
                "accent_b": str(b_row["accent_pattern"].item()),
                "speaker_ab": speaker,
                "speaker_x": speaker,
                "file_a": a_row["#file"].item(),
                "file_b": b_row["#file"].item(),
                "file_x": x_file,
                "is_catch": True,
                "correct_answer": correct_answer,
                "audio_source": "syn",
            }
        )

    if len(rows) < n_catch:
        print(
            f"  Warning: Could only generate {len(rows)} catch trials (requested {n_catch})"
        )

    return pl.DataFrame(rows)


def counterbalance_answers(triplets: list[dict], seed: int) -> list[dict]:
    """Randomly swap A and B for ~50% of triplets.

    Skips catch trials since they already have correct_answer set
    and swapping would break the identity matching.
    """
    random.seed(seed)
    result = []
    for triplet in triplets:
        # Skip catch trials - they're already counterbalanced
        if triplet.get("is_catch"):
            result.append(triplet)
            continue

        if random.random() < 0.5:
            # Swap A and B
            result.append(
                {
                    **triplet,
                    "file_a": triplet["file_b"],
                    "file_b": triplet["file_a"],
                    "accent_a": triplet["accent_b"],
                    "accent_b": triplet["accent_a"],
                    "correct_answer": "B",
                }
            )
        else:
            result.append({**triplet, "correct_answer": "A"})
    return result


def assign_to_participants(
    triplets: list[dict],
    catch_trials: list[dict],
    trials_per_participant: int,
    min_responses_per_recording: int,
    seed: int,
    min_participants: int = 15,
    max_repeats_per_participant: int = 2,
) -> list[list[dict]]:
    """Assign triplets to participants minimizing per-participant recording repeats.

    Uses round-robin to space any repeated recordings within a participant's sequence.
    """
    # Calculate how many participants we need
    # Each triplet should be seen by enough participants to ensure recording coverage
    n_triplets = len(triplets)
    n_catch = len(catch_trials)
    regular_per_participant = trials_per_participant - n_catch

    if regular_per_participant <= 0:
        raise ValueError(
            "trials_per_participant must be greater than number of catch trials"
        )

    # Count recording appearances needed
    all_recordings = set()
    for t in triplets:
        all_recordings.update(get_recordings_in_triplet(t))

    # Each participant will see regular_per_participant triplets
    # Each triplet has 3 recordings
    # We want each recording to appear min_responses_per_recording times total

    # Build recording -> triplet index mapping for prioritization
    recording_to_triplet_indices: dict[str, list[int]] = defaultdict(list)
    for i, triplet in enumerate(triplets):
        for r in get_recordings_in_triplet(triplet):
            recording_to_triplet_indices[r].append(i)

    def compute_min_participants(target_regular: int) -> int:
        if target_regular <= 0:
            raise ValueError("regular trials per participant must be positive")
        # Each recording needs min_responses_per_recording appearances total
        # Each triplet has 3 recordings, each participant sees target_regular triplets
        # So each participant contributes target_regular * 3 recording slots
        total_recording_slots_needed = len(all_recordings) * min_responses_per_recording
        recording_slots_per_participant = target_regular * 3
        min_participants = max(
            1,
            (total_recording_slots_needed + recording_slots_per_participant - 1)
            // recording_slots_per_participant,
        )

        # If each participant can see all triplets, compute tighter bound per recording.
        if n_triplets <= target_regular:
            required_per_recording = 1
            for r in all_recordings:
                per_participant_for_r = len(recording_to_triplet_indices[r])
                if per_participant_for_r <= 0:
                    continue
                required = (
                    min_responses_per_recording + per_participant_for_r - 1
                ) // per_participant_for_r
                required_per_recording = max(required_per_recording, required)
            min_participants = max(min_participants, required_per_recording)

        return min_participants

    # Pre-count recordings used in catch trials (same for all participants)
    catch_recording_counts = Counter()
    for ct in catch_trials:
        for r in get_recordings_in_triplet(ct):
            catch_recording_counts[r] += 1

    def assign_once(target_regular: int) -> tuple[list[list[dict]], int, Counter]:
        # Track how many times each recording has been assigned
        recording_assignment_counts = Counter()

        # Track which triplets have been assigned
        triplet_assigned_count = Counter()

        min_participants_clamped = max(1, min_participants)
        n_participants = max(
            min_participants_clamped, compute_min_participants(target_regular)
        )
        print(
            f"  Target: {n_participants} participant lists "
            f"(minimum {min_participants_clamped}, may increase for coverage)"
        )
        print(
            f"  {target_regular} regular + {n_catch} catch = {target_regular + n_catch} trials each"
        )

        participant_lists = []
        max_participants = n_participants * 100  # Limit to avoid infinite loop
        p_idx = 0
        min_filled = target_regular

        while p_idx < max_participants:
            # Select triplets for this participant
            # Start with catch trial recordings to account for them in repeat constraint
            participant_recordings = Counter(catch_recording_counts)
            participant_triplets = []
            used_in_this_participant = set()

            # Prioritize triplets containing under-covered recordings
            under_covered_recs = {
                r
                for r in all_recordings
                if recording_assignment_counts[r] < min_responses_per_recording
            }

            # Sort candidates: triplets with under-covered recordings first, then by coverage pressure
            candidates = list(range(n_triplets))

            def priority_key(t_idx):
                recs = get_recordings_in_triplet(triplets[t_idx])
                n_under = len(recs & under_covered_recs)
                total_assigned = sum(recording_assignment_counts[r] for r in recs)
                return (-n_under, total_assigned, triplet_assigned_count[t_idx], t_idx)

            candidates.sort(key=priority_key)

            # Two-pass selection: prefer no repeats, allow repeats as fallback
            # Include catch trial recordings in used set
            used_recordings = set(catch_recording_counts.keys())

            # First pass: only triplets with no recording repeats
            for t_idx in candidates:
                if len(participant_triplets) >= target_regular:
                    break
                if t_idx in used_in_this_participant:
                    continue

                triplet = triplets[t_idx]
                recs = get_recordings_in_triplet(triplet)

                # Skip if any recording already used OR would exceed max repeats
                if recs & used_recordings:
                    continue
                if any(
                    participant_recordings[r] >= max_repeats_per_participant
                    for r in recs
                ):
                    continue

                participant_triplets.append(triplet)
                used_in_this_participant.add(t_idx)
                used_recordings.update(recs)
                for r in recs:
                    participant_recordings[r] += 1
                    recording_assignment_counts[r] += 1
                triplet_assigned_count[t_idx] += 1

            # Second pass: allow limited repeats if needed to fill remaining slots
            if len(participant_triplets) < target_regular:
                for t_idx in candidates:
                    if len(participant_triplets) >= target_regular:
                        break
                    if t_idx in used_in_this_participant:
                        continue

                    triplet = triplets[t_idx]
                    recs = get_recordings_in_triplet(triplet)

                    # Skip if any recording would exceed max repeats
                    if any(
                        participant_recordings[r] >= max_repeats_per_participant
                        for r in recs
                    ):
                        continue

                    participant_triplets.append(triplet)
                    used_in_this_participant.add(t_idx)
                    for r in recs:
                        participant_recordings[r] += 1
                        recording_assignment_counts[r] += 1
                    triplet_assigned_count[t_idx] += 1

            min_filled = min(min_filled, len(participant_triplets))

            # Warn if we couldn't fill all slots
            if len(participant_triplets) < target_regular:
                print(
                    f"  Warning: Participant {p_idx} has only {len(participant_triplets)}/{target_regular} "
                    f"regular trials (max_repeats={max_repeats_per_participant} constraint)"
                )

            # Add catch trials (distributed evenly)
            all_trials = participant_triplets + catch_trials.copy()

            # Round-robin reorder to maximize spacing of repeated recordings
            all_trials = round_robin_order(all_trials, seed + p_idx)

            # Counterbalance A/B answers
            all_trials = counterbalance_answers(all_trials, seed + p_idx + 1000)

            participant_lists.append(all_trials)
            p_idx += 1

            # Check if we've met our target
            if p_idx >= n_participants:
                under_covered = [
                    r
                    for r, c in recording_assignment_counts.items()
                    if c < min_responses_per_recording
                ]
                if not under_covered:
                    break  # Coverage achieved

        return participant_lists, min_filled, recording_assignment_counts

    # Check if coverage is achievable: each recording must appear in enough triplets
    impossible_recordings = [
        r
        for r in all_recordings
        if len(recording_to_triplet_indices[r]) < min_responses_per_recording
    ]
    if impossible_recordings:
        print(
            f"  Note: {len(impossible_recordings)} recordings appear in < {min_responses_per_recording} triplets"
        )
        print(
            f"         These cannot reach target coverage regardless of participant count"
        )

    target_regular = regular_per_participant
    participant_lists, min_filled, recording_assignment_counts = assign_once(
        target_regular
    )
    if min_filled < target_regular:
        raise RuntimeError(
            f"Unable to assign {target_regular} regular trials per participant without exceeding "
            f"max_repeats_per_participant={max_repeats_per_participant}. "
            "Try increasing the triplet pool or reducing trials per participant."
        )

    print(f"  Generated {len(participant_lists)} participant lists")
    print(
        f"  {target_regular} regular + {n_catch} catch = {target_regular + n_catch} trials each"
    )

    # Report coverage stats
    under_covered = [
        r
        for r, c in recording_assignment_counts.items()
        if c < min_responses_per_recording
    ]
    if under_covered:
        # Coverage was required but not achieved
        raise RuntimeError(
            f"Failed to achieve coverage: {len(under_covered)} recordings have "
            f"< {min_responses_per_recording} appearances after {len(participant_lists)} participants. "
            f"Try increasing --trials-per-participant or decreasing --min-responses-per-recording."
        )
    else:
        print(
            f"  All {len(all_recordings)} recordings have >= {min_responses_per_recording} appearances"
        )

    min_count = (
        min(recording_assignment_counts.values()) if recording_assignment_counts else 0
    )
    max_count = (
        max(recording_assignment_counts.values()) if recording_assignment_counts else 0
    )
    print(f"  Recording appearances (total): min={min_count}, max={max_count}")

    # Report within-participant repeats
    max_within_participant = 0
    for plist in participant_lists:
        rec_counts = Counter()
        for t in plist:
            for r in get_recordings_in_triplet(t):
                rec_counts[r] += 1
        if rec_counts:
            max_within_participant = max(
                max_within_participant, max(rec_counts.values())
            )
    print(f"  Max repeats within a participant: {max_within_participant}")

    return participant_lists, max_within_participant


def round_robin_order(triplets: list[dict], seed: int) -> list[dict]:
    """Reorder triplets to maximize spacing between repeated recordings.

    Uses a greedy approach: at each position, select the triplet whose recordings
    have been heard least recently.
    """
    if not triplets:
        return triplets

    random.seed(seed)
    random.shuffle(triplets)  # Start with random order

    remaining = triplets.copy()
    ordered = []

    # Track last position each recording was heard
    last_heard = defaultdict(lambda: -float("inf"))

    while remaining:
        current_pos = len(ordered)

        # Score each remaining triplet by minimum time since any of its recordings was heard
        best_idx = 0
        best_min_gap = -float("inf")

        for i, triplet in enumerate(remaining):
            recs = get_recordings_in_triplet(triplet)
            min_gap = min(current_pos - last_heard[r] for r in recs)

            # Add small random tiebreaker
            min_gap += random.random() * 0.1

            if min_gap > best_min_gap:
                best_min_gap = min_gap
                best_idx = i

        triplet = remaining.pop(best_idx)
        ordered.append(triplet)

        for r in get_recordings_in_triplet(triplet):
            last_heard[r] = current_pos

    return ordered


def main():
    parser = argparse.ArgumentParser(
        description="Generate triplets for human ABX experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., stress, pitch_accent, tone)",
    )
    parser.add_argument(
        "--trials-per-participant",
        type=int,
        default=100,
        help="Number of trials per participant (including catch trials)",
    )
    parser.add_argument(
        "--min-responses-per-recording",
        type=int,
        default=5,
        help="Minimum times each recording should be heard (across all participants)",
    )
    parser.add_argument(
        "--catch-ratio",
        type=float,
        default=0.1,
        help="Ratio of catch trials to regular trials",
    )
    parser.add_argument(
        "--max-size-group",
        type=int,
        default=None,
        help="Max items for A, B, or X per cell (fastabx Subsampler). Defaults to no subsampling",
    )
    parser.add_argument(
        "--max-x-across",
        type=int,
        default=None,
        help="Max X speakers per (A,B) pair (fastabx Subsampler). Defaults to no subsampling",
    )
    parser.add_argument(
        "--pinyin-freq-file",
        type=Path,
        default=None,
        help="Path to pinyin frequency file (for Mandarin filtering)",
    )
    parser.add_argument(
        "--top-pinyins",
        type=int,
        default=None,
        help="Select only top N most frequent pinyins (requires --pinyin-freq-file)",
    )
    parser.add_argument(
        "--min-participants",
        type=int,
        default=15,
        help="Minimum number of participants (generator may increase to meet coverage)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("human_abx"), help="Output directory"
    )
    parser.add_argument(
        "--materialize-audio",
        action="store_true",
        help="Also materialize audio clips for web deployment",
    )

    args = parser.parse_args()

    # Load the prepared item file for the requested dataset.
    item_path = Path("abx_items") / args.dataset / "items.csv"
    if not item_path.exists():
        raise FileNotFoundError(f"Item file not found: {item_path}")

    print(f"Loading items from {item_path}...")
    labels = pl.read_csv(item_path, schema_overrides={"#file": pl.String})
    print(f"  {len(labels)} items")

    # Load synthesized items for catch trials
    syn_item_path = Path("abx_items") / f"{args.dataset}_syn" / "items.csv"
    if not syn_item_path.exists():
        raise FileNotFoundError(f"Synthesized item file not found: {syn_item_path}")
    syn_labels = pl.read_csv(syn_item_path, schema_overrides={"#file": pl.String})
    print(
        f"Loaded synthesized items for catch trials from {syn_item_path} ({len(syn_labels)} items)"
    )

    # Filter by pinyin frequency if specified
    if args.pinyin_freq_file:
        if not args.top_pinyins:
            raise ValueError("--top-pinyins required when using --pinyin-freq-file")

        print(
            f"Filtering to top {args.top_pinyins} pinyins from {args.pinyin_freq_file}..."
        )
        freq_data = []
        with open(args.pinyin_freq_file) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    freq_data.append((parts[1], int(parts[2])))  # pinyin, frequency

        # Sort by frequency and take top N
        freq_data.sort(key=lambda x: -x[1])
        top_pinyins = {p for p, _ in freq_data[: args.top_pinyins]}
        print(
            f"  Selected pinyins: {sorted(top_pinyins)[:10]}... ({len(top_pinyins)} total)"
        )

        # Filter labels - phone_sequence is the pinyin for tone
        original_count = len(labels)
        labels = labels.filter(pl.col("phone_sequence").is_in(top_pinyins))
        print(f"  Filtered: {original_count} -> {len(labels)} items")

        # Update item_path to use filtered data (write to generated/)
        generated_dir = args.output_dir / "generated"
        generated_dir.mkdir(parents=True, exist_ok=True)
        filtered_item_path = generated_dir / f"{args.dataset}_filtered_items.csv"
        labels.write_csv(filtered_item_path)
        item_path = filtered_item_path

    # Create dummy dataset and build task with subsampling
    print("Building ABX task with subsampling...")
    dataset = dummy_dataset_from_item(item_path, frequency=None)
    subsampler = Subsampler(args.max_size_group, args.max_x_across, seed=args.seed)
    task = Task(
        dataset,
        on="accent_pattern",
        by=["phone_sequence"],
        across=["speaker"],
        subsampler=subsampler,
    )
    print(f"  {len(task)} cells")

    # Enumerate triplets
    print("Enumerating triplets...")
    all_triplets = enumerate_triplets(task, labels).sort(
        [
            "phone_sequence",
            "accent_a",
            "accent_b",
            "speaker_ab",
            "speaker_x",
            "file_a",
            "file_b",
            "file_x",
        ]
    )
    print(f"  {len(all_triplets)} triplets from subsampled cells")

    # Use full triplet set - assignment algorithm handles coverage constraints
    triplet_list = all_triplets.to_dicts()

    # Add metadata for regular trials
    for t in triplet_list:
        t["phone_sequence_b"] = t["phone_sequence"]  # Same word for A, B, X
        t["is_catch"] = False
        t["audio_source"] = "main"

    # Generate catch trials from synthesized speech
    n_catch = max(1, int(args.trials_per_participant * args.catch_ratio))
    print(f"Generating {n_catch} catch trials from synthesized speech...")
    catch_trials = generate_catch_trials(syn_labels, n_catch, args.seed).sort(
        [
            "phone_sequence",
            "speaker_ab",
            "file_a",
            "file_b",
            "file_x",
            "correct_answer",
        ]
    )
    catch_list = catch_trials.to_dicts()
    print(f"  Generated {len(catch_list)} catch trials")

    # Assign to participants
    print("Assigning triplets to participants...")
    participant_lists, max_within_participant = assign_to_participants(
        triplet_list,
        catch_list,
        args.trials_per_participant,
        args.min_responses_per_recording,
        args.seed,
        args.min_participants,
    )

    trials_per_participant_actual = (
        len(participant_lists[0]) if participant_lists else 0
    )
    catch_per_participant_actual = (
        sum(1 for t in participant_lists[0] if t.get("is_catch"))
        if participant_lists
        else 0
    )

    # Save outputs
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generated_dir = args.output_dir / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    # Save master triplet pool (intermediate file)
    pool_path = generated_dir / f"{args.dataset}_triplet_pool.csv"
    pl.DataFrame(triplet_list).write_csv(pool_path)
    print(f"\nWrote triplet pool to {pool_path}")

    # Save participant lists directly to web/lists/ for deployment
    lists_dir = args.output_dir / "web" / "lists" / args.dataset
    lists_dir.mkdir(parents=True, exist_ok=True)
    # Remove stale lists from previous runs for this dataset
    for old_list in lists_dir.glob("participant_*.csv"):
        old_list.unlink()

    for i, plist in enumerate(participant_lists):
        list_path = lists_dir / f"participant_{i:03d}.csv"
        pl.DataFrame(plist).write_csv(list_path)

    print(f"Wrote {len(participant_lists)} participant lists to {lists_dir}/")

    # Save list of unique audio files used across all participants.
    used_files: set[str] = set()
    used_files_main: set[str] = set()
    used_files_syn: set[str] = set()
    for plist in participant_lists:
        for trial in plist:
            source = trial.get("audio_source")
            for key in ("file_a", "file_b", "file_x"):
                file_id = trial.get(key)
                if not file_id:
                    continue
                file_id = str(file_id)
                used_files.add(file_id)
                if source == "syn":
                    used_files_syn.add(file_id)
                else:
                    used_files_main.add(file_id)

    used_audio_path = generated_dir / f"{args.dataset}_used_audio.csv"
    used_audio_main_path = generated_dir / f"{args.dataset}_used_audio_main.csv"
    used_audio_syn_path = generated_dir / f"{args.dataset}_used_audio_syn.csv"

    pl.DataFrame({"#file": sorted(used_files)}).write_csv(used_audio_path)
    pl.DataFrame({"#file": sorted(used_files_main)}).write_csv(used_audio_main_path)
    pl.DataFrame({"#file": sorted(used_files_syn)}).write_csv(used_audio_syn_path)
    print(f"Wrote used audio list to {used_audio_path}")
    print(f"Wrote used main audio list to {used_audio_main_path}")
    print(f"Wrote used syn audio list to {used_audio_syn_path}")

    # Save summary
    summary_path = generated_dir / f"{args.dataset}_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Total items: {len(labels)}\n")
        f.write(f"Total triplets: {len(triplet_list)}\n")
        f.write(f"Catch trials per participant: {catch_per_participant_actual}\n")
        f.write(f"Catch trial source: {syn_item_path}\n")
        f.write(f"Trials per participant: {trials_per_participant_actual}\n")
        f.write(f"Number of participants: {len(participant_lists)}\n")
        f.write(f"Min responses per recording: {args.min_responses_per_recording}\n")
        f.write(
            f"Subsampler: max_size_group={args.max_size_group}, max_x_across={args.max_x_across}\n"
        )
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Used audio list: {used_audio_path}\n")
        f.write(f"Used main audio list: {used_audio_main_path}\n")
        f.write(f"Used syn audio list: {used_audio_syn_path}\n")

        # Recording coverage stats
        all_recordings = set()
        recording_counts = Counter()
        for plist in participant_lists:
            for t in plist:
                for r in get_recordings_in_triplet(t):
                    all_recordings.add(r)
                    recording_counts[r] += 1

        f.write(f"\nRecording coverage (across all participants):\n")
        f.write(f"  Total unique recordings: {len(all_recordings)}\n")
        f.write(f"  Min appearances: {min(recording_counts.values())}\n")
        f.write(f"  Max appearances: {max(recording_counts.values())}\n")
        f.write(
            f"  Mean appearances: {sum(recording_counts.values()) / len(recording_counts):.1f}\n"
        )
        # Coverage distribution
        counts_by_appearance = Counter(recording_counts.values())
        total_recordings = len(recording_counts)
        f.write(f"\nRecording appearance distribution:\n")
        for k in sorted(counts_by_appearance):
            count = counts_by_appearance[k]
            pct = (count / total_recordings * 100) if total_recordings else 0.0
            f.write(f"  {k}: {count} ({pct:.1f}%)\n")

        # Identity of lowest-coverage recordings (below 5th percentile)
        counts_sorted = sorted(recording_counts.values())
        if counts_sorted:
            p10_index = max(0, math.ceil(0.05 * len(counts_sorted)) - 1)
            p10 = counts_sorted[p10_index]
            low_recs = sorted(
                ((r, c) for r, c in recording_counts.items() if c < p10),
                key=lambda rc: (rc[1], rc[0]),
            )
            f.write(
                f"\nRecordings with < 5th percentile appearances (p10={p10}): {len(low_recs)}\n"
            )
            max_listed = 50
            for r, c in low_recs[:max_listed]:
                f.write(f"  {r}: {c}\n")
            if len(low_recs) > max_listed:
                f.write(f"  ... {len(low_recs) - max_listed} more\n")

        f.write(f"\nWithin-participant repeats:\n")
        f.write(
            f"  Max times a recording heard by one participant: {max_within_participant}\n"
        )

    print(f"Wrote summary to {summary_path}")

    # Materialize audio clips if requested
    if args.materialize_audio:
        print("\nMaterializing audio clips...")
        web_audio_dir = args.output_dir / "web" / "audio"

        audio_root_main = Path(
            (Path("abx_items") / args.dataset / "audio_path.txt").read_text().strip()
        )
        main_items_path = item_path
        print("  Using prepared clipped items")

        main_items_files = set(
            pl.read_csv(main_items_path, columns=["#file"])["#file"]
            .cast(pl.Utf8)
            .to_list()
        )
        missing_files = used_files_main - main_items_files
        if missing_files:
            missing_list = sorted(missing_files)
            preview = ", ".join(missing_list[:10])
            raise FileNotFoundError(
                "Missing required audio files in items.csv for materialization. "
                f"{len(missing_list)} files are referenced by participant lists but not present in "
                f"{main_items_path}. First few: {preview}"
            )

        main_output = web_audio_dir / args.dataset
        n_main = materialize_audio_clips(
            main_items_path,
            audio_root_main,
            main_output,
            used_files_main,
            fade_duration=0.0,
        )
        print(f"  Materialized {n_main} main clips to {main_output}")

        # Synthesized audio for catch trials (always clean cut - already isolated words)
        audio_root_syn = Path(
            (Path("abx_items") / f"{args.dataset}_syn" / "audio_path.txt")
            .read_text()
            .strip()
        )
        syn_output = web_audio_dir / f"{args.dataset}_syn"
        n_syn = materialize_audio_clips(
            syn_item_path, audio_root_syn, syn_output, used_files_syn, fade_duration=0.0
        )
        print(f"  Materialized {n_syn} synthesized clips to {syn_output}")


if __name__ == "__main__":
    main()
