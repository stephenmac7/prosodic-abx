# Human ABX Experiment Design

## Overview

This experiment evaluates how well humans discriminate prosodic minimal pairs, providing a baseline for comparing SSL model performance.

## Datasets

### English Lexical Stress
- **Items**: 598 recordings from 10 speakers
- **Contrasts**: 15 noun-verb minimal pairs (e.g., "CONduct" vs "conDUCT")
- **Excluded words**: "research", "transfer" (not true minimal pairs)
- **Participants needed**: 33
- **Trials per participant**: 70 (63 regular + 7 catch)

### Japanese Pitch Accent
- **Items**: 909 recordings from 10 speakers
- **Contrasts**: ~46 pitch accent minimal pairs (e.g., あめ "rain" vs "candy")
- **Participants needed**: 18
- **Trials per participant**: 100 (90 regular + 10 catch)

### Mandarin Lexical Tone
- **Items**: 1008 recordings (subsampled to top 50 most frequent pinyins)
- **Contrasts**: 6 tone pairs (T1-T2, T1-T3, T1-T4, T2-T3, T2-T4, T3-T4)
- **Participants needed**: 15
- **Trials per participant**: 100 (90 regular + 10 catch)
- **Note**: Unlike English/Japanese, no per-recording QC coverage

(We set a minimum of 15 participants for each dataset to ensure stable estimates.)

## ABX Task Design

### Trial Structure
Each trial presents three audio clips:
- **A**: Reference recording (accent pattern 1)
- **B**: Contrast recording (accent pattern 2, same speaker as A)
- **X**: Probe recording (matches A's accent pattern, different speaker)

Participant task: "Is X more similar to A or B?".
Correct answer is counterbalanced (~50% A, ~50% B).

### Catch Trials
- ~10% of trials
- Use synthesized speech from the corresponding `*_syn` dataset
- A, B, and X are all from the same TTS speaker
- Either A=X (correct answer: A) or B=X (correct answer: B)
- Simple identity matching, so should be trivial for attentive listeners
- Used to filter unreliable participants (runtime screening: >30% failure rate after 2 trials)

## Sampling Strategy

### Recording Coverage (English/Japanese)
- **Japanese**: Each recording appears ≥5 times across all participants
- **English**: Each recording appears ≥2 times (mean ~11 times)
- Enables per-recording accuracy analysis for recording QC

### Recording Coverage (Mandarin)
- No minimum coverage requirement—annotations are reliable
- Fixed at 15 participants for human baseline comparison

### Within-Participant Design
- **No-repeat preference**: Algorithm prefers triplets where no recording is heard twice
- **Fallback**: Allows minimal repeats if needed (did not happen with our dataset / parameters) 
- **Round-robin ordering**: Any repeated recordings are maximally spaced

### Balanced Coverage
- **Recording Prioritization**: Assignment algorithm uses a multi-stage priority queue:
  1. **Under-covered recordings**: Priority to recordings below the minimum coverage threshold.
  2. **Global rarity**: Secondary preference for recordings with fewer total assignments (balancing coverage beyond the minimum).
  3. **Triplet freshness**: Tertiary preference for triplets not yet seen by other participants.
- **Answer Counterbalancing**: Correct answer (A vs. B) is randomized (~50/50 split) for both regular and catch trials.
- **Contrast Direction**: Triplet pool generation ensures both directions (Target=Pattern1 vs. Target=Pattern2) are represented.

## Directory Structure

```
human_abx/
├── DESIGN.md                  # This document
├── generate_human_abx.py      # Generation + audio materialization
├── deploy.py                  # Deploy to web server
├── .gitignore                 # Excludes generated files
├── generated/                 # Intermediate files (gitignored)
│   ├── {dataset}_triplet_pool.csv
│   ├── {dataset}_used_audio*.csv
│   ├── {dataset}_summary.txt
│   └── {dataset}_filtered_items.csv  (Mandarin only)
└── web/
    ├── index.html             # Experiment UI
    ├── app.js                 # Frontend logic
    ├── styles.css             # Styling
    ├── assign.php             # Prolific assignment handler
    ├── save_responses.php     # Response recording
    ├── lists/                 # Participant lists (gitignored)
    │   └── {dataset}/participant_*.csv
    └── audio/                 # Materialized clips (gitignored)
        ├── {dataset}/         # Main audio
        └── {dataset}_syn/     # Catch trial audio
```

### Participant List Format
Each CSV contains columns:
- `phone_sequence`: Word/mora/pinyin sequence for A and X
- `phone_sequence_b`: Word/mora/pinyin sequence for B (differs for catch trials)
- `accent_a`, `accent_b`: Accent pattern labels
- `speaker_ab`, `speaker_x`: Speaker IDs
- `file_a`, `file_b`, `file_x`: Audio file paths (relative to audio_path.txt)
- `is_catch`: Boolean flag for catch trials
- `correct_answer`: "A" or "B"
- `audio_source`: "main" for regular trials, "syn" for catch trials (use corresponding audio_path.txt)

## Generating Lists and Audio

```bash
cd /home/smcintosh/fastabx

# English stress
uv run python human_abx/generate_human_abx.py \
    --dataset stress \
    --trials-per-participant 70 \
    --min-responses-per-recording 2 \
    --materialize-audio

# Japanese pitch accent
uv run python human_abx/generate_human_abx.py \
    --dataset pitch_accent \
    --trials-per-participant 100 \
    --min-responses-per-recording 5 \
    --materialize-audio

# Mandarin tone
uv run python human_abx/generate_human_abx.py \
    --dataset mandarin_tone \
    --trials-per-participant 100 \
    --min-responses-per-recording 1 \
    --pinyin-freq-file metadata/junda_syllable_freq_without_tones.txt \
    --top-pinyins 50 \
    --materialize-audio
```

Omit `--materialize-audio` if you only want to regenerate participant lists.
The summary now includes a recording-appearance distribution so you can verify coverage
when using lower minimums.

## Web Deployment

### Server Requirements
- PHP 7.4+
- Write permissions for the `data/` directory (created automatically)

### Deployment Steps

1. Generate participant lists and audio (see "Generating Lists and Audio" above).

2. Deploy to web server:
   ```bash
   uv run python human_abx/deploy.py --target ~/public_html/human_abx
   ```
   Use `--clean` to remove an existing deployment first (deletes all response data!).

3. Verify permissions:
   Ensure the web server can write to the `data/` directory within the deployed folder.

### Assignment Logic
The `assign.php` script handles participant management:
- Assigns available participant lists in a round-robin fashion.
- Locks lists to prevent concurrent assignment.
- Tracks status (in_progress, completed, screened_out).
- Redirects users to the main experiment (`index.html`) with their assigned list.

### Testing
You can test the experiment without Prolific integration in two ways:

1. **Direct Link**: Bypass assignment logic and test a specific list.
   ```
   https://yourserver.com/human_abx/index.html?list=lists/stress/participant_000.csv
   ```

2. **Assignment System**: Test the automatic assignment logic with a custom ID.
   ```
   https://yourserver.com/human_abx/assign.php?dataset=stress&participant_id=test_user_01
   ```

### Response Data
Responses are saved to `data/` within the deployment directory:
- `responses_{participant}_{list}_{timestamp}.csv` - Individual response files
- `submissions.log` - Master log of all submissions
- `assignments_{dataset}.json` - State file tracking which lists are assigned/completed

### URL Parameters

| Parameter | Description |
|-----------|-------------|
| `dataset` | Dataset to assign from (stress, pitch_accent, mandarin_tone) |
| `list` | Path to participant list CSV (direct access) |
| `PROLIFIC_PID` | Prolific participant ID (passed by Prolific) |
| `participant_id` | Generic participant ID (fallback if no Prolific PID) |
| `skip` | Set `skip=true` to skip the tutorial (for testing) |

## Prolific Integration

### Configuration
Edit `web/save_responses.php` (or the deployed copy) to set Prolific-specific URLs:

- `$completion_url`: Redirect for successful completion.
- `$attention_fail_url`: Redirect for participants who fail catch trials (screened out).
- `$bonus_url`: Redirect for participants with high accuracy (eligible for bonus).
- `$bonus_threshold`: Accuracy threshold for bonus (default 0.80).

### Study URL
Set your Prolific study URL to point to `assign.php` with the necessary parameters:

```
https://yourserver.com/human_abx/assign.php?dataset=stress&PROLIFIC_PID={{%PROLIFIC_PID%}}&STUDY_ID={{%STUDY_ID%}}&SESSION_ID={{%SESSION_ID%}}
```

### Bonus Payments
Participants who achieve high accuracy (>80% by default) on the main ABX trials are automatically redirected to the `$bonus_url`. The `submissions.log` file records `COMPLETED_BONUS` for these users, facilitating easy bonus administration.