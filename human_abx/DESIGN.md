# Human ABX Experiment Design

## Overview

This experiment evaluates how well humans discriminate prosodic minimal pairs, providing:
1. A baseline for comparing SSL model performance (all three languages)
2. Quality control for annotation validation (English stress and Japanese pitch accent only—Mandarin annotations are already reliable)

## Datasets

### English Lexical Stress
- **Items**: 598 recordings from 10 speakers
- **Contrasts**: 15 noun-verb minimal pairs (e.g., "CONduct" vs "conDUCT")
- **Excluded words**: "research", "transfer" (not true minimal pairs)
- **Participants needed**: 15 (Generated 32 lists for coverage)
- **Trials per participant**: 70 (63 regular + 7 catch)

### Japanese Pitch Accent
- **Items**: 909 recordings from 10 speakers
- **Contrasts**: ~46 pitch accent minimal pairs (e.g., あめ "rain" vs "candy")
- **Participants needed**: 15 (Generated 18 lists for coverage)
- **Trials per participant**: 100 (90 regular + 10 catch)

### Mandarin Lexical Tone
- **Items**: 1008 recordings (subsampled to top 50 most frequent pinyins)
- **Contrasts**: 6 tone pairs (T1-T2, T1-T3, T1-T4, T2-T3, T2-T4, T3-T4)
- **Participants needed**: 15 (Generated 15 lists)
- **Trials per participant**: 100 (90 regular + 10 catch)
- **Note**: Unlike English/Japanese, this is for human baseline only—annotations are already reliable, so no per-recording QC coverage is required.

## ABX Task Design

### Trial Structure
Each trial presents three audio clips:
- **A**: Reference recording (accent pattern 1)
- **B**: Contrast recording (accent pattern 2, same speaker as A)
- **X**: Probe recording (matches A's accent pattern, different speaker)

Participant task: "Is X more similar to A or B?"

Correct answer is counterbalanced (~50% A, ~50% B).

### Across-Speaker Mode
- A and B are from the same speaker
- X is from a different speaker
- Tests whether prosodic contrast is perceivable across speaker variability

### Catch Trials
- ~10% of trials (10 per participant)
- Use synthesized speech from the corresponding `*_syn` dataset
- A, B, and X are all from the same TTS speaker
- Either A=X (correct answer: A) or B=X (correct answer: B)
- Simple identity matching—no phonetic contrast to parse
- Should be trivially easy for attentive listeners
- Used to filter unreliable participants (runtime screening: >30% failure rate after 2 trials)
- **Note**: We target at least 15 participants for each dataset to ensure stable estimates.

## Sampling Strategy

### Recording Coverage (English/Japanese)
- Each recording appears ≥5 times across all participants
- Enables per-recording accuracy analysis for annotation QC
- Recordings with low accuracy → candidates for re-annotation

### Recording Coverage (Mandarin)
- No minimum coverage requirement—annotations are reliable
- Fixed at 15 participants for human baseline comparison

### Within-Participant Design
- **No-repeat preference**: Algorithm prefers triplets where no recording is heard twice
- **Fallback**: Allows minimal repeats if needed (max observed: 2)
- **Round-robin ordering**: Any repeated recordings are maximally spaced

### Balanced Coverage
- Triplets sampled evenly across all minimal pairs
- Both contrast directions tested (accent 1→2 and 2→1)

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

# Mandarin tone (human baseline only, no per-recording QC)
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

## Prolific Deployment

### Setup

1. Generate participant lists and audio (see "Generating Lists and Audio" above)

2. Deploy to web server:
```bash
uv run python human_abx/deploy.py --target ~/public_html/human_abx
```

Use `--clean` to remove an existing deployment first (deletes all response data!).

3. Configure `save_responses.php`:
   - Edit `$completion_url` to your Prolific completion URL
   - Verify `$num_lists` in `assign.php` matches generated participant lists

### Prolific Study URL

Set your study URL to:
```
https://yourserver.com/human_abx/assign.php?dataset=stress&PROLIFIC_PID={{%PROLIFIC_PID%}}&STUDY_ID={{%STUDY_ID%}}&SESSION_ID={{%SESSION_ID%}}
```

The `assign.php` script will:
- Assign participants to lists in round-robin fashion
- Track assignments to ensure balanced distribution
- Redirect to the experiment with all necessary parameters

### Response Data

Responses are saved to `~/public_html/human_abx/data/`:
- `responses_{participant}_{list}_{timestamp}.csv` - Individual response files
- `submissions.log` - Master log of all submissions (includes accuracy and bonus status)
- `assignments_{dataset}.json` - Participant-to-list assignments

### Bonus Payments

Participants who achieve high accuracy on ABX trials (non-catch) are automatically redirected to a bonus completion URL. Configure this in `save_responses.php`:
- `$bonus_url`: Prolific completion URL for bonus-eligible participants
- `$bonus_threshold`: Accuracy threshold (set to 80%)

The submissions log records `COMPLETED_BONUS` status for bonus-qualifying participants.

### URL Parameters

| Parameter | Description |
|-----------|-------------|
| `list` | Path to participant list CSV |
| `submit` | URL to save_responses.php |
| `PROLIFIC_PID` | Prolific participant ID |
| `STUDY_ID` | Prolific study ID |
| `SESSION_ID` | Prolific session ID |
| `skip` | Skip tutorial (for testing) |

Note: `completion_url` is configured in `save_responses.php` and returned after successful submission.

## Analysis

### Participant Filtering
1. Score catch trials per participant
2. Exclude participants screened out at runtime or with high catch trial failure rate

### Recording-Level QC (English/Japanese only)
1. Compute accuracy per recording across all appearances
2. Flag recordings with accuracy significantly below mean
3. Re-examine flagged recordings for annotation errors
4. If labels are corrected, re-score affected triplets (human responses remain valid—only the "correct answer" changes)
5. Exclude triplets where recordings no longer form valid contrasts after relabeling

### Contrast-Level Results
1. Aggregate accuracy by minimal pair
2. Compare with SSL model ABX error rates
3. Identify contrasts where humans >> models (or vice versa)
