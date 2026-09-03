# Prosodic ABX

This repository contains code for running Prosodic ABX experiments and related
analysis scripts. The paper for Prosodic ABX is available at https://arxiv.org/abs/2604.02102

## Installation

```bash
uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate
bash install.sh
```

## Data

Prepare each natural-speech dataset explicitly:

```bash
python prepare_data.py --stress
python prepare_data.py --pitch-accent
```

For Mandarin tone experiments, download the MCAE-monosyllable dataset from
https://osf.io/h3uem. This project uses
`raw_data/tone and pinyin information.xlsx` and `Audio_files/neutral`. Place
them as:

```text
data/tone/tone and pinyin information.xlsx
data/tone/neutral.zip
```

Then run:

```bash
python prepare_data.py --tone
```

Prepare the fixed synthesized-speech corpora used in the paper:

```bash
python prepare_data.py --stress-syn
python prepare_data.py --stress-kokoro
python prepare_data.py --pitch-accent-syn
python prepare_data.py --tone-syn
```

## Generate Items

Generate fastabx item files from the prepared data:

```bash
python generate_items/stress.py
python generate_items/pitch_accent.py
python generate_items/pitch_accent.py --in-context
python generate_items/tone.py
python generate_items/stress_syn.py
python generate_items/stress_kokoro.py
python generate_items/pitch_accent_syn.py
python generate_items/tone_syn.py
```

## Run ABX

To run the full set of tasks and models while keeping feature files on a
temporary disk, use:

```bash
FEATURE_ROOT=/localdisk/$USER/prosody-abx-features \
bash scripts/run_abx_streaming.sh
```

The script extracts features for one task/model pair, runs ABX, then removes
those feature files after the result has been written.

## Reproduce Paper Figures

After ABX results have been generated, reproduce the paper-facing figures and
tables with:

```bash
bash scripts/reproduce_paper_figures.sh
```

By default, the plotting scripts write only the core summaries or paper
figures. Pass `--diagnostics` to the analysis scripts to also save intermediate
plots and per-model statistics.

## Synthesis Corpus

The synthesized speech used in the paper is released through the Hugging Face
dataset as fixed audio files. Use `prepare_data.py` above to reproduce the
paper's synthesized-speech experiments with the same audio.

The synthesis scripts are kept for users who want to regenerate the synthesized
corpora.

Google Text-to-Speech synthesis requires a Google Cloud project with the
Text-to-Speech API enabled, billing enabled, and application-default credentials
configured. Kokoro English synthesis uses Montreal Forced Aligner; install MFA
following the official MFA documentation before running the Kokoro script.

```bash
python synthesize_wavs/synth_en.py
python synthesize_wavs/synth_jp.py
python synthesize_wavs/synth_zh.py
python synthesize_wavs/synth_en_kokoro.py --mfa-command /path/to/mfa
```
