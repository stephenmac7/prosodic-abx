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

Google Text-to-Speech synthesis requires a Google Cloud project with the
Text-to-Speech API enabled, billing enabled, and application-default credentials
configured for the account you want to use.

Kokoro English synthesis uses Montreal Forced Aligner for word alignment. Install
MFA following the official MFA documentation, and make sure the `mfa` command is
available before running the Kokoro synthesis script.

## Data

Prepare the English stress and Japanese pitch accent data:

```bash
python prepare_data.py
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

## Synthesis Corpus

```bash
python synthesize_wavs/synth_en.py
python synthesize_wavs/synth_jp.py
python synthesize_wavs/synth_zh.py
python synthesize_wavs/synth_en_kokoro.py --mfa-command /path/to/mfa
```



