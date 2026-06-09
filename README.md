# Prosodic ABX

This repository contains code for running Prosodic ABX experiments and related
analysis scripts. The paper for Prosodic ABX is available at https://arxiv.org/abs/2604.02102

## Installation

The recommended setup uses Python 3.12 and a local `venv`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
bash install.sh
```

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
python prepare_data.py -tone
```


