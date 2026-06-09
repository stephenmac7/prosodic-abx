#!/usr/bin/env bash
set -euo pipefail

# Install dependencies for the Prosodic ABX codebase.
# The default PyTorch build targets CUDA 12.8. Override PYTORCH_INDEX_URL if a
# different PyTorch build is needed.

PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
FASTABX_VERSION="${FASTABX_VERSION:-0.6.1}"

python -m pip install -U pip setuptools wheel

python -m pip install --no-cache-dir \
  torch==2.9.0 \
  torchaudio==2.9.0 \
  --index-url "${PYTORCH_INDEX_URL}"

python -m pip install --no-cache-dir \
  setuptools-scm==9.2.2 \
  ninja \
  numpy==2.3.5 \
  wheel

SETUPTOOLS_SCM_PRETEND_VERSION="${FASTABX_VERSION}" python -m pip install \
  --no-cache-dir \
  --no-build-isolation \
  --force-reinstall \
  --no-deps \
  "fastabx==${FASTABX_VERSION}"

python -m pip install --no-cache-dir -r requirements.txt

python - <<'PY'
import torch
import torchaudio
import torchcodec
import fastabx
import tgt
import polars

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("environment ok")
PY
