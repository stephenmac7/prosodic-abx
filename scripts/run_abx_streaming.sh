#!/usr/bin/env bash
# Run feature extraction and ABX scoring one model at a time.
#
# This script keeps disk usage low by deleting each model's feature directory
# after its ABX results have been written successfully.
#
# Usage:
#   bash scripts/run_abx_streaming.sh
#
# Optional environment variables:
#   MODELS_FILE=models_to_test.txt
#   FEATURE_ROOT=features
#   TASKS="stress pitch_accent"
#   PYTHON=python
#   LOW_MEMORY=1

set -euo pipefail

MODELS_FILE="${MODELS_FILE:-models_to_test.txt}"
FEATURE_ROOT="${FEATURE_ROOT:-features}"
PYTHON="${PYTHON:-python}"
LOW_MEMORY="${LOW_MEMORY:-1}"

DEFAULT_TASKS=(
  stress
  pitch_accent
  stress_syn
  stress_kokoro
  pitch_accent_syn
  pitch_accent_in_context
  tone_syn
  tone
)

if [[ -n "${TASKS:-}" ]]; then
  read -r -a TASK_LIST <<< "$TASKS"
else
  TASK_LIST=("${DEFAULT_TASKS[@]}")
fi

if [[ ! -f "$MODELS_FILE" ]]; then
  echo "Model list not found: $MODELS_FILE" >&2
  exit 1
fi

model_key() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]'
}

echo "Models file: $MODELS_FILE"
echo "Feature root: $FEATURE_ROOT"
echo "Tasks: ${TASK_LIST[*]}"
echo

for task in "${TASK_LIST[@]}"; do
  item_dir="abx_items/$task"
  item_file="$item_dir/items.csv"

  if [[ ! -f "$item_file" ]]; then
    echo "Skipping $task: missing $item_file"
    continue
  fi

  echo "=============================="
  echo "Task: $task"
  echo "=============================="

  while IFS= read -r model || [[ -n "$model" ]]; do
    [[ -z "$model" ]] && continue
    [[ "$model" =~ ^[[:space:]]*# ]] && continue

    key="$(model_key "$model")"
    result_csv="results/$task/$key.csv"
    feature_dir="$FEATURE_ROOT/$task/$key"

    echo
    echo "Model: $model"

    if [[ -f "$result_csv" ]]; then
      echo "Result already exists, skipping: $result_csv"
      continue
    fi

    if [[ -d "$feature_dir" ]]; then
      echo "Removing stale feature directory: $feature_dir"
      rm -rf "$feature_dir"
    fi

    if [[ "$LOW_MEMORY" == "1" ]]; then
      "$PYTHON" extract_features.py "$item_dir" "$FEATURE_ROOT" --model "$model" --low-memory
    else
      "$PYTHON" extract_features.py "$item_dir" "$FEATURE_ROOT" --model "$model"
    fi

    "$PYTHON" run_abx.py "$item_dir" "$FEATURE_ROOT" --model "$model"

    echo "Removing feature directory after successful ABX: $feature_dir"
    rm -rf "$feature_dir"
  done < "$MODELS_FILE"
done

echo
echo "All requested ABX runs finished."
