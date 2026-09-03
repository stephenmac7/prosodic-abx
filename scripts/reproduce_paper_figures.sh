#!/usr/bin/env bash
set -euo pipefail

# Reproduce the paper-facing figures and tables from existing ABX results.
# This script keeps plots/ focused on the artifacts used in the paper.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if command -v uv >/dev/null 2>&1; then
  PYTHON=(uv run python)
else
  PYTHON=(python)
fi

PLOTS_DIR="$ROOT_DIR/plots"
rm -rf "$PLOTS_DIR"
mkdir -p "$PLOTS_DIR"

"${PYTHON[@]}" scripts/plot_human_SSL_bootstrap.py
mv plots/human_ssl/human_ssl_boxplot.png plots/human_ssl_boxplot.png
mv plots/human_ssl/human_ssl_boxplot.pdf plots/human_ssl_boxplot.pdf
rm -rf plots/human_ssl

"${PYTHON[@]}" scripts/plot_word_error_rate_bootstrapped.py \
  data/human_abx/stress
mv plots/human_abx/human_vs_machine_word_error_English_stress.png \
  plots/human_vs_machine_word_error_English_stress.png
mv plots/human_abx/human_vs_machine_word_error_English_stress.pdf \
  plots/human_vs_machine_word_error_English_stress.pdf
rm -rf plots/human_abx

synth_tmp="$(mktemp -d)"
trap 'rm -rf "$synth_tmp"' EXIT

"${PYTHON[@]}" scripts/plot_synth_proxy_stats.py \
  pitch_accent pitch_accent_syn \
  --output-dir "$synth_tmp/pitch_accent_vs_pitch_accent_syn"
"${PYTHON[@]}" scripts/plot_synth_proxy_stats.py \
  tone tone_syn \
  --output-dir "$synth_tmp/tone_vs_tone_syn"
"${PYTHON[@]}" scripts/plot_synth_proxy_stats.py \
  stress stress_syn \
  --output-dir "$synth_tmp/stress_vs_stress_syn"
"${PYTHON[@]}" scripts/plot_synth_proxy_stats.py \
  stress stress_kokoro \
  --output-dir "$synth_tmp/stress_vs_stress_kokoro"

mkdir -p plots/synth_proxy_stats
"${PYTHON[@]}" scripts/format_synth_proxy_table.py \
  --row "Japanese (G)" "$synth_tmp/pitch_accent_vs_pitch_accent_syn/summary.json" \
  --row "Mandarin (G)" "$synth_tmp/tone_vs_tone_syn/summary.json" \
  --row "English (G)" "$synth_tmp/stress_vs_stress_syn/summary.json" \
  --row "English (K)" "$synth_tmp/stress_vs_stress_kokoro/summary.json" \
  --output plots/synth_proxy_stats/synth_proxy_table.tex

"${PYTHON[@]}" scripts/analyze_in_context.py pitch_accent

"${PYTHON[@]}" scripts/plot_cross_analysis.py

echo "Paper figures and tables written to plots/."
