#!/usr/bin/env python3
"""
Format a LaTeX table from synth proxy summary.json files.

Usage:
  uv run python scripts/format_synth_proxy_table.py \
    --row "Japanese (S)" plots/synth_proxy_stats/pitch_accent_vs_pitch_accent_syn/summary.json \
    --row "Mandarin (S)" plots/synth_proxy_stats/mandarin_tone_vs_mandarin_tone_syn/summary.json \
    --row "English (S)" plots/synth_proxy_stats/stress_vs_stress_syn/summary.json \
    --row "English (K)" plots/synth_proxy_stats/stress_vs_stress_kokoro/summary.json \
    --label "tab:synth-proxy"
"""

import argparse
import json
from pathlib import Path


def _parse_median_iqr(value):
    """Parse 'median [q1, q3]' string, return (median, q1, q3) or None."""
    if value is None or value == "N/A":
        return None
    try:
        parts = str(value).split()
        median = float(parts[0])
        q1 = float(parts[1].strip("[,"))
        q3 = float(parts[2].strip("],"))
        return (median, q1, q3)
    except (ValueError, IndexError):
        return None


def _format_value(value, decimals, show_iqr=False):
    if value is None:
        return "N/A"
    if isinstance(value, tuple):
        median, q1, q3 = value
        if show_iqr:
            return f"{median:.{decimals}f}$^{{{q3:.{decimals}f}}}_{{{q1:.{decimals}f}}}$"
        return f"{median:.{decimals}f}"
    return f"{value:.{decimals}f}"


def _format_percent_value(value, decimals, show_iqr=False):
    if value is None:
        return "N/A"
    if isinstance(value, tuple):
        median, q1, q3 = value
        if show_iqr:
            return f"{median * 100:.{decimals}f}$^{{{q3 * 100:.{decimals}f}}}_{{{q1 * 100:.{decimals}f}}}$"
        return f"{median * 100:.{decimals}f}"
    return f"{value * 100:.{decimals}f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--row",
        action="append",
        nargs=2,
        metavar=("LABEL", "JSON_PATH"),
        required=True,
        help="Row label and summary.json path.",
    )
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--regret-decimals", type=int, default=2)
    parser.add_argument("--corr-decimals", type=int, default=2)
    parser.add_argument("--show-iqr", action="store_true", help="Include IQR for median columns")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows = []
    for label, path_str in args.row:
        path = Path(path_str)
        data = json.loads(path.read_text())
        layer = data["layerwise"]
        regret_median = _parse_median_iqr(layer.get("regret_median_iqr"))
        pearson_median = _parse_median_iqr(layer.get("pearson_median_iqr"))

        global_delta = data["global"].get("model_uniform_minus_global")
        global_percentile = data["global"].get("global_percentile")

        rows.append(
            {
                "label": label,
                "regret": _format_percent_value(regret_median, args.regret_decimals, args.show_iqr),
                "pearson": _format_value(pearson_median, args.corr_decimals, args.show_iqr),
                "global_percentile": _format_value(global_percentile, 0),
                "global_delta": _format_percent_value(global_delta, args.regret_decimals),
            }
        )

    caption_text = (
        f"\\textbf{{TTS-based layer and model selection quality.}} "
        f"Subscript $m$ denotes median across models. "
        f"$\\Delta$ ABX is the improvement in ABX score when using synthesized speech to select a model and layer vs. random. "
        f"S = standard TTS; K = Kokoro."
    )
    lines = [
        r"\begin{table}[t]",
        f"\\caption{{{caption_text}}}",
    ]
    if args.label:
        lines.append(f"\\label{{{args.label}}}")
    lines.extend(
        [
            r"\centering",
            r"\small",
            r"\begin{tabular}{l c c c c}",
            r"\toprule",
            r" & \multicolumn{2}{c}{Local} & \multicolumn{2}{c}{Global} \\",
            r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}",
            r" & Regret$_m$ (\%) & $r_m$ & Percentile & $\Delta$ ABX (\%) \\",
            r"\midrule",
        ]
    )

    for row in rows:
        lines.append(
            f"{row['label']} & {row['regret']} & {row['pearson']} & {row['global_percentile']} & {row['global_delta']} \\\\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )

    output = "\n".join(lines) + "\n"
    if args.output:
        args.output.write_text(output)
    else:
        print(output)


if __name__ == "__main__":
    main()
