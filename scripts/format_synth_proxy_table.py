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
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--percent-decimals", type=int, default=0)
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
        topk_key = f"top{args.topk}_agreement_rate"
        topk_rate = layer.get(topk_key)
        regret_median = _parse_median_iqr(layer.get("regret_median_iqr"))
        spearman_median = _parse_median_iqr(layer.get("spearman_median_iqr"))
        pearson_median = _parse_median_iqr(layer.get("pearson_median_iqr"))

        global_regret = data["global"].get("global_regret")
        global_delta = data["global"].get("model_uniform_minus_global")
        global_percentile = data["global"].get("global_percentile")

        rows.append(
            {
                "label": label,
                "topk": _format_percent_value(topk_rate, args.percent_decimals),
                "regret": _format_percent_value(regret_median, args.regret_decimals, args.show_iqr),
                "spearman": _format_value(spearman_median, args.corr_decimals, args.show_iqr),
                "pearson": _format_value(pearson_median, args.corr_decimals, args.show_iqr),
                "global_regret": _format_percent_value(global_regret, args.regret_decimals),
                "global_delta": _format_percent_value(global_delta, args.regret_decimals),
                "global_percentile": _format_value(global_percentile, args.percent_decimals),
            }
        )

    caption_text = (
        f"\\textbf{{Layer and global selection quality.}} "
        f"Subscript $m$ denotes median across models. "
        f"Top-{args.topk} is the percentage of models where the best layer on synthesized speech is among the "
        f"top-{args.topk} layers on natural speech. "
        f"Percentile is the percentile rank of the natural ABX score of the best model+layer according to the synthesized dataset (higher is better). "
        f"$\\Delta$ ABX vs random is the improvement in ABX score when using synthesized speech to select a model and layer vs. random. "
        f"S = standard TTS; K = Kokoro."
    )
    lines = [
        r"\begin{table*}[t]",
        f"\\caption{{{caption_text}}}",
    ]
    if args.label:
        lines.append(f"\\label{{{args.label}}}")
    lines.extend(
        [
            r"\centering",
            r"\small",
            r"\begin{tabular}{l c c c c c c c}",
            r"\toprule",
            r"Task & \multicolumn{4}{c}{Local (layer selection)} & \multicolumn{3}{c}{Global (model+layer selection)} \\",
            r"\cmidrule(lr){2-5} \cmidrule(lr){6-8}",
            f" & Top-{args.topk} (\\%) & Regret$_m$ (\\%) & $\\rho_m$ & $r_m$ & Regret (\\%) & Percentile & $\\Delta$ ABX vs random (\\%) \\\\",
            r"\midrule",
        ]
    )

    for row in rows:
        lines.append(
            f"{row['label']} & {row['topk']} & {row['regret']} & {row['spearman']} & {row['pearson']} & {row['global_regret']} & {row['global_percentile']} & {row['global_delta']} \\\\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
        ]
    )

    output = "\n".join(lines) + "\n"
    if args.output:
        args.output.write_text(output)
    else:
        print(output)


if __name__ == "__main__":
    main()
