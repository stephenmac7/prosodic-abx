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


def _parse_ci(ci_str):
    """Parse '[lower, upper]' string, return (lower, upper) or None."""
    if ci_str is None or ci_str == "N/A":
        return None
    try:
        stripped = ci_str.strip("[]")
        parts = stripped.split(",")
        return (float(parts[0].strip()), float(parts[1].strip()))
    except (ValueError, IndexError):
        return None


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


def _format_ci_latex(ci, decimals=2, scale=1.0):
    """Format CI as LaTeX: $[lower, upper]$."""
    if ci is None:
        return "N/A"
    lo, hi = ci
    lo_s = f"{lo * scale:.{decimals}f}"
    hi_s = f"{hi * scale:.{decimals}f}"
    # Add \phantom{-} for positive values to align with negatives
    if lo * scale >= 0:
        lo_s = f"\\phantom{{-}}{lo_s}"
    return f"$[{lo_s},$ & ${hi_s}]$"


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
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows = []
    for label, path_str in args.row:
        path = Path(path_str)
        data = json.loads(path.read_text())
        layer = data["layerwise"]
        glob = data["global"]

        regret_parsed = _parse_median_iqr(layer.get("regret_median_iqr"))
        regret_pct = f"{regret_parsed[0] * 100:.{args.regret_decimals}f}" if regret_parsed else "N/A"

        pearson_parsed = _parse_median_iqr(layer.get("pearson_median_iqr"))
        pearson_str = f"{pearson_parsed[0]:.{args.corr_decimals}f}" if pearson_parsed else "N/A"

        rho = glob.get("model_rank_rho")
        rho_str = f"{rho:.{args.corr_decimals}f}" if rho is not None else "N/A"

        rho_ci = _parse_ci(glob.get("model_rank_rho_ci"))
        rho_ci_str = _format_ci_latex(rho_ci, args.corr_decimals)

        rows.append(
            {
                "label": label,
                "regret": regret_pct,
                "pearson": pearson_str,
                "rho": rho_str,
                "rho_ci": rho_ci_str,
            }
        )

    caption_text = (
        r"\textbf{TTS proxy quality for layer and model selection.} "
        r"Subscript $m$ denotes median across models. "
        r"Regret is the increase in error rate on natural speech when choosing a layer based on synthesized speech. "
        r"$\rho_\textrm{model}$ is the Spearman rank correlation of best-layer error rates across models. "
        r"S = standard TTS; K = Kokoro."
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
            r"\resizebox{\columnwidth}{!}{%",
            r"\begin{tabular}{l c c @{\quad} c @{\enspace} l @{\;} r}",
            r"\toprule",
            r" & $r_m$ & Regret$_m$ (\%) & \multicolumn{3}{c}{$\rho_\textrm{model}$ [95\% CI]} \\",
            r"\cmidrule(r){2-3} \cmidrule(l){4-6}",
        ]
    )

    for row in rows:
        lines.append(
            f"{row['label']} & {row['pearson']} & {row['regret']} & {row['rho']} & {row['rho_ci']}\\\\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
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
