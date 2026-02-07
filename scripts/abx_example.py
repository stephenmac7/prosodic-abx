#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ABX DTW example using precomputed features (japanese-hubert-large layer 18).

This script:
1) Loads precomputed layer-18 features from features.pt.
2) Computes DTW alignments for multiple A↔X and B↔X pairs.
3) Averages curves and plots them on X's timeline with phoneme markers.
"""

from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR.parent / "plots" / "example"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "japanese-hubert-large"
LAYER_IDX = 18  # 1-based, consistent with extract_features.py
FEATURES_PT = Path(
    "/home/sunhaitong/fastabx/features/pitch_accent/"
    "japanese-hubert-large/l18/features.pt"
)

X_TEXTGRID = Path(
    "/home/sunhaitong/fastabx/scripts/example_wavs/mfa-aligned/"
    "S008_20_りょうしん_両親.TextGrid"
)
X_PREFIX = "S008_20"


def _load_features_map(features_pt: Path):
    if not features_pt.exists():
        raise FileNotFoundError(f"Missing features file: {features_pt}")
    data = torch.load(features_pt, weights_only=False)
    if "features" not in data:
        raise ValueError(f"Invalid features file (missing 'features'): {features_pt}")
    return data["features"], float(data.get("frequency", 50.0))


def _find_key(prefix: str, keys: list[str]) -> str:
    matches = [k for k in keys if prefix in k]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        # Prefer the shortest match (e.g., S008/S008_20_...) to avoid unrelated keys.
        return sorted(matches, key=len)[0]
    raise KeyError(f"No key found containing: {prefix}")


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    return x / denom


def dtw_path(cost: np.ndarray) -> list[tuple[int, int]]:
    n, m = cost.shape
    dp = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    dp[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i, j] = cost[i - 1, j - 1] + min(
                dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1]
            )

    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        prev = np.argmin([dp[i - 1, j - 1], dp[i - 1, j], dp[i, j - 1]])
        if prev == 0:
            i -= 1
            j -= 1
        elif prev == 1:
            i -= 1
        else:
            j -= 1

    path.reverse()
    return path


def dtw_avg_cost_per_x_frame(ref: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Compute the DTW cost curve on X's timeline.

    For each frame j in X, this returns the *average* DTW alignment cost of all
    reference frames i that align to j along the optimal DTW path.
    """
    # Normalize each frame vector to use cosine distance.
    ref = _normalize_rows(ref)
    x = _normalize_rows(x)

    # Pairwise cosine distance matrix: ref_frames x x_frames
    cost = 1.0 - np.matmul(ref, x.T)

    # DTW alignment path (i, j) pairs.
    path = dtw_path(cost)

    # Collect all costs aligned to each X frame j.
    aligned = [[] for _ in range(x.shape[0])]
    for i, j in path:
        aligned[j].append(cost[i, j])

    # Average cost per X frame (this is the curve you described).
    curve = np.array([np.mean(vals) if vals else np.nan for vals in aligned])

    # If any X frames have no aligned points (rare), fill by interpolation.
    if np.isnan(curve).any():
        idx = np.arange(len(curve))
        valid = ~np.isnan(curve)
        curve = np.interp(idx, idx[valid], curve[valid])

    return curve


def parse_phone_intervals(textgrid_path: Path):
    """Parse (xmin, xmax, text) for the 'phones' IntervalTier."""
    # Some MFA TextGrid exports are UTF-16 encoded.
    lines = textgrid_path.read_text(encoding="utf-16").splitlines()
    intervals = []

    in_phones = False
    current = {}
    for line in lines:
        line = line.strip()

        if line.startswith("item [") and in_phones:
            # A new tier starts; stop once we leave the phones tier.
            break

        if line.startswith("name ="):
            tier_name = line.split("=", 1)[1].strip().strip('"')
            in_phones = tier_name == "phones"
            continue

        if not in_phones:
            continue

        if line.startswith("xmin ="):
            current["xmin"] = float(line.split("=", 1)[1].strip())
        elif line.startswith("xmax ="):
            current["xmax"] = float(line.split("=", 1)[1].strip())
        elif line.startswith("text ="):
            text = line.split("=", 1)[1].strip().strip('"')
            current["text"] = text
            if {"xmin", "xmax", "text"} <= current.keys():
                intervals.append((current["xmin"], current["xmax"], current["text"]))
                current = {}

    if not intervals:
        raise ValueError(f"No phone intervals found in {textgrid_path}")

    return intervals


def _prepare_waveform(audio: torch.Tensor, sr: int, max_points: int = 5000):
    """Return time axis and normalized waveform for lightweight plotting."""
    y = audio.squeeze(0).cpu().numpy()
    if y.size == 0:
        return np.array([]), np.array([])
    if y.size > max_points:
        step = int(np.ceil(y.size / max_points))
        y = y[::step]
    y = y / (np.max(np.abs(y)) + 1e-8)
    t = np.linspace(0, len(audio.squeeze(0)) / sr, num=y.size)
    return t, y


def plot_curves_with_phonemes(
    t,
    curve_a,
    curve_b,
    out_path: Path,
    textgrid_path: Path,
):
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    # White background.
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # DTW curves (A↔X, B↔X).
    # ax.plot(
    #     t,
    #     curve_a,
    #     color="#7aa6d8",
    #     lw=1.6,
    #     marker="s",
    #     markersize=5.5,
    #     markerfacecolor="#2b6cb0",
    #     markeredgewidth=0.0,
    #     label="A ↔ X",
    # )
    # ax.plot(
    #     t,
    #     curve_b,
    #     color="#e0a07a",
    #     lw=1.6,
    #     marker="s",
    #     markersize=5.5,
    #     markerfacecolor="#c05621",
    #     markeredgewidth=0.0,
    #     label="B ↔ X",
    # )

    # BX - AX difference curve (on its own hidden y-axis).
    diff = curve_b - curve_a
    ax_diff = ax.twinx()
    ax_diff.plot(
        t,
        diff,
        color="#1f77b4",
        lw=1.8,
        label="Dis(B,X)-Dis(A,X)",
    )
    # Zero reference line for the difference.
    ax_diff.axhline(0.0, color="#4a4a4a", lw=1.0, linestyle="-", alpha=0.8)
    # Increase amplitude visually by widening the symmetric limits.
    max_abs = float(np.max(np.abs(diff))) if diff.size else 1.0
    ax_diff.set_ylim(-1.5 * max_abs, 1.5 * max_abs)
    ax_diff.set_yticks([])
    ax_diff.spines["right"].set_visible(False)
    ax_diff.spines["left"].set_visible(False)
    ax_diff.spines["top"].set_visible(False)
    ax_diff.spines["bottom"].set_visible(False)

    # Shade positive/negative regions between diff and zero.
    ax_diff.fill_between(
        t,
        0,
        diff,
        where=diff >= 0,
        color="#6baed6",
        alpha=0.25,
        interpolate=True,
        zorder=0,
    )
    ax_diff.fill_between(
        t,
        0,
        diff,
        where=diff < 0,
        color="#fdae6b",
        alpha=0.25,
        interpolate=True,
        zorder=0,
    )

    # Waveform on a secondary axis (kept subtle)
    # t_wav, y_wav = _prepare_waveform(audio_x, sr)
    # ax_wav = ax.twinx()
    # ax_wav.plot(t_wav, y_wav, color="#bfbfbf", lw=0.8, alpha=0.4, zorder=0)
    # ax_wav.set_ylim(-1.0, 1.0)
    # ax_wav.set_yticks([])
    # ax_wav.spines["right"].set_visible(False)
    # ax_wav.spines["left"].set_visible(False)
    # ax_wav.spines["top"].set_visible(False)
    # ax_wav.spines["bottom"].set_visible(False)

    # Overlay phoneme boundaries aligned to X time axis.
    if textgrid_path.exists():
        intervals = parse_phone_intervals(textgrid_path)
        # Draw boundaries first
        for idx, (t0, t1, label) in enumerate(intervals):
            ax.axvline(t0, color="#4a4a4a", linewidth=1.0, alpha=0.7, zorder=1)
            if idx == len(intervals) - 1:
                ax.axvline(t1, color="#4a4a4a", linewidth=1.0, alpha=0.7, zorder=1)
        # Then draw phoneme labels on top of everything
        for (t0, t1, label) in intervals:
            mid = 0.5 * (t0 + t1)
            ax.text(
                mid,
                0.5,
                label,
                ha="center",
                va="center",
                transform=ax.get_xaxis_transform(),
                fontsize=24,
                fontweight="bold",
                color="#1f2a44",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9, edgecolor="none"),
                zorder=10,
            )
    else:
        print(f"Warning: TextGrid not found, skipping phoneme overlay: {textgrid_path}")

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_yticks([])
    ax.set_title("")
    ax.grid(False)
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#6baed6", edgecolor="none", alpha=0.25, label="Dis(B,X) > Dis(A,X)"),
        Patch(facecolor="#fdae6b", edgecolor="none", alpha=0.25, label="Dis(B,X) < Dis(A,X)"),
    ]
    ax.legend(handles=legend_handles, frameon=True, framealpha=0.95, loc="upper left")

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.0)

    # X-axis label
    ax.set_xlabel("Time in X (s)")

    # Trim view to [0, waveform_end] if waveform is enabled, otherwise use t.
    if textgrid_path.exists():
        try:
            xmax = intervals[-1][1]
            ax.set_xlim(0.0, float(xmax))
        except Exception:
            if t.size > 0:
                ax.set_xlim(0.0, float(t[-1]))
    elif t.size > 0:
        ax.set_xlim(0.0, float(t[-1]))

    # Add an arrow at the end of the x-axis with unit label.
    # No axis arrow; keep the x-axis clean.


    plt.tight_layout()
    plt.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.savefig(out_path.with_suffix(".pdf"), dpi=200, bbox_inches="tight")
    plt.close()


def main():
    features_map, frequency = _load_features_map(FEATURES_PT)
    keys = [k for k in features_map.keys() if isinstance(k, str)]

    x_key = _find_key(X_PREFIX, keys)
    feats_x = features_map[x_key].cpu().numpy()

    a_keys = []
    for i in range(1, 11):
        if i == 8:
            continue
        prefix = f"S{i:03d}_20"
        try:
            a_keys.append(_find_key(prefix, keys))
        except KeyError:
            print(f"Warning: missing A key with prefix {prefix}")

    b_keys = []
    for i in range(1, 11):
        if i == 8:
            continue
        prefix = f"S{i:03d}_19"
        try:
            b_keys.append(_find_key(prefix, keys))
        except KeyError:
            print(f"Warning: missing B key with prefix {prefix}")

    if not a_keys or not b_keys:
        raise SystemExit("No A/B keys found in features.pt")

    a_curves = [
        dtw_avg_cost_per_x_frame(features_map[k].cpu().numpy(), feats_x)
        for k in a_keys
    ]
    b_curves = [
        dtw_avg_cost_per_x_frame(features_map[k].cpu().numpy(), feats_x)
        for k in b_keys
    ]

    curve_a = np.mean(np.stack(a_curves, axis=0), axis=0)
    curve_b = np.mean(np.stack(b_curves, axis=0), axis=0)

    time_x = np.arange(len(curve_a)) / frequency
    out_path = OUT_DIR / "ryoshin_dtw_japanese-hubert-large_l18"
    plot_curves_with_phonemes(
        time_x,
        curve_a,
        curve_b,
        out_path,
        X_TEXTGRID,
    )

    print(f"Saved: {out_path.with_suffix('.png')}")
    print(f"Saved: {out_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
