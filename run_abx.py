"""Run ABX discrimination test for Japanese pitch accent patterns.

This script runs ABX tests using the fastabx library to evaluate how well
speech representations (e.g., HuBERT, WavLM) discriminate pitch accent patterns.

Two modes:
  - ACROSS speaker (default): Tests if accent patterns are discriminable across different speakers
    ON: accent_pattern, BY: phone_sequence, ACROSS: speaker

  - WITHIN speaker (--within): Tests if accent patterns are discriminable for the same speaker
    ON: accent_pattern, BY: phone_sequence + speaker

Optional extra constraints:
  - Use --extra-by to add additional BY constraints (e.g., prev_phone next_phone, context_set, part_of_word).
  - Use --ax-match to enforce that A and X match on specific label columns (asymmetric constraint).

Workflow:
    # 1. Generate item file
    python generate_abx_dataset.py --min-moras 2

    # 2. Extract features (saves consolidated features.pt per layer)
    python extract_features.py abx_items/pitch_accent features --model HUBERT_BASE
    # Results: features/pitch_accent/hubert_base/l{1..12}/features.pt

    # 3. Run ABX test (evaluates all layers automatically)
    python run_abx.py abx_items/pitch_accent features --model HUBERT_BASE

Usage:
    python run_abx.py abx_items/pitch_accent features --model HUBERT_BASE
    python run_abx.py abx_items/pitch_accent features --model HUBERT_BASE --within
    python run_abx.py abx_items/pitch_accent features --model HUBERT_BASE --both
    python run_abx.py abx_items/pitch_accent features --model HUBERT_BASE --extra-by prev_phone next_phone
    python run_abx.py abx_items/stress features --model hubert --ax-match context_set part_of_word
"""

import argparse
import csv
from pathlib import Path

import polars as pl
import torch
from fastabx import Dataset, Score, Subsampler, Task
from fastabx.dataset import InMemoryAccessor


def build_dataset(item_path: Path, feature_file: Path) -> Dataset:
    """Load dataset from consolidated feature file.

    Args:
        item_path: Path to items.csv
        feature_file: Path to features.pt containing {"features": {...}, "frequency": float}

    Returns:
        Dataset ready for ABX evaluation
    """
    print(f"Loading features from {feature_file}...")
    data = torch.load(feature_file, weights_only=False)
    features_dict = data["features"]
    frequency = data["frequency"]
    print(f"Loaded {len(features_dict)} files (frequency={frequency} Hz)")

    # Read items
    df = pl.read_csv(item_path, schema_overrides={"#file": pl.String, "onset": pl.String, "offset": pl.String})
    df = df.with_columns(
        df["onset"].str.to_decimal(inference_length=len(df)),
        df["offset"].str.to_decimal(inference_length=len(df)),
    )

    # Compute frontiers (matches fastabx item_frontiers logic)
    # start = ceil(onset * freq - 0.5), end = floor(offset * freq - 0.5) + 1
    df = df.with_columns([
        ((pl.col("onset") * frequency - 0.5).ceil().cast(pl.Int64)).alias("_start"),
        ((pl.col("offset") * frequency - 0.5).floor().cast(pl.Int64) + 1).alias("_end"),
    ])

    # Slice features
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sliced = []
    for row in df.iter_rows(named=True):
        feat = features_dict[row["#file"]]
        start = row["_start"]
        end = min(row["_end"], feat.size(0))  # clamp to feature length
        sliced.append(feat[start:end].to(device))

    # Build indices and concatenate
    indices, pos = {}, 0
    for i, s in enumerate(sliced):
        indices[i] = (pos, pos + s.size(0))
        pos += s.size(0)

    labels = df.drop("_start", "_end")
    return Dataset(labels=labels, accessor=InMemoryAccessor(indices, torch.cat(sliced)))


def maybe_normalize(dataset: Dataset, distance: str) -> None:
    """Normalize in-place for angular/cosine distances."""
    if distance in ("angular", "cosine"):
        print("Normalizing features...")
        dataset.normalize_()


def compute_abx(
    dataset: Dataset,
    *,
    distance: str,
    across: bool,
    extra_by: list[str],
    ax_match: list[str],
    max_size_group: int | None,
    max_x_across: int | None,
    subsample_seed: int,
) -> float:
    """Compute ABX error rate from a preloaded dataset."""
    # Build task
    # ON: accent_pattern (what we discriminate)
    by_conditions = ["phone_sequence"]
    across_conditions = None

    if extra_by:
        by_conditions.extend(extra_by)
        print(f"Using extra BY constraints: {extra_by}")

    if across:
        # ACROSS speaker: A and B have same speaker, X has different speaker
        across_conditions = ["speaker"]
        print("Using across-speaker mode")
    else:
        # WITHIN speaker: A, B, X all have same speaker
        by_conditions.append("speaker")

    print(f"Building ABX task (ON: accent_pattern, BY: {by_conditions}, ACROSS: {across_conditions})...")
    subsampler = None
    if max_size_group is not None or (across and max_x_across is not None):
        subsampler = Subsampler(max_size_group, max_x_across, seed=subsample_seed)
        if subsampler.description(with_across=bool(across_conditions)):
            print(f"Subsampling: {subsampler.description(with_across=bool(across_conditions))}")

    task = Task(dataset, on="accent_pattern", by=by_conditions, across=across_conditions, subsampler=subsampler)

    print(f"Number of cells: {len(task)}")

    if len(task) == 0:
        msg = "No valid ABX cells could be formed."
        if not across:
            msg += (" Within-speaker mode requires at least 2 items per (phone_sequence, speaker, accent_pattern). "
                    "Consider using the default across-speaker mode for datasets with single recordings per condition.")
        raise ValueError(msg)

    # Score
    print(f"Computing ABX scores with {distance} distance...")
    constraints = None
    if ax_match:
        constraints = [pl.col(f"{c}_a") == pl.col(f"{c}_x") for c in ax_match]
        print(f"Using A/X match constraints: {ax_match}")
    score = Score(task, distance, constraints=constraints)

    # Collapse: average over speakers first (giving equal weight per phone_sequence + contrast),
    # then final mean over all (phone_sequence, contrast) combinations. If extra BY constraints
    # are provided, collapse across them before speaker.
    collapse_levels = ["speaker"]
    if extra_by:
        # Collapse extra BY constraints before speaker
        collapse_levels = [tuple(extra_by)] + collapse_levels

    return score.collapse(levels=collapse_levels)


def main():
    parser = argparse.ArgumentParser(
        description="Run ABX discrimination test for pitch accent patterns.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("dataset_path", type=Path,
                        help="Path to dataset directory (containing items.csv)")
    parser.add_argument("features", type=Path,
                        help="Root path for feature directory (features will be loaded from features/{dataset}/{model}/...)")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name (e.g., HUBERT_BASE, WAVLM_BASE)")

    parser.add_argument("--distance", type=str, default="angular",
                        choices=["angular", "cosine", "euclidean", "kl", "kl_symmetric", "identical"],
                        help="Distance metric")
    parser.add_argument("--extra-by", nargs="+", default=[],
                        help="Additional label columns to include in BY constraints (e.g., prev_phone next_phone)")
    parser.add_argument("--ax-match", nargs="+", default=[],
                        help="Label columns that must match between A and X (asymmetric constraint)")
    parser.add_argument("--within", action="store_true",
                        help="Test within speakers (A,B,X same speaker) instead of across")
    parser.add_argument("--both", action="store_true",
                        help="Compute within and across speaker scores using one dataset load")
    parser.add_argument("--max-size-group", type=int, default=None,
                        help="Max items for A, B, or X in each cell (subsampling)")
    parser.add_argument("--max-x-across", type=int, default=None,
                        help="Max X values per (A,B) in across mode (subsampling)")
    parser.add_argument("--subsample-seed", type=int, default=0,
                        help="Random seed for subsampling")

    args = parser.parse_args()

    # Resolve item file and dataset name
    item_path = args.dataset_path / "items.csv"
    dataset_name = args.dataset_path.name

    if not item_path.exists():
         raise FileNotFoundError(f"Item file not found: {item_path}")

    modes: list[tuple[str, bool]]
    if args.both:
        if args.within:
            print("Note: --within ignored because --both is set.")
        modes = [("within", False), ("across", True)]
    else:
        across_flag = not args.within
        modes = [("within" if args.within else "across", across_flag)]

    # Build feature directory path: features/{dataset_name}/{model_name}/
    features_dir_dataset_name = dataset_name if not dataset_name.startswith("csj_") else "csj"
    model_dir = args.features / features_dir_dataset_name / args.model.lower()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    # Find feature files: either l{i}/features.pt or features.pt (for MFCC/FBANK)
    def is_layer_dir(name: str) -> bool:
        return len(name) > 1 and name[0] == "l" and name[1:].isdigit()

    layer_dirs = [p for p in model_dir.iterdir() if p.is_dir() and is_layer_dir(p.name)]

    if layer_dirs:
        # Neural model with layer subdirectories
        layer_dirs = sorted(layer_dirs, key=lambda p: int(p.name[1:]))
        feature_sets = [(p.name, p / "features.pt") for p in layer_dirs]
    else:
        # MFCC/FBANK with single features.pt
        feature_file = model_dir / "features.pt"
        if not feature_file.exists():
            raise FileNotFoundError(f"Feature file not found: {feature_file}")
        feature_sets = [("features", feature_file)]

    # Determine output CSV path and check if results already exist
    results_dir = Path("results") / dataset_name
    csv_name = f"{args.model.lower()}.csv"
    output_csv = results_dir / csv_name

    if output_csv.exists():
        raise FileExistsError(f"Results already exist: {output_csv}")

    results: list[dict[str, str]] = []
    for layer_name, feature_file in feature_sets:
        if len(feature_sets) > 1:
            print(f"\n=== {layer_name} ===")

        dataset = build_dataset(item_path, feature_file)
        maybe_normalize(dataset, args.distance)

        for mode_name, across_flag in modes:
            if args.both:
                print(f"\n=== {mode_name.capitalize()}-speaker ===")
            error_rate = compute_abx(
                dataset,
                distance=args.distance,
                across=across_flag,
                extra_by=args.extra_by,
                ax_match=args.ax_match,
                max_size_group=args.max_size_group,
                max_x_across=args.max_x_across,
                subsample_seed=args.subsample_seed,
            )
            label = f" ({mode_name})" if args.both else ""
            print(f"ABX Error Rate{label}: {error_rate:.4f} ({error_rate * 100:.2f}%)")
            results.append(
                {
                    "layer": layer_name,
                    "mode": mode_name,
                    "error_rate": f"{error_rate:.6f}",
                }
            )

    # Write results
    results_dir.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["layer", "mode", "error_rate"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nWrote results to {output_csv}")


if __name__ == "__main__":
    main()
