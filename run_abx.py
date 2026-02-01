"""Run ABX discrimination test for prosodic patterns (Across-Speaker).

This script runs ABX tests using the fastabx library to evaluate how well
speech representations (e.g., HuBERT, WavLM) discriminate prosodic patterns
across different speakers.

Logic:
  - ON: accent_pattern
  - BY: phone_sequence
  - ACROSS: speaker (A and B share speaker, X has different speaker)

Usage:
    python run_abx.py abx_items/pitch_accent features --model HUBERT_BASE
    python run_abx.py abx_items/pitch_accent features --model HUBERT_BASE --extra-by prev_phone
"""

import argparse
import csv
from pathlib import Path

import polars as pl
import torch
from fastabx import Dataset, Score, Subsampler, Task
from fastabx.dataset import InMemoryAccessor


def build_dataset(item_path: Path, feature_file: Path) -> Dataset:
    """Load dataset from consolidated feature file."""
    print(f"Loading features from {feature_file}...")
    data = torch.load(feature_file, weights_only=False)
    features_dict = data["features"]
    frequency = data["frequency"]
    print(f"Loaded {len(features_dict)} files (frequency={frequency} Hz)")

    # Read items
    df = pl.read_csv(
        item_path,
        schema_overrides={"#file": pl.String, "onset": pl.String, "offset": pl.String},
    )
    df = df.with_columns(
        df["onset"].str.to_decimal(inference_length=len(df)),
        df["offset"].str.to_decimal(inference_length=len(df)),
    )

    # Compute frontiers (matches fastabx item_frontiers logic)
    df = df.with_columns(
        [
            ((pl.col("onset") * frequency - 0.5).ceil().cast(pl.Int64)).alias("_start"),
            ((pl.col("offset") * frequency - 0.5).floor().cast(pl.Int64) + 1).alias(
                "_end"
            ),
        ]
    )

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
    extra_by: list[str],
    ax_match: list[str],
    max_size_group: int | None,
    max_x_across: int | None,
    subsample_seed: int,
) -> tuple[float, pl.DataFrame]:
    """Compute ABX error rate from a preloaded dataset."""
    # Standard ABX config:
    # ON: accent_pattern (what we discriminate)
    # BY: phone_sequence (phonetic control)
    # ACROSS: speaker (A and B match speaker, X is different)
    by_conditions = ["phone_sequence"]
    across_conditions = ["speaker"]

    if extra_by:
        by_conditions.extend(extra_by)
        print(f"Using extra BY constraints: {extra_by}")

    print(
        f"Building ABX task (ON: accent_pattern, BY: {by_conditions}, ACROSS: {across_conditions})..."
    )

    subsampler = None
    if max_size_group is not None or max_x_across is not None:
        subsampler = Subsampler(max_size_group, max_x_across, seed=subsample_seed)
        if subsampler.description(with_across=True):
            print(f"Subsampling: {subsampler.description(with_across=True)}")

    task = Task(
        dataset,
        on="accent_pattern",
        by=by_conditions,
        across=across_conditions,
        subsampler=subsampler,
    )

    print(f"Number of cells: {len(task)}")

    if len(task) == 0:
        raise ValueError(
            "No valid ABX cells could be formed. Check your data constraints."
        )

    # Score
    print(f"Computing ABX scores with {distance} distance...")
    constraints = None
    if ax_match:
        constraints = [pl.col(f"{c}_a") == pl.col(f"{c}_x") for c in ax_match]
        print(f"Using A/X match constraints: {ax_match}")

    score = Score(task, distance, constraints=constraints)

    # Collapse: average over speakers first (giving equal weight per phone_sequence + contrast),
    # then final mean over all (phone_sequence, contrast) combinations.
    collapse_levels = ["speaker"]
    if extra_by:
        collapse_levels = [tuple(extra_by)] + collapse_levels

    return score.collapse(levels=collapse_levels), score


def main():
    parser = argparse.ArgumentParser(
        description="Run ABX discrimination test for pitch accent patterns.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "dataset_path",
        type=Path,
        help="Path to dataset directory (containing items.csv)",
    )
    parser.add_argument("features", type=Path, help="Root path for feature directory")
    parser.add_argument(
        "--model", type=str, required=True, help="Model name (e.g., HUBERT_BASE)"
    )

    parser.add_argument(
        "--distance",
        type=str,
        default="angular",
        choices=["angular", "cosine", "euclidean", "kl", "kl_symmetric", "identical"],
        help="Distance metric",
    )
    parser.add_argument(
        "--extra-by",
        nargs="+",
        default=[],
        help="Additional label columns to include in BY constraints",
    )
    parser.add_argument(
        "--ax-match",
        nargs="+",
        default=[],
        help="Label columns that must match between A and X",
    )
    parser.add_argument(
        "--max-size-group",
        type=int,
        default=None,
        help="Max items for A, B, or X in each cell",
    )
    parser.add_argument(
        "--max-x-across",
        type=int,
        default=None,
        help="Max X values per (A,B) in across mode",
    )
    parser.add_argument(
        "--subsample-seed", type=int, default=0, help="Random seed for subsampling"
    )

    args = parser.parse_args()

    # Resolve paths
    item_path = args.dataset_path / "items.csv"
    if not item_path.exists():
        raise FileNotFoundError(f"Item file not found: {item_path}")

    dataset_name = args.dataset_path.name
    features_dir_dataset_name = (
        dataset_name if not dataset_name.startswith("csj_") else "csj"
    )
    model_dir = args.features / features_dir_dataset_name / args.model.lower()

    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    # Find feature files
    def is_layer_dir(name: str) -> bool:
        return len(name) > 1 and name[0] == "l" and name[1:].isdigit()

    layer_dirs = [p for p in model_dir.iterdir() if p.is_dir() and is_layer_dir(p.name)]

    if layer_dirs:
        layer_dirs = sorted(layer_dirs, key=lambda p: int(p.name[1:]))
        feature_sets = [(p.name, p / "features.pt") for p in layer_dirs]
    else:
        feature_file = model_dir / "features.pt"
        if not feature_file.exists():
            raise FileNotFoundError(f"Feature file not found: {feature_file}")
        feature_sets = [("features", feature_file)]

    # Prepare output
    results_dir = Path("results") / dataset_name
    output_csv = results_dir / f"{args.model.lower()}.csv"

    if output_csv.exists():
        raise FileExistsError(f"Results already exist: {output_csv}")

    results = []

    for layer_name, feature_file in feature_sets:
        if len(feature_sets) > 1:
            print(f"\n=== {layer_name} ===")

        dataset = build_dataset(item_path, feature_file)
        maybe_normalize(dataset, args.distance)

        error_rate, score = compute_abx(
            dataset,
            distance=args.distance,
            extra_by=args.extra_by,
            ax_match=args.ax_match,
            max_size_group=args.max_size_group,
            max_x_across=args.max_x_across,
            subsample_seed=args.subsample_seed,
        )

        # Save detailed cell scores
        detail_csv = results_dir / "cells" / args.model.lower() / f"{layer_name}.csv"
        detail_csv.parent.mkdir(parents=True, exist_ok=True)
        score.write_csv(detail_csv)

        print(f"ABX Error Rate: {error_rate:.4f} ({error_rate * 100:.2f}%)")

        results.append(
            {
                "layer": layer_name,
                "mode": "across",
                "error_rate": f"{error_rate:.6f}",
            }
        )

    # Write summary results
    results_dir.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["layer", "mode", "error_rate"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nWrote results to {output_csv}")


if __name__ == "__main__":
    main()
