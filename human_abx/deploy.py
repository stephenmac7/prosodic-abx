#!/usr/bin/env python3
"""Deploy human ABX experiment to web server.

Usage:
    python deploy.py --target ~/public_html/human_abx
    python deploy.py --target ~/public_html/human_abx --clean  # Remove existing first
"""

import argparse
import shutil
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Deploy human ABX experiment to web server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--target", type=Path, required=True,
        help="Target directory for deployment (e.g., ~/public_html/human_abx)"
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="Remove existing deployment before copying (deletes all data!)"
    )
    parser.add_argument(
        "--source", type=Path, default=Path(__file__).parent / "web",
        help="Source web directory"
    )

    args = parser.parse_args()
    target = args.target.expanduser()
    source = args.source

    if not source.exists():
        raise FileNotFoundError(f"Source directory not found: {source}")

    # Check for required subdirectories
    required = ["lists", "audio"]
    missing = [d for d in required if not (source / d).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required directories in {source}: {missing}\n"
            "Run generate_human_abx.py --materialize-audio first."
        )

    if args.clean:
        if target.exists():
            print(f"Removing existing deployment: {target}")
            shutil.rmtree(target)

    # Create target directory
    target.mkdir(parents=True, exist_ok=True)

    # Use rsync to deploy (trailing slash on source copies contents)
    print(f"Deploying {source} -> {target}")
    result = subprocess.run(
        [
            "rsync", "-a", "--delete",
            "--exclude", "data/",  # Preserve existing response data
            f"{source}/",
            f"{target}/",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"rsync failed with exit code {result.returncode}")

    # Create data directory if it doesn't exist
    data_dir = target / "data"
    data_dir.mkdir(exist_ok=True)

    # Summary
    print("\nDeployment complete!")
    print(f"  Target: {target}")

    # Count lists
    lists_dir = target / "lists"
    for dataset_dir in sorted(lists_dir.iterdir()):
        if dataset_dir.is_dir():
            n_lists = len(list(dataset_dir.glob("participant_*.csv")))
            print(f"  {dataset_dir.name}: {n_lists} lists")


if __name__ == "__main__":
    main()
