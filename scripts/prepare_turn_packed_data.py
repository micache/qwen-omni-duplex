#!/usr/bin/env python3
"""Download/validate only the selected TASTE layout and MUSAN noise subset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from huggingface_hub import snapshot_download

from duplex.turn_packed import DATASET_ID, NOISE_DATASET_ID, load_local_taste


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--taste-root", type=Path, default=Path("data/TASTE-IF-SFT-48K"))
    parser.add_argument("--noise-root", type=Path, default=Path("data/musan"))
    parser.add_argument("--download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.download:
        snapshot_download(
            repo_id=DATASET_ID,
            repo_type="dataset",
            revision="main",
            local_dir=args.taste_root,
            cache_dir=args.taste_root / ".hub_cache",
            allow_patterns=("README.md", "data/*.parquet"),
        )
        snapshot_download(
            repo_id=NOISE_DATASET_ID,
            repo_type="dataset",
            revision="main",
            local_dir=args.noise_root,
            cache_dir=args.noise_root / ".hub_cache",
            allow_patterns=("README.md", "noise/*.parquet", "licenses/noise/*"),
        )
    train = load_local_taste(args.taste_root, split="train")
    dev = load_local_taste(args.taste_root, split="dev")
    noise_files = sorted((args.noise_root / "noise").glob("train-*.parquet"))
    if not noise_files:
        raise FileNotFoundError(f"No MUSAN noise shards under {args.noise_root / 'noise'}.")
    print(f"TASTE conversations: train={len(train)} dev={len(dev)}")
    print(f"MUSAN noise shards: {len(noise_files)}")


if __name__ == "__main__":
    main()
