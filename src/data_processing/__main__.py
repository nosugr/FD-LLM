"""Run with: python -m src.data_processing."""

import argparse
from pathlib import Path

from .pipeline import DEFAULT_CONFIG, build_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess the 19-class CWRU dataset")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    metadata = build_dataset(args.config)
    for source in metadata["sources"]:
        print(f"{source['label']:2d} {source['source_file']:35s} "
              f"{source['signal_field']:15s} points={source['signal_length']:6d} "
              f"windows={source['available_windows']:3d} gap={source['unused_gap_windows']:3d}")
    print(f"train={metadata['train_count']} test={metadata['test_count']} "
          f"classes={metadata['class_count']}")


if __name__ == "__main__":
    main()
