import argparse
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
from rich.progress import track

from emulator_bench.common import (
    DEFAULT_BASE_DIR,
    DEFAULT_SOURCE_DATA_DIR,
    REQUIRED_COLUMNS,
    configure_visible_gpus,
    console,
    copy_dataset_splits,
    discover_split_jobs,
    read_table,
    require_columns,
)
from emulator_bench.ec import ec_label, is_unlabeled_ec


def validate_split(path: Path, allow_provisional: bool = False) -> pd.DataFrame:
    df = read_table(path)
    require_columns(df, REQUIRED_COLUMNS, path)
    bad = []
    unlabeled = 0
    for idx, value in enumerate(df["ec_number"].astype(str).tolist()):
        if is_unlabeled_ec(value):
            unlabeled += 1
            continue
        try:
            ec_label(value, "l1", allow_provisional=allow_provisional)
        except ValueError as exc:
            bad.append((idx, value, str(exc)))
            if len(bad) >= 5:
                break
    if bad:
        raise ValueError(f"Invalid EC labels in {path}: {bad}")
    if unlabeled:
        console.print(f"{path}: {unlabeled} unlabeled EC rows will be excluded from training/evaluation")
    return df


def validate_jobs(base_dir: Path = DEFAULT_BASE_DIR, allow_provisional: bool = False) -> None:
    jobs = discover_split_jobs(base_dir)
    if not jobs:
        raise FileNotFoundError(f"No train/val/test split jobs found under {base_dir}")
    for job in track(jobs, description="Validating splits", console=console):
        for path in (job.train_path, job.val_path, job.test_path):
            validate_split(path, allow_provisional=allow_provisional)
    console.print(f"Validated {len(jobs)} split jobs under {base_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", default=str(DEFAULT_SOURCE_DATA_DIR))
    parser.add_argument("--base_dir", default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow_provisional", action="store_true")
    parser.add_argument("--gpus")
    args = parser.parse_args()
    configure_visible_gpus(args.gpus)

    copied = copy_dataset_splits(Path(args.source_dir), Path(args.base_dir), overwrite=args.overwrite)
    console.print(f"Copied {len(copied)} files into {args.base_dir}")
    validate_jobs(Path(args.base_dir), allow_provisional=args.allow_provisional)


if __name__ == "__main__":
    main()
