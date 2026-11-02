"""Compute and cache the features of the augmented train reactions and the test reactions of the selected splits."""

from __future__ import annotations

import argparse

from . import features
from .augment import augment
from .data import discover_splits, read_rows
from .utils import DEFAULT_DATASET_ROOT, DEFAULT_OUT_ROOT, DEFAULT_SPLIT_GROUPS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--split_groups", nargs="+", default=list(DEFAULT_SPLIT_GROUPS))
    parser.add_argument("--device", default=None, help="cpu or cuda; default is cuda when available")
    args = parser.parse_args()
    reactions = set()
    for split in discover_splits(args.dataset_root, tuple(args.split_groups)):
        reactions |= {a for r in read_rows(split.path / "train.parquet")[0]["rxn_smiles"].unique() for a in augment(r)}
        reactions |= set(read_rows(split.path / "test.parquet")[0]["rxn_smiles"])
    folder = features.build(args.out_root, sorted(reactions), args.device)
    print(f"[build_features] {len(reactions)} reactions in {folder}", flush=True)


if __name__ == "__main__":
    main()
