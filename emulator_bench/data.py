from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .utils import DEFAULT_SPLIT_GROUPS

SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Split:
    group: str
    name: str  # "all" for a flat group, else the threshold_* folder
    path: Path

    @property
    def id(self) -> str:
        return f"{self.group}/{self.name}"


def discover_splits(dataset_root: str | Path, groups: tuple[str, ...] = DEFAULT_SPLIT_GROUPS) -> list[Split]:
    """The reaction-only split folders of the given groups. A group is flat or holds threshold_* folders."""
    root = Path(dataset_root)
    found = []
    for group in groups:
        folders = sorted((root / group).glob("threshold_*")) or [root / group]
        for folder in folders:
            if all((folder / f"{s}.parquet").exists() for s in SPLITS):
                found.append(Split(group, folder.name if folder.name.startswith("threshold_") else "all", folder))
    if not found:
        raise FileNotFoundError(f"No split folders for {groups} under {root}")
    return found


def read_rows(path: Path) -> tuple[pd.DataFrame, dict]:
    """Rows with a reaction and an EC. Rows without them are dropped and counted."""
    frame = pd.read_parquet(path, columns=["rxn_smiles", "ec_number"])
    raw = len(frame)
    frame = frame.dropna().astype(str)
    frame["rxn_smiles"] = frame["rxn_smiles"].str.strip()
    frame["ec_number"] = frame["ec_number"].str.strip()
    frame = frame[frame["rxn_smiles"].str.contains(">>") & (frame["ec_number"] != "")]
    return frame.reset_index(drop=True), {"raw_rows": raw, "dropped_rows": raw - len(frame)}
