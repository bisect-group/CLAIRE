import csv
import hashlib
import json
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from rich.console import Console
from rich.table import Table


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DATA_DIR = Path("/home/adhil/github/EMULaToR/data/processed/datasets/enzyme_retrieval_dataset")
DEFAULT_BASE_DIR = Path("/home/adhil/github/EMULaToR/data/processed/baselines/CLAIRE")
DEFAULT_EMBEDDINGS_DIR = DEFAULT_BASE_DIR / "embeddings"
DEFAULT_RESULTS_DIRNAME = "claire_results"
DEFAULT_LEVELS = ["l1", "l2", "l3", "l4"]
DEFAULT_DEVICE = "cuda:0"
SPLIT_FILENAMES = ("train", "val", "test")
REQUIRED_COLUMNS = ("rxn_smiles", "ec_number", "uniprot_date")

console = Console()


@dataclass(frozen=True)
class SplitJob:
    split_group: str
    split_name: str
    root_dir: Path
    train_path: Path
    val_path: Path
    test_path: Path

    def as_dict(self) -> Dict[str, str]:
        return {
            "split_group": self.split_group,
            "split_name": self.split_name,
            "root_dir": str(self.root_dir),
            "train_path": str(self.train_path),
            "val_path": str(self.val_path),
            "test_path": str(self.test_path),
        }


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_rxn_smiles(value: str) -> str:
    return str(value).strip()


def feature_cache_path(embeddings_dir: Path, rxn_smiles: str) -> Path:
    key = stable_hash(normalize_rxn_smiles(rxn_smiles))
    return Path(embeddings_dir) / "reactions" / "rxnfp_drfp_512" / key[:2] / f"{key}.npz"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def atomic_save_npz(path: Path, **payload) -> None:
    ensure_parent(path)
    tmp_path = Path(str(path) + f".tmp.{os.getpid()}")
    np.savez_compressed(tmp_path, **payload)
    generated = Path(str(tmp_path) + ".npz")
    if generated.exists():
        generated.replace(path)
    else:
        tmp_path.replace(path)


def atomic_write_text(path: Path, text: str) -> None:
    ensure_parent(path)
    tmp_path = Path(str(path) + f".tmp.{os.getpid()}")
    tmp_path.write_text(text)
    tmp_path.replace(path)


def save_json(path: Path, payload: Dict) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True))


def load_json(path: Path) -> Dict:
    return json.loads(Path(path).read_text())


def append_csv_row(path: Path, row: Dict) -> None:
    ensure_parent(path)
    exists = path.exists()
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_torch() -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


def amp_dtype_for_device(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    major, _minor = torch.cuda.get_device_capability(device)
    return torch.bfloat16 if major >= 8 else torch.float16


def resolve_device(device_name: str = DEFAULT_DEVICE) -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    requested = torch.device(device_name)
    if requested.type != "cuda":
        return requested
    count = torch.cuda.device_count()
    index = requested.index if requested.index is not None else 0
    if index is not None and index < count:
        return requested
    if count == 1:
        console.print(f"Requested {device_name}, but only one CUDA device is visible; using cuda:0")
        return torch.device("cuda:0")
    raise ValueError(f"Requested {device_name}, but only {count} CUDA devices are visible")


def configure_visible_gpus(gpus: Optional[str]) -> None:
    if gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus)
        console.print(f"Using physical GPU(s) {gpus}; in-process CUDA devices start at cuda:0")


def subprocess_env(gpus: Optional[str] = None) -> Dict[str, str]:
    env = os.environ.copy()
    if gpus:
        env["CUDA_VISIBLE_DEVICES"] = str(gpus)
    return env


def read_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}")


def write_table(df: pd.DataFrame, path: Path) -> None:
    ensure_parent(path)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
        return
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
        return
    raise ValueError(f"Unsupported table format: {path}")


def require_columns(df: pd.DataFrame, required: Iterable[str], path: Path) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing} in {path}")


def split_file(directory: Path, stem: str) -> Optional[Path]:
    for suffix in (".parquet", ".csv"):
        candidate = Path(directory) / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def copy_dataset_splits(
    source_dir: Path = DEFAULT_SOURCE_DATA_DIR,
    base_dir: Path = DEFAULT_BASE_DIR,
    overwrite: bool = False,
) -> List[Path]:
    source_dir = Path(source_dir)
    base_dir = Path(base_dir)
    copied: List[Path] = []
    for src in source_dir.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(source_dir)
        dest = base_dir / rel
        if dest.exists() and not overwrite and dest.stat().st_size == src.stat().st_size:
            continue
        ensure_parent(dest)
        shutil.copy2(src, dest)
        copied.append(dest)
    return copied


def discover_split_jobs(base_dir: Path = DEFAULT_BASE_DIR, split_groups: Optional[Sequence[str]] = None) -> List[SplitJob]:
    base_dir = Path(base_dir)
    jobs: List[SplitJob] = []
    candidates: List[Path] = []
    for child in sorted(base_dir.iterdir()) if base_dir.exists() else []:
        if not child.is_dir() or child.name in {"embeddings", "optuna_studies", "bench_summaries"}:
            continue
        if split_groups and child.name not in split_groups:
            continue
        candidates.append(child)
        for grandchild in sorted(child.iterdir()):
            if grandchild.is_dir():
                candidates.append(grandchild)

    for directory in candidates:
        train = split_file(directory, "train")
        val = split_file(directory, "val")
        test = split_file(directory, "test")
        if not (train and val and test):
            continue
        rel = directory.relative_to(base_dir)
        group = rel.parts[0]
        split_name = rel.parts[-1] if len(rel.parts) > 1 else group
        jobs.append(SplitJob(group, split_name, directory, train, val, test))
    return jobs


def render_jobs_table(jobs: Sequence[SplitJob], title: str = "CLAIRE split jobs") -> None:
    table = Table(title=title)
    table.add_column("group")
    table.add_column("split")
    table.add_column("root")
    for job in jobs:
        table.add_row(job.split_group, job.split_name, str(job.root_dir))
    console.print(table)


def completed_run(out_dir: Path) -> bool:
    out_dir = Path(out_dir)
    return (out_dir / "metrics.json").exists() and (out_dir / "predictions.csv").exists()
