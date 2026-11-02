from __future__ import annotations

import functools
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Callable

BASELINE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BASELINE_ROOT.parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "processed" / "datasets" / "enzyme_retrieval_dataset"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "outputs" / "baselines" / "CLAIRE"
TASK = "enzyme_retrieval"
DEFAULT_SEEDS = (666, 777, 888)
DEFAULT_SPLIT_GROUPS = (
    "random_splits_grouped_rxn_key",
    "uniprot_time_splits",
    "reaction_drfp_tanimoto_splits",
)
HPARAMS_PATH = Path(__file__).with_name("default_hparams.json")
FULL_EC = re.compile(r"\d+\.\d+\.\d+\.\d+")


def load_hparams() -> dict:
    return json.loads(HPARAMS_PATH.read_text())["hparams"]


@functools.cache
def original_modules():
    """The original dev code: the training script (model, dataset) and dev.utils.utils."""
    import importlib.util

    if str(BASELINE_ROOT) not in sys.path:
        sys.path.insert(0, str(BASELINE_ROOT))
    spec = importlib.util.spec_from_file_location("claire_train", BASELINE_ROOT / "dev" / "training" / "train-pred_rxn_EC.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def atomic_write(path: str | Path, write: Callable[[str], None]) -> None:
    """Call write(tmp_path) with a temp file in the target folder, then rename it onto path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    os.close(fd)
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def write_json(path: str | Path, data: object) -> None:
    text = json.dumps(data, indent=2, sort_keys=True) + "\n"
    atomic_write(path, lambda tmp: Path(tmp).write_text(text))


def seed_dir(out_root: str | Path, group: str, name: str, seed: int) -> Path:
    return Path(out_root) / TASK / group / name / f"seed_{seed}"
