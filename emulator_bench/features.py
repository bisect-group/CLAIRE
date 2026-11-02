"""CLAIRE input: rxnfp (256) followed by DRFP (256), computed once per unique reaction string and cached.

rxnfp follows data/embedding/embedding.py (default model, CLS embedding). DRFP follows the `drfp` command
of that script with `-d 256`: the command's defaults are radius 3, rings, and no rooted central atoms.
"""

from __future__ import annotations

import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from .augment import canonical
from .utils import atomic_write, write_json

FEATURES_VERSION = 1
BATCH_SIZE = 128
DIM = 256


def cache_dir(out_root: str | Path) -> Path:
    return Path(out_root) / "embeddings" / f"rxnfp_drfp_v{FEATURES_VERSION}"


def fingerprint() -> dict:
    """Everything the cached values depend on."""
    import importlib.metadata as md

    return {
        "features_version": FEATURES_VERSION,
        "kind": "rxnfp_bert_ft_cls_256+drfp_256_r3_rings_unrooted",
        "rxnfp": md.version("rxnfp"),
        "transformers": md.version("transformers"),
        "drfp": md.version("drfp"),
        "rdkit": md.version("rdkit"),
        "dtype": "float32",
    }


def rxnfp(reactions: list[str], device: str | None = None) -> np.ndarray:
    import torch
    from rxnfp.transformer_fingerprints import (
        RXNBERTFingerprintGenerator,
        get_default_model_and_tokenizer,
    )

    model, tokenizer = get_default_model_and_tokenizer(force_no_cuda=device == "cpu")
    generator = RXNBERTFingerprintGenerator(model, tokenizer, force_no_cuda=device == "cpu")
    order = np.argsort([len(r) for r in reactions], kind="stable")
    out = np.empty((len(reactions), DIM), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(order), BATCH_SIZE):
            rows = order[start : start + BATCH_SIZE]
            out[rows] = np.asarray(generator.convert_batch([reactions[i] for i in rows]), dtype=np.float32)
    return out


def _drfp(chunk: list[str]) -> np.ndarray:
    from drfp import DrfpEncoder

    return np.asarray(DrfpEncoder.encode(chunk, n_folded_length=DIM, root_central_atom=False), dtype=np.float32)


def drfp(reactions: list[str], workers: int = 8) -> np.ndarray:
    """One encoding per sorted-component form of a reaction, shared by its orderings."""
    keys = [canonical(r) for r in reactions]
    unique = sorted(set(keys))
    chunks = [unique[i : i + 2000] for i in range(0, len(unique), 2000)]
    with Pool(workers) as pool:
        encoded = np.concatenate(pool.map(_drfp, chunks))
    index = pd.Series(np.arange(len(unique)), index=unique)
    return encoded[index.loc[keys].to_numpy()]


def compute(reactions: list[str], device: str | None = None) -> np.ndarray:
    return np.concatenate([rxnfp(reactions, device), drfp(reactions)], axis=1)


def _save_npy(path: str, array: np.ndarray) -> None:
    with open(path, "wb") as handle:
        np.save(handle, array)


def build(out_root: str | Path, reactions: list[str], device: str | None = None) -> Path:
    """Add the missing reactions to the cache. Run it once, before the runs."""
    folder = cache_dir(out_root)
    meta_path = folder / "meta.json"
    known, matrix = [], np.empty((0, 2 * DIM), dtype=np.float32)
    if meta_path.exists():
        if json.loads(meta_path.read_text())["fingerprint"] != fingerprint():
            raise ValueError(f"{folder} was built with other settings; raise FEATURES_VERSION")
        known = pd.read_parquet(folder / "reactions.parquet")["rxn_smiles"].tolist()
        matrix = np.load(folder / "features.npy")
    missing = sorted(set(reactions) - set(known))
    if missing:
        matrix = np.concatenate([matrix, compute(missing, device)])
        known = known + missing
        atomic_write(folder / "reactions.parquet", lambda t: pd.DataFrame({"rxn_smiles": known}).to_parquet(t, index=False))
        atomic_write(folder / "features.npy", lambda t: _save_npy(t, matrix))
        write_json(meta_path, {"fingerprint": fingerprint(), "rows": len(known)})
    return folder


def load(out_root: str | Path, reactions: list[str]) -> np.ndarray:
    """Features of the reactions, one row each (the same reaction may repeat)."""
    folder = cache_dir(out_root)
    known = pd.read_parquet(folder / "reactions.parquet")["rxn_smiles"]
    index = pd.Series(np.arange(len(known)), index=known)
    return np.load(folder / "features.npy", mmap_mode="r")[index.loc[reactions].to_numpy()]
