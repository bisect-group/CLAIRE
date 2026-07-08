import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from emulator_bench.common import (
    DEFAULT_BASE_DIR,
    DEFAULT_DEVICE,
    DEFAULT_EMBEDDINGS_DIR,
    atomic_save_npz,
    console,
    discover_split_jobs,
    feature_cache_path,
    normalize_rxn_smiles,
    read_table,
    configure_visible_gpus,
    resolve_device,
)


def _load_rxnfp_generator():
    from rxnfp.transformer_fingerprints import RXNBERTFingerprintGenerator, get_default_model_and_tokenizer

    model, tokenizer = get_default_model_and_tokenizer()
    return RXNBERTFingerprintGenerator(model, tokenizer)


def _encode_drfp(smiles: Sequence[str], dim: int = 256) -> np.ndarray:
    from drfp import DrfpEncoder

    encoded = DrfpEncoder.encode(list(smiles), n_folded_length=dim, show_progress_bar=False)
    return np.asarray(encoded, dtype=np.float32)


def _encode_rxnfp(generator, smiles: Sequence[str]) -> np.ndarray:
    encoded = generator.convert_batch(list(smiles))
    return np.asarray(encoded, dtype=np.float32)


def _save_feature_batch(embeddings_dir: Path, batch: Sequence[str], rxnfp: np.ndarray, drfp: np.ndarray) -> int:
    if rxnfp.shape[1] != 256 or drfp.shape[1] != 256:
        raise ValueError(f"Expected 256-d rxnfp and DRFP, got {rxnfp.shape} and {drfp.shape}")
    features = np.concatenate([rxnfp, drfp], axis=1).astype(np.float32)
    for smiles, feature, rxnfp_item, drfp_item in zip(batch, features, rxnfp, drfp):
        atomic_save_npz(
            feature_cache_path(embeddings_dir, smiles),
            feature=feature,
            rxnfp=rxnfp_item.astype(np.float32),
            drfp=drfp_item.astype(np.float32),
            rxn_smiles=np.asarray(smiles),
        )
    return len(batch)


def collect_reactions(base_dir: Path, split_groups: Sequence[str] | None = None) -> List[str]:
    jobs = discover_split_jobs(base_dir, split_groups=split_groups)
    if not jobs:
        raise FileNotFoundError(f"No split jobs found under {base_dir}")
    seen = set()
    ordered: List[str] = []
    for job in jobs:
        for path in (job.train_path, job.val_path, job.test_path):
            df = read_table(path)
            for value in df["rxn_smiles"].astype(str):
                smiles = normalize_rxn_smiles(value)
                if smiles and smiles not in seen:
                    seen.add(smiles)
                    ordered.append(smiles)
    return ordered


def cache_reactions(
    reactions: Sequence[str],
    embeddings_dir: Path = DEFAULT_EMBEDDINGS_DIR,
    batch_size: int = 128,
    rxnfp_batch_size: int = 128,
    device_name: str = DEFAULT_DEVICE,
    drfp_workers: int = 1,
    save_workers: int = 2,
    overwrite: bool = False,
) -> Dict[str, int]:
    embeddings_dir = Path(embeddings_dir)
    pending = [rxn for rxn in reactions if overwrite or not feature_cache_path(embeddings_dir, rxn).exists()]
    stats = {"total": len(reactions), "cached": len(reactions) - len(pending), "computed": 0}
    if not pending:
        console.print(f"All {len(reactions)} reaction features are already cached")
        return stats
    work_batch_size = max(1, min(int(batch_size), int(rxnfp_batch_size)))

    device = resolve_device(device_name)
    if device.type == "cuda":
        import torch

        torch.cuda.set_device(device)
    generator = _load_rxnfp_generator()
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )
    with progress:
        task = progress.add_task("Caching reaction features", total=len(pending))
        batches = [list(pending[start : start + work_batch_size]) for start in range(0, len(pending), work_batch_size)]
        save_futures: List[Future] = []
        drfp_workers = max(1, int(drfp_workers))
        save_workers = max(1, int(save_workers))
        with ThreadPoolExecutor(max_workers=drfp_workers) as drfp_pool, ThreadPoolExecutor(max_workers=save_workers) as save_pool:
            drfp_futures: Dict[int, Future] = {0: drfp_pool.submit(_encode_drfp, batches[0], 256)}
            for index, batch in enumerate(batches):
                next_index = index + 1
                if next_index < len(batches):
                    drfp_futures[next_index] = drfp_pool.submit(_encode_drfp, batches[next_index], 256)

                rxnfp = _encode_rxnfp(generator, batch)
                drfp = drfp_futures.pop(index).result()
                save_futures.append(save_pool.submit(_save_feature_batch, embeddings_dir, batch, rxnfp, drfp))

                while len(save_futures) >= save_workers * 2:
                    completed = save_futures.pop(0).result()
                    stats["computed"] += completed
                    progress.advance(task, completed)

            for future in save_futures:
                completed = future.result()
                stats["computed"] += completed
                progress.advance(task, completed)
    console.print(f"Cached {stats['computed']} reactions; reused {stats['cached']}")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--embeddings_dir", default=str(DEFAULT_EMBEDDINGS_DIR))
    parser.add_argument("--split_groups", nargs="*")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--rxnfp_batch_size", type=int, default=128)
    parser.add_argument("--drfp_workers", type=int, default=1)
    parser.add_argument("--save_workers", type=int, default=2)
    parser.add_argument("--gpus")
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    configure_visible_gpus(args.gpus)

    reactions = collect_reactions(Path(args.base_dir), split_groups=args.split_groups)
    console.print(f"Discovered {len(reactions)} unique reactions")
    cache_reactions(
        reactions,
        Path(args.embeddings_dir),
        batch_size=args.batch_size,
        rxnfp_batch_size=args.rxnfp_batch_size,
        device_name=args.device,
        drfp_workers=args.drfp_workers,
        save_workers=args.save_workers,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
