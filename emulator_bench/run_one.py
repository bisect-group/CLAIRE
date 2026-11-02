"""Train CLAIRE's EC model on one reaction-only split and seed, then rank the ECs of the test reactions."""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch

from . import features, fit
from .augment import augment
from .data import discover_splits, read_rows
from .results import ec_level_accuracy
from .utils import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_OUT_ROOT,
    DEFAULT_SEEDS,
    DEFAULT_SPLIT_GROUPS,
    atomic_write,
    load_hparams,
    seed_dir,
    write_json,
)


def run(split, seed: int, args: argparse.Namespace) -> dict:
    out = seed_dir(args.out_root, split.group, split.name, seed)
    if (out / "metrics.json").exists() and not args.overwrite:
        return {}
    out.mkdir(parents=True, exist_ok=True)
    hparams = load_hparams()
    epochs = args.epochs or hparams["epochs"]
    device = torch.device(args.device)
    frames, stats = {}, {}
    for name in ("train", "test"):
        frames[name], stats[name] = read_rows(split.path / f"{name}.parquet")
    if args.sample_rows:
        frames = {name: frame.head(args.sample_rows) for name, frame in frames.items()}
    fit_rows = frames["train"]
    test = frames["test"]
    queries = test["rxn_smiles"].drop_duplicates().tolist()
    pairs = [(a, ec) for r, ec in zip(fit_rows["rxn_smiles"], fit_rows["ec_number"]) for a in augment(r)]
    x = torch.from_numpy(features.load(args.out_root, [a for a, _ in pairs])).to(device)
    groups = fit.Groups([ec for _, ec in pairs])
    started = time.time()
    model, losses = fit.train(
        x, groups, hparams, seed, epochs, run_dir=out, amp=args.amp,
        checkpoint_every=args.checkpoint_every, log_every=args.log_every,
    )  # fmt: skip
    torch.save(model.state_dict(), out / "lastmodel.pth")
    q = torch.from_numpy(features.load(args.out_root, queries)).to(device)
    ec_index, ec_dist = fit.rank(model, x, groups, q, hparams["top_k"])
    ranked = {
        r: ([groups.ec_names[j] for j in idx], (-d).astype(float).tolist())
        for r, idx, d in zip(queries, ec_index, ec_dist)
    }
    predictions = pd.DataFrame(
        {
            "query": test["rxn_smiles"],
            "ec_number": test["ec_number"],
            "top_ec": test["rxn_smiles"].map(lambda r: ranked[r][0]),
            "top_score": test["rxn_smiles"].map(lambda r: ranked[r][1]),
        }
    )
    atomic_write(out / "test_predictions.parquet", lambda tmp: predictions.to_parquet(tmp, index=False))
    accuracy = ec_level_accuracy(predictions, drop_preliminary=False)
    accuracy_no_b = ec_level_accuracy(predictions, drop_preliminary=True)
    write_json(out / "ec_level_accuracy.json", accuracy)
    write_json(out / "ec_level_accuracy_no_b.json", accuracy_no_b)
    pd.DataFrame({"epoch": np.arange(1, len(losses) + 1), "train_loss": losses}).to_csv(out / "logfile.csv", index=False)
    metrics = {
        "split": split.id, "seed": seed, "epochs": epochs, "n_ecs": len(groups.ec_names),
        "n_fit_rows": len(fit_rows), "n_augmented_rows": len(pairs), "rows": stats, "precision": "bf16" if args.amp else "fp32",
        "elapsed_seconds": time.time() - started, "final_loss": losses[-1],
        "ec_level_accuracy": accuracy, "ec_level_accuracy_no_b": accuracy_no_b, "hparams": hparams,
    }  # fmt: skip
    write_json(out / "metrics.json", metrics)
    print(f"[run_one] {split.id} seed {seed}: {accuracy_no_b['k=1']}", flush=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--split_groups", nargs="+", default=list(DEFAULT_SPLIT_GROUPS))
    parser.add_argument("--split_name", default=None, help="Only this split name, e.g. threshold_0.425 or all")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true", help="bf16 autocast on CUDA. The original trains in fp32.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--epochs", type=int, default=None, help="Test-only override of the epoch count")
    parser.add_argument("--sample_rows", type=int, default=0, help="Test-only: first rows of each split")
    parser.add_argument("--checkpoint_every", type=int, default=100)
    parser.add_argument("--log_every", type=int, default=10)
    args = parser.parse_args()
    for split in discover_splits(args.dataset_root, tuple(args.split_groups)):
        if args.split_name and split.name != args.split_name:
            continue
        for seed in args.seeds:
            run(split, seed, args)


if __name__ == "__main__":
    main()
