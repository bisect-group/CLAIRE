import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from torch.utils.data import DataLoader

from emulator_bench.common import (
    DEFAULT_DEVICE,
    DEFAULT_EMBEDDINGS_DIR,
    amp_dtype_for_device,
    configure_torch,
    configure_visible_gpus,
    console,
    feature_cache_path,
    read_table,
    resolve_device,
    save_json,
    set_seed,
)
from emulator_bench.ec import build_level_vocab, ec_label, is_unlabeled_ec
from emulator_bench.evaluate import evaluate_checkpoint, load_feature_matrix
from emulator_bench.modeling import LayerNormNet, TripletFeatureDataset


def labels_for_level(df: pd.DataFrame, level: str, allow_provisional: bool = False) -> List[str]:
    labels = []
    for value in df["ec_number"].astype(str):
        labels.append(ec_label(value, level, allow_provisional=allow_provisional))
    return labels


def drop_unusable_labels(df: pd.DataFrame, level: str, allow_provisional: bool = False) -> pd.DataFrame:
    keep = []
    for value in df["ec_number"].astype(str):
        if is_unlabeled_ec(value):
            keep.append(False)
            continue
        try:
            ec_label(value, level, allow_provisional=allow_provisional)
            keep.append(True)
        except ValueError:
            keep.append(False)
    return df.loc[keep].reset_index(drop=True)


def filter_to_train_vocab(df: pd.DataFrame, level: str, vocab: set[str], allow_provisional: bool = False) -> pd.DataFrame:
    keep = []
    for value in df["ec_number"].astype(str):
        try:
            keep.append(ec_label(value, level, allow_provisional=allow_provisional) in vocab)
        except ValueError:
            keep.append(False)
    return df.loc[keep].reset_index(drop=True)


def save_checkpoint(path: Path, model, optimizer, epoch: int, best_loss: float, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_loss": best_loss,
            "args": vars(args),
        },
        tmp_path,
    )
    tmp_path.replace(path)


def quarantine_checkpoint(path: Path, error: Exception) -> None:
    corrupt_path = path.with_name(f"{path.name}.corrupt.{int(time.time())}")
    try:
        path.replace(corrupt_path)
        console.print(f"[yellow]Ignoring corrupt checkpoint {path}; moved to {corrupt_path}: {error}[/yellow]")
    except OSError as move_error:
        console.print(f"[yellow]Ignoring corrupt checkpoint {path}; failed to move it: {move_error}; original error: {error}[/yellow]")


def should_log_epoch(epoch: int, start_epoch: int, end_epoch: int, interval: int) -> bool:
    return epoch == start_epoch or epoch == end_epoch or epoch % max(1, interval) == 0


def emit_progress(payload: Dict) -> None:
    print(json.dumps({"claire_progress": payload}, sort_keys=True), flush=True)


def train_one(args: argparse.Namespace) -> Dict:
    set_seed(args.seed)
    configure_torch()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df = read_table(Path(args.train_path))
    val_df = read_table(Path(args.val_path))
    test_df = read_table(Path(args.test_path))

    train_df = drop_unusable_labels(train_df, args.level, allow_provisional=args.allow_provisional)
    val_df = drop_unusable_labels(val_df, args.level, allow_provisional=args.allow_provisional)
    test_df = drop_unusable_labels(test_df, args.level, allow_provisional=args.allow_provisional)
    train_vocab = build_level_vocab(train_df["ec_number"].astype(str), args.level, allow_provisional=args.allow_provisional)
    if len(train_vocab) < 2:
        raise ValueError(f"Need at least two train labels for {args.level}; got {len(train_vocab)}")
    train_labels = labels_for_level(train_df, args.level, allow_provisional=args.allow_provisional)
    val_df = filter_to_train_vocab(val_df, args.level, train_vocab, allow_provisional=args.allow_provisional)
    test_df = filter_to_train_vocab(test_df, args.level, train_vocab, allow_provisional=args.allow_provisional)
    if val_df.empty or test_df.empty:
        raise ValueError("Validation/test became empty after filtering labels not present in train vocabulary")

    features = load_feature_matrix(train_df, Path(args.embeddings_dir))
    dataset = TripletFeatureDataset(features, train_labels)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.num_workers > 0 and args.persistent_workers,
    )

    device = resolve_device(args.device)
    model = LayerNormNet(args.input_dim, args.hidden_dim, args.out_dim, args.num_layers, drop_out=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps)
    criterion = nn.TripletMarginLoss(margin=args.margin, reduction="mean")
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda" and amp_dtype_for_device(device) == torch.float16)
    autocast_dtype = amp_dtype_for_device(device)

    last_path = out_dir / "checkpoint_last.pt"
    best_path = out_dir / "bestmodel_state_dict.pth"
    start_epoch = 1
    best_loss = float("inf")
    if last_path.exists() and not args.overwrite:
        try:
            checkpoint = torch.load(last_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = int(checkpoint["epoch"]) + 1
            best_loss = float(checkpoint.get("best_loss", best_loss))
            console.print(f"Resuming {args.level} from epoch {start_epoch}")
        except Exception as error:
            quarantine_checkpoint(last_path, error)

    log_rows: List[Dict] = []
    total_epochs = max(0, args.epochs - start_epoch + 1)
    progress = None
    epoch_task = None
    if args.progress_mode == "live":
        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        )
        progress.start()
        epoch_task = progress.add_task(f"Training {args.run_name}", total=total_epochs)
    elif args.progress_mode == "lines":
        console.print(f"[bold]{args.run_name}[/bold] starting epochs {start_epoch}-{args.epochs}")
    elif args.progress_mode == "json":
        emit_progress({"event": "start", "run_name": args.run_name, "start_epoch": start_epoch, "epochs": args.epochs, "total": total_epochs})

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            start_time = time.time()
            model.train()
            total_loss = 0.0
            steps = 0
            for anchor, positive, negative in loader:
                anchor = anchor.to(device=device, dtype=torch.float32, non_blocking=True)
                positive = positive.to(device=device, dtype=torch.float32, non_blocking=True)
                negative = negative.to(device=device, dtype=torch.float32, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, dtype=autocast_dtype, enabled=args.amp and device.type == "cuda"):
                    loss = criterion(model(anchor), model(positive), model(negative))
                scaler.scale(loss).backward()
                if args.clip_grad > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                scaler.step(optimizer)
                scaler.update()
                total_loss += float(loss.detach().cpu())
                steps += 1
            train_loss = total_loss / max(1, steps)
            if train_loss < best_loss:
                best_loss = train_loss
                save_checkpoint(best_path, model, optimizer, epoch, best_loss, args)
            save_checkpoint(last_path, model, optimizer, epoch, best_loss, args)
            epoch_seconds = time.time() - start_time
            row = {"epoch": epoch, "train_loss": train_loss, "seconds": epoch_seconds, "best_loss": best_loss}
            log_rows.append(row)
            pd.DataFrame(log_rows).to_csv(out_dir / "logfile.csv", index=False)
            if progress is not None and epoch_task is not None:
                progress.advance(epoch_task)
            elif args.progress_mode == "lines" and should_log_epoch(epoch, start_epoch, args.epochs, args.progress_interval):
                pct = 100.0 * (epoch - start_epoch + 1) / max(1, total_epochs)
                console.print(
                    f"{args.run_name} epoch {epoch}/{args.epochs} "
                    f"({pct:.1f}%) loss={train_loss:.6f} best={best_loss:.6f} seconds={epoch_seconds:.2f}"
                )
            elif args.progress_mode == "json" and should_log_epoch(epoch, start_epoch, args.epochs, args.progress_interval):
                emit_progress(
                    {
                        "event": "progress",
                        "run_name": args.run_name,
                        "epoch": epoch,
                        "epochs": args.epochs,
                        "completed": epoch - start_epoch + 1,
                        "total": total_epochs,
                        "loss": train_loss,
                        "best_loss": best_loss,
                        "seconds": epoch_seconds,
                    }
                )
    finally:
        if progress is not None:
            progress.stop()
    if args.progress_mode == "json":
        emit_progress({"event": "done", "run_name": args.run_name, "completed": total_epochs, "total": total_epochs})

    checkpoint_for_eval = best_path if best_path.exists() else last_path
    val_out = out_dir / "val"
    test_out = out_dir / "test"
    val_metrics = evaluate_checkpoint(
        Path(args.train_path),
        Path(args.val_path),
        checkpoint_for_eval,
        val_out,
        args.level,
        embeddings_dir=Path(args.embeddings_dir),
        device_name=args.device,
        topk=args.topk,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        num_layers=args.num_layers,
    )
    test_metrics = evaluate_checkpoint(
        Path(args.train_path),
        Path(args.test_path),
        checkpoint_for_eval,
        test_out,
        args.level,
        embeddings_dir=Path(args.embeddings_dir),
        device_name=args.device,
        topk=args.topk,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        num_layers=args.num_layers,
    )
    summary = {
        "level": args.level,
        "train_path": args.train_path,
        "val_path": args.val_path,
        "test_path": args.test_path,
        "num_train": len(train_df),
        "num_val_in_vocab": len(val_df),
        "num_test_in_vocab": len(test_df),
        "num_labels": len(train_vocab),
        "best_loss": best_loss,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    save_json(out_dir / "run_summary.json", summary)
    table = Table(title=f"CLAIRE {args.level} test metrics")
    table.add_column("metric")
    table.add_column("value")
    for key, value in test_metrics.items():
        table.add_row(key, f"{value:.6f}")
    console.print(table)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--val_path", required=True)
    parser.add_argument("--test_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--level", choices=["l1", "l2", "l3", "l4"], required=True)
    parser.add_argument("--embeddings_dir", default=str(DEFAULT_EMBEDDINGS_DIR))
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--gpus")
    parser.add_argument("--input_dim", type=int, default=512)
    parser.add_argument("--hidden_dim", type=int, default=1280)
    parser.add_argument("--out_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=6000)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true", default=True)
    parser.add_argument("--persistent_workers", action="store_true", default=True)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--clip_grad", type=float, default=0.0)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--progress_mode", choices=["live", "lines", "json", "none"], default="live")
    parser.add_argument("--progress_interval", type=int, default=10)
    parser.add_argument("--allow_provisional", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.progress_interval < 1:
        raise SystemExit("--progress_interval must be >= 1")
    if args.run_name is None:
        args.run_name = f"{args.level} seed={args.seed}"
    return args


if __name__ == "__main__":
    parsed_args = parse_args()
    configure_visible_gpus(parsed_args.gpus)
    train_one(parsed_args)
