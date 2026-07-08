import argparse
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
from rich.progress import track
from sklearn.metrics import accuracy_score, f1_score

from emulator_bench.common import (
    DEFAULT_DEVICE,
    DEFAULT_EMBEDDINGS_DIR,
    configure_visible_gpus,
    console,
    feature_cache_path,
    read_table,
    resolve_device,
    save_json,
)
from emulator_bench.ec import ec_label, is_unlabeled_ec, prefix_accuracy
from emulator_bench.modeling import LayerNormNet, embed_features


def load_feature_matrix(df: pd.DataFrame, embeddings_dir: Path) -> np.ndarray:
    rows = []
    for smiles in df["rxn_smiles"].astype(str):
        path = feature_cache_path(embeddings_dir, smiles)
        if not path.exists():
            raise FileNotFoundError(f"Missing cached feature for reaction: {smiles}")
        rows.append(np.load(path)["feature"].astype(np.float32))
    return np.vstack(rows).astype(np.float32)


def drop_unusable_labels(df: pd.DataFrame, level: str) -> pd.DataFrame:
    keep = []
    for value in df["ec_number"].astype(str):
        if is_unlabeled_ec(value):
            keep.append(False)
            continue
        try:
            ec_label(value, level)
            keep.append(True)
        except ValueError:
            keep.append(False)
    return df.loc[keep].reset_index(drop=True)


def nearest_neighbor_predictions(
    train_embeddings: torch.Tensor,
    test_embeddings: torch.Tensor,
    train_labels: Sequence[str],
    topk: int = 3,
    distance_batch_size: int = 1024,
) -> List[List[str]]:
    predictions: List[List[str]] = []
    topk = min(topk, len(train_labels))
    labels = list(train_labels)
    for start in track(range(0, len(test_embeddings), distance_batch_size), description="Nearest-neighbor eval", console=console):
        batch = test_embeddings[start : start + distance_batch_size]
        distances = torch.cdist(batch, train_embeddings)
        indices = torch.topk(distances, k=topk, largest=False).indices.cpu().numpy()
        for row in indices:
            seen = []
            for index in row:
                label = labels[int(index)]
                if label not in seen:
                    seen.append(label)
            predictions.append(seen)
    return predictions


def compute_metrics(
    y_true: Sequence[str],
    ranked_predictions: Sequence[Sequence[str]],
    source_true_ec: Sequence[str] | None = None,
    prediction_level: str = "l4",
    max_topk: int = 5,
) -> Dict[str, float]:
    y_pred = [preds[0] if preds else "" for preds in ranked_predictions]
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    requested_ks = sorted({k for k in (1, 3, 5, max_topk) if 1 <= k <= max_topk})
    for k in requested_ks:
        total = len(y_true)
        correct = sum(1 for true, preds in zip(y_true, ranked_predictions) if true in list(preds)[:k])
        metrics[f"top{k}_accuracy"] = float(correct / total) if total else 0.0
    if source_true_ec is not None:
        allowed = ["l1", "l2", "l3", "l4"][: ["l1", "l2", "l3", "l4"].index(prediction_level) + 1]
        for level in allowed:
            try:
                metrics[f"{level}_prefix_accuracy"] = float(prefix_accuracy(source_true_ec, y_pred, level))
            except ValueError:
                metrics[f"{level}_prefix_accuracy"] = 0.0
    return metrics


def evaluate_checkpoint(
    train_path: Path,
    eval_path: Path,
    checkpoint_path: Path,
    out_dir: Path,
    level: str,
    embeddings_dir: Path = DEFAULT_EMBEDDINGS_DIR,
    device_name: str = DEFAULT_DEVICE,
    topk: int = 5,
    input_dim: int = 512,
    hidden_dim: int = 1280,
    out_dim: int = 128,
    num_layers: int = 5,
) -> Dict[str, float]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(device_name)

    train_df = drop_unusable_labels(read_table(train_path), level)
    eval_df = drop_unusable_labels(read_table(eval_path), level)
    train_labels = [ec_label(value, level) for value in train_df["ec_number"].astype(str)]
    eval_labels = [ec_label(value, level) for value in eval_df["ec_number"].astype(str)]

    train_features = load_feature_matrix(train_df, embeddings_dir)
    eval_features = load_feature_matrix(eval_df, embeddings_dir)
    model = LayerNormNet(input_dim, hidden_dim, out_dim, num_layers).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint)

    train_emb = embed_features(model, train_features, device)
    eval_emb = embed_features(model, eval_features, device)
    ranked = nearest_neighbor_predictions(train_emb, eval_emb, train_labels, topk=topk)
    metrics = compute_metrics(
        eval_labels,
        ranked,
        source_true_ec=eval_df["ec_number"].astype(str).tolist(),
        prediction_level=level,
        max_topk=topk,
    )

    rows = []
    for idx, (true_label, preds) in enumerate(zip(eval_labels, ranked)):
        row = {"row_index": idx, "true_label": true_label, "pred_label": preds[0] if preds else ""}
        for pred_idx, label in enumerate(preds, start=1):
            row[f"top{pred_idx}"] = label
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "predictions.csv", index=False)
    save_json(out_dir / "metrics.json", metrics)
    console.print(metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--eval_path", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--level", choices=["l1", "l2", "l3", "l4"], required=True)
    parser.add_argument("--embeddings_dir", default=str(DEFAULT_EMBEDDINGS_DIR))
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--gpus")
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()
    configure_visible_gpus(args.gpus)
    evaluate_checkpoint(
        Path(args.train_path),
        Path(args.eval_path),
        Path(args.checkpoint_path),
        Path(args.out_dir),
        args.level,
        embeddings_dir=Path(args.embeddings_dir),
        device_name=args.device,
        topk=args.topk,
    )


if __name__ == "__main__":
    main()
