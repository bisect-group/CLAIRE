"""Training and ranking of the CLAIRE model (dev/training/train-pred_rxn_EC.py, dev/prediction/inference_EC.py).

The model, the loss, the optimizer and the loop follow the original. The original builds the triplets in a
DataLoader with Python `random` calls and torch tensors per sample. Here the features stay on the GPU and the
triplets come from sampler.py, which draws the same numbers.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from . import sampler
from .utils import original_modules

TOP_K = 50


class Groups:
    """The rows of each EC. ECs are numbered in order of first appearance, as the keys of the original dict."""

    def __init__(self, labels: list[str]):
        self.ec_names = list(dict.fromkeys(labels))
        index = {ec: i for i, ec in enumerate(self.ec_names)}
        self.row_ec = np.array([index[ec] for ec in labels], dtype=np.int64)
        self.ec_rows = np.argsort(self.row_ec, kind="stable").astype(np.int64)
        self.ec_ptr = np.concatenate([[0], np.cumsum(np.bincount(self.row_ec, minlength=len(self.ec_names)))]).astype(np.int64)


def epoch_order(n: int) -> np.ndarray:
    """The anchor order of one epoch of the original `DataLoader(shuffle=True)`: the same two torch RNG draws and permutation."""
    torch.empty((), dtype=torch.int64).random_()  # base seed of the loader iterator
    seed = int(torch.empty((), dtype=torch.int64).random_())  # seed of the random sampler
    return torch.randperm(n, generator=torch.Generator().manual_seed(seed)).numpy()


def epoch_triplets(stream: sampler.MTStream, order: np.ndarray, groups: Groups) -> np.ndarray:
    out = np.empty((len(order), 3), dtype=np.int64)
    done = 0
    while done < len(order):
        words = stream.window(sampler.MARGIN_WORDS + 64 * (len(order) - done))
        n, used = sampler.sample_triplets(words, order[done:], groups.row_ec, groups.ec_ptr, groups.ec_rows, out[done:])
        stream.advance(used)
        done += n
    return out


def save_checkpoint(path: Path, state: dict) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def train(
    x: torch.Tensor,
    groups: Groups,
    hparams: dict,
    seed: int,
    epochs: int,
    run_dir: Path | None = None,
    amp: bool = False,
    checkpoint_every: int = 100,
    log_every: int = 10,
):
    """Train on the feature rows `x`. Returns the model after the last epoch and the loss of each epoch."""
    module = original_modules()
    device = x.device
    module.seed(seed)
    stream = sampler.MTStream(seed)
    batch, adaptive = hparams["batch_size"], hparams["adaptive_rate"]
    model = module.LayerNormNet(
        x.shape[1], hparams["hidden_dim"], hparams["out_dim"], hparams["num_layers"], device, torch.float32, hparams["dropout"]
    )
    adam = lambda: torch.optim.Adam(model.parameters(), lr=hparams["learning_rate"], betas=tuple(hparams["betas"]))  # noqa: E731
    optimizer = adam()
    criterion = torch.nn.TripletMarginLoss(margin=hparams["margin"], reduction="mean")
    losses: list[float] = []
    start = 1
    checkpoint_path = run_dir / "checkpoint_last.pt" if run_dir else None
    if checkpoint_path is not None and checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"].cpu(), device)
        stream.load_state(state["stream"])
        losses, start = list(state["losses"]), state["epoch"] + 1
    started = time.time()
    for epoch in range(start, epochs + 1):
        if epoch % adaptive == 0:
            optimizer = adam()
        order = epoch_order(len(x))
        triplets = torch.from_numpy(epoch_triplets(stream, order, groups)).to(device)
        model.train()
        step_losses = []
        for b in range(0, len(order), batch):
            a, p, n = (x[triplets[b : b + batch, j]] for j in range(3))
            optimizer.zero_grad()
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                loss = criterion(model(a), model(p), model(n))
            loss.backward()
            optimizer.step()
            step_losses.append(loss.detach().float())
        losses.append(float(torch.stack(step_losses).double().sum()) / len(step_losses))
        if epoch % log_every == 0 or epoch == epochs:
            print(json.dumps({"epoch": epoch, "epochs": epochs, "loss": losses[-1], "seconds": time.time() - started}), flush=True)
        if checkpoint_path is not None and (epoch % checkpoint_every == 0 or epoch == epochs):
            save_checkpoint(
                checkpoint_path,
                {
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                    "stream": stream.state(), "losses": losses,
                },
            )  # fmt: skip
    return model, losses


@torch.no_grad()
def rank(model, x: torch.Tensor, groups: Groups, queries: torch.Tensor, top_k: int = TOP_K):
    """ECs ordered by the distance of the query to the nearest reference reaction of the EC.

    The model is in eval mode. The distances are computed per query, as in dev.utils.utils.dist_map.
    Returns the indices of the nearest `top_k` ECs and their distances, per query.
    """
    model.eval()
    reference = torch.cat([model(x[i : i + 8192]) for i in range(0, len(x), 8192)])
    row_ec = torch.from_numpy(groups.row_ec).to(x.device)
    embedded = model(queries)
    ec_index, ec_dist = [], []
    for start in range(0, len(queries), 256):
        nearest = torch.full((len(embedded[start : start + 256]), len(groups.ec_names)), torch.inf, device=x.device)
        for i, query in enumerate(embedded[start : start + 256]):
            nearest[i].scatter_reduce_(0, row_ec, (query.unsqueeze(0) - reference).norm(dim=1, p=2), "amin")
        values, order = torch.sort(nearest, dim=1, stable=True)
        ec_index.append(order[:, :top_k].cpu())
        ec_dist.append(values[:, :top_k].cpu())
    return torch.cat(ec_index).numpy(), torch.cat(ec_dist).numpy()
