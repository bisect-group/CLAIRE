"""GPU check of the bf16 option (`--amp`): the bf16 run against the fp32 run, bounded by a noise floor.

The floor is the gap between the bf16 run and a bf16 run whose initial weights are perturbed by the bf16 unit
roundoff, 2**-8 relative. The gap of the bf16 run to the fp32 run must stay within K times that floor.
With EQUIV_NEGATIVE_CONTROL=1 the gap run starts from weights times 1.05 and the test must fail.
Dropout is 0 here: fp32 and bf16 dropout draw different masks.
"""

import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from emulator_bench import fit
from emulator_bench.utils import original_modules

K = 10
U = 2.0**-8
HPARAMS = {
    "input_dim": 512, "hidden_dim": 1280, "out_dim": 128, "num_layers": 5, "dropout": 0.0, "learning_rate": 1e-4,
    "betas": [0.9, 0.999], "margin": 1.0, "batch_size": 1000, "adaptive_rate": 200,
}  # fmt: skip


def fixture():
    rng = np.random.RandomState(0)
    labels = [f"1.{i % 40}.1.{i % 7}" for i in range(4000)]
    centers = rng.randn(len(set(labels)), 512)
    index = {ec: i for i, ec in enumerate(dict.fromkeys(labels))}
    x = centers[[index[ec] for ec in labels]] + 0.5 * rng.randn(len(labels), 512)
    return labels, torch.from_numpy(x.astype(np.float32))


MODULE = original_modules()
ORIGINAL_MODEL = MODULE.LayerNormNet


def patch_init(monkeypatch, scale=1.0, rel=0.0):
    """The original model class, with the initial weights scaled and perturbed by `rel` relative noise."""

    def build(*args, **kwargs):
        model = ORIGINAL_MODEL(*args, **kwargs)
        generator = torch.Generator().manual_seed(123)
        with torch.no_grad():
            for p in model.parameters():
                p.mul_(scale)
                p.add_(rel * p.abs() * torch.randn(p.shape, generator=generator).to(p.device))
        return model

    monkeypatch.setattr(fit, "original_modules", lambda: SimpleNamespace(LayerNormNet=build, seed=MODULE.seed))


@pytest.mark.gpu
def test_bf16_training_tracks_fp32_within_noise(monkeypatch):
    labels, x = fixture()
    x = x.cuda()
    groups = fit.Groups(labels)

    def run(amp, scale=1.0, rel=0.0):
        patch_init(monkeypatch, scale, rel)
        model, losses = fit.train(x, groups, HPARAMS, seed=0, epochs=5, amp=amp)
        with torch.no_grad():
            return torch.tensor(losses, dtype=torch.float64), model.eval()(x[:512]).double()

    ref, noise, clean = run(False), run(True, rel=U), run(True)
    prod = run(True, scale=1.05) if os.environ.get("EQUIV_NEGATIVE_CONTROL") == "1" else clean
    gap, floor = (prod[0] - ref[0]).abs().max().item(), (noise[0] - clean[0]).abs().max().item()
    gap0, floor0 = (prod[0][0] - ref[0][0]).abs().item(), (noise[0][0] - clean[0][0]).abs().item()
    tiny = 1e-7 * ref[0].abs().max().item()
    final_gap, final_floor = (prod[1] - ref[1]).abs().mean().item(), (noise[1] - clean[1]).abs().mean().item()
    print(
        f"CLAIRE bf16: max loss gap={gap:.3e}, floor={floor:.3e}, ratio={gap / max(floor, tiny):.2f}, K={K}; "
        f"epoch-1 gap={gap0:.3e}, floor={floor0:.3e}, ratio={gap0 / max(floor0, tiny):.2f}; "
        f"final embedding gap={final_gap:.3e}, floor={final_floor:.3e}, ratio={final_gap / max(final_floor, tiny):.2f}"
    )
    assert gap0 <= K * max(floor0, tiny)
    assert gap <= K * max(floor, tiny)
    assert final_gap <= K * max(final_floor, tiny)
