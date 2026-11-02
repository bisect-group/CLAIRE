"""The wrapper against the original CLAIRE code on a tiny synthetic fixture (CPU, no downloads)."""

import pickle
import random
import sys

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from emulator_bench import augment as aug
from emulator_bench import fit, sampler
from emulator_bench.utils import original_modules

DIM = 16
HPARAMS = {
    "hidden_dim": 32, "out_dim": 8, "num_layers": 5, "dropout": 0.1, "learning_rate": 5e-4, "betas": [0.9, 0.999],
    "margin": 1.0, "batch_size": 6000, "adaptive_rate": 2,
}  # fmt: skip


def fixture(n_rows: int = 80):
    """Rows of ECs with one to many rows, so that single-row ECs and uneven EC sizes are drawn."""
    rng = np.random.RandomState(0)
    labels = [f"1.{i % 9}.1.{i % 4}" for i in range(n_rows - 6)] + ["2.1.1.1", "2.1.1.2", "2.1.1.3", "2.1.1.3", "9.9.9.9", "9.9.9.8"]
    rng.shuffle(labels)
    labels = [str(label) for label in labels]
    return labels, rng.randn(n_rows, DIM)


def original_dataset(labels, matrix):
    module = original_modules()
    esm_emb = {}
    for i, label in enumerate(labels):
        esm_emb.setdefault(label, []).append(torch.Tensor(matrix[i]))
    return module.Dataset(esm_emb, matrix, labels)


def test_sampler_matches_original_dataset():
    labels, matrix = fixture()
    dataset = original_dataset(labels, matrix)
    x = torch.from_numpy(matrix.astype(np.float32))
    by_row = {row.numpy().tobytes(): i for i, row in enumerate(x)}
    groups = fit.Groups(labels)
    stream = sampler.MTStream(7)
    random.seed(7)
    for epoch in range(4):
        order = np.random.RandomState(epoch).permutation(len(dataset))
        expected = [[by_row[t[0].float().numpy().tobytes()] for t in dataset[int(i)]] for i in order]
        assert fit.epoch_triplets(stream, order, groups).tolist() == expected


def run_original_training(tmp_path, monkeypatch, labels, matrix, epochs, batch=None):
    for folder in ("data/pred_rxn_EC123", "results/model/pred_rxn_EC123", "results/loss/pred_rxn_EC123", "training"):
        (tmp_path / folder).mkdir(parents=True)
    esm_emb = {}
    for i, label in enumerate(labels):
        esm_emb.setdefault(label, []).append(torch.Tensor(matrix[i]))
    pickle.dump(esm_emb, open(tmp_path / "data/pred_rxn_EC123/esm_emb_dict_ec3.pkl", "wb"))
    pickle.dump(matrix, open(tmp_path / "data/model_lookup_train.pkl", "wb"))
    pickle.dump(labels, open(tmp_path / "data/pred_rxn_EC123/labels_train_ec3.pkl", "wb"))
    monkeypatch.chdir(tmp_path / "training")
    module = original_modules()
    if batch:  # the original fixes the batch size at 6000; a small one makes the fixture span several batches
        monkeypatch.setattr(module, "DataLoader", lambda ds, batch_size, shuffle: DataLoader(ds, batch_size=batch, shuffle=shuffle))
    argv = ["train", "-e", str(epochs), "-i", str(DIM), "-d", "32", "-o", "8", "-l", "5", "-lr", "5e-4", "--adaptive_rate", "2"]
    monkeypatch.setattr(sys, "argv", argv)
    module.main()
    return torch.load(tmp_path / "results/model/pred_rxn_EC123" / f"train_{epochs}_final.pth")


@pytest.mark.parametrize("batch", [None, 32])
def test_training_matches_original_script(tmp_path, monkeypatch, batch):
    torch.use_deterministic_algorithms(True)
    labels, matrix = fixture()
    state = run_original_training(tmp_path, monkeypatch, labels, matrix, epochs=5, batch=batch)
    x = torch.from_numpy(matrix.astype(np.float32))
    model, losses = fit.train(x, fit.Groups(labels), {**HPARAMS, "batch_size": batch or 6000}, seed=1234, epochs=5)
    assert len(losses) == 5
    assert all(torch.equal(model.state_dict()[k], v) for k, v in state.items())


def test_resume_gives_the_same_model(tmp_path):
    labels, matrix = fixture()
    x, groups = torch.from_numpy(matrix.astype(np.float32)), fit.Groups(labels)
    hparams = {**HPARAMS, "batch_size": 32}
    straight, _ = fit.train(x, groups, hparams, seed=5, epochs=6)
    fit.train(x, groups, hparams, seed=5, epochs=3, run_dir=tmp_path, checkpoint_every=3)
    resumed, _ = fit.train(x, groups, hparams, seed=5, epochs=6, run_dir=tmp_path, checkpoint_every=3)
    assert all(torch.equal(v, resumed.state_dict()[k]) for k, v in straight.state_dict().items())


def test_rank_matches_original_distances():
    from dev.utils.utils import dist_map

    labels, matrix = fixture()
    x, groups = torch.from_numpy(matrix.astype(np.float32)), fit.Groups(labels)
    module = original_modules()
    torch.manual_seed(0)
    model = module.LayerNormNet(DIM, 32, 8, 5, torch.device("cpu"), torch.float32)
    queries = torch.randn(9, DIM)
    index, dist = fit.rank(model, x, groups, queries, top_k=5)
    model.eval()
    # One tag per reference row, so the original distance map keeps every distance. The nearest row of an EC is its distance.
    original = dist_map([str(q) for q in range(9)], model(queries), [str(i) for i in range(len(x))], model(x))
    for q in range(9):
        nearest: dict[str, float] = {}
        for i, label in enumerate(labels):
            nearest[label] = min(nearest.get(label, np.inf), original[str(q)][str(i)])
        expected = sorted(nearest.items(), key=lambda kv: kv[1])[:5]
        assert [groups.ec_names[j] for j in index[q]] == [ec for ec, _ in expected]
        assert np.array_equal(dist[q], np.array([d for _, d in expected], dtype=np.float32))


def test_augmentation_rows():
    rxn = "A.B.C>>D.E"
    rows = aug.augment(rxn)
    assert rows[0] == rxn and len(rows) == 10
    assert aug.augment("A.B>>C.D") == ["A.B>>C.D", "A.B>>D.C", "B.A>>C.D", "B.A>>D.C"]
    assert aug.augment("A.A>>C") == ["A.A>>C"]
    big = "A.B.C.D>>E.F"
    assert aug.augment(big) == aug.augment(big) and len(aug.augment(big)) == 10 and aug.augment(big)[0] == big
    assert {aug.canonical(r) for r in aug.augment(big)} == {aug.canonical(big)}


def test_features_match_the_original_recipes():
    from drfp import DrfpEncoder
    from rxnfp.transformer_fingerprints import RXNBERTFingerprintGenerator, get_default_model_and_tokenizer

    from emulator_bench import features

    reactions = aug.augment("NC(=O)c1ccc[n+](C)c1.NCCC=O.O>>NCCC(=O)O.Cc1ccc[n+](C)c1") + ["CCO>>CC=O", "CC(=O)O.OCC>>CC(=O)OCC.O"]
    model, tokenizer = get_default_model_and_tokenizer(force_no_cuda=True)
    generator = RXNBERTFingerprintGenerator(model, tokenizer, force_no_cuda=True)
    single = np.array([generator.convert(r) for r in reactions], dtype=np.float32)
    # The `drfp` command of data/embedding/embedding.py: each reaction on its own, `-d 256`, defaults otherwise.
    direct = np.array([DrfpEncoder.encode([r], n_folded_length=256, root_central_atom=False)[0] for r in reactions], dtype=np.float32)
    both = features.compute(reactions, device="cpu")
    assert np.abs(both[:, :256] - single).max() < 1e-4
    assert np.array_equal(both[:, 256:], direct)


def test_stream_state_round_trip():
    a, b = sampler.MTStream(3), sampler.MTStream(3)
    a.window(100)
    a.advance(40)
    b.load_state(a.state())
    assert np.array_equal(a.window(10)[:10], b.window(10)[:10])
    random.seed(3)
    assert [int(w) for w in sampler.MTStream(3).window(5)[:5]] == [random.getrandbits(32) for _ in range(5)]




@pytest.mark.parametrize("n, batch", [(7, 6000), (100, 32), (20001, 6000)])
def test_epoch_order_matches_dataloader(n, batch):
    from torch.utils.data import DataLoader

    class Indices(torch.utils.data.Dataset):
        def __len__(self):
            return n

        def __getitem__(self, i):
            return i

    torch.manual_seed(5)
    loader = DataLoader(Indices(), batch_size=batch, shuffle=True)
    expected = [torch.cat(list(loader)).numpy() for _ in range(3)]
    state = torch.get_rng_state()
    torch.manual_seed(5)
    assert all(np.array_equal(fit.epoch_order(n), e) for e in expected)
    assert torch.equal(torch.get_rng_state(), state)
