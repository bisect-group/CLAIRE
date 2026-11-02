# CLAIRE wrapper for EMULaToR enzyme retrieval

This folder runs CLAIRE (reaction to EC) on the reaction-only EMULaToR splits.
It trains with the published settings only. It does not tune anything.
The input is the reaction SMILES (rxnfp and DRFP features). The model does not use protein data.

Splits: `random_splits_grouped_rxn_key`, `uniprot_time_splits` and `reaction_drfp_tanimoto_splits/threshold_*`
of the protein-filtered `enzyme_retrieval_dataset` (the MLP baseline copy under
`outputs/baselines/MLP/data/classification_protein/`). Pass it with `--dataset_root`. Seeds: 666, 777, 888.
After B-EC removal the test sets hold 1797 (random), 3163 (time) and 1740 (Tanimoto) queries, the same as MLP.

## Run

All commands run from the submodule root (`baselines/CLAIRE`). `commands.txt` has the full list.

```bash
PY=python   # an env with torch, numba, pandas, pyarrow, rxnfp, drfp, rdkit, transformers<5, setuptools<81
$PY -m emulator_bench.run_all --gpus 1 --runs_per_gpu 1 --amp --dataset_root <protein-filtered dir>
```

- `--gpus 0 1`: each run gets one GPU. Use `--gpus 1` on e2ea100_2 (GPU 0 belongs to another user).
- `--runs_per_gpu N`: runs at the same time on one GPU. One run fills the GPU, so 1 is best.
- `--amp`: bf16 autocast, about 3.6 times faster. It is off by default (fp32, as the original).
- `--epochs N`: for tests only. The default is the published 2000.

`run_all` builds the feature cache first, runs every split and seed, then writes `summary_by_seed.csv` and `summary.csv`
under `OUT/enzyme_retrieval/`. A finished run is skipped. An unfinished run resumes from `checkpoint_last.pt`.

## Output

`OUT/enzyme_retrieval/<split_group>/<name|all>/seed_<s>/` holds `test_predictions.parquet` (top 50 ECs and scores per
test row), `ec_level_accuracy.json`, `ec_level_accuracy_no_b.json`, `metrics.json`, `logfile.csv`, `lastmodel.pth` and
`checkpoint_last.pt`. `OUT/embeddings/rxnfp_drfp_v1/` holds the feature cache (about 830 MB).

## Check

```bash
bash emulator_bench/verify_equivalence.sh                    # CPU, wrapper against the original code
bash emulator_bench/verify_equivalence.sh --gpu              # only the bf16 noise-floor test, needs a CUDA device
bash emulator_bench/verify_equivalence.sh --negative-control # must fail on purpose
```

`REPORT.md` has the settings, the deviations and the equivalence numbers.
The older wrapper files (`train.py`, `tune_optuna.py`, `launch_parallel_*`, ...) are not part of this path.
