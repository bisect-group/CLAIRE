# CLAIRE emulator_bench

This bench adds split-driven EMULaToR retraining to CLAIRE without modifying the original `dev/` scripts.

## Inputs

The source EMULaToR dataset is:

```text
/home/adhil/github/EMULaToR/data/processed/datasets/enzyme_retrieval_dataset
```

The bench copies it to:

```text
/home/adhil/github/EMULaToR/data/processed/baselines/CLAIRE
```

Supported split groups are:

- `random_splits`
- `uniprot_time_splits`
- `reaction_drfp_tanimoto_splits/threshold_*`

Each split file must have:

- `rxn_smiles`
- `ec_number`
- `uniprot_date`

CLAIRE is reaction-only, so sequence files are ignored.

## Features And Cache

Each unique reaction SMILES is embedded once:

- rxnfp: 256 values
- DRFP: 256 values
- concatenated feature: 512 values

The cache root is:

```text
/home/adhil/github/EMULaToR/data/processed/baselines/CLAIRE/embeddings/reactions/rxnfp_drfp_512
```

Cache files are keyed by SHA-256 of normalized `rxn_smiles`. Files are written atomically and reused across all splits, EC levels, Optuna trials, and retraining runs.

## EC Labels

The default strategy follows the CLAIRE paper style with separate level models:

- L1: `1`
- L2: `1.1`
- L3: `1.1.1`
- L4: `1.1.1.1`

L1 must be `1-7`. L2, L3, and L4 are not validated by synthetic numeric caps. Allowed labels come from the observed train vocabulary for each split and EC level. Provisional `n` labels are excluded by default.

## Training

Default settings preserve the paper comparison path:

- input dimension: `512`
- hidden dimension: `1280`
- output dimension: `128`
- layers: `5`
- loss: Triplet Margin Loss
- margin: `1`
- learning rate: `1e-4`
- epochs: `2000`
- physical GPU selection: `--gpus 2` or another CUDA-visible physical GPU id
- multiple physical GPUs: `--gpus 2,3`
- parallel training processes per physical GPU: `--parallel_runs_per_gpu 2`
- repeated seeds: `--seeds 1234 2345 3456`
- parallel launcher progress is append-only by default, with `--progress_interval 10`
- in-process device: `cuda:0` after `--gpus` masks the selected GPU

The wrapper enables TF32 when available and uses AMP with bf16 on Ampere-or-newer CUDA devices, otherwise fp16.

Feature caching pipelines CPU and GPU work:

- DRFP for the next batch runs in a background worker while rxnfp embeds the current batch on GPU.
- Cache file writes run in background workers so the next rxnfp batch can start sooner.
- Default rxnfp cache batch size is `128`; increase `--rxnfp_batch_size` or `--cache_rxnfp_batch_size` only if GPU utilization is low and memory has headroom.

## Outputs

Each split/level/seed run writes:

- `checkpoint_last.pt`
- `bestmodel_state_dict.pth`
- `logfile.csv`
- `val/predictions.csv`
- `val/metrics.json`
- `test/predictions.csv`
- `test/metrics.json`
- `run_summary.json`

## Resume Behavior

Re-running a command:

- skips copied files with matching size
- skips existing feature cache files
- resumes training from `checkpoint_last.pt`
- skips completed split/level jobs in `run_split_benchmarks.py` unless `--overwrite` is passed

## Main Commands

Validate and copy splits:

```bash
conda run -n mldb python -m emulator_bench.dataset_adapter --gpus 2
```

Cache all reaction features:

```bash
conda run -n mldb python -m emulator_bench.cache_features \
  --gpus 2 \
  --batch_size 128 \
  --rxnfp_batch_size 128 \
  --drfp_workers 1 \
  --save_workers 2
```

Run all split and EC-level benchmarks with original settings:

```bash
conda run -n mldb python -m emulator_bench.run_split_benchmarks \
  --gpus 2 \
  --cache_batch_size 128 \
  --cache_rxnfp_batch_size 128 \
  --cache_drfp_workers 1 \
  --cache_save_workers 2
```

Run all split and EC-level benchmarks for multiple seeds, with two training runs per selected GPU:

```bash
conda run -n mldb python -m emulator_bench.run_split_benchmarks \
  --gpus 2,3 \
  --parallel_runs_per_gpu 2 \
  --seeds 1234 2345 3456 \
  --progress_interval 10 \
  --cache_batch_size 128 \
  --cache_rxnfp_batch_size 128 \
  --cache_drfp_workers 1 \
  --cache_save_workers 2
```

Short smoke test:

```bash
conda run -n mldb python -m emulator_bench.run_split_benchmarks \
  --gpus 2 \
  --split_groups random_splits \
  --levels l1 \
  --epochs 1 \
  --batch_size 128
```
