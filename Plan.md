# CLAIRE EMULaToR Bench Plan

## Baseline Findings

CLAIRE predicts EC numbers from reaction SMILES using two reaction featurizers: rxnfp and DRFP. The released README describes concatenating a 256-dimensional rxnfp vector with a 256-dimensional DRFP vector to form the 512-dimensional model input. Training uses a fully connected LayerNorm network with Triplet Margin Loss. The paper trained separate models for first-, second-, and third-level EC prediction; this bench extends that pattern with a fourth full-EC model for EMULaToR evaluation.

The original training and inference scripts contain fixed relative paths and a hardcoded `/root/CLAIRE` import path, so the bench will not call or edit those scripts. It will keep the original architecture and loss behavior in wrapper-owned code under `emulator_bench/`.

## Dataset Mapping

Source splits live under `/home/adhil/github/EMULaToR/data/processed/datasets/enzyme_retrieval_dataset` and are copied to `/home/adhil/github/EMULaToR/data/processed/baselines/CLAIRE`. Supported groups are `random_splits`, `uniprot_time_splits`, and `reaction_drfp_tanimoto_splits/threshold_*`.

Each split table must contain `rxn_smiles`, `ec_number`, and `uniprot_date`. CLAIRE uses only `rxn_smiles` as input and `ec_number` as the source label. Sequence artifacts are intentionally ignored because CLAIRE is reaction-only.

## Implementation

Add a self-contained `emulator_bench/` package with split copying, feature caching, training, evaluation, Optuna, and launch scripts. Reaction embeddings are cached once globally under `/home/adhil/github/EMULaToR/data/processed/baselines/CLAIRE/embeddings`, keyed by normalized reaction SMILES hash. Every split, EC level, Optuna trial, and retrain reuses those files.

Default single-GPU commands use environment `mldb` and select the physical GPU with `--gpus`, for example `--gpus 2`. Inside the process, the selected visible GPU is addressed as logical `cuda:0`. Training defaults preserve the paper settings used for comparison: 512 input dimension, 1280 hidden dimension, 128 output dimension, 5 layers, Triplet Margin Loss margin 1, 2000 epochs, and learning rate `1e-4`.

EC validation uses official syntax rather than synthetic numeric ranges. L1 must be 1-7. L2 and L3 have no fixed upper bound, with `99` treated as "other" only if observed. L4 has no fixed numeric cap. Dash partials are allowed only in lower unspecified positions, and provisional `n` labels are excluded by default. Train/eval vocabularies are built from the observed train split for each EC level.

## Execution Sequence

1. Copy split files into the CLAIRE baseline data root.
2. Cache DRFP and rxnfp features for every unique reaction that appears in selected splits.
3. Train separate EC-level models for L1, L2, L3, and L4.
4. Evaluate each model with nearest-neighbor lookup against the train split.
5. Aggregate per-level weighted F1, accuracy, top-k accuracy, and EC prefix accuracy with `python -m emulator_bench.aggregate_results`. The wrapper reports validation and test partitions separately, validates metrics against saved predictions, and writes per-seed CSV, grouped CSV/JSON, and diagnostics artifacts.
6. Optionally run Optuna over retraining-safe optimization settings and launch retraining from best hyperparameters.

## Smoke Test

Use `conda run -n mldb ... --gpus 2` and a tiny sampled split while preserving the same cache/train/eval path. The smoke test should verify split copying, schema validation, EC parsing, DRFP/rxnfp cache writes and resume behavior, Rich progress rendering, one short training epoch, prediction/metric output, and skip-on-rerun behavior.
