# CLAIRE on EMULaToR enzyme retrieval (reaction to EC)

## 1. Method summary

CLAIRE predicts the EC number of a reaction. The input is the rxnfp embedding (256 numbers) joined with the DRFP
fingerprint (256 bits) of the reaction. A network with 5 layers (1280 units, LayerNorm, dropout 0.1) maps it to 128 numbers.
It trains with a triplet loss. The anchor is a reaction. The positive is a reaction of the same EC.
The negative is a reaction of another EC, drawn at random. The training set is augmented: the order of the
reactants and the order of the products are shuffled. Adam restarts every 200 epochs.
At test time each query is embedded and the ECs are ranked by the distance to the nearest training reaction of the EC.
The paper trains three models (EC level 1, 2 and 3). The wrapper trains one model on the full four-level EC.

## 2. Original default settings

| Setting | Value | Source |
|---|---|---|
| Input | rxnfp 256 + DRFP 256 | paper Methods; `data/embedding/embedding.py` |
| Layers / hidden / output | 5 / 1280 / 128 | paper; `dev/training/train-pred_rxn_EC.py:18-20` |
| Dropout | 0.1 | `train-pred_rxn_EC.py:27` |
| Optimizer | Adam, betas (0.9, 0.999) | `train-pred_rxn_EC.py:96` |
| Learning rate | 1e-4 | paper (Hyperparameter tuning). The script default is 5e-4 (line 15). Open issue. |
| Loss | TripletMarginLoss, margin 1 | paper (alpha = 1); `train-pred_rxn_EC.py:97` |
| Batch size | 6000 | `train-pred_rxn_EC.py:100` |
| Epochs | 2000 | `train-pred_rxn_EC.py:16` |
| Adam restart | every 200 epochs | `train-pred_rxn_EC.py:21,106` |
| Augmentation | shuffle reactants and products | paper Methods |
| Training rows | train set only | `train-pred_rxn_EC.py` (the validation rows are not used) |
| Final model | last epoch | README (`..._final.pth`) |

The values are in `default_hparams.json` with their sources.

## 3. What the wrapper does

- `data.py` reads `train` and `test` of each reaction-only split folder. It never re-splits.
  Rows without a reaction or an EC are dropped and counted in `metrics.json`.
- `augment.py` makes the training rows: the reaction, then its other reactant and product orders.
- `features.py` computes rxnfp and DRFP once per unique reaction string and caches them in
  `OUT/embeddings/rxnfp_drfp_v1/` (reaction list, 512-wide float32 matrix, library versions).
  DRFP does not depend on the order, so it is computed once per sorted form of a reaction.
- `fit.py` trains with the original `LayerNormNet`, loss and optimizer. The original draws triplets with Python
  `random` in a DataLoader. The wrapper keeps the features on the GPU and draws the same triplets with `sampler.py`.
- `fit.rank` ranks the ECs of each test reaction by the distance to the nearest training reaction of the EC (top 50).
  The score is the negative distance.
- `run_one.py` runs one split and seed. It writes `lastmodel.pth`, `checkpoint_last.pt` (resumable),
  `test_predictions.parquet`, `ec_level_accuracy.json`, `ec_level_accuracy_no_b.json`, `metrics.json`, `logfile.csv`.
- `run_all.py` builds the features, runs all splits and seeds on the given GPUs, then writes the summary files.
- Scoring is EC level accuracy (levels 1 to 4, k = 1 to 50). The `_no_b` file drops preliminary ECs from the truth
  and from the ranked lists. The test queries are the unique `rxn_smiles` of the test file (not augmented).

## 4. Files added, removed or changed

Added in `emulator_bench/`: `default_hparams.json`, `utils.py`, `data.py`, `augment.py`, `features.py`, `sampler.py`,
`fit.py`, `run_one.py`, `build_features.py`, `run_all.py`, `results.py`, `verify_equivalence.sh`, `REPORT.md`, `README.md`, `commands.txt`.
Added in `tests/`: `conftest.py`, `test_equivalence.py`, `test_gpu_precision.py`.
Nothing is removed. Original code in `dev/` is not changed.
The older files stay and are not used by the production path: `aggregate_results.py`, `cache_features.py`, `common.py`,
`dataset_adapter.py`, `ec.py`, `evaluate.py`, `modeling.py`, `queue_pipeline.py`, `train.py`, `tune_optuna.py`,
`launch_parallel_optuna.py`, `launch_parallel_retrain_from_optuna.py`, `run_split_benchmarks.py`,
`default_hparams_claire_original.json`. The Optuna files search hyperparameters. The production path does not call them.

## 5. Bugs fixed

The older wrapper tuned hyperparameters with Optuna. The new path uses the published settings only.

## 6. Deviations and why behavior is unchanged

| Change | Evidence that the result is the same |
|---|---|
| Triplets drawn by numba from the Mersenne Twister words of `random.seed(seed)` | Same triplets as the original `Dataset` for 4 epochs (`test_sampler_matches_original_dataset`). |
| Epoch order from `torch.randperm` with the two RNG draws of `DataLoader(shuffle=True)` | Same order and torch RNG state as a real DataLoader (`test_epoch_order_matches_dataloader`). |
| Features kept on the GPU, no pickles | Final weights bitwise equal to `main()` of the original script, with batch size 6000 and with 32 (`test_training_matches_original_script`). |
| Ranking per query with a scatter minimum | Same EC order and bitwise equal distances to `dev.utils.utils.dist_map` plus a minimum per EC (`test_rank_matches_original_distances`). |
| DRFP once per sorted form of the reaction | Equal to DRFP of each ordering (`test_features_match_the_original_recipes`). |
| Batched rxnfp, sorted by length | Max difference to one-by-one conversion is below 1e-4 (same test). |
| Resume from `checkpoint_last.pt` | Bitwise equal to a straight run (`test_resume_gives_the_same_model`). |

Deviations that come from the data and the released material:

- One model for the full four-level EC. The paper trains models for EC levels 1, 2 and 3.
- Ranking uses the nearest training reaction of each EC (paper). The released `dist_map` stores one distance per EC
  label and keeps the last training row of the label, so it depends on the row order.
- The augmentation code is not released. `augment.py` copies the rule seen in the released `train_augmented.csv`:
  all distinct orders when a reaction has at most 10, else 10 orders drawn with replacement. The draws are seeded by the reaction.
- DRFP: the `drfp` command of `embedding.py` has defaults radius 3, rings, no root atoms. The wrapper uses the same call.
  With the installed `drfp` 0.3.7 and rdkit 2026.9 it gives about 70 percent of the on-bits of the released `drfp/model_lookup_train.pkl`
  and not the same bits (a different RDKit version is the likely cause). rxnfp equals the released file to 9e-6.

The default precision is fp32. `--amp` is bf16 autocast. It is off by default and the default result does not change with the flag.

## 7. Commands

See `commands.txt`. All runs: `python -m emulator_bench.run_all --gpus 1 --runs_per_gpu 1 --amp --dataset_root <protein-filtered dir>`
(3 split folders x 3 seeds = 9 runs: `random_splits_grouped_rxn_key`, `uniprot_time_splits`, `reaction_drfp_tanimoto_splits/threshold_0.425`; seeds 666, 777, 888).

## 8. Smoke test

`run_one --epochs 2 --sample_rows 64` writes all files. On e2ea100_2 `run_all --amp --epochs 2` ran all three splits
and gave the query counts above.
The `_no_b` accuracy agrees with `src/utils/ec_level_accuracy.py` on the same predictions.

## 9. Equivalence verdict

EQUIVALENT-EXCEPT-DOCUMENTED. On the CPU fixture the wrapper gives bitwise equal weights to the original script.
The documented deviations are in section 6. Run `bash emulator_bench/verify_equivalence.sh` (CPU, 11 tests, no skips).

GPU check of bf16 (`verify_equivalence.sh --gpu`): gap of the bf16 run to the fp32 run, divided by the noise floor
(the bf16 run with initial weights perturbed by 2^-8). K = 10.

| Quantity | Ratio |
|---|---|
| Max loss gap over epochs | 0.62 |
| Epoch-1 loss gap | 0.70 |
| Final embedding gap | 0.80 |

Negative control (initial weights times 1.05, `--negative-control`): the max loss ratio is 11.99 and the epoch-1 ratio is 14.35, so the test fails as it must.

## 10. Open issues

- Learning rate: the paper says 1e-4, the script default is 5e-4. The wrapper uses 1e-4.
- Output level: one full-EC model instead of three models for levels 1 to 3 (user decision). Levels 1 to 3 come from its ranking.
- Augmentation rule, DRFP bits and ranking rule differ from the release as listed in section 6.
- The wrapper does not use the validation rows, as the released script.
- The test queries come from the protein-filtered files given with `--dataset_root`. The counts match MLP (1797, 3163, 1740).
- rxnfp needs `transformers<5` and `setuptools<81`.

Speed on e2ea100_2 (A100 80 GB, GPU 1). Protein-filtered splits, seconds per epoch:

| Split | augmented rows | fp32 | bf16 (`--amp`) |
|---|---|---|---|
| `random_splits_grouped_rxn_key` | 465,860 | - | 0.93 |
| `uniprot_time_splits` | 444,262 | - | 0.89 |
| `reaction_drfp_tanimoto_splits/threshold_0.425` | 489,893 | 3.38 | 0.94 |

One bf16 seed (three splits, 2000 epochs each) takes 1.5 h. The nine runs take 4.6 h. fp32 takes about 17 h.
Sharing the GPU with Enzyformer slows both runs by 1.7x to 1.8x, so sharing gives no gain over running them in turn.
The feature cache holds the reactions of all three splits (525 MB on this host).
