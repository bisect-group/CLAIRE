import argparse
from pathlib import Path

import optuna

from emulator_bench.common import DEFAULT_DEVICE, DEFAULT_EMBEDDINGS_DIR, configure_visible_gpus, console, save_json
from emulator_bench.train import parse_args as parse_train_args
from emulator_bench.train import train_one


def objective(trial: optuna.Trial, base_args: argparse.Namespace) -> float:
    args = argparse.Namespace(**vars(base_args))
    args.lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    args.batch_size = trial.suggest_categorical("batch_size", [1024, 2048, 4096, 6000, 8192])
    args.dropout = trial.suggest_float("dropout", 0.0, 0.3)
    args.margin = trial.suggest_categorical("margin", [0.5, 1.0, 1.5, 2.0])
    args.clip_grad = trial.suggest_categorical("clip_grad", [0.0, 1.0, 5.0])
    args.out_dir = str(Path(base_args.out_dir) / f"trial_{trial.number}")
    summary = train_one(args)
    return float(summary["val_metrics"]["weighted_f1"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--val_path", required=True)
    parser.add_argument("--test_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--level", choices=["l1", "l2", "l3", "l4"], required=True)
    parser.add_argument("--embeddings_dir", default=str(DEFAULT_EMBEDDINGS_DIR))
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--gpus")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--study_name", default="claire_optuna")
    parser.add_argument("--storage")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    configure_visible_gpus(args.gpus)

    train_args = argparse.Namespace(
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
        out_dir=args.out_dir,
        level=args.level,
        embeddings_dir=args.embeddings_dir,
        device=args.device,
        gpus=args.gpus,
        input_dim=512,
        hidden_dim=1280,
        out_dim=128,
        num_layers=5,
        dropout=0.1,
        lr=1e-4,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        margin=1.0,
        epochs=args.epochs,
        batch_size=6000,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        amp=True,
        clip_grad=0.0,
        topk=5,
        seed=args.seed,
        allow_provisional=False,
        overwrite=False,
    )
    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
    )
    study.optimize(lambda trial: objective(trial, train_args), n_trials=args.n_trials)
    result = {"best_value": study.best_value, "best_hparams": study.best_params}
    save_json(Path(args.out_dir) / "optuna_best_hparams.json", result)
    console.print(result)


if __name__ == "__main__":
    main()
