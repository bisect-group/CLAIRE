import argparse
import subprocess
import sys
from pathlib import Path

from emulator_bench.common import DEFAULT_DEVICE, REPO_ROOT, console, subprocess_env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--val_path", required=True)
    parser.add_argument("--test_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--level", required=True, choices=["l1", "l2", "l3", "l4"])
    parser.add_argument("--gpus", nargs="+", default=["1"])
    parser.add_argument("--devices", nargs="+", default=["cuda:0"])
    parser.add_argument("--n_trials_per_worker", type=int, default=5)
    parser.add_argument("--storage", required=True)
    parser.add_argument("--study_name", default="claire_optuna")
    parser.add_argument("--epochs", type=int, default=2000)
    args = parser.parse_args()

    processes = []
    for index, gpu in enumerate(args.gpus):
        device = args.devices[index] if index < len(args.devices) else args.devices[0]
        out_dir = Path(args.out_dir) / f"worker_{index}"
        cmd = [
            sys.executable,
            "-m",
            "emulator_bench.tune_optuna",
            "--train_path",
            args.train_path,
            "--val_path",
            args.val_path,
            "--test_path",
            args.test_path,
            "--out_dir",
            str(out_dir),
            "--level",
            args.level,
            "--device",
            device,
            "--gpus",
            gpu,
            "--n_trials",
            str(args.n_trials_per_worker),
            "--storage",
            args.storage,
            "--study_name",
            args.study_name,
            "--epochs",
            str(args.epochs),
        ]
        console.print(" ".join(cmd))
        processes.append(subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=subprocess_env(gpu)))
    failures = [process.wait() for process in processes]
    if any(code != 0 for code in failures):
        raise SystemExit(f"Optuna worker failure codes: {failures}")


if __name__ == "__main__":
    main()
