import argparse
import subprocess
import sys
from itertools import cycle
from pathlib import Path

from emulator_bench.common import DEFAULT_BASE_DIR, DEFAULT_RESULTS_DIRNAME, REPO_ROOT, console, discover_split_jobs, load_json, subprocess_env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--levels", nargs="*", default=["l1", "l2", "l3", "l4"], choices=["l1", "l2", "l3", "l4"])
    parser.add_argument("--gpus", nargs="+", default=["1"])
    parser.add_argument("--devices", nargs="+", default=["cuda:0"])
    parser.add_argument("--hparams_json", required=True)
    parser.add_argument("--results_dirname", default="claire_retrain_from_optuna")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    hparams = load_json(Path(args.hparams_json)).get("best_hparams", load_json(Path(args.hparams_json)))
    gpu_iter = cycle(args.gpus)
    processes = []
    for job in discover_split_jobs(Path(args.base_dir)):
        for level in args.levels:
            gpu = next(gpu_iter)
            device = args.devices[0]
            out_dir = job.root_dir / args.results_dirname / level / f"seed_{args.seed}"
            cmd = [
                sys.executable,
                "-m",
                "emulator_bench.train",
                "--train_path",
                str(job.train_path),
                "--val_path",
                str(job.val_path),
                "--test_path",
                str(job.test_path),
                "--out_dir",
                str(out_dir),
                "--level",
                level,
                "--device",
                device,
                "--gpus",
                gpu,
                "--epochs",
                str(args.epochs),
                "--seed",
                str(args.seed),
            ]
            for key in ("lr", "batch_size", "dropout", "margin", "clip_grad"):
                if key in hparams:
                    cmd.extend([f"--{key}", str(hparams[key])])
            console.print(" ".join(cmd))
            processes.append(subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=subprocess_env(gpu)))
            if len(processes) >= len(args.gpus):
                code = processes.pop(0).wait()
                if code != 0:
                    raise SystemExit(code)
    for process in processes:
        code = process.wait()
        if code != 0:
            raise SystemExit(code)


if __name__ == "__main__":
    main()
