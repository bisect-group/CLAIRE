"""Launch every split and seed: features once, then the runs on the given GPUs, then the summaries."""

from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .data import discover_splits
from .utils import (
    BASELINE_ROOT,
    DEFAULT_DATASET_ROOT,
    DEFAULT_OUT_ROOT,
    DEFAULT_SEEDS,
    DEFAULT_SPLIT_GROUPS,
    TASK,
    seed_dir,
)


def run_logged(command: list[str], log: Path, env: dict | None = None) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as handle:
        return subprocess.run(command, cwd=BASELINE_ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--split_groups", nargs="+", default=list(DEFAULT_SPLIT_GROUPS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--gpus", nargs="+", required=True, help="GPU ids; each run gets one through CUDA_VISIBLE_DEVICES")
    parser.add_argument("--runs_per_gpu", type=int, default=1)
    parser.add_argument("--amp", action="store_true", help="bf16 autocast on Ampere or newer. The default keeps fp32.")
    parser.add_argument("--epochs", type=int, default=None, help="Test-only override of the epoch count")
    parser.add_argument("--dry_run", action="store_true", help="List the runs and stop")
    args = parser.parse_args()

    splits = discover_splits(args.dataset_root, tuple(args.split_groups))
    jobs = [(s, seed) for s in splits for seed in args.seeds]
    for split, seed in jobs:
        print(f"run {split.id} seed {seed} -> {seed_dir(args.out_root, split.group, split.name, seed)}", flush=True)
    if args.dry_run:
        return
    py = [sys.executable, "-m"]
    common = ["--dataset_root", args.dataset_root, "--out_root", args.out_root, "--split_groups", *args.split_groups]
    logs = Path(args.out_root) / "logs" / TASK
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(args.gpus[0])}
    if run_logged([*py, "emulator_bench.build_features", *common], logs / "build_features.log", env):
        sys.exit(f"build_features failed; see {logs}")

    slots: queue.Queue = queue.Queue()
    for _ in range(args.runs_per_gpu):
        for gpu in args.gpus:
            slots.put(gpu)

    def run(job) -> bool:
        split, seed = job
        gpu = slots.get()
        try:
            command = [
                *py, "emulator_bench.run_one", *common[:4], "--split_groups", split.group,
                "--split_name", split.name, "--seeds", str(seed),
            ]  # fmt: skip
            if args.amp:
                command.append("--amp")
            if args.epochs is not None:
                command += ["--epochs", str(args.epochs)]
            log = logs / f"{split.group}__{split.name}__seed_{seed}.log"
            ok = run_logged(command, log, {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}) == 0
            print(f"{'done' if ok else 'FAILED'} {split.id} seed {seed} (gpu {gpu}); log {log}", flush=True)
            return ok
        finally:
            slots.put(gpu)

    with ThreadPoolExecutor(max_workers=len(args.gpus) * args.runs_per_gpu) as pool:
        results = list(pool.map(run, jobs))
    if not all(results):
        sys.exit(1)
    subprocess.run([*py, "emulator_bench.results", "--out_root", args.out_root], cwd=BASELINE_ROOT, check=True)


if __name__ == "__main__":
    main()
