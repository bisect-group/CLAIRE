import argparse
import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import List, Sequence

from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table

from emulator_bench.cache_features import cache_reactions, collect_reactions
from emulator_bench.common import (
    DEFAULT_BASE_DIR,
    DEFAULT_DEVICE,
    DEFAULT_EMBEDDINGS_DIR,
    DEFAULT_LEVELS,
    DEFAULT_RESULTS_DIRNAME,
    DEFAULT_SOURCE_DATA_DIR,
    REPO_ROOT,
    completed_run,
    configure_visible_gpus,
    console,
    copy_dataset_splits,
    discover_split_jobs,
    render_jobs_table,
    subprocess_env,
)
from emulator_bench.dataset_adapter import validate_jobs


def parse_gpu_ids(gpus: str | None) -> list[str]:
    if not gpus:
        return []
    return [gpu.strip() for gpu in gpus.replace(",", " ").split() if gpu.strip()]


def run_command(cmd: List[str], gpus: str | None = None) -> None:
    console.print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), env=subprocess_env(gpus))


def make_train_command(
    args: argparse.Namespace,
    job,
    level: str,
    seed: int,
    out_dir: Path,
    gpu: str | None,
    progress_mode: str = "live",
) -> list[str]:
    run_name = f"{job.split_group}/{job.split_name} {level} seed={seed}"
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
        "--embeddings_dir",
        args.embeddings_dir,
        "--device",
        args.device,
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
        "--seed",
        str(seed),
        "--topk",
        str(args.topk),
        "--run_name",
        run_name,
        "--progress_mode",
        progress_mode,
        "--progress_interval",
        str(args.progress_interval),
    ]
    if args.allow_provisional:
        cmd.append("--allow_provisional")
    if args.overwrite:
        cmd.append("--overwrite")
    if gpu:
        cmd.extend(["--gpus", gpu])
    return cmd


def run_parallel(queued: Sequence[tuple], args: argparse.Namespace, gpu_ids: Sequence[str]) -> None:
    slots = list(gpu_ids) if gpu_ids else [None]
    slots = slots * max(1, args.parallel_runs_per_gpu)
    running: list[dict] = []
    next_slot = 0
    next_job = 0
    messages: queue.Queue[tuple[str, str]] = queue.Queue()

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )

    def reader(process: subprocess.Popen, description: str) -> None:
        assert process.stdout is not None
        for line in process.stdout:
            messages.put((description, line.rstrip()))

    def launch_one(progress_obj: Progress) -> bool:
        nonlocal next_job, next_slot
        if next_job >= len(queued) or len(running) >= len(slots):
            return False
        job, level, seed, out_dir = queued[next_job]
        next_job += 1
        gpu = slots[next_slot % len(slots)]
        next_slot += 1
        cmd = make_train_command(args, job, level, seed, out_dir, gpu, progress_mode="json")
        description = f"{job.split_group}/{job.split_name} {level} seed={seed} gpu={gpu or 'visible'}"
        env = subprocess_env(gpu)
        env["PYTHONUNBUFFERED"] = "1"
        task_id = progress_obj.add_task(description, total=args.epochs, visible=True)
        process = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        thread = threading.Thread(target=reader, args=(process, description), daemon=True)
        thread.start()
        running.append({"process": process, "description": description, "task_id": task_id, "thread": thread})
        return True

    with progress:
        while len(running) < len(slots) and launch_one(progress):
            pass
        while running:
            try:
                description, line = messages.get(timeout=0.2)
            except queue.Empty:
                description = ""
                line = ""
            if line:
                try:
                    payload = json.loads(line).get("claire_progress")
                except json.JSONDecodeError:
                    payload = None
                if payload:
                    for item in running:
                        if item["description"] == description:
                            total = int(payload.get("total") or args.epochs)
                            completed = int(payload.get("completed") or 0)
                            epoch = payload.get("epoch")
                            if payload.get("event") == "start":
                                total = int(payload.get("total") or args.epochs)
                                completed = 0
                            progress.update(
                                item["task_id"],
                                total=total,
                                completed=completed,
                                description=f"{description} epoch {epoch or ''}".rstrip(),
                            )
                            break
                else:
                    progress.console.print(line)

            for item in list(running):
                code = item["process"].poll()
                if code is None:
                    continue
                item["thread"].join(timeout=0.2)
                if code != 0:
                    for other in running:
                        if other is not item:
                            other["process"].terminate()
                    raise SystemExit(f"{item['description']} failed with exit code {code}")
                progress.remove_task(item["task_id"])
                running.remove(item)
                launch_one(progress)
            time.sleep(0.05)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", default=str(DEFAULT_SOURCE_DATA_DIR))
    parser.add_argument("--base_dir", default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--embeddings_dir", default=str(DEFAULT_EMBEDDINGS_DIR))
    parser.add_argument("--results_dirname", default=DEFAULT_RESULTS_DIRNAME)
    parser.add_argument("--levels", nargs="*", default=DEFAULT_LEVELS, choices=DEFAULT_LEVELS)
    parser.add_argument("--split_groups", nargs="*")
    parser.add_argument("--gpus")
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=6000)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--parallel_runs_per_gpu", type=int, default=1)
    parser.add_argument("--progress_interval", type=int, default=10)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--cache_batch_size", type=int, default=128)
    parser.add_argument("--cache_rxnfp_batch_size", type=int, default=128)
    parser.add_argument("--cache_drfp_workers", type=int, default=1)
    parser.add_argument("--cache_save_workers", type=int, default=2)
    parser.add_argument("--skip_copy", action="store_true")
    parser.add_argument("--skip_cache", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow_provisional", action="store_true")
    args = parser.parse_args()
    if args.parallel_runs_per_gpu < 1:
        raise SystemExit("--parallel_runs_per_gpu must be >= 1")
    if args.progress_interval < 1:
        raise SystemExit("--progress_interval must be >= 1")
    configure_visible_gpus(args.gpus)
    gpu_ids = parse_gpu_ids(args.gpus)
    seeds = args.seeds if args.seeds else [args.seed]

    base_dir = Path(args.base_dir)
    if not args.skip_copy:
        copied = copy_dataset_splits(Path(args.source_dir), base_dir, overwrite=args.overwrite)
        console.print(f"Copied {len(copied)} files into {base_dir}")
    validate_jobs(base_dir, allow_provisional=args.allow_provisional)
    jobs = discover_split_jobs(base_dir, split_groups=args.split_groups)
    render_jobs_table(jobs)
    if not args.skip_cache:
        reactions = collect_reactions(base_dir, split_groups=args.split_groups)
        cache_reactions(
            reactions,
            Path(args.embeddings_dir),
            batch_size=args.cache_batch_size,
            rxnfp_batch_size=args.cache_rxnfp_batch_size,
            device_name=args.device,
            drfp_workers=args.cache_drfp_workers,
            save_workers=args.cache_save_workers,
            overwrite=args.overwrite,
        )

    table = Table(title="CLAIRE benchmark queue")
    table.add_column("group")
    table.add_column("split")
    table.add_column("level")
    table.add_column("seed")
    table.add_column("status")
    queued = []
    for job in jobs:
        for level in args.levels:
            for seed in seeds:
                out_dir = job.root_dir / args.results_dirname / level / f"seed_{seed}"
                status = "overwrite" if args.overwrite else ("skip" if completed_run(out_dir / "test") else "run")
                table.add_row(job.split_group, job.split_name, level, str(seed), status)
                if status != "skip":
                    queued.append((job, level, seed, out_dir))
    console.print(table)

    if not queued:
        return
    if len(gpu_ids) * args.parallel_runs_per_gpu > 1:
        run_parallel(queued, args, gpu_ids)
        return
    for job, level, seed, out_dir in queued:
        gpu = gpu_ids[0] if gpu_ids else None
        run_command(make_train_command(args, job, level, seed, out_dir, gpu), gpus=gpu)


if __name__ == "__main__":
    main()
