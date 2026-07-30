"""Aggregate completed CLAIRE EMULaToR benchmark evaluations.

The command deliberately treats every seed/partition independently.  It never
pools predictions across seeds: benchmark summaries are the mean and sample
standard deviation of the per-seed metrics.
"""

import argparse
import csv
import io
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd
from rich.table import Table

from emulator_bench.common import DEFAULT_BASE_DIR, DEFAULT_RESULTS_DIRNAME, atomic_write_text, console, load_json
from emulator_bench.evaluate import compute_metrics


PARTITIONS = ("val", "test")
METRIC_TOLERANCE = 1e-9


def csv_text(rows: Iterable[Dict[str, Any]], fieldnames: List[str]) -> str:
    handle = io.StringIO()
    writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue()


def numeric_metrics(path: Path) -> Dict[str, float]:
    payload = load_json(path)
    if not isinstance(payload, dict) or not payload:
        raise ValueError("metrics.json must contain a non-empty object")
    metrics = {}
    for key, value in payload.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"metric {key!r} is not a finite number")
        metrics[str(key)] = float(value)
    return metrics


def recompute_metrics(predictions_path: Path, level: str) -> Tuple[Dict[str, float], int]:
    # EC labels such as L1 ``3`` must remain strings.  Pandas otherwise turns
    # them into floats (``3.0``), which makes a correct saved metric look bad.
    frame = pd.read_csv(predictions_path, dtype=str, keep_default_na=False)
    required = {"true_label", "pred_label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"predictions.csv is missing columns: {', '.join(sorted(missing))}")
    ranked = []
    for _, row in frame.iterrows():
        labels = []
        for column in sorted((name for name in frame.columns if name.startswith("top") and name[3:].isdigit()), key=lambda name: int(name[3:])):
            value = row[column]
            if value:
                labels.append(str(value))
        if not labels and row["pred_label"]:
            labels.append(str(row["pred_label"]))
        ranked.append(labels)
    truth = frame["true_label"].tolist()
    return compute_metrics(truth, ranked, source_true_ec=truth, prediction_level=level), len(frame)


def validate_metrics(saved: Dict[str, float], predictions_path: Path, level: str) -> Tuple[int, List[str]]:
    recomputed, row_count = recompute_metrics(predictions_path, level)
    mismatches = []
    for name, value in saved.items():
        if name not in recomputed:
            continue
        if not math.isclose(value, recomputed[name], rel_tol=METRIC_TOLERANCE, abs_tol=METRIC_TOLERANCE):
            mismatches.append(f"{name}: saved={value:.12g}, recomputed={recomputed[name]:.12g}")
    return row_count, mismatches


def discover_runs(base_dir: Path, results_dirname: str, split_groups: List[str] | None) -> List[Dict[str, Any]]:
    records = []
    for results_dir in sorted(path for path in base_dir.rglob(results_dirname) if path.is_dir()):
        split_root = results_dir.parent
        relative = split_root.relative_to(base_dir)
        if not relative.parts:
            continue
        split_group = relative.parts[0]
        if split_groups and split_group not in split_groups:
            continue
        split_name = relative.as_posix()
        for level_dir in sorted(path for path in results_dir.iterdir() if path.is_dir() and path.name in {"l1", "l2", "l3", "l4"}):
            for seed_dir in sorted(path for path in level_dir.iterdir() if path.is_dir() and path.name.startswith("seed_")):
                seed_text = seed_dir.name.removeprefix("seed_")
                try:
                    seed = int(seed_text)
                except ValueError:
                    seed = seed_text
                for partition in PARTITIONS:
                    records.append({
                        "split_group": split_group,
                        "split_name": split_name,
                        "level": level_dir.name,
                        "seed": seed,
                        "partition": partition,
                        "run_dir": seed_dir,
                    })
    return records


def inspect_record(record: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(record)
    partition_dir = result["run_dir"] / result["partition"]
    metrics_path = partition_dir / "metrics.json"
    predictions_path = partition_dir / "predictions.csv"
    result.update({"run_dir": str(result["run_dir"]), "row_count": None, "status": "valid", "error": "", "metrics": {}})
    missing = [str(path.name) for path in (metrics_path, predictions_path) if not path.is_file()]
    if missing:
        result.update(status="missing", error=f"missing required artifact(s): {', '.join(missing)}")
        return result
    try:
        metrics = numeric_metrics(metrics_path)
        row_count, mismatches = validate_metrics(metrics, predictions_path, result["level"])
        if mismatches:
            result.update(status="invalid", error="; ".join(mismatches))
        result.update(metrics=metrics, row_count=row_count)
    except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        result.update(status="invalid", error=str(error))
    return result


def aggregate(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    groups: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record["split_group"], record["split_name"], record["level"], record["partition"])].append(record)
    summary_rows = []
    summary_json = []
    for identity, members in sorted(groups.items()):
        valid = [member for member in members if member["status"] == "valid"]
        metric_names = sorted({name for member in valid for name in member["metrics"]})
        summary = {
            "split_group": identity[0], "split_name": identity[1], "level": identity[2], "partition": identity[3],
            "runs_discovered": len(members), "runs_completed": len(valid),
            "runs_missing": sum(member["status"] == "missing" for member in members),
            "runs_invalid": sum(member["status"] == "invalid" for member in members),
            "completed_seeds": ",".join(str(member["seed"]) for member in valid),
        }
        metrics_json = {}
        for name in metric_names:
            values = [member["metrics"][name] for member in valid if name in member["metrics"]]
            metric_summary = {"count": len(values), "mean": mean(values) if values else None, "sample_std": stdev(values) if len(values) > 1 else 0.0 if values else None}
            metrics_json[name] = metric_summary
            summary[f"{name}_mean"] = metric_summary["mean"]
            summary[f"{name}_sample_std"] = metric_summary["sample_std"]
        summary_rows.append(summary)
        summary_json.append({**{key: value for key, value in summary.items() if not key.endswith("_mean") and not key.endswith("_sample_std")}, "metrics": metrics_json})
    return summary_rows, summary_json


def write_reports(output_dir: Path, records: List[Dict[str, Any]], summary_rows: List[Dict[str, Any]], summary_json: List[Dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_names = sorted({name for record in records for name in record["metrics"]})
    per_run_rows = [{key: value for key, value in record.items() if key not in {"metrics"}} | record["metrics"] for record in records]
    run_fields = ["split_group", "split_name", "level", "seed", "partition", "run_dir", "status", "row_count", "error"] + metric_names
    summary_metrics = sorted({key for row in summary_rows for key in row if key.endswith("_mean") or key.endswith("_sample_std")})
    summary_fields = ["split_group", "split_name", "level", "partition", "runs_discovered", "runs_completed", "runs_missing", "runs_invalid", "completed_seeds"] + summary_metrics
    diagnostics = [{key: value for key, value in record.items() if key != "metrics"} for record in records if record["status"] != "valid"]
    atomic_write_text(output_dir / "per_run_metrics.csv", csv_text(per_run_rows, run_fields))
    atomic_write_text(output_dir / "summary_metrics.csv", csv_text(summary_rows, summary_fields))
    atomic_write_text(output_dir / "summary_metrics.json", __import__("json").dumps(summary_json, indent=2, sort_keys=True) + "\n")
    atomic_write_text(output_dir / "diagnostics.json", __import__("json").dumps(diagnostics, indent=2, sort_keys=True) + "\n")


def render_summary(summary_rows: List[Dict[str, Any]], output_dir: Path) -> None:
    table = Table(title="CLAIRE aggregated metrics")
    for column in ("split", "level", "partition", "completed", "missing", "invalid"):
        table.add_column(column)
    for row in summary_rows:
        table.add_row(row["split_name"], row["level"], row["partition"], str(row["runs_completed"]), str(row["runs_missing"]), str(row["runs_invalid"]))
    console.print(table)
    console.print(f"Reports written to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate CLAIRE benchmark metrics across seeds.")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--results-dirname", default=DEFAULT_RESULTS_DIRNAME)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--split-groups", nargs="*")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero if any run is missing or invalid after writing reports.")
    args = parser.parse_args()
    output_dir = args.output_dir or args.base_dir / "aggregated_metrics"
    records = [inspect_record(record) for record in discover_runs(args.base_dir, args.results_dirname, args.split_groups)]
    if not records:
        parser.error(f"No {args.results_dirname!r} seed runs found under {args.base_dir}")
    summary_rows, summary_json = aggregate(records)
    write_reports(output_dir, records, summary_rows, summary_json)
    render_summary(summary_rows, output_dir)
    if args.strict and any(record["status"] != "valid" for record in records):
        raise SystemExit("Strict aggregation failed: one or more runs are missing or invalid; see diagnostics.json.")


if __name__ == "__main__":
    main()
