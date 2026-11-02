"""EC level accuracy of ranked EC predictions, and the summary CSVs of all runs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import DEFAULT_OUT_ROOT, FULL_EC, TASK, atomic_write

EC_LEVELS = (1, 2, 3, 4)
KS = (1, 3, 5, 10, 20, 30, 40, 50)
MISS_EC = "0.0.0.0"


def matched_levels(true_ec: str, pred_ec: str) -> int:
    matched = 0
    for pred_part, true_part in zip(pred_ec.split("."), true_ec.split(".")):
        if pred_part != true_part:
            break
        matched += 1
    return matched


def ec_level_accuracy(frame: pd.DataFrame, drop_preliminary: bool) -> dict:
    """Per test reaction and k: share of its true ECs that the top-k ECs match to level n (best prefix match), mean x 100.

    `frame` has one row per test row with `query`, `ec_number` and `top_ec` (best first). With `drop_preliminary`,
    preliminary ECs (such as 3.4.23.B9) leave the truth and the ranked lists, and reactions without a full EC are not scored.
    """
    if drop_preliminary:
        frame = frame[frame["ec_number"].str.fullmatch(FULL_EC)]
    true_ecs, ranked = [], []
    for _, rows in frame.groupby("query", sort=False):
        top = list(rows["top_ec"].iloc[0])
        ranked.append([ec for ec in top if re.fullmatch(FULL_EC, ec)] if drop_preliminary else top)
        true_ecs.append(sorted(set(rows["ec_number"])))
    result = {}
    for k in KS:
        scores = np.zeros((len(true_ecs), len(EC_LEVELS)))
        for i, (trues, ranking) in enumerate(zip(true_ecs, ranked)):
            top = ranking[:k] or [MISS_EC]
            best = np.array([max(matched_levels(t, p) for p in top) for t in trues])
            scores[i] = [(best >= n).mean() for n in EC_LEVELS]
        means = scores.mean(axis=0) * 100.0
        result[f"k={k}"] = {f"level_{n}_accuracy": float(m) for n, m in zip(EC_LEVELS, means)}
    return {**result, "n_queries": len(true_ecs), "n_rows": len(frame)}


def aggregate(out_root: str | Path) -> None:
    """Rebuild summary_by_seed.csv and summary.csv from the saved predictions of every finished run."""
    root = Path(out_root) / TASK
    rows = []
    for path in sorted(root.glob("**/seed_*/test_predictions.parquet")):
        frame = pd.read_parquet(path)
        accuracy = ec_level_accuracy(frame, drop_preliminary=True)
        folder = path.parent
        for k in KS:
            rows.append(
                {
                    "split_group": "/".join(folder.parent.parent.relative_to(root).parts),
                    "split_name": folder.parent.name,
                    "seed": int(folder.name.split("_")[1]),
                    "k": k,
                    "queries": accuracy["n_queries"],
                    **{f"level_{n}": accuracy[f"k={k}"][f"level_{n}_accuracy"] for n in EC_LEVELS},
                }
            )
    if not rows:
        raise FileNotFoundError(f"No test_predictions.parquet under {root}")
    by_seed = pd.DataFrame(rows)
    keys = ["split_group", "split_name", "k"]
    summary = by_seed.groupby(keys)[[f"level_{n}" for n in EC_LEVELS]].agg(["mean", "std"])
    summary.columns = [f"{level}_{stat}" for level, stat in summary.columns]
    summary["seeds"] = by_seed.groupby(keys)["seed"].nunique()
    atomic_write(root / "summary_by_seed.csv", lambda tmp: by_seed.to_csv(tmp, index=False))
    atomic_write(root / "summary.csv", lambda tmp: summary.reset_index().to_csv(tmp, index=False))
    print(f"[results] wrote {root / 'summary.csv'} ({len(by_seed)} rows)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rebuild the CLAIRE retrieval summaries from saved predictions")
    parser.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    aggregate(parser.parse_args().out_root)
