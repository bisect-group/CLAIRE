import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from emulator_bench.aggregate_results import aggregate, discover_runs, inspect_record, main
from emulator_bench.evaluate import compute_metrics


class AggregateResultsTest(unittest.TestCase):
    def write_run(self, base, split, level, seed, partition="test", valid=True, corrupt_metrics=False):
        directory = base / split / "claire_results" / level / f"seed_{seed}" / partition
        directory.mkdir(parents=True)
        frame = pd.DataFrame({"true_label": ["1", "2"], "pred_label": ["1", "1"], "top1": ["1", "1"], "top2": ["2", "2"]})
        frame.to_csv(directory / "predictions.csv", index=False)
        if corrupt_metrics:
            (directory / "metrics.json").write_text("{")
        elif valid:
            metrics = compute_metrics(["1", "2"], [["1", "2"], ["1", "2"]], ["1", "2"], "l1")
            (directory / "metrics.json").write_text(json.dumps(metrics))
        return directory

    def test_discovers_layouts_and_aggregates_sample_std(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self.write_run(base, "random_splits", "l1", 1)
            self.write_run(base, "uniprot_time_splits", "l1", 2)
            self.write_run(base, "reaction_drfp_tanimoto_splits/threshold_0.15", "l1", 3)
            records = [inspect_record(record) for record in discover_runs(base, "claire_results", None)]
            self.assertEqual(len(records), 6)  # test and missing val for each seed
            summaries, _ = aggregate(records)
            self.assertEqual(sum(row["runs_completed"] for row in summaries), 3)
            self.assertEqual(sum(row["runs_missing"] for row in summaries), 3)
            self.assertEqual({row["partition"] for row in summaries}, {"test", "val"})

    def test_marks_missing_malformed_and_metric_mismatch_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self.write_run(base, "random_splits", "l1", 1)
            corrupt = self.write_run(base, "random_splits", "l1", 2, corrupt_metrics=True)
            mismatch = self.write_run(base, "random_splits", "l1", 3)
            metrics_path = mismatch / "metrics.json"
            payload = json.loads(metrics_path.read_text())
            payload["accuracy"] = 0.0
            metrics_path.write_text(json.dumps(payload))
            records = [inspect_record(record) for record in discover_runs(base, "claire_results", None)]
            statuses = {record["seed"]: record["status"] for record in records if record["partition"] == "test"}
            self.assertEqual(statuses, {1: "valid", 2: "invalid", 3: "invalid"})

    def test_strict_mode_fails_after_writing_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "base"
            output = Path(tmp) / "out"
            self.write_run(base, "random_splits", "l1", 1)
            with patch("sys.argv", ["aggregate_results", "--base-dir", str(base), "--output-dir", str(output), "--strict"]):
                with self.assertRaises(SystemExit):
                    main()
            self.assertTrue((output / "diagnostics.json").is_file())


if __name__ == "__main__":
    unittest.main()
