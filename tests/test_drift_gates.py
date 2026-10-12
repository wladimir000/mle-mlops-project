import copy
import json
import math
import unittest
from pathlib import Path


ROOT = Path("monitoring/reports")
E2E = ROOT / "e2e_independent"

COLUMNS = {
    "PULocationID": "jensenshannon",
    "DOLocationID": "jensenshannon",
    "trip_distance": "wasserstein",
}

EXPECTED_DRIFT = {
    "baseline": {
        "PULocationID": False,
        "DOLocationID": False,
        "trip_distance": False,
    },
    "shifted": {
        "PULocationID": False,
        "DOLocationID": True,
        "trip_distance": True,
    },
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, label):
    require(
        type(value) in (int, float)
        and math.isfinite(value),
        f"Invalid number: {label}",
    )
    return float(value)


def load_json(path):
    require(path.is_file(), f"Missing: {path}")

    with path.open(
        encoding="utf-8-sig"
    ) as stream:
        return json.load(stream)


def calibration_thresholds():
    data = load_json(
        ROOT / "drift_calibration_5000.json"
    )

    require(
        data.get("sample_size_each_group") == 5000,
        "Calibration sample size mismatch",
    )

    values = data.get("candidate_thresholds")
    require(
        isinstance(values, dict),
        "Calibration thresholds missing",
    )
    require(
        set(values) == set(COLUMNS),
        "Calibration feature mismatch",
    )

    result = {}

    for column in COLUMNS:
        threshold = number(
            values[column],
            f"calibration {column}",
        )
        require(
            0 < threshold < 1,
            f"Invalid calibration threshold: {column}",
        )
        result[column] = threshold

    return result


def same_float(actual, expected, label):
    actual = number(actual, label)
    expected = number(expected, label + " expected")

    require(
        math.isclose(
            actual,
            expected,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ),
        f"{label}: {actual} != {expected}",
    )


def evaluate(report, thresholds, scenario):
    require(
        scenario in EXPECTED_DRIFT,
        "Unknown scenario",
    )
    require(
        isinstance(report, dict),
        "Invalid report root",
    )

    metrics = report.get("metrics")

    require(
        isinstance(metrics, list)
        and len(metrics) == 4,
        "Expected exactly four metrics",
    )

    require(
        all(isinstance(m, dict) for m in metrics),
        "Invalid metric object",
    )

    counts = [
        m for m in metrics
        if m.get("config", {}).get("type")
        == "evidently:metric_v2:DriftedColumnsCount"
    ]

    require(
        len(counts) == 1,
        "Expected one DriftedColumnsCount",
    )

    summary = counts[0]
    config = summary["config"]
    value = summary.get("value")

    require(
        isinstance(value, dict),
        "Invalid summary value",
    )

    require(
        config.get("columns") == list(COLUMNS),
        "Unexpected column order",
    )

    same_float(
        config.get("drift_share"),
        0.5,
        "Dataset drift share threshold",
    )

    require(
        config.get("per_column_method") == COLUMNS,
        "Summary method configuration mismatch",
    )

    configured_thresholds = config.get(
        "per_column_threshold"
    )

    require(
        isinstance(configured_thresholds, dict)
        and set(configured_thresholds) == set(COLUMNS),
        "Summary threshold configuration invalid",
    )

    for column in COLUMNS:
        same_float(
            configured_thresholds[column],
            thresholds[column],
            f"Summary threshold {column}",
        )

    individual = [
        m for m in metrics
        if m.get("config", {}).get("type")
        == "evidently:metric_v2:ValueDrift"
    ]

    require(
        len(individual) == 3,
        "Expected three ValueDrift metrics",
    )

    seen = {}
    decisions = {}

    for metric in individual:
        cfg = metric["config"]
        column = cfg.get("column")

        require(
            column in COLUMNS,
            f"Unexpected feature: {column}",
        )
        require(
            column not in seen,
            f"Duplicate feature: {column}",
        )
        require(
            cfg.get("method") == COLUMNS[column],
            f"Wrong method: {column}",
        )

        threshold = thresholds[column]

        same_float(
            cfg.get("threshold"),
            threshold,
            f"ValueDrift threshold {column}",
        )

        score = number(
            metric.get("value"),
            f"Drift score {column}",
        )

        require(
            score >= 0,
            f"Negative drift score: {column}",
        )

        seen[column] = score
        decisions[column] = score >= threshold

    require(
        set(seen) == set(COLUMNS),
        "Missing feature metric",
    )

    expected = EXPECTED_DRIFT[scenario]

    require(
        decisions == expected,
        (
            f"{scenario} drift decisions differ: "
            f"{decisions}"
        ),
    )

    count = sum(decisions.values())
    share = count / len(COLUMNS)

    same_float(
        value.get("count"),
        count,
        "DriftedColumnsCount.value.count",
    )

    same_float(
        value.get("share"),
        share,
        "DriftedColumnsCount.value.share",
    )

    dataset_drift = share >= 0.5

    require(
        dataset_drift == (scenario == "shifted"),
        "Dataset-level drift decision mismatch",
    )

    return {
        "scenario": scenario,
        "scores": seen,
        "decisions": decisions,
        "drift_count": count,
        "drift_share": share,
        "dataset_drift": dataset_drift,
    }


class DriftGateTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.thresholds = calibration_thresholds()
        cls.reports = {
            name: load_json(
                E2E / name / "taxi_live_api_drift.json"
            )
            for name in EXPECTED_DRIFT
        }

    def test_g01_baseline(self):
        result = evaluate(
            self.reports["baseline"],
            self.thresholds,
            "baseline",
        )
        self.assertEqual(result["drift_count"], 0)
        self.assertFalse(result["dataset_drift"])
        print("G01 BASELINE: PASS")

    def test_g02_shifted(self):
        result = evaluate(
            self.reports["shifted"],
            self.thresholds,
            "shifted",
        )
        self.assertEqual(result["drift_count"], 2)
        self.assertTrue(result["dataset_drift"])
        print("G02 SYNTHETIC DRIFT: PASS")

    def test_g03_missing_metric_rejected(self):
        report = copy.deepcopy(
            self.reports["baseline"]
        )
        report["metrics"].pop()

        with self.assertRaises(ValueError):
            evaluate(
                report,
                self.thresholds,
                "baseline",
            )

        print("G03 MISSING METRIC: REJECTED")

    def test_g04_wrong_threshold_rejected(self):
        report = copy.deepcopy(
            self.reports["baseline"]
        )

        metric = next(
            m for m in report["metrics"]
            if m["config"]["type"]
            == "evidently:metric_v2:ValueDrift"
        )

        metric["config"]["threshold"] = 999.0

        with self.assertRaises(ValueError):
            evaluate(
                report,
                self.thresholds,
                "baseline",
            )

        print("G04 WRONG THRESHOLD: REJECTED")

    def test_g05_corrupt_summary_rejected(self):
        report = copy.deepcopy(
            self.reports["shifted"]
        )

        report["metrics"][0]["value"]["count"] = 0

        with self.assertRaises(ValueError):
            evaluate(
                report,
                self.thresholds,
                "shifted",
            )

        print("G05 CORRUPT SUMMARY: REJECTED")

    def test_g06_duplicate_feature_rejected(self):
        report = copy.deepcopy(
            self.reports["baseline"]
        )

        report["metrics"][2]["config"]["column"] = (
            "PULocationID"
        )

        with self.assertRaises(ValueError):
            evaluate(
                report,
                self.thresholds,
                "baseline",
            )

        print("G06 DUPLICATE FEATURE: REJECTED")

    def test_g07_invalid_score_rejected(self):
        report = copy.deepcopy(
            self.reports["baseline"]
        )

        report["metrics"][1]["value"] = float("nan")

        with self.assertRaises(ValueError):
            evaluate(
                report,
                self.thresholds,
                "baseline",
            )

        print("G07 INVALID SCORE: REJECTED")

    def test_g08_wrong_method_rejected(self):
        report = copy.deepcopy(
            self.reports["baseline"]
        )

        report["metrics"][1]["config"]["method"] = (
            "ks"
        )

        with self.assertRaises(ValueError):
            evaluate(
                report,
                self.thresholds,
                "baseline",
            )

        print("G08 WRONG METHOD: REJECTED")


if __name__ == "__main__":
    unittest.main(verbosity=2)