from src.monitor_reference import load_reference
import os
import argparse
import hashlib
import json
import sqlite3

from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset

from src.preprocess import FEATURES, preprocess


DATABASE = Path(os.getenv("TAXI_MONITOR_DB", "data/api_predictions.sqlite3"))
REFERENCE = Path(os.getenv("TAXI_MONITOR_REFERENCE", "data/green_tripdata_2024-01.parquet"))
CALIBRATION = Path(
    "monitoring/reports/drift_calibration_5000.json"
)
OUTPUT = Path(os.getenv("TAXI_MONITOR_OUTPUT", "monitoring/reports"))

SAMPLE_SIZE = 5000


def read_api_inputs():
    if not DATABASE.is_file():
        return pd.DataFrame(columns=FEATURES)

    with closing(
        sqlite3.connect(
            f"file:{DATABASE.resolve().as_posix()}?mode=ro",
            uri=True,
            timeout=10,
        )
    ) as connection:
        frame = pd.read_sql_query(
            """
            SELECT
                PULocationID,
                DOLocationID,
                trip_distance
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
            """,
            connection,
            params=(SAMPLE_SIZE,),
        )

    if not frame.empty:
        frame["trip_distance"] = pd.to_numeric(
            frame["trip_distance"], errors="raise"
        )

        for column in ("PULocationID", "DOLocationID"):
            frame[column] = frame[column].astype(str)

    return frame[FEATURES].copy()


def load_thresholds():
    if not CALIBRATION.is_file():
        raise FileNotFoundError(
            f"Missing calibration: {CALIBRATION}"
        )

    config = json.loads(
        CALIBRATION.read_text(encoding="utf-8-sig")
    )

    if config["sample_size_each_group"] != SAMPLE_SIZE:
        raise ValueError("Calibration sample size mismatch")

    thresholds = config["candidate_thresholds"]

    for column in FEATURES:
        if column not in thresholds:
            raise ValueError(
                f"Threshold missing: {column}"
            )

    return {
        column: float(thresholds[column])
        for column in FEATURES
    }


def make_dataset(frame):
    definition = DataDefinition(
        numerical_columns=["trip_distance"],
        categorical_columns=[
            "PULocationID",
            "DOLocationID",
        ],
    )

    return Dataset.from_pandas(
        frame,
        data_definition=definition,
    )


def run_monitor():
    print("=== LIVE API DRIFT MONITOR ===", flush=True)

    current = read_api_inputs()
    count = len(current)

    print(f"AVAILABLE API ROWS: {count}")
    print(f"REQUIRED API ROWS: {SAMPLE_SIZE}")

    if count < SAMPLE_SIZE:
        print("STATUS: INSUFFICIENT_DATA")
        print(
            "No drift conclusion. "
            "Collect more real API inputs."
        )
        return False

    if not REFERENCE.is_file():
        raise FileNotFoundError(REFERENCE)

    thresholds = load_thresholds()

    prepared = load_reference(REFERENCE)

    if len(prepared) < SAMPLE_SIZE:
        raise ValueError("Insufficient reference rows")

    reference = prepared.sample(
        n=SAMPLE_SIZE,
        random_state=20261012,
        replace=False,
    )

    if current[FEATURES].isna().any().any():
        raise ValueError("Missing API feature values")

    if not current["trip_distance"].between(
        0.1, 100
    ).all():
        raise ValueError("Invalid API trip distances")

    preset = DataDriftPreset(
        columns=FEATURES,
        drift_share=0.5,
        per_column_method={
            "PULocationID": "jensenshannon",
            "DOLocationID": "jensenshannon",
            "trip_distance": "wasserstein",
        },
        per_column_threshold=thresholds,
    )

    report = Report([preset])

    snapshot = report.run(
        make_dataset(current),
        make_dataset(reference),
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)

    html_path = OUTPUT / "taxi_live_api_drift.html"
    json_path = OUTPUT / "taxi_live_api_drift.json"
    metadata_path = OUTPUT / "taxi_live_api_drift_metadata.json"

    snapshot.save_html(str(html_path))

    result_json = snapshot.json()
    parsed = json.loads(result_json)

    if not isinstance(parsed, dict) or not parsed.get("metrics"):
        raise RuntimeError(
            "Evidently output missing metrics"
        )

    json_path.write_text(
        result_json,
        encoding="utf-8",
    )

    metadata = {
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "source": "SQLite API prediction inputs",
        "reference": str(REFERENCE),
        "api_sample_size": count,
        "reference_sample_size": len(reference),
        "methods": {
            "PULocationID": "jensenshannon",
            "DOLocationID": "jensenshannon",
            "trip_distance": "wasserstein",
        },
        "candidate_thresholds": thresholds,
        "calibration_sha256": hashlib.sha256(
            CALIBRATION.read_bytes()
        ).hexdigest(),
        "scope": "historical replay",
        "limitations": [
            "No ground-truth model quality measurement",
            "Report is generated on demand",
            "Latest API requests are not necessarily independent",
            "Calibration thresholds remain provisional",
        ],
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    for path in (html_path, json_path, metadata_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Invalid output: {path}")

    print(f"HTML REPORT: {html_path}")
    print(f"JSON REPORT: {json_path}")
    print(f"METADATA: {metadata_path}")
    print("REPORT EXPORT: PASS")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--require-report",
        action="store_true",
    )

    args = parser.parse_args()

    created = run_monitor()

    if args.require_report and not created:
        raise SystemExit(2)