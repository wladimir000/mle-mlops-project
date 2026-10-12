import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from src.preprocess import FEATURES, preprocess


DATA_PATH = Path(
    "data/green_tripdata_2024-01.parquet"
)

OUTPUT_DIR = Path("monitoring/reports")

NUMERIC_COLUMNS = ["trip_distance"]
CATEGORICAL_COLUMNS = [
    "PULocationID",
    "DOLocationID",
]


def load_reference_data(sample_size: int):
    """Load and prepare distinct monitoring samples."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    raw = pd.read_parquet(DATA_PATH)
    prepared = preprocess(raw)

    if len(prepared) < sample_size * 2:
        raise ValueError(
            "Not enough rows for independent samples"
        )

    # Use the actual model input features.
    prepared = prepared[FEATURES].copy()

    sampled = prepared.sample(
        n=sample_size * 2,
        random_state=42,
        replace=False,
    )

    reference = sampled.iloc[:sample_size].copy()
    current = sampled.iloc[sample_size:].copy()

    return (
        reference.reset_index(drop=True),
        current.reset_index(drop=True),
    )


def make_dataset(frame: pd.DataFrame):
    """Attach explicit Evidently feature types."""

    definition = DataDefinition(
        numerical_columns=NUMERIC_COLUMNS,
        categorical_columns=CATEGORICAL_COLUMNS,
    )

    return Dataset.from_pandas(
        frame,
        data_definition=definition,
    )


def generate_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    report_name: str,
    metadata: dict,
):
    """Generate and persist an Evidently drift report."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    reference_dataset = make_dataset(reference)
    current_dataset = make_dataset(current)

    report = Report([
        DataDriftPreset(),
    ])

    snapshot = report.run(
        current_dataset,
        reference_dataset,
    )

    html_path = OUTPUT_DIR / f"{report_name}.html"
    json_path = OUTPUT_DIR / f"{report_name}.json"
    metadata_path = (
        OUTPUT_DIR / f"{report_name}_metadata.json"
    )

    snapshot.save_html(str(html_path))

    json_path.write_text(
        snapshot.json(),
        encoding="utf-8",
    )

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    # Verify saved output
    if not html_path.is_file():
        raise RuntimeError("HTML report missing")

    if html_path.stat().st_size == 0:
        raise RuntimeError("HTML report is empty")

    result = json.loads(
        json_path.read_text(encoding="utf-8")
    )

    if not isinstance(result, dict):
        raise RuntimeError("Invalid report JSON")

    print(f"REPORT: {report_name}", flush=True)
    print(f"REFERENCE ROWS: {len(reference)}")
    print(f"CURRENT ROWS: {len(current)}")
    print(f"HTML: {html_path}")
    print(f"JSON: {json_path}")
    print("STATUS: PASS", flush=True)

    return html_path


def main(sample_size: int):
    print("=== TAXI DATA DRIFT MONITORING ===")

    reference, current = load_reference_data(
        sample_size
    )

    baseline_metadata = {
        "scenario": "baseline",
        "source": str(DATA_PATH),
        "sample_size": sample_size,
        "seed": 42,
        "simulated": False,
        "description": (
            "Disjoint samples from the same "
            "January 2024 taxi dataset."
        ),
    }

    generate_report(
        reference,
        current,
        "taxi_baseline_drift",
        baseline_metadata,
    )

    # Simulate a strong numerical feature shift.
    shifted = current.copy()

    shifted["trip_distance"] = np.clip(
        shifted["trip_distance"].astype(float) * 3.0,
        0.1,
        100.0,
    )

    shifted_metadata = {
        "scenario": "simulated_drift",
        "source": str(DATA_PATH),
        "sample_size": sample_size,
        "seed": 42,
        "simulated": True,
        "transformation": (
            "trip_distance multiplied by 3, "
            "clipped to 0.1-100."
        ),
    }

    generate_report(
        reference,
        shifted,
        "taxi_simulated_drift",
        shifted_metadata,
    )

    print("=== EVIDENTLY REPORTS COMPLETED ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sample-size",
        type=int,
        default=1500,
    )

    args = parser.parse_args()

    if args.sample_size < 100:
        parser.error("sample-size must be >= 100")

    main(args.sample_size)
