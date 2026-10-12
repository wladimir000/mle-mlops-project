import json
from pathlib import Path

import numpy as np
import pandas as pd

from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset

from src.preprocess import FEATURES, preprocess


ROOT = Path("monitoring/reports")

CALIBRATION = ROOT / "drift_calibration_5000.json"

DATA_FILE = Path(
    "data/green_tripdata_2024-01.parquet"
)

COLUMNS = [
    "PULocationID",
    "DOLocationID",
    "trip_distance",
]


def load_thresholds():
    calibration = json.loads(
        CALIBRATION.read_text(encoding="utf-8-sig")
    )

    if calibration["sample_size_each_group"] != 5000:
        raise ValueError("Unexpected calibration sample size")

    thresholds = calibration["candidate_thresholds"]

    for column in COLUMNS:
        value = float(thresholds[column])

        if not 0 < value < 1:
            raise ValueError(
                f"Invalid threshold for {column}: {value}"
            )

    return {
        column: float(thresholds[column])
        for column in COLUMNS
    }


def prepare_samples():
    df = preprocess(
        pd.read_parquet(DATA_FILE)
    )[FEATURES].copy()

    if len(df) < 10000:
        raise ValueError("Insufficient reference data")

    samples = df.sample(
        n=10000,
        random_state=20261013,
        replace=False,
    ).reset_index(drop=True)

    reference = samples.iloc[:5000].copy()
    current = samples.iloc[5000:].copy()

    return reference, current


def to_evidently(frame):
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


def run_report(name, reference, current, thresholds):
    preset = DataDriftPreset(
        columns=COLUMNS,
        drift_share=0.5,
        per_column_method={
            "PULocationID": "jensenshannon",
            "DOLocationID": "jensenshannon",
            "trip_distance": "wasserstein",
        },
        per_column_threshold=thresholds,
    )

    report = Report([preset])

    result = report.run(
        to_evidently(current),
        to_evidently(reference),
    )

    ROOT.mkdir(parents=True, exist_ok=True)

    html = ROOT / f"{name}.html"
    output = ROOT / f"{name}.json"

    result.save_html(str(html))
    output.write_text(
        result.json(),
        encoding="utf-8",
    )

    if not html.exists() or html.stat().st_size == 0:
        raise RuntimeError(f"HTML output invalid: {html}")

    data = json.loads(
        output.read_text(encoding="utf-8")
    )

    if not isinstance(data, dict) or not data.get("metrics"):
        raise RuntimeError(
            f"Evidently metrics missing: {output}"
        )

    print(f"REPORT: {html}")
    print(f"JSON: {output}")
    print("EXPORT: PASS", flush=True)


def main():
    print("=== CALIBRATED EVIDENTLY VALIDATION ===")

    thresholds = load_thresholds()

    print("METHODS:")
    print("Locations: Jensen-Shannon")
    print("Distance: Wasserstein")

    print("CANDIDATE THRESHOLDS:")
    for name, threshold in thresholds.items():
        print(f"{name}: {threshold:.6f}")

    reference, current = prepare_samples()

    print("\n=== BASELINE ===")

    run_report(
        "taxi_calibrated_baseline",
        reference,
        current,
        thresholds,
    )

    print("\n=== SYNTHETIC DRIFT ===")

    shifted = current.copy()

    shifted["trip_distance"] = np.clip(
        shifted["trip_distance"].astype(float) * 3,
        0.1,
        100.0,
    )

    # Shift destination distribution.
    common_destination = (
        reference["DOLocationID"]
        .mode()
        .iloc[0]
    )

    rng = np.random.default_rng(20261013)

    mask = rng.random(len(shifted)) < 0.8

    shifted.loc[
        mask, "DOLocationID"
    ] = common_destination

    run_report(
        "taxi_calibrated_simulated_drift",
        reference,
        shifted,
        thresholds,
    )

    print("\nREPORT GENERATION: PASS")
    print(
        "Review Evidently metric scores and "
        "drift decisions before acceptance."
    )


if __name__ == "__main__":
    main()
