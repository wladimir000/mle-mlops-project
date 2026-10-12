import pandas as pd

from src.preprocess import FEATURES, preprocess


RAW_REQUIRED = {
    "lpep_pickup_datetime",
    "lpep_dropoff_datetime",
    "PULocationID",
    "DOLocationID",
    "trip_distance",
}


def load_reference(path):
    frame = pd.read_parquet(path)

    if RAW_REQUIRED.issubset(frame.columns):
        result = preprocess(frame)[FEATURES].copy()

    elif set(FEATURES).issubset(frame.columns):
        result = frame[FEATURES].copy()

    else:
        raise ValueError(
            "Unsupported reference format: "
            f"{list(frame.columns)}"
        )

    if result.empty:
        raise ValueError("Reference dataset is empty")

    for column in ("PULocationID", "DOLocationID"):
        if result[column].isna().any():
            raise ValueError(
                f"Missing values in {column}"
            )
        result[column] = result[column].astype(str)

    result["trip_distance"] = pd.to_numeric(
        result["trip_distance"],
        errors="raise",
    )

    if result[FEATURES].isna().any().any():
        raise ValueError("Reference contains missing values")

    if not result["trip_distance"].between(
        0.1, 100
    ).all():
        raise ValueError(
            "Reference contains invalid trip distances"
        )

    return result[FEATURES].copy()
