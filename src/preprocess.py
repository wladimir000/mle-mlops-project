import pandas as pd

FEATURES = ["PULocationID", "DOLocationID", "trip_distance"]
TARGET = "duration"


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess NYC Green Taxi trip records."""

    df = df.copy()

    df[TARGET] = (
        df["lpep_dropoff_datetime"]
        - df["lpep_pickup_datetime"]
    ).dt.total_seconds() / 60

    df = df[df[TARGET].between(1, 60)]
    df = df[df["trip_distance"].between(0.1, 100)]

    df[["PULocationID", "DOLocationID"]] = (
        df[["PULocationID", "DOLocationID"]].astype(str)
    )

    return df[FEATURES + [TARGET]].copy()
