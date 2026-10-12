import argparse
import os
from pathlib import Path
from urllib.request import urlretrieve

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from dotenv import load_dotenv
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from src.preprocess import FEATURES, TARGET, preprocess


EXPERIMENT_NAME = "green-taxi-monitoring"
MODEL_NAME = "green-taxi-ride-duration"


def load_data(year: int, month: int) -> pd.DataFrame:
    """Load cached data or download NYC Green Taxi parquet."""

    path = Path(
        f"data/green_tripdata_{year}-{month:02d}.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        url = (
            "https://d37ci6vzurychx.cloudfront.net/"
            f"trip-data/green_tripdata_{year}-{month:02d}.parquet"
        )
        print(f"Downloading dataset: {url}", flush=True)
        urlretrieve(url, path)

    print(f"Loading dataset: {path}", flush=True)
    return pd.read_parquet(path)


def train(year: int = 2024, month: int = 1) -> float:
    """Train, evaluate, save and register the model."""

    load_dotenv()

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "sqlite:///mlflow.db"
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_data(year, month)
    df = preprocess(df)

    if len(df) < 2:
        raise ValueError("Not enough valid trips for training.")

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        random_state=42,
        test_size=0.2
    )

    print(f"Total trips: {len(df)}", flush=True)
    print(f"Training rows: {len(X_train)}", flush=True)
    print(f"Testing rows: {len(X_test)}", flush=True)

    with mlflow.start_run() as run:

        mlflow.set_tags({
            "model": "linear regression",
            "dataset": "green-taxi",
            "year": str(year),
            "month": str(month),
            "features": ", ".join(FEATURES),
            "target": TARGET,
            "pipeline": "standalone-training"
        })

        mlflow.log_params({
            "year": year,
            "month": month,
            "test_size": 0.2,
            "random_state": 42,
            "model_type": "LinearRegression"
        })

        pipeline = make_pipeline(
            DictVectorizer(),
            LinearRegression()
        )

        print("Training model...", flush=True)

        pipeline.fit(
            X_train.to_dict(orient="records"),
            y_train
        )

        predictions = pipeline.predict(
            X_test.to_dict(orient="records")
        )

        rmse = root_mean_squared_error(
            y_test,
            predictions
        )

        mlflow.log_metric("rmse", float(rmse))

        print(f"RMSE: {rmse:.4f} minutes", flush=True)

        # Save a local artifact for the later API
        model_path = Path("models/model.joblib")
        model_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(pipeline, model_path)
        print(f"Local model saved: {model_path}", flush=True)

        # Log model in MLflow
        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="mlflow-model",
            serialization_format="skops"
        )

        client = MlflowClient()

        try:
            client.create_registered_model(
                name=MODEL_NAME,
                tags={"framework": "scikit-learn"},
                description="NYC Green Taxi ride duration regression model"
            )
        except MlflowException as exc:
            if exc.error_code != "RESOURCE_ALREADY_EXISTS":
                raise

        version = client.create_model_version(
            name=MODEL_NAME,
            source=model_info.model_uri,
            run_id=run.info.run_id
        )

        client.set_registered_model_alias(
            name=MODEL_NAME,
            alias="production",
            version=version.version
        )

        print(f"MLflow run ID: {run.info.run_id}", flush=True)
        print(f"Registered model: {MODEL_NAME}", flush=True)
        print(f"Model version: {version.version}", flush=True)
        print("Alias: production", flush=True)

    return float(rmse)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--month", type=int, default=1)

    args = parser.parse_args()

    train(year=args.year, month=args.month)
