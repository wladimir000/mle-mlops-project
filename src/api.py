from contextlib import asynccontextmanager
from pathlib import Path
import math
import os

import joblib
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field


DEFAULT_MODEL_PATH = Path("models/model.joblib")


class TaxiTrip(BaseModel):
    model_config = ConfigDict(extra="forbid")

    PULocationID: int = Field(ge=0)
    DOLocationID: int = Field(ge=0)
    trip_distance: float = Field(ge=0.1, le=100)


class PredictionResponse(BaseModel):
    predicted_duration_minutes: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = Path(
        os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))
    )

    if not model_path.is_file():
        raise RuntimeError(
            f"Model artifact not found: {model_path}"
        )

    # Only load trusted artifacts produced by our training code.
    app.state.model = joblib.load(model_path)

    if not hasattr(app.state.model, "predict"):
        raise RuntimeError("Invalid model: predict() missing")

    yield

    del app.state.model


app = FastAPI(
    title="NYC Green Taxi Duration API",
    description=(
        "Historical replay using completed taxi trip features. "
        "trip_distance is measured during the trip."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health(request: Request):
    ready = hasattr(request.app.state, "model")

    if not ready:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )

    return {
        "status": "ok",
        "model_loaded": True,
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(trip: TaxiTrip, request: Request):
    model = getattr(request.app.state, "model", None)

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )

    features = {
        "PULocationID": str(trip.PULocationID),
        "DOLocationID": str(trip.DOLocationID),
        "trip_distance": trip.trip_distance,
    }

    try:
        prediction = float(model.predict([features])[0])
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        ) from exc

    if not math.isfinite(prediction):
        raise HTTPException(
            status_code=500,
            detail="Non-finite model prediction"
        )

    from src.prediction_log import log_prediction
    log_prediction(features, prediction)

    return PredictionResponse(
        predicted_duration_minutes=prediction
    )

# Prometheus instrumentation
from src.metrics import install_metrics
install_metrics(app)
