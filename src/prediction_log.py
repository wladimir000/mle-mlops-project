import logging
import os
import sqlite3

from contextlib import closing
from pathlib import Path


logger = logging.getLogger(__name__)

DEFAULT_DATABASE = "data/api_predictions.sqlite3"


def get_database_path():
    return Path(
        os.getenv(
            "TAXI_PREDICTION_DB",
            DEFAULT_DATABASE,
        )
    )


def log_prediction(features, prediction):
    """
    Persist a completed prediction.

    Database errors must not block inference.
    """

    database = get_database_path()

    try:
        database.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with closing(
            sqlite3.connect(
                database,
                timeout=5.0,
            )
        ) as connection:

            with connection:

                connection.execute("""
                    CREATE TABLE IF NOT EXISTS predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        created_utc TEXT NOT NULL
                            DEFAULT CURRENT_TIMESTAMP,
                        PULocationID TEXT NOT NULL,
                        DOLocationID TEXT NOT NULL,
                        trip_distance REAL NOT NULL,
                        prediction_minutes REAL NOT NULL
                    )
                """)

                connection.execute(
                    """
                    INSERT INTO predictions (
                        PULocationID,
                        DOLocationID,
                        trip_distance,
                        prediction_minutes
                    )
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        str(features["PULocationID"]),
                        str(features["DOLocationID"]),
                        float(features["trip_distance"]),
                        float(prediction),
                    ),
                )

        return True

    except (
        OSError,
        sqlite3.Error,
        ValueError,
        TypeError,
    ):
        logger.exception(
            "Prediction logging failed"
        )
        return False