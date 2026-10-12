import json
import os
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
from contextlib import closing
from pathlib import Path

import httpx
import pandas as pd

from src.preprocess import FEATURES, preprocess


HOST = "127.0.0.1"
PORT = 8001
BASE_URL = f"http://{HOST}:{PORT}"
SAMPLE_SIZE = 5000

REFERENCE = Path(
    "data/green_tripdata_2024-01.parquet"
)


def wait_for_server(process, client):
    for _ in range(100):
        if process.poll() is not None:
            raise RuntimeError(
                f"API exited early: {process.returncode}"
            )

        try:
            response = client.get("/health")
            if (
                response.status_code == 200
                and response.json().get("model_loaded")
            ):
                return
        except httpx.RequestError:
            pass

        time.sleep(0.2)

    raise TimeoutError("Test API failed to start")


def main():
    print("=== ISOLATED E2E MONITORING ===", flush=True)

    with socket.socket() as sock:
        if sock.connect_ex((HOST, PORT)) == 0:
            raise RuntimeError(
                "Port 8001 occupied. No test started."
            )

    raw = pd.read_parquet(REFERENCE)

    prepared = preprocess(raw)[FEATURES].copy()

    if len(prepared) < SAMPLE_SIZE:
        raise RuntimeError(
            "Insufficient source data"
        )

    sample = prepared.sample(
        n=SAMPLE_SIZE,
        random_state=20261012,
        replace=False,
    )

    with tempfile.TemporaryDirectory(
        prefix="taxi_e2e_"
    ) as directory:

        root = Path(directory)
        db = root / "test_predictions.sqlite3"
        reports = root / "reports"
        api_log = Path("monitoring/reports/e2e_validation/api_test.log").resolve()
        api_log.parent.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env["TAXI_PREDICTION_DB"] = str(db)
        env["TAXI_MONITOR_DB"] = str(db)
        env["TAXI_MONITOR_OUTPUT"] = str(reports)

        process = None

        try:
            with api_log.open("w", encoding="utf-8") as log:

                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "uvicorn",
                        "src.api:app",
                        "--host",
                        HOST,
                        "--port",
                        str(PORT),
                        "--no-access-log",
                    ],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )

                with httpx.Client(
                    base_url=BASE_URL,
                    timeout=20.0,
                ) as client:

                    wait_for_server(process, client)
                    print("TEST API: PASS", flush=True)

                    for i, row in enumerate(
                        sample.itertuples(index=False),
                        start=1,
                    ):
                        payload = {
                            "PULocationID": int(
                                row.PULocationID
                            ),
                            "DOLocationID": int(
                                row.DOLocationID
                            ),
                            "trip_distance": float(
                                row.trip_distance
                            ),
                        }

                        response = client.post(
                            "/predict",
                            json=payload,
                        )

                        if response.status_code != 200:
                            raise RuntimeError(
                                f"Prediction {i} failed: "
                                f"{response.text[:300]}"
                            )

                        if "predicted_duration_minutes" not in (
                            response.json()
                        ):
                            raise RuntimeError(
                                f"Prediction field missing: {i}"
                            )

                        if i % 500 == 0:
                            print(
                                f"API REQUESTS: {i}/{SAMPLE_SIZE}",
                                flush=True,
                            )

                with closing(
                    sqlite3.connect(db)
                ) as connection:
                    count = connection.execute(
                        "SELECT COUNT(*) FROM predictions"
                    ).fetchone()[0]

                print("SQLITE ROWS:", count, flush=True)

                if count != SAMPLE_SIZE:
                    raise RuntimeError(
                        f"Expected {SAMPLE_SIZE}, got {count}"
                    )

                print(
                    "API -> SQLITE: PASS",
                    flush=True,
                )

                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "src.monitor_api",
                        "--require-report",
                    ],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=180,
                )

                print(result.stdout, flush=True)

                if result.returncode != 0:
                    print(result.stderr, flush=True)
                    raise RuntimeError(
                        "Evidently report generation failed"
                    )

                html = reports / "taxi_live_api_drift.html"
                report_json = (
                    reports / "taxi_live_api_drift.json"
                )
                metadata = (
                    reports /
                    "taxi_live_api_drift_metadata.json"
                )

                for path in (html, report_json, metadata):
                    if not path.is_file():
                        raise RuntimeError(
                            f"Missing report: {path}"
                        )

                    if path.stat().st_size == 0:
                        raise RuntimeError(
                            f"Empty report: {path}"
                        )

                data = json.loads(
                    report_json.read_text(
                        encoding="utf-8"
                    )
                )

                if not data.get("metrics"):
                    raise RuntimeError(
                        "Evidently report has no metrics"
                    )

                destination = Path(
                    "monitoring/reports/e2e_validation"
                )
                destination.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                for source in (
                    html,
                    report_json,
                    metadata,
                ):
                    import shutil
                    shutil.copy2(
                        source,
                        destination / source.name,
                    )

                print(
                    "SQLITE -> EVIDENTLY: PASS",
                    flush=True,
                )
                print(
                    "REPORTS:",
                    destination,
                    flush=True,
                )
                print(
                    "E2E TECHNICAL TEST: PASS",
                    flush=True,
                )

        except Exception:
            if api_log.exists():
                print(
                    "=== TEST API LOG TAIL ===",
                    flush=True,
                )
                print(
                    "\n".join(
                        api_log.read_text(
                            encoding="utf-8",
                            errors="replace",
                        ).splitlines()[-30:]
                    ),
                    flush=True,
                )
            raise

        finally:
            if process is not None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)


if __name__ == "__main__":
    main()