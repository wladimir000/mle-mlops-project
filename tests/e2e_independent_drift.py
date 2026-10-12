import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from contextlib import closing
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from src.preprocess import FEATURES, preprocess


ROOT = Path.cwd()
SOURCE = ROOT / "data/green_tripdata_2024-01.parquet"
RESULTS = ROOT / "monitoring/reports/e2e_independent"

N = 5000
PORT = 8002
BASE_URL = f"http://127.0.0.1:{PORT}"


def prepare_groups():
    raw = pd.read_parquet(SOURCE)
    prepared = preprocess(raw)[FEATURES].copy()

    if len(prepared) < 3 * N:
        raise RuntimeError(
            f"Need at least {3 * N} valid rows"
        )

    # Three non-overlapping source-row groups.
    selected = prepared.sample(
        n=3 * N,
        random_state=20261017,
        replace=False,
    )

    reference = selected.iloc[:N].copy()
    baseline = selected.iloc[N:2*N].copy()
    drift = selected.iloc[2*N:3*N].copy()

    groups = [reference, baseline, drift]

    for i in range(3):
        for j in range(i + 1, 3):
            overlap = set(groups[i].index) & set(
                groups[j].index
            )
            assert not overlap, (
                "Source rows overlap"
            )

    print("SOURCE-ROW DISJOINTNESS: PASS", flush=True)

    # Intentional synthetic shift.
    shifted = drift.copy()

    shifted["trip_distance"] = np.clip(
        shifted["trip_distance"].astype(float) * 3,
        0.1,
        100.0,
    )

    destination = (
        reference["DOLocationID"].mode().iloc[0]
    )

    rng = np.random.default_rng(20261017)
    mask = rng.random(N) < 0.8

    shifted.loc[mask, "DOLocationID"] = destination

    return reference, baseline, shifted


def wait_for_api(process, client):
    for _ in range(100):
        if process.poll() is not None:
            raise RuntimeError(
                f"Test API exited: {process.returncode}"
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

    raise TimeoutError(
        "Test API did not become ready"
    )


def run_scenario(name, reference, current):
    print(f"\n=== SCENARIO: {name} ===", flush=True)

    scenario_dir = RESULTS / name
    scenario_dir.mkdir(parents=True, exist_ok=True)

    # Files stored outside TemporaryDirectory
    # to avoid Windows log-file cleanup locks.
    api_log = scenario_dir / "api.log"

    with tempfile.TemporaryDirectory(
        prefix=f"taxi_{name}_"
    ) as directory:

        tmp = Path(directory)

        database = tmp / "predictions.sqlite3"
        reference_path = tmp / "reference.parquet"
        report_dir = tmp / "reports"

        reference.to_parquet(
            reference_path,
            index=False,
        )

        env = os.environ.copy()
        env["TAXI_PREDICTION_DB"] = str(database)
        env["TAXI_MONITOR_DB"] = str(database)
        env["TAXI_MONITOR_REFERENCE"] = str(
            reference_path
        )
        env["TAXI_MONITOR_OUTPUT"] = str(report_dir)

        process = None

        try:
            with api_log.open(
                "w", encoding="utf-8"
            ) as log:

                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "uvicorn",
                        "src.api:app",
                        "--host",
                        "127.0.0.1",
                        "--port",
                        str(PORT),
                        "--no-access-log",
                    ],
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )

                with httpx.Client(
                    base_url=BASE_URL,
                    timeout=20.0,
                ) as client:

                    wait_for_api(process, client)

                    print("API READY: PASS", flush=True)

                    for number, row in enumerate(
                        current.itertuples(index=False),
                        1,
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
                                f"Prediction {number}: "
                                f"{response.status_code}"
                            )

                        if (
                            "predicted_duration_minutes"
                            not in response.json()
                        ):
                            raise RuntimeError(
                                "Prediction response missing"
                            )

                        if number % 1000 == 0:
                            print(
                                f"REQUESTS: {number}/{N}",
                                flush=True,
                            )

                with closing(
                    sqlite3.connect(database)
                ) as connection:
                    rows = connection.execute(
                        "SELECT COUNT(*) FROM predictions"
                    ).fetchone()[0]

                if rows != N:
                    raise RuntimeError(
                        f"Expected {N} rows, got {rows}"
                    )

                print("SQLITE ROWS:", rows, flush=True)

                monitor = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "src.monitor_api",
                        "--require-report",
                    ],
                    cwd=ROOT,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=180,
                )

                print(monitor.stdout, flush=True)

                if monitor.returncode:
                    print(monitor.stderr, flush=True)
                    raise RuntimeError(
                        "Evidently monitor failed"
                    )

                files = [
                    "taxi_live_api_drift.html",
                    "taxi_live_api_drift.json",
                    "taxi_live_api_drift_metadata.json",
                ]

                for filename in files:
                    source = report_dir / filename
                    if not source.exists():
                        raise RuntimeError(
                            f"Missing {source}"
                        )
                    shutil.copy2(
                        source,
                        scenario_dir / filename,
                    )

                report_data = json.loads(
                    (
                        scenario_dir /
                        "taxi_live_api_drift.json"
                    ).read_text(encoding="utf-8")
                )

                # Evidently 0.7 JSON structure must
                # be inspected, not guessed.
                if not report_data.get("metrics"):
                    raise RuntimeError(
                        "Missing Evidently metrics"
                    )

                print(
                    "REPORT GENERATION: PASS",
                    flush=True,
                )

        finally:
            if process is not None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)

    print(
        f"SCENARIO {name}: COMPLETE",
        flush=True,
    )


def main():
    RESULTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    reference, baseline, shifted = prepare_groups()

    run_scenario("baseline", reference, baseline)
    run_scenario("shifted", reference, shifted)

    print("\nE2E REPORTS GENERATED: PASS")
    print("REPORT ROOT:", RESULTS)
    print(
        "Review Dataset Drift decisions "
        "in both reports before acceptance."
    )


if __name__ == "__main__":
    main()