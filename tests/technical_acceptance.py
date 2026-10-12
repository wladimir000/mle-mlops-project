import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path.cwd()
OUTPUT = ROOT / "monitoring/reports/acceptance"
OUTPUT.mkdir(parents=True, exist_ok=True)

results = []


def run_gate(name, command):
    print(f"\n=== {name} ===", flush=True)

    process = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )

    output = process.stdout + process.stderr

    (OUTPUT / f"{name}.log").write_text(
        output,
        encoding="utf-8",
    )

    status = (
        "PASS" if process.returncode == 0
        else "FAIL"
    )

    results.append({
        "gate": name,
        "status": status,
        "exit_code": process.returncode,
    })

    print(output[-2500:], flush=True)
    print(f"{name}: {status}", flush=True)


def check_file(name, path):
    print(f"\n=== {name} ===")

    target = ROOT / path

    status = (
        "PASS"
        if target.is_file() and target.stat().st_size > 0
        else "FAIL"
    )

    results.append({
        "gate": name,
        "status": status,
        "path": str(path),
    })

    print(f"{name}: {status}")


def main():
    print("=== MLOPS TECHNICAL ACCEPTANCE ===")

    run_gate(
        "G01_SOURCE_SYNTAX",
        [
            sys.executable,
            "-m",
            "compileall",
            "-q",
            "src",
            "tests",
        ],
    )

    run_gate(
        "G02_API_TESTS",
        [
            sys.executable,
            "-m",
            "unittest",
            "tests.test_api",
            "-v",
        ],
    )

    run_gate(
        "G03_DRIFT_GATES",
        [
            sys.executable,
            "-m",
            "unittest",
            "tests.test_drift_gates",
            "-v",
        ],
    )

    check_file(
        "G04_TRAINED_MODEL",
        "models/model.joblib",
    )

    check_file(
        "G05_DVC_POINTER",
        "data/green_tripdata_2024-01.parquet.dvc",
    )

    check_file(
        "G06_GITHUB_WORKFLOW",
        ".github/workflows/taxi-training.yml",
    )

    check_file(
        "G07_DOCKER_COMPOSE",
        "docker-compose.yml",
    )

    check_file(
        "G08_PROMETHEUS_CONFIG",
        "monitoring/prometheus/prometheus.yml",
    )

    check_file(
        "G09_GRAFANA_DASHBOARD",
        "monitoring/grafana/dashboards/taxi-api.json",
    )

    for scenario in ("baseline", "shifted"):
        check_file(
            f"G10_{scenario.upper()}_REPORT",
            (
                "monitoring/reports/e2e_independent/"
                f"{scenario}/taxi_live_api_drift.json"
            ),
        )

    report = {
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "results": results,
        "total": len(results),
        "passed": sum(
            r["status"] == "PASS"
            for r in results
        ),
        "failed": sum(
            r["status"] == "FAIL"
            for r in results
        ),
    }

    destination = OUTPUT / "acceptance.json"

    destination.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n=== FINAL SUMMARY ===")
    print("TOTAL:", report["total"])
    print("PASSED:", report["passed"])
    print("FAILED:", report["failed"])
    print("REPORT:", destination)

    if report["failed"]:
        raise SystemExit(1)

    print("TECHNICAL STATIC ACCEPTANCE: PASS")


if __name__ == "__main__":
    main()