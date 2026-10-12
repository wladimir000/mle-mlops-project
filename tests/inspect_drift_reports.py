import json
from pathlib import Path


ROOT = Path("monitoring/reports/e2e_independent")


def compact(value, depth=0):
    """Display report structure without huge distributions."""
    if depth >= 5:
        return "..."

    if isinstance(value, dict):
        result = {}

        for key, item in value.items():
            lower = str(key).lower()

            if any(word in lower for word in (
                "histogram", "distribution",
                "plot", "graph", "image",
                "render", "visualization",
            )):
                continue

            result[key] = compact(item, depth + 1)

        return result

    if isinstance(value, list):
        return [
            compact(item, depth + 1)
            for item in value[:4]
        ]

    if isinstance(value, str):
        return value[:160]

    return value


def inspect_report(name):
    path = (
        ROOT / name / "taxi_live_api_drift.json"
    )

    report = json.loads(
        path.read_text(encoding="utf-8-sig")
    )

    if not isinstance(report, dict):
        raise ValueError("Invalid JSON root")

    metrics = report.get("metrics")

    if not isinstance(metrics, list) or not metrics:
        raise ValueError("Missing metrics list")

    print(f"\n=== {name.upper()} ===")

    print("ROOT KEYS:", sorted(report.keys()))
    print("METRIC COUNT:", len(metrics))

    for index, metric in enumerate(metrics):
        print(f"\nMETRIC {index}")

        if isinstance(metric, dict):
            print(
                json.dumps(
                    compact(metric),
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                )[:4500]
            )
        else:
            print(
                repr(metric)[:500]
            )

    print(f"\n{name.upper()} JSON STRUCTURE: PASS")


def main():
    inspect_report("baseline")
    inspect_report("shifted")

    print("\nREPORT INSPECTION: PASS")


if __name__ == "__main__":
    main()