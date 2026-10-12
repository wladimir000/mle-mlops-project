"""Offline calibration of taxi feature drift thresholds (diagnostic only)."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

from src.preprocess import FEATURES, preprocess

FEATURE_TYPES = {
    "PULocationID": "categorical",
    "DOLocationID": "categorical",
    "trip_distance": "numerical",
}


def js_distance(a: pd.Series, b: pd.Series) -> float:
    a = a.astype(str)
    b = b.astype(str)
    categories = a.value_counts().index.union(b.value_counts().index)
    p = a.value_counts().reindex(categories, fill_value=0).to_numpy(dtype=float)
    q = b.value_counts().reindex(categories, fill_value=0).to_numpy(dtype=float)
    return float(jensenshannon(p / p.sum(), q / q.sum()))


def distance_scores(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    std = max(float(reference["trip_distance"].std(ddof=0)), 1e-8)
    return {
        "PULocationID": js_distance(reference["PULocationID"], current["PULocationID"]),
        "DOLocationID": js_distance(reference["DOLocationID"], current["DOLocationID"]),
        "trip_distance": float(wasserstein_distance(
            reference["trip_distance"], current["trip_distance"]
        ) / std),
    }


def draw_pair(pool: pd.DataFrame, n: int, seed: int):
    indexes = np.random.default_rng(seed).choice(len(pool), size=2 * n, replace=False)
    return pool.iloc[indexes[:n]].copy(), pool.iloc[indexes[n:]].copy()


def run_one(train_pool, validation_pool, n, runs, seed):
    calibration_scores = [
        distance_scores(*draw_pair(train_pool, n, seed + i))
        for i in range(runs)
    ]
    validation_scores = [
        distance_scores(*draw_pair(validation_pool, n, seed + 10000 + i))
        for i in range(runs)
    ]

    thresholds = {
        feature: float(np.quantile(
            [r[feature] for r in calibration_scores], 0.95
        ))
        for feature in FEATURE_TYPES
    }

    holdout_rates = {
        feature: float(np.mean([
            r[feature] > thresholds[feature] for r in validation_scores
        ]))
        for feature in FEATURE_TYPES
    }

    # Dataset-level threshold > 0.5 means at least 2/3 features must drift.
    holdout_dataset_rate = float(np.mean([
        sum(r[col] > thresholds[col] for col in FEATURE_TYPES) >= 2
        for r in validation_scores
    ]))

    ref, curr = draw_pair(validation_pool, n, seed + 90000)
    shifted = curr.copy()
    shifted["trip_distance"] = np.clip(
        shifted["trip_distance"].astype(float) * 3, 0.1, 100.0
    )
    # Use an existing zone category, but strongly change its frequency.
    modal_zone = str(ref["DOLocationID"].mode().iloc[0])
    positions = np.random.default_rng(seed + 90001).choice(
        len(shifted), size=int(0.8 * len(shifted)), replace=False
    )
    shifted.iloc[positions, shifted.columns.get_loc("DOLocationID")] = modal_zone
    shifted_scores = distance_scores(ref, shifted)
    shifted_alerts = {
        k: shifted_scores[k] > thresholds[k] for k in FEATURE_TYPES
    }

    result = {
        "sample_size_each_group": n,
        "calibration_repeats": runs,
        "holdout_repeats": runs,
        "training_validation_pools_disjoint": True,
        "distance_methods": {
            "PULocationID": "scipy Jensen-Shannon distance",
            "DOLocationID": "scipy Jensen-Shannon distance",
            "trip_distance": "Wasserstein distance / reference std",
        },
        "threshold_method": "95th percentile of calibration distances",
        "candidate_thresholds": thresholds,
        "holdout_alarm_rate_by_feature": holdout_rates,
        "holdout_dataset_alarm_rate_2_of_3": holdout_dataset_rate,
        "positive_control_scores": shifted_scores,
        "positive_control_detected_by_feature": shifted_alerts,
        "positive_control_dataset_detected_2_of_3": sum(shifted_alerts.values()) >= 2,
        "positive_control": "trip_distance x3 (clipped) and 80% dropoff destinations set to common existing ID",
        "caveats": [
            "Calibration uses SciPy, not the native Evidently threshold configuration.",
            "Repeated random splits reuse records inside each disjoint pool.",
            "A synthetic shift does not demonstrate real-world model degradation.",
            "Candidate thresholds are not yet installed into Evidently.",
        ],
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[1500, 5000])
    parser.add_argument("--runs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20261012)
    args = parser.parse_args()
    if args.runs < 20 or not args.sizes or min(args.sizes) < 100:
        parser.error("Use at least 20 runs and sample sizes >= 100")

    raw_path = Path("data/green_tripdata_2024-01.parquet")
    if not raw_path.is_file():
        raise FileNotFoundError(f"Missing training data: {raw_path}")
    prepared = preprocess(pd.read_parquet(raw_path))[FEATURES].copy()
    if not prepared["trip_distance"].notna().all():
        raise ValueError("Missing trip_distance values")
    prepared = prepared.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    midpoint = len(prepared) // 2
    train_pool, validation_pool = prepared.iloc[:midpoint], prepared.iloc[midpoint:]
    if min(len(train_pool), len(validation_pool)) < 2 * max(args.sizes):
        raise ValueError(
            f"Need >= {4*max(args.sizes)} valid rows for disjoint calibration/validation pools. "
            f"Found {len(prepared)}."
        )

    out = Path("monitoring/reports")
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for n in args.sizes:
        print(f"CALIBRATING: {n} rows/group, {args.runs} calibration + {args.runs} holdout pairs", flush=True)
        result = run_one(train_pool, validation_pool, n, args.runs, args.seed + n)
        results.append(result)
        path = out / f"drift_calibration_{n}.json"
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print("FEATURE           CANDIDATE_THRESHOLD  HOLDOUT_ALARM_RATE  SIMULATED_SHIFT")
        for feature in FEATURE_TYPES:
            print(f"{feature:18s} {result['candidate_thresholds'][feature]:19.6f} "
                  f"{result['holdout_alarm_rate_by_feature'][feature]:18.1%} "
                  f"{str(result['positive_control_detected_by_feature'][feature]):>15s}")
        print(f"Dataset-level baseline alarm rate: {result['holdout_dataset_alarm_rate_2_of_3']:.1%}")
        print(f"Synthetic combined drift detected: {result['positive_control_dataset_detected_2_of_3']}")
        print(f"REPORT: {path}\n", flush=True)
    if any(not r["positive_control_detected_by_feature"][f]
           for r in results for f in ["trip_distance", "DOLocationID"]):
        raise RuntimeError("Positive control missed a deliberately shifted feature")
    print("PASS: Calibration outputs generated. Review false-alarm rates before applying thresholds.")


if __name__ == "__main__":
    main()