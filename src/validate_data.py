"""
validate_data.py
----------------
Sanity-checks the generated dataset against expected distributions.
Run this after generate_data.py to confirm data looks realistic.

Usage:
    python src/validate_data.py
    python src/validate_data.py --input data/raw/rides.csv
"""

import argparse
import sys

import pandas as pd
import numpy as np


def load(path: str) -> pd.DataFrame:
    print(f"\nLoading: {path}")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def check(condition: bool, msg: str, warn: bool = False) -> bool:
    status = "PASS" if condition else ("WARN" if warn else "FAIL")
    icon   = "✓" if condition else ("⚠" if warn else "✗")
    print(f"  {icon} [{status}] {msg}")
    return condition


def run_checks(df: pd.DataFrame) -> int:
    failures = 0
    print("\n── Schema checks ───────────────────────────────────────────")
    expected_cols = {
        "ride_id", "timestamp", "city", "pickup_zone", "vehicle_type",
        "wait_time_min", "surge_multiplier", "distance_km", "fare_inr",
        "driver_rating", "is_completed", "weather", "is_festival", "is_ipl_day"
    }
    missing = expected_cols - set(df.columns)
    if not check(len(missing) == 0, f"All 14 columns present (missing: {missing})"):
        failures += 1

    print("\n── Volume checks ───────────────────────────────────────────")
    if not check(len(df) >= 490_000, f"Row count ≥ 490K  (got {len(df):,})"):
        failures += 1

    cities = ["Delhi NCR", "Mumbai", "Bengaluru", "Chennai", "Hyderabad"]
    if not check(set(df["city"].unique()) == set(cities), "All 5 cities present"):
        failures += 1

    date_min = df["timestamp"].min().date()
    date_max = df["timestamp"].max().date()
    if not check(str(date_min) <= "2023-01-31", f"Data starts in Jan 2023  (got {date_min})"):
        failures += 1
    if not check(str(date_max) >= "2024-06-01", f"Data ends in Jun 2024   (got {date_max})"):
        failures += 1

    print("\n── Distribution checks ─────────────────────────────────────")
    # Wait time: median should be 5–10 min
    med_wait = df["wait_time_min"].median()
    if not check(4.0 <= med_wait <= 12.0, f"Median wait_time_min in [4, 12]  (got {med_wait:.1f})"):
        failures += 1

    # Surge: mean should be 1.1–1.8
    mean_surge = df["surge_multiplier"].mean()
    if not check(1.0 <= mean_surge <= 2.0, f"Mean surge_multiplier in [1.0, 2.0]  (got {mean_surge:.2f})"):
        failures += 1

    # Fare: median ₹100–₹300
    med_fare = df["fare_inr"].median()
    if not check(80 <= med_fare <= 400, f"Median fare_inr in [80, 400]  (got ₹{med_fare:.0f})"):
        failures += 1

    # Completion rate: 80–97%
    comp_rate = df["is_completed"].mean() * 100
    if not check(80 <= comp_rate <= 97, f"Completion rate in [80%, 97%]  (got {comp_rate:.1f}%)"):
        failures += 1

    # Driver rating: mean 3.8–4.6
    mean_rating = df["driver_rating"].median()
    if not check(3.5 <= mean_rating <= 4.8, f"Median driver_rating in [3.5, 4.8]  (got {mean_rating:.1f})"):
        failures += 1

    print("\n── Vehicle type checks ─────────────────────────────────────")
    valid_vt = {"bike_taxi", "auto", "economy", "premium", "shared"}
    if not check(set(df["vehicle_type"].unique()) == valid_vt, "All 5 vehicle types present"):
        failures += 1

    print("\n── Weather checks ──────────────────────────────────────────")
    valid_weather = {"clear", "light_rain", "heavy_rain", "fog"}
    if not check(set(df["weather"].unique()).issubset(valid_weather), "Only valid weather values"):
        failures += 1
    clear_pct = (df["weather"] == "clear").mean() * 100
    check(50 <= clear_pct <= 85, f"Clear weather 50–85% of rides  (got {clear_pct:.1f}%)", warn=True)

    print("\n── Data quality checks ─────────────────────────────────────")
    null_wait   = df["wait_time_min"].isna().sum()
    null_rating = df["driver_rating"].isna().sum()
    null_zone   = df["pickup_zone"].isna().sum()
    check(null_wait > 0,   f"Null wait_time_min injected  ({null_wait:,} rows)", warn=(null_wait == 0))
    check(null_rating > 0, f"Null driver_rating injected  ({null_rating:,} rows)", warn=(null_rating == 0))
    check(null_zone > 0,   f"Null pickup_zone injected    ({null_zone:,} rows)", warn=(null_zone == 0))

    outlier_fares = ((df["fare_inr"] <= 0) | (df["fare_inr"] > 5000)).sum()
    check(outlier_fares > 0, f"Outlier fares injected  ({outlier_fares:,} rows)", warn=(outlier_fares == 0))

    dupes = df.duplicated(subset=["city", "timestamp", "vehicle_type", "distance_km"]).sum()
    check(dupes > 0, f"Duplicate rows present  ({dupes:,})", warn=(dupes == 0))

    print("\n── Temporal pattern checks ─────────────────────────────────")
    df["hour"] = df["timestamp"].dt.hour
    peak_pct   = df["hour"].isin([7, 8, 9, 17, 18, 19]).mean() * 100
    if not check(15 <= peak_pct <= 70, f"15–70% of rides in peak hours  (got {peak_pct:.1f}%)"):
        failures += 1

    festival_rides = df["is_festival"].sum()
    ipl_rides      = df["is_ipl_day"].sum()
    check(festival_rides > 0, f"Festival rides exist  ({festival_rides:,})", warn=(festival_rides == 0))
    check(ipl_rides > 0,      f"IPL-day rides exist  ({ipl_rides:,})", warn=(ipl_rides == 0))

    return failures


def print_result(failures: int) -> None:
    print(f"\n{'='*55}")
    if failures == 0:
        print("  ALL CHECKS PASSED — dataset looks great!")
        print("  Ready for Phase 2: data pipeline & preprocessing.")
    else:
        print(f"  {failures} CHECK(S) FAILED — review output above.")
    print(f"{'='*55}\n")


def main():
    parser = argparse.ArgumentParser(description="Validate generated ride dataset")
    parser.add_argument("--input", type=str, default="data/raw/rides.csv")
    args = parser.parse_args()

    df = load(args.input)
    failures = run_checks(df)
    print_result(failures)
    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
