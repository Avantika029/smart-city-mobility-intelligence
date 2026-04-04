"""
pipeline.py
-----------
Data pipeline for the Smart City Mobility Intelligence System.

Stages:
  1. Load        — read raw CSV, parse types
  2. Validate    — schema + type checks before touching data
  3. Clean       — handle nulls, duplicates, outliers
  4. Engineer    — create 15+ features for EDA and modelling
  5. Encode      — label-encode categoricals
  6. Store       — save to DuckDB + processed CSV
  7. Report      — print quality report with before/after stats

Usage:
    python src/pipeline.py
    python src/pipeline.py --input data/raw/rides.csv --output data/processed/rides_clean.csv
"""

import argparse
import os
import json
from datetime import datetime

import numpy as np
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────

RAW_PATH       = "data/raw/rides.csv"
PROCESSED_PATH = "data/processed/rides_clean.csv"
DB_PATH        = "data/processed/mobility.duckdb"
REPORT_PATH    = "reports/quality/pipeline_report.json"

VALID_CITIES        = {"Delhi NCR", "Mumbai", "Bengaluru", "Chennai", "Hyderabad"}
VALID_VEHICLE_TYPES = {"bike_taxi", "auto", "economy", "premium", "shared"}
VALID_WEATHER       = {"clear", "light_rain", "heavy_rain", "fog"}

FARE_MIN, FARE_MAX         = 10.0, 5000.0
DISTANCE_MIN, DISTANCE_MAX = 0.5, 80.0
WAIT_MIN, WAIT_MAX         = 0.5, 120.0
SURGE_MIN, SURGE_MAX       = 1.0, 4.0
RATING_MIN, RATING_MAX     = 1.0, 5.0

# City-zone lookup for zone-level imputation
CITY_ZONES = {
    "Delhi NCR":  ["Connaught Place", "Karol Bagh", "Lajpat Nagar", "Saket", "Dwarka",
                   "Rohini", "Noida Sector 18", "Gurgaon Cyber City", "Hauz Khas",
                   "Nehru Place", "Janakpuri", "Vasant Kunj", "Greater Noida",
                   "Faridabad", "Ghaziabad"],
    "Mumbai":     ["Bandra", "Andheri West", "Andheri East", "Juhu", "Borivali",
                   "Thane", "Kurla", "Dadar", "Lower Parel", "BKC", "Powai",
                   "Malad", "Navi Mumbai", "Vashi", "Chembur"],
    "Bengaluru":  ["Koramangala", "Indiranagar", "Whitefield", "Electronic City",
                   "HSR Layout", "Jayanagar", "Hebbal", "Marathahalli", "MG Road",
                   "Yelahanka", "Bannerghatta", "Sarjapur", "BTM Layout",
                   "Bellandur", "Rajajinagar"],
    "Chennai":    ["T. Nagar", "Anna Nagar", "Adyar", "Velachery", "Tambaram",
                   "Porur", "Nungambakkam", "Mylapore", "Perambur", "Chromepet",
                   "Sholinganallur", "OMR", "Guindy", "Kodambakkam", "Koyambedu"],
    "Hyderabad":  ["Hitech City", "Madhapur", "Gachibowli", "Banjara Hills",
                   "Jubilee Hills", "Secunderabad", "Ameerpet", "Kukatpally",
                   "LB Nagar", "Dilsukhnagar", "Uppal", "Kompally",
                   "Kondapur", "Manikonda", "Miyapur"],
}


# ── Stage 1: Load ─────────────────────────────────────────────────────────────

def load_raw(path: str) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print("  STAGE 1 — Load")
    print(f"{'='*60}")

    df = pd.read_csv(
        path,
        parse_dates=["timestamp"],
        dtype={
            "ride_id":          str,
            "city":             str,
            "pickup_zone":      str,
            "vehicle_type":     str,
            "weather":          str,
            "is_completed":     bool,
            "is_festival":      bool,
            "is_ipl_day":       bool,
        }
    )

    print(f"  Loaded   : {len(df):,} rows × {df.shape[1]} columns")
    print(f"  Date range: {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
    print(f"  Memory   : {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    return df


# ── Stage 2: Validate ─────────────────────────────────────────────────────────

def validate_schema(df: pd.DataFrame) -> dict:
    print(f"\n{'='*60}")
    print("  STAGE 2 — Validate")
    print(f"{'='*60}")

    issues = {}

    # Check required columns
    required = {"ride_id","timestamp","city","pickup_zone","vehicle_type",
                "wait_time_min","surge_multiplier","distance_km","fare_inr",
                "driver_rating","is_completed","weather","is_festival","is_ipl_day"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        issues["missing_columns"] = list(missing_cols)
        print(f"  ✗ Missing columns: {missing_cols}")
    else:
        print(f"  ✓ All 14 columns present")

    # Invalid categoricals
    bad_cities   = df[~df["city"].isin(VALID_CITIES)]["city"].unique().tolist()
    bad_vehicles = df[~df["vehicle_type"].isin(VALID_VEHICLE_TYPES)]["vehicle_type"].unique().tolist()
    bad_weather  = df[~df["weather"].isin(VALID_WEATHER)]["weather"].unique().tolist()

    if bad_cities:   issues["invalid_cities"]   = bad_cities
    if bad_vehicles: issues["invalid_vehicles"] = bad_vehicles
    if bad_weather:  issues["invalid_weather"]  = bad_weather

    print(f"  ✓ City values valid     (bad: {len(bad_cities)})")
    print(f"  ✓ Vehicle values valid  (bad: {len(bad_vehicles)})")
    print(f"  ✓ Weather values valid  (bad: {len(bad_weather)})")

    # Null counts
    null_counts = df.isnull().sum()
    nulls = null_counts[null_counts > 0].to_dict()
    issues["nulls_before_cleaning"] = nulls
    print(f"\n  Null counts before cleaning:")
    for col, n in nulls.items():
        print(f"    {col:<22} {n:>6,}  ({n/len(df)*100:.1f}%)")

    # Outlier counts
    outlier_fare     = ((df["fare_inr"] < FARE_MIN) | (df["fare_inr"] > FARE_MAX)).sum()
    outlier_distance = ((df["distance_km"] < DISTANCE_MIN) | (df["distance_km"] > DISTANCE_MAX)).sum()
    outlier_wait     = (df["wait_time_min"] > WAIT_MAX).sum()
    issues["outliers_before_cleaning"] = {
        "fare_inr": int(outlier_fare),
        "distance_km": int(outlier_distance),
        "wait_time_min": int(outlier_wait),
    }
    print(f"\n  Outliers before cleaning:")
    print(f"    fare_inr      {outlier_fare:>6,} rows outside [₹{FARE_MIN}, ₹{FARE_MAX}]")
    print(f"    distance_km   {outlier_distance:>6,} rows outside [{DISTANCE_MIN}, {DISTANCE_MAX}] km")
    print(f"    wait_time_min {outlier_wait:>6,} rows > {WAIT_MAX} min")

    # Duplicate check
    dupe_count = df.duplicated(subset=["city","timestamp","vehicle_type","distance_km"]).sum()
    issues["duplicates_before_cleaning"] = int(dupe_count)
    print(f"\n  Duplicate rows: {dupe_count:,}")

    return issues


# ── Stage 3: Clean ────────────────────────────────────────────────────────────

def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    print(f"\n{'='*60}")
    print("  STAGE 3 — Clean")
    print(f"{'='*60}")

    report = {}
    n_start = len(df)

    # 3a. Remove duplicates
    before = len(df)
    df = df.drop_duplicates(subset=["city", "timestamp", "vehicle_type", "distance_km"])
    removed_dupes = before - len(df)
    report["duplicates_removed"] = removed_dupes
    print(f"  ✓ Duplicates removed    : {removed_dupes:,} rows")

    # 3b. Drop rows with invalid categoricals
    before = len(df)
    df = df[df["city"].isin(VALID_CITIES)]
    df = df[df["vehicle_type"].isin(VALID_VEHICLE_TYPES)]
    df = df[df["weather"].isin(VALID_WEATHER)]
    removed_invalid = before - len(df)
    report["invalid_categoricals_removed"] = removed_invalid
    print(f"  ✓ Invalid category rows : {removed_invalid:,} rows removed")

    # 3c. Impute null wait_time_min — median per city + vehicle_type
    null_wait = df["wait_time_min"].isna().sum()
    city_vt_median = (
        df.groupby(["city", "vehicle_type"])["wait_time_min"]
        .median()
        .rename("wait_median")
    )
    df = df.join(city_vt_median, on=["city", "vehicle_type"])
    df["wait_time_min"] = df["wait_time_min"].fillna(df["wait_median"])
    df = df.drop(columns=["wait_median"])
    # Fallback: global median for any remaining nulls
    global_wait_median = df["wait_time_min"].median()
    df["wait_time_min"] = df["wait_time_min"].fillna(global_wait_median)
    report["wait_time_imputed"] = int(null_wait)
    print(f"  ✓ wait_time_min imputed : {null_wait:,} nulls → city+vehicle median")

    # 3d. Impute null driver_rating — median per city
    null_rating = df["driver_rating"].isna().sum()
    city_rating_median = df.groupby("city")["driver_rating"].median().rename("rating_median")
    df = df.join(city_rating_median, on="city")
    df["driver_rating"] = df["driver_rating"].fillna(df["rating_median"])
    df = df.drop(columns=["rating_median"])
    report["driver_rating_imputed"] = int(null_rating)
    print(f"  ✓ driver_rating imputed : {null_rating:,} nulls → city median")

    # 3e. Impute null pickup_zone — random zone from city's zone list
    null_zone = df["pickup_zone"].isna().sum()
    def impute_zone(row):
        if pd.isna(row["pickup_zone"]):
            zones = CITY_ZONES.get(row["city"], ["Unknown"])
            return np.random.choice(zones)
        return row["pickup_zone"]
    if null_zone > 0:
        np.random.seed(42)
        df["pickup_zone"] = df.apply(impute_zone, axis=1)
    report["pickup_zone_imputed"] = int(null_zone)
    print(f"  ✓ pickup_zone imputed   : {null_zone:,} nulls → random city zone")

    # 3f. Cap outlier fares (Winsorisation — don't drop, just cap)
    outlier_fare = ((df["fare_inr"] < FARE_MIN) | (df["fare_inr"] > FARE_MAX)).sum()
    df["fare_inr"] = df["fare_inr"].clip(lower=FARE_MIN, upper=FARE_MAX)
    report["fare_outliers_capped"] = int(outlier_fare)
    print(f"  ✓ fare_inr capped       : {outlier_fare:,} outliers → [₹{FARE_MIN}, ₹{FARE_MAX}]")

    # 3g. Cap outlier distances
    outlier_dist = ((df["distance_km"] < DISTANCE_MIN) | (df["distance_km"] > DISTANCE_MAX)).sum()
    df["distance_km"] = df["distance_km"].clip(lower=DISTANCE_MIN, upper=DISTANCE_MAX)
    report["distance_outliers_capped"] = int(outlier_dist)
    print(f"  ✓ distance_km capped    : {outlier_dist:,} outliers → [{DISTANCE_MIN}, {DISTANCE_MAX}] km")

    # 3h. Cap outlier wait times
    outlier_wait = (df["wait_time_min"] > WAIT_MAX).sum()
    df["wait_time_min"] = df["wait_time_min"].clip(lower=WAIT_MIN, upper=WAIT_MAX)
    report["wait_outliers_capped"] = int(outlier_wait)
    print(f"  ✓ wait_time_min capped  : {outlier_wait:,} outliers → [{WAIT_MIN}, {WAIT_MAX}] min")

    # 3i. Clip surge and rating to valid ranges
    df["surge_multiplier"] = df["surge_multiplier"].clip(SURGE_MIN, SURGE_MAX)
    df["driver_rating"]    = df["driver_rating"].clip(RATING_MIN, RATING_MAX)

    # 3j. Final null check — should be zero
    remaining_nulls = df.isnull().sum().sum()
    report["remaining_nulls"] = int(remaining_nulls)
    if remaining_nulls == 0:
        print(f"  ✓ Null check            : 0 nulls remaining")
    else:
        print(f"  ✗ WARNING: {remaining_nulls} nulls still present")

    n_end = len(df)
    report["rows_before"] = n_start
    report["rows_after"]  = n_end
    report["rows_removed"] = n_start - n_end
    print(f"\n  Rows: {n_start:,} → {n_end:,}  (removed {n_start - n_end:,})")

    return df.reset_index(drop=True), report


# ── Stage 4: Feature Engineering ──────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print("  STAGE 4 — Feature Engineering")
    print(f"{'='*60}")

    # ── Time features ──────────────────────────────────────────────────────
    df["hour"]          = df["timestamp"].dt.hour
    df["day_of_week"]   = df["timestamp"].dt.dayofweek        # 0=Mon, 6=Sun
    df["day_name"]      = df["timestamp"].dt.day_name()
    df["month"]         = df["timestamp"].dt.month
    df["month_name"]    = df["timestamp"].dt.month_name()
    df["week_of_year"]  = df["timestamp"].dt.isocalendar().week.astype(int)
    df["quarter"]       = df["timestamp"].dt.quarter
    df["date"]          = df["timestamp"].dt.date
    print("  ✓ Time features         : hour, day_of_week, month, quarter, week_of_year")

    # ── Peak hour flag ─────────────────────────────────────────────────────
    # Morning rush: 7–9 AM | Evening rush: 5–8 PM
    df["is_peak_hour"] = df["hour"].isin([7, 8, 9, 17, 18, 19, 20]).astype(int)
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
    df["is_night"]     = df["hour"].isin([22, 23, 0, 1, 2, 3, 4]).astype(int)
    df["is_late_night"]= df["hour"].isin([0, 1, 2, 3, 4]).astype(int)
    print("  ✓ Period flags          : is_peak_hour, is_weekend, is_night, is_late_night")

    # ── Season (India-specific) ────────────────────────────────────────────
    def get_season(month: int) -> str:
        if month in (12, 1, 2):   return "winter"
        if month in (3, 4, 5):    return "summer"
        if month in (6, 7, 8, 9): return "monsoon"
        return "post_monsoon"                          # Oct–Nov

    df["season"] = df["month"].map(get_season)
    print("  ✓ Season                : winter / summer / monsoon / post_monsoon")

    # ── Rain flag ──────────────────────────────────────────────────────────
    df["is_raining"]      = df["weather"].isin(["light_rain", "heavy_rain"]).astype(int)
    df["is_heavy_rain"]   = (df["weather"] == "heavy_rain").astype(int)
    df["is_fog"]          = (df["weather"] == "fog").astype(int)
    print("  ✓ Weather flags         : is_raining, is_heavy_rain, is_fog")

    # ── Fare per km ────────────────────────────────────────────────────────
    df["fare_per_km"] = (df["fare_inr"] / df["distance_km"]).round(2)
    print("  ✓ fare_per_km           : fare_inr / distance_km")

    # ── Demand pressure score ──────────────────────────────────────────────
    # Composite score: surge + peak + rain + festival + IPL
    df["demand_pressure"] = (
        df["surge_multiplier"]
        + df["is_peak_hour"] * 0.5
        + df["is_raining"]   * 0.4
        + df["is_festival"].astype(int) * 0.6
        + df["is_ipl_day"].astype(int)  * 0.5
    ).round(3)
    print("  ✓ demand_pressure       : composite demand score")

    # ── High surge flag ────────────────────────────────────────────────────
    df["is_high_surge"] = (df["surge_multiplier"] >= 2.0).astype(int)
    df["surge_band"] = pd.cut(
        df["surge_multiplier"],
        bins=[0, 1.0, 1.5, 2.0, 2.5, 5.0],
        labels=["no_surge", "low", "medium", "high", "extreme"],
        right=True
    ).astype(str)
    print("  ✓ Surge features        : is_high_surge, surge_band")

    # ── Wait time bands ────────────────────────────────────────────────────
    df["wait_band"] = pd.cut(
        df["wait_time_min"],
        bins=[0, 3, 6, 10, 15, 120],
        labels=["very_fast", "fast", "normal", "slow", "very_slow"],
        right=True
    ).astype(str)
    print("  ✓ wait_band             : very_fast / fast / normal / slow / very_slow")

    # ── Hourly ride volume (demand proxy) ──────────────────────────────────
    # Count rides per city+date+hour — useful as a demand feature for ML
    hourly_vol = (
        df.groupby(["city", "date", "hour"])
        .size()
        .reset_index(name="hourly_ride_volume")
    )
    df = df.merge(hourly_vol, on=["city", "date", "hour"], how="left")
    print("  ✓ hourly_ride_volume    : rides per city × date × hour")

    # ── High demand flag ───────────────────────────────────────────────────
    city_vol_75 = df.groupby("city")["hourly_ride_volume"].quantile(0.75)
    df["is_high_demand"] = df.apply(
        lambda r: int(r["hourly_ride_volume"] >= city_vol_75[r["city"]]), axis=1
    )
    print("  ✓ is_high_demand        : hourly volume > 75th percentile for city")

    # ── Event day composite ────────────────────────────────────────────────
    df["is_event_day"] = (
        df["is_festival"].astype(int) | df["is_ipl_day"].astype(int)
    )
    print("  ✓ is_event_day          : festival OR IPL match day")

    # ── Time since midnight (continuous) ──────────────────────────────────
    df["minutes_since_midnight"] = df["hour"] * 60 + df["timestamp"].dt.minute
    print("  ✓ minutes_since_midnight: continuous time feature for ML")

    total_features = 20
    print(f"\n  Total new features added: {total_features}")
    return df


# ── Stage 5: Encode Categoricals ──────────────────────────────────────────────

def encode(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    print(f"\n{'='*60}")
    print("  STAGE 5 — Encode Categoricals")
    print(f"{'='*60}")

    encodings = {}

    # Label encode city
    city_map = {c: i for i, c in enumerate(sorted(df["city"].unique()))}
    df["city_code"] = df["city"].map(city_map)
    encodings["city"] = city_map
    print(f"  ✓ city_code      : {city_map}")

    # Label encode vehicle_type
    vt_map = {v: i for i, v in enumerate(sorted(df["vehicle_type"].unique()))}
    df["vehicle_code"] = df["vehicle_type"].map(vt_map)
    encodings["vehicle_type"] = vt_map
    print(f"  ✓ vehicle_code   : {vt_map}")

    # Label encode weather
    w_map = {w: i for i, w in enumerate(sorted(df["weather"].unique()))}
    df["weather_code"] = df["weather"].map(w_map)
    encodings["weather"] = w_map
    print(f"  ✓ weather_code   : {w_map}")

    # Label encode season
    s_map = {"winter": 0, "summer": 1, "monsoon": 2, "post_monsoon": 3}
    df["season_code"] = df["season"].map(s_map)
    encodings["season"] = s_map
    print(f"  ✓ season_code    : {s_map}")

    return df, encodings


# ── Stage 6: Store ────────────────────────────────────────────────────────────

def store(df: pd.DataFrame, csv_path: str, db_path: str) -> None:
    print(f"\n{'='*60}")
    print("  STAGE 6 — Store")
    print(f"{'='*60}")

    # Save processed CSV
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    # Convert date column to string for CSV compatibility
    df_save = df.copy()
    df_save["date"] = df_save["date"].astype(str)
    df_save.to_csv(csv_path, index=False)
    size_mb = os.path.getsize(csv_path) / (1024 * 1024)
    print(f"  ✓ CSV saved      : {csv_path}  ({size_mb:.1f} MB)")

    # Save to DuckDB
    try:
        import duckdb
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        con = duckdb.connect(db_path)
        con.execute("DROP TABLE IF EXISTS rides")
        con.execute("CREATE TABLE rides AS SELECT * FROM df_save")
        row_count = con.execute("SELECT COUNT(*) FROM rides").fetchone()[0]
        con.close()
        print(f"  ✓ DuckDB saved   : {db_path}  ({row_count:,} rows in 'rides' table)")
    except ImportError:
        print("  ⚠ DuckDB not installed — skipping DB storage")
        print("    Install with: pip install duckdb")


# ── Stage 7: Quality Report ───────────────────────────────────────────────────

def quality_report(df: pd.DataFrame, validation: dict, cleaning: dict,
                   encodings: dict, report_path: str) -> None:
    print(f"\n{'='*60}")
    print("  STAGE 7 — Quality Report")
    print(f"{'='*60}")

    report = {
        "generated_at": datetime.now().isoformat(),
        "final_shape": {"rows": len(df), "columns": df.shape[1]},
        "date_range": {
            "start": str(df["timestamp"].min().date()),
            "end":   str(df["timestamp"].max().date()),
        },
        "validation": validation,
        "cleaning":   cleaning,
        "encodings":  encodings,
        "final_columns": list(df.columns),
        "final_stats": {
            col: {
                "mean":   round(df[col].mean(), 3),
                "median": round(df[col].median(), 3),
                "std":    round(df[col].std(), 3),
                "min":    round(df[col].min(), 3),
                "max":    round(df[col].max(), 3),
            }
            for col in ["wait_time_min", "surge_multiplier", "distance_km",
                        "fare_inr", "driver_rating", "demand_pressure",
                        "fare_per_km", "hourly_ride_volume"]
        },
        "feature_distributions": {
            "city":         df["city"].value_counts().to_dict(),
            "vehicle_type": df["vehicle_type"].value_counts().to_dict(),
            "weather":      df["weather"].value_counts().to_dict(),
            "season":       df["season"].value_counts().to_dict(),
            "surge_band":   df["surge_band"].value_counts().to_dict(),
            "wait_band":    df["wait_band"].value_counts().to_dict(),
        },
        "flag_rates": {
            "is_peak_hour":    round(df["is_peak_hour"].mean(), 4),
            "is_weekend":      round(df["is_weekend"].mean(), 4),
            "is_raining":      round(df["is_raining"].mean(), 4),
            "is_festival":     round(df["is_festival"].mean(), 4),
            "is_ipl_day":      round(df["is_ipl_day"].mean(), 4),
            "is_high_surge":   round(df["is_high_surge"].mean(), 4),
            "is_high_demand":  round(df["is_high_demand"].mean(), 4),
            "is_event_day":    round(df["is_event_day"].mean(), 4),
            "is_completed":    round(df["is_completed"].mean(), 4),
        }
    }

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"  ✓ Report saved   : {report_path}")
    print(f"\n  Final dataset    : {len(df):,} rows × {df.shape[1]} columns")
    print(f"\n  Flag rates:")
    for flag, rate in report["flag_rates"].items():
        print(f"    {flag:<22} {rate*100:>5.1f}%")

    print(f"\n  Key statistics (clean data):")
    for col, stats in report["final_stats"].items():
        print(f"    {col:<22}  median={stats['median']}  std={stats['std']}")


# ── Summary printer ───────────────────────────────────────────────────────────

def print_final_summary(df: pd.DataFrame) -> None:
    print(f"\n{'='*60}")
    print("  PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"\n  Output columns ({df.shape[1]} total):")

    original = ["ride_id","timestamp","city","pickup_zone","vehicle_type",
                "wait_time_min","surge_multiplier","distance_km","fare_inr",
                "driver_rating","is_completed","weather","is_festival","is_ipl_day"]
    engineered = [c for c in df.columns if c not in original]

    print(f"    Original  ({len(original)}) : {', '.join(original)}")
    print(f"    Engineered ({len(engineered)}): {', '.join(engineered)}")
    print(f"\n  Next step → open notebooks/02_eda.ipynb  (Phase 3)\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_pipeline(input_path: str, output_csv: str, output_db: str,
                 report_path: str) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print("  Smart City Mobility — Data Pipeline")
    print(f"  Input  : {input_path}")
    print(f"  Output : {output_csv}")
    print(f"{'='*60}")

    df                    = load_raw(input_path)
    validation_report     = validate_schema(df)
    df, cleaning_report   = clean(df)
    df                    = engineer_features(df)
    df, encodings         = encode(df)
    store(df, output_csv, output_db)
    quality_report(df, validation_report, cleaning_report, encodings, report_path)
    print_final_summary(df)

    return df


def main():
    parser = argparse.ArgumentParser(description="Smart City Mobility data pipeline")
    parser.add_argument("--input",   default=RAW_PATH,       help="Raw CSV path")
    parser.add_argument("--output",  default=PROCESSED_PATH,  help="Processed CSV path")
    parser.add_argument("--db",      default=DB_PATH,         help="DuckDB path")
    parser.add_argument("--report",  default=REPORT_PATH,     help="Quality report JSON path")
    args = parser.parse_args()

    run_pipeline(args.input, args.output, args.db, args.report)


if __name__ == "__main__":
    main()
