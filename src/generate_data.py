"""
generate_data.py
----------------
Synthetic ride dataset generator for the Smart City Mobility Intelligence System.

Generates 500,000+ realistic ride records across 5 Indian metro cities with:
  - Time-aware demand patterns (rush hours, late night, weekends)
  - City-specific zone geography
  - Weather effects (monsoon season, fog in Delhi winters)
  - Festival and IPL event demand spikes
  - Deliberate data quality issues for EDA realism
  - Realistic fare calculations per vehicle type

Usage:
    python src/generate_data.py
    python src/generate_data.py --rows 100000 --output data/raw/rides_sample.csv
"""

import argparse
import os
import uuid
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc="", unit="", ncols=None, **kwargs):
        total = kwargs.get("total", None)
        n = total or (len(iterable) if hasattr(iterable, "__len__") else None)
        for i, item in enumerate(iterable):
            if n and (i % max(1, n // 20) == 0 or i == n - 1):
                pct = int((i + 1) / n * 100)
                print(f"\r  {desc}: {pct:>3}% ({i+1:,}/{n:,})", end="", flush=True)
            yield item
        if n:
            print()

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ── City & Zone Configuration ─────────────────────────────────────────────────
CITY_CONFIG = {
    "Delhi NCR": {
        "zones": [
            "Connaught Place", "Karol Bagh", "Lajpat Nagar", "Saket",
            "Dwarka", "Rohini", "Noida Sector 18", "Gurgaon Cyber City",
            "Hauz Khas", "Nehru Place", "Janakpuri", "Vasant Kunj",
            "Greater Noida", "Faridabad", "Ghaziabad",
        ],
        "base_demand": 1.30,
        "base_fare_per_km": 12,
        "monsoon_intensity": 0.70,
        "fog_season": True,
    },
    "Mumbai": {
        "zones": [
            "Bandra", "Andheri West", "Andheri East", "Juhu",
            "Borivali", "Thane", "Kurla", "Dadar",
            "Lower Parel", "BKC", "Powai", "Malad",
            "Navi Mumbai", "Vashi", "Chembur",
        ],
        "base_demand": 1.25,
        "base_fare_per_km": 14,
        "monsoon_intensity": 1.40,
        "fog_season": False,
    },
    "Bengaluru": {
        "zones": [
            "Koramangala", "Indiranagar", "Whitefield", "Electronic City",
            "HSR Layout", "Jayanagar", "Hebbal", "Marathahalli",
            "MG Road", "Yelahanka", "Bannerghatta", "Sarjapur",
            "BTM Layout", "Bellandur", "Rajajinagar",
        ],
        "base_demand": 1.10,
        "base_fare_per_km": 13,
        "monsoon_intensity": 0.90,
        "fog_season": False,
    },
    "Chennai": {
        "zones": [
            "T. Nagar", "Anna Nagar", "Adyar", "Velachery",
            "Tambaram", "Porur", "Nungambakkam", "Mylapore",
            "Perambur", "Chromepet", "Sholinganallur", "OMR",
            "Guindy", "Kodambakkam", "Koyambedu",
        ],
        "base_demand": 0.95,
        "base_fare_per_km": 11,
        "monsoon_intensity": 1.10,
        "fog_season": False,
    },
    "Hyderabad": {
        "zones": [
            "Hitech City", "Madhapur", "Gachibowli", "Banjara Hills",
            "Jubilee Hills", "Secunderabad", "Ameerpet", "Kukatpally",
            "LB Nagar", "Dilsukhnagar", "Uppal", "Kompally",
            "Kondapur", "Manikonda", "Miyapur",
        ],
        "base_demand": 0.90,
        "base_fare_per_km": 11,
        "monsoon_intensity": 0.80,
        "fog_season": False,
    },
}

# ── Vehicle Type Configuration ────────────────────────────────────────────────
VEHICLE_CONFIG = {
    "bike_taxi": {"share": 0.28, "base_fare": 25, "per_km": 8,  "wait_base": 3.0},
    "auto":      {"share": 0.24, "base_fare": 30, "per_km": 11, "wait_base": 4.5},
    "economy":   {"share": 0.30, "base_fare": 50, "per_km": 14, "wait_base": 5.5},
    "premium":   {"share": 0.10, "base_fare": 100,"per_km": 22, "wait_base": 7.0},
    "shared":    {"share": 0.08, "base_fare": 20, "per_km": 6,  "wait_base": 8.0},
}

# ── Festival Calendar ─────────────────────────────────────────────────────────
FESTIVAL_DATES = {
    "2023-01-26", "2023-03-08", "2023-04-04", "2023-04-14",
    "2023-08-15", "2023-09-19", "2023-10-02", "2023-10-24",
    "2023-11-12", "2023-11-13", "2023-11-27", "2023-12-25",
    "2023-12-31", "2024-01-01", "2024-01-22", "2024-01-26",
    "2024-03-25", "2024-04-11", "2024-05-23", "2024-06-17",
}

# ── IPL Match Dates Per City ──────────────────────────────────────────────────
IPL_MATCHES = {
    "Delhi NCR": {
        "2023-04-01","2023-04-08","2023-04-15","2023-04-22","2023-04-29",
        "2023-05-06","2023-05-13","2024-03-24","2024-04-06","2024-04-14",
        "2024-04-21","2024-04-28","2024-05-05","2024-05-12",
    },
    "Mumbai": {
        "2023-04-02","2023-04-09","2023-04-16","2023-04-23","2023-04-30",
        "2023-05-07","2023-05-14","2024-03-23","2024-03-30","2024-04-07",
        "2024-04-13","2024-04-20","2024-04-27","2024-05-04",
    },
    "Bengaluru": {
        "2023-04-03","2023-04-10","2023-04-17","2023-04-24","2023-05-01",
        "2023-05-08","2024-03-22","2024-03-29","2024-04-08","2024-04-15",
        "2024-04-22","2024-04-29","2024-05-06",
    },
    "Chennai": {
        "2023-04-04","2023-04-11","2023-04-18","2023-04-25","2023-05-02",
        "2023-05-09","2023-05-16","2024-03-25","2024-04-01","2024-04-09",
        "2024-04-16","2024-04-23","2024-04-30","2024-05-07",
    },
    "Hyderabad": {
        "2023-04-05","2023-04-12","2023-04-19","2023-04-26","2023-05-03",
        "2023-05-10","2024-03-26","2024-04-02","2024-04-10","2024-04-17",
        "2024-04-24","2024-05-01","2024-05-08",
    },
}

# ── Demand Curve (hour → multiplier) ─────────────────────────────────────────
HOUR_DEMAND = {
    0: 0.40, 1: 0.20, 2: 0.15, 3: 0.12, 4: 0.18,
    5: 0.45, 6: 0.75, 7: 1.20, 8: 1.50, 9: 1.10,
    10: 0.80, 11: 0.75, 12: 0.90, 13: 1.00, 14: 0.85,
    15: 0.80, 16: 0.90, 17: 1.40, 18: 1.70, 19: 1.60,
    20: 1.30, 21: 1.10, 22: 0.90, 23: 0.60,
}

WEATHER_DEMAND = {"clear": 1.00, "light_rain": 1.35, "heavy_rain": 1.75, "fog": 1.20}
WEATHER_WAIT   = {"clear": 1.00, "light_rain": 1.25, "heavy_rain": 1.80, "fog": 1.40}


# ── Helper Functions ──────────────────────────────────────────────────────────

def get_weather(ts: datetime, city: str, rng: np.random.Generator) -> str:
    month = ts.month
    cfg = CITY_CONFIG[city]

    if city == "Delhi NCR" and month in (12, 1, 2):
        if rng.random() < 0.25:
            return "fog"

    if city == "Mumbai" and month in (6, 7, 8, 9):
        r, i = rng.random(), cfg["monsoon_intensity"]
        if r < 0.15 * i: return "heavy_rain"
        if r < 0.45 * i: return "light_rain"

    if city == "Chennai" and month in (10, 11, 12):
        r, i = rng.random(), cfg["monsoon_intensity"]
        if r < 0.12 * i: return "heavy_rain"
        if r < 0.40 * i: return "light_rain"

    if month in (6, 7, 8, 9):
        r, i = rng.random(), cfg["monsoon_intensity"]
        if r < 0.08 * i: return "heavy_rain"
        if r < 0.30 * i: return "light_rain"

    return "clear"


def get_surge(d_mult: float, weather: str, is_festival: bool, is_ipl: bool,
              rng: np.random.Generator) -> float:
    if d_mult >= 1.6:
        surge = rng.uniform(1.5, 2.5)
    elif d_mult >= 1.2:
        surge = rng.uniform(1.0, 1.75)
    else:
        surge = rng.uniform(1.0, 1.25)

    if weather == "heavy_rain": surge += rng.uniform(0.5, 1.2)
    elif weather == "light_rain": surge += rng.uniform(0.1, 0.4)
    elif weather == "fog": surge += rng.uniform(0.15, 0.35)
    if is_festival: surge += rng.uniform(0.3, 0.8)
    if is_ipl:      surge += rng.uniform(0.4, 1.0)

    return round(min(surge, 4.0) * 4) / 4   # snap to 0.25 steps


def get_wait(vehicle_type: str, d_mult: float, weather: str,
             zone_idx: int, n_zones: int, rng: np.random.Generator) -> float:
    base = VEHICLE_CONFIG[vehicle_type]["wait_base"]
    demand_penalty = max(0, (d_mult - 1.0) * 3.5)
    zone_penalty   = (zone_idx / n_zones) * 4.0
    noise          = rng.normal(0, 1.2)
    wait = (base + demand_penalty + zone_penalty) * WEATHER_WAIT[weather] + noise
    return max(1.0, round(wait, 1))


def get_fare(vehicle_type: str, dist_km: float, surge: float,
             rng: np.random.Generator) -> float:
    cfg  = VEHICLE_CONFIG[vehicle_type]
    base = cfg["base_fare"] + cfg["per_km"] * dist_km
    return round(base * surge * rng.uniform(0.95, 1.05), 1)


def inject_quality_issues(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    n = len(df)
    df.loc[rng.choice(n, size=int(n * 0.020), replace=False), "wait_time_min"]    = np.nan
    df.loc[rng.choice(n, size=int(n * 0.010), replace=False), "driver_rating"]    = np.nan
    df.loc[rng.choice(n, size=int(n * 0.005), replace=False), "pickup_zone"]      = np.nan
    bad_fare_idx = rng.choice(n, size=int(n * 0.005), replace=False)
    df.loc[bad_fare_idx, "fare_inr"] = rng.choice([0.0, 9999.0, -1.0], size=len(bad_fare_idx))

    dup_idx    = rng.choice(n, size=int(n * 0.003), replace=False)
    duplicates = df.iloc[dup_idx].copy()
    df = pd.concat([df, duplicates], ignore_index=True)
    return df.sample(frac=1, random_state=SEED).reset_index(drop=True)


# ── Main Generator ────────────────────────────────────────────────────────────

def generate_rides(n_rows: int = 500_000) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print(f"  Smart City Mobility — Synthetic Data Generator")
    print(f"  Target rows : {n_rows:,}")
    print(f"  Date range  : Jan 2023 – Jun 2024  (18 months)")
    print(f"  Cities      : {', '.join(CITY_CONFIG.keys())}")
    print(f"{'='*60}\n")

    rng = np.random.default_rng(SEED)

    start_date    = datetime(2023, 1, 1)
    end_date      = datetime(2024, 6, 30, 23, 59, 59)
    total_seconds = int((end_date - start_date).total_seconds())

    cities        = list(CITY_CONFIG.keys())
    city_weights  = np.array([CITY_CONFIG[c]["base_demand"] for c in cities])
    city_weights /= city_weights.sum()

    vehicle_types   = list(VEHICLE_CONFIG.keys())
    vehicle_weights = np.array([VEHICLE_CONFIG[v]["share"] for v in vehicle_types])
    vehicle_weights /= vehicle_weights.sum()

    peak_hours = [7, 8, 9, 17, 18, 19, 20]

    records = []

    for _ in tqdm(range(n_rows), desc="Generating rides", unit="rides", ncols=70):

        # 1. City
        city  = str(rng.choice(cities, p=city_weights))
        zones = CITY_CONFIG[city]["zones"]
        n_z   = len(zones)

        # 2. Timestamp with demand-based hour bias
        ts    = start_date + timedelta(seconds=int(rng.integers(0, total_seconds)))
        hour  = ts.hour
        d_mult = HOUR_DEMAND[hour]
        if rng.random() > d_mult / 1.7:
            hour   = int(rng.choice(peak_hours))
            ts     = ts.replace(hour=hour, minute=int(rng.integers(0, 60)))
            d_mult = HOUR_DEMAND[hour]

        date_str = ts.strftime("%Y-%m-%d")

        # 3. Weekend modifier
        if ts.weekday() >= 5:
            d_mult *= 0.85
            if 18 <= hour <= 23:
                d_mult *= 1.25

        # 4. Festival & IPL
        is_festival = date_str in FESTIVAL_DATES
        is_ipl      = date_str in IPL_MATCHES.get(city, set())
        if is_festival: d_mult *= float(rng.uniform(1.4, 2.2))
        if is_ipl:
            d_mult *= float(rng.uniform(1.8, 3.0) if 21 <= hour <= 23 else rng.uniform(1.2, 1.6))

        # 5. Weather
        weather = get_weather(ts, city, rng)
        d_mult *= WEATHER_DEMAND[weather]

        # 6. Zone (central zones weighted 2×)
        zone_probs = np.ones(n_z)
        zone_probs[:5] *= 2.0
        zone_probs /= zone_probs.sum()
        zone_idx    = int(rng.choice(n_z, p=zone_probs))
        pickup_zone = zones[zone_idx]

        # 7. Vehicle type
        vehicle_type = str(rng.choice(vehicle_types, p=vehicle_weights))

        # 8. Core metrics
        surge    = get_surge(d_mult, weather, is_festival, is_ipl, rng)
        dist_km  = float(np.clip(rng.lognormal(1.8, 0.7), 1.0, 13.0))
        fare     = get_fare(vehicle_type, dist_km, surge, rng)
        wait     = get_wait(vehicle_type, d_mult, weather, zone_idx, n_z, rng)

        # 9. Driver rating
        rating = float(np.clip(
            rng.normal(4.3, 0.3) - max(0, (surge - 1.5) * 0.2)
            - (0.15 if weather in ("heavy_rain", "fog") else 0),
            1.0, 5.0
        ))
        rating = round(rating * 2) / 2

        # 10. Completion (cancellations spike in heavy rain + high surge)
        cancel_prob = 0.05
        if weather == "heavy_rain": cancel_prob += 0.12
        if surge >= 2.5:            cancel_prob += 0.08
        is_completed = bool(rng.random() > cancel_prob)

        records.append({
            "ride_id":          f"RD{uuid.uuid4().hex[:12].upper()}",
            "timestamp":        ts.strftime("%Y-%m-%d %H:%M:%S"),
            "city":             city,
            "pickup_zone":      pickup_zone,
            "vehicle_type":     vehicle_type,
            "wait_time_min":    wait,
            "surge_multiplier": surge,
            "distance_km":      round(dist_km, 2),
            "fare_inr":         fare,
            "driver_rating":    rating,
            "is_completed":     is_completed,
            "weather":          weather,
            "is_festival":      is_festival,
            "is_ipl_day":       is_ipl,
        })

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"\n✓ Generated {len(df):,} raw records")
    return df


def print_summary(df: pd.DataFrame) -> None:
    print(f"\n{'='*60}")
    print("  DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"  Shape       : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Date range  : {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")

    print(f"\n  Rows per city:")
    for city, count in df["city"].value_counts().items():
        print(f"    {city:<22} {count:>8,}  ({count/len(df)*100:.1f}%)")

    print(f"\n  Vehicle type split:")
    for vt, count in df["vehicle_type"].value_counts().items():
        print(f"    {vt:<14} {count:>8,}  ({count/len(df)*100:.1f}%)")

    print(f"\n  Key statistics:")
    print(f"    Median fare_inr      : ₹{df['fare_inr'].median():.0f}")
    print(f"    Median wait_time_min : {df['wait_time_min'].median():.1f} min")
    print(f"    Mean surge_mult      : {df['surge_multiplier'].mean():.2f}×")
    print(f"    Completion rate      : {df['is_completed'].mean()*100:.1f}%")
    print(f"    Festival rides       : {df['is_festival'].sum():,}  ({df['is_festival'].mean()*100:.1f}%)")
    print(f"    IPL-day rides        : {df['is_ipl_day'].sum():,}  ({df['is_ipl_day'].mean()*100:.1f}%)")

    print(f"\n  Null counts (quality issues injected):")
    null_counts = df.isnull().sum()
    for col, n in null_counts[null_counts > 0].items():
        print(f"    {col:<22} {n:>6,} nulls")

    print(f"\n  Outlier fares (<= 0 or > ₹5000):")
    outliers = ((df["fare_inr"] <= 0) | (df["fare_inr"] > 5000)).sum()
    print(f"    {outliers:,} rows")
    print(f"\n{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Generate Smart City Mobility synthetic dataset")
    parser.add_argument("--rows",   type=int, default=500_000,
                        help="Number of ride records to generate (default: 500000)")
    parser.add_argument("--output", type=str, default="data/raw/rides.csv",
                        help="Output CSV file path")
    parser.add_argument("--clean",  action="store_true",
                        help="Skip injecting data quality issues")
    args = parser.parse_args()

    df = generate_rides(n_rows=args.rows)

    if not args.clean:
        print("\nInjecting data quality issues...")
        rng = np.random.default_rng(SEED + 1)
        df  = inject_quality_issues(df, rng)
        print(f"  Final dataset size: {len(df):,} rows (includes injected duplicates)")

    print_summary(df)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"✓ Saved → {args.output}  ({size_mb:.1f} MB)")
    print(f"\n  Next step → python src/pipeline.py\n")


if __name__ == "__main__":
    main()
