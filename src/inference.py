"""
inference.py
------------
Prediction API for the Smart City Mobility Intelligence System.

Loads all trained models and exposes three simple functions:
  predict_demand()       -> predicted rides for a city/hour
  predict_wait_time()    -> predicted wait time in minutes
  predict_surge_risk()   -> surge probability (0–1) + flag

Used by the Streamlit dashboard (Phase 6) and any other consumer.

Usage:
    from src.inference import MobilityPredictor
    predictor = MobilityPredictor()
    result = predictor.predict_all(city='Mumbai', hour=18, ...)
"""

import json
import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

MODEL_DIR = Path("models")

# ── Encodings (must match pipeline.py) ───────────────────────────────────────
CITY_CODES    = {'Bengaluru': 0, 'Chennai': 1, 'Delhi NCR': 2, 'Hyderabad': 3, 'Mumbai': 4}
VEHICLE_CODES = {'auto': 0, 'bike_taxi': 1, 'economy': 2, 'premium': 3, 'shared': 4}
WEATHER_CODES = {'clear': 0, 'fog': 1, 'heavy_rain': 2, 'light_rain': 3}
SEASON_CODES  = {'winter': 0, 'summer': 1, 'monsoon': 2, 'post_monsoon': 3}

CITY_NAMES    = {v: k for k, v in CITY_CODES.items()}
VEHICLE_NAMES = {v: k for k, v in VEHICLE_CODES.items()}

# Demand model feature columns (must match train_demand_model.py)
DEMAND_FEATURES = [
    "hour", "day_of_week", "month", "quarter",
    "is_peak_hour", "is_weekend",
    "is_raining", "is_heavy_rain", "is_fog",
    "is_festival", "is_ipl_day",
    "season_code",
    "rides_lag_1d", "rides_lag_7d",
    "rides_roll_3d", "rides_roll_7d",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]

WAIT_FEATURES = [
    "hour", "day_of_week", "month", "quarter",
    "is_peak_hour", "is_weekend", "is_night",
    "is_raining", "is_heavy_rain", "is_fog",
    "is_festival", "is_ipl_day",
    "surge_multiplier", "demand_pressure", "hourly_ride_volume",
    "city_code", "vehicle_code", "weather_code", "season_code",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]

SURGE_FEATURES = [
    "hour", "day_of_week", "month", "quarter",
    "is_peak_hour", "is_weekend", "is_night",
    "is_raining", "is_heavy_rain", "is_fog",
    "is_festival", "is_ipl_day",
    "demand_pressure", "hourly_ride_volume",
    "city_code", "weather_code", "season_code",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]


def _get_season(month: int) -> str:
    if month in (12, 1, 2):    return "winter"
    if month in (3, 4, 5):     return "summer"
    if month in (6, 7, 8, 9):  return "monsoon"
    return "post_monsoon"


def _cyclical(value: float, period: float) -> tuple[float, float]:
    return (
        float(np.sin(2 * np.pi * value / period)),
        float(np.cos(2 * np.pi * value / period)),
    )


class MobilityPredictor:
    """
    Loads all trained models and provides a unified prediction interface.

    Example:
        predictor = MobilityPredictor()
        result = predictor.predict_all(
            city='Mumbai', hour=18, day_of_week=1, month=7,
            weather='heavy_rain', vehicle_type='economy',
            is_festival=False, is_ipl_day=False,
        )
        print(result)
    """

    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.models    = {}
        self.meta      = {}
        self._load_models()

    def _load_models(self) -> None:
        model_files = {
            "wait_time":    "wait_time_model.pkl",
            "surge":        "surge_model.pkl",
            "cancellation": "cancellation_model.pkl",
        }
        demand_files = {
            city.replace(' ', '_'): f"demand_{city.replace(' ', '_')}.pkl"
            for city in CITY_CODES
        }

        # Load wait/surge/cancel models
        for name, fname in model_files.items():
            path = self.model_dir / fname
            if path.exists():
                with open(path, 'rb') as f:
                    self.models[name] = pickle.load(f)
                meta_path = self.model_dir / fname.replace('.pkl', '_meta.json')
                if meta_path.exists():
                    with open(meta_path) as f:
                        self.meta[name] = json.load(f)
            else:
                self.models[name] = None

        # Load demand models per city
        self.models["demand"] = {}
        for city_key, fname in demand_files.items():
            path = self.model_dir / fname
            if path.exists():
                with open(path, 'rb') as f:
                    self.models["demand"][city_key] = pickle.load(f)

    def _build_features(
        self,
        city: str,
        hour: int,
        day_of_week: int,
        month: int,
        weather: str,
        vehicle_type: str,
        is_festival: bool,
        is_ipl_day: bool,
        surge_multiplier: float = 1.5,
        hourly_ride_volume: int = 15,
        demand_pressure: float = 1.8,
        rides_lag_1d: float = 15.0,
        rides_lag_7d: float = 15.0,
        rides_roll_3d: float = 15.0,
        rides_roll_7d: float = 15.0,
    ) -> dict:
        """Build the full feature dict for all models."""
        season     = _get_season(month)
        hour_sin, hour_cos = _cyclical(hour, 24)
        dow_sin,  dow_cos  = _cyclical(day_of_week, 7)
        quarter = (month - 1) // 3 + 1

        return {
            "hour":               hour,
            "day_of_week":        day_of_week,
            "month":              month,
            "quarter":            quarter,
            "is_peak_hour":       int(hour in [7, 8, 9, 17, 18, 19, 20]),
            "is_weekend":         int(day_of_week >= 5),
            "is_night":           int(hour in [22, 23, 0, 1, 2, 3, 4]),
            "is_raining":         int(weather in ["light_rain", "heavy_rain"]),
            "is_heavy_rain":      int(weather == "heavy_rain"),
            "is_fog":             int(weather == "fog"),
            "is_festival":        int(is_festival),
            "is_ipl_day":         int(is_ipl_day),
            "surge_multiplier":   surge_multiplier,
            "demand_pressure":    demand_pressure,
            "hourly_ride_volume": hourly_ride_volume,
            "city_code":          CITY_CODES.get(city, 2),
            "vehicle_code":       VEHICLE_CODES.get(vehicle_type, 2),
            "weather_code":       WEATHER_CODES.get(weather, 0),
            "season_code":        SEASON_CODES.get(season, 0),
            "hour_sin":           hour_sin,
            "hour_cos":           hour_cos,
            "dow_sin":            dow_sin,
            "dow_cos":            dow_cos,
            "rides_lag_1d":       rides_lag_1d,
            "rides_lag_7d":       rides_lag_7d,
            "rides_roll_3d":      rides_roll_3d,
            "rides_roll_7d":      rides_roll_7d,
        }

    def predict_wait_time(self, features: dict) -> float:
        """Predict wait time in minutes."""
        if self.models.get("wait_time") is None:
            return 7.5  # fallback average
        X = pd.DataFrame([{k: features[k] for k in WAIT_FEATURES}])
        pred = self.models["wait_time"].predict(X)[0]
        return round(float(np.clip(pred, 0.5, 60.0)), 1)

    def predict_surge_risk(self, features: dict) -> dict:
        """Predict surge probability and classification."""
        if self.models.get("surge") is None:
            return {"probability": 0.3, "is_high_surge": False, "label": "Normal"}
        X    = pd.DataFrame([{k: features[k] for k in SURGE_FEATURES}])
        prob = float(self.models["surge"].predict_proba(X)[0][1])
        flag = prob >= 0.5
        label = (
            "Extreme surge risk" if prob >= 0.8 else
            "High surge risk"    if prob >= 0.6 else
            "Moderate risk"      if prob >= 0.4 else
            "Normal"
        )
        return {
            "probability":   round(prob, 3),
            "is_high_surge": bool(flag),
            "label":         label,
            "surge_estimate": round(1.0 + prob * 2.5, 2),
        }

    def predict_cancellation_risk(self, features: dict) -> dict:
        """Predict cancellation probability."""
        CANCEL_FEATURES = [
            "hour", "day_of_week", "month", "is_peak_hour", "is_weekend",
            "is_night", "is_raining", "is_heavy_rain", "is_fog",
            "is_festival", "is_ipl_day", "surge_multiplier", "demand_pressure",
            "city_code", "vehicle_code", "weather_code", "season_code",
        ]
        if self.models.get("cancellation") is None:
            return {"probability": 0.06, "label": "Low risk"}
        X    = pd.DataFrame([{k: features[k] for k in CANCEL_FEATURES}])
        prob = float(self.models["cancellation"].predict_proba(X)[0][1])
        label = (
            "High cancellation risk" if prob >= 0.2 else
            "Moderate risk"          if prob >= 0.12 else
            "Low risk"
        )
        return {"probability": round(prob, 3), "label": label}

    def predict_all(
        self,
        city: str,
        hour: int,
        day_of_week: int = 1,
        month: int = 6,
        weather: str = "clear",
        vehicle_type: str = "economy",
        is_festival: bool = False,
        is_ipl_day: bool = False,
        surge_multiplier: float = 1.5,
        hourly_ride_volume: int = 15,
        **kwargs,
    ) -> dict:
        """
        Run all 3 predictions and return a unified result dict.

        Returns:
            {
              'city': 'Mumbai',
              'hour': 18,
              'weather': 'heavy_rain',
              'wait_time_min': 9.2,
              'surge': {'probability': 0.72, 'label': 'High surge risk', ...},
              'cancellation': {'probability': 0.18, 'label': 'Moderate risk'},
              'demand_pressure': 2.3,
              'recommendations': [...],
            }
        """
        # Estimate demand_pressure from inputs
        dp = (
            surge_multiplier
            + (0.5 if hour in [7,8,9,17,18,19,20] else 0)
            + (0.4 if weather in ["light_rain","heavy_rain"] else 0)
            + (0.6 if is_festival else 0)
            + (0.5 if is_ipl_day  else 0)
        )

        features = self._build_features(
            city=city, hour=hour, day_of_week=day_of_week, month=month,
            weather=weather, vehicle_type=vehicle_type,
            is_festival=is_festival, is_ipl_day=is_ipl_day,
            surge_multiplier=surge_multiplier,
            hourly_ride_volume=hourly_ride_volume,
            demand_pressure=round(dp, 2),
        )

        wait      = self.predict_wait_time(features)
        surge     = self.predict_surge_risk(features)
        cancel    = self.predict_cancellation_risk(features)

        # Smart recommendations
        recs = []
        if wait > 10:
            recs.append(f"Dispatch additional drivers to {city} — predicted wait {wait:.0f} min")
        if surge["is_high_surge"]:
            recs.append(f"Surge alert ({surge['label']}) — pre-position fleet before {hour}:00")
        if weather == "heavy_rain":
            recs.append("Heavy rain: activate rain surge protocol, expect 3× cancellations")
        if is_festival:
            recs.append("Festival day: deploy surge standby pool from 6 PM onwards")
        if is_ipl_day and hour >= 21:
            recs.append("Post-IPL demand spike expected — activate 150+ standby vehicles")
        if not recs:
            recs.append("Demand is normal — maintain current fleet distribution")

        return {
            "city":             city,
            "hour":             hour,
            "weather":          weather,
            "vehicle_type":     vehicle_type,
            "is_festival":      is_festival,
            "is_ipl_day":       is_ipl_day,
            "wait_time_min":    wait,
            "surge":            surge,
            "cancellation":     cancel,
            "demand_pressure":  round(dp, 2),
            "recommendations":  recs,
        }

    def predict_hourly_profile(
        self,
        city: str,
        month: int = 6,
        day_of_week: int = 1,
        weather: str = "clear",
        is_festival: bool = False,
        is_ipl_day: bool = False,
    ) -> pd.DataFrame:
        """Predict wait time and surge risk for all 24 hours of a day."""
        results = []
        for hour in range(24):
            r = self.predict_all(
                city=city, hour=hour, day_of_week=day_of_week,
                month=month, weather=weather, is_festival=is_festival,
                is_ipl_day=is_ipl_day,
            )
            results.append({
                "hour":           hour,
                "wait_time_min":  r["wait_time_min"],
                "surge_prob":     r["surge"]["probability"],
                "surge_label":    r["surge"]["label"],
                "cancel_prob":    r["cancellation"]["probability"],
                "demand_pressure":r["demand_pressure"],
            })
        return pd.DataFrame(results)

    def model_summary(self) -> None:
        """Print a summary of all loaded models and their metrics."""
        print("\nLoaded models:")
        for name, meta in self.meta.items():
            print(f"  {name:<15} {meta.get('model_name',''):<22} {meta.get('metrics',{})}")
        demand_cities = list(self.models.get("demand", {}).keys())
        print(f"  demand models  : {len(demand_cities)} cities — {', '.join(demand_cities)}")


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading MobilityPredictor...")
    p = MobilityPredictor()
    p.model_summary()

    print("\n── Test 1: Mumbai, 6PM, heavy rain ──")
    r = p.predict_all(city="Mumbai", hour=18, month=7,
                       weather="heavy_rain", vehicle_type="economy")
    print(f"  Wait time   : {r['wait_time_min']} min")
    print(f"  Surge risk  : {r['surge']['label']} ({r['surge']['probability']:.0%})")
    print(f"  Cancellation: {r['cancellation']['label']} ({r['cancellation']['probability']:.0%})")
    print(f"  Recs: {r['recommendations']}")

    print("\n── Test 2: Delhi NCR, 8AM, clear, festival day ──")
    r2 = p.predict_all(city="Delhi NCR", hour=8, month=11,
                        weather="clear", is_festival=True)
    print(f"  Wait time   : {r2['wait_time_min']} min")
    print(f"  Surge risk  : {r2['surge']['label']} ({r2['surge']['probability']:.0%})")
    print(f"  Recs: {r2['recommendations']}")

    print("\n── Test 3: 24-hour profile for Bengaluru (Monday, monsoon) ──")
    profile = p.predict_hourly_profile("Bengaluru", month=7, day_of_week=0, weather="light_rain")
    print(profile[["hour","wait_time_min","surge_prob","surge_label"]].to_string(index=False))
