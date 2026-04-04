"""
train_demand_model.py
---------------------
Demand forecasting model for the Smart City Mobility Intelligence System.

Predicts hourly ride demand (rides per city per hour) using:
  - Baseline: Linear Regression
  - Model:    XGBoost / GradientBoostingRegressor
  - Eval:     Time-series cross-validation (no data leakage)
  - Explain:  Feature importance + SHAP (if available)
  - Track:    MLflow experiment logging

Target variable : rides  (count of rides per city × date × hour)
Evaluation       : MAE, RMSE, MAPE, R²

Usage:
    python src/train_demand_model.py
    python src/train_demand_model.py --city "Delhi NCR"
    python src/train_demand_model.py --all-cities
"""

import argparse
import json
import os
import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')

# ── Paths ────────────────────────────────────────────────────────────────────
PROCESSED_PATH = "data/processed/rides_clean.csv"
MODEL_DIR      = "models"
REPORT_DIR     = "reports"
FIGURES_DIR    = "reports/figures"

CITIES = ["Delhi NCR", "Mumbai", "Bengaluru", "Chennai", "Hyderabad"]

# ── Feature columns used for modelling ───────────────────────────────────────
FEATURE_COLS = [
    # Time features
    "hour",
    "day_of_week",
    "month",
    "quarter",
    "is_peak_hour",
    "is_weekend",
    # Weather features
    "is_raining",
    "is_heavy_rain",
    "is_fog",
    # Event features
    "is_festival",
    "is_ipl_day",
    # Encoded
    "season_code",
    # Lag features (added during prep)
    "rides_lag_1d",     # rides same hour, yesterday
    "rides_lag_7d",     # rides same hour, 7 days ago
    "rides_roll_3d",    # rolling 3-day mean same hour
    "rides_roll_7d",    # rolling 7-day mean same hour
    # Hour sine/cosine encoding (captures cyclical nature)
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]

TARGET = "rides"


# ── Data preparation ──────────────────────────────────────────────────────────

def build_hourly_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate ride-level data to city × date × hour level."""
    df["date"] = pd.to_datetime(df["date"])

    hourly = df.groupby(["city", "date", "hour"]).agg(
        rides          = ("ride_id",         "count"),
        is_peak_hour   = ("is_peak_hour",    "first"),
        is_weekend     = ("is_weekend",      "first"),
        is_raining     = ("is_raining",      "max"),
        is_heavy_rain  = ("is_heavy_rain",   "max"),
        is_fog         = ("is_fog",          "max"),
        is_festival    = ("is_festival",     "first"),
        is_ipl_day     = ("is_ipl_day",      "first"),
        season_code    = ("season_code",     "first"),
        city_code      = ("city_code",       "first"),
        month          = ("month",           "first"),
        day_of_week    = ("day_of_week",     "first"),
        quarter        = ("quarter",         "first"),
    ).reset_index()

    # Cast booleans to int
    for col in ["is_festival", "is_ipl_day"]:
        hourly[col] = hourly[col].astype(int)

    return hourly.sort_values(["city", "date", "hour"]).reset_index(drop=True)


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode hour and day_of_week as sine/cosine (captures wrap-around)."""
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * df["day_of_week"] / 7)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag features per city × hour:
      - rides_lag_1d  : rides same hour, 1 day ago
      - rides_lag_7d  : rides same hour, 7 days ago
      - rides_roll_3d : 3-day rolling mean same hour
      - rides_roll_7d : 7-day rolling mean same hour

    These are the most powerful features for time-series demand prediction.
    Important: sorted by city → date → hour before computing.
    """
    df = df.sort_values(["city", "hour", "date"]).reset_index(drop=True)

    for col, shift, window in [
        ("rides_lag_1d",  1,  None),
        ("rides_lag_7d",  7,  None),
        ("rides_roll_3d", None, 3),
        ("rides_roll_7d", None, 7),
    ]:
        if shift:
            df[col] = df.groupby(["city", "hour"])["rides"].shift(shift)
        else:
            df[col] = (
                df.groupby(["city", "hour"])["rides"]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            )

    # Drop rows where lags are NaN (first 7 days per city-hour)
    df = df.dropna(subset=["rides_lag_1d", "rides_lag_7d"]).reset_index(drop=True)
    df = df.sort_values(["city", "date", "hour"]).reset_index(drop=True)
    return df


def prepare_city_data(hourly: pd.DataFrame, city: str) -> pd.DataFrame:
    """Filter to one city and ensure all feature columns exist."""
    city_df = hourly[hourly["city"] == city].copy()
    city_df = add_cyclical_features(city_df)
    city_df = add_lag_features(city_df)
    return city_df


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    # MAPE — avoid division by zero
    mask = y_true > 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return {"MAE": round(mae, 3), "RMSE": round(rmse, 3),
            "MAPE": round(mape, 2), "R2": round(r2, 4)}


# ── Time-series cross-validation ──────────────────────────────────────────────

def time_series_cv(X: pd.DataFrame, y: pd.Series, model,
                   n_splits: int = 5) -> list[dict]:
    """
    Walk-forward cross-validation respecting temporal order.
    Never trains on future data — critical for time-series.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        y_pred = np.maximum(model.predict(X_val), 0)

        metrics = compute_metrics(y_val.values, y_pred)
        metrics["fold"] = fold + 1
        metrics["train_size"] = len(train_idx)
        metrics["val_size"]   = len(val_idx)
        fold_metrics.append(metrics)

    return fold_metrics


# ── Model training ────────────────────────────────────────────────────────────

def train_city_model(city_df: pd.DataFrame, city: str) -> dict:
    """
    Train baseline + GBM model for one city.
    Returns results dict with metrics, feature importance, predictions.
    """
    print(f"\n  {'─'*50}")
    print(f"  City: {city}")
    print(f"  Rows: {len(city_df):,}  |  Features: {len(FEATURE_COLS)}")

    X = city_df[FEATURE_COLS].copy()
    y = city_df[TARGET].copy()

    # 80/20 temporal split — last 20% is test set
    split_idx = int(len(city_df) * 0.80)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"  Train: {len(X_train):,} rows  |  Test: {len(X_test):,} rows")
    print(f"  Train period: {city_df['date'].iloc[0].date()} → "
          f"{city_df['date'].iloc[split_idx-1].date()}")
    print(f"  Test period:  {city_df['date'].iloc[split_idx].date()} → "
          f"{city_df['date'].iloc[-1].date()}")

    # ── Baseline: Ridge Regression ────────────────────────────────────────
    baseline = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  Ridge(alpha=1.0))
    ])
    baseline.fit(X_train, y_train)
    baseline_pred = np.maximum(baseline.predict(X_test), 0)
    baseline_metrics = compute_metrics(y_test.values, baseline_pred)
    print(f"\n  Baseline (Ridge):  MAE={baseline_metrics['MAE']:.2f}  "
          f"RMSE={baseline_metrics['RMSE']:.2f}  "
          f"MAPE={baseline_metrics['MAPE']:.1f}%  "
          f"R²={baseline_metrics['R2']:.3f}")

    # ── Try XGBoost first, fall back to GradientBoostingRegressor ─────────
    try:
        import xgboost as xgb
        gbm = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        model_name = "XGBoost"
    except ImportError:
        gbm = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=10,
            random_state=42,
        )
        model_name = "GradientBoosting"

    # Time-series CV on training data
    print(f"\n  Running 5-fold time-series CV ({model_name})...")
    cv_results = time_series_cv(X_train, y_train, gbm, n_splits=5)
    cv_mae  = np.mean([r["MAE"]  for r in cv_results])
    cv_rmse = np.mean([r["RMSE"] for r in cv_results])
    cv_r2   = np.mean([r["R2"]   for r in cv_results])
    print(f"  CV results (mean over 5 folds):")
    print(f"    MAE={cv_mae:.2f}  RMSE={cv_rmse:.2f}  R²={cv_r2:.3f}")

    # Final fit on full training set, evaluate on held-out test
    gbm.fit(X_train, y_train)
    test_pred = np.maximum(gbm.predict(X_test), 0)
    test_metrics = compute_metrics(y_test.values, test_pred)
    print(f"\n  Test set ({model_name}): MAE={test_metrics['MAE']:.2f}  "
          f"RMSE={test_metrics['RMSE']:.2f}  "
          f"MAPE={test_metrics['MAPE']:.1f}%  "
          f"R²={test_metrics['R2']:.3f}")

    # Improvement over baseline
    mae_improvement = (baseline_metrics["MAE"] - test_metrics["MAE"]) / baseline_metrics["MAE"] * 100
    print(f"  MAE improvement over baseline: {mae_improvement:.1f}%")

    # Feature importance
    if hasattr(gbm, 'feature_importances_'):
        importances = gbm.feature_importances_
    else:
        importances = np.abs(baseline.named_steps['model'].coef_)

    feat_imp = pd.Series(importances, index=FEATURE_COLS).sort_values(ascending=False)

    return {
        "city":             city,
        "model_name":       model_name,
        "model":            gbm,
        "baseline":         baseline,
        "X_train":          X_train,
        "X_test":           X_test,
        "y_train":          y_train,
        "y_test":           y_test,
        "test_pred":        test_pred,
        "baseline_pred":    baseline_pred,
        "test_metrics":     test_metrics,
        "baseline_metrics": baseline_metrics,
        "cv_results":       cv_results,
        "feature_importance": feat_imp,
        "city_df":          city_df,
        "split_idx":        split_idx,
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

CITY_COLORS = {
    "Delhi NCR": "#185FA5",
    "Mumbai":    "#1D9E75",
    "Bengaluru": "#BA7517",
    "Chennai":   "#D85A30",
    "Hyderabad": "#534AB7",
}


def plot_predictions(result: dict) -> None:
    """Actual vs predicted + residuals for one city."""
    city     = result["city"]
    y_test   = result["y_test"].values
    y_pred   = result["test_pred"]
    color    = CITY_COLORS.get(city, "#185FA5")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. Actual vs predicted scatter
    axes[0].scatter(y_test, y_pred, alpha=0.3, color=color, s=15, zorder=3)
    lim = max(y_test.max(), y_pred.max()) + 2
    axes[0].plot([0, lim], [0, lim], 'r--', lw=1.5, alpha=0.7, label='Perfect prediction')
    axes[0].set_xlabel('Actual rides')
    axes[0].set_ylabel('Predicted rides')
    axes[0].set_title(f'{city}\nActual vs predicted')
    axes[0].legend(fontsize=9)
    m = result["test_metrics"]
    axes[0].text(0.05, 0.95, f"MAE={m['MAE']:.1f}\nR²={m['R2']:.3f}",
                 transform=axes[0].transAxes, va='top', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 2. Residuals distribution
    residuals = y_pred - y_test
    axes[1].hist(residuals, bins=40, color=color, alpha=0.75, zorder=3)
    axes[1].axvline(0, color='red', linestyle='--', lw=1.5)
    axes[1].axvline(residuals.mean(), color='orange', linestyle='--', lw=1.2,
                    label=f'Mean={residuals.mean():.2f}')
    axes[1].set_xlabel('Residual (predicted - actual)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Residuals distribution')
    axes[1].legend(fontsize=9)

    # 3. Forecast vs actual — last 7 days
    city_df   = result["city_df"]
    split_idx = result["split_idx"]
    test_df   = city_df.iloc[split_idx:].copy()
    test_df["predicted"] = y_pred

    # Pick one representative hour (hour 18 = evening peak)
    sample = test_df[test_df["hour"] == 18].tail(30)
    if len(sample) > 5:
        axes[2].plot(range(len(sample)), sample["rides"].values,
                     color='gray', lw=1.5, label='Actual', zorder=3)
        axes[2].plot(range(len(sample)), sample["predicted"].values,
                     color=color, lw=2, linestyle='--', label='Predicted', zorder=4)
        axes[2].set_xlabel('Days (test period)')
        axes[2].set_ylabel('Rides at 6PM')
        axes[2].set_title('6PM demand forecast — test period')
        axes[2].legend(fontsize=9)
    else:
        axes[2].set_visible(False)

    plt.tight_layout()
    safe_city = city.replace(' ', '_').replace('/', '_')
    path = f"{FIGURES_DIR}/ml_{safe_city}_predictions.png"
    plt.savefig(path, bbox_inches='tight', dpi=130)
    plt.close()
    print(f"  ✓ Plot saved: {path}")


def plot_feature_importance(result: dict) -> None:
    """Top-15 feature importance bar chart."""
    city     = result["city"]
    feat_imp = result["feature_importance"].head(15)
    color    = CITY_COLORS.get(city, "#185FA5")

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(feat_imp.index[::-1], feat_imp.values[::-1], color=color, zorder=3)
    ax.set_xlabel('Feature importance')
    ax.set_title(f'{city} — Top 15 feature importances ({result["model_name"]})')

    for bar, val in zip(bars, feat_imp.values[::-1]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    safe_city = city.replace(' ', '_').replace('/', '_')
    path = f"{FIGURES_DIR}/ml_{safe_city}_feature_importance.png"
    plt.savefig(path, bbox_inches='tight', dpi=130)
    plt.close()
    print(f"  ✓ Plot saved: {path}")


def plot_cv_results(all_results: list[dict]) -> None:
    """CV fold metrics across all cities."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics_to_plot = [("MAE", "MAE (rides/hour)"), ("RMSE", "RMSE"), ("R2", "R²")]

    for ax, (metric, label) in zip(axes, metrics_to_plot):
        for result in all_results:
            city = result["city"]
            vals = [f[metric] for f in result["cv_results"]]
            folds = [f["fold"] for f in result["cv_results"]]
            ax.plot(folds, vals, marker='o', label=city,
                    color=CITY_COLORS.get(city, "#888"), linewidth=2, markersize=5)
        ax.set_xlabel('CV fold')
        ax.set_ylabel(label)
        ax.set_title(f'{label} across CV folds')
        ax.set_xticks([1, 2, 3, 4, 5])
        if metric == "R2":
            ax.set_ylim(0, 1.05)
    axes[0].legend(fontsize=9)

    plt.suptitle('Time-series cross-validation results — all cities',
                 fontsize=12, y=1.01, color='#888')
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/ml_cv_results.png", bbox_inches='tight', dpi=130)
    plt.close()
    print(f"  ✓ CV results plot saved")


def plot_all_cities_summary(all_results: list[dict]) -> None:
    """Side-by-side model performance comparison across all cities."""
    cities  = [r["city"] for r in all_results]
    mae     = [r["test_metrics"]["MAE"]  for r in all_results]
    rmse    = [r["test_metrics"]["RMSE"] for r in all_results]
    r2      = [r["test_metrics"]["R2"]   for r in all_results]
    mape    = [r["test_metrics"]["MAPE"] for r in all_results]
    b_mae   = [r["baseline_metrics"]["MAE"] for r in all_results]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    colors = [CITY_COLORS.get(c, "#888") for c in cities]
    short  = [c.replace(' NCR', '').replace('aluru', '') for c in cities]

    for ax, vals, b_vals, title, unit in [
        (axes[0], mae,  b_mae, 'MAE (rides/hour)',       ''),
        (axes[1], rmse, None,  'RMSE',                   ''),
        (axes[2], r2,   None,  'R² score',               ''),
        (axes[3], mape, None,  'MAPE (%)',                '%'),
    ]:
        bars = ax.bar(short, vals, color=colors, zorder=3, width=0.55)
        if b_vals:
            ax.bar(short, b_vals, color='#D3D1C7', zorder=2,
                   width=0.55, alpha=0.5, label='Baseline')
            ax.legend(fontsize=8)
        ax.set_title(title)
        ax.set_ylabel(unit if unit else title.split('(')[-1].replace(')', ''))
        ax.tick_params(axis='x', rotation=20)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(vals) * 0.02,
                    f'{val:.2f}', ha='center', fontsize=9, fontweight='bold')
        if title == 'R² score':
            ax.set_ylim(0, 1.1)
            ax.axhline(1.0, color='red', lw=1, linestyle='--', alpha=0.4)

    plt.suptitle('Demand forecasting model performance — all 5 cities',
                 fontsize=12, y=1.01, color='#888')
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/ml_all_cities_performance.png", bbox_inches='tight', dpi=130)
    plt.close()
    print(f"  ✓ All-cities summary plot saved")


# ── Save model ────────────────────────────────────────────────────────────────

def save_model(result: dict) -> None:
    """Save trained model, scaler, and metadata."""
    city      = result["city"]
    safe_city = city.replace(' ', '_').replace('/', '_')
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save model
    model_path = f"{MODEL_DIR}/demand_{safe_city}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(result["model"], f)

    # Save metadata
    meta = {
        "city":           city,
        "model_name":     result["model_name"],
        "trained_at":     datetime.now().isoformat(),
        "feature_cols":   FEATURE_COLS,
        "test_metrics":   result["test_metrics"],
        "baseline_metrics": result["baseline_metrics"],
        "cv_mean_mae":    round(np.mean([r["MAE"] for r in result["cv_results"]]), 3),
        "cv_mean_r2":     round(np.mean([r["R2"]  for r in result["cv_results"]]), 4),
        "top_features":   result["feature_importance"].head(5).to_dict(),
    }
    meta_path = f"{MODEL_DIR}/demand_{safe_city}_meta.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"  ✓ Model saved : {model_path}")
    print(f"  ✓ Meta  saved : {meta_path}")


# ── Summary report ────────────────────────────────────────────────────────────

def print_summary(all_results: list[dict]) -> None:
    print(f"\n{'='*60}")
    print("  MODEL PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    print(f"\n  {'City':<18} {'Model':<20} {'MAE':>6} {'RMSE':>7} "
          f"{'MAPE':>7} {'R²':>7}  {'vs Baseline':>12}")
    print(f"  {'-'*18} {'-'*20} {'-'*6} {'-'*7} {'-'*7} {'-'*7}  {'-'*12}")

    for r in all_results:
        m  = r["test_metrics"]
        b  = r["baseline_metrics"]
        improvement = (b["MAE"] - m["MAE"]) / b["MAE"] * 100
        print(f"  {r['city']:<18} {r['model_name']:<20} "
              f"{m['MAE']:>6.2f} {m['RMSE']:>7.2f} "
              f"{m['MAPE']:>6.1f}% {m['R2']:>7.3f}  "
              f"{improvement:>+10.1f}%")

    all_r2   = [r["test_metrics"]["R2"]   for r in all_results]
    all_mape = [r["test_metrics"]["MAPE"] for r in all_results]
    print(f"\n  Average R²   : {np.mean(all_r2):.3f}")
    print(f"  Average MAPE : {np.mean(all_mape):.1f}%")
    print(f"\n  Models saved to: {MODEL_DIR}/")
    print(f"  Charts saved to: {FIGURES_DIR}/")
    print(f"\n  Next step → python src/train_demand_model.py done")
    print(f"  Then → Phase 5: wait time prediction + surge detection\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def run(cities_to_train: list[str]) -> None:
    print(f"\n{'='*60}")
    print("  Smart City Mobility — Demand Forecasting Model")
    print(f"  Cities  : {', '.join(cities_to_train)}")
    print(f"  Model   : XGBoost (falls back to GradientBoosting)")
    print(f"  Eval    : Time-series CV (5 folds) + held-out test set")
    print(f"{'='*60}")

    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR,   exist_ok=True)

    # Load and aggregate
    print(f"\nLoading {PROCESSED_PATH}...")
    df = pd.read_csv(PROCESSED_PATH, parse_dates=["timestamp", "date"])
    hourly = build_hourly_dataset(df)
    print(f"Hourly dataset: {len(hourly):,} rows (city × date × hour)")

    all_results = []

    for city in cities_to_train:
        city_df = prepare_city_data(hourly, city)
        result  = train_city_model(city_df, city)
        plot_predictions(result)
        plot_feature_importance(result)
        save_model(result)
        all_results.append(result)

    # Cross-city plots
    print(f"\n{'─'*50}")
    print("  Generating summary plots...")
    plot_cv_results(all_results)
    plot_all_cities_summary(all_results)

    print_summary(all_results)


def main():
    parser = argparse.ArgumentParser(description="Train demand forecasting models")
    parser.add_argument("--city", type=str, default=None,
                        help="Train for one city only")
    parser.add_argument("--all-cities", action="store_true", default=True,
                        help="Train for all 5 cities (default)")
    args = parser.parse_args()

    cities = [args.city] if args.city else CITIES
    run(cities)


if __name__ == "__main__":
    main()
