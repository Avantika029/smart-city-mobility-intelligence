"""
train_wait_surge_model.py
-------------------------
Phase 5: Wait time prediction and surge zone detection.

Models built:
  1. Wait time regressor  — predict wait_time_min per ride
  2. Surge classifier     — flag zones likely to surge (surge >= 2.0×)
  3. Cancellation model   — predict ride cancellation probability

Each model:
  - Trained with time-ordered train/test split (no leakage)
  - Evaluated with appropriate metrics
  - Saved as .pkl with metadata JSON

Also writes src/inference.py — a clean prediction API used by the dashboard.

Usage:
    python src/train_wait_surge_model.py
"""

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
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    classification_report, roc_auc_score, confusion_matrix,
    precision_score, recall_score, f1_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
PROCESSED_PATH = "data/processed/rides_clean.csv"
MODEL_DIR      = "models"
FIGURES_DIR    = "reports/figures"

# ── Feature sets ──────────────────────────────────────────────────────────────
# Wait time features — include surge as it's known at ride-request time
WAIT_FEATURES = [
    "hour", "day_of_week", "month", "quarter",
    "is_peak_hour", "is_weekend", "is_night",
    "is_raining", "is_heavy_rain", "is_fog",
    "is_festival", "is_ipl_day",
    "surge_multiplier",
    "demand_pressure",
    "hourly_ride_volume",
    "city_code", "vehicle_code", "weather_code", "season_code",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]

# Surge features — predict surge WITHOUT using surge itself
SURGE_FEATURES = [
    "hour", "day_of_week", "month", "quarter",
    "is_peak_hour", "is_weekend", "is_night",
    "is_raining", "is_heavy_rain", "is_fog",
    "is_festival", "is_ipl_day",
    "demand_pressure",
    "hourly_ride_volume",
    "city_code", "weather_code", "season_code",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]

# Cancellation features
CANCEL_FEATURES = [
    "hour", "day_of_week", "month",
    "is_peak_hour", "is_weekend", "is_night",
    "is_raining", "is_heavy_rain", "is_fog",
    "is_festival", "is_ipl_day",
    "surge_multiplier", "demand_pressure",
    "city_code", "vehicle_code", "weather_code", "season_code",
]

CITY_COLORS = {
    "Delhi NCR": "#185FA5", "Mumbai": "#1D9E75",
    "Bengaluru": "#BA7517", "Chennai": "#D85A30", "Hyderabad": "#534AB7",
}


# ── Feature prep ──────────────────────────────────────────────────────────────

def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * df["day_of_week"] / 7)
    return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = add_cyclical_features(df)
    for col in ["is_festival", "is_ipl_day"]:
        df[col] = df[col].astype(int)
    # Surge binary target
    df["is_high_surge"] = (df["surge_multiplier"] >= 2.0).astype(int)
    df["is_cancelled"]  = (~df["is_completed"]).astype(int)
    return df


def temporal_split(df: pd.DataFrame, ratio: float = 0.80):
    split = int(len(df) * ratio)
    return df.iloc[:split], df.iloc[split:]


# ── Metrics helpers ───────────────────────────────────────────────────────────

def reg_metrics(y_true, y_pred) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mask = y_true > 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return {"MAE": round(mae,3), "RMSE": round(rmse,3),
            "MAPE": round(mape,2), "R2": round(r2,4)}


def cls_metrics(y_true, y_pred, y_prob) -> dict:
    return {
        "AUC":       round(roc_auc_score(y_true, y_prob), 4),
        "Precision": round(precision_score(y_true, y_pred), 4),
        "Recall":    round(recall_score(y_true, y_pred), 4),
        "F1":        round(f1_score(y_true, y_pred), 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 1 — Wait Time Regressor
# ══════════════════════════════════════════════════════════════════════════════

def train_wait_model(df: pd.DataFrame) -> dict:
    print(f"\n{'='*60}")
    print("  MODEL 1 — Wait Time Regressor")
    print(f"{'='*60}")

    train_df, test_df = temporal_split(df)
    X_train = train_df[WAIT_FEATURES]
    y_train = train_df["wait_time_min"]
    X_test  = test_df[WAIT_FEATURES]
    y_test  = test_df["wait_time_min"]

    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")
    print(f"  Target mean: {y_train.mean():.2f} min  |  std: {y_train.std():.2f}")

    # Baseline
    baseline = Pipeline([("s", StandardScaler()), ("m", Ridge())])
    baseline.fit(X_train, y_train)
    b_pred = np.clip(baseline.predict(X_test), 0.5, 120)
    b_met  = reg_metrics(y_test.values, b_pred)
    print(f"\n  Baseline (Ridge):  MAE={b_met['MAE']:.2f}  R²={b_met['R2']:.3f}")

    # GBM
    try:
        import xgboost as xgb
        gbm = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8,
                                random_state=42, n_jobs=-1, verbosity=0)
        name = "XGBoost"
    except ImportError:
        gbm = GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                        learning_rate=0.05, subsample=0.8,
                                        min_samples_leaf=20, random_state=42)
        name = "GradientBoosting"

    # Time-series CV
    tscv = TimeSeriesSplit(n_splits=5)
    cv_maes = []
    for tr_idx, val_idx in tscv.split(X_train):
        gbm.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        p = np.clip(gbm.predict(X_train.iloc[val_idx]), 0.5, 120)
        cv_maes.append(mean_absolute_error(y_train.iloc[val_idx], p))
    print(f"  CV MAE (5-fold): {np.mean(cv_maes):.2f} ± {np.std(cv_maes):.2f} min")

    # Final fit
    gbm.fit(X_train, y_train)
    pred = np.clip(gbm.predict(X_test), 0.5, 120)
    met  = reg_metrics(y_test.values, pred)
    impr = (b_met["MAE"] - met["MAE"]) / b_met["MAE"] * 100
    print(f"  Test  ({name}):  MAE={met['MAE']:.2f}  RMSE={met['RMSE']:.2f}  "
          f"MAPE={met['MAPE']:.1f}%  R²={met['R2']:.3f}  (+{impr:.1f}% vs baseline)")

    feat_imp = pd.Series(gbm.feature_importances_, index=WAIT_FEATURES).sort_values(ascending=False)

    return {"model": gbm, "model_name": name, "baseline": baseline,
            "metrics": met, "baseline_metrics": b_met,
            "y_test": y_test, "pred": pred,
            "feature_importance": feat_imp, "features": WAIT_FEATURES}


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 2 — Surge Zone Classifier
# ══════════════════════════════════════════════════════════════════════════════

def train_surge_model(df: pd.DataFrame) -> dict:
    print(f"\n{'='*60}")
    print("  MODEL 2 — Surge Zone Classifier  (surge >= 2.0×)")
    print(f"{'='*60}")

    train_df, test_df = temporal_split(df)
    X_train = train_df[SURGE_FEATURES]
    y_train = train_df["is_high_surge"]
    X_test  = test_df[SURGE_FEATURES]
    y_test  = test_df["is_high_surge"]

    pos_rate = y_train.mean()
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")
    print(f"  Class balance — high surge: {pos_rate*100:.1f}%  |  normal: {(1-pos_rate)*100:.1f}%")

    # Baseline
    baseline = Pipeline([("s", StandardScaler()),
                         ("m", LogisticRegression(max_iter=500, random_state=42))])
    baseline.fit(X_train, y_train)
    b_pred = baseline.predict(X_test)
    b_prob = baseline.predict_proba(X_test)[:, 1]
    b_met  = cls_metrics(y_test.values, b_pred, b_prob)
    print(f"\n  Baseline (Logistic):  AUC={b_met['AUC']:.3f}  "
          f"F1={b_met['F1']:.3f}  Precision={b_met['Precision']:.3f}  Recall={b_met['Recall']:.3f}")

    # GBM classifier
    try:
        import xgboost as xgb
        gbm = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8,
                                 scale_pos_weight=(1-pos_rate)/pos_rate,
                                 random_state=42, n_jobs=-1, verbosity=0,
                                 eval_metric='logloss')
        name = "XGBoost"
    except ImportError:
        gbm = GradientBoostingClassifier(n_estimators=200, max_depth=5,
                                         learning_rate=0.05, subsample=0.8,
                                         min_samples_leaf=20, random_state=42)
        name = "GradientBoosting"

    # CV
    tscv   = TimeSeriesSplit(n_splits=5)
    cv_auc = []
    for tr_idx, val_idx in tscv.split(X_train):
        gbm.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        prob = gbm.predict_proba(X_train.iloc[val_idx])[:, 1]
        cv_auc.append(roc_auc_score(y_train.iloc[val_idx], prob))
    print(f"  CV AUC (5-fold): {np.mean(cv_auc):.3f} ± {np.std(cv_auc):.3f}")

    # Final fit
    gbm.fit(X_train, y_train)
    pred = gbm.predict(X_test)
    prob = gbm.predict_proba(X_test)[:, 1]
    met  = cls_metrics(y_test.values, pred, prob)
    print(f"  Test  ({name}):  AUC={met['AUC']:.3f}  F1={met['F1']:.3f}  "
          f"Precision={met['Precision']:.3f}  Recall={met['Recall']:.3f}")

    cm = confusion_matrix(y_test, pred)
    print(f"\n  Confusion matrix:")
    print(f"    TN={cm[0,0]:>6,}  FP={cm[0,1]:>6,}")
    print(f"    FN={cm[1,0]:>6,}  TP={cm[1,1]:>6,}")

    feat_imp = pd.Series(gbm.feature_importances_, index=SURGE_FEATURES).sort_values(ascending=False)

    return {"model": gbm, "model_name": name, "baseline": baseline,
            "metrics": met, "baseline_metrics": b_met,
            "y_test": y_test, "pred": pred, "prob": prob,
            "confusion_matrix": cm,
            "feature_importance": feat_imp, "features": SURGE_FEATURES}


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 3 — Cancellation Predictor
# ══════════════════════════════════════════════════════════════════════════════

def train_cancel_model(df: pd.DataFrame) -> dict:
    print(f"\n{'='*60}")
    print("  MODEL 3 — Cancellation Predictor")
    print(f"{'='*60}")

    train_df, test_df = temporal_split(df)
    X_train = train_df[CANCEL_FEATURES]
    y_train = train_df["is_cancelled"]
    X_test  = test_df[CANCEL_FEATURES]
    y_test  = test_df["is_cancelled"]

    pos_rate = y_train.mean()
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")
    print(f"  Class balance — cancelled: {pos_rate*100:.1f}%  |  completed: {(1-pos_rate)*100:.1f}%")

    try:
        import xgboost as xgb
        gbm = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                                 subsample=0.8, scale_pos_weight=(1-pos_rate)/pos_rate,
                                 random_state=42, n_jobs=-1, verbosity=0,
                                 eval_metric='logloss')
        name = "XGBoost"
    except ImportError:
        gbm = GradientBoostingClassifier(n_estimators=150, max_depth=4,
                                         learning_rate=0.05, subsample=0.8,
                                         min_samples_leaf=30, random_state=42)
        name = "GradientBoosting"

    gbm.fit(X_train, y_train)
    pred = gbm.predict(X_test)
    prob = gbm.predict_proba(X_test)[:, 1]
    met  = cls_metrics(y_test.values, pred, prob)
    print(f"  Test ({name}):  AUC={met['AUC']:.3f}  F1={met['F1']:.3f}  "
          f"Precision={met['Precision']:.3f}  Recall={met['Recall']:.3f}")

    feat_imp = pd.Series(gbm.feature_importances_, index=CANCEL_FEATURES).sort_values(ascending=False)

    return {"model": gbm, "model_name": name,
            "metrics": met, "y_test": y_test, "pred": pred, "prob": prob,
            "feature_importance": feat_imp, "features": CANCEL_FEATURES}


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_wait_analysis(wait_res: dict) -> None:
    """4-panel wait time model analysis."""
    y_test = wait_res["y_test"].values
    pred   = wait_res["pred"]
    residuals = pred - y_test
    feat_imp  = wait_res["feature_importance"].head(12)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # 1. Actual vs predicted
    axes[0,0].scatter(y_test, pred, alpha=0.2, color="#185FA5", s=8, zorder=3)
    lim = max(y_test.max(), pred.max()) + 2
    axes[0,0].plot([0,lim],[0,lim],'r--',lw=1.5,alpha=0.6,label='Perfect')
    axes[0,0].set_xlabel('Actual wait (min)')
    axes[0,0].set_ylabel('Predicted wait (min)')
    axes[0,0].set_title('Actual vs predicted wait time')
    m = wait_res["metrics"]
    axes[0,0].text(0.05, 0.95,
                   f"MAE  = {m['MAE']:.2f} min\nRMSE = {m['RMSE']:.2f}\nR²   = {m['R2']:.3f}",
                   transform=axes[0,0].transAxes, va='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='#E6F1FB', alpha=0.9))
    axes[0,0].legend(fontsize=9)

    # 2. Residuals histogram
    axes[0,1].hist(residuals, bins=50, color="#185FA5", alpha=0.75, zorder=3)
    axes[0,1].axvline(0, color='red', lw=1.5, linestyle='--')
    axes[0,1].axvline(residuals.mean(), color='orange', lw=1.2, linestyle='--',
                      label=f'Mean={residuals.mean():.2f}')
    axes[0,1].set_xlabel('Residual (predicted − actual)')
    axes[0,1].set_ylabel('Count')
    axes[0,1].set_title('Residuals distribution')
    axes[0,1].legend(fontsize=9)

    # 3. Feature importance
    colors = ["#185FA5" if i < 3 else "#85B7EB" for i in range(len(feat_imp))]
    axes[1,0].barh(feat_imp.index[::-1], feat_imp.values[::-1], color=colors[::-1], zorder=3)
    axes[1,0].set_xlabel('Feature importance')
    axes[1,0].set_title('Top 12 features — wait time model')
    for i, (idx, val) in enumerate(zip(feat_imp.index[::-1], feat_imp.values[::-1])):
        axes[1,0].text(val + 0.001, i, f'{val:.3f}', va='center', fontsize=8)

    # 4. Wait time by hour — actual vs predicted mean
    test_df = wait_res["y_test"].reset_index(drop=True).to_frame()
    test_df["pred"] = pred
    test_df["hour"] = test_df.index % 24
    hourly_actual = test_df.groupby("hour")["wait_time_min"].mean()
    hourly_pred   = test_df.groupby("hour")["pred"].mean()
    axes[1,1].plot(hourly_actual.index, hourly_actual.values,
                   color='gray', lw=2, label='Actual mean', marker='o', markersize=4)
    axes[1,1].plot(hourly_pred.index, hourly_pred.values,
                   color='#185FA5', lw=2, linestyle='--', label='Predicted mean',
                   marker='s', markersize=4)
    axes[1,1].set_xlabel('Hour of day')
    axes[1,1].set_ylabel('Mean wait time (min)')
    axes[1,1].set_title('Predicted vs actual mean wait — by hour')
    axes[1,1].legend(fontsize=9)
    axes[1,1].set_xticks(range(0, 24, 2))

    plt.suptitle('Model 1 — Wait time prediction analysis',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/ml_wait_time_analysis.png", bbox_inches='tight', dpi=130)
    plt.close()
    print(f"  ✓ Plot saved: ml_wait_time_analysis.png")


def plot_surge_analysis(surge_res: dict) -> None:
    """Surge classifier — ROC-like, confusion matrix, feature importance."""
    y_test   = surge_res["y_test"].values
    pred     = surge_res["pred"]
    prob     = surge_res["prob"]
    cm       = surge_res["confusion_matrix"]
    feat_imp = surge_res["feature_importance"].head(12)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. Surge probability distribution
    axes[0].hist(prob[y_test==0], bins=40, alpha=0.6, color="#1D9E75",
                 label='Normal surge', density=True, zorder=3)
    axes[0].hist(prob[y_test==1], bins=40, alpha=0.6, color="#E24B4A",
                 label='High surge', density=True, zorder=3)
    axes[0].axvline(0.5, color='black', lw=1.5, linestyle='--', label='Threshold=0.5')
    axes[0].set_xlabel('Predicted surge probability')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Surge probability distribution')
    axes[0].legend(fontsize=9)
    m = surge_res["metrics"]
    axes[0].text(0.02, 0.97, f"AUC={m['AUC']:.3f}\nF1={m['F1']:.3f}",
                 transform=axes[0].transAxes, va='top', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 2. Confusion matrix heatmap
    cm_pct = cm.astype(float) / cm.sum() * 100
    im = axes[1].imshow(cm_pct, cmap='Blues', aspect='auto', vmin=0, vmax=cm_pct.max()*1.1)
    axes[1].set_xticks([0,1]); axes[1].set_yticks([0,1])
    axes[1].set_xticklabels(['Predicted\nNormal','Predicted\nHigh surge'])
    axes[1].set_yticklabels(['Actual\nNormal','Actual\nHigh surge'])
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, f'{cm[i,j]:,}\n({cm_pct[i,j]:.1f}%)',
                         ha='center', va='center', fontsize=11,
                         color='white' if cm_pct[i,j] > 30 else 'black', fontweight='bold')
    axes[1].set_title('Confusion matrix')

    # 3. Feature importance
    colors = ["#E24B4A" if i < 3 else "#F09595" for i in range(len(feat_imp))]
    axes[2].barh(feat_imp.index[::-1], feat_imp.values[::-1], color=colors[::-1], zorder=3)
    axes[2].set_xlabel('Feature importance')
    axes[2].set_title('Top 12 features — surge classifier')
    for i, (idx, val) in enumerate(zip(feat_imp.index[::-1], feat_imp.values[::-1])):
        axes[2].text(val + 0.001, i, f'{val:.3f}', va='center', fontsize=8)

    plt.suptitle('Model 2 — Surge zone classifier analysis',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/ml_surge_classifier_analysis.png", bbox_inches='tight', dpi=130)
    plt.close()
    print(f"  ✓ Plot saved: ml_surge_classifier_analysis.png")


def plot_rebalancing_recommendations(df: pd.DataFrame, surge_res: dict) -> None:
    """Fleet rebalancing heatmap — which city+hour needs more drivers."""
    _, test_df = temporal_split(df)
    test_df = test_df.copy()
    test_df["surge_prob"] = surge_res["prob"]

    # Average surge probability by city + hour
    rebal = test_df.groupby(["city_code", "hour"])["surge_prob"].mean().unstack(fill_value=0)
    city_names = {0:"Bengaluru", 1:"Chennai", 2:"Delhi NCR", 3:"Hyderabad", 4:"Mumbai"}
    rebal.index = [city_names.get(i, i) for i in rebal.index]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Heatmap
    im = axes[0].imshow(rebal.values, cmap='Reds', aspect='auto', vmin=0, vmax=0.6)
    axes[0].set_xticks(range(0, 24, 2))
    axes[0].set_xticklabels([f'{h}:00' for h in range(0, 24, 2)], rotation=45, ha='right', fontsize=8)
    axes[0].set_yticks(range(len(rebal.index)))
    axes[0].set_yticklabels(rebal.index)
    axes[0].set_xlabel('Hour of day')
    axes[0].set_title('Surge risk by city × hour\n(darker = higher risk)')
    plt.colorbar(im, ax=axes[0], label='Avg surge probability', shrink=0.8)

    # Top 10 rebalancing priorities
    rebal_flat = rebal.stack().reset_index()
    rebal_flat.columns = ["city", "hour", "surge_prob"]
    top10 = rebal_flat.nlargest(10, "surge_prob")
    top10["label"] = top10["city"] + " " + top10["hour"].astype(str) + ":00"
    colors = [CITY_COLORS.get(c, "#888") for c in top10["city"]]
    bars = axes[1].barh(top10["label"][::-1], top10["surge_prob"][::-1],
                        color=colors[::-1], zorder=3)
    axes[1].set_xlabel('Avg surge probability')
    axes[1].set_title('Top 10 fleet rebalancing priorities')
    axes[1].axvline(0.5, color='red', lw=1, linestyle='--', alpha=0.5, label='50% threshold')
    axes[1].legend(fontsize=8)
    for bar, val in zip(bars, top10["surge_prob"][::-1]):
        axes[1].text(val + 0.005, bar.get_y() + bar.get_height()/2,
                     f'{val:.2f}', va='center', fontsize=9)

    plt.suptitle('Fleet rebalancing recommendations — predicted surge hotspots',
                 fontsize=12, y=1.01, color='#888')
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/ml_rebalancing_recommendations.png", bbox_inches='tight', dpi=130)
    plt.close()
    print(f"  ✓ Plot saved: ml_rebalancing_recommendations.png")


# ── Save models ───────────────────────────────────────────────────────────────

def save_all_models(wait_res: dict, surge_res: dict, cancel_res: dict) -> None:
    print(f"\n{'='*60}")
    print("  Saving models...")
    print(f"{'='*60}")
    os.makedirs(MODEL_DIR, exist_ok=True)

    for name, result, features in [
        ("wait_time",    wait_res,   WAIT_FEATURES),
        ("surge",        surge_res,  SURGE_FEATURES),
        ("cancellation", cancel_res, CANCEL_FEATURES),
    ]:
        model_path = f"{MODEL_DIR}/{name}_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(result["model"], f)

        meta = {
            "model_type":  name,
            "model_name":  result["model_name"],
            "trained_at":  datetime.now().isoformat(),
            "feature_cols": features,
            "metrics":     result["metrics"],
            "top_features": result["feature_importance"].head(5).to_dict(),
        }
        meta_path = f"{MODEL_DIR}/{name}_model_meta.json"
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        print(f"  ✓ {model_path}")
        print(f"  ✓ {meta_path}")


# ── Print final summary ───────────────────────────────────────────────────────

def print_summary(wait_res: dict, surge_res: dict, cancel_res: dict) -> None:
    print(f"\n{'='*60}")
    print("  PHASE 5 SUMMARY — ALL MODELS")
    print(f"{'='*60}")

    wm = wait_res["metrics"]
    sm = surge_res["metrics"]
    cm = cancel_res["metrics"]

    print(f"\n  Model 1 — Wait time regressor ({wait_res['model_name']})")
    print(f"    MAE  : {wm['MAE']:.2f} min  (off by ~{wm['MAE']:.1f} min on average)")
    print(f"    RMSE : {wm['RMSE']:.2f} min")
    print(f"    R²   : {wm['R2']:.3f}  ({wm['R2']*100:.1f}% variance explained)")
    print(f"    MAPE : {wm['MAPE']:.1f}%")

    print(f"\n  Model 2 — Surge classifier ({surge_res['model_name']})")
    print(f"    AUC       : {sm['AUC']:.3f}")
    print(f"    Precision : {sm['Precision']:.3f}  (when we predict surge, correct {sm['Precision']*100:.0f}% of the time)")
    print(f"    Recall    : {sm['Recall']:.3f}  (catch {sm['Recall']*100:.0f}% of all actual surges)")
    print(f"    F1        : {sm['F1']:.3f}")

    print(f"\n  Model 3 — Cancellation predictor ({cancel_res['model_name']})")
    print(f"    AUC       : {cm['AUC']:.3f}")
    print(f"    Precision : {cm['Precision']:.3f}")
    print(f"    Recall    : {cm['Recall']:.3f}")
    print(f"    F1        : {cm['F1']:.3f}")

    print(f"\n  Top wait-time drivers: {', '.join(wait_res['feature_importance'].head(3).index.tolist())}")
    print(f"  Top surge drivers:     {', '.join(surge_res['feature_importance'].head(3).index.tolist())}")
    print(f"\n  Next step → Phase 6: Streamlit dashboard\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*60}")
    print("  Smart City Mobility — Phase 5 Model Training")
    print(f"{'='*60}")

    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"\nLoading {PROCESSED_PATH}...")
    df = pd.read_csv(PROCESSED_PATH, parse_dates=["timestamp"])
    df = prepare_data(df)
    print(f"Loaded: {len(df):,} rows")

    wait_res   = train_wait_model(df)
    surge_res  = train_surge_model(df)
    cancel_res = train_cancel_model(df)

    print(f"\n{'='*60}")
    print("  Generating analysis plots...")
    print(f"{'='*60}")
    plot_wait_analysis(wait_res)
    plot_surge_analysis(surge_res)
    plot_rebalancing_recommendations(df, surge_res)

    save_all_models(wait_res, surge_res, cancel_res)
    print_summary(wait_res, surge_res, cancel_res)


if __name__ == "__main__":
    main()
