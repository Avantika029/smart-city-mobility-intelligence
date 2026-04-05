# 🚗 Smart City Mobility Intelligence System

> An end-to-end data science project analyzing and predicting urban mobility patterns across 5 major Indian cities using machine learning.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?style=flat-square)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=flat-square)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## 📌 Objective

Analyze 500,000+ ride records across **Delhi NCR, Mumbai, Bengaluru, Chennai, and Hyderabad** to:
- Identify demand patterns by hour, day, season, weather, and events
- Predict hourly ride demand per city (ML forecasting)
- Predict wait times and flag surge zones proactively
- Deliver actionable fleet rebalancing recommendations

---

## 🏗️ Project Structure

```
smart_city_mobility/
├── data/
│   ├── raw/                  # Generated synthetic ride dataset (500K+ rows)
│   └── processed/            # Cleaned + feature-engineered dataset (43 columns)
├── notebooks/
│   ├── 01_phase1_exploration.ipynb   # Dataset overview
│   └── 02_eda.ipynb                  # Full EDA with 10 charts
├── src/
│   ├── generate_data.py      # Synthetic data generator
│   ├── validate_data.py      # 18 automated data quality checks
│   ├── pipeline.py           # 7-stage cleaning + feature engineering pipeline
│   ├── train_demand_model.py # Demand forecasting model (Phase 4)
│   ├── train_wait_surge_model.py  # Wait time + surge models (Phase 5)
│   └── inference.py          # Unified prediction API for dashboard
├── models/                   # Trained model artifacts (.pkl)
├── reports/
│   └── figures/              # 20+ saved charts (EDA + ML analysis)
├── app.py                    # Streamlit dashboard
├── dashboard.html            # Standalone BI dashboard (no install needed)
├── requirements.txt
└── packages.txt
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| Total records | 501,500 (after quality injection) |
| Cleaned records | 499,989 |
| Date range | Jan 2023 – Jun 2024 (18 months) |
| Cities | Delhi NCR, Mumbai, Bengaluru, Chennai, Hyderabad |
| Original columns | 14 |
| Engineered features | 29 (total: 43 columns) |

### Column Dictionary

| Column | Type | Description |
|---|---|---|
| ride_id | str | Unique ride identifier |
| timestamp | datetime | Ride request time (IST) |
| city | str | One of 5 metro cities |
| pickup_zone | str | Named urban zone within city |
| vehicle_type | str | bike_taxi / auto / economy / premium / shared |
| wait_time_min | float | Minutes from request to pickup |
| surge_multiplier | float | Fare surge factor (1.0 = no surge) |
| distance_km | float | Ride distance in km |
| fare_inr | float | Final fare in Indian Rupees |
| driver_rating | float | Driver rating (1.0–5.0) |
| is_completed | bool | Whether ride was completed |
| weather | str | clear / light_rain / heavy_rain / fog |
| is_festival | bool | Public holiday or major festival day |
| is_ipl_day | bool | IPL match day in that city |

### Engineered Features (29)
`hour`, `day_of_week`, `month`, `quarter`, `week_of_year`, `season`,
`is_peak_hour`, `is_weekend`, `is_night`, `is_late_night`,
`is_raining`, `is_heavy_rain`, `is_fog`,
`fare_per_km`, `demand_pressure`, `hourly_ride_volume`,
`is_high_surge`, `surge_band`, `wait_band`,
`is_high_demand`, `is_event_day`, `minutes_since_midnight`,
`city_code`, `vehicle_code`, `weather_code`, `season_code`,
`hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`

---

## 🔬 Key Findings (EDA)

| Finding | Metric |
|---|---|
| Busiest hour | 6 PM — 56,264 rides |
| Peak hours share | 73.4% of all rides in 7–9 AM and 5–8 PM |
| Heavy rain impact on wait | +152% vs clear weather |
| Heavy rain impact on surge | +84% vs clear weather |
| Festival day fare increase | +60% vs normal days |
| Festival day wait increase | +42% vs normal days |
| Monsoon surge (Jun–Sep) | 1.71× average |
| Surge predicts fare | r = 0.85 (strongest correlation) |
| High surge rides (≥2×) | 25.1% of all rides |
| Mumbai highest wait city | 8.2 min median |
| Worst wait zones | Chembur, Koyambedu, Ghaziabad (peripheral zones) |
| IPL post-match spike | 3× demand surge at 9–11 PM |

---

## 🤖 ML Models

### Model 1 — Demand Forecasting
Predicts **hourly ride demand per city** (rides per hour).

| City | MAE | RMSE | R² | vs Baseline |
|---|---|---|---|---|
| Delhi NCR | 2.24 | 3.16 | 0.891 | +2.9% |
| Mumbai | 2.04 | 2.88 | 0.892 | +1.2% |
| Bengaluru | 2.00 | 2.88 | 0.867 | +0.1% |
| Chennai | 1.88 | 2.68 | 0.850 | +2.3% |
| Hyderabad | 1.90 | 2.73 | 0.835 | +0.8% |
| **Average** | **2.01** | **2.87** | **0.867** | **+1.5%** |

### Model 2 — Wait Time Prediction
Predicts **wait time in minutes** per ride.

| Metric | Value |
|---|---|
| MAE | 1.41 min |
| RMSE | 1.77 min |
| MAPE | 21.2% |
| R² | 0.745 |

### Model 3 — Surge Zone Classifier
Classifies whether a zone will experience **high surge (≥ 2.0×)**.

| Metric | Value |
|---|---|
| AUC | 1.00 |
| F1 Score | 1.00 |
| Precision | 1.00 |
| Recall | 1.00 |

### Model 4 — Cancellation Predictor
Predicts **ride cancellation probability**.

| Metric | Value |
|---|---|
| AUC | 0.62 |

### Key ML decisions
- **Time-series cross-validation** (5 walk-forward folds) — no data leakage
- **Lag features** (yesterday's demand, 7-day rolling mean) — most powerful predictors
- **Cyclical encoding** (sine/cosine for hour and day-of-week)
- **Temporal train/test split** — last 20% held out as unseen test data
- **Baseline comparison** — Ridge regression used as benchmark

---

## 🖥️ Dashboard

### Streamlit App
Interactive web dashboard with 4 tabs:
- **Live predictor** — real-time wait time + surge risk predictions with gauges
- **EDA explorer** — all 10 interactive charts with city/weather filters
- **Model insights** — ML performance charts and feature importances
- **About** — project overview and key findings

```bash
streamlit run app.py
```

### Standalone BI Dashboard
Open `dashboard.html` in any browser — no install required.

Features:
- 5 live KPI cards (update on filter change)
- 8 interactive charts: hourly demand, city breakdown, monthly trend,
  weather impact, demand heatmap, vehicle analysis, surge bands, event impact
- City filter and weather filter
- Top 10 hotspot zones table
- City performance comparison table

---

## ⚙️ Setup & Usage

```bash
# 1. Clone the repo
git clone https://github.com/Avantika029/smart-city-mobility-intelligence.git
cd smart-city-mobility-intelligence

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate the dataset
python src/generate_data.py          # Creates data/raw/rides.csv (500K rows)

# 4. Validate data quality
python src/validate_data.py          # Runs 18 automated checks

# 5. Run the pipeline
python src/pipeline.py               # Cleans data, engineers 29 features

# 6. Train demand forecasting models
python src/train_demand_model.py     # Trains 5 city models (Phase 4)

# 7. Train wait time + surge models
python src/train_wait_surge_model.py # Trains 3 more models (Phase 5)

# 8. Launch the dashboard
streamlit run app.py

# Or open the standalone BI dashboard
open dashboard.html                  # Works in any browser
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Data | pandas, NumPy, DuckDB |
| Machine Learning | scikit-learn, XGBoost, LightGBM |
| Explainability | SHAP |
| Visualisation | Matplotlib, Seaborn, Plotly |
| Dashboard | Streamlit, Chart.js (HTML) |
| Tracking | MLflow |
| Dev | Git, GitHub |

---

## 📁 Phase Summary

| Phase | Description | Status |
|---|---|---|
| Phase 1 | Project setup + synthetic data generation (501K rows) | ✅ Complete |
| Phase 2 | Data pipeline — cleaning, imputation, 29 features | ✅ Complete |
| Phase 3 | EDA — 10 charts, key findings documented | ✅ Complete |
| Phase 4 | Demand forecasting model — avg R²=0.867 | ✅ Complete |
| Phase 5 | Wait time + surge + cancellation models + inference API | ✅ Complete |
| Phase 6 | Streamlit dashboard + standalone BI dashboard | ✅ Complete |

---

## 👩‍💻 Author

**Avantika** — Data Analyst / Data Science Portfolio Project

🔗 [GitHub](https://github.com/Avantika029/smart-city-mobility-intelligence)
