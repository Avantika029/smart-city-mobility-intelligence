# Smart City Mobility Intelligence System

An end-to-end data science project analyzing and predicting urban mobility patterns
across 5 major Indian cities — Delhi NCR, Mumbai, Bengaluru, Chennai, and Hyderabad.

## Objective
Analyze ride data to optimize traffic flow, reduce wait times, and improve ride
allocation using machine learning.

## Project Structure
```
smart_city_mobility/
├── data/
│   ├── raw/              # Generated synthetic ride dataset
│   ├── processed/        # Cleaned, feature-engineered data
│   └── external/         # Weather, events, holiday calendars
├── notebooks/            # EDA and modelling Jupyter notebooks
├── src/                  # Reusable Python modules
│   ├── generate_data.py  # Synthetic data generator
│   ├── pipeline.py       # Data cleaning & feature engineering
│   └── inference.py      # Model prediction utilities
├── models/               # Trained model artifacts
├── reports/
│   ├── figures/          # Saved charts (PNG + HTML)
│   └── quality/          # Data quality reports
└── tests/                # Unit tests
```

## Dataset
- 500,000+ synthetic ride records across 5 Indian cities
- 18-month date range (Jan 2023 – Jun 2024)
- Realistic patterns: rush hours, monsoon demand, festivals, IPL events

## Column Dictionary

| Column            | Type     | Description                                      |
|-------------------|----------|--------------------------------------------------|
| ride_id           | str      | Unique ride identifier                           |
| timestamp         | datetime | Ride request time (IST)                          |
| city              | str      | One of 5 metro cities                            |
| pickup_zone       | str      | Named urban zone within city                     |
| vehicle_type      | str      | bike_taxi / auto / economy / premium / shared    |
| wait_time_min     | float    | Minutes from request to pickup                   |
| surge_multiplier  | float    | Fare surge factor (1.0 = no surge)               |
| distance_km       | float    | Ride distance in km                              |
| fare_inr          | float    | Final fare in Indian Rupees                      |
| driver_rating     | float    | Driver rating (1.0–5.0)                          |
| is_completed      | bool     | Whether ride was completed                       |
| weather           | str      | clear / light_rain / heavy_rain / fog            |
| is_festival       | bool     | Public holiday or major festival day             |
| is_ipl_day        | bool     | IPL match day in that city                       |

## Setup
```bash
git clone <your-repo-url>
cd smart_city_mobility
pip install -r requirements.txt
python src/generate_data.py        # Generates data/raw/rides.csv (~500K rows)
jupyter notebook notebooks/        # Open EDA notebooks
```

## Key Findings
*(To be filled after EDA — Phase 3)*

## Tech Stack
Python · pandas · NumPy · scikit-learn · XGBoost · LightGBM · SHAP · Streamlit · Plotly · DuckDB

## Author
Portfolio Project — Smart City Mobility Intelligence System
