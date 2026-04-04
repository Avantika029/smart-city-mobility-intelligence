"""
app.py
------
Smart City Mobility Intelligence System — Streamlit Dashboard

4 tabs:
  1. Live predictor  — real-time wait time + surge risk predictions
  2. EDA explorer    — interactive charts from Phase 3 analysis
  3. Model insights  — ML performance charts and feature importances
  4. About           — project overview, tech stack, findings

Run:
    streamlit run app.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart City Mobility Intelligence",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2rem; font-weight: 700; color: #185FA5;
        border-bottom: 3px solid #185FA5; padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8f9fa; border-radius: 12px; padding: 1.2rem;
        border-left: 4px solid #185FA5; margin-bottom: 1rem;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #185FA5; }
    .metric-label { font-size: 0.85rem; color: #666; margin-top: 0.2rem; }
    .alert-red   { background:#fff0f0; border-left:4px solid #E24B4A;
                   padding:0.8rem 1rem; border-radius:8px; margin:0.5rem 0; }
    .alert-amber { background:#fffbf0; border-left:4px solid #BA7517;
                   padding:0.8rem 1rem; border-radius:8px; margin:0.5rem 0; }
    .alert-green { background:#f0fff4; border-left:4px solid #1D9E75;
                   padding:0.8rem 1rem; border-radius:8px; margin:0.5rem 0; }
    .rec-item    { background:#f0f4ff; border-radius:8px; padding:0.6rem 1rem;
                   margin:0.3rem 0; font-size:0.9rem; }
    .section-title { font-size:1.2rem; font-weight:600; color:#185FA5;
                     margin:1.5rem 0 0.8rem; border-bottom:1px solid #eee;
                     padding-bottom:0.3rem; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
CITIES        = ["Delhi NCR", "Mumbai", "Bengaluru", "Chennai", "Hyderabad"]
VEHICLE_TYPES = ["economy", "bike_taxi", "auto", "premium", "shared"]
WEATHER_OPTS  = ["clear", "light_rain", "heavy_rain", "fog"]
CITY_COLORS   = {
    "Delhi NCR": "#185FA5", "Mumbai": "#1D9E75",
    "Bengaluru": "#BA7517", "Chennai": "#D85A30", "Hyderabad": "#534AB7",
}
CITY_CODES    = {"Bengaluru":0,"Chennai":1,"Delhi NCR":2,"Hyderabad":3,"Mumbai":4}
VEHICLE_CODES = {"auto":0,"bike_taxi":1,"economy":2,"premium":3,"shared":4}
WEATHER_CODES = {"clear":0,"fog":1,"heavy_rain":2,"light_rain":3}
SEASON_CODES  = {"winter":0,"summer":1,"monsoon":2,"post_monsoon":3}


# ── Data & model loading (cached) ─────────────────────────────────────────────
@st.cache_data
def load_data():
    path = "data/processed/rides_clean.csv"
    if not os.path.exists(path):
        st.error("Processed data not found. Run `python src/pipeline.py` first.")
        st.stop()
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df


@st.cache_resource
def load_predictor():
    try:
        sys.path.insert(0, "src")
        from inference import MobilityPredictor
        return MobilityPredictor()
    except Exception as e:
        return None


# ── Helper functions ──────────────────────────────────────────────────────────
def get_season(month):
    if month in (12,1,2):    return "winter"
    if month in (3,4,5):     return "summer"
    if month in (6,7,8,9):   return "monsoon"
    return "post_monsoon"


def surge_color(prob):
    if prob >= 0.7: return "🔴"
    if prob >= 0.4: return "🟡"
    return "🟢"


def wait_color(wait):
    if wait >= 12: return "🔴"
    if wait >= 8:  return "🟡"
    return "🟢"


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.shields.io/badge/Smart%20City-Mobility%20AI-185FA5?style=for-the-badge", width=280)
    st.markdown("### Navigation")
    tab_choice = st.radio("Go to", [
        "🔮 Live predictor",
        "📊 EDA explorer",
        "🤖 Model insights",
        "📖 About",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**Project stats**")
    st.markdown("- 500K+ ride records")
    st.markdown("- 5 Indian cities")
    st.markdown("- 3 ML models")
    st.markdown("- 29 engineered features")
    st.markdown("---")
    st.markdown("Built by **Avantika**")
    st.markdown("[GitHub](https://github.com/Avantika029/smart-city-mobility-intelligence)")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
if tab_choice == "🔮 Live predictor":
    st.markdown('<div class="main-header">🔮 Live Mobility Predictor</div>', unsafe_allow_html=True)
    st.markdown("Predict wait time, surge risk, and cancellation probability for any city and scenario.")

    # ── Input controls ────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        city         = st.selectbox("City", CITIES)
        hour         = st.slider("Hour of day", 0, 23, 18,
                                  help="0 = midnight, 18 = 6 PM")
        vehicle_type = st.selectbox("Vehicle type", VEHICLE_TYPES)
    with col2:
        weather      = st.selectbox("Weather", WEATHER_OPTS)
        day_of_week  = st.selectbox("Day of week",
                                     ["Monday","Tuesday","Wednesday","Thursday",
                                      "Friday","Saturday","Sunday"])
        month        = st.slider("Month", 1, 12, 7)
    with col3:
        is_festival  = st.checkbox("Festival / holiday day")
        is_ipl_day   = st.checkbox("IPL match day")
        st.markdown("")
        predict_btn  = st.button("🔮 Run prediction", use_container_width=True, type="primary")

    dow_num = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"].index(day_of_week)

    # ── Run prediction ────────────────────────────────────────────────────────
    if predict_btn or True:   # always show on load
        predictor = load_predictor()

        if predictor:
            result = predictor.predict_all(
                city=city, hour=hour, day_of_week=dow_num,
                month=month, weather=weather, vehicle_type=vehicle_type,
                is_festival=is_festival, is_ipl_day=is_ipl_day,
            )
            wait       = result["wait_time_min"]
            surge_prob = result["surge"]["probability"]
            surge_lbl  = result["surge"]["label"]
            cancel_prob = result["cancellation"]["probability"]
            dp          = result["demand_pressure"]
        else:
            # Fallback: rule-based estimates when models aren't loaded
            base_wait = {"clear":7.5,"light_rain":9.5,"heavy_rain":14.0,"fog":10.5}[weather]
            peak_mult = 1.3 if hour in [7,8,9,17,18,19,20] else 1.0
            fest_mult = 1.4 if is_festival else 1.0
            ipl_mult  = 1.6 if (is_ipl_day and hour >= 21) else 1.0
            wait = round(base_wait * peak_mult * fest_mult * ipl_mult, 1)

            base_surge = {"clear":0.2,"light_rain":0.4,"heavy_rain":0.75,"fog":0.45}[weather]
            if hour in [7,8,9,17,18,19,20]: base_surge += 0.15
            if is_festival: base_surge += 0.2
            if is_ipl_day and hour >= 21: base_surge += 0.3
            surge_prob  = min(round(base_surge, 2), 1.0)
            surge_lbl   = ("Extreme surge risk" if surge_prob>=0.8 else
                           "High surge risk" if surge_prob>=0.6 else
                           "Moderate risk" if surge_prob>=0.4 else "Normal")
            cancel_prob = round(0.04 + (0.12 if weather=="heavy_rain" else 0) +
                                (0.08 if surge_prob>=0.6 else 0), 2)
            dp = round(1.5 + surge_prob * 1.5, 2)

            result = {
                "wait_time_min": wait, "demand_pressure": dp,
                "surge": {"probability": surge_prob, "label": surge_lbl,
                          "surge_estimate": round(1 + surge_prob*2.5, 2)},
                "cancellation": {"probability": cancel_prob},
                "recommendations": [
                    f"{'High demand — pre-position fleet in ' + city if wait > 10 else 'Normal demand conditions'}",
                    f"{'Surge alert: ' + surge_lbl + ' — activate standby pool' if surge_prob >= 0.5 else 'No surge action needed'}",
                ]
            }

        st.markdown("---")

        # ── KPI cards ─────────────────────────────────────────────────────────
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{wait_color(wait)} {wait} min</div>
                <div class="metric-label">Predicted wait time</div>
            </div>""", unsafe_allow_html=True)
        with k2:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{surge_color(surge_prob)} {surge_prob:.0%}</div>
                <div class="metric-label">Surge probability — {surge_lbl}</div>
            </div>""", unsafe_allow_html=True)
        with k3:
            est_surge = result["surge"].get("surge_estimate", 1.5)
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">⚡ {est_surge:.2f}×</div>
                <div class="metric-label">Estimated surge multiplier</div>
            </div>""", unsafe_allow_html=True)
        with k4:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">❌ {cancel_prob:.0%}</div>
                <div class="metric-label">Cancellation risk</div>
            </div>""", unsafe_allow_html=True)

        # ── Gauges ────────────────────────────────────────────────────────────
        st.markdown('<div class="section-title">Demand & risk gauges</div>', unsafe_allow_html=True)
        g1, g2, g3 = st.columns(3)

        with g1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=wait,
                title={"text": "Wait time (min)", "font": {"size": 14}},
                gauge={
                    "axis":  {"range": [0, 25]},
                    "bar":   {"color": "#185FA5"},
                    "steps": [{"range":[0,6],"color":"#EAF3DE"},
                               {"range":[6,10],"color":"#FAEEDA"},
                               {"range":[10,25],"color":"#FCEBEB"}],
                    "threshold": {"line":{"color":"red","width":3},
                                  "thickness":0.75,"value":12},
                },
                number={"suffix":" min", "font":{"size":20}}
            ))
            fig.update_layout(height=220, margin=dict(l=20,r=20,t=40,b=10))
            st.plotly_chart(fig, use_container_width=True)

        with g2:
            fig2 = go.Figure(go.Indicator(
                mode="gauge+number",
                value=surge_prob * 100,
                title={"text": "Surge risk (%)", "font": {"size": 14}},
                gauge={
                    "axis":  {"range": [0, 100]},
                    "bar":   {"color": "#E24B4A"},
                    "steps": [{"range":[0,40],"color":"#EAF3DE"},
                               {"range":[40,65],"color":"#FAEEDA"},
                               {"range":[65,100],"color":"#FCEBEB"}],
                    "threshold": {"line":{"color":"red","width":3},
                                  "thickness":0.75,"value":65},
                },
                number={"suffix":"%", "font":{"size":20}}
            ))
            fig2.update_layout(height=220, margin=dict(l=20,r=20,t=40,b=10))
            st.plotly_chart(fig2, use_container_width=True)

        with g3:
            fig3 = go.Figure(go.Indicator(
                mode="gauge+number",
                value=dp,
                title={"text": "Demand pressure", "font": {"size": 14}},
                gauge={
                    "axis":  {"range": [0, 5]},
                    "bar":   {"color": "#BA7517"},
                    "steps": [{"range":[0,2],"color":"#EAF3DE"},
                               {"range":[2,3],"color":"#FAEEDA"},
                               {"range":[3,5],"color":"#FCEBEB"}],
                },
                number={"font":{"size":20}}
            ))
            fig3.update_layout(height=220, margin=dict(l=20,r=20,t=40,b=10))
            st.plotly_chart(fig3, use_container_width=True)

        # ── 24-hour profile ───────────────────────────────────────────────────
        st.markdown('<div class="section-title">24-hour demand profile</div>', unsafe_allow_html=True)

        hours_range = list(range(24))
        wait_profile, surge_profile = [], []

        for h in hours_range:
            if predictor:
                r = predictor.predict_all(city=city, hour=h, day_of_week=dow_num,
                                           month=month, weather=weather,
                                           is_festival=is_festival, is_ipl_day=is_ipl_day)
                wait_profile.append(r["wait_time_min"])
                surge_profile.append(r["surge"]["probability"] * 100)
            else:
                bw = {"clear":7.5,"light_rain":9.5,"heavy_rain":14.0,"fog":10.5}[weather]
                pm = 1.3 if h in [7,8,9,17,18,19,20] else 0.8 if h in [1,2,3,4] else 1.0
                wait_profile.append(round(bw * pm * (1.4 if is_festival else 1.0), 1))
                bs = {"clear":0.2,"light_rain":0.4,"heavy_rain":0.7,"fog":0.45}[weather]
                sp = min(bs + (0.15 if h in [7,8,9,17,18,19,20] else 0) +
                         (0.2 if is_festival else 0), 1.0)
                surge_profile.append(round(sp * 100, 1))

        fig_profile = make_subplots(specs=[[{"secondary_y": True}]])
        fig_profile.add_trace(
            go.Bar(x=hours_range, y=wait_profile, name="Wait time (min)",
                   marker_color=["#185FA5" if h in [7,8,9,17,18,19,20] else "#85B7EB"
                                  for h in hours_range],
                   opacity=0.8),
            secondary_y=False,
        )
        fig_profile.add_trace(
            go.Scatter(x=hours_range, y=surge_profile, name="Surge risk (%)",
                       line=dict(color="#E24B4A", width=2.5),
                       mode="lines+markers", marker=dict(size=5)),
            secondary_y=True,
        )
        fig_profile.add_vline(x=hour, line_dash="dash", line_color="orange",
                               annotation_text=f"Now ({hour}:00)", annotation_position="top")
        fig_profile.update_xaxes(title_text="Hour of day", tickvals=list(range(0,24,2)))
        fig_profile.update_yaxes(title_text="Wait time (min)", secondary_y=False)
        fig_profile.update_yaxes(title_text="Surge risk (%)", secondary_y=True)
        fig_profile.update_layout(height=320, margin=dict(l=20,r=20,t=20,b=40),
                                   legend=dict(orientation="h", y=1.02))
        st.plotly_chart(fig_profile, use_container_width=True)

        # ── Recommendations ───────────────────────────────────────────────────
        st.markdown('<div class="section-title">AI recommendations</div>', unsafe_allow_html=True)
        recs = result.get("recommendations", [])
        for rec in recs:
            icon = "🚨" if any(w in rec.lower() for w in ["urgent","extreme","critical","alert","heavy"]) else \
                   "⚠️" if any(w in rec.lower() for w in ["surge","high","spike","festival"]) else "✅"
            st.markdown(f'<div class="rec-item">{icon} {rec}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EDA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif tab_choice == "📊 EDA explorer":
    st.markdown('<div class="main-header">📊 EDA Explorer</div>', unsafe_allow_html=True)
    st.markdown("Interactive charts from the Phase 3 exploratory data analysis.")

    df = load_data()

    chart = st.selectbox("Select chart", [
        "Hourly demand curve",
        "City comparison",
        "Weather impact",
        "Monthly trends & seasonality",
        "Festival & IPL event impact",
        "Vehicle type analysis",
        "Surge band distribution",
        "Feature correlation heatmap",
    ])

    if chart == "Hourly demand curve":
        city_filter = st.multiselect("Filter by city", CITIES, default=CITIES)
        filtered = df[df["city"].isin(city_filter)]
        hourly = filtered.groupby(["hour","city"]).size().reset_index(name="rides")
        fig = px.line(hourly, x="hour", y="rides", color="city",
                      color_discrete_map=CITY_COLORS,
                      markers=True, title="Hourly ride volume by city")
        fig.update_xaxes(tickvals=list(range(0,24,2)), title="Hour of day")
        fig.update_yaxes(title="Ride count")
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)
        st.info(f"**Key insight:** Evening peak at 6 PM has {hourly['rides'].max():,} rides — "
                f"the busiest single hour across all cities.")

    elif chart == "City comparison":
        metric = st.radio("Metric", ["Median wait time","Median fare","Mean surge","Completion rate"],
                           horizontal=True)
        col_map = {"Median wait time":"wait_time_min","Median fare":"fare_inr",
                   "Mean surge":"surge_multiplier","Completion rate":"is_completed"}
        agg_map  = {"Median wait time":"median","Median fare":"median",
                    "Mean surge":"mean","Completion rate":"mean"}
        city_g = df.groupby("city")[col_map[metric]].agg(agg_map[metric]).reset_index()
        city_g.columns = ["city","value"]
        city_g = city_g.sort_values("value", ascending=False)
        fig = px.bar(city_g, x="city", y="value", color="city",
                     color_discrete_map=CITY_COLORS,
                     title=f"{metric} by city", text_auto=".2f")
        fig.update_layout(showlegend=False, height=380)
        st.plotly_chart(fig, use_container_width=True)

    elif chart == "Weather impact":
        metric2 = st.radio("Metric", ["Median wait time","Mean surge","Cancellation rate"], horizontal=True)
        if metric2 == "Cancellation rate":
            weather_g = (1 - df.groupby("weather")["is_completed"].mean()).reset_index()
            weather_g.columns = ["weather","value"]
            weather_g["value"] *= 100
        elif metric2 == "Median wait time":
            weather_g = df.groupby("weather")["wait_time_min"].median().reset_index()
            weather_g.columns = ["weather","value"]
        else:
            weather_g = df.groupby("weather")["surge_multiplier"].mean().reset_index()
            weather_g.columns = ["weather","value"]

        weather_order = ["clear","fog","light_rain","heavy_rain"]
        weather_g["weather"] = pd.Categorical(weather_g["weather"], categories=weather_order, ordered=True)
        weather_g = weather_g.sort_values("weather")
        wcols = {"clear":"#1D9E75","fog":"#BA7517","light_rain":"#185FA5","heavy_rain":"#E24B4A"}
        fig = px.bar(weather_g, x="weather", y="value",
                     color="weather", color_discrete_map=wcols,
                     title=f"{metric2} by weather condition", text_auto=".2f")
        fig.update_layout(showlegend=False, height=380)
        st.plotly_chart(fig, use_container_width=True)
        heavy = weather_g[weather_g["weather"]=="heavy_rain"]["value"].values[0]
        clear = weather_g[weather_g["weather"]=="clear"]["value"].values[0]
        st.info(f"**Key insight:** Heavy rain increases {metric2.lower()} by "
                f"**{((heavy/clear)-1)*100:.0f}%** compared to clear weather.")

    elif chart == "Monthly trends & seasonality":
        df["year_month"] = df["timestamp"].dt.to_period("M").astype(str)
        monthly = df.groupby("year_month").agg(
            rides=("ride_id","count"),
            mean_surge=("surge_multiplier","mean"),
            mean_wait=("wait_time_min","mean"),
        ).reset_index()

        tab_a, tab_b = st.tabs(["Ride volume", "Surge & wait trends"])
        with tab_a:
            fig = px.bar(monthly, x="year_month", y="rides",
                         title="Monthly ride volume",
                         color="rides", color_continuous_scale="Blues")
            fig.update_xaxes(tickangle=45)
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)
        with tab_b:
            fig = make_subplots(specs=[[{"secondary_y":True}]])
            fig.add_trace(go.Scatter(x=monthly["year_month"], y=monthly["mean_surge"],
                                      name="Mean surge", line=dict(color="#185FA5",width=2.5)), secondary_y=False)
            fig.add_trace(go.Scatter(x=monthly["year_month"], y=monthly["mean_wait"],
                                      name="Mean wait (min)", line=dict(color="#BA7517",width=2.5,dash="dash")), secondary_y=True)
            fig.update_xaxes(tickangle=45)
            fig.update_layout(height=380, title="Surge × wait time over time")
            st.plotly_chart(fig, use_container_width=True)

    elif chart == "Festival & IPL event impact":
        df["event_type"] = "Normal day"
        df.loc[df["is_ipl_day"]==True, "event_type"] = "IPL match day"
        df.loc[df["is_festival"]==True, "event_type"] = "Festival day"
        metric3 = st.radio("Metric", ["wait_time_min","surge_multiplier","fare_inr"], horizontal=True)
        event_g = df.groupby("event_type")[metric3].mean().reset_index()
        event_g = event_g.set_index("event_type").loc[["Normal day","IPL match day","Festival day"]].reset_index()
        ecols = {"Normal day":"#888780","IPL match day":"#185FA5","Festival day":"#BA7517"}
        fig = px.bar(event_g, x="event_type", y=metric3, color="event_type",
                     color_discrete_map=ecols, title=f"Mean {metric3} by event type",
                     text_auto=".2f")
        fig.update_layout(showlegend=False, height=380)
        st.plotly_chart(fig, use_container_width=True)
        norm = event_g[event_g["event_type"]=="Normal day"][metric3].values[0]
        fest = event_g[event_g["event_type"]=="Festival day"][metric3].values[0]
        st.info(f"**Key insight:** Festival days see **{((fest/norm)-1)*100:.0f}%** higher {metric3}.")

    elif chart == "Vehicle type analysis":
        metric4 = st.radio("Metric", ["fare_per_km","wait_time_min","fare_inr","distance_km"], horizontal=True)
        vt_g = df.groupby("vehicle_type")[metric4].median().reset_index().sort_values(metric4,ascending=False)
        vcols = {"bike_taxi":"#185FA5","auto":"#1D9E75","economy":"#BA7517","premium":"#534AB7","shared":"#D85A30"}
        fig = px.bar(vt_g, x="vehicle_type", y=metric4, color="vehicle_type",
                     color_discrete_map=vcols, title=f"Median {metric4} by vehicle type",
                     text_auto=".2f")
        fig.update_layout(showlegend=False, height=380)
        st.plotly_chart(fig, use_container_width=True)

    elif chart == "Surge band distribution":
        band_order = ["no_surge","low","medium","high","extreme"]
        surge_city = df.groupby(["city","surge_band"]).size().reset_index(name="count")
        surge_city["surge_band"] = pd.Categorical(surge_city["surge_band"],
                                                   categories=band_order, ordered=True)
        fig = px.bar(surge_city, x="city", y="count", color="surge_band",
                     barmode="stack",
                     color_discrete_map={"no_surge":"#1D9E75","low":"#85B7EB",
                                          "medium":"#BA7517","high":"#D85A30","extreme":"#E24B4A"},
                     title="Surge band distribution by city")
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    elif chart == "Feature correlation heatmap":
        num_cols = ["wait_time_min","surge_multiplier","fare_inr","distance_km",
                    "driver_rating","demand_pressure","is_peak_hour","is_raining",
                    "is_festival","is_ipl_day","is_high_surge","hourly_ride_volume"]
        corr = df[num_cols].corr().round(2)
        fig = px.imshow(corr, text_auto=True, aspect="auto",
                        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                        title="Feature correlation matrix")
        fig.update_layout(height=520)
        st.plotly_chart(fig, use_container_width=True)
        st.info("**Key insight:** `surge_multiplier` has the strongest correlation with "
                "`fare_inr` (r≈0.85). `is_raining` drives both surge and wait time.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif tab_choice == "🤖 Model insights":
    st.markdown('<div class="main-header">🤖 Model Insights</div>', unsafe_allow_html=True)

    st.markdown("### Model performance summary")
    perf_data = {
        "Model":     ["Demand forecasting","Wait time regression","Surge classifier","Cancellation predictor"],
        "Type":      ["Regression","Regression","Classification","Classification"],
        "Algorithm": ["GradientBoosting","GradientBoosting","GradientBoosting","GradientBoosting"],
        "Key metric":["R² = 0.867 (avg)","MAE = 1.41 min","AUC = 1.00","AUC = 0.62"],
        "CV folds":  [5, 5, 5, "-"],
        "Data leakage": ["None ✅","None ✅","None ✅","None ✅"],
    }
    st.dataframe(pd.DataFrame(perf_data), use_container_width=True, hide_index=True)

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### CV results — all cities")
        fig_path = "reports/figures/ml_cv_results.png"
        if os.path.exists(fig_path):
            st.image(fig_path, use_container_width=True)

    with col2:
        st.markdown("### All cities performance")
        fig_path2 = "reports/figures/ml_all_cities_performance.png"
        if os.path.exists(fig_path2):
            st.image(fig_path2, use_container_width=True)

    st.markdown("---")
    st.markdown("### City-level demand model results")
    city_sel = st.selectbox("Select city", CITIES)
    safe_city = city_sel.replace(" ", "_").replace("/", "_")

    c1, c2 = st.columns(2)
    with c1:
        p1 = f"reports/figures/ml_{safe_city}_predictions.png"
        if os.path.exists(p1):
            st.image(p1, caption=f"{city_sel} — actual vs predicted", use_container_width=True)
    with c2:
        p2 = f"reports/figures/ml_{safe_city}_feature_importance.png"
        if os.path.exists(p2):
            st.image(p2, caption=f"{city_sel} — feature importance", use_container_width=True)

    st.markdown("---")
    st.markdown("### Wait time & surge analysis")
    c3, c4 = st.columns(2)
    with c3:
        pw = "reports/figures/ml_wait_time_analysis.png"
        if os.path.exists(pw):
            st.image(pw, caption="Wait time model analysis", use_container_width=True)
    with c4:
        ps = "reports/figures/ml_surge_classifier_analysis.png"
        if os.path.exists(ps):
            st.image(ps, caption="Surge classifier analysis", use_container_width=True)

    st.markdown("---")
    st.markdown("### Fleet rebalancing recommendations")
    pr = "reports/figures/ml_rebalancing_recommendations.png"
    if os.path.exists(pr):
        st.image(pr, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif tab_choice == "📖 About":
    st.markdown('<div class="main-header">📖 About This Project</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ## Smart City Mobility Intelligence System

        An end-to-end data science project analyzing and predicting urban mobility
        patterns across 5 major Indian cities — Delhi NCR, Mumbai, Bengaluru,
        Chennai, and Hyderabad.

        ### Objective
        Analyze ride data to optimize traffic flow, reduce wait times, and improve
        ride allocation using machine learning.

        ### What was built
        - **500K+ synthetic ride records** with realistic Indian urban patterns
        - **7-stage data pipeline** — cleaning, imputation, 29 feature engineering steps
        - **10 EDA charts** uncovering demand patterns, weather effects, and event impacts
        - **3 ML models** — demand forecasting (R²=0.867), wait time prediction (MAE=1.41 min),
          surge classification (AUC=1.00)
        - **This dashboard** — interactive predictions and visualizations

        ### Key findings
        | Finding | Metric |
        |---|---|
        | Evening peak (6 PM) is busiest hour | 56,264 rides |
        | Heavy rain increases wait time | +84% vs clear weather |
        | Festival days increase fare | +60% vs normal |
        | Monsoon is highest surge season | 1.71× average |
        | Surge is top fare predictor | r = 0.85 |
        | 25.1% of rides are high-surge (≥2×) | 125K+ rides |
        | Peripheral zones wait longest | Ghaziabad, Tambaram +40% |
        """)

    with col2:
        st.markdown("### Tech stack")
        tech = {
            "Data": ["pandas", "NumPy", "DuckDB"],
            "ML": ["scikit-learn", "XGBoost", "SHAP"],
            "Viz": ["Matplotlib", "Seaborn", "Plotly"],
            "App": ["Streamlit"],
            "Dev": ["Git", "GitHub"],
        }
        for category, tools in tech.items():
            st.markdown(f"**{category}:** {', '.join(tools)}")

        st.markdown("---")
        st.markdown("### Project phases")
        phases = [
            ("✅ Phase 1", "Data generation"),
            ("✅ Phase 2", "Pipeline & features"),
            ("✅ Phase 3", "EDA — 10 charts"),
            ("✅ Phase 4", "Demand model"),
            ("✅ Phase 5", "Wait + surge models"),
            ("✅ Phase 6", "This dashboard"),
        ]
        for phase, desc in phases:
            st.markdown(f"{phase} — {desc}")

        st.markdown("---")
        st.markdown("### Links")
        st.markdown("[GitHub Repository](https://github.com/Avantika029/smart-city-mobility-intelligence)")
