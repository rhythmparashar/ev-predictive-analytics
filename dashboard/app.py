"""
EV Telemetry Intelligence Dashboard
Stable Professional Version (Fixed Date 26)
Run: streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time

# ───────────────────────── CONFIG ─────────────────────────
st.set_page_config(page_title="EV SOC Intelligence", layout="wide")

TARGET = "y_soc_t_plus_300s"
DATA_DIR = Path("data/gold/window_features")
MODEL_PATH = Path("models/soc_xgb_baseline.json")

FIXED_DATE = "2026-02-26"
VEHICLE_ID = "EV01"
ROW_LIMIT = 20000

# ───────────────────────── LOAD MODEL ─────────────────────────
@st.cache_resource
def load_model():
    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)
    return model

# ───────────────────────── LOAD DATA ─────────────────────────
@st.cache_data
def load_day():
    base = DATA_DIR / f"dt={FIXED_DATE}" / f"vehicle_id={VEHICLE_ID}"
    files = list(base.glob("*.parquet"))

    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except:
            continue

    if not dfs:
        return None

    df = pd.concat(dfs)

    if len(df) > ROW_LIMIT:
        df = df.sample(ROW_LIMIT).sort_values("timestamp")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df

# ───────────────────────── RUN PREDICTIONS ─────────────────────────
def run_predictions(df, model):

    if "label_available" in df.columns:
        df = df[df["label_available"] == 1].copy()

    feature_cols = model.get_booster().feature_names
    feature_cols = [f for f in feature_cols if f in df.columns]

    X = df[feature_cols].fillna(0).astype("float32")

    df["pred"] = model.predict(X)
    df["error"] = df["pred"] - df[TARGET]
    df["abs_error"] = df["error"].abs()

    return df

# ───────────────────────── LOAD EVERYTHING ─────────────────────────
st.title("⚡ EV SOC Intelligence Dashboard")
st.caption(f"Vehicle: {VEHICLE_ID} | Date: {FIXED_DATE}")

df_raw = load_day()
if df_raw is None:
    st.error("No data found.")
    st.stop()

model = load_model()
df = run_predictions(df_raw, model)

page = st.radio(
    "Navigate",
    ["Overview", "Technical Analysis", "Single Trip Deep Dive", "Live Replay"],
    horizontal=True
)

# ═════════════════════════════════════════════
# OVERVIEW
# ═════════════════════════════════════════════
if page == "Overview":

    mae = mean_absolute_error(df[TARGET], df["pred"])
    rmse = np.sqrt(mean_squared_error(df[TARGET], df["pred"]))
    bias = df["error"].mean()

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{mae:.3f}%")
    c2.metric("RMSE", f"{rmse:.3f}%")
    c3.metric("Bias", f"{bias:+.3f}%")

    # SOC vs Predicted
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df[TARGET],
                             mode="lines", name="Actual SOC",
                             line=dict(color="#3b82f6", width=3)))
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["pred"],
                             mode="lines", name="Predicted SOC",
                             line=dict(color="#f59e0b", dash="dash")))
    fig.update_layout(template="plotly_dark", height=500,
                      title="SOC Actual vs Predicted (5-Min Forecast)")
    st.plotly_chart(fig, width="stretch")

    # Error Distribution
    fig2 = px.histogram(df, x="error", nbins=60, title="Prediction Error Distribution")
    fig2.update_layout(template="plotly_dark")
    st.plotly_chart(fig2, width="stretch")

# ═════════════════════════════════════════════
# TECHNICAL ANALYSIS
# ═════════════════════════════════════════════
elif page == "Technical Analysis":

    # Feature Importance
    importance = pd.Series(model.feature_importances_,
                           index=model.get_booster().feature_names)
    importance = importance.sort_values(ascending=False).head(20)

    fig = px.bar(importance[::-1],
                 orientation="h",
                 title="Top 20 Feature Importance")
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, width="stretch")

    # Scatter Actual vs Pred
    fig2 = px.scatter(df.sample(min(3000, len(df))),
                      x=TARGET, y="pred",
                      title="Actual vs Predicted Scatter")
    fig2.update_layout(template="plotly_dark")
    st.plotly_chart(fig2, width="stretch")

    # Error vs Power
    if "output_power_kw" in df.columns:
        fig3 = px.scatter(df.sample(min(3000, len(df))),
                          x="output_power_kw", y="error",
                          title="Error vs Output Power")
        fig3.update_layout(template="plotly_dark")
        st.plotly_chart(fig3, width="stretch")

# ═════════════════════════════════════════════
# SINGLE TRIP
# ═════════════════════════════════════════════
elif page == "Single Trip Deep Dive":

    trip_ids = sorted(df["trip_id"].unique())
    selected_trip = st.selectbox("Select Trip", trip_ids)

    tdf = df[df["trip_id"] == selected_trip].sort_values("timestamp")

    mae_trip = tdf["abs_error"].mean()
    soc_drop = tdf[TARGET].iloc[0] - tdf[TARGET].iloc[-1]

    c1, c2 = st.columns(2)
    c1.metric("Trip MAE", f"{mae_trip:.3f}%")
    c2.metric("SOC Drop", f"{soc_drop:.2f}%")

    # SOC Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tdf["timestamp"], y=tdf[TARGET],
                             mode="lines", name="Actual",
                             line=dict(color="#3b82f6", width=3)))
    fig.add_trace(go.Scatter(x=tdf["timestamp"], y=tdf["pred"],
                             mode="lines", name="Predicted",
                             line=dict(color="#f59e0b", dash="dash")))
    fig.update_layout(template="plotly_dark",
                      height=500,
                      title=f"Trip {selected_trip} SOC Analysis")
    st.plotly_chart(fig, width="stretch")

    # Error over time
    fig2 = px.bar(tdf, x="timestamp", y="error",
                  title="Prediction Error Over Time")
    fig2.update_layout(template="plotly_dark")
    st.plotly_chart(fig2, width="stretch")

    # Power graph
    if "output_power_kw" in tdf.columns:
        fig3 = px.line(tdf, x="timestamp", y="output_power_kw",
                       title="Output Power During Trip")
        fig3.update_layout(template="plotly_dark")
        st.plotly_chart(fig3, width="stretch")

# ═════════════════════════════════════════════
# LIVE REPLAY
# ═════════════════════════════════════════════
elif page == "Live Replay":

    st.subheader("🔴 Live SOC Replay (Real Model Output)")

    trip_ids = sorted(df["trip_id"].unique())
    selected_trip = st.selectbox("Select Trip", trip_ids, key="live_trip")

    tdf = df[df["trip_id"] == selected_trip].sort_values("timestamp").reset_index(drop=True)

    speed = st.slider("Playback Speed (ms)", 20, 200, 40)

    chart_placeholder = st.empty()
    metrics_placeholder = st.empty()

    actual_x, actual_y = [], []
    pred_x, pred_y = [], []

    for i in range(len(tdf)):

        row = tdf.iloc[i]

        actual_x.append(row["timestamp"])
        actual_y.append(row[TARGET])

        pred_x.append(row["timestamp"])
        pred_y.append(row["pred"])

        with metrics_placeholder.container():
            c1, c2, c3 = st.columns(3)
            c1.metric("Actual SOC", f"{row[TARGET]:.2f}%")
            c2.metric("Predicted (5m)", f"{row['pred']:.2f}%")
            c3.metric("Error", f"{row['error']:+.2f}%")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=actual_x, y=actual_y,
                                 mode="lines", name="Actual",
                                 line=dict(color="#3b82f6", width=3)))
        fig.add_trace(go.Scatter(x=pred_x, y=pred_y,
                                 mode="lines", name="Predicted",
                                 line=dict(color="#f59e0b", dash="dash")))
        fig.update_layout(template="plotly_dark", height=450)

        chart_placeholder.plotly_chart(fig, width="stretch")

        time.sleep(speed / 1000)