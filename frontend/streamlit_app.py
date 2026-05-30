"""
streamlit_app.py — Pearls AQI Predictor Dashboard
═══════════════════════════════════════════════════
Satisfies PDF requirement: "Use Streamlit/Gradio and Flask/FastAPI for the web app"

Run locally:  streamlit run frontend/streamlit_app.py
Calls the Flask backend API at localhost:5000 (or BACKEND_URL env var).
"""

import os
import requests
import pandas as pd
import streamlit as st

# ── Config ────────────────────────────────────────────
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:5000")

st.set_page_config(
    page_title="Pearls AQI Predictor — Karachi",
    page_icon="🌫️",
    layout="wide",
)

# ── Header ────────────────────────────────────────────
st.title("🌫️ Pearls AQI Predictor")
st.caption("Real-time Air Quality Index forecasting for Karachi — 10Pearls Shine Program")

# ── Fetch data from Flask API ──────────────────────────
@st.cache_data(ttl=300)  # cache for 5 minutes
def fetch_prediction():
    try:
        r = requests.get(f"{BACKEND_URL}/api/predict", timeout=10)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)

data, error = fetch_prediction()

if error:
    st.error(f"Could not reach Flask API: {error}")
    st.info("Make sure Flask backend is running: `cd backend && python app.py`")
    st.stop()

# ── Alert Banner ──────────────────────────────────────
alert = data.get("alert")
if alert:
    color = alert.get("color", "#ef4444")
    st.markdown(
        f'<div style="background:{color}22;border-left:4px solid {color};'
        f'padding:0.75rem 1rem;border-radius:8px;margin-bottom:1rem;">'
        f'<strong>{alert["message"]}</strong></div>',
        unsafe_allow_html=True
    )

# ── Current AQI ───────────────────────────────────────
current = data["current"]
aqi     = current["aqi"]
cat     = current.get("category", {})

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Current AQI", aqi)
col2.metric("PM2.5 (µg/m³)", current["pm25"])
col3.metric("Temperature (°C)", current["temperature"])
col4.metric("Humidity (%)", current["humidity"])
col5.metric("Wind (m/s)", current["wind_speed"])

st.markdown("---")

# ── 3-Day Forecast ────────────────────────────────────
st.subheader("3-Day AQI Forecast")

forecasts = data.get("forecast", [])
if forecasts:
    cols = st.columns(3)
    for i, fc in enumerate(forecasts):
        with cols[i]:
            color = fc.get("color", "#888")
            st.markdown(
                f'<div style="background:{color}22;border:1px solid {color};'
                f'border-radius:12px;padding:1rem;text-align:center;">'
                f'<div style="font-size:13px;color:#888">{fc["weekday"]}, {fc["date"]}</div>'
                f'<div style="font-size:36px;font-weight:700;color:{color}">{fc["aqi"]}</div>'
                f'<div style="font-size:14px;color:{color}">{fc["category"]}</div>'
                f'<div style="font-size:12px;color:#888;margin-top:4px">'
                f'Confidence: {fc["confidence"]}% · Model: {fc["model_used"].upper()}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    # Forecast chart
    st.markdown("####")
    chart_df = pd.DataFrame([
        {"Horizon": f["horizon"], "Forecast AQI": f["aqi"]} for f in forecasts
    ])
    chart_df = pd.concat([
        pd.DataFrame([{"Horizon": "Now", "Forecast AQI": aqi}]),
        chart_df
    ])
    st.line_chart(chart_df.set_index("Horizon"))

st.markdown("---")

# ── SHAP Feature Importance ───────────────────────────
st.subheader("SHAP Feature Importance (24h forecast)")
st.caption("Which features drove this prediction the most")

shap_data = data.get("shap", [])
if shap_data:
    shap_df = pd.DataFrame(shap_data)
    shap_df["abs"] = shap_df["shap_value"].abs()
    shap_df = shap_df.sort_values("abs", ascending=True)

    # Color bars: positive = red (increases AQI), negative = green (reduces AQI)
    import plotly.graph_objects as go
    colors = ["#ef4444" if v > 0 else "#22c55e" for v in shap_df["shap_value"]]
    fig = go.Figure(go.Bar(
        x=shap_df["shap_value"],
        y=shap_df["label"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.1f}" for v in shap_df["shap_value"]],
        textposition="outside",
    ))
    fig.update_layout(
        xaxis_title="SHAP value (impact on AQI prediction)",
        margin=dict(l=10, r=40, t=10, b=10),
        height=350,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Red = increases AQI prediction | Green = decreases AQI prediction")

st.markdown("---")

# ── Model Performance Metrics ──────────────────────────
st.subheader("Model Performance")
metrics = data.get("metrics", {})
if metrics:
    m1, m2, m3 = st.columns(3)
    m1.metric("RMSE (24h)", f"{metrics.get('rmse', 'N/A'):.2f}" if isinstance(metrics.get('rmse'), float) else "N/A")
    m2.metric("MAE (24h)",  f"{metrics.get('mae',  'N/A'):.2f}" if isinstance(metrics.get('mae'),  float) else "N/A")
    m3.metric("R² (24h)",   f"{metrics.get('r2',   'N/A'):.3f}" if isinstance(metrics.get('r2'),   float) else "N/A")
else:
    st.info("Run the training pipeline to see model metrics.")

# ── Pollutant Breakdown ───────────────────────────────
st.markdown("---")
st.subheader("🧪 Pollutant Breakdown")
pollutants = {
    "PM2.5 (µg/m³)": current["pm25"],
    "PM10 (µg/m³)":  current["pm10"],
    "NO₂ (µg/m³)":   current["no2"],
    "O₃ (µg/m³)":    current["o3"],
    "SO₂ (µg/m³)":   current["so2"],
    "CO (mg/m³)":     current["co"],
}
poll_df = pd.DataFrame(list(pollutants.items()), columns=["Pollutant", "Value"])
st.dataframe(poll_df, use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────
st.markdown("---")
st.caption(
    f"Last updated: {data.get('fetched_at', 'N/A')} | "
    "Data: OpenWeather API | "
    "Models: Hopsworks Model Registry | "
    "10Pearls Shine Program 2025"
)