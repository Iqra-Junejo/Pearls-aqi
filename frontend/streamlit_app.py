"""
streamlit_app.py — Pearls AQI Predictor
Standalone dashboard — calls OpenWeather API directly.
Deployable on Streamlit Cloud without a separate Flask backend.

Set secrets in Streamlit Cloud:
  OPENWEATHER_API_KEY = "your_key"
"""

import math
import os
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
import streamlit as st

# ── Config ────────────────────────────────────────────
st.set_page_config(
    page_title="Pearls AQI Predictor — Karachi",
    page_icon="🌫️",
    layout="wide",
)

OW_KEY  = os.getenv("OPENWEATHER_API_KEY") or st.secrets.get("OPENWEATHER_API_KEY", "")
LAT, LON = 24.8607, 67.0011
CITY     = "Karachi"

AQI_CATS = [
    (0,   50,  "Good",                           "#22c55e"),
    (51,  100, "Moderate",                       "#eab308"),
    (101, 150, "Unhealthy for Sensitive Groups", "#f97316"),
    (151, 200, "Unhealthy",                      "#ef4444"),
    (201, 300, "Very Unhealthy",                 "#a855f7"),
    (301, 500, "Hazardous",                      "#be123c"),
]

def get_category(aqi):
    for lo, hi, label, color in AQI_CATS:
        if lo <= aqi <= hi:
            return label, color
    return "Hazardous", "#be123c"

def pm25_to_aqi(c):
    for lo_c, hi_c, lo_i, hi_i in [
        (0.0,12.0,0,50),(12.1,35.4,51,100),(35.5,55.4,101,150),
        (55.5,150.4,151,200),(150.5,250.4,201,300),(250.5,500.4,301,500),
    ]:
        if lo_c <= c <= hi_c:
            return round((hi_i-lo_i)/(hi_c-lo_c)*(c-lo_c)+lo_i)
    return 500


# ── Fetch live data ───────────────────────────────────
@st.cache_data(ttl=1800)
def fetch_live():
    if not OW_KEY:
        return None, None
    try:
        w = requests.get(f"https://api.openweathermap.org/data/2.5/weather",
            params={"lat":LAT,"lon":LON,"appid":OW_KEY,"units":"metric"}, timeout=10).json()
        p = requests.get(f"https://api.openweathermap.org/data/2.5/air_pollution",
            params={"lat":LAT,"lon":LON,"appid":OW_KEY}, timeout=10).json()
        comp = p["list"][0]["components"]
        pm25 = comp.get("pm2_5", 0)
        weather = {
            "temperature": round(w["main"]["temp"], 1),
            "humidity":    w["main"]["humidity"],
            "pressure":    w["main"]["pressure"],
            "wind_speed":  round(w["wind"]["speed"], 1),
            "visibility":  round(w.get("visibility", 10000)/1000, 1),
            "clouds":      w["clouds"]["all"],
        }
        pollution = {
            "aqi":  pm25_to_aqi(pm25),
            "pm25": round(pm25, 2),
            "pm10": round(comp.get("pm10", 0), 2),
            "no2":  round(comp.get("no2",  0), 2),
            "o3":   round(comp.get("o3",   0), 2),
            "so2":  round(comp.get("so2",  0), 2),
            "co":   round(comp.get("co",   0)/1000, 4),
            "nh3":  round(comp.get("nh3",  0), 2),
        }
        return weather, pollution
    except Exception as e:
        return None, None


@st.cache_data(ttl=1800)
def fetch_forecast():
    """5-day forecast from OpenWeather — extract AQI proxy from PM2.5."""
    if not OW_KEY:
        return []
    try:
        r = requests.get("https://api.openweathermap.org/data/2.5/air_pollution/forecast",
            params={"lat":LAT,"lon":LON,"appid":OW_KEY}, timeout=10).json()
        seen_days = {}
        for item in r.get("list", []):
            dt   = datetime.fromtimestamp(item["dt"], tz=timezone.utc)
            day  = dt.strftime("%Y-%m-%d")
            pm25 = item["components"].get("pm2_5", 0)
            aqi  = pm25_to_aqi(pm25)
            if day not in seen_days:
                seen_days[day] = {"date": dt.strftime("%b %d"), "weekday": dt.strftime("%A"),
                                  "aqi": aqi, "pm25": round(pm25,2)}
        days = list(seen_days.values())[1:4]  # next 3 days
        return days
    except:
        return []


@st.cache_data(ttl=3600)
def fetch_history_24h():
    """Last 24h of pollution data."""
    if not OW_KEY:
        return pd.DataFrame()
    try:
        now   = int(datetime.now(timezone.utc).timestamp())
        start = now - 86400
        r = requests.get("https://api.openweathermap.org/data/2.5/air_pollution/history",
            params={"lat":LAT,"lon":LON,"appid":OW_KEY,"start":start,"end":now}, timeout=10).json()
        records = []
        for item in r.get("list", []):
            dt   = datetime.fromtimestamp(item["dt"], tz=timezone.utc)
            pm25 = item["components"].get("pm2_5", 0)
            records.append({"time": dt, "aqi": pm25_to_aqi(pm25),
                            "pm25": round(pm25,2), "pm10": round(item["components"].get("pm10",0),2)})
        return pd.DataFrame(records)
    except:
        return pd.DataFrame()


# ══════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════

st.title("🌫️ Pearls AQI Predictor")
st.caption(f"Real-time Air Quality Index forecasting for {CITY} | 10Pearls Shine Program 2025")

if not OW_KEY:
    st.error("⚠️ OPENWEATHER_API_KEY not set. Add it in Streamlit Cloud secrets.")
    st.stop()

weather, pollution = fetch_live()

if not weather:
    st.error("Could not fetch live data. Check your API key.")
    st.stop()

aqi = pollution["aqi"]
cat_label, cat_color = get_category(aqi)

# ── Alert ─────────────────────────────────────────────
if aqi > 150:
    msg = {
        300: f"🚨 HAZARDOUS air (AQI={aqi}). Stay indoors. Avoid ALL outdoor activity.",
        200: f"⚠️ Very Unhealthy air (AQI={aqi}). Avoid outdoor activity. Close windows.",
        150: f"⚠️ Unhealthy air (AQI={aqi}). Reduce outdoor exertion. Wear N95 mask.",
    }
    for threshold, m in msg.items():
        if aqi > threshold:
            st.markdown(
                f'<div style="background:{cat_color}22;border-left:4px solid {cat_color};'
                f'padding:0.75rem 1rem;border-radius:8px;margin-bottom:1rem;">'
                f'<strong>{m}</strong></div>', unsafe_allow_html=True)
            break

# ── Current AQI Hero ──────────────────────────────────
col_aqi, col_metrics = st.columns([1, 3])
with col_aqi:
    st.markdown(
        f'<div style="background:{cat_color}22;border:2px solid {cat_color};'
        f'border-radius:16px;padding:1.5rem;text-align:center;">'
        f'<div style="font-size:13px;color:#888;margin-bottom:4px">Current AQI</div>'
        f'<div style="font-size:56px;font-weight:700;color:{cat_color}">{aqi}</div>'
        f'<div style="font-size:14px;color:{cat_color}">{cat_label}</div>'
        f'<div style="font-size:11px;color:#888;margin-top:4px">{CITY} • Live</div>'
        f'</div>', unsafe_allow_html=True)

with col_metrics:
    m1,m2,m3,m4,m5,m6 = st.columns(6)
    m1.metric("PM2.5 µg/m³", pollution["pm25"])
    m2.metric("PM10 µg/m³",  pollution["pm10"])
    m3.metric("Temp °C",     weather["temperature"])
    m4.metric("Humidity %",  weather["humidity"])
    m5.metric("Wind m/s",    weather["wind_speed"])
    m6.metric("NO₂ µg/m³",  pollution["no2"])

st.markdown("---")

# ── 3-Day Forecast ────────────────────────────────────
st.subheader("3-Day AQI Forecast")
forecast = fetch_forecast()

if forecast:
    cols = st.columns(3)
    for i, fc in enumerate(forecast):
        label, color = get_category(fc["aqi"])
        with cols[i]:
            st.markdown(
                f'<div style="background:{color}22;border:1px solid {color};'
                f'border-radius:12px;padding:1rem;text-align:center;">'
                f'<div style="font-size:13px;color:#888">{fc["weekday"]}, {fc["date"]}</div>'
                f'<div style="font-size:40px;font-weight:700;color:{color}">{fc["aqi"]}</div>'
                f'<div style="font-size:14px;color:{color}">{label}</div>'
                f'<div style="font-size:12px;color:#888;margin-top:4px">PM2.5: {fc["pm25"]} µg/m³</div>'
                f'</div>', unsafe_allow_html=True)

    # Forecast chart
    st.markdown("####")
    now_label = datetime.now(timezone.utc).strftime("%b %d")
    chart_data = [{"Day": f"Now ({now_label})", "AQI": aqi}] + \
                 [{"Day": f'{fc["weekday"][:3]} ({fc["date"]})', "AQI": fc["aqi"]} for fc in forecast]
    df_chart = pd.DataFrame(chart_data)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_chart["Day"], y=df_chart["AQI"],
        mode="lines+markers+text",
        text=df_chart["AQI"], textposition="top center",
        line=dict(color=cat_color, width=3),
        marker=dict(size=10, color=cat_color),
    ))
    fig.add_hrect(y0=0,   y1=50,  fillcolor="#22c55e", opacity=0.05, line_width=0)
    fig.add_hrect(y0=51,  y1=100, fillcolor="#eab308", opacity=0.05, line_width=0)
    fig.add_hrect(y0=101, y1=150, fillcolor="#f97316", opacity=0.05, line_width=0)
    fig.add_hrect(y0=151, y1=200, fillcolor="#ef4444", opacity=0.05, line_width=0)
    fig.update_layout(
        yaxis_title="AQI", xaxis_title="",
        margin=dict(l=10,r=10,t=20,b=10), height=280,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ── 24h History Chart ─────────────────────────────────
st.subheader("Last 24 Hours — AQI Trend")
df_hist = fetch_history_24h()

if not df_hist.empty:
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=df_hist["time"], y=df_hist["aqi"],
        mode="lines", name="AQI",
        line=dict(color=cat_color, width=2),
        fill="tozeroy", fillcolor=f"{cat_color}22",
    ))
    fig2.add_trace(go.Scatter(
        x=df_hist["time"], y=df_hist["pm25"],
        mode="lines", name="PM2.5",
        line=dict(color="#378ADD", width=1.5, dash="dot"),
    ))
    fig2.update_layout(
        yaxis_title="Value", xaxis_title="Time (UTC)",
        margin=dict(l=10,r=10,t=10,b=10), height=260,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("Historical data requires OpenWeather paid plan or will load shortly.")

st.markdown("---")

# ── Pollutant Breakdown ───────────────────────────────
st.subheader("🧪 Pollutant Breakdown")
col_p1, col_p2 = st.columns(2)

pollutants = {
    "PM2.5 (µg/m³)": pollution["pm25"],
    "PM10 (µg/m³)":  pollution["pm10"],
    "NO₂ (µg/m³)":   pollution["no2"],
    "O₃ (µg/m³)":    pollution["o3"],
    "SO₂ (µg/m³)":   pollution["so2"],
    "CO (mg/m³)":     pollution["co"],
    "NH₃ (µg/m³)":   pollution["nh3"],
}

with col_p1:
    fig3 = go.Figure(go.Bar(
        x=list(pollutants.values()),
        y=list(pollutants.keys()),
        orientation="h",
        marker_color=cat_color,
    ))
    fig3.update_layout(
        margin=dict(l=10,r=10,t=10,b=10), height=280,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Concentration"
    )
    st.plotly_chart(fig3, use_container_width=True)

with col_p2:
    poll_df = pd.DataFrame(list(pollutants.items()), columns=["Pollutant","Value"])
    st.dataframe(poll_df, use_container_width=True, hide_index=True)

st.markdown("---")

# ── Model Performance ─────────────────────────────────
st.subheader("ML Model Performance")
st.caption("5 models trained per horizon: Ridge, Random Forest, LightGBM, XGBoost, LSTM (TensorFlow)")

metrics_data = {
    "Horizon": ["24h", "48h", "72h"],
    "Best Model": ["XGBoost", "Random Forest", "LSTM"],
    "RMSE": [11.81, 11.81, 12.02],
    "MAE":  [9.65,  9.53,  9.74],
    "R²":   [0.035, 0.034, 0.003],
    "Acc±15 AQI": ["77.6%", "78.7%", "78.5%"],
}
st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)
st.caption("Models trained on 2,160 records. Best model per horizon selected automatically by lowest RMSE.")

# ── AQI Scale Reference ───────────────────────────────
st.markdown("---")
st.subheader("AQI Scale Reference")
scale_cols = st.columns(6)
for i, (lo, hi, label, color) in enumerate(AQI_CATS):
    with scale_cols[i]:
        st.markdown(
            f'<div style="background:{color}22;border:1px solid {color};'
            f'border-radius:8px;padding:0.5rem;text-align:center;">'
            f'<div style="font-size:11px;font-weight:600;color:{color}">{label}</div>'
            f'<div style="font-size:12px;color:#888">{lo}–{hi}</div>'
            f'</div>', unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────
st.markdown("---")
st.caption(
    f"Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} | "
    "Data: OpenWeather API | Feature Store: Hopsworks | "
    "Pipeline: GitHub Actions (hourly) | 10Pearls Shine Program 2025"
)