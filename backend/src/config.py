"""config.py — Central config for Pearls AQI Backend"""
import os
from dotenv import load_dotenv
load_dotenv()

# ── OpenWeather ───────────────────────────────────────
OW_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
OW_BASE    = "https://api.openweathermap.org"
CITY       = os.getenv("CITY", "karachi")
CITY_LAT   = float(os.getenv("CITY_LAT", "24.8607"))
CITY_LON   = float(os.getenv("CITY_LON",  "67.0011"))

# ── Hopsworks ─────────────────────────────────────────
HW_API_KEY  = os.getenv("HOPSWORKS_API_KEY", "")
HW_PROJECT  = os.getenv("HOPSWORKS_PROJECT", "pearls_aqi")
FG_NAME     = "aqi_features"          # Feature Group name
FG_VERSION  = 1
MR_NAME     = "aqi_model"             # Model Registry name

# ── Local fallback (when Hopsworks not configured) ────
LOCAL_DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "data")
LOCAL_MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# ── Horizons ──────────────────────────────────────────
HORIZONS = [24, 48, 72]

# ── 62 Feature columns ────────────────────────────────
FEATURE_COLS = [
    # Raw pollutants
    "aqi_raw","pm25","pm10","no2","o3","so2","co","nh3",
    # Weather
    "temperature","feels_like","humidity","pressure",
    "wind_speed","wind_deg","visibility","clouds",
    # Time features
    "hour","day_of_week","month","day_of_month",
    "is_weekend","is_morning","is_evening","is_night",
    "hour_sin","hour_cos","day_sin","day_cos",
    # Lag features
    "aqi_lag_1","aqi_lag_2","aqi_lag_3","aqi_lag_6",
    "aqi_lag_12","aqi_lag_24","aqi_lag_48",
    "pm25_lag_1","pm25_lag_6","pm25_lag_24",
    "pm10_lag_1","pm10_lag_24",
    "temp_lag_1","temp_lag_24",
    "humidity_lag_1","humidity_lag_24",
    # Rolling averages
    "aqi_rolling_mean_3","aqi_rolling_mean_6",
    "aqi_rolling_mean_12","aqi_rolling_mean_24",
    "aqi_rolling_std_6","aqi_rolling_std_24",
    "pm25_rolling_mean_6","pm25_rolling_mean_24",
    "pm10_rolling_mean_24","temp_rolling_mean_24",
    "humidity_rolling_mean_24",
    # Derived
    "aqi_change_1h","aqi_change_6h","aqi_change_24h",
    "pm25_pm10_ratio","wind_humidity_interaction",
    "heat_index","aqi_trend_slope",
]

# ── AQI Categories ────────────────────────────────────
AQI_CATS = [
    (0,   50,  "Good",                           "#22c55e"),
    (51,  100, "Moderate",                       "#eab308"),
    (101, 150, "Unhealthy for Sensitive Groups", "#f97316"),
    (151, 200, "Unhealthy",                      "#ef4444"),
    (201, 300, "Very Unhealthy",                 "#a855f7"),
    (301, 500, "Hazardous",                      "#be123c"),
]

def get_aqi_category(aqi: int) -> dict:
    for lo, hi, label, color in AQI_CATS:
        if lo <= aqi <= hi:
            return {"label": label, "color": color}
    return {"label": "Hazardous", "color": "#be123c"}

SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "dev-secret")
PORT       = int(os.getenv("PORT", 5000))
DEBUG      = os.getenv("FLASK_ENV", "production") == "development"
