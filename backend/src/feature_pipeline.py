"""
feature_pipeline.py  ←  Step 1 from PDF
═══════════════════════════════════════════════════════
1. Fetches raw weather + pollutant data from OpenWeather API
2. Computes 62 features (time-based, lag, rolling, derived)
3. Stores features in Hopsworks Feature Store

Runs automatically every hour via GitHub Actions CI/CD.
"""

import os, sys, math, logging
from datetime import datetime, timedelta, timezone

import requests
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    OW_API_KEY, OW_BASE, CITY, CITY_LAT, CITY_LON,
    HW_API_KEY, HW_PROJECT, FG_NAME, FG_VERSION,
    LOCAL_DATA_DIR, FEATURE_COLS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)
os.makedirs(LOCAL_DATA_DIR, exist_ok=True)


# ═══════════════════════════════════════
#  1. FETCH FROM OPENWEATHER API
# ═══════════════════════════════════════

def fetch_weather() -> dict:
    """Fetches current weather — temperature, humidity, wind, etc."""
    r = requests.get(f"{OW_BASE}/data/2.5/weather", params={
        "lat": CITY_LAT, "lon": CITY_LON,
        "appid": OW_API_KEY, "units": "metric"
    }, timeout=10)
    r.raise_for_status()
    d = r.json()
    return {
        "temperature": round(d["main"]["temp"], 2),
        "feels_like":  round(d["main"]["feels_like"], 2),
        "humidity":    d["main"]["humidity"],
        "pressure":    d["main"]["pressure"],
        "wind_speed":  round(d["wind"]["speed"], 2),
        "wind_deg":    d["wind"].get("deg", 0),
        "visibility":  round(d.get("visibility", 10000) / 1000, 2),
        "clouds":      d["clouds"]["all"],
    }


def fetch_pollution() -> dict:
    """Fetches AQI + all pollutants from OpenWeather Air Pollution API."""
    r = requests.get(f"{OW_BASE}/data/2.5/air_pollution", params={
        "lat": CITY_LAT, "lon": CITY_LON, "appid": OW_API_KEY
    }, timeout=10)
    r.raise_for_status()
    d    = r.json()
    comp = d["list"][0]["components"]
    pm25 = comp.get("pm2_5", 0)
    return {
        "aqi_raw": d["list"][0]["main"]["aqi"],
        "aqi":     _pm25_to_us_aqi(pm25),
        "pm25":    round(pm25, 2),
        "pm10":    round(comp.get("pm10",  0), 2),
        "no2":     round(comp.get("no2",   0), 2),
        "o3":      round(comp.get("o3",    0), 2),
        "so2":     round(comp.get("so2",   0), 2),
        "co":      round(comp.get("co",    0) / 1000, 4),
        "nh3":     round(comp.get("nh3",   0), 2),
    }


def _pm25_to_us_aqi(c: float) -> int:
    """Convert PM2.5 µg/m³ → US EPA AQI (0-500 scale)."""
    for lo_c, hi_c, lo_i, hi_i in [
        (0.0, 12.0, 0, 50), (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150), (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300), (250.5, 500.4, 301, 500),
    ]:
        if lo_c <= c <= hi_c:
            return round((hi_i - lo_i) / (hi_c - lo_c) * (c - lo_c) + lo_i)
    return 500


# ═══════════════════════════════════════
#  2. COMPUTE FEATURES (62 total)
# ═══════════════════════════════════════

def engineer_features(raw: dict, history: list) -> dict:
    """
    Computes all features from raw data + history.
    Includes: time-based, lag, rolling averages, derived features.
    """
    now = datetime.now(timezone.utc)
    f   = {**raw}

    # Time-based features (hour, day, month)
    f.update({
        "timestamp":    now.isoformat(),
        "city":         CITY,
        "hour":         now.hour,
        "day_of_week":  now.weekday(),
        "month":        now.month,
        "day_of_month": now.day,
        "is_weekend":   int(now.weekday() >= 5),
        "is_morning":   int(6  <= now.hour < 12),
        "is_evening":   int(17 <= now.hour < 21),
        "is_night":     int(now.hour >= 21 or now.hour < 6),
        # Cyclical encoding so model understands time is circular
        "hour_sin": round(math.sin(2*math.pi*now.hour/24), 6),
        "hour_cos": round(math.cos(2*math.pi*now.hour/24), 6),
        "day_sin":  round(math.sin(2*math.pi*now.weekday()/7), 6),
        "day_cos":  round(math.cos(2*math.pi*now.weekday()/7), 6),
    })

    def lag(key, n):
        vals = [h.get(key, raw.get(key, 0)) for h in history]
        return vals[n-1] if len(vals) >= n else raw.get(key, 0)

    def rmean(key, n):
        vals = [h.get(key, raw.get(key, 0)) for h in history[:n]]
        return round(float(np.mean(vals)) if vals else raw.get(key, 0), 4)

    def rstd(key, n):
        vals = [h.get(key, 0) for h in history[:n]]
        return round(float(np.std(vals)) if len(vals)>1 else 0, 4)

    # Lag features (AQI change rate — per PDF requirement)
    for n in [1,2,3,6,12,24,48]:
        f[f"aqi_lag_{n}"] = lag("aqi", n)
    for n in [1,6,24]:
        f[f"pm25_lag_{n}"] = lag("pm25", n)
    for n in [1,24]:
        f[f"pm10_lag_{n}"]     = lag("pm10", n)
        f[f"temp_lag_{n}"]     = lag("temperature", n)
        f[f"humidity_lag_{n}"] = lag("humidity", n)

    # Rolling averages
    for n in [3,6,12,24]:
        f[f"aqi_rolling_mean_{n}"] = rmean("aqi", n)
    f["aqi_rolling_std_6"]       = rstd("aqi", 6)
    f["aqi_rolling_std_24"]      = rstd("aqi", 24)
    f["pm25_rolling_mean_6"]     = rmean("pm25", 6)
    f["pm25_rolling_mean_24"]    = rmean("pm25", 24)
    f["pm10_rolling_mean_24"]    = rmean("pm10", 24)
    f["temp_rolling_mean_24"]    = rmean("temperature", 24)
    f["humidity_rolling_mean_24"]= rmean("humidity", 24)

    # Derived features (AQI change rate — per PDF requirement)
    f["aqi_change_1h"]  = raw["aqi"] - lag("aqi", 1)
    f["aqi_change_6h"]  = raw["aqi"] - lag("aqi", 6)
    f["aqi_change_24h"] = raw["aqi"] - lag("aqi", 24)
    f["pm25_pm10_ratio"]           = round(raw["pm25"] / max(raw["pm10"], 0.01), 4)
    f["wind_humidity_interaction"] = round(raw["wind_speed"] * raw["humidity"] / 100, 4)

    T, H = raw["temperature"], raw["humidity"]
    f["heat_index"] = round(T + 0.33*(H/100*6.105*np.exp(17.27*T/(237.7+T)))-4, 2)

    hist_aqi = [h.get("aqi", raw["aqi"]) for h in history[:6]]
    f["aqi_trend_slope"] = round(float(np.polyfit(range(len(hist_aqi)), hist_aqi, 1)[0]), 4) \
                           if len(hist_aqi) > 1 else 0.0

    # Target labels (filled by backfill / update_targets)
    f["target_aqi_24h"] = None
    f["target_aqi_48h"] = None
    f["target_aqi_72h"] = None

    return f


# ═══════════════════════════════════════
#  3. STORE IN HOPSWORKS FEATURE STORE
# ═══════════════════════════════════════

def get_feature_store():
    """Connect to Hopsworks Feature Store."""
    if not HW_API_KEY:
        log.warning("HOPSWORKS_API_KEY not set — using local CSV fallback.")
        return None
    try:
        import hopsworks
        proj = hopsworks.login(api_key_value=HW_API_KEY, project=HW_PROJECT)
        return proj.get_feature_store()
    except Exception as e:
        log.error(f"Hopsworks connection failed: {e}")
        return None


def save_features(record: dict, fs=None):
    """Save feature record to Hopsworks Feature Store (+ local CSV backup)."""
    import pandas as pd

    df = pd.DataFrame([record])

    # ── Hopsworks ──────────────────────────────────
    if fs is not None:
        try:
            fg = fs.get_or_create_feature_group(
                name=FG_NAME,
                version=FG_VERSION,
                primary_key=["timestamp", "city"],
                description="Hourly AQI features for Karachi — Pearls AQI",
                event_time="timestamp",
            )
            fg.insert(df, write_options={"wait_for_job": False})
            log.info(f"Saved to Hopsworks Feature Group '{FG_NAME}' ✓")
        except Exception as e:
            log.error(f"Hopsworks insert failed: {e}")

    # ── Local CSV backup ───────────────────────────
    csv_path = os.path.join(LOCAL_DATA_DIR, "features.csv")
    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        df = pd.concat([existing, df]).drop_duplicates(subset=["timestamp","city"])
    df.to_csv(csv_path, index=False)
    log.info(f"Saved to local CSV: {csv_path}")


def get_history_from_store(fs=None, hours=72) -> list:
    """Load recent records from Hopsworks (or local CSV) for lag features."""
    import pandas as pd

    if fs is not None:
        try:
            fg  = fs.get_feature_group(FG_NAME, version=FG_VERSION)
            df  = fg.read()
            df  = df.sort_values("timestamp", ascending=False).head(hours)
            return df.to_dict("records")
        except Exception as e:
            log.warning(f"Hopsworks read failed: {e}, using local CSV")

    csv_path = os.path.join(LOCAL_DATA_DIR, "features.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path).sort_values("timestamp", ascending=False).head(hours)
        return df.to_dict("records")
    return []


def update_targets(fs=None):
    """Back-fill target labels (what AQI was 24h/48h/72h later)."""
    import pandas as pd

    csv_path = os.path.join(LOCAL_DATA_DIR, "features.csv")
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path).sort_values("timestamp").reset_index(drop=True)
    for i, row in df.iterrows():
        for h in [24, 48, 72]:
            fi = i + h
            if fi < len(df):
                df.at[i, f"target_aqi_{h}h"] = df.at[fi, "aqi"]
    df.to_csv(csv_path, index=False)
    log.info("Target labels updated ✓")


# ═══════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════

def run():
    if not OW_API_KEY:
        log.error("OPENWEATHER_API_KEY not set in .env!")
        return None

    log.info(f"Feature pipeline starting — {CITY.upper()}...")

    # 1. Fetch raw data
    weather   = fetch_weather()
    pollution = fetch_pollution()
    raw       = {**weather, **pollution}
    log.info(f"Live: AQI={raw['aqi']}  PM2.5={raw['pm25']}µg/m³  Temp={raw['temperature']}°C")

    # 2. Compute features
    fs      = get_feature_store()
    history = get_history_from_store(fs)
    record  = engineer_features(raw, history)

    # 3. Store in Feature Store
    save_features(record, fs)
    update_targets(fs)

    log.info("Feature pipeline complete ✓")
    return record


if __name__ == "__main__":
    run()
