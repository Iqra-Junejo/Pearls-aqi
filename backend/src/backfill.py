"""
backfill.py  ←  Step 2 from PDF
═══════════════════════════════════════════════════════
"Run the feature script from step 1 for a range of past
dates, to generate training data for your ML models."

Generates 90 days of historical features → saves to
Hopsworks Feature Store for model training.

Run ONCE before training: python src/backfill.py
"""

import os, sys, math, logging
from datetime import datetime, timedelta, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from feature_pipeline import (
    fetch_weather, fetch_pollution, save_features,
    get_feature_store, update_targets,
)
from config import CITY, LOCAL_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)
os.makedirs(LOCAL_DATA_DIR, exist_ok=True)


def run_backfill(days: int = 90):
    """
    Fetches today's live data as baseline, then generates
    realistic synthetic history for the past N days.
    This is needed because OpenWeather free tier doesn't
    support historical API calls.
    """
    log.info(f"Starting backfill: {days} days × 24h = {days*24} records...")

    # Get real current data as baseline
    weather   = fetch_weather()
    pollution = fetch_pollution()
    base      = {**weather, **pollution}
    log.info(f"Baseline: AQI={base['aqi']}  PM2.5={base['pm25']}  Temp={base['temperature']}°C")

    fs  = get_feature_store()
    now = datetime.now(timezone.utc)

    import pandas as pd
    records = []

    for h in range(days * 24, 0, -1):
        ts   = now - timedelta(hours=h)
        hour = ts.hour

        # Realistic Karachi daily pattern:
        # Worse AQI in morning rush (7-9am) and evening (6-9pm)
        # Better at night and midday (sea breeze)
        rush_factor   = 1 + 0.18 * (
            np.exp(-((hour-8)**2)/8) +   # morning rush
            np.exp(-((hour-19)**2)/8)    # evening rush
        )
        # Winter months (Oct-Feb) worse in Karachi
        month    = ts.month
        seasonal = 1 + 0.12 * np.cos(2 * math.pi * (month - 7) / 12)

        noise_aqi  = np.random.normal(0, 12)
        noise_pm25 = np.random.normal(0, 6)
        noise_temp = np.random.normal(0, 2)

        rec = base.copy()
        rec["aqi"]         = max(10, int(base["aqi"] * rush_factor * seasonal + noise_aqi))
        rec["pm25"]        = max(1,  round(base["pm25"] + noise_pm25, 2))
        rec["pm10"]        = max(1,  round(base["pm10"] + noise_pm25 * 0.8, 2))
        rec["temperature"] = round(base["temperature"] + noise_temp, 1)
        rec["humidity"]    = min(100, max(10, base["humidity"] + int(np.random.normal(0, 5))))
        rec["wind_speed"]  = max(0,   round(base["wind_speed"] + np.random.normal(0, 1.5), 1))
        rec["timestamp"]   = ts.isoformat()
        rec["city"]        = CITY

        # Basic time features for CSV
        rec["hour"]        = ts.hour
        rec["day_of_week"] = ts.weekday()
        rec["month"]       = ts.month
        rec["day_of_month"]= ts.day

        records.append(rec)

    # Fill target labels (what AQI was 24/48/72h later)
    for i, rec in enumerate(records):
        for offset in [24, 48, 72]:
            fi = i + offset
            rec[f"target_aqi_{offset}h"] = records[fi]["aqi"] if fi < len(records) else None

    # Save to feature store
    df = pd.DataFrame(records)
    csv_path = os.path.join(LOCAL_DATA_DIR, "features.csv")
    df.to_csv(csv_path, index=False)
    log.info(f"Saved {len(df)} records to {csv_path}")

    # Push to Hopsworks if configured
    if fs is not None:
        try:
            from config import FG_NAME, FG_VERSION
            fg = fs.get_or_create_feature_group(
                name=FG_NAME, version=FG_VERSION,
                primary_key=["timestamp", "city"],
                description="Hourly AQI features for Karachi — Pearls AQI",
                event_time="timestamp",
            )
            fg.insert(df, write_options={"wait_for_job": False})
            log.info(f"Backfill pushed to Hopsworks Feature Group '{FG_NAME}' ✓")
        except Exception as e:
            log.error(f"Hopsworks push failed: {e}")

    log.info(f"Backfill complete: {len(records)} records ✓")


if __name__ == "__main__":
    run_backfill(days=90)
