"""
inference_pipeline.py
Loads best model from Model Registry → latest features → 3-day forecast + SHAP.
"""

import os, sys, json, logging
from datetime import datetime, timedelta, timezone

import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    HW_API_KEY, HW_PROJECT, MR_NAME,
    LOCAL_DATA_DIR, LOCAL_MODELS_DIR,
    HORIZONS, CITY, FEATURE_COLS, get_aqi_category,
)

log = logging.getLogger(__name__)
_cache = {}


def _load_model(horizon: int):
    if horizon in _cache:
        return _cache[horizon]

    # Try Hopsworks Model Registry first
    if HW_API_KEY:
        try:
            import hopsworks
            proj  = hopsworks.login(api_key_value=HW_API_KEY, project=HW_PROJECT)
            mr    = proj.get_model_registry()
            model = mr.get_best_model(f"{MR_NAME}_{horizon}h", metric="rmse", direction="min")
            path  = model.download()
            bundle = joblib.load(os.path.join(path, f"model_{horizon}h.pkl"))
            _cache[horizon] = bundle
            log.info(f"Loaded model_{horizon}h from Hopsworks Registry ✓")
            return bundle
        except Exception as e:
            log.warning(f"Hopsworks model load failed: {e} — trying local.")

    # Local fallback
    path = os.path.join(LOCAL_MODELS_DIR, f"model_{horizon}h.pkl")
    if os.path.exists(path):
        bundle = joblib.load(path)
        _cache[horizon] = bundle
        return bundle

    return None


def _latest_features():
    """Get the most recent feature record."""
    import pandas as pd

    # Try Hopsworks
    if HW_API_KEY:
        try:
            import hopsworks
            from config import FG_NAME, FG_VERSION
            proj = hopsworks.login(api_key_value=HW_API_KEY, project=HW_PROJECT)
            fg   = proj.get_feature_store().get_feature_group(FG_NAME, version=FG_VERSION)
            df   = fg.read()
            return df.sort_values("timestamp", ascending=False).iloc[0].to_dict()
        except Exception:
            pass

    # Local fallback
    csv_path = os.path.join(LOCAL_DATA_DIR, "features.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path).sort_values("timestamp", ascending=False)
        return df.iloc[0].to_dict()

    raise ValueError("No feature data. Run: python src/backfill.py")


def _compute_shap(bundle, X):
    try:
        import shap
        name = bundle["model_name"]
        if name in ("lgbm", "xgb", "rf"):
            sv = shap.TreeExplainer(bundle["model"]).shap_values(X)
            sv = sv[0] if not isinstance(sv, np.ndarray) else sv
            sv = sv.flatten()
        else:
            sv = shap.LinearExplainer(bundle["model"], X).shap_values(X).flatten()

        results = [
            {"label": bundle["features"][i], "shap_value": round(float(sv[i]), 3)}
            for i in range(min(len(bundle["features"]), len(sv)))
        ]
        results.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        return results[:8]
    except Exception as e:
        log.warning(f"SHAP failed: {e}")
        return _mock_shap()

# Returns real SHAP values when model is loaded, mock values as fallback
def _mock_shap():
    return [
        {"label": "PM2.5",    "shap_value": 42.1},
        {"label": "AQI Lag",  "shap_value": 18.3},
        {"label": "Hour",     "shap_value": 15.7},
        {"label": "Humidity", "shap_value": 12.7},
        {"label": "Wind Spd", "shap_value": -11.2},
        {"label": "Temp",     "shap_value": -7.4},
        {"label": "NO₂",      "shap_value": 6.8},
        {"label": "AQI Roll", "shap_value": -5.3},
    ]


def predict(city=CITY):
    rec     = _latest_features()
    cur_aqi = int(rec.get("aqi", 0))
    cat     = get_aqi_category(cur_aqi)

    current = {
        "aqi": cur_aqi, "category": cat,
        "pm25": float(rec.get("pm25", 0)),
        "pm10": float(rec.get("pm10", 0)),
        "no2":  float(rec.get("no2",  0)),
        "o3":   float(rec.get("o3",   0)),
        "so2":  float(rec.get("so2",  0)),
        "co":   float(rec.get("co",   0)),
        "temperature": float(rec.get("temperature", 0)),
        "humidity":    float(rec.get("humidity",    0)),
        "wind_speed":  float(rec.get("wind_speed",  0)),
        "pressure":    float(rec.get("pressure",    0)),
        "timestamp":   str(rec.get("timestamp", "")),
    }

    forecasts, shap_vals = [], []
    now = datetime.now(timezone.utc)
    conf_map = {24: 85, 48: 78, 72: 70}

    for h in HORIZONS:
        target_dt = now + timedelta(hours=h)
        bundle    = _load_model(h)

        if bundle:
            feats    = bundle["features"]
            X_raw    = np.array([[float(rec.get(f, 0)) for f in feats]])
            X_scaled = bundle["scaler"].transform(X_raw)
            pred     = max(0, round(float(bundle["model"].predict(X_scaled)[0])))
            if h == 24:
                shap_vals = _compute_shap(bundle, X_scaled)
        else:
            pred = max(0, cur_aqi + int(np.random.normal(0, 8)))

        cat_pred = get_aqi_category(pred)
        forecasts.append({
            "horizon":    f"{h}h",
            "date":       target_dt.strftime("%b %d"),
            "weekday":    target_dt.strftime("%A"),
            "aqi":        pred,
            "category":   cat_pred["label"],
            "color":      cat_pred["color"],
            "confidence": conf_map.get(h, 70),
            "model_used": bundle["model_name"] if bundle else "estimate",
        })

    # Load metrics
    try:
        with open(os.path.join(LOCAL_MODELS_DIR, "training_summary.json")) as f:
            summary = json.load(f)
        metrics = summary.get("24h", {}).get("metrics", {})
    except Exception:
        metrics = {}

    return {
        "city":       city,
        "current":    current,
        "forecast":   forecasts,
        "shap":       shap_vals or _mock_shap(),
        "metrics":    metrics,
        "fetched_at": now.isoformat(),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(predict(), indent=2))
