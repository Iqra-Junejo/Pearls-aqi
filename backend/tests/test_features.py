"""
tests/test_features.py
Unit tests for feature engineering and AQI conversion.

Run: cd backend && python -m pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from feature_pipeline import _pm25_to_us_aqi, engineer_features
from config import get_aqi_category


def test_pm25_to_aqi_good():
    """PM2.5 of 5 µg/m³ should map to AQI in Good range (0-50)."""
    result = _pm25_to_us_aqi(5.0)
    assert 0 <= result <= 50, f"Expected Good range (0-50), got {result}"


def test_pm25_to_aqi_moderate():
    """PM2.5 of 20 µg/m³ should map to Moderate range (51-100)."""
    result = _pm25_to_us_aqi(20.0)
    assert 51 <= result <= 100, f"Expected Moderate range (51-100), got {result}"


def test_pm25_to_aqi_unhealthy():
    """PM2.5 of 40 µg/m³ should map to Unhealthy for Sensitive Groups (101-150)."""
    result = _pm25_to_us_aqi(40.0)
    assert result >= 101, f"Expected >= 101, got {result}"


def test_pm25_to_aqi_caps_at_500():
    """Very high PM2.5 should cap at AQI 500."""
    result = _pm25_to_us_aqi(600)
    assert result == 500


def test_aqi_category_good():
    cat = get_aqi_category(25)
    assert cat["label"] == "Good"
    assert cat["color"] == "#22c55e"


def test_aqi_category_moderate():
    cat = get_aqi_category(75)
    assert cat["label"] == "Moderate"


def test_aqi_category_unhealthy():
    cat = get_aqi_category(165)
    assert cat["label"] == "Unhealthy"


def test_aqi_category_hazardous():
    cat = get_aqi_category(350)
    assert cat["label"] == "Hazardous"


def test_feature_engineering_time_features():
    """engineer_features should produce cyclical time encoding."""
    raw = _sample_raw()
    features = engineer_features(raw, history=[])
    assert "hour_sin" in features
    assert "hour_cos" in features
    assert "day_sin" in features
    assert -1.0 <= features["hour_sin"] <= 1.0


def test_feature_engineering_derived():
    """engineer_features should produce AQI change rate features."""
    raw = _sample_raw()
    features = engineer_features(raw, history=[])
    assert "aqi_change_1h" in features
    assert "aqi_change_6h" in features
    assert "aqi_change_24h" in features
    assert "pm25_pm10_ratio" in features
    assert "aqi_trend_slope" in features


def test_feature_engineering_targets_are_none():
    """Targets should be None when just collected — filled later by backfill."""
    raw = _sample_raw()
    features = engineer_features(raw, history=[])
    assert features["target_aqi_24h"] is None
    assert features["target_aqi_48h"] is None
    assert features["target_aqi_72h"] is None


def test_feature_engineering_with_history():
    """Lag features should use history when available."""
    raw = _sample_raw()
    history = [dict(raw, aqi=90), dict(raw, aqi=85)]  # 2 past records
    features = engineer_features(raw, history=history)
    assert features["aqi_lag_1"] == 90
    assert features["aqi_lag_2"] == 85

    
def test_aqi_category_very_unhealthy():
    cat = get_aqi_category(250)
    assert cat["label"] == "Very Unhealthy"

# ── Helpers ───────────────────────────────────────────

def _sample_raw():
    return {
        "aqi": 100, "pm25": 35.0, "pm10": 60.0, "no2": 15.0,
        "o3": 8.0, "so2": 3.0, "co": 0.6, "nh3": 2.0,
        "aqi_raw": 3, "temperature": 33.0, "feels_like": 36.0,
        "humidity": 65, "pressure": 1008, "wind_speed": 2.5,
        "wind_deg": 200, "visibility": 7.0, "clouds": 40,
    }
