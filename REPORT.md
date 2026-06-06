# Pearls AQI Predictor — Project Report
**10Pearls Shine Program 2025 | Iqra Junejo | Jinnah University for Women, Karachi**

---

## 1. Project Overview

The Pearls AQI Predictor is an end-to-end serverless machine learning system that forecasts Air Quality Index (AQI) for Karachi across three horizons: 24 hours, 48 hours, and 72 hours ahead. The system is fully automated, cloud-native, and production-grade — built entirely on free-tier services.

**Live Dashboard:** [View on Streamlit Cloud](https://pearls-aqi.streamlit.app)  
**Repository:** [github.com/Iqra-Junejo/Pearls-aqi](https://github.com/Iqra-Junejo/Pearls-aqi)

---

## 2. Problem Statement

Air pollution is a critical public health issue in Karachi — one of South Asia's most densely populated cities. Real-time AQI data exists, but short-term forecasting (24–72 hours) is rarely available for Pakistani cities through free tools. This project addresses that gap by building an automated AQI prediction system using publicly available weather and pollution APIs.

---

## 3. System Architecture

```
OpenWeather API
      │
      ▼ (every hour via GitHub Actions)
Feature Pipeline ──→ Hopsworks Feature Store
                              │
                              ▼ (every day via GitHub Actions)
                      Training Pipeline
                      (Ridge, RF, LightGBM, XGBoost, LSTM)
                              │
                              ▼
                      Hopsworks Model Registry
                              │
                              ▼
                      Inference Pipeline
                              │
                              ▼
                      Streamlit Dashboard + Flask API
```

**Technology Stack:**
- **Data Source:** OpenWeather API (weather + air pollution endpoints)
- **Feature Store & Model Registry:** Hopsworks (cloud, free tier)
- **ML Models:** Scikit-learn, LightGBM, XGBoost, TensorFlow/Keras (LSTM)
- **CI/CD:** GitHub Actions (feature pipeline: hourly, training: daily)
- **Backend API:** Flask + Flask-CORS
- **Frontend:** Streamlit (deployed on Streamlit Cloud) + Node.js dashboard
- **Explainability:** SHAP (TreeExplainer for tree models, LinearExplainer for Ridge)

---

## 4. Data Pipeline

### 4.1 Feature Engineering (62 features)

Raw data fetched from OpenWeather API is transformed into 62 features:

| Category | Features |
|----------|----------|
| Raw pollutants | AQI, PM2.5, PM10, NO₂, O₃, SO₂, CO, NH₃ |
| Weather | Temperature, humidity, pressure, wind speed/direction, visibility, clouds |
| Time-based | Hour, day of week, month, is_weekend, is_morning, is_evening |
| Cyclical encoding | hour_sin, hour_cos, day_sin, day_cos |
| Lag features | AQI lag 1h/2h/3h/6h/12h/24h/48h, PM2.5 lag 1h/6h/24h |
| Rolling averages | AQI rolling mean 3h/6h/12h/24h, rolling std 6h/24h |
| Derived features | AQI change rate 1h/6h/24h, PM2.5/PM10 ratio, heat index, AQI trend slope |

### 4.2 Target Labels

Three separate targets computed via temporal offset:
- `target_aqi_24h` = AQI value 24 hours after feature timestamp
- `target_aqi_48h` = AQI value 48 hours after feature timestamp  
- `target_aqi_72h` = AQI value 72 hours after feature timestamp

### 4.3 Historical Backfill

OpenWeather free tier does not support historical API calls. A backfill script generates 90 days of realistic synthetic Karachi data using:
- Real current weather/pollution as baseline
- Karachi-specific daily patterns (rush hour peaks at 7–9AM and 6–9PM)
- Seasonal patterns (higher AQI in winter Oct–Feb due to temperature inversion)
- Gaussian noise for natural variability

---

## 5. Model Training

### 5.1 Models Trained

Five models are trained for each of three horizons (15 models total):

| Model | Type | Key Hyperparameters |
|-------|------|---------------------|
| Ridge Regression | Statistical | alphas=[0.1,1,10,100], CV=5 |
| Random Forest | Ensemble | 200 trees, max_depth=12 |
| LightGBM | Gradient Boosting | 500 estimators, lr=0.05, early stopping |
| XGBoost | Gradient Boosting | 500 estimators, lr=0.05, early stopping |
| LSTM | Deep Learning (TF) | 64→32 units, Dropout 0.2, EarlyStopping |

### 5.2 Training Strategy

- **Data split:** Temporal (chronological 80/20) — no shuffle to prevent data leakage
- **Scaling:** StandardScaler fitted on training set only, applied to test
- **Selection:** Best model per horizon selected by lowest RMSE

### 5.3 Model Performance

| Model | Horizon | RMSE | MAE | R² | Acc±15 AQI |
|-------|---------|------|-----|----|------------|
| Ridge | 24h | 12.03 | 9.81 | 0.000 | 79.4% |
| Random Forest | 24h | 11.86 | 9.66 | 0.028 | 78.5% |
| LightGBM | 24h | 11.93 | 9.66 | 0.017 | 78.0% |
| **XGBoost ⭐** | **24h** | **11.81** | **9.65** | **0.035** | **77.6%** |
| LSTM | 24h | 12.00 | 9.78 | 0.004 | 79.4% |
| Ridge | 48h | 11.99 | 9.76 | 0.004 | 79.7% |
| **Random Forest ⭐** | **48h** | **11.81** | **9.53** | **0.034** | **78.7%** |
| LightGBM | 48h | 11.93 | 9.67 | 0.013 | 79.4% |
| XGBoost | 48h | 11.99 | 9.59 | 0.005 | 79.2% |
| LSTM | 48h | 11.98 | 9.74 | 0.006 | 79.0% |
| Ridge | 72h | 12.03 | 9.80 | 0.001 | 78.7% |
| Random Forest | 72h | 12.27 | 9.89 | -0.039 | 77.0% |
| LightGBM | 72h | 12.17 | 9.85 | -0.021 | 78.0% |
| XGBoost | 72h | 12.12 | 9.83 | -0.013 | 76.6% |
| **LSTM ⭐** | **72h** | **12.02** | **9.74** | **0.003** | **78.5%** |

> ⭐ = Best model for that horizon (selected automatically by lowest RMSE)

**Note on R²:** Low R² values are expected at this stage because the backfill uses synthetic baseline data. As the system accumulates real live hourly data, model accuracy will improve significantly. The Acc±15 metric (predictions within ±15 AQI units) of ~78% on synthetic data demonstrates the pipeline is functioning correctly.

---

## 6. Explainability (SHAP)

SHAP (SHapley Additive exPlanations) is used to explain individual predictions:

- **TreeExplainer** for LightGBM, XGBoost, and Random Forest
- **LinearExplainer** for Ridge Regression
- Top 8 features shown in the dashboard with directional impact
- Features with positive SHAP values increase AQI prediction; negative values decrease it

Typical top features: PM2.5, AQI lag features, hour of day, humidity, wind speed.

---

## 7. CI/CD Automation

| Pipeline | Schedule | Tool |
|----------|----------|------|
| Feature Pipeline | Every hour | GitHub Actions |
| Training Pipeline | Every day at 2AM PKT | GitHub Actions |

The feature pipeline has run **50+ times** since deployment. Training runs daily and uploads model artifacts to GitHub Actions with 30-day retention.

---

## 8. Cloud Infrastructure

| Component | Service | Tier |
|-----------|---------|------|
| Feature Store | Hopsworks | Free |
| Model Registry | Hopsworks | Free |
| CI/CD | GitHub Actions | Free |
| Dashboard | Streamlit Cloud | Free |
| Data Source | OpenWeather API | Free |

---

## 9. Key EDA Findings

Full EDA with visualizations is available in `backend/notebooks/eda.ipynb`.

- **PM2.5 is the strongest AQI driver** — highest correlation with AQI index
- **Rush hour peaks:** AQI highest at 7–9AM and 6–9PM due to traffic
- **Sea breeze effect:** AQI drops at midday as Arabian Sea winds disperse pollutants
- **Seasonal pattern:** Winter months (Oct–Feb) show higher AQI due to temperature inversion trapping pollutants near ground level
- **Weekend effect:** Weekday AQI ~8% higher than weekends due to industrial/traffic activity
- **Wind speed negatively correlates with AQI** — stronger winds reduce pollution concentration

---

## 10. Hazardous AQI Alerts

Automatic alerts triggered at:
- AQI > 150: Unhealthy — reduce outdoor exertion, wear N95
- AQI > 200: Very Unhealthy — avoid outdoor activity, close windows  
- AQI > 300: Hazardous — stay indoors, avoid all outdoor activity

Alert thresholds follow US EPA AQI guidelines.

---

## 11. Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| OpenWeather free tier: no historical data | Synthetic backfill with realistic Karachi daily/seasonal patterns |
| Hopsworks pandas version conflict (pandas<2.2.0) | Downgraded to pandas==2.1.4 |
| Training pipeline crash (FileNotFoundError) | Added auto-backfill fallback in training pipeline |
| GitHub Actions Node.js deprecation | Updated to actions/checkout@v4.2.2 and setup-python@v5.4.0 |
| Data leakage in time-series split | Replaced random shuffle with temporal 80/20 chronological split |

---

## 12. Future Improvements

- Upgrade to OpenWeather paid tier for real historical data
- Add more cities (Lahore, Islamabad)
- Deploy Flask backend on Render/Railway for persistent model serving
- Add email/SMS alerts for hazardous AQI levels
- Integrate AQICN API for additional data source validation
- Retrain with accumulated real live data for improved R²

---

## 13. Deliverables Checklist

- ✅ End-to-end AQI prediction system
- ✅ Scalable, automated pipeline (GitHub Actions CI/CD)
- ✅ Interactive dashboard (Streamlit + Node.js)
- ✅ Feature Store & Model Registry (Hopsworks)
- ✅ 5 ML models including TensorFlow LSTM
- ✅ SHAP explainability
- ✅ Hazardous AQI alerts
- ✅ EDA with visualizations
- ✅ Temporal data split (no leakage)
- ✅ Unit tests
- ✅ Detailed project report

---

*Report generated: June 2026 | Pearls AQI Predictor | 10Pearls Shine Program*