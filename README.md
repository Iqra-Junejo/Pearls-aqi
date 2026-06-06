# 🌫️ Pearls AQI Predictor
### 10Pearls Data Science Internship Project — 2025

> Predict Air Quality Index (AQI) for Karachi 3 days ahead using a 100% serverless ML stack.

---

<div align="center">

**👩‍💻 Developed by Iqra Junejo**
**Data Science Intern — 10Pearls**
Karachi, Pakistan · 2025

</div>

---

## 📐 Architecture (from project PDF)

```
Weather & Pollution API (OpenWeather)
        │
        ▼  (runs every hour via GitHub Actions)
┌──────────────────┐
│ Feature Pipeline │  → engineers 62 features (time, lag, rolling, derived)
└──────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  Feature Store + Model      │  ← Hopsworks (free tier)
│  Registry (Hopsworks)       │
└─────────────────────────────┘
        │                │
        ▼ (every day)    ▼ (on predict)
┌──────────────────┐  ┌──────────────────────┐
│ Training Pipeline│  │  Inference Pipeline  │
│ 4 models × 3     │  │  loads best model    │
│ horizons = 12    │  │  → 3-day forecast    │
└──────────────────┘  └──────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │   Web Dashboard  │
                    │  Flask + Node.js │
                    │  SHAP · Alerts   │
                    └──────────────────┘
```

---

## 📁 Project Structure

```
Pearls-aqi/
├── backend/
│   ├── app.py                        ← Flask REST API
│   ├── requirements.txt
│   ├── .env.example
│   ├── src/
│   │   ├── config.py                 ← All env vars & constants
│   │   ├── feature_pipeline.py       ← Step 1: Fetch → Engineer → Hopsworks
│   │   ├── backfill.py               ← Step 2: Historical data generation
│   │   ├── training_pipeline.py      ← Step 3: Train 4 models × 3 horizons
│   │   └── inference_pipeline.py     ← Step 4: Load model → Predict → SHAP
│   ├── notebooks/
│   │   └── eda.ipynb                 ← EDA: trends, correlations, SHAP plots
│   └── .github/workflows/
│       ├── feature_pipeline.yml      ← CI/CD: runs EVERY HOUR automatically
│       └── training_pipeline.yml     ← CI/CD: runs DAILY at 2AM PKT
│
└── frontend/
    ├── server.js                     ← Bun/Node Express server
    ├── bunfig.toml                   ← Bun config
    ├── package.json
    ├── .env.example
    ├── views/index.html              ← Dashboard UI
    └── public/
        ├── css/main.css
        └── js/
            ├── app.js
            └── particles.js
```

---

## 🔑 API Keys Needed (both FREE)

| Service | URL | What it gives you |
|---|---|---|
| OpenWeather | https://openweathermap.org/api | Real-time AQI + weather |
| Hopsworks | https://www.hopsworks.ai | Feature Store + Model Registry |

---

## ⚙️ Setup & Run

### Backend
```bash
cd backend
python -m venv venv

# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
copy .env.example .env        # Fill your API keys

# Step 1: Backfill 90 days of historical data (run ONCE)
python src/backfill.py

# Step 2: Train all 12 models
python src/training_pipeline.py

# Step 3: Start Flask API
python app.py                 # → http://localhost:5000
```

### Frontend (using Bun — faster & safer than npm)
```bash
cd frontend

# Install Bun (one time only):
# Windows PowerShell (run as admin):
powershell -c "irm bun.sh/install.ps1 | iex"
# Mac/Linux:
curl -fsSL https://bun.sh/install | bash

# Install deps & run:
bun install
bun run dev                   # → http://localhost:3000
```

---

## 🐙 Git Setup Commands

### First time — create repo and push everything:
```bash
# 1. Go to github.com → New Repository → name it "Pearls-aqi" → Create

# 2. In your project root folder (Pearls-aqi/):
git init
git add .
git commit -m "feat: initial commit — Pearls AQI Predictor by Iqra Junejo"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/Pearls-aqi.git
git push -u origin main
```

### Add GitHub Secrets (for CI/CD to work):
```
GitHub repo → Settings → Secrets and variables → Actions → New repository secret

Add these 3 secrets:
  OPENWEATHER_API_KEY   → your OpenWeather key
  HOPSWORKS_API_KEY     → your Hopsworks API key
  HOPSWORKS_PROJECT     → your Hopsworks project name
```

### Daily workflow commands:
```bash
git add .
git commit -m "update: description of what you changed"
git push
```

### Check CI/CD status:
```
GitHub repo → Actions tab → you'll see hourly + daily runs
```

---
## Model Performance

Models trained on 2,160 records (90-day backfill + live hourly data).
5 models evaluated per horizon: Ridge, Random Forest, LightGBM, XGBoost, LSTM (TensorFlow).

| Model | Horizon | RMSE | MAE | R² | Acc±15 |
|-------|---------|------|-----|----|--------|
| XGBoost ⭐ | 24h | 11.81 | 9.65 | 0.035 | 77.6% |
| Random Forest ⭐ | 48h | 11.81 | 9.53 | 0.034 | 78.7% |
| LSTM ⭐ | 72h | 12.02 | 9.74 | 0.003 | 78.5% |

> Best model per horizon selected automatically by RMSE.
> R² is low due to synthetic backfill baseline — accuracy improves as live hourly data accumulates.

## 🤖 What Runs Automatically (CI/CD)

```
Every Hour — even when you're offline or sleeping:
  GitHub Actions → OpenWeather API → Karachi live AQI + weather
  → Engineer 62 features → Save to Hopsworks Feature Store ✓

Every Day at 2AM PKT:
  GitHub Actions → Load features from Hopsworks
  → Train LightGBM + XGBoost + Random Forest + Ridge
  → For 24h / 48h / 72h = 12 models total
  → Pick best per horizon → Save to Hopsworks Model Registry ✓
```

---

## 📊 Deliverables (per PDF requirements)

- ✅ End-to-end AQI prediction system
- ✅ Scalable automated pipeline (GitHub Actions)
- ✅ Interactive dashboard (real-time + 3-day forecast)
- ✅ SHAP feature importance explanations
- ✅ Hazardous AQI alerts
- ✅ EDA notebook (trends, correlations)
- ✅ Multiple models (RF, Ridge, LightGBM, XGBoost)
- ✅ Feature Store (Hopsworks)
- ✅ Model Registry (Hopsworks)

---
## EDA Notebook
Full exploratory data analysis available in `backend/notebooks/eda.ipynb`
<div align="center">

**Pearls AQI · 10Pearls Data Science Internship · 2026 Cohort-8**
Built by **Iqra Junejo** · Karachi, Pakistan

</div>
