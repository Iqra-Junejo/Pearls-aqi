"""
training_pipeline.py  ←  Step 3 from PDF
═══════════════════════════════════════════════════════
1. Fetches historical (features, targets) from Feature Store
2. Trains + evaluates: Random Forest, Ridge, LightGBM, XGBoost
   (Statistical → Deep Learning per PDF guidelines)
3. Stores the best model in Hopsworks Model Registry

Runs automatically every day via GitHub Actions CI/CD.
"""

import os, sys, json, logging
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    HW_API_KEY, HW_PROJECT, FG_NAME, FG_VERSION, MR_NAME,
    LOCAL_DATA_DIR, LOCAL_MODELS_DIR, HORIZONS, FEATURE_COLS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)
os.makedirs(LOCAL_MODELS_DIR, exist_ok=True)


# ═══════════════════════════════════════
#  1. FETCH FEATURES FROM FEATURE STORE
# ═══════════════════════════════════════

def load_features() -> pd.DataFrame:
    """Load features from Hopsworks Feature Store (or local CSV fallback)."""

    # Try Hopsworks first
    if HW_API_KEY:
        try:
            import hopsworks
            proj = hopsworks.login(api_key_value=HW_API_KEY, project=HW_PROJECT)
            fs   = proj.get_feature_store()
            fg   = fs.get_feature_group(FG_NAME, version=FG_VERSION)
            df   = fg.read()
            log.info(f"Loaded {len(df)} records from Hopsworks Feature Store.")
            return df
        except Exception as e:
            log.warning(f"Hopsworks load failed: {e} — using local CSV.")

    # Local fallback
    csv_path = os.path.join(LOCAL_DATA_DIR, "features.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            "No feature data found.\n"
            "Run first: python src/backfill.py"
        )
    df = pd.read_csv(csv_path)
    log.info(f"Loaded {len(df)} records from local CSV.")
    return df


# ═══════════════════════════════════════
#  EVALUATION
# ═══════════════════════════════════════

def evaluate(name, horizon, y_true, y_pred) -> dict:
    rmse  = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae   = float(mean_absolute_error(y_true, y_pred))
    r2    = float(r2_score(y_true, y_pred))
    acc15 = float(np.mean(np.abs(y_true - y_pred) <= 15) * 100)
    log.info(f"  [{name:14s}] RMSE={rmse:.2f}  MAE={mae:.2f}  R²={r2:.3f}  Acc±15={acc15:.1f}%")
    return {"model": name, "horizon": horizon, "rmse": rmse,
            "mae": mae, "r2": r2, "accuracy_15": acc15,
            "trained_at": datetime.utcnow().isoformat()}


# ═══════════════════════════════════════
#  2. TRAIN MULTIPLE MODELS
# ═══════════════════════════════════════

def train_all(Xtr, Xte, ytr, yte, horizon):
    """Train all 4 models, return (name, model, metrics) list."""
    results = []

    # Statistical model 1 — Ridge Regression
    try:
        m = RidgeCV(alphas=[0.1, 1, 10, 100], cv=5)
        m.fit(Xtr, ytr)
        results.append(("ridge", m, evaluate("Ridge", horizon, yte, m.predict(Xte))))
    except Exception as e:
        log.error(f"Ridge failed: {e}")

    # Statistical model 2 — Random Forest
    try:
        m = RandomForestRegressor(n_estimators=200, max_depth=12,
            min_samples_leaf=5, n_jobs=-1, random_state=42)
        m.fit(Xtr, ytr)
        results.append(("rf", m, evaluate("RandomForest", horizon, yte, m.predict(Xte))))
    except Exception as e:
        log.error(f"RF failed: {e}")

    # Advanced model 1 — LightGBM
    try:
        import lightgbm as lgb
        m = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05,
            num_leaves=63, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, random_state=42, verbose=-1)
        m.fit(Xtr, ytr, eval_set=[(Xte, yte)],
              callbacks=[lgb.early_stopping(50, verbose=False)])
        results.append(("lgbm", m, evaluate("LightGBM", horizon, yte, m.predict(Xte))))
    except Exception as e:
        log.error(f"LightGBM failed: {e}")

    # Advanced model 2 — XGBoost
    try:
        import xgboost as xgb
        m = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
            subsample=0.8, early_stopping_rounds=50, random_state=42, verbosity=0)
        m.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)
        results.append(("xgb", m, evaluate("XGBoost", horizon, yte, m.predict(Xte))))
    except Exception as e:
        log.error(f"XGBoost failed: {e}")

    return results


# ═══════════════════════════════════════
#  3. SAVE TO MODEL REGISTRY (Hopsworks)
# ═══════════════════════════════════════

def save_to_registry(bundle: dict, metrics: dict, horizon: int):
    """Save best model to Hopsworks Model Registry."""
    if not HW_API_KEY:
        log.info("Hopsworks not configured — saving locally only.")
        return

    try:
        import hopsworks
        proj = hopsworks.login(api_key_value=HW_API_KEY, project=HW_PROJECT)
        mr   = proj.get_model_registry()

        # Save model to temp file first
        tmp  = os.path.join(LOCAL_MODELS_DIR, f"tmp_{horizon}h.pkl")
        joblib.dump(bundle, tmp)

        # Register in Hopsworks
        model = mr.python.create_model(
            name=f"{MR_NAME}_{horizon}h",
            metrics=metrics,
            description=f"Pearls AQI {horizon}h forecast — best: {bundle['model_name']}",
            input_example={"features": bundle["features"][:5]},
            model_schema=None,
        )
        model.save(tmp)
        os.remove(tmp)
        log.info(f"Model {MR_NAME}_{horizon}h saved to Hopsworks Model Registry ✓")

    except Exception as e:
        log.error(f"Hopsworks model registry failed: {e}")


# ═══════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════

def run():
    df = load_features()

    if len(df) < 50:
        log.warning(f"Only {len(df)} records. Run backfill first for better results.")

    all_results = {}

    for h in HORIZONS:
        log.info(f"\n{'='*50}\nTraining for {h}h horizon\n{'='*50}")

        target = f"target_aqi_{h}h"
        df_h   = df.dropna(subset=[target]).copy()
        feats  = [f for f in FEATURE_COLS if f in df_h.columns]
        df_h[feats] = df_h[feats].fillna(df_h[feats].median())

        X, y  = df_h[feats].values, df_h[target].values.astype(float)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, shuffle=False)
        sc    = StandardScaler()
        Xtr   = sc.fit_transform(Xtr)
        Xte   = sc.transform(Xte)

        log.info(f"Train={len(Xtr)}  Test={len(Xte)}  Features={len(feats)}")

        results = train_all(Xtr, Xte, ytr, yte, h)
        if not results:
            log.error(f"All models failed for {h}h!")
            continue

        # Pick best by RMSE
        best_name, best_model, best_metrics = min(results, key=lambda x: x[2]["rmse"])
        log.info(f"\n🏆 Best {h}h: {best_name.upper()} (RMSE={best_metrics['rmse']:.2f})")

        bundle = {
            "model":       best_model,
            "scaler":      sc,
            "model_name":  best_name,
            "horizon":     h,
            "features":    feats,
            "metrics":     best_metrics,
            "all_results": {n: m for n, _, m in results},
        }

        # Save locally
        local_path = os.path.join(LOCAL_MODELS_DIR, f"model_{h}h.pkl")
        joblib.dump(bundle, local_path)
        log.info(f"Saved → {local_path}")

        # Save to Hopsworks Model Registry
        save_to_registry(bundle, best_metrics, h)

        all_results[f"{h}h"] = {
            "best":    best_name,
            "metrics": best_metrics,
            "all":     {n: m for n, _, m in results},
        }

    # Save summary
    summary_path = os.path.join(LOCAL_MODELS_DIR, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"\nSummary saved → {summary_path}")
    log.info("\n✅ Training pipeline complete!")


if __name__ == "__main__":
    run()
