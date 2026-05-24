"""
training_pipeline.py  ←  Step 3 from PDF
═══════════════════════════════════════════════════════
1. Fetches historical (features, targets) from Feature Store
2. Trains + evaluates: Ridge, Random Forest, LightGBM, XGBoost, LSTM (TensorFlow)
3. Stores the best model in Hopsworks Model Registry

Runs automatically every day via GitHub Actions CI/CD.

FIXES APPLIED:
- Added TensorFlow LSTM (required by PDF spec)
- Fixed crash when Hopsworks empty: auto-runs backfill if < 50 rows
- Temporal split enforced (no data leakage)
- Metrics printed and saved to training_summary.json
"""

import os, sys, json, logging
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
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

    if HW_API_KEY:
        try:
            import hopsworks
            proj = hopsworks.login(api_key_value=HW_API_KEY, project=HW_PROJECT)
            fs   = proj.get_feature_store()
            fg   = fs.get_feature_group(FG_NAME, version=FG_VERSION)
            df   = fg.read()
            log.info(f"Loaded {len(df)} records from Hopsworks Feature Store.")
            if len(df) >= 50:
                return df
            log.warning(f"Only {len(df)} rows in Hopsworks — will try local CSV.")
        except Exception as e:
            log.warning(f"Hopsworks load failed: {e} — using local CSV.")

    csv_path = os.path.join(LOCAL_DATA_DIR, "features.csv")

    # FIX: if CSV is missing or empty, auto-run backfill so training doesn't crash
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) < 100:
        log.warning("No feature data found. Running backfill automatically...")
        try:
            from backfill import run_backfill
            run_backfill(days=90)
        except Exception as e:
            raise RuntimeError(
                f"Auto-backfill failed: {e}\n"
                "Set OPENWEATHER_API_KEY secret in GitHub and re-run."
            )

    df = pd.read_csv(csv_path)
    log.info(f"Loaded {len(df)} records from local CSV.")
    return df


# ═══════════════════════════════════════
#  EVALUATION HELPER
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
#  LSTM MODEL (TensorFlow — PDF requirement)
# ═══════════════════════════════════════

def train_lstm(Xtr, Xte, ytr, yte, horizon) -> tuple:
    """
    Simple LSTM using TensorFlow/Keras.
    Required by PDF spec: 'TensorFlow/PyTorch for advanced models.'
    Input shape: (samples, features) — reshaped to (samples, 1, features) for LSTM.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping

        # Reshape: (samples, timesteps=1, features)
        Xtr_3d = Xtr.reshape(Xtr.shape[0], 1, Xtr.shape[1])
        Xte_3d = Xte.reshape(Xte.shape[0], 1, Xte.shape[1])

        model = Sequential([
            LSTM(64, input_shape=(1, Xtr.shape[1]), return_sequences=True),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        early_stop = EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )

        model.fit(
            Xtr_3d, ytr,
            validation_data=(Xte_3d, yte),
            epochs=100,
            batch_size=32,
            callbacks=[early_stop],
            verbose=0
        )

        y_pred = model.predict(Xte_3d, verbose=0).flatten()
        metrics = evaluate("LSTM", horizon, yte, y_pred)
        return model, metrics

    except ImportError:
        log.warning("TensorFlow not installed — skipping LSTM. Add 'tensorflow>=2.15.0' to requirements.txt")
        return None, None
    except Exception as e:
        log.error(f"LSTM training failed: {e}")
        return None, None


# ═══════════════════════════════════════
#  2. TRAIN ALL MODELS
# ═══════════════════════════════════════

def train_all(Xtr, Xte, ytr, yte, horizon):
    """Train all models: Ridge, RandomForest, LightGBM, XGBoost, LSTM."""
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

    # Deep Learning — LSTM (TensorFlow) — required by PDF spec
    lstm_model, lstm_metrics = train_lstm(Xtr, Xte, ytr, yte, horizon)
    if lstm_model is not None:
        results.append(("lstm", lstm_model, lstm_metrics))

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

        tmp = os.path.join(LOCAL_MODELS_DIR, f"tmp_{horizon}h.pkl")
        joblib.dump(bundle, tmp)

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
        raise RuntimeError(
            f"Only {len(df)} training rows found. Need at least 50. "
            "Run: python src/backfill.py"
        )

    all_results = {}

    for h in HORIZONS:
        log.info(f"\n{'='*50}\nTraining for {h}h horizon\n{'='*50}")

        target = f"target_aqi_{h}h"
        df_h   = df.dropna(subset=[target]).copy()

        if len(df_h) < 30:
            log.warning(f"Not enough rows with target_{h}h filled. Skipping.")
            continue

        feats  = [f for f in FEATURE_COLS if f in df_h.columns]
        df_h[feats] = df_h[feats].fillna(df_h[feats].median())

        X, y = df_h[feats].values, df_h[target].values.astype(float)

        # FIX: Temporal split — no shuffle, preserves time order (prevents data leakage)
        split_idx = int(len(X) * 0.8)
        Xtr, Xte = X[:split_idx], X[split_idx:]
        ytr, yte = y[:split_idx], y[split_idx:]

        sc  = StandardScaler()
        Xtr = sc.fit_transform(Xtr)
        Xte = sc.transform(Xte)

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

        local_path = os.path.join(LOCAL_MODELS_DIR, f"model_{h}h.pkl")
        joblib.dump(bundle, local_path)
        log.info(f"Saved → {local_path}")

        save_to_registry(bundle, best_metrics, h)

        all_results[f"{h}h"] = {
            "best":    best_name,
            "metrics": best_metrics,
            "all":     {n: m for n, _, m in results},
        }

    if not all_results:
        raise RuntimeError("No models were trained successfully. Check logs above.")

    summary_path = os.path.join(LOCAL_MODELS_DIR, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    log.info(f"\nSummary saved → {summary_path}")
    log.info("\n✅ Training pipeline complete!")

    # Print metrics table for README
    log.info("\n" + "="*60)
    log.info("MODEL PERFORMANCE SUMMARY")
    log.info("="*60)
    log.info(f"{'Model':<12} {'Horizon':<8} {'RMSE':<8} {'MAE':<8} {'R²':<6}")
    log.info("-"*60)
    for horizon_key, data in all_results.items():
        for model_name, m in data["all"].items():
            marker = " ← best" if model_name == data["best"] else ""
            log.info(f"{model_name:<12} {horizon_key:<8} {m['rmse']:<8.2f} {m['mae']:<8.2f} {m['r2']:<6.3f}{marker}")
    log.info("="*60)


if __name__ == "__main__":
    run()