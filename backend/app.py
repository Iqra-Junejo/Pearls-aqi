"""app.py — Flask API for Pearls AQI Predictor"""
import os, sys, json, logging
from datetime import datetime, timezone
from flask import Flask, jsonify, request
from flask_cors import CORS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from config import SECRET_KEY, PORT, DEBUG

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
app = Flask(__name__)
app.secret_key = SECRET_KEY
CORS(app)


@app.route("/api/predict")
def predict():
    city = request.args.get("city", "karachi")
    try:
        from inference_pipeline import predict as run
        result = run(city)

        # Add hazardous AQI alert (per PDF requirement)
        aqi = result["current"]["aqi"]
        if aqi > 300:
            result["alert"] = {"level": "HAZARDOUS", "color": "#be123c",
                "message": f"⚠️ HAZARDOUS air quality (AQI={aqi}). Stay indoors. Avoid all outdoor activity."}
        elif aqi > 200:
            result["alert"] = {"level": "VERY_UNHEALTHY", "color": "#a855f7",
                "message": f"⚠️ Very Unhealthy air (AQI={aqi}). Avoid outdoor activity. Close windows."}
        elif aqi > 150:
            result["alert"] = {"level": "UNHEALTHY", "color": "#ef4444",
                "message": f"⚠️ Unhealthy air (AQI={aqi}). Reduce outdoor exertion. Wear N95 mask."}
        else:
            result["alert"] = None

        return jsonify(result)
    except Exception as e:
        logging.error(f"Predict error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/current")
def current():
    """Live fetch directly from OpenWeather — no ML."""
    try:
        from feature_pipeline import fetch_weather, fetch_pollution
        from config import CITY
        return jsonify({**fetch_weather(), **fetch_pollution(), "city": CITY,
                        "fetched_at": datetime.now(timezone.utc).isoformat()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/history")
def history():
    """Last 7 days of hourly readings."""
    try:
        import pandas as pd
        from config import LOCAL_DATA_DIR
        csv_path = os.path.join(LOCAL_DATA_DIR, "features.csv")
        if not os.path.exists(csv_path):
            return jsonify([])
        df = pd.read_csv(csv_path).sort_values("timestamp", ascending=False).head(168)
        cols = ["timestamp","aqi","pm25","temperature","humidity"]
        cols = [c for c in cols if c in df.columns]
        return jsonify(list(reversed(df[cols].to_dict("records"))))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/metrics")
def metrics():
    from config import LOCAL_MODELS_DIR
    path = os.path.join(LOCAL_MODELS_DIR, "training_summary.json")
    if not os.path.exists(path):
        return jsonify({"error": "No models yet. Run training_pipeline.py"}), 404
    with open(path) as f:
        return jsonify(json.load(f))


@app.route("/api/health")
def health():
    from config import LOCAL_DATA_DIR, LOCAL_MODELS_DIR
    import pandas as pd

    out = {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}

    # Data check
    csv_path = os.path.join(LOCAL_DATA_DIR, "features.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        out["data"] = {"records": len(df), "latest": df["timestamp"].max() if "timestamp" in df else "—"}
    else:
        out["data"] = {"records": 0, "hint": "Run: python src/backfill.py"}

    # Models check
    out["models"] = {
        f"{h}h": "ready" if os.path.exists(os.path.join(LOCAL_MODELS_DIR, f"model_{h}h.pkl"))
                 else "missing — run training_pipeline.py"
        for h in [24, 48, 72]
    }
    return jsonify(out)


@app.errorhandler(404)
def not_found(e): return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def server_err(e): return jsonify({"error": "Server error"}), 500


if __name__ == "__main__":
    print(f"\n🌫️  Pearls AQI Backend → http://localhost:{PORT}")
    print(f"   /api/predict  /api/current  /api/history  /api/health\n")
    app.run(host="0.0.0.0", port=PORT, debug=DEBUG)
