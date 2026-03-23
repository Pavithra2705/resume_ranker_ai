"""
Fraud Detection - Flask Backend
================================
Run: python app.py
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for
import pickle, json, os, numpy as np, pandas as pd
from utils.predictor import FraudPredictor

app = Flask(__name__)
predictor = FraudPredictor()

# ─────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────

@app.route("/")
def index():
    stats = predictor.get_dashboard_stats()
    return render_template("index.html", stats=stats)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        result = predictor.predict_single(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/predict_csv", methods=["POST"])
def predict_csv():
    try:
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No file uploaded"}), 400
        df = pd.read_csv(file)
        results = predictor.predict_batch(df)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/metrics")
def metrics():
    try:
        with open("models/eval_results.json") as f:
            data = json.load(f)
        return jsonify(data)
    except:
        return jsonify({"error": "Models not trained yet. Run train_model.py first."})


@app.route("/health")
def health():
    return jsonify({"status": "ok", "models_loaded": predictor.is_ready()})


if __name__ == "__main__":
    print("\n🚀 Fraud Detection Dashboard running at http://127.0.0.1:5000\n")
    app.run(debug=True, port=5000)
