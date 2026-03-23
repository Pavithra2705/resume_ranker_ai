"""
utils/predictor.py — Core prediction logic
"""

import pickle, json, os
import numpy as np
import pandas as pd


class FraudPredictor:

    MODEL_DIR = "models"

    def __init__(self):
        self.xgb     = None
        self.iso     = None
        self.lr      = None
        self.scaler  = None
        self.features = None
        self._load_models()

    # ─────────────────────────────────────
    # Load
    # ─────────────────────────────────────
    def _load_models(self):
        try:
            self.xgb    = pickle.load(open(f"{self.MODEL_DIR}/xgb_model.pkl",  "rb"))
            self.iso    = pickle.load(open(f"{self.MODEL_DIR}/iso_forest.pkl", "rb"))
            self.lr     = pickle.load(open(f"{self.MODEL_DIR}/lr_model.pkl",   "rb"))
            self.scaler = pickle.load(open(f"{self.MODEL_DIR}/scaler.pkl",     "rb"))
            with open(f"{self.MODEL_DIR}/feature_names.json") as f:
                self.features = json.load(f)
            print("✅ All models loaded successfully.")
        except FileNotFoundError:
            print("⚠  Models not found. Run train_model.py first.")

    def is_ready(self):
        return all([self.xgb, self.iso, self.lr, self.scaler, self.features])

    # ─────────────────────────────────────
    # Single prediction
    # ─────────────────────────────────────
    def predict_single(self, data: dict) -> dict:
        if not self.is_ready():
            return {"error": "Models not loaded. Run train_model.py first."}

        # Build feature vector
        row = []
        for feat in self.features:
            val = data.get(feat, 0.0)
            row.append(float(val))

        X = np.array(row).reshape(1, -1)
        X_sc = self.scaler.transform(X)

        # XGBoost score
        xgb_prob  = float(self.xgb.predict_proba(X_sc)[0][1])
        xgb_pred  = int(self.xgb.predict(X_sc)[0])

        # Isolation Forest anomaly score (-1 = anomaly)
        iso_score = int(self.iso.predict(X_sc)[0])
        anomaly   = iso_score == -1

        # LR baseline
        lr_prob   = float(self.lr.predict_proba(X_sc)[0][1])

        # Ensemble: weighted vote
        ensemble_score = round(0.7 * xgb_prob + 0.3 * lr_prob, 4)
        is_fraud = ensemble_score > 0.89

        risk_level = (
            "🔴 HIGH RISK"    if ensemble_score > 0.85 else
            "🟠 MEDIUM RISK"  if ensemble_score > 0.65 else
            "🟢 LOW RISK"
        )

        return {
            "is_fraud"       : bool(is_fraud),
            "ensemble_score" : ensemble_score,
            "xgb_prob"       : round(xgb_prob, 4),
            "lr_prob"        : round(lr_prob, 4),
            "anomaly_flag"   : anomaly,
            "risk_level"     : risk_level,
            "verdict"        : "⚠️ FRAUDULENT" if is_fraud else "✅ LEGITIMATE"
        }

    # ─────────────────────────────────────
    # Batch CSV prediction
    # ─────────────────────────────────────
    def predict_batch(self, df: pd.DataFrame) -> dict:
        if not self.is_ready():
            return {"error": "Models not loaded."}

        # Align columns
        for feat in self.features:
            if feat not in df.columns:
                df[feat] = 0.0
        X = df[self.features].values.astype(float)
        X_sc = self.scaler.transform(X)

        xgb_probs  = self.xgb.predict_proba(X_sc)[:, 1]
        lr_probs   = self.lr.predict_proba(X_sc)[:, 1]
        ensemble   = 0.7 * xgb_probs + 0.3 * lr_probs
        predictions = (ensemble > 0.89).astype(int)

        fraud_count = int(predictions.sum())
        legit_count = int(len(predictions) - fraud_count)

        rows = []
        for i, (pred, score) in enumerate(zip(predictions, ensemble)):
            rows.append({
                "row"     : i + 1,
                "verdict" : "FRAUD" if pred else "LEGIT",
                "score"   : round(float(score), 4),
                "risk"    : "HIGH" if score > 0.75 else "MEDIUM" if score > 0.45 else "LOW"
            })

        return {
            "total"       : len(predictions),
            "fraud_count" : fraud_count,
            "legit_count" : legit_count,
            "fraud_rate"  : round(fraud_count / len(predictions) * 100, 2),
            "rows"        : rows[:100]   # cap at 100 for UI
        }

    # ─────────────────────────────────────
    # Dashboard stats
    # ─────────────────────────────────────
    def get_dashboard_stats(self) -> dict:
        try:
            with open(f"{self.MODEL_DIR}/eval_results.json") as f:
                results = json.load(f)
            xgb = results.get("XGBoost", {})
            lr  = results.get("LogisticRegression", {})
            return {
                "xgb_auc"  : xgb.get("auc", "N/A"),
                "xgb_f1"   : xgb.get("f1",  "N/A"),
                "lr_auc"   : lr.get("auc",   "N/A"),
                "lr_f1"    : lr.get("f1",    "N/A"),
                "models_ready": self.is_ready()
            }
        except:
            return {"models_ready": False}
