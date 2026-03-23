# 🛡️ FraudShield — AI-Powered Credit Card Fraud Detection

## Project Overview
End-to-end fraud detection system using an **ensemble of XGBoost + Isolation Forest + Logistic Regression**, built on the Kaggle Credit Card Fraud dataset (284,807 transactions, 492 frauds).

---

## 📁 Project Structure
```
fraud_detection/
├── train_model.py          # Full training pipeline (run once)
├── app.py                  # Flask web app
├── requirements.txt
├── utils/
│   └── predictor.py        # Prediction logic + ensemble
├── templates/
│   └── index.html          # Dashboard UI
├── static/
│   └── plots/              # ROC curve + feature importance (auto-generated)
├── models/                 # Saved models (auto-generated)
│   ├── xgb_model.pkl
│   ├── iso_forest.pkl
│   ├── lr_model.pkl
│   ├── scaler.pkl
│   ├── feature_names.json
│   └── eval_results.json
└── data/
    └── creditcard.csv      # Download from Kaggle (or auto-generated synthetic)
```

---

## 🚀 Setup & Run

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — (Optional) Download real dataset
Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
Place `creditcard.csv` in the `data/` folder.

> If you skip this step, a synthetic dataset is auto-generated so the app still works.

### Step 3 — Train the model
```bash
python train_model.py
```
This will:
- Load/generate dataset
- Preprocess + apply SMOTE for class imbalance
- Train XGBoost, Isolation Forest, Logistic Regression
- Save all models to `/models/`
- Save ROC curve + feature importance plots to `/static/plots/`
- Print evaluation metrics

### Step 4 — Launch the dashboard
```bash
python app.py
```
Open: http://127.0.0.1:5000

---

## 🔬 ML Pipeline Explained

### 1. Data Preprocessing
- StandardScaler on Amount and Time columns
- Features V1–V28 are already PCA-transformed in Kaggle dataset

### 2. Class Imbalance Handling
- SMOTE (Synthetic Minority Oversampling Technique)
- Original ratio: ~0.17% fraud → balanced to 50/50 for training

### 3. Models
| Model | Role | Why |
|-------|------|-----|
| XGBoost | Primary classifier | Best for tabular, handles imbalance well |
| Isolation Forest | Anomaly detection | Unsupervised — catches unusual patterns |
| Logistic Regression | Baseline | Interpretable, fast, good probability calibration |

### 4. Ensemble
```
final_score = 0.7 × XGBoost_prob + 0.3 × LR_prob
```

### 5. Evaluation Metrics
- ROC-AUC (primary — handles imbalance well)
- F1-Score (harmonic mean of precision & recall)
- Confusion Matrix

---

## 📊 Expected Results (Real Kaggle Dataset)
| Metric | XGBoost | Logistic Regression |
|--------|---------|---------------------|
| AUC    | ~0.98+  | ~0.97               |
| F1     | ~0.86+  | ~0.72               |

---

## 🌐 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Dashboard UI |
| `/predict` | POST | Single transaction JSON → fraud verdict |
| `/predict_csv` | POST | Upload CSV → batch predictions |
| `/metrics` | GET | Model evaluation metrics (JSON) |
| `/health` | GET | System health check |

### Example `/predict` request:
```json
POST /predict
{
  "V1": -3.04, "V2": 2.14, "V3": -3.43,
  "V4": 2.54,  "V5": -2.48, "V6": 2.54,
  "V7": -3.09, "V8": 0.56,  "V9": -0.89,
  "V10": -0.32, "Amount_scaled": 2.1, "Time_scaled": -0.5
}
```

---

## ❓ Interview Questions — Specific to This Project

**Q: Why XGBoost for fraud detection?**
A: XGBoost excels at tabular data, handles class imbalance via `scale_pos_weight`, is fast, and provides feature importance for explainability.

**Q: How did you handle the severe class imbalance (0.17% fraud)?**
A: SMOTE on training data + ROC-AUC as evaluation metric instead of accuracy. Accuracy alone would be misleading (99.83% by always predicting legit).

**Q: Why use an ensemble instead of just XGBoost?**
A: Isolation Forest adds an unsupervised anomaly detection layer catching patterns XGBoost might miss. LR provides calibrated probabilities and a reliable baseline.

**Q: What is Isolation Forest?**
A: An unsupervised anomaly detection algorithm that isolates observations by randomly selecting features and split values. Fraudulent (anomalous) points are isolated in fewer steps.

**Q: What is ROC-AUC and why prefer it here?**
A: Area Under the ROC Curve measures the model's ability to discriminate between classes across all thresholds. Critical for imbalanced datasets where accuracy is misleading.

---

## 👨‍💻 Tech Stack
- **Python 3.9+**
- **XGBoost** — primary classifier
- **Scikit-learn** — preprocessing, LR, Isolation Forest, metrics
- **imbalanced-learn** — SMOTE
- **Flask** — REST API + web dashboard
- **Matplotlib** — ROC + feature importance plots
- **Pandas/NumPy** — data wrangling
