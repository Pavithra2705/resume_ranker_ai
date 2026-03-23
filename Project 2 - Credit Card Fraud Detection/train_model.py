"""
Fraud Detection - Model Training Pipeline
==========================================
Dataset: Kaggle Credit Card Fraud Detection
Run this script ONCE to train and save the model.

NOTE: Uses GradientBoostingClassifier (sklearn) as XGBoost equivalent.
On machines with internet: pip install xgboost imbalanced-learn
and swap GradientBoostingClassifier with xgb.XGBClassifier.
"""

import pandas as pd
import numpy as np
import os, pickle, json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve, f1_score)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

DATA_PATH = "data/creditcard.csv"
MODEL_DIR = "models"

def load_data(path):
    print("[1/6] Loading dataset …")
    if not os.path.exists(path):
        print("     ⚠  creditcard.csv not found — generating synthetic data …")
        np.random.seed(42)
        n, fraud_n = 28480, 49
        X_legit  = np.random.randn(n - fraud_n, 28)
        X_fraud  = np.random.randn(fraud_n, 28) * 2 + np.tile(
            [-2,2,-2,2,-1.5,1,-1.5,.5,-.5,.3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], (fraud_n,1))
        amt_l = np.abs(np.random.exponential(88, n-fraud_n))
        amt_f = np.abs(np.random.exponential(250, fraud_n))
        t_l   = np.linspace(0, 172792, n-fraud_n)
        t_f   = np.random.uniform(0, 172792, fraud_n)
        X = np.vstack([X_legit, X_fraud])
        amounts = np.concatenate([amt_l, amt_f])
        times   = np.concatenate([t_l, t_f])
        labels  = np.concatenate([np.zeros(n-fraud_n), np.ones(fraud_n)])
        cols = [f"V{i}" for i in range(1,29)] + ["Amount","Time"]
        df = pd.DataFrame(np.column_stack([X, amounts, times]), columns=cols)
        df["Class"] = labels.astype(int)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        os.makedirs("data", exist_ok=True)
        df.to_csv(path, index=False)
        print(f"     Synthetic dataset → {path}  {df.shape}")
    df = pd.read_csv(path)
    print(f"     Shape: {df.shape}  |  Fraud: {df['Class'].sum()}  |  Legit: {(df['Class']==0).sum()}")
    return df

def preprocess(df):
    print("[2/6] Preprocessing …")
    df = df.copy().dropna()
    df["Amount_scaled"] = StandardScaler().fit_transform(df[["Amount"]])
    df["Time_scaled"]   = StandardScaler().fit_transform(df[["Time"]])
    df.drop(["Amount","Time"], axis=1, inplace=True)
    X, y = df.drop("Class", axis=1), df["Class"]
    feature_names = list(X.columns)
    with open(f"{MODEL_DIR}/feature_names.json","w") as f: json.dump(feature_names, f)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    # Balance classes (oversample fraud)
    print("     Balancing classes …")
    train_df = pd.concat([X_train, y_train], axis=1)
    legit, fraud = train_df[train_df.Class==0], train_df[train_df.Class==1]
    fraud_up = resample(fraud, replace=True, n_samples=len(legit)//2, random_state=42)
    bal = pd.concat([legit, fraud_up]).sample(frac=1, random_state=42)
    Xb, yb = bal.drop("Class",axis=1), bal["Class"]
    print(f"     Balanced: {yb.value_counts().to_dict()}")

    scaler = StandardScaler()
    Xb_sc  = scaler.fit_transform(Xb)
    Xt_sc  = scaler.transform(X_test)
    pickle.dump(scaler, open(f"{MODEL_DIR}/scaler.pkl","wb"))
    return Xb_sc, Xt_sc, yb.values, y_test.values, feature_names

def train_models(X_train, y_train):
    print("[3/6] Training models …")
    gbc = GradientBoostingClassifier(n_estimators=150, max_depth=5,
          learning_rate=0.05, subsample=0.8, random_state=42)
    gbc.fit(X_train, y_train)
    pickle.dump(gbc, open(f"{MODEL_DIR}/xgb_model.pkl","wb"))
    print("     ✓ Gradient Boosting (XGBoost-equiv)")

    iso = IsolationForest(n_estimators=100, contamination=0.005, random_state=42)
    iso.fit(X_train)
    pickle.dump(iso, open(f"{MODEL_DIR}/iso_forest.pkl","wb"))
    print("     ✓ Isolation Forest")

    lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    lr.fit(X_train, y_train)
    pickle.dump(lr, open(f"{MODEL_DIR}/lr_model.pkl","wb"))
    print("     ✓ Logistic Regression")
    return gbc, iso, lr

def evaluate(models, X_test, y_test):
    print("[4/6] Evaluating …")
    gbc, iso, lr = models
    results = {}
    for name, model in [("XGBoost", gbc), ("LogisticRegression", lr)]:
        yp = model.predict(X_test)
        yprob = model.predict_proba(X_test)[:,1]
        rep = classification_report(y_test, yp, output_dict=True)
        results[name] = {
            "auc": round(roc_auc_score(y_test, yprob), 4),
            "f1":  round(f1_score(y_test, yp), 4),
            "precision": round(rep.get("1",{}).get("precision",0), 4),
            "recall":    round(rep.get("1",{}).get("recall",0), 4),
        }
        print(f"     {name} → AUC:{results[name]['auc']}  F1:{results[name]['f1']}  "
              f"Prec:{results[name]['precision']}  Rec:{results[name]['recall']}")
    with open(f"{MODEL_DIR}/eval_results.json","w") as f: json.dump(results, f, indent=2)
    return results

def save_plots(models, X_test, y_test, feature_names):
    print("[5/6] Saving plots …")
    gbc, iso, lr = models
    os.makedirs("static/plots", exist_ok=True)

    # ROC
    fig, ax = plt.subplots(figsize=(7,5), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    for name, model, color in [("GradBoost", gbc, "#00e5ff"), ("LogReg", lr, "#ff6b6b")]:
        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:,1])
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0,1],[0,1],"--",color="#444",lw=1)
    ax.set_xlabel("FPR",color="white"); ax.set_ylabel("TPR",color="white")
    ax.set_title("ROC Curve — Fraud Detection",color="white",fontsize=13)
    ax.tick_params(colors="white"); ax.legend(facecolor="#1a1f2e",labelcolor="white")
    for s in ax.spines.values(): s.set_edgecolor("#333")
    plt.tight_layout(); plt.savefig("static/plots/roc_curve.png",dpi=150,bbox_inches="tight"); plt.close()

    # Feature Importance
    imp = gbc.feature_importances_; idx = np.argsort(imp)[-15:]
    fig, ax = plt.subplots(figsize=(8,6),facecolor="#0d1117"); ax.set_facecolor("#0d1117")
    ax.barh([feature_names[i] for i in idx], imp[idx], color=plt.cm.cool(np.linspace(.3,1,len(idx))))
    ax.set_xlabel("Score",color="white"); ax.set_title("Top 15 Feature Importances",color="white",fontsize=13)
    ax.tick_params(colors="white")
    for s in ax.spines.values(): s.set_edgecolor("#333")
    plt.tight_layout(); plt.savefig("static/plots/feature_importance.png",dpi=150,bbox_inches="tight"); plt.close()

    # Confusion Matrix
    cm = confusion_matrix(y_test, gbc.predict(X_test))
    fig, ax = plt.subplots(figsize=(5,4),facecolor="#0d1117"); ax.set_facecolor("#0d1117")
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Legit","Fraud"],color="white"); ax.set_yticklabels(["Legit","Fraud"],color="white")
    ax.set_title("Confusion Matrix",color="white",fontsize=12); ax.tick_params(colors="white")
    for (i,j),v in np.ndenumerate(cm):
        ax.text(j,i,str(v),ha='center',va='center',
                color='white' if cm[i,j]<cm.max()/2 else 'black',fontsize=14,fontweight='bold')
    plt.tight_layout(); plt.savefig("static/plots/confusion_matrix.png",dpi=150,bbox_inches="tight"); plt.close()
    print("     ✓ Plots saved → static/plots/")

if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test, feat = preprocess(df)
    models = train_models(X_train, y_train)
    evaluate(models, X_test, y_test)
    save_plots(models, X_test, y_test, feat)
    print("\n✅ Training complete! Run: python app.py → http://127.0.0.1:5000")
