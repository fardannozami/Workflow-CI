"""
modelling.py - Heart Disease Prediction Model
Digunakan oleh MLProject + GitHub Actions CI
Kriteria 3 Basic
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, roc_auc_score
)
import os, sys, warnings
warnings.filterwarnings('ignore')

# ===== Konfigurasi =====
DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else 'heart_preprocessing'
print(f"Data directory: {DATA_DIR}")

# ===== Load data =====
print("=" * 50)
print("LOADING DATA")
print("=" * 50)
X_train = pd.read_csv(os.path.join(DATA_DIR, 'X_train.csv'))
X_test = pd.read_csv(os.path.join(DATA_DIR, 'X_test.csv'))
y_train = pd.read_csv(os.path.join(DATA_DIR, 'y_train.csv')).squeeze('columns')
y_test = pd.read_csv(os.path.join(DATA_DIR, 'y_test.csv')).squeeze('columns')

print(f"X_train: {X_train.shape}")
print(f"X_test:  {X_test.shape}")

# ===== MLflow =====
mlflow.set_experiment("Heart_Disease_CI")

with mlflow.start_run(run_name="CI_Training"):
    mlflow.sklearn.autolog()

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"\nAccuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)

    mlflow.sklearn.log_model(model, "heart_disease_model")

    print(f"\nRun ID: {mlflow.active_run().info.run_id}")
    print("[OK] CI Training Selesai")
