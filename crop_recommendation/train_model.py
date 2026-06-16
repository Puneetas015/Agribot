"""
crop_recommendation/train_model.py
===================================
Trains a Random Forest classifier on the Kaggle Crop Recommendation dataset
(https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)
and saves the model + label encoder to core/ml_models/.

Usage:
    python crop_recommendation/train_model.py

Expected CSV columns:
    N, P, K, temperature, humidity, ph, rainfall, label
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# ── Paths ────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "Crop_recommendation.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "core", "ml_models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "crop_rf_model.pkl")
LE_PATH    = os.path.join(MODEL_DIR, "label_encoder.pkl")

FEATURE_COLS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET_COL   = "label"


def load_data(path: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    print(f"Dataset shape : {df.shape}")
    print(f"Crops         : {sorted(df[TARGET_COL].unique())}")
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    return X, y


def train(X: pd.DataFrame, y: pd.Series) -> tuple:
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # ── Evaluation ───────────────────────────────────────────
    y_pred = clf.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy : {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # 5-fold cross-validation on the full dataset
    cv_scores = cross_val_score(clf, X, y_enc, cv=5, scoring="accuracy", n_jobs=-1)
    print(f"5-Fold CV     : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── Feature importance ───────────────────────────────────
    importances = pd.Series(clf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("\nFeature Importances:")
    print(importances.to_string())

    return clf, le


def save_artifacts(clf, le):
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(le,  LE_PATH)
    print(f"\n✅  Model saved    → {MODEL_PATH}")
    print(f"✅  Encoder saved  → {LE_PATH}")


if __name__ == "__main__":
    X, y = load_data(DATA_PATH)
    clf, le = train(X, y)
    save_artifacts(clf, le)
