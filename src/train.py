"""
train.py
--------
Trains multiple ML models for customer churn prediction
and saves the best model to the models/ directory.
"""

import os
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

PROCESSED_DATA_PATH = os.path.join("data", "processed")
MODELS_PATH = "models"


def load_processed_data():
    """Load preprocessed train/test splits."""
    X_train = np.load(os.path.join(PROCESSED_DATA_PATH, "X_train.npy"))
    X_test  = np.load(os.path.join(PROCESSED_DATA_PATH, "X_test.npy"))
    y_train = np.load(os.path.join(PROCESSED_DATA_PATH, "y_train.npy"))
    y_test  = np.load(os.path.join(PROCESSED_DATA_PATH, "y_test.npy"))
    return X_train, X_test, y_train, y_test


def get_models() -> dict:
    """Return a dictionary of models to train."""
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest":       RandomForestClassifier(n_estimators=200, random_state=42),
        "XGBoost":            XGBClassifier(n_estimators=200, use_label_encoder=False,
                                            eval_metric="logloss", random_state=42),
        "LightGBM":           LGBMClassifier(n_estimators=200, random_state=42),
    }


def train_and_evaluate(models: dict, X_train, X_test, y_train, y_test) -> dict:
    """Train each model and collect AUC scores."""
    results = {}
    for name, model in models.items():
        print(f"\n[→] Training {name}...")
        model.fit(X_train, y_train)
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        results[name] = {"model": model, "auc": auc}
        print(f"   AUC: {auc:.4f}")
        print(classification_report(y_test, y_pred))
    return results


def save_best_model(results: dict):
    """Save the best-performing model to disk."""
    os.makedirs(MODELS_PATH, exist_ok=True)
    best_name = max(results, key=lambda k: results[k]["auc"])
    best_model = results[best_name]["model"]
    save_path = os.path.join(MODELS_PATH, f"{best_name}_best.joblib")
    joblib.dump(best_model, save_path)
    print(f"\n[✓] Best model: {best_name} (AUC={results[best_name]['auc']:.4f})")
    print(f"[✓] Saved to {save_path}")
    return best_name


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_processed_data()
    models  = get_models()
    results = train_and_evaluate(models, X_train, X_test, y_train, y_test)
    save_best_model(results)
