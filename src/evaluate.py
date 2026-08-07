"""
evaluate.py
-----------
Evaluates the trained churn model and generates
performance reports, confusion matrix, and ROC curve plots.
"""

import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve
)

MODELS_PATH  = "models"
REPORTS_PATH = "reports"
PROCESSED_DATA_PATH = os.path.join("data", "processed")


def load_model(model_filename: str):
    """Load a saved model from the models/ directory."""
    return joblib.load(os.path.join(MODELS_PATH, model_filename))


def load_test_data():
    X_test = np.load(os.path.join(PROCESSED_DATA_PATH, "X_test.npy"))
    y_test = np.load(os.path.join(PROCESSED_DATA_PATH, "y_test.npy"))
    return X_test, y_test


def plot_confusion_matrix(y_true, y_pred, model_name: str):
    os.makedirs(REPORTS_PATH, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"])
    plt.title(f"Confusion Matrix — {model_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_PATH, f"confusion_matrix_{model_name}.png"), dpi=150)
    plt.close()
    print(f"[✓] Saved confusion matrix for {model_name}")


def plot_roc_curve(y_true, y_proba, model_name: str):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color="#6C63FF", lw=2,
             label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve — {model_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_PATH, f"roc_curve_{model_name}.png"), dpi=150)
    plt.close()
    print(f"[✓] Saved ROC curve for {model_name}")


def generate_report(y_true, y_pred, model_name: str):
    report = classification_report(y_true, y_pred, target_names=["No Churn", "Churn"])
    report_path = os.path.join(REPORTS_PATH, f"report_{model_name}.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"[✓] Saved classification report for {model_name}")
    return report


if __name__ == "__main__":
    model_file = "XGBoost_best.joblib"   # Change as needed
    model_name = model_file.replace("_best.joblib", "")
    model = load_model(model_file)
    X_test, y_test = load_test_data()
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    plot_confusion_matrix(y_test, y_pred, model_name)
    plot_roc_curve(y_test, y_proba, model_name)
    report = generate_report(y_test, y_pred, model_name)
    print("\n--- Classification Report ---")
    print(report)
