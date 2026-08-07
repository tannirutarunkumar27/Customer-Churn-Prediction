"""
utils.py
--------
Shared utility functions used across the churn prediction pipeline.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── Logging ──────────────────────────────────────────────────────────────────
def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger."""
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
    )
    return logging.getLogger(name)


# ── File I/O ─────────────────────────────────────────────────────────────────
def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def save_json(data: dict, filepath: str):
    """Save a dictionary as a JSON file."""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)
    print(f"[✓] Saved JSON: {filepath}")


def load_json(filepath: str) -> dict:
    """Load a JSON file into a dictionary."""
    with open(filepath, "r") as f:
        return json.load(f)


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_feature_importance(feature_names: list, importances: np.ndarray,
                             title: str = "Feature Importance", save_path: str = None):
    """Plot and optionally save a bar chart of feature importances."""
    idx = np.argsort(importances)[::-1][:15]  # top 15
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[idx], y=np.array(feature_names)[idx], palette="viridis")
    plt.title(title)
    plt.xlabel("Importance Score")
    plt.tight_layout()
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150)
        print(f"[✓] Saved feature importance plot: {save_path}")
    plt.show()


def plot_class_distribution(y: pd.Series, title: str = "Class Distribution"):
    """Plot the distribution of target classes."""
    plt.figure(figsize=(5, 4))
    counts = y.value_counts()
    sns.barplot(x=counts.index.astype(str), y=counts.values, palette="coolwarm")
    plt.title(title)
    plt.xlabel("Churn")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


# ── Metrics ────────────────────────────────────────────────────────────────
def metrics_summary(y_true, y_pred, y_proba) -> dict:
    """Return a summary of key classification metrics."""
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score,
        recall_score, roc_auc_score
    )
    return {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall":    round(recall_score(y_true, y_pred), 4),
        "f1_score":  round(f1_score(y_true, y_pred), 4),
        "roc_auc":   round(roc_auc_score(y_true, y_proba), 4),
    }
