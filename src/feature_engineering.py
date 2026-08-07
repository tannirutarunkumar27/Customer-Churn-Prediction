"""
feature_engineering.py
-----------------------
Creates new features and performs additional transformations
on the preprocessed churn dataset.
"""

import pandas as pd
import numpy as np


def create_tenure_groups(df: pd.DataFrame, tenure_col: str = "tenure") -> pd.DataFrame:
    """Bin tenure into categorical groups."""
    bins = [0, 12, 24, 48, 72]
    labels = ["0-1yr", "1-2yr", "2-4yr", "4-6yr"]
    df["tenure_group"] = pd.cut(df[tenure_col], bins=bins, labels=labels, right=True)
    print("[✓] Created tenure_group feature")
    return df


def create_monthly_charge_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Ratio of monthly charges to total charges."""
    if "MonthlyCharges" in df.columns and "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["charge_ratio"] = df["MonthlyCharges"] / (df["TotalCharges"] + 1)
        print("[✓] Created charge_ratio feature")
    return df


def flag_high_value_customer(df: pd.DataFrame, threshold: float = 70.0) -> pd.DataFrame:
    """Flag customers with monthly charges above threshold."""
    if "MonthlyCharges" in df.columns:
        df["high_value"] = (df["MonthlyCharges"] > threshold).astype(int)
        print(f"[✓] Created high_value flag (threshold={threshold})")
    return df


def apply_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps."""
    df = create_tenure_groups(df)
    df = create_monthly_charge_ratio(df)
    df = flag_high_value_customer(df)
    return df


if __name__ == "__main__":
    # Example usage
    sample_df = pd.DataFrame({
        "tenure": [5, 15, 30, 55],
        "MonthlyCharges": [45.0, 75.0, 90.0, 110.0],
        "TotalCharges": ["225", "1125", "2700", "6050"]
    })
    result = apply_all_features(sample_df)
    print(result.head())
