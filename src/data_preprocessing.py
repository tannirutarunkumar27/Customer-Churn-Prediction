"""
data_preprocessing.py
---------------------
Handles loading, cleaning, encoding, and splitting the raw churn dataset.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

RAW_DATA_PATH = os.path.join("data", "raw")
PROCESSED_DATA_PATH = os.path.join("data", "processed")


def load_data(filename: str) -> pd.DataFrame:
    """Load raw CSV data from the data/raw directory."""
    filepath = os.path.join(RAW_DATA_PATH, filename)
    df = pd.read_csv(filepath)
    print(f"[✓] Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def drop_irrelevant_columns(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Drop columns that are not useful for prediction."""
    df = df.drop(columns=[c for c in cols if c in df.columns])
    print(f"[✓] Dropped columns: {cols}")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill or drop missing values."""
    num_cols = df.select_dtypes(include=np.number).columns
    cat_cols = df.select_dtypes(include="object").columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])
    print("[✓] Handled missing values")
    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode categorical features."""
    le = LabelEncoder()
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    print(f"[✓] Encoded categorical columns: {list(cat_cols)}")
    return df


def split_and_scale(df: pd.DataFrame, target: str, test_size: float = 0.2):
    """Split data into train/test sets and scale features."""
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"[✓] Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def save_processed(X_train, X_test, y_train, y_test):
    """Save processed arrays to the data/processed directory."""
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    np.save(os.path.join(PROCESSED_DATA_PATH, "X_train.npy"), X_train)
    np.save(os.path.join(PROCESSED_DATA_PATH, "X_test.npy"), X_test)
    np.save(os.path.join(PROCESSED_DATA_PATH, "y_train.npy"), y_train)
    np.save(os.path.join(PROCESSED_DATA_PATH, "y_test.npy"), y_test)
    print("[✓] Saved processed data to data/processed/")


if __name__ == "__main__":
    df = load_data("churn_data.csv")
    df = drop_irrelevant_columns(df, ["customerID"])
    df = handle_missing_values(df)
    df = encode_categorical(df)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(df, target="Churn")
    save_processed(X_train, X_test, y_train, y_test)
