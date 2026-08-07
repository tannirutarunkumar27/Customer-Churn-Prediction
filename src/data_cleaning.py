"""
data_cleaning.py
----------------
Phase 3: Complete Data Cleaning Pipeline
  1. Load raw data
  2. Convert signup_date, extract date features
  3. Drop ID columns (customer_id, signup_date raw)
  4. Handle missing values (median imputation)
  5. Encode categorical features (Label Encoding)
  6. Handle outliers (cap at 1st/99th percentile for skewed features)
  7. Verify dtypes
  8. Save cleaned dataset to data/processed/churn_clean.csv

Run: python src/data_cleaning.py
"""

import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

DIVIDER = "=" * 70
def section(t): print(f"\n{DIVIDER}\n  {t}\n{DIVIDER}")

RAW_PATH     = os.path.join("data","raw","customer_churn_1M.csv")
CLEAN_PATH   = os.path.join("data","processed","churn_clean.csv")
os.makedirs(os.path.join("data","processed"), exist_ok=True)

NUMERICAL = [
    "age","annual_income","tenure","monthlycharges","totalcharges",
    "customer_satisfaction","num_complaints","num_service_calls",
    "late_payments","avg_monthly_gb","days_since_last_interaction","credit_score",
]
CATEGORICAL = ["gender","education","marital_status","contract","payment_method","paperless_billing"]
BINARY      = ["has_phone_service","has_internet_service","has_online_security","has_online_backup",
               "has_device_protection","has_tech_support","has_streaming_tv","has_streaming_movies","senior_citizen"]
TARGET      = "churn"

# Outlier capping — only for heavily right-skewed features
OUTLIER_CAP_COLS = ["annual_income","totalcharges","avg_monthly_gb","days_since_last_interaction","num_complaints"]


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD
# ═════════════════════════════════════════════════════════════════════════════
section("STEP 1 — Load Raw Data")
df = pd.read_csv(RAW_PATH)
print(f"  Raw shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"  Columns: {list(df.columns)}")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — CONVERT signup_date + EXTRACT DATE FEATURES
# ═════════════════════════════════════════════════════════════════════════════
section("STEP 2 — Convert signup_date & Extract Features")
df["signup_date"]    = pd.to_datetime(df["signup_date"])
df["signup_year"]    = df["signup_date"].dt.year
df["signup_month"]   = df["signup_date"].dt.month
df["signup_quarter"] = df["signup_date"].dt.quarter
print(f"  signup_date dtype  : {df['signup_date'].dtype}  ✓")
print(f"  Extracted: signup_year, signup_month, signup_quarter")
print(f"  Date range: {df['signup_date'].min().date()} → {df['signup_date'].max().date()}")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — DROP UNNECESSARY COLUMNS
# ═════════════════════════════════════════════════════════════════════════════
section("STEP 3 — Drop Unnecessary Columns")
drop_cols = ["customer_id", "signup_date"]
df.drop(columns=drop_cols, inplace=True)
print(f"  Dropped: {drop_cols}")
print(f"  Shape after drop: {df.shape}")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 4 — HANDLE MISSING VALUES
# ═════════════════════════════════════════════════════════════════════════════
section("STEP 4 — Handle Missing Values")
missing_before = df.isnull().sum()
cols_with_missing = missing_before[missing_before > 0]

print(f"\n  {'Column':<35} {'Missing':>10} {'Strategy'}")
print("  " + "-" * 65)
for col, cnt in cols_with_missing.items():
    pct = cnt / len(df) * 100
    median_val = df[col].median()
    df[col].fillna(median_val, inplace=True)
    print(f"  {col:<35} {cnt:>8,} ({pct:.2f}%)  → filled with median ({median_val:.2f})")

# Verify
remaining = df.isnull().sum().sum()
print(f"\n  Total missing after imputation: {remaining}")
print(f"  {'✓ No missing values remain' if remaining == 0 else '⚠ Still has missing values!'}")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 5 — ENCODE CATEGORICAL FEATURES
# ═════════════════════════════════════════════════════════════════════════════
section("STEP 5 — Encode Categorical Features (Label Encoding)")

label_encoders = {}
print(f"\n  {'Column':<25} {'Unique Values':<40} {'Encoding'}")
print("  " + "-" * 75)
for col in CATEGORICAL:
    le = LabelEncoder()
    unique_vals = df[col].unique().tolist()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"  {col:<25} {str(unique_vals)[:38]:<40} {mapping}")

# paperless_billing: Yes/No → already handled by LabelEncoder
print(f"\n  All categoricals encoded. Dtypes now:")
for col in CATEGORICAL:
    print(f"    {col}: {df[col].dtype}")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 6 — HANDLE OUTLIERS (Capping at 1st / 99th Percentile)
# ═════════════════════════════════════════════════════════════════════════════
section("STEP 6 — Handle Outliers (Winsorize at 1st/99th percentile)")

print(f"\n  {'Column':<35} {'P1':>10} {'P99':>10} {'Capped Low':>12} {'Capped High':>12}")
print("  " + "-" * 82)
for col in OUTLIER_CAP_COLS:
    p1  = df[col].quantile(0.01)
    p99 = df[col].quantile(0.99)
    low_count  = (df[col] < p1).sum()
    high_count = (df[col] > p99).sum()
    df[col] = df[col].clip(lower=p1, upper=p99)
    print(f"  {col:<35} {p1:>10.2f} {p99:>10.2f} {low_count:>12,} {high_count:>12,}")

print(f"\n  Note: Features like age, tenure, credit_score, monthlycharges")
print(f"  have minimal/no outliers — NOT capped (preserve natural range).")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 7 — VERIFY DATA TYPES
# ═════════════════════════════════════════════════════════════════════════════
section("STEP 7 — Verify Final Data Types")

print(f"\n  {'Column':<35} {'Dtype':<12} {'Non-Null':>10} {'Min':>10} {'Max':>10}")
print("  " + "-" * 82)
for col in df.columns:
    print(f"  {col:<35} {str(df[col].dtype):<12} {df[col].notnull().sum():>10,} "
          f"{df[col].min():>10.2f} {df[col].max():>10.2f}")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 8 — SAVE CLEANED DATASET
# ═════════════════════════════════════════════════════════════════════════════
section("STEP 8 — Save Cleaned Dataset")

df.to_csv(CLEAN_PATH, index=False)
size_mb = os.path.getsize(CLEAN_PATH) / (1024 * 1024)
print(f"\n  Saved: {CLEAN_PATH}")
print(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"  File size: {size_mb:.1f} MB")
print(f"\n  Final column list ({df.shape[1]} total):")
for i, col in enumerate(df.columns, 1):
    print(f"    {i:>2}. {col}")

print(f"\n{DIVIDER}")
print(f"  ✓ DATA CLEANING COMPLETE")
print(f"  Output: {CLEAN_PATH}")
print(f"{DIVIDER}\n")
