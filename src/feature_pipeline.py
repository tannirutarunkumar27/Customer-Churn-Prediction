"""
feature_pipeline.py
-------------------
Phase 4: Feature Engineering  (20 new features)
Phase 5: Feature Encoding     (One-Hot + Label Encoding)
Phase 6: Feature Scaling      (Standard, MinMax, Robust)

Output: data/processed/churn_features.csv  (ML-ready)
Run   : python src/feature_pipeline.py
"""

import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
)
import joblib

DIVIDER = "=" * 70
def section(t): print(f"\n{DIVIDER}\n  {t}\n{DIVIDER}")

RAW_PATH   = os.path.join("data", "raw", "customer_churn_1M.csv")
OUT_PATH   = os.path.join("data", "processed", "churn_features.csv")
SCALER_DIR = os.path.join("models", "scalers")
os.makedirs(os.path.join("data", "processed"), exist_ok=True)
os.makedirs(SCALER_DIR, exist_ok=True)

TARGET = "churn"
BINARY_SERVICES = [
    "has_phone_service","has_internet_service","has_online_security",
    "has_online_backup","has_device_protection","has_tech_support",
    "has_streaming_tv","has_streaming_movies",
]

# ═════════════════════════════════════════════════════════════════════════════
# LOAD & BASE CLEANING
# ═════════════════════════════════════════════════════════════════════════════
section("LOAD & BASE CLEANING")
df = pd.read_csv(RAW_PATH)
print(f"  Raw shape: {df.shape}")

df["signup_date"] = pd.to_datetime(df["signup_date"])
df["signup_year"]    = df["signup_date"].dt.year
df["signup_month"]   = df["signup_date"].dt.month
df["signup_quarter"] = df["signup_date"].dt.quarter
df.drop(columns=["customer_id", "signup_date"], inplace=True)

# Impute missing with median
for col in ["annual_income","customer_satisfaction","num_complaints","avg_monthly_gb","credit_score"]:
    df[col].fillna(df[col].median(), inplace=True)

# Cap outliers (1st/99th pct)
for col in ["annual_income","totalcharges","avg_monthly_gb","days_since_last_interaction","num_complaints"]:
    p1, p99 = df[col].quantile(0.01), df[col].quantile(0.99)
    df[col] = df[col].clip(p1, p99)

print(f"  After cleaning: {df.shape}")


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 4 — FEATURE ENGINEERING (20 features)
# ═════════════════════════════════════════════════════════════════════════════
section("PHASE 4 — FEATURE ENGINEERING")

# 1. Tenure in days
df["tenure_days"] = df["tenure"] * 30
print("  [1]  tenure_days = tenure * 30")

# 2. Customer Lifetime Value
df["customer_lifetime_value"] = df["monthlycharges"] * df["tenure"]
print("  [2]  customer_lifetime_value = monthlycharges * tenure")

# 3. Annual Revenue estimate
df["annual_revenue"] = df["monthlycharges"] * 12
print("  [3]  annual_revenue = monthlycharges * 12")

# 4. Avg spend per service
df["avg_spend_per_service"] = df["monthlycharges"] / (df["num_services"].replace(0, 1))
print("  [4]  avg_spend_per_service = monthlycharges / num_services")

# 5. Monthly charge as % of monthly income
df["charge_to_income_ratio"] = df["monthlycharges"] / (df["annual_income"] / 12 + 1)
print("  [5]  charge_to_income_ratio = monthlycharges / (annual_income/12)")

# 6. Complaint rate per month of tenure
df["complaint_rate"] = df["num_complaints"] / (df["tenure"] + 1)
print("  [6]  complaint_rate = num_complaints / (tenure + 1)")

# 7. Service call rate per month
df["service_call_rate"] = df["num_service_calls"] / (df["tenure"] + 1)
print("  [7]  service_call_rate = num_service_calls / (tenure + 1)")

# 8. Total active services (from binary flags)
df["total_active_services"] = df[BINARY_SERVICES].sum(axis=1)
print("  [8]  total_active_services = sum of all has_* service flags")

# 9. GB per service
df["gb_per_service"] = df["avg_monthly_gb"] / (df["total_active_services"].replace(0, 1))
print("  [9]  gb_per_service = avg_monthly_gb / total_active_services")

# 10. Risk score (complaints + calls + late payments combined)
df["risk_score"] = (df["num_complaints"] + df["num_service_calls"] + df["late_payments"]) / 3
print("  [10] risk_score = (num_complaints + num_service_calls + late_payments) / 3")

# 11. Satisfaction-to-complaint ratio
df["satisfaction_complaint_ratio"] = df["customer_satisfaction"] / (df["num_complaints"] + 1)
print("  [11] satisfaction_complaint_ratio = customer_satisfaction / (num_complaints + 1)")

# 12. High-value customer flag (monthly bill > 100)
df["is_high_value"] = (df["monthlycharges"] > 100).astype(int)
print("  [12] is_high_value = monthlycharges > $100")

# 13. Long-term customer flag (tenure > 24 months)
df["is_long_term"] = (df["tenure"] > 24).astype(int)
print("  [13] is_long_term = tenure > 24 months")

# 14. Month-to-month contract flag
df["is_month_to_month"] = (df["contract"] == "month_to_month").astype(int)
print("  [14] is_month_to_month = contract == 'month_to_month'")

# 15. High-risk payment flag (electronic check → highest churn)
df["is_high_risk_payment"] = (df["payment_method"] == "electronic_check").astype(int)
print("  [15] is_high_risk_payment = payment_method == 'electronic_check'")

# 16. Has late payment flag
df["has_late_payment"] = (df["late_payments"] > 0).astype(int)
print("  [16] has_late_payment = late_payments > 0")

# 17. Low satisfaction flag
df["is_low_satisfaction"] = (df["customer_satisfaction"] < 4).astype(int)
print("  [17] is_low_satisfaction = customer_satisfaction < 4")

# 18. High complaint flag
df["is_high_complaint"] = (df["num_complaints"] >= 2).astype(int)
print("  [18] is_high_complaint = num_complaints >= 2")

# 19. Senior + high charges flag (senior citizens with high bills at risk)
df["is_senior_high_charges"] = ((df["senior_citizen"] == 1) & (df["monthlycharges"] > 80)).astype(int)
print("  [19] is_senior_high_charges = senior_citizen AND monthlycharges > $80")

# 20. Premium customer (high satisfaction + high value)
df["is_premium"] = ((df["customer_satisfaction"] >= 8) & (df["monthlycharges"] > 90)).astype(int)
print("  [20] is_premium = satisfaction >= 8 AND monthlycharges > $90")

ENG_FEATURES = [
    "tenure_days","customer_lifetime_value","annual_revenue","avg_spend_per_service",
    "charge_to_income_ratio","complaint_rate","service_call_rate","total_active_services",
    "gb_per_service","risk_score","satisfaction_complaint_ratio",
    "is_high_value","is_long_term","is_month_to_month","is_high_risk_payment",
    "has_late_payment","is_low_satisfaction","is_high_complaint",
    "is_senior_high_charges","is_premium",
]
print(f"\n  Total engineered features: {len(ENG_FEATURES)}")
print(f"\n  SAMPLE STATS of new features:")
print(f"  {'Feature':<35} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
print("  " + "-" * 80)
for f in ENG_FEATURES:
    print(f"  {f:<35} {df[f].mean():>10.3f} {df[f].std():>10.3f} "
          f"{df[f].min():>10.3f} {df[f].max():>10.3f}")

# Churn signal check for new features
print(f"\n  CHURN SIGNAL — New Features (mean diff):")
print(f"  {'Feature':<35} {'No Churn':>12} {'Churn':>12} {'Diff%':>10}")
print("  " + "-" * 72)
for f in ENG_FEATURES:
    m0 = df.loc[df[TARGET]==0, f].mean()
    m1 = df.loc[df[TARGET]==1, f].mean()
    diff = (m1 - m0) / (abs(m0) + 1e-9) * 100
    marker = " <-- STRONG" if abs(diff) > 15 else ""
    print(f"  {f:<35} {m0:>12.4f} {m1:>12.4f} {diff:>+10.1f}%{marker}")


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 5 — FEATURE ENCODING
# ═════════════════════════════════════════════════════════════════════════════
section("PHASE 5 — FEATURE ENCODING")

# Nominal categoricals → One-Hot Encoding (no natural order)
NOMINAL_CATS = ["gender", "education", "marital_status", "payment_method"]

# Ordinal categoricals → Label Encoding with meaningful order
ORDINAL_MAP = {
    "contract":          {"month_to_month": 0, "one_year": 1, "two_year": 2},
    "paperless_billing": {"No": 0, "Yes": 1},
}

print("\n  [A] ONE-HOT ENCODING (Nominal Categoricals)")
print(f"  Columns: {NOMINAL_CATS}")
df_encoded = pd.get_dummies(df, columns=NOMINAL_CATS, drop_first=False, dtype=int)
new_ohe_cols = [c for c in df_encoded.columns if c not in df.columns]
print(f"  Created {len(new_ohe_cols)} dummy columns: {new_ohe_cols}")

print("\n  [B] LABEL / ORDINAL ENCODING")
for col, mapping in ORDINAL_MAP.items():
    df_encoded[col] = df_encoded[col].map(mapping)
    print(f"  {col}: {mapping}")

print(f"\n  Shape after encoding: {df_encoded.shape}")
print(f"  All dtypes numeric: {all(df_encoded.dtypes != 'object')}")
print(f"\n  Encoded column list ({df_encoded.shape[1]} cols):")
for i, c in enumerate(df_encoded.columns, 1):
    print(f"    {i:>2}. {c}")


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 6 — FEATURE SCALING
# ═════════════════════════════════════════════════════════════════════════════
section("PHASE 6 — FEATURE SCALING")

# Separate X and y
y = df_encoded[TARGET].copy()
X = df_encoded.drop(columns=[TARGET])

# Columns to scale (continuous numerical only — not binary/flag columns)
SCALE_COLS = [
    "age","annual_income","tenure","monthlycharges","totalcharges",
    "customer_satisfaction","num_complaints","num_service_calls","late_payments",
    "avg_monthly_gb","days_since_last_interaction","credit_score","dependents",
    "num_services","tenure_days","customer_lifetime_value","annual_revenue",
    "avg_spend_per_service","charge_to_income_ratio","complaint_rate",
    "service_call_rate","total_active_services","gb_per_service","risk_score",
    "satisfaction_complaint_ratio","signup_year","signup_month","signup_quarter",
]
# Keep only cols that exist in X
SCALE_COLS = [c for c in SCALE_COLS if c in X.columns]

print(f"\n  Columns to scale: {len(SCALE_COLS)}")
print(f"  Binary/flag columns left unscaled (already 0/1)")

# ── StandardScaler ────────────────────────────────────────────────────────────
print("\n  [1] StandardScaler — (mean=0, std=1) — best for normally distributed features")
ss = StandardScaler()
X_std = X.copy()
X_std[SCALE_COLS] = ss.fit_transform(X[SCALE_COLS])
joblib.dump(ss, os.path.join(SCALER_DIR, "standard_scaler.joblib"))
print(f"  Saved: models/scalers/standard_scaler.joblib")
print(f"  Sample (first 3 scaled cols after StandardScaler):")
print(X_std[SCALE_COLS[:3]].describe().loc[["mean","std","min","max"]].round(3).to_string())

# ── MinMaxScaler ──────────────────────────────────────────────────────────────
print("\n  [2] MinMaxScaler — (range 0 to 1) — best for bounded/neural network features")
mm = MinMaxScaler()
X_mm = X.copy()
X_mm[SCALE_COLS] = mm.fit_transform(X[SCALE_COLS])
joblib.dump(mm, os.path.join(SCALER_DIR, "minmax_scaler.joblib"))
print(f"  Saved: models/scalers/minmax_scaler.joblib")
print(f"  Sample (first 3 scaled cols after MinMaxScaler):")
print(X_mm[SCALE_COLS[:3]].describe().loc[["mean","std","min","max"]].round(3).to_string())

# ── RobustScaler ──────────────────────────────────────────────────────────────
print("\n  [3] RobustScaler — (IQR-based) — best for features with outliers")
rs = RobustScaler()
X_rb = X.copy()
X_rb[SCALE_COLS] = rs.fit_transform(X[SCALE_COLS])
joblib.dump(rs, os.path.join(SCALER_DIR, "robust_scaler.joblib"))
print(f"  Saved: models/scalers/robust_scaler.joblib")
print(f"  Sample (first 3 scaled cols after RobustScaler):")
print(X_rb[SCALE_COLS[:3]].describe().loc[["mean","std","min","max"]].round(3).to_string())

# Scaler decision guide
print("""
  SCALER USAGE GUIDE:
  +------------------------------------------------------------------+
  | StandardScaler  -> Logistic Regression, SVM, KNN, PCA           |
  | MinMaxScaler    -> Neural Networks, features needing [0,1] range |
  | RobustScaler    -> Tree models (XGBoost, RF) with outliers       |
  +------------------------------------------------------------------+
  Note: Tree-based models (XGBoost, LightGBM, RandomForest) are
  scale-invariant and do NOT require scaling, but scaling won't hurt.
""")

# ── SAVE FINAL ML-READY DATASET ───────────────────────────────────────────────
section("SAVE ML-READY DATASET (StandardScaler version)")

# Combine X (std-scaled) + y → final dataset
df_final = X_std.copy()
df_final[TARGET] = y.values
df_final.to_csv(OUT_PATH, index=False)

size_mb = os.path.getsize(OUT_PATH) / (1024**2)
print(f"\n  Saved: {OUT_PATH}")
print(f"  Shape: {df_final.shape[0]:,} rows x {df_final.shape[1]} columns")
print(f"  Size : {size_mb:.1f} MB")
print(f"\n  Churn distribution in final dataset:")
print(f"  No Churn: {(y==0).sum():,} ({(y==0).mean()*100:.2f}%)")
print(f"  Churn   : {(y==1).sum():,} ({(y==1).mean()*100:.2f}%)")

print(f"""
{DIVIDER}
  PIPELINE COMPLETE:
  [✓] Phase 4: 20 engineered features created
  [✓] Phase 5: One-Hot Encoding (4 nominal) + Ordinal Encoding (2)
  [✓] Phase 6: StandardScaler, MinMaxScaler, RobustScaler applied & saved
  [✓] Final ML-ready dataset: {OUT_PATH}
  [✓] Scalers saved to: models/scalers/
{DIVIDER}
""")
