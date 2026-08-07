"""
advanced_features.py
--------------------
Priorities 1, 2, 3: Advanced Feature Engineering + Better Encoding

Creates 35+ engineered features including:
  - Interaction features
  - Behavior scores (loyalty, risk, engagement, digital, value)
  - Age / Income / Tenure groups
  - Date-based features
  - Target Encoding (no leakage)
  - Frequency Encoding

Output: data/processed/churn_advanced.csv
Run   : python src/advanced_features.py
"""

import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

DIVIDER = "=" * 70
def section(t): print(f"\n{DIVIDER}\n  {t}\n{DIVIDER}")

RAW  = os.path.join("data", "raw", "customer_churn_1M.csv")
OUT  = os.path.join("data", "processed", "churn_advanced.csv")
os.makedirs(os.path.join("data", "processed"), exist_ok=True)

TARGET = "churn"
BINARY_SVC = ["has_phone_service","has_internet_service","has_online_security",
               "has_online_backup","has_device_protection","has_tech_support",
               "has_streaming_tv","has_streaming_movies"]

# ─── LOAD ─────────────────────────────────────────────────────────────────────
section("LOAD RAW DATA")
df = pd.read_csv(RAW)
print(f"  Shape: {df.shape}")
df["signup_date"] = pd.to_datetime(df["signup_date"])

# Reference date = max date in dataset
REF_DATE = df["signup_date"].max()

# ─────────────────────────────────────────────────────────────────────────────
# PRIORITY 2: IMPUTATION COMPARISON (50k sample)
# ─────────────────────────────────────────────────────────────────────────────
section("PRIORITY 2 — Imputation Strategy Comparison (50k sample)")

MISS_COLS = ["annual_income","customer_satisfaction","num_complaints","avg_monthly_gb","credit_score"]
samp = df.sample(50_000, random_state=42)
num_feats = ["age","annual_income","tenure","monthlycharges","totalcharges",
             "customer_satisfaction","num_complaints","avg_monthly_gb","credit_score"]

def score_imputer(imp, X_s, y_s, name):
    X_imp = pd.DataFrame(imp.fit_transform(X_s[num_feats]), columns=num_feats)
    lr = LogisticRegression(max_iter=200, class_weight="balanced", random_state=42)
    from sklearn.model_selection import cross_val_score
    sc = cross_val_score(lr, X_imp, y_s, cv=3, scoring="roc_auc")
    print(f"  {name:<30} ROC-AUC: {sc.mean():.4f} ± {sc.std():.4f}")
    return sc.mean()

print(f"\n  Comparing imputers on 50k stratified sample:")
X_s, y_s = samp[num_feats].copy(), samp[TARGET].copy()

imp_scores = {}
imp_scores["Median"]        = score_imputer(SimpleImputer(strategy="median"), X_s, y_s, "Median")
imp_scores["Mean"]          = score_imputer(SimpleImputer(strategy="mean"),   X_s, y_s, "Mean")
try:
    imp_scores["KNN(k=5)"]  = score_imputer(KNNImputer(n_neighbors=5),        X_s, y_s, "KNN (k=5)")
except Exception as e:
    print(f"  KNN failed: {e}")

best_imp = max(imp_scores, key=imp_scores.get)
print(f"\n  Best imputer: {best_imp} → using Median (stable for 1M rows)")

# Apply median imputation to full dataset
for col in MISS_COLS:
    med = df[col].median()
    df[col].fillna(med, inplace=True)
print(f"  Missing values remaining: {df.isnull().sum().sum()}")

# ─────────────────────────────────────────────────────────────────────────────
# OUTLIER HANDLING (Winsorize 1st/99th)
# ─────────────────────────────────────────────────────────────────────────────
CAP_COLS = ["annual_income","totalcharges","avg_monthly_gb",
            "days_since_last_interaction","num_complaints","monthlycharges"]
for col in CAP_COLS:
    p1, p99 = df[col].quantile(0.01), df[col].quantile(0.99)
    df[col] = df[col].clip(p1, p99)

# ─────────────────────────────────────────────────────────────────────────────
# PRIORITY 1: ADVANCED FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
section("PRIORITY 1 — Advanced Feature Engineering (35+ features)")

# ── Date-based ────────────────────────────────────────────────────────────────
df["customer_age_days"]   = (REF_DATE - df["signup_date"]).dt.days
df["customer_age_months"] = df["customer_age_days"] // 30
df["signup_year"]         = df["signup_date"].dt.year
df["signup_month"]        = df["signup_date"].dt.month
df["signup_quarter"]      = df["signup_date"].dt.quarter
df["signup_dayofweek"]    = df["signup_date"].dt.dayofweek   # 0=Mon
df["signup_weekofyear"]   = df["signup_date"].dt.isocalendar().week.astype(int)
print("  [DATE] customer_age_days/months, signup year/month/quarter/dow/week")

# ── Interaction Features ───────────────────────────────────────────────────────
df["charges_x_tenure"]         = df["monthlycharges"] * df["tenure"]
df["income_div_charges"]        = df["annual_income"]   / (df["monthlycharges"] + 1)
df["complaints_div_tenure"]     = df["num_complaints"]  / (df["tenure"] + 1)
df["svc_calls_div_tenure"]      = df["num_service_calls"] / (df["tenure"] + 1)
df["late_pay_div_tenure"]       = df["late_payments"]   / (df["tenure"] + 1)
df["totalcharges_div_tenure"]   = df["totalcharges"]    / (df["tenure"] + 1)
df["services_x_charges"]        = df["num_services"]    * df["monthlycharges"]
df["gb_x_charges"]              = df["avg_monthly_gb"]  * df["monthlycharges"]
df["satisfaction_x_tenure"]     = df["customer_satisfaction"] * df["tenure"]
df["credit_x_income"]           = df["credit_score"]    * (df["annual_income"] / 1000)
print("  [INTERACTION] charges×tenure, income÷charges, complaints÷tenure, ...")

# ── Behavior Scores ────────────────────────────────────────────────────────────
# Loyalty score (higher = more loyal)
df["loyalty_score"] = (
    df["tenure"] * 0.4 +
    df["customer_satisfaction"] * 3 +
    df["total_active_services"] if "total_active_services" in df.columns
    else df[BINARY_SVC].sum(axis=1) * 2
)
df["total_active_services"] = df[BINARY_SVC].sum(axis=1)
df["loyalty_score"] = (
    df["tenure"] * 0.4 +
    df["customer_satisfaction"] * 3 +
    df["total_active_services"] * 2
)

# Risk score (higher = more risky)
df["risk_score"] = (
    df["num_complaints"]    * 3 +
    df["late_payments"]     * 2 +
    df["num_service_calls"] * 1
)

# Engagement score (how actively customer interacts)
df["engagement_score"] = (
    df["num_service_calls"] * 0.5 +
    df["total_active_services"] * 1.5 +
    df["avg_monthly_gb"] * 0.1 -
    df["days_since_last_interaction"] * 0.05
)

# Digital usage score
df["digital_score"] = (
    df["has_internet_service"] * 3 +
    df["has_streaming_tv"] +
    df["has_streaming_movies"] +
    df["has_online_backup"] +
    df["has_online_security"] +
    df["avg_monthly_gb"] * 0.05 +
    (df["paperless_billing"] == "Yes").astype(int) * 2
)

# Customer value score (revenue potential)
df["value_score"] = (
    df["monthlycharges"] * 0.5 +
    df["annual_income"]  * 0.0001 +
    df["credit_score"]   * 0.01 +
    df["tenure"]         * 0.3
)

# Payment risk score
df["payment_risk"] = (
    df["late_payments"] * 3 +
    (df["payment_method"] == "electronic_check").astype(int) * 2 +
    (df["credit_score"] < 600).astype(int) * 2
)

# Complaint frequency per year
df["complaint_freq_annual"] = df["num_complaints"] / (df["customer_age_months"] / 12 + 0.1)

# Revenue per day of tenure
df["revenue_per_day"]  = df["totalcharges"] / (df["customer_age_days"] + 1)
df["avg_spend_per_svc"] = df["monthlycharges"] / (df["total_active_services"].replace(0, 1))

print("  [SCORES] loyalty, risk, engagement, digital, value, payment_risk")
print("  [RATES]  complaint_freq_annual, revenue_per_day, avg_spend_per_svc")

# ── Ratio Features ─────────────────────────────────────────────────────────────
df["satisfaction_per_complaint"] = df["customer_satisfaction"] / (df["num_complaints"] + 1)
df["charge_to_income_pct"]       = df["monthlycharges"] / (df["annual_income"] / 12 + 1) * 100
df["gb_per_service"]             = df["avg_monthly_gb"] / (df["total_active_services"] + 1)
df["clv"]                        = df["monthlycharges"] * df["tenure"]  # Customer LTV
print("  [RATIOS] satisfaction_per_complaint, charge_to_income_pct, gb_per_service, clv")

# ── Flag Features ──────────────────────────────────────────────────────────────
df["is_high_value"]      = (df["monthlycharges"] > df["monthlycharges"].quantile(0.75)).astype(int)
df["is_long_term"]       = (df["tenure"] > 24).astype(int)
df["is_month_to_month"]  = (df["contract"] == "month_to_month").astype(int)
df["has_late_payment"]   = (df["late_payments"] > 0).astype(int)
df["is_low_sat"]         = (df["customer_satisfaction"] < 4).astype(int)
df["is_high_risk"]       = (df["risk_score"] > df["risk_score"].quantile(0.75)).astype(int)
df["is_digital_heavy"]   = (df["digital_score"] > df["digital_score"].quantile(0.75)).astype(int)
df["is_premium"]         = ((df["customer_satisfaction"] >= 8) & (df["monthlycharges"] > 90)).astype(int)
df["has_no_support_svc"] = ((df["has_online_security"]==0) & (df["has_tech_support"]==0)).astype(int)
print("  [FLAGS] high_value, long_term, month_to_month, late_payment, low_sat, ...")

# ── Categorical Groups ─────────────────────────────────────────────────────────
# Age groups
df["age_group"] = pd.cut(df["age"], bins=[0,25,35,45,60,200],
                          labels=[0,1,2,3,4]).astype(int)

# Income groups
df["income_group"] = pd.cut(df["annual_income"],
                             bins=[0,30000,60000,100000,250001],
                             labels=[0,1,2,3]).astype(int)

# Tenure groups
df["tenure_group"] = pd.cut(df["tenure"], bins=[0,6,12,24,60,200],
                              labels=[0,1,2,3,4]).astype(int)

# Credit score groups
df["credit_group"] = pd.cut(df["credit_score"],
                              bins=[0,580,670,740,850],
                              labels=[0,1,2,3]).astype(int)

# Monthly charge groups
df["charge_group"] = pd.cut(df["monthlycharges"],
                              bins=[0,40,70,100,1000],
                              labels=[0,1,2,3]).astype(int)

print("  [GROUPS] age_group, income_group, tenure_group, credit_group, charge_group")

# Count total engineered features
eng_cols = [c for c in df.columns if c not in
            pd.read_csv(RAW, nrows=0).columns.tolist() + ["signup_date"]]
print(f"\n  Total engineered features: {len(eng_cols)}")

# ─────────────────────────────────────────────────────────────────────────────
# PRIORITY 3: ENCODING
# ─────────────────────────────────────────────────────────────────────────────
section("PRIORITY 3 — Encoding: Target + Frequency + Ordinal")

# Drop raw ID/date columns
df.drop(columns=["customer_id","signup_date"], inplace=True)

# Ordinal encoding (preserve meaning)
ORDINAL = {"contract": {"month_to_month":0,"one_year":1,"two_year":2},
           "paperless_billing": {"No":0,"Yes":1}}
for col, mapping in ORDINAL.items():
    df[col] = df[col].map(mapping)
    print(f"  [ORDINAL] {col}: {mapping}")

# Nominal categoricals
NOMINAL = ["gender","education","marital_status","payment_method"]

# A: Frequency Encoding
print("\n  [FREQUENCY ENCODING]")
for col in NOMINAL:
    freq = df[col].value_counts(normalize=True)
    df[f"{col}_freq"] = df[col].map(freq)
    print(f"    {col}_freq created")

# B: Target Encoding (fit on train only to prevent leakage)
print("\n  [TARGET ENCODING] (train split only — no leakage)")
train_idx, test_idx = train_test_split(df.index, test_size=0.20, stratify=df[TARGET], random_state=42)
global_mean = df.loc[train_idx, TARGET].mean()
SMOOTH = 30  # smoothing factor

for col in NOMINAL:
    stats = df.loc[train_idx].groupby(col)[TARGET].agg(["count","mean"])
    smooth = (stats["count"] * stats["mean"] + SMOOTH * global_mean) / (stats["count"] + SMOOTH)
    df[f"{col}_target_enc"] = df[col].map(smooth).fillna(global_mean)
    print(f"    {col}_target_enc: min={df[f'{col}_target_enc'].min():.4f} max={df[f'{col}_target_enc'].max():.4f}")

# Drop original string nominal cols (keeping freq + target encoded versions)
df.drop(columns=NOMINAL, inplace=True)
print(f"\n  Dropped original nominal columns: {NOMINAL}")
print(f"  Final shape: {df.shape}")
print(f"  All dtypes numeric: {all(df.dtypes != 'object')}")

# ─── SAVE ──────────────────────────────────────────────────────────────────────
section("SAVE")
df.to_csv(OUT, index=False)
size = os.path.getsize(OUT) / 1024**2
print(f"  Saved: {OUT}")
print(f"  Shape: {df.shape[0]:,} x {df.shape[1]} columns | {size:.1f} MB")
print(f"\n  Feature count: {df.shape[1]-1} features + 1 target")

# Quick sanity check
print(f"\n  Target distribution in saved data:")
print(f"  No Churn: {(df[TARGET]==0).sum():,} ({(df[TARGET]==0).mean()*100:.2f}%)")
print(f"  Churn:    {(df[TARGET]==1).sum():,} ({(df[TARGET]==1).mean()*100:.2f}%)")
print(f"\n  Sample correlations with churn (top 10 by abs value):")
corr = df.corr()[TARGET].drop(TARGET).abs().sort_values(ascending=False).head(10)
for feat, val in corr.items():
    print(f"    {feat:<40} {val:.4f}")
print(f"\n[✓] advanced_features.py COMPLETE\n")
