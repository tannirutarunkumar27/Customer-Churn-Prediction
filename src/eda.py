"""
eda.py
------
Steps 2–6: Dataset Understanding, Loading, Data Types,
           Missing Values, and Duplicate Detection.

Run: python src/eda.py
"""

import pandas as pd
import numpy as np
import os

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
RAW_DATA_PATH = os.path.join("data", "raw", "customer_churn_1M.csv")
DIVIDER       = "=" * 70


def section(title: str):
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — UNDERSTAND THE DATASET
# ─────────────────────────────────────────────────────────────────────────────
def step2_dataset_understanding(df: pd.DataFrame):
    section("STEP 2 — UNDERSTAND THE DATASET")

    print(f"\n  Rows    : {df.shape[0]:,}")
    print(f"  Columns : {df.shape[1]}")
    print(f"  Target  : 'churn'  (0 = No Churn, 1 = Churn)")

    print("\n  TARGET DISTRIBUTION:")
    vc = df["churn"].value_counts()
    pct = df["churn"].value_counts(normalize=True) * 100
    for label, count in vc.items():
        meaning = "No Churn" if label == 0 else "Churn"
        print(f"    {label} ({meaning}): {count:>8,}  ({pct[label]:.2f}%)")

    # Column Data Dictionary
    data_dict = {
        "customer_id":               ("Unique customer identifier",              "ID",          "CUST0000000001"),
        "signup_date":               ("Date & time customer signed up",          "DateTime",    "2022-12-12 12:53"),
        "age":                       ("Customer age in years",                   "Numerical",   "43"),
        "gender":                    ("Customer gender",                         "Categorical", "Male / Female"),
        "annual_income":             ("Annual income in USD",                    "Numerical",   "55085.25"),
        "education":                 ("Highest education level",                 "Categorical", "college / master"),
        "marital_status":            ("Marital status",                          "Categorical", "married / single"),
        "dependents":                ("Number of dependents",                    "Numerical",   "1"),
        "tenure":                    ("Months with the company",                 "Numerical",   "24"),
        "contract":                  ("Contract type",                           "Categorical", "month-to-month / one_year / two_year"),
        "payment_method":            ("Payment method used",                     "Categorical", "electronic_check / bank_transfer"),
        "paperless_billing":         ("Whether paperless billing is enabled",    "Categorical", "Yes / No"),
        "senior_citizen":            ("Whether customer is a senior citizen",    "Binary",      "0 / 1"),
        "monthlycharges":            ("Monthly bill in USD",                     "Numerical",   "67.20"),
        "totalcharges":              ("Total charges billed to date in USD",     "Numerical",   "144.39"),
        "num_services":              ("Number of services subscribed",           "Numerical",   "1"),
        "has_phone_service":         ("Has phone service",                       "Binary",      "0 / 1"),
        "has_internet_service":      ("Has internet service",                    "Binary",      "0 / 1"),
        "has_online_security":       ("Has online security add-on",              "Binary",      "0 / 1"),
        "has_online_backup":         ("Has online backup add-on",                "Binary",      "0 / 1"),
        "has_device_protection":     ("Has device protection add-on",            "Binary",      "0 / 1"),
        "has_tech_support":          ("Has tech support add-on",                 "Binary",      "0 / 1"),
        "has_streaming_tv":          ("Has streaming TV service",                "Binary",      "0 / 1"),
        "has_streaming_movies":      ("Has streaming movies service",            "Binary",      "0 / 1"),
        "customer_satisfaction":     ("Satisfaction score (1–10)",               "Numerical",   "9.0"),
        "num_complaints":            ("Number of complaints filed",              "Numerical",   "0.0"),
        "num_service_calls":         ("Number of service/support calls made",    "Numerical",   "0"),
        "late_payments":             ("Number of late payments",                 "Numerical",   "0"),
        "avg_monthly_gb":            ("Average monthly data usage in GB",        "Numerical",   "109.63"),
        "days_since_last_interaction": ("Days since last customer interaction",  "Numerical",   "16"),
        "credit_score":              ("Customer credit score",                   "Numerical",   "585.0"),
        "churn":                     ("Target — did customer churn?",            "Target",      "0 (No) / 1 (Yes)"),
    }

    print("\n\n  COLUMN DATA DICTIONARY:")
    header = f"  {'Column':<30} {'Description':<45} {'Type':<13} {'Example'}"
    print(header)
    print("  " + "-" * 110)
    for col, (desc, dtype, example) in data_dict.items():
        print(f"  {col:<30} {desc:<45} {dtype:<13} {example}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — LOAD THE DATASET
# ─────────────────────────────────────────────────────────────────────────────
def step3_load_dataset(df: pd.DataFrame):
    section("STEP 3 — LOAD THE DATASET")

    print(f"\n  ✔  Dataset loaded from: {RAW_DATA_PATH}")
    print(f"\n  Shape : {df.shape[0]:,} rows × {df.shape[1]} columns")

    print("\n  COLUMN NAMES:")
    for i, col in enumerate(df.columns, 1):
        print(f"    {i:>2}. {col}")

    print("\n  FIRST 5 ROWS:")
    print(df.head(5).to_string(index=False))

    print("\n  LAST 5 ROWS:")
    print(df.tail(5).to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — INSPECT DATA TYPES
# ─────────────────────────────────────────────────────────────────────────────
def step4_data_types(df: pd.DataFrame):
    section("STEP 4 — INSPECT DATA TYPES")

    int_cols   = df.select_dtypes(include="int64").columns.tolist()
    float_cols = df.select_dtypes(include="float64").columns.tolist()
    obj_cols   = df.select_dtypes(include="object").columns.tolist()
    bool_cols  = df.select_dtypes(include="bool").columns.tolist()

    print(f"\n  Integer  columns ({len(int_cols)}): {int_cols}")
    print(f"\n  Float    columns ({len(float_cols)}): {float_cols}")
    print(f"\n  Object   columns ({len(obj_cols)}): {obj_cols}")
    print(f"\n  Boolean  columns ({len(bool_cols)}): {bool_cols if bool_cols else 'None'}")

    # Q1: Are any numeric columns stored as strings?
    print("\n  ── Q1: Any numeric columns stored as strings? ──")
    suspicious = []
    for col in obj_cols:
        if col in ("customer_id", "signup_date"):
            continue
        sample = df[col].dropna().head(100)
        try:
            pd.to_numeric(sample)
            suspicious.append(col)
        except (ValueError, TypeError):
            pass
    if suspicious:
        print(f"  ⚠  Suspicious (look numeric but stored as object): {suspicious}")
    else:
        print("  ✔  No numeric-looking columns stored as strings.")

    # Q2: Are dates stored correctly?
    print("\n  ── Q2: Date columns stored correctly? ──")
    date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
    for col in date_cols:
        dtype = df[col].dtype
        print(f"  Column '{col}': dtype = {dtype}  → ", end="")
        if str(dtype) == "object":
            print("⚠  Stored as string — should be parsed as datetime.")
        else:
            print("✔  Already datetime.")

    print("\n  FULL DTYPE TABLE:")
    print(f"  {'Column':<35} {'Dtype':<12} {'Category'}")
    print("  " + "-" * 65)
    category_map = {}
    for col in df.columns:
        dt = str(df[col].dtype)
        if dt == "int64":   cat = "Integer"
        elif dt == "float64": cat = "Float"
        elif dt == "bool":  cat = "Boolean"
        else:               cat = "Object/String"
        print(f"  {col:<35} {dt:<12} {cat}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — CHECK MISSING VALUES
# ─────────────────────────────────────────────────────────────────────────────
def step5_missing_values(df: pd.DataFrame):
    section("STEP 5 — CHECK MISSING VALUES")

    total_rows = len(df)
    missing    = df.isnull().sum()
    pct        = (missing / total_rows * 100).round(4)

    missing_df = pd.DataFrame({
        "Column":         missing.index,
        "Missing Values": missing.values,
        "Percentage (%)": pct.values,
    })
    missing_df = missing_df.sort_values("Missing Values", ascending=False)

    print(f"\n  Total rows: {total_rows:,}\n")
    print(f"  {'Column':<35} {'Missing Values':>16} {'Percentage (%)':>16}")
    print("  " + "-" * 72)
    for _, row in missing_df.iterrows():
        flag = "  ⚠" if row["Missing Values"] > 0 else "  ✔"
        print(f"{flag} {row['Column']:<33} {int(row['Missing Values']):>16,} {row['Percentage (%)']:>15.4f}%")

    cols_with_missing = missing_df[missing_df["Missing Values"] > 0]
    print(f"\n  Summary: {len(cols_with_missing)} column(s) have missing values:")
    for _, row in cols_with_missing.iterrows():
        print(f"    • {row['Column']}: {int(row['Missing Values']):,} missing ({row['Percentage (%)']:.4f}%)")

    print("\n  RECOMMENDED ACTIONS:")
    for _, row in cols_with_missing.iterrows():
        col  = row["Column"]
        pct_ = row["Percentage (%)"]
        dtype = str(df[col].dtype)
        if pct_ < 5:
            action = "Fill with median" if "float" in dtype or "int" in dtype else "Fill with mode"
        elif pct_ < 20:
            action = "Consider median/mode imputation or a flag column"
        else:
            action = "Consider dropping or using a model-based imputation"
        print(f"    • {col:<35} ({pct_:.2f}% missing) → {action}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — CHECK DUPLICATE RECORDS
# ─────────────────────────────────────────────────────────────────────────────
def step6_duplicates(df: pd.DataFrame):
    section("STEP 6 — CHECK DUPLICATE RECORDS")

    dup_rows = df.duplicated().sum()
    dup_ids  = df["customer_id"].duplicated().sum()

    print(f"\n  Duplicate rows            : {dup_rows:,}")
    print(f"  Duplicate customer_ids    : {dup_ids:,}")

    if dup_rows == 0:
        print("\n  ✔  No duplicate rows found. Dataset is clean.")
    else:
        print(f"\n  ⚠  Found {dup_rows:,} duplicate rows. Sample:")
        print(df[df.duplicated()].head(5).to_string(index=False))
        print("\n  ANALYSIS:")
        print("    • Why might duplicates exist?")
        print("      – ETL pipeline bug (data loaded twice)")
        print("      – Multiple records from different source systems")
        print("      – Legitimate repeat entries (e.g., same action twice)")
        print("    • Are they valid? → Inspect if IDs differ (data errors) or same (true dups)")
        print("    • Should they be removed? → Yes, if rows are identical and IDs match.")

    if dup_ids == 0:
        print("\n  ✔  All customer_ids are unique. No ID-level duplicates.")
    else:
        print(f"\n  ⚠  {dup_ids:,} duplicate customer_ids found.")
        print("    → One customer may have multiple records (e.g., re-subscriptions).")
        print("    → Decide: keep latest record per customer, or aggregate.")

    # Bonus: check if any row is near-duplicate (same ID, different churn)
    if "customer_id" in df.columns:
        conflict = (
            df.groupby("customer_id")["churn"]
            .nunique()
            .reset_index()
        )
        conflict_count = (conflict["churn"] > 1).sum()
        print(f"\n  Customers with conflicting churn labels : {conflict_count:,}")
        if conflict_count == 0:
            print("  ✔  No conflicting churn labels detected.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + DIVIDER)
    print("  CUSTOMER CHURN — EXPLORATORY DATA ANALYSIS (Steps 2–6)")
    print(DIVIDER)

    # Load once, share across all steps
    print(f"\n  Loading: {RAW_DATA_PATH} ...")
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"  ✔  Loaded successfully.")

    step2_dataset_understanding(df)
    step3_load_dataset(df)
    step4_data_types(df)
    step5_missing_values(df)
    step6_duplicates(df)

    print(f"\n\n{DIVIDER}")
    print("  ✅  EDA Complete — Ready for preprocessing (src/data_preprocessing.py)")
    print(DIVIDER + "\n")
