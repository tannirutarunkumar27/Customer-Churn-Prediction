"""
eda_analysis.py
---------------
Full EDA script: Steps 2-6 (Today's tasks)
  Step 2 : Convert signup_date, inspect date range
  Step 3 : Separate features into Numerical / Categorical / Binary
  Step 4 : Target variable analysis
  Step 5 : Numerical features — distributions, box plots, stats
  Step 6 : Categorical features — counts, percentages

Run: python src/eda_analysis.py
Plots saved to: reports/figures/
"""

import os, sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#1a1a2e",
    "axes.edgecolor":   "#3a3a5c",
    "axes.labelcolor":  "#e0e0e0",
    "xtick.color":      "#b0b0c0",
    "ytick.color":      "#b0b0c0",
    "text.color":       "#e0e0e0",
    "grid.color":       "#2a2a4a",
    "grid.linewidth":   0.5,
    "font.family":      "sans-serif",
    "axes.titlesize":   13,
    "axes.labelsize":   11,
})
PALETTE_MAIN   = ["#6C63FF", "#FF6584"]
PALETTE_CAT    = ["#6C63FF","#FF6584","#43BCCD","#F7B731","#a29bfe","#fd79a8"]
FIG_DIR        = os.path.join("reports", "figures")
RAW_DATA_PATH  = os.path.join("data", "raw", "customer_churn_1M.csv")
DIVIDER        = "=" * 70

os.makedirs(FIG_DIR, exist_ok=True)


def section(title):
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


def save_fig(name):
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path, dpi=140, bbox_inches="tight", facecolor=plt.gcf().get_facecolor())
    plt.close("all")
    print(f"  [saved] {path}")


# ─────────────────────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────────────────────
print("\nLoading dataset...")
df = pd.read_csv(RAW_DATA_PATH)
print(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — CONVERT signup_date
# ═════════════════════════════════════════════════════════════════════════════
section("STEP 2 — Convert signup_date to datetime")

df["signup_date"] = pd.to_datetime(df["signup_date"])
print(f"\n  dtype after conversion : {df['signup_date'].dtype}")
print(f"  Earliest signup date   : {df['signup_date'].min()}")
print(f"  Latest signup date     : {df['signup_date'].max()}")
date_range = df["signup_date"].max() - df["signup_date"].min()
print(f"  Date range             : {date_range.days:,} days  (~{date_range.days // 365} years)")

# Extract date parts (saved for later use in feature engineering)
df["signup_year"]    = df["signup_date"].dt.year
df["signup_month"]   = df["signup_date"].dt.month
df["signup_quarter"] = df["signup_date"].dt.quarter
print(f"\n  Extracted: signup_year, signup_month, signup_quarter")
print(f"  Years present: {sorted(df['signup_year'].unique())}")
print(f"  Months range : {df['signup_month'].min()} – {df['signup_month'].max()}")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — SEPARATE FEATURES
# ═════════════════════════════════════════════════════════════════════════════
section("STEP 3 — Separate Features")

NUMERICAL = [
    "age", "annual_income", "tenure", "monthlycharges", "totalcharges",
    "customer_satisfaction", "num_complaints", "num_service_calls",
    "late_payments", "avg_monthly_gb", "days_since_last_interaction", "credit_score",
]
CATEGORICAL = [
    "gender", "education", "marital_status", "contract",
    "payment_method", "paperless_billing",
]
BINARY = [
    "has_phone_service", "has_internet_service", "has_online_security",
    "has_online_backup", "has_device_protection", "has_tech_support",
    "has_streaming_tv", "has_streaming_movies", "senior_citizen",
]
ID_COLS  = ["customer_id", "signup_date", "signup_year", "signup_month", "signup_quarter"]
TARGET   = "churn"

print(f"\n  Numerical   ({len(NUMERICAL)}): {NUMERICAL}")
print(f"\n  Categorical ({len(CATEGORICAL)}): {CATEGORICAL}")
print(f"\n  Binary      ({len(BINARY)}): {BINARY}")
print(f"\n  ID/Date     ({len(ID_COLS)}): {ID_COLS}")
print(f"\n  Target      : {TARGET}")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 4 — TARGET VARIABLE ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
section("STEP 4 — Target Variable Analysis (churn)")

vc   = df[TARGET].value_counts()
pct  = df[TARGET].value_counts(normalize=True) * 100
print(f"\n  {'Label':<12} {'Count':>10} {'Percentage':>12}")
print("  " + "-" * 38)
for label in [0, 1]:
    name = "No Churn" if label == 0 else "Churn"
    print(f"  {name:<12} {vc[label]:>10,} {pct[label]:>11.2f}%")

print(f"\n  OBSERVATION:")
print(f"  Around {pct[0]:.1f}% of customers stayed while ~{pct[1]:.1f}% churned.")
print(f"  This is a HIGHLY IMBALANCED dataset — must use stratified splits,")
print(f"  SMOTE/class_weight, and AUC/F1 (not accuracy) as evaluation metric.")

# Bar chart
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Target Variable: Churn Distribution", fontsize=15, color="#a78bfa", y=1.01)

labels_str = ["No Churn (0)", "Churn (1)"]
counts_    = [vc[0], vc[1]]
bars = axes[0].bar(labels_str, counts_, color=PALETTE_MAIN, width=0.5, edgecolor="none")
axes[0].set_title("Count")
axes[0].set_ylabel("Number of Customers")
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
for bar, cnt in zip(bars, counts_):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5000,
                 f"{cnt:,}", ha="center", fontsize=11, color="#e0e0e0")

wedges, texts, autotexts = axes[1].pie(
    [pct[0], pct[1]],
    labels=labels_str,
    colors=PALETTE_MAIN,
    autopct="%1.1f%%",
    startangle=140,
    wedgeprops=dict(edgecolor="#0f1117", linewidth=2),
)
for at in autotexts:
    at.set_color("#0f1117"); at.set_fontsize(12); at.set_fontweight("bold")
axes[1].set_title("Percentage")
plt.tight_layout()
save_fig("step4_churn_distribution.png")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 5 — NUMERICAL FEATURES
# ═════════════════════════════════════════════════════════════════════════════
section("STEP 5 — Numerical Features Analysis")

# Summary statistics table
stats_df = df[NUMERICAL].describe().T
stats_df["median"] = df[NUMERICAL].median()
stats_df["missing"] = df[NUMERICAL].isnull().sum()
stats_df["missing_%"] = (df[NUMERICAL].isnull().sum() / len(df) * 100).round(2)

print(f"\n  SUMMARY STATISTICS:")
print(f"\n  {'Feature':<30} {'Mean':>10} {'Median':>10} {'Std':>10} "
      f"{'Min':>10} {'Max':>10} {'Missing%':>10}")
print("  " + "-" * 95)
for feat in NUMERICAL:
    d = stats_df.loc[feat]
    print(f"  {feat:<30} {d['mean']:>10.2f} {d['median']:>10.2f} {d['std']:>10.2f} "
          f"{d['min']:>10.2f} {d['max']:>10.2f} {d['missing_%']:>9.2f}%")

# Histograms — all numerical in one grid
print("\n  Plotting histograms...")
n_cols = 3
n_rows = (len(NUMERICAL) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3.5))
fig.suptitle("Numerical Features — Distributions", fontsize=15, color="#a78bfa")
axes_flat = axes.flatten()
for i, feat in enumerate(NUMERICAL):
    ax = axes_flat[i]
    data = df[feat].dropna()
    ax.hist(data, bins=50, color="#6C63FF", edgecolor="none", alpha=0.85)
    ax.axvline(data.mean(),   color="#FF6584", lw=1.5, linestyle="--", label=f"Mean={data.mean():.1f}")
    ax.axvline(data.median(), color="#F7B731", lw=1.5, linestyle="-",  label=f"Median={data.median():.1f}")
    ax.set_title(feat)
    ax.legend(fontsize=7, framealpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
for j in range(i + 1, len(axes_flat)):
    axes_flat[j].set_visible(False)
plt.tight_layout()
save_fig("step5_numerical_histograms.png")

# Box plots — outlier detection
print("  Plotting box plots...")
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3.5))
fig.suptitle("Numerical Features — Box Plots (Outlier Detection)", fontsize=15, color="#a78bfa")
axes_flat = axes.flatten()
for i, feat in enumerate(NUMERICAL):
    ax = axes_flat[i]
    data = df[feat].dropna()
    bp = ax.boxplot(data, vert=True, patch_artist=True,
                    boxprops=dict(facecolor="#6C63FF", color="#a78bfa"),
                    medianprops=dict(color="#FF6584", linewidth=2),
                    whiskerprops=dict(color="#b0b0c0"),
                    capprops=dict(color="#b0b0c0"),
                    flierprops=dict(marker=".", color="#FF6584", alpha=0.3, markersize=2))
    # IQR-based outlier count
    Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
    IQR    = Q3 - Q1
    n_out  = ((data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR)).sum()
    ax.set_title(f"{feat}\n({n_out:,} outliers)", fontsize=10)
    ax.set_xticks([])
for j in range(i + 1, len(axes_flat)):
    axes_flat[j].set_visible(False)
plt.tight_layout()
save_fig("step5_numerical_boxplots.png")

# Outlier summary
print("\n  OUTLIER SUMMARY (IQR method):")
print(f"  {'Feature':<30} {'Q1':>8} {'Q3':>8} {'IQR':>8} {'Outliers':>10} {'%':>8}")
print("  " + "-" * 80)
for feat in NUMERICAL:
    data = df[feat].dropna()
    Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
    IQR    = Q3 - Q1
    n_out  = ((data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR)).sum()
    pct_   = n_out / len(data) * 100
    print(f"  {feat:<30} {Q1:>8.2f} {Q3:>8.2f} {IQR:>8.2f} {n_out:>10,} {pct_:>7.2f}%")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 6 — CATEGORICAL FEATURES
# ═════════════════════════════════════════════════════════════════════════════
section("STEP 6 — Categorical Features Analysis")

print(f"\n  {'Feature':<20} {'Value':<25} {'Count':>10} {'Percentage':>12}")
print("  " + "-" * 72)
for feat in CATEGORICAL:
    vc_   = df[feat].value_counts()
    pct_  = df[feat].value_counts(normalize=True) * 100
    for j, (val, cnt) in enumerate(vc_.items()):
        feat_label = feat if j == 0 else ""
        print(f"  {feat_label:<20} {str(val):<25} {cnt:>10,} {pct_[val]:>11.2f}%")
    print("  " + "  " * 34)

# Count plots grid
print("\n  Plotting categorical count plots...")
n_cols = 2
n_rows = (len(CATEGORICAL) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 4))
fig.suptitle("Categorical Features — Value Counts", fontsize=15, color="#a78bfa")
axes_flat = axes.flatten()
for i, feat in enumerate(CATEGORICAL):
    ax = axes_flat[i]
    vc_  = df[feat].value_counts()
    bars = ax.bar(vc_.index.astype(str), vc_.values,
                  color=PALETTE_CAT[:len(vc_)], edgecolor="none", alpha=0.9)
    ax.set_title(feat)
    ax.set_ylabel("Count")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.tick_params(axis="x", rotation=25)
    for bar, cnt in zip(bars, vc_.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vc_.values) * 0.01,
                f"{cnt / len(df) * 100:.1f}%", ha="center", fontsize=9, color="#e0e0e0")
for j in range(i + 1, len(axes_flat)):
    axes_flat[j].set_visible(False)
plt.tight_layout()
save_fig("step6_categorical_countplots.png")

# Binary features summary
section("STEP 6b — Binary Features Analysis")
print(f"\n  {'Feature':<25} {'=0 Count':>12} {'=0 %':>8} {'=1 Count':>12} {'=1 %':>8}")
print("  " + "-" * 72)
for feat in BINARY + [TARGET]:
    c0 = (df[feat] == 0).sum()
    c1 = (df[feat] == 1).sum()
    p0 = c0 / len(df) * 100
    p1 = c1 / len(df) * 100
    print(f"  {feat:<25} {c0:>12,} {p0:>7.2f}% {c1:>12,} {p1:>7.2f}%")

# Binary stacked bar
print("\n  Plotting binary feature chart...")
fig, ax = plt.subplots(figsize=(13, 5))
fig.suptitle("Binary Features — Adoption Rate (% = 1)", fontsize=14, color="#a78bfa")
all_binary = BINARY + [TARGET]
rates = [(df[f] == 1).mean() * 100 for f in all_binary]
bars  = ax.barh(all_binary, rates, color="#6C63FF", edgecolor="none", alpha=0.85)
ax.barh(all_binary, [100 - r for r in rates], left=rates, color="#2a2a4a", edgecolor="none", alpha=0.5)
for bar, rate in zip(bars, rates):
    ax.text(rate + 1, bar.get_y() + bar.get_height() / 2,
            f"{rate:.1f}%", va="center", fontsize=10)
ax.set_xlim(0, 110)
ax.set_xlabel("Adoption Rate (%)")
ax.axvline(50, color="#FF6584", lw=1, linestyle="--", alpha=0.5, label="50%")
ax.legend()
plt.tight_layout()
save_fig("step6_binary_adoption_rates.png")

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
section("SUMMARY — All Plots Saved")
for f in sorted(os.listdir(FIG_DIR)):
    print(f"  reports/figures/{f}")

print(f"""
{DIVIDER}
  TODAY'S TASKS COMPLETE:
  [x] signup_date converted to datetime
  [x] Features separated (Numerical / Categorical / Binary)
  [x] Target variable analyzed (churn distribution)
  [x] All numerical features — histograms + box plots + stats
  [x] All categorical features — counts + percentages + plots
  [x] Binary features adoption rates plotted
{DIVIDER}
""")
