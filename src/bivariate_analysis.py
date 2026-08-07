"""
bivariate_analysis.py
---------------------
Step 2: Bivariate Analysis — Every feature vs Churn
  - Numerical vs Churn : Box plots + Violin plots + mean comparison
  - Categorical vs Churn: Churn rate bars + stacked bars
  - Binary vs Churn    : Churn rate per flag

Run: python src/bivariate_analysis.py
Plots saved: reports/figures/biv_*.png
"""

import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1117", "axes.facecolor": "#1a1a2e",
    "axes.edgecolor": "#3a3a5c",   "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#b0b0c0",      "ytick.color": "#b0b0c0",
    "text.color": "#e0e0e0",       "grid.color": "#2a2a4a",
    "grid.linewidth": 0.5,         "font.family": "sans-serif",
    "axes.titlesize": 12,          "axes.labelsize": 10,
})
C0, C1  = "#6C63FF", "#FF6584"          # No-Churn color, Churn color
PALETTE = {0: C0, 1: C1}
FIG_DIR = os.path.join("reports", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

NUMERICAL = [
    "age", "annual_income", "tenure", "monthlycharges", "totalcharges",
    "customer_satisfaction", "num_complaints", "num_service_calls",
    "late_payments", "avg_monthly_gb", "days_since_last_interaction", "credit_score",
]
CATEGORICAL = ["gender", "education", "marital_status", "contract", "payment_method", "paperless_billing"]
BINARY      = ["has_phone_service","has_internet_service","has_online_security","has_online_backup",
                "has_device_protection","has_tech_support","has_streaming_tv","has_streaming_movies","senior_citizen"]
TARGET = "churn"
DIVIDER = "=" * 70

def section(t): print(f"\n{DIVIDER}\n  {t}\n{DIVIDER}")
def save_fig(n):
    p = os.path.join(FIG_DIR, n)
    plt.savefig(p, dpi=140, bbox_inches="tight", facecolor=plt.gcf().get_facecolor())
    plt.close("all"); print(f"  [saved] {p}")


# ─── LOAD ─────────────────────────────────────────────────────────────────────
print("\nLoading dataset...")
df = pd.read_csv(os.path.join("data","raw","customer_churn_1M.csv"))
df["signup_date"] = pd.to_datetime(df["signup_date"])
print(f"  Shape: {df.shape[0]:,} x {df.shape[1]}")
# Small sample for violin/pair plots (speed)
SAMPLE = df.sample(n=80_000, random_state=42)


# ══════════════════════════════════════════════════════════════════════════════
# A. NUMERICAL vs CHURN
# ══════════════════════════════════════════════════════════════════════════════
section("A. Numerical Features vs Churn — Mean Comparison")

print(f"\n  {'Feature':<30} {'Mean (No Churn)':>18} {'Mean (Churn)':>15} {'Diff%':>8} {'Signal'}")
print("  " + "-" * 80)
insights = []
for feat in NUMERICAL:
    m0 = df.loc[df[TARGET]==0, feat].mean()
    m1 = df.loc[df[TARGET]==1, feat].mean()
    diff = (m1 - m0) / (m0 + 1e-9) * 100
    arrow = "↑ Churn higher" if diff > 5 else ("↓ Churn lower" if diff < -5 else "≈ Similar")
    insights.append((feat, m0, m1, diff, arrow))
    print(f"  {feat:<30} {m0:>18.3f} {m1:>15.3f} {diff:>+8.1f}% {arrow}")

# Box plots grid — full dataset stats, sampled for speed of rendering
section("A. Box Plots: Numerical vs Churn")
n_cols = 3; n_rows = (len(NUMERICAL) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows*3.8))
fig.suptitle("Numerical Features — Distribution by Churn", fontsize=14, color="#a78bfa")
axes_f = axes.flatten()
for i, feat in enumerate(NUMERICAL):
    ax = axes_f[i]
    groups = [SAMPLE.loc[SAMPLE[TARGET]==0, feat].dropna(),
              SAMPLE.loc[SAMPLE[TARGET]==1, feat].dropna()]
    bp = ax.boxplot(groups, patch_artist=True, widths=0.5,
                    boxprops=dict(linewidth=1.2),
                    medianprops=dict(linewidth=2.5, color="#F7B731"),
                    whiskerprops=dict(color="#b0b0c0"),
                    capprops=dict(color="#b0b0c0"),
                    flierprops=dict(marker=".", markersize=1.5, alpha=0.2))
    bp["boxes"][0].set_facecolor(C0)
    bp["boxes"][1].set_facecolor(C1)
    ax.set_title(feat); ax.set_xticks([1,2]); ax.set_xticklabels(["No Churn","Churn"])
    m0 = df.loc[df[TARGET]==0, feat].mean()
    m1 = df.loc[df[TARGET]==1, feat].mean()
    diff = (m1-m0)/(m0+1e-9)*100
    ax.set_xlabel(f"Δ={diff:+.1f}%", fontsize=8, color="#a0a0c0")
for j in range(i+1, len(axes_f)): axes_f[j].set_visible(False)
plt.tight_layout(); save_fig("biv_numerical_boxplots.png")

# Violin plots (sampled)
section("A. Violin Plots: Numerical vs Churn (80k sample)")
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows*3.8))
fig.suptitle("Numerical Features — Violin Plots by Churn (sampled)", fontsize=14, color="#a78bfa")
axes_f = axes.flatten()
for i, feat in enumerate(NUMERICAL):
    ax = axes_f[i]
    data_plot = SAMPLE[[feat, TARGET]].dropna()
    parts = ax.violinplot(
        [data_plot.loc[data_plot[TARGET]==0, feat],
         data_plot.loc[data_plot[TARGET]==1, feat]],
        positions=[1,2], showmedians=True, showextrema=False)
    for pc, col in zip(parts["bodies"], [C0, C1]):
        pc.set_facecolor(col); pc.set_alpha(0.75)
    parts["cmedians"].set_color("#F7B731"); parts["cmedians"].set_linewidth(2)
    ax.set_title(feat); ax.set_xticks([1,2]); ax.set_xticklabels(["No Churn","Churn"])
for j in range(i+1, len(axes_f)): axes_f[j].set_visible(False)
plt.tight_layout(); save_fig("biv_numerical_violins.png")


# ══════════════════════════════════════════════════════════════════════════════
# B. CATEGORICAL vs CHURN
# ══════════════════════════════════════════════════════════════════════════════
section("B. Categorical Features vs Churn — Churn Rate per Category")

print(f"\n  {'Feature':<20} {'Category':<25} {'Churn Rate %':>14} {'Count':>10}")
print("  " + "-" * 75)
cat_churn_rates = {}
for feat in CATEGORICAL:
    rates = df.groupby(feat)[TARGET].agg(["mean","count"]).reset_index()
    rates.columns = [feat, "churn_rate", "count"]
    rates["churn_rate"] *= 100
    rates = rates.sort_values("churn_rate", ascending=False)
    cat_churn_rates[feat] = rates
    for _, row in rates.iterrows():
        print(f"  {feat:<20} {str(row[feat]):<25} {row['churn_rate']:>13.2f}% {int(row['count']):>10,}")
    print()

# Churn rate bar charts per categorical feature
section("B. Churn Rate Bar Charts — Categorical")
n_cols = 2; n_rows = (len(CATEGORICAL) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows*4.5))
fig.suptitle("Churn Rate by Categorical Feature", fontsize=14, color="#a78bfa")
axes_f = axes.flatten()
for i, feat in enumerate(CATEGORICAL):
    ax = axes_f[i]
    rates = cat_churn_rates[feat].sort_values("churn_rate", ascending=True)
    colors = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, len(rates)))
    bars = ax.barh(rates[feat].astype(str), rates["churn_rate"], color=colors, edgecolor="none")
    ax.set_title(f"Churn Rate (%) — {feat}")
    ax.set_xlabel("Churn Rate (%)")
    ax.axvline(df[TARGET].mean()*100, color="#F7B731", linestyle="--", lw=1.5,
               label=f"Avg {df[TARGET].mean()*100:.1f}%")
    ax.legend(fontsize=8, framealpha=0.3)
    for bar, rate in zip(bars, rates["churn_rate"]):
        ax.text(rate+0.1, bar.get_y()+bar.get_height()/2,
                f"{rate:.1f}%", va="center", fontsize=9)
    ax.tick_params(axis="y", labelsize=9)
for j in range(i+1, len(axes_f)): axes_f[j].set_visible(False)
plt.tight_layout(); save_fig("biv_categorical_churnrates.png")

# Stacked bar charts
section("B. Stacked Bar Charts — Categorical")
fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows*4.5))
fig.suptitle("Churn vs No-Churn Composition by Category", fontsize=14, color="#a78bfa")
axes_f = axes.flatten()
for i, feat in enumerate(CATEGORICAL):
    ax = axes_f[i]
    ct = df.groupby([feat, TARGET]).size().unstack(fill_value=0)
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    labels = ct_pct.index.astype(str)
    ax.bar(labels, ct_pct[0], color=C0, label="No Churn", alpha=0.9)
    ax.bar(labels, ct_pct[1], bottom=ct_pct[0], color=C1, label="Churn", alpha=0.9)
    ax.set_title(feat); ax.set_ylabel("Percentage (%)"); ax.set_ylim(0, 110)
    ax.legend(fontsize=8, framealpha=0.3)
    ax.tick_params(axis="x", rotation=25)
    for j_idx, (x_pos, churn_pct) in enumerate(zip(range(len(labels)), ct_pct[1])):
        ax.text(x_pos, ct_pct[0].iloc[j_idx] + churn_pct/2,
                f"{churn_pct:.1f}%", ha="center", va="center",
                fontsize=8, color="white", fontweight="bold")
for j in range(i+1, len(axes_f)): axes_f[j].set_visible(False)
plt.tight_layout(); save_fig("biv_categorical_stacked.png")


# ══════════════════════════════════════════════════════════════════════════════
# C. BINARY vs CHURN
# ══════════════════════════════════════════════════════════════════════════════
section("C. Binary Features vs Churn — Churn Rate per Flag")

print(f"\n  {'Feature':<30} {'Churn% (=0)':>13} {'Churn% (=1)':>13} {'Diff':>8} {'Direction'}")
print("  " + "-" * 80)
bin_results = []
for feat in BINARY:
    r0 = df.loc[df[feat]==0, TARGET].mean() * 100
    r1 = df.loc[df[feat]==1, TARGET].mean() * 100
    diff = r1 - r0
    direction = "Service REDUCES churn" if diff < -1 else ("Service INCREASES churn" if diff > 1 else "≈ No effect")
    bin_results.append((feat, r0, r1, diff, direction))
    print(f"  {feat:<30} {r0:>12.2f}% {r1:>12.2f}% {diff:>+8.2f}% {direction}")

fig, ax = plt.subplots(figsize=(12, 6))
fig.suptitle("Binary Features — Churn Rate Comparison (=0 vs =1)", fontsize=14, color="#a78bfa")
x = np.arange(len(BINARY))
w = 0.35
r0s = [b[1] for b in bin_results]; r1s = [b[2] for b in bin_results]
ax.bar(x - w/2, r0s, w, label="Not subscribed (=0)", color=C0, alpha=0.9)
ax.bar(x + w/2, r1s, w, label="Subscribed (=1)", color=C1, alpha=0.9)
ax.set_xticks(x); ax.set_xticklabels(BINARY, rotation=40, ha="right", fontsize=8)
ax.set_ylabel("Churn Rate (%)"); ax.legend(framealpha=0.3)
ax.axhline(df[TARGET].mean()*100, color="#F7B731", linestyle="--", lw=1.5, label=f"Avg {df[TARGET].mean()*100:.1f}%")
plt.tight_layout(); save_fig("biv_binary_churnrates.png")


# ══════════════════════════════════════════════════════════════════════════════
# D. BUSINESS INSIGHTS SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
section("D. BUSINESS INSIGHTS — Who Churns the Most?")

print("""
  NUMERICAL FEATURES:
  ─────────────────────────────────────────────────────
  [Check the mean comparison table above for exact numbers]
  Key drivers expected: tenure, customer_satisfaction,
  num_complaints, late_payments, days_since_last_interaction

  CATEGORICAL FEATURES:
  ─────────────────────────────────────────────────────
  [See churn rate tables above for exact percentages]

  BINARY FEATURES:
  ─────────────────────────────────────────────────────
  Services that REDUCE churn → add value, customers stay
  Services that INCREASE churn → pricing issue or service quality

  TOP 3 BUSINESS QUESTIONS ANSWERED:
  1. Who churns the most? → See categorical churn rates
  2. Which customers are riskiest? → High complaints, low satisfaction,
     month-to-month contract, high daily charges
  3. Which services reduce churn? → See binary features table above
""")

section("BIVARIATE ANALYSIS COMPLETE — Plots saved to reports/figures/")
for f in sorted(os.listdir(FIG_DIR)):
    if f.startswith("biv_"):
        print(f"  reports/figures/{f}")
