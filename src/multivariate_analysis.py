"""
multivariate_analysis.py
------------------------
Step 3: Multivariate Analysis
  - Correlation matrix
  - Pivot table heatmaps (4 interactions)
  - Pair plots (20k sample)

Run: python src/multivariate_analysis.py
"""

import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({
    "figure.facecolor":"#0f1117","axes.facecolor":"#1a1a2e",
    "axes.edgecolor":"#3a3a5c","axes.labelcolor":"#e0e0e0",
    "xtick.color":"#b0b0c0","ytick.color":"#b0b0c0",
    "text.color":"#e0e0e0","grid.color":"#2a2a4a",
    "font.family":"sans-serif","axes.titlesize":12,
})
C0, C1  = "#6C63FF", "#FF6584"
FIG_DIR = os.path.join("reports","figures")
os.makedirs(FIG_DIR, exist_ok=True)
NUMERICAL = [
    "age","annual_income","tenure","monthlycharges","totalcharges",
    "customer_satisfaction","num_complaints","num_service_calls",
    "late_payments","avg_monthly_gb","days_since_last_interaction","credit_score",
]
TARGET  = "churn"
DIVIDER = "=" * 70
def section(t): print(f"\n{DIVIDER}\n  {t}\n{DIVIDER}")
def save_fig(n):
    p = os.path.join(FIG_DIR, n)
    plt.savefig(p, dpi=140, bbox_inches="tight", facecolor=plt.gcf().get_facecolor())
    plt.close("all"); print(f"  [saved] {p}")

print("\nLoading dataset...")
df = pd.read_csv(os.path.join("data","raw","customer_churn_1M.csv"))
print(f"  Shape: {df.shape[0]:,} x {df.shape[1]}")

# ── 1. CORRELATION MATRIX ────────────────────────────────────────────────────
section("1. Correlation Matrix")
corr_cols = NUMERICAL + [TARGET]
corr = df[corr_cols].corr()

churn_corr = corr[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
print(f"\n  {'Feature':<35} {'Corr with Churn':>16} {'Strength'}")
print("  " + "-" * 65)
for feat, val in churn_corr.items():
    strength = "STRONG" if abs(val)>0.3 else ("MODERATE" if abs(val)>0.1 else "WEAK")
    print(f"  {feat:<35} {val:>+16.4f}  {strength}")

fig, ax = plt.subplots(figsize=(14, 11))
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask, k=1)] = True
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
            cmap=sns.diverging_palette(250,10,as_cmap=True),
            center=0, linewidths=0.4, linecolor="#1a1a2e",
            annot_kws={"size":8}, ax=ax, cbar_kws={"shrink":0.8})
ax.set_title("Correlation Matrix — Numerical Features + Churn", fontsize=14, color="#a78bfa", pad=15)
plt.xticks(rotation=40, ha="right", fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout(); save_fig("multi_correlation_matrix.png")

# ── 2. PIVOT TABLE HEATMAPS ──────────────────────────────────────────────────
section("2. Pivot Heatmaps — Churn Rate by 2 Features")

def pivot_heatmap(df_, row_feat, col_feat, title, bins_row=None, bins_col=None,
                  labels_row=None, labels_col=None, figname=None):
    tmp = df_.copy()
    if bins_row: tmp[row_feat] = pd.cut(tmp[row_feat], bins=bins_row, labels=labels_row)
    if bins_col: tmp[col_feat] = pd.cut(tmp[col_feat], bins=bins_col, labels=labels_col)
    pivot = tmp.groupby([row_feat, col_feat])[TARGET].mean().unstack() * 100
    print(f"\n  {title}")
    print(pivot.round(2).to_string())
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd",
                linewidths=0.4, linecolor="#1a1a2e",
                annot_kws={"size":9}, ax=ax, cbar_kws={"label":"Churn Rate (%)"})
    ax.set_title(title, fontsize=13, color="#a78bfa", pad=12)
    plt.tight_layout()
    if figname: save_fig(figname)

pivot_heatmap(df, "contract", "tenure",
    title="Churn Rate (%) — Contract Type x Tenure Groups",
    bins_col=[0,12,24,48,72], labels_col=["0-12mo","12-24mo","24-48mo","48-72mo"],
    figname="multi_pivot_contract_tenure.png")

pivot_heatmap(df, "annual_income", "monthlycharges",
    title="Churn Rate (%) — Income Bracket x Monthly Charges",
    bins_row=[0,30000,60000,100000,250001], labels_row=["<30K","30-60K","60-100K",">100K"],
    bins_col=[0,50,75,100,855], labels_col=["<$50","$50-75","$75-100",">$100"],
    figname="multi_pivot_income_charges.png")

pivot_heatmap(df, "customer_satisfaction", "num_complaints",
    title="Churn Rate (%) — Satisfaction x Complaints",
    bins_row=[0,3,6,9], labels_row=["Low(1-3)","Mid(4-6)","High(7-9)"],
    bins_col=[-0.1,0,1,2,7], labels_col=["0","1","2","3+"],
    figname="multi_pivot_satisfaction_complaints.png")

# Internet + Tech Support pivot
tmp2 = df.copy()
tmp2["has_internet_service"] = tmp2["has_internet_service"].map({0:"No Internet",1:"Has Internet"})
tmp2["has_tech_support"]     = tmp2["has_tech_support"].map({0:"No Tech Support",1:"Has Tech Support"})
pt = tmp2.groupby(["has_internet_service","has_tech_support"])[TARGET].mean().unstack() * 100
print(f"\n  Internet x Tech Support Churn Rate:\n{pt.round(2).to_string()}")
fig, ax = plt.subplots(figsize=(7,4))
sns.heatmap(pt, annot=True, fmt=".2f", cmap="YlOrRd", linewidths=0.4,
            ax=ax, annot_kws={"size":12}, cbar_kws={"label":"Churn Rate (%)"})
ax.set_title("Churn Rate (%) — Internet Service x Tech Support", fontsize=12, color="#a78bfa")
plt.tight_layout(); save_fig("multi_pivot_internet_techsupport.png")

# ── 3. PAIR PLOT ─────────────────────────────────────────────────────────────
section("3. Pair Plot — Key Features (20k sample)")
key_feats = ["tenure","monthlycharges","customer_satisfaction","num_complaints","credit_score",TARGET]
SMALL = df[key_feats].sample(n=20_000, random_state=42).dropna()
SMALL[TARGET] = SMALL[TARGET].astype(str)
print("  Generating pair plot...")
g = sns.pairplot(SMALL, hue=TARGET, palette={"0":C0,"1":C1},
                 plot_kws={"alpha":0.2,"s":8,"edgecolors":"none"}, diag_kind="kde")
g.figure.suptitle("Pair Plot — Key Features (20k sample)", y=1.01, fontsize=13, color="#a78bfa")
g.figure.patch.set_facecolor("#0f1117")
for ax in g.axes.flatten():
    if ax: ax.set_facecolor("#1a1a2e")
save_fig("multi_pairplot.png")

section("MULTIVARIATE COMPLETE")
for f in sorted(os.listdir(FIG_DIR)):
    if f.startswith("multi_"): print(f"  reports/figures/{f}")
