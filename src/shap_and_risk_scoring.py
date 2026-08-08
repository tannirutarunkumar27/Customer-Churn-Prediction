"""
shap_and_risk_scoring.py
------------------------
Step 1: SHAP Explainable AI
  - SHAP Summary Plot (global feature importance)
  - SHAP Bar Plot     (mean |SHAP| per feature)
  - SHAP Force Plot   (single prediction explanation, saved as HTML)
  - Business insight text

Step 2: Customer Risk Scoring
  - Assign churn probability to every test customer
  - Segment into Low / Medium / High / Critical risk tiers
  - Print top at-risk customers and save CSV report

Run: python src/shap_and_risk_scoring.py
"""
import os, warnings, time
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from imblearn.under_sampling import RandomUnderSampler

# ── dirs ──────────────────────────────────────────────────────────────────────
FIG_DIR    = os.path.join("reports", "figures")
MODEL_DIR  = "models"
REPORT_DIR = "reports"
os.makedirs(FIG_DIR,    exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "#0f1117", "axes.facecolor": "#1a1a2e",
    "axes.edgecolor":   "#3a3a5c", "axes.labelcolor": "#e0e0e0",
    "xtick.color":      "#b0b0c0", "ytick.color":     "#b0b0c0",
    "text.color":       "#e0e0e0", "grid.color":      "#2a2a4a",
    "font.family":      "sans-serif",
})
PAL = ["#6C63FF", "#FF6584", "#43BCCD", "#F7B731", "#00b894"]

D = "=" * 70
def sec(t):  print(f"\n{D}\n  {t}\n{D}")
def sfig(n):
    p = os.path.join(FIG_DIR, n)
    plt.savefig(p, dpi=140, bbox_inches="tight", facecolor=plt.gcf().get_facecolor())
    plt.close("all")
    print(f"  [saved] {p}")

TARGET = "churn"

# ===================================================
# LOAD & PREPARE DATA  (same pipeline as training)
# ===================================================
sec("LOAD DATA & PREPARE (same pipeline as final_optimization.py)")
DATA_PATH = os.path.join("data", "processed", "churn_advanced.csv")
df = pd.read_csv(DATA_PATH)

# Same noise-reduction as final_optimization.py
corr = df.corr()[TARGET].drop(TARGET).abs()
weak = corr[corr < 0.01].index.tolist()
df.drop(columns=weak, inplace=True)
print(f"  Kept {df.shape[1]-1} features after noise reduction (dropped {len(weak)})")

X = df.drop(columns=[TARGET])
y = df[TARGET]
FEATURE_NAMES = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

rus = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
X_train_res, y_train_res = rus.fit_resample(X_train, y_train)
print(f"  Train (resampled): {X_train_res.shape[0]:,} | Test: {X_test.shape[0]:,}")

# ===================================================
# TRAIN BEST MODEL  (HistGB — highest ROC-AUC)
# or load from disk if already saved
# ===================================================
sec("TRAIN / LOAD MODEL")
MODEL_PATH = os.path.join(MODEL_DIR, "final_HistGB.joblib")

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print(f"  Loaded existing model: {MODEL_PATH}")
else:
    from sklearn.ensemble import HistGradientBoostingClassifier
    model = HistGradientBoostingClassifier(
        max_iter=500, max_depth=9, learning_rate=0.03,
        min_samples_leaf=40, l2_regularization=0.2,
        random_state=42,
    )
    t0 = time.time()
    model.fit(X_train_res, y_train_res)
    print(f"  Trained in {time.time()-t0:.1f}s")
    joblib.dump(model, MODEL_PATH)

yproba = model.predict_proba(X_test)[:, 1]
ypred  = (yproba >= 0.50).astype(int)
print(f"  Accuracy : {accuracy_score(y_test, ypred):.4f}")
print(f"  Recall   : {recall_score(y_test, ypred):.4f}")
print(f"  ROC-AUC  : {roc_auc_score(y_test, yproba):.4f}")

# ===================================================
# STEP 1 — SHAP  (sample 5k for speed)
# ===================================================
sec("STEP 1: SHAP EXPLAINABLE AI")

SHAP_SAMPLE = 5_000
idx_shap = np.random.RandomState(42).choice(len(X_test), SHAP_SAMPLE, replace=False)
X_shap   = X_test.iloc[idx_shap].reset_index(drop=True)

print(f"  Computing SHAP values on {SHAP_SAMPLE:,} test samples ...")
t0 = time.time()
explainer   = shap.Explainer(model, X_shap, feature_names=FEATURE_NAMES)
shap_values = explainer(X_shap, check_additivity=False)
print(f"  SHAP computation done in {time.time()-t0:.1f}s")

# ── 1a. SHAP Summary Plot (beeswarm) ─────────────────────────────────────────
print("\n  Generating SHAP Summary Plot (beeswarm) ...")
fig, ax = plt.subplots(figsize=(12, 10))
shap.summary_plot(
    shap_values, X_shap,
    plot_type="dot",
    max_display=20,
    show=False,
    color_bar_label="Feature Value",
)
plt.title("SHAP Summary Plot — Top 20 Features", color="#a78bfa", fontsize=14, pad=14)
plt.tight_layout()
sfig("shap_summary_beeswarm.png")

# ── 1b. SHAP Bar Plot (mean |SHAP| = global importance) ──────────────────────
print("  Generating SHAP Bar Plot (global feature importance) ...")
shap_mean = np.abs(shap_values.values).mean(axis=0)
shap_importance = pd.Series(shap_mean, index=FEATURE_NAMES).sort_values(ascending=True).tail(20)

fig, ax = plt.subplots(figsize=(11, 9))
colors = [PAL[0]] * len(shap_importance)
colors[-1] = PAL[1]   # top feature highlighted
ax.barh(shap_importance.index, shap_importance.values, color=colors, alpha=0.88)
ax.set_xlabel("Mean |SHAP Value|  (average impact on model output)", fontsize=11)
ax.set_title("SHAP Global Feature Importance — Top 20", color="#a78bfa", fontsize=14)
ax.axvline(shap_importance.values.mean(), color="white", lw=1.2, linestyle="--", alpha=0.5, label="Mean")
ax.legend(framealpha=0.3, fontsize=9)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
sfig("shap_bar_importance.png")

# ── 1c. SHAP Force Plot (HTML for a churn sample) ────────────────────────────
print("  Generating SHAP Force Plots (HTML) for sample predictions ...")
actual_churn_idx = np.where(y_test.iloc[idx_shap].values == 1)[0]
if len(actual_churn_idx) > 0:
    sample_idx = actual_churn_idx[0]   # first true churner in SHAP subset
else:
    sample_idx = 0

force_html_path = os.path.join(FIG_DIR, "shap_force_plot_churn_sample.html")
shap.initjs()
force_plot = shap.force_plot(
    explainer.expected_value,
    shap_values.values[sample_idx],
    X_shap.iloc[sample_idx],
    feature_names=FEATURE_NAMES,
    matplotlib=False,
)
shap.save_html(force_html_path, force_plot)
print(f"  [saved] {force_html_path}")

# ── 1d. Waterfall Plot for the same sample ───────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 9))
shap.plots.waterfall(shap_values[sample_idx], max_display=15, show=False)
plt.title(f"SHAP Waterfall Plot — Churner (Sample #{sample_idx})", color="#a78bfa", fontsize=13)
plt.tight_layout()
sfig("shap_waterfall_churn_sample.png")

# ── 1e. Business Insight Text ─────────────────────────────────────────────────
top5_features = pd.Series(shap_mean, index=FEATURE_NAMES).sort_values(ascending=False).head(5)
print("\n" + "─" * 65)
print("  BUSINESS INSIGHT:")
print("  The top drivers of churn in this model are:")
for i, (feat, val) in enumerate(top5_features.items(), 1):
    print(f"    {i}. {feat:<40}  (SHAP impact: {val:.4f})")
top_names = ", ".join(top5_features.index[:3].tolist())
print(f"\n  Summary: '{top_names}, and other key factors")
print(f"  were the strongest contributors to predicted churn.'")
print("─" * 65)

# ── 1f. Save SHAP importance to CSV ──────────────────────────────────────────
shap_df = pd.Series(shap_mean, index=FEATURE_NAMES)\
            .sort_values(ascending=False)\
            .reset_index()
shap_df.columns = ["feature", "mean_abs_shap"]
shap_df.to_csv(os.path.join(REPORT_DIR, "shap_feature_importance.csv"), index=False)
print(f"\n  [saved] reports/shap_feature_importance.csv")

# ===================================================
# STEP 2 — CUSTOMER RISK SCORING
# ===================================================
sec("STEP 2: CUSTOMER RISK SCORING")

# Build risk score table from the full test set
risk_df = pd.DataFrame({
    "churn_probability": np.round(yproba, 4),
    "actual_churn":      y_test.values,
})
risk_df.index = range(len(risk_df))

# Assign risk tiers
def assign_risk(p):
    if p >= 0.70: return "Critical"
    elif p >= 0.50: return "High"
    elif p >= 0.30: return "Medium"
    else:           return "Low"

risk_df["risk_tier"] = risk_df["churn_probability"].apply(assign_risk)
risk_df["risk_tier"] = pd.Categorical(
    risk_df["risk_tier"],
    categories=["Low", "Medium", "High", "Critical"],
    ordered=True,
)

# ── Print segment summary ─────────────────────────────────────────────────────
print(f"\n  {'Risk Tier':<12} {'Customers':>12} {'Pct of Base':>13} {'Actual Churn%':>15}")
print("  " + "-" * 56)
total = len(risk_df)
for tier in ["Critical", "High", "Medium", "Low"]:
    sub = risk_df[risk_df["risk_tier"] == tier]
    pct_base   = len(sub) / total * 100
    churn_rate = sub["actual_churn"].mean() * 100
    print(f"  {tier:<12} {len(sub):>12,} {pct_base:>12.2f}% {churn_rate:>14.2f}%")

# ── Top 10 Critical customers ─────────────────────────────────────────────────
top_risk = risk_df.sort_values("churn_probability", ascending=False).head(10).copy()
top_risk.index = [f"Customer_{i}" for i in range(1, len(top_risk)+1)]
print(f"\n  Top 10 Highest-Risk Customers:")
print(f"  {'Customer':<14} {'Churn Prob':>12} {'Risk Tier':>12} {'Actual Churn':>14}")
print("  " + "-" * 56)
for cust, row in top_risk.iterrows():
    actual = "YES" if row["actual_churn"] == 1 else "NO"
    print(f"  {cust:<14} {row['churn_probability']:>12.4f} {row['risk_tier']:>12} {actual:>14}")

# ── Save full risk report ─────────────────────────────────────────────────────
risk_df.to_csv(os.path.join(REPORT_DIR, "customer_risk_scores.csv"), index=True)
print(f"\n  [saved] reports/customer_risk_scores.csv  ({len(risk_df):,} customers scored)")

# ── Risk Tier Distribution Bar Chart ─────────────────────────────────────────
tier_counts = risk_df["risk_tier"].value_counts().reindex(["Critical","High","Medium","Low"])
tier_colors = [PAL[1], PAL[0], PAL[2], PAL[4]]

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("Customer Risk Segmentation Dashboard", color="#a78bfa", fontsize=15, y=1.01)

# Bar chart — count per tier
axes[0].bar(tier_counts.index, tier_counts.values, color=tier_colors, alpha=0.9, width=0.55)
for i, (k, v) in enumerate(tier_counts.items()):
    axes[0].text(i, v + tier_counts.max()*0.01, f"{v:,}", ha="center", va="bottom",
                 color="white", fontsize=10, fontweight="bold")
axes[0].set_xlabel("Risk Tier"); axes[0].set_ylabel("Number of Customers")
axes[0].set_title("Customers per Risk Tier", color="#e0e0e0", fontsize=12)
axes[0].grid(axis="y", alpha=0.3)

# Bar chart — actual churn rate per tier
churn_by_tier = risk_df.groupby("risk_tier", observed=False)["actual_churn"].mean() * 100
churn_by_tier = churn_by_tier.reindex(["Critical","High","Medium","Low"])
axes[1].bar(churn_by_tier.index, churn_by_tier.values, color=tier_colors, alpha=0.9, width=0.55)
for i, v in enumerate(churn_by_tier.values):
    axes[1].text(i, v + 0.5, f"{v:.1f}%", ha="center", va="bottom",
                 color="white", fontsize=10, fontweight="bold")
axes[1].set_xlabel("Risk Tier"); axes[1].set_ylabel("Actual Churn Rate (%)")
axes[1].set_title("Actual Churn Rate per Risk Tier", color="#e0e0e0", fontsize=12)
axes[1].grid(axis="y", alpha=0.3)

plt.tight_layout()
sfig("risk_segmentation_dashboard.png")

# ── Churn Probability Distribution per Tier ───────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 6))
tier_order = ["Critical","High","Medium","Low"]
for tier, col in zip(tier_order, tier_colors):
    sub = risk_df[risk_df["risk_tier"] == tier]["churn_probability"]
    ax.hist(sub, bins=40, color=col, alpha=0.65, label=f"{tier} ({len(sub):,})", density=True)
ax.set_xlabel("Predicted Churn Probability", fontsize=11)
ax.set_ylabel("Density")
ax.set_title("Churn Probability Distribution by Risk Tier", color="#a78bfa", fontsize=13)
ax.axvline(0.30, color="white", lw=1, linestyle=":", alpha=0.6)
ax.axvline(0.50, color="white", lw=1, linestyle=":", alpha=0.6)
ax.axvline(0.70, color="white", lw=1, linestyle=":", alpha=0.6)
ax.legend(framealpha=0.3, fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
sfig("risk_probability_distribution.png")

# ===================================================
# FINAL SUMMARY
# ===================================================
sec("FINAL SUMMARY")
print(f"""
  SHAP Explainability Outputs:
    reports/figures/shap_summary_beeswarm.png     (global patterns)
    reports/figures/shap_bar_importance.png        (feature rankings)
    reports/figures/shap_waterfall_churn_sample.png(individual prediction)
    reports/figures/shap_force_plot_churn_sample.html (interactive force plot)
    reports/shap_feature_importance.csv            (all features ranked)

  Customer Risk Scoring Outputs:
    reports/customer_risk_scores.csv               ({len(risk_df):,} customers)
    reports/figures/risk_segmentation_dashboard.png
    reports/figures/risk_probability_distribution.png

  Risk Tier Summary:
    - Critical (>= 0.70 prob): {(risk_df['risk_tier']=='Critical').sum():>8,} customers
    - High     (0.50 - 0.70):  {(risk_df['risk_tier']=='High').sum():>8,} customers
    - Medium   (0.30 - 0.50):  {(risk_df['risk_tier']=='Medium').sum():>8,} customers
    - Low      (<  0.30 prob): {(risk_df['risk_tier']=='Low').sum():>8,} customers

  Top 3 churn drivers (SHAP):
    {chr(10).join([f"    {i+1}. {f}" for i, f in enumerate(top5_features.index[:3].tolist())])}
""")
print("[OK] shap_and_risk_scoring.py COMPLETE")
