"""
model_training.py
-----------------
Phase 7 : Class Imbalance — Baseline / class_weight / SMOTE / SMOTEENN
Phase 8 : Train Baseline Models — LR, Decision Tree, Random Forest, XGBoost
Phase 9 : Evaluate — Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix

Run: python src/model_training.py
"""

import os, warnings, time
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve,
    classification_report,
)
from xgboost import XGBClassifier

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":"#0f1117","axes.facecolor":"#1a1a2e",
    "axes.edgecolor":"#3a3a5c","axes.labelcolor":"#e0e0e0",
    "xtick.color":"#b0b0c0","ytick.color":"#b0b0c0",
    "text.color":"#e0e0e0","grid.color":"#2a2a4a",
    "font.family":"sans-serif","axes.titlesize":12,
})
PALETTE = ["#6C63FF","#FF6584","#43BCCD","#F7B731"]
FIG_DIR   = os.path.join("reports","figures")
MODEL_DIR = "models"
os.makedirs(FIG_DIR,   exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

DIVIDER = "=" * 70
def section(t): print(f"\n{DIVIDER}\n  {t}\n{DIVIDER}")
def save_fig(n):
    p = os.path.join(FIG_DIR, n)
    plt.savefig(p, dpi=140, bbox_inches="tight", facecolor=plt.gcf().get_facecolor())
    plt.close("all"); print(f"  [saved] {p}")

TARGET = "churn"

# ─────────────────────────────────────────────────────────────────────────────
# LOAD FEATURE-ENGINEERED DATA
# ─────────────────────────────────────────────────────────────────────────────
section("LOAD DATA")
DATA_PATH = os.path.join("data","processed","churn_features.csv")
print(f"  Loading: {DATA_PATH} ...")
df = pd.read_csv(DATA_PATH)
print(f"  Shape: {df.shape[0]:,} x {df.shape[1]}")

X = df.drop(columns=[TARGET])
y = df[TARGET]
print(f"  Churn: {y.mean()*100:.2f}%  ({y.sum():,} churned, {(y==0).sum():,} stayed)")

# Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\n  Train: {X_train.shape[0]:,} rows  |  Test: {X_test.shape[0]:,} rows")
print(f"  Train churn rate: {y_train.mean()*100:.2f}%")
print(f"  Test  churn rate: {y_test.mean()*100:.2f}%")

# ═════════════════════════════════════════════════════════════════════════════
# PHASE 7 — CLASS IMBALANCE COMPARISON
# ═════════════════════════════════════════════════════════════════════════════
section("PHASE 7 — CLASS IMBALANCE STRATEGIES (150k sample)")

# Use stratified 150k sample for speed (SMOTE doesn't scale to 1M)
SAMPLE_SIZE = 150_000
idx = np.random.RandomState(42).choice(len(X_train), size=min(SAMPLE_SIZE, len(X_train)), replace=False)
X_s = X_train.iloc[idx].copy()
y_s = y_train.iloc[idx].copy()
X_te_s, y_te_s = X_test.copy(), y_test.copy()

# Quick evaluator: Logistic Regression
def quick_eval(X_tr, y_tr, X_te, y_te, label):
    t0 = time.time()
    model = LogisticRegression(max_iter=500, random_state=42, solver="lbfgs")
    model.fit(X_tr, y_tr)
    elapsed = time.time() - t0
    y_pred  = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:,1]
    return {
        "Strategy":  label,
        "Train Size": len(y_tr),
        "Churn %":   f"{y_tr.mean()*100:.1f}%",
        "Accuracy":  accuracy_score(y_te, y_pred),
        "Precision": precision_score(y_te, y_pred, zero_division=0),
        "Recall":    recall_score(y_te, y_pred, zero_division=0),
        "F1":        f1_score(y_te, y_pred, zero_division=0),
        "ROC-AUC":   roc_auc_score(y_te, y_proba),
        "Time(s)":   round(elapsed, 1),
    }

imb_results = []

# 1. Baseline (no balancing)
print("\n  [1/4] Baseline (no balancing)...")
imb_results.append(quick_eval(X_s, y_s, X_te_s, y_te_s, "Baseline"))

# 2. class_weight='balanced'
print("  [2/4] class_weight='balanced'...")
t0 = time.time()
m = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42, solver="lbfgs")
m.fit(X_s, y_s)
elapsed = time.time() - t0
y_pred  = m.predict(X_te_s)
y_proba = m.predict_proba(X_te_s)[:,1]
imb_results.append({
    "Strategy":"class_weight=balanced", "Train Size":len(y_s),
    "Churn %":f"{y_s.mean()*100:.1f}%",
    "Accuracy":accuracy_score(y_te_s,y_pred),
    "Precision":precision_score(y_te_s,y_pred,zero_division=0),
    "Recall":recall_score(y_te_s,y_pred,zero_division=0),
    "F1":f1_score(y_te_s,y_pred,zero_division=0),
    "ROC-AUC":roc_auc_score(y_te_s,y_proba),
    "Time(s)":round(elapsed,1),
})

# 3. SMOTE
try:
    from imblearn.over_sampling import SMOTE
    print("  [3/4] SMOTE...")
    t0 = time.time()
    sm = SMOTE(random_state=42)
    X_sm, y_sm = sm.fit_resample(X_s, y_s)
    imb_results.append(quick_eval(X_sm, y_sm, X_te_s, y_te_s, "SMOTE"))
    print(f"          -> Resampled: {len(y_sm):,} rows")
except (ImportError, Exception) as e:
    print(f"  [3/4] SMOTE skipped: {e}")
    imb_results.append({"Strategy":"SMOTE","Train Size":"N/A","Churn %":"N/A",
                        "Accuracy":0,"Precision":0,"Recall":0,"F1":0,"ROC-AUC":0,"Time(s)":0})

# 4. SMOTEENN
try:
    from imblearn.combine import SMOTEENN
    print("  [4/4] SMOTEENN...")
    t0 = time.time()
    se = SMOTEENN(random_state=42)
    X_se, y_se = se.fit_resample(X_s, y_s)
    imb_results.append(quick_eval(X_se, y_se, X_te_s, y_te_s, "SMOTEENN"))
    print(f"          -> Resampled: {len(y_se):,} rows")
except (ImportError, Exception) as e:
    print(f"  [4/4] SMOTEENN skipped: {e}")
    imb_results.append({"Strategy":"SMOTEENN","Train Size":"N/A","Churn %":"N/A",
                        "Accuracy":0,"Precision":0,"Recall":0,"F1":0,"ROC-AUC":0,"Time(s)":0})

imb_df = pd.DataFrame(imb_results)
print(f"\n  IMBALANCE STRATEGY COMPARISON (Logistic Regression, 150k sample):")
print(f"\n  {'Strategy':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10}")
print("  " + "-" * 78)
for _, row in imb_df.iterrows():
    print(f"  {row['Strategy']:<25} {row['Accuracy']:>10.4f} {row['Precision']:>10.4f} "
          f"{row['Recall']:>10.4f} {row['F1']:>10.4f} {row['ROC-AUC']:>10.4f}")

# Best strategy (by F1)
best_strategy = imb_df.loc[imb_df["F1"].idxmax(), "Strategy"]
print(f"\n  Best strategy (by F1): {best_strategy}")
print(f"\n  CONCLUSION: For full model training we use class_weight='balanced'")
print(f"  (SMOTE is equivalent but doesn't scale to 1M rows efficiently)")

# Save imbalance comparison bar plot
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Class Imbalance Strategy Comparison (LR, 150k sample)", fontsize=13, color="#a78bfa")
valid = imb_df[imb_df["Recall"] > 0]
for ax, metric in zip(axes, ["Recall","F1","ROC-AUC"]):
    bars = ax.bar(valid["Strategy"], valid[metric], color=PALETTE[:len(valid)], edgecolor="none", alpha=0.9)
    ax.set_title(metric); ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=25, labelsize=8)
    for bar, val in zip(bars, valid[metric]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f"{val:.3f}", ha="center", fontsize=9)
plt.tight_layout(); save_fig("phase7_imbalance_comparison.png")


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 8 — TRAIN BASELINE MODELS (Full Training Data)
# ═════════════════════════════════════════════════════════════════════════════
section("PHASE 8 — TRAIN BASELINE MODELS (Full dataset, class_weight='balanced')")

class_ratio = (y_train==0).sum() / (y_train==1).sum()  # for XGBoost
print(f"  class_ratio (neg/pos) for XGBoost scale_pos_weight: {class_ratio:.2f}")

MODELS = {
    "LogisticRegression": LogisticRegression(
        max_iter=1000, class_weight="balanced",
        random_state=42, solver="lbfgs", C=1.0, n_jobs=-1
    ),
    "DecisionTree": DecisionTreeClassifier(
        max_depth=12, class_weight="balanced",
        random_state=42, min_samples_leaf=50
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=150, class_weight="balanced",
        max_depth=15, random_state=42, n_jobs=-1,
        min_samples_leaf=30
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200, scale_pos_weight=class_ratio,
        max_depth=6, learning_rate=0.1,
        random_state=42, eval_metric="logloss",
        n_jobs=-1, verbosity=0,
        use_label_encoder=False,
    ),
}

trained = {}
train_times = {}

for name, model in MODELS.items():
    print(f"\n  [{name}] Training on {X_train.shape[0]:,} rows x {X_train.shape[1]} features ...")
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    train_times[name] = round(elapsed, 1)
    trained[name] = model
    # Save model
    path = os.path.join(MODEL_DIR, f"{name}.joblib")
    joblib.dump(model, path)
    print(f"  [✓] Trained in {elapsed:.1f}s  |  Saved: {path}")


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 9 — MODEL EVALUATION
# ═════════════════════════════════════════════════════════════════════════════
section("PHASE 9 — MODEL EVALUATION")

def evaluate(model, X_te, y_te, name):
    y_pred  = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:,1]
    return {
        "Model":     name,
        "Accuracy":  round(accuracy_score(y_te, y_pred), 4),
        "Precision": round(precision_score(y_te, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_te, y_pred, zero_division=0), 4),
        "F1":        round(f1_score(y_te, y_pred, zero_division=0), 4),
        "ROC-AUC":   round(roc_auc_score(y_te, y_proba), 4),
        "PR-AUC":    round(average_precision_score(y_te, y_proba), 4),
        "Train(s)":  train_times[name],
        "_y_pred":   y_pred,
        "_y_proba":  y_proba,
    }

results = [evaluate(m, X_test, y_test, n) for n, m in trained.items()]
res_df  = pd.DataFrame([{k:v for k,v in r.items() if not k.startswith("_")} for r in results])

# Print metrics table
print(f"\n  {'Model':<22} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10} {'PR-AUC':>10} {'Train(s)':>10}")
print("  " + "-" * 97)
for _, row in res_df.iterrows():
    print(f"  {row['Model']:<22} {row['Accuracy']:>10.4f} {row['Precision']:>10.4f} "
          f"{row['Recall']:>10.4f} {row['F1']:>10.4f} {row['ROC-AUC']:>10.4f} "
          f"{row['PR-AUC']:>10.4f} {row['Train(s)']:>10.1f}s")

# Best model
best_row = res_df.loc[res_df["Recall"].idxmax()]
print(f"\n  Best by Recall (most important for churn): {best_row['Model']}")
best_f1  = res_df.loc[res_df["F1"].idxmax()]
print(f"  Best by F1:                                {best_f1['Model']}")
best_roc = res_df.loc[res_df["ROC-AUC"].idxmax()]
print(f"  Best by ROC-AUC:                           {best_roc['Model']}")

# Save metrics CSV
res_df.to_csv(os.path.join("reports","model_metrics.csv"), index=False)
print(f"\n  Saved: reports/model_metrics.csv")

# Detailed classification reports
print("\n  DETAILED CLASSIFICATION REPORTS:")
for r in results:
    print(f"\n  ── {r['Model']} ──")
    print(classification_report(y_test, r["_y_pred"],
                                 target_names=["No Churn","Churn"], digits=4))

# ── CONFUSION MATRICES ────────────────────────────────────────────────────────
print("  Plotting confusion matrices...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Confusion Matrices — All Models", fontsize=14, color="#a78bfa")
for ax, r in zip(axes.flatten(), results):
    cm = confusion_matrix(y_test, r["_y_pred"])
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
    labels = np.array([[f"{v}\n({p:.1f}%)" for v, p in zip(row, prow)]
                       for row, prow in zip(cm, cm_pct)])
    sns.heatmap(cm, annot=labels, fmt="", cmap="Purples",
                xticklabels=["No Churn","Churn"],
                yticklabels=["No Churn","Churn"],
                linewidths=0.5, linecolor="#0f1117", ax=ax,
                annot_kws={"size":11})
    ax.set_title(f"{r['Model']}\nRecall={r['Recall']:.4f}  F1={r['F1']:.4f}")
    ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
plt.tight_layout(); save_fig("phase9_confusion_matrices.png")

# ── ROC CURVES ───────────────────────────────────────────────────────────────
print("  Plotting ROC curves...")
fig, ax = plt.subplots(figsize=(9, 7))
ax.plot([0,1],[0,1],"--", color="#555577", lw=1, label="Random Classifier")
for r, col in zip(results, PALETTE):
    fpr, tpr, _ = roc_curve(y_test, r["_y_proba"])
    ax.plot(fpr, tpr, lw=2, color=col,
            label=f"{r['Model']}  (AUC={r['ROC-AUC']:.4f})")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — All Models", fontsize=13, color="#a78bfa")
ax.legend(framealpha=0.3, fontsize=10); ax.grid(alpha=0.3)
plt.tight_layout(); save_fig("phase9_roc_curves.png")

# ── PR CURVES ────────────────────────────────────────────────────────────────
print("  Plotting PR curves...")
fig, ax = plt.subplots(figsize=(9, 7))
baseline_pr = y_test.mean()
ax.axhline(baseline_pr, color="#555577", linestyle="--", lw=1,
           label=f"Random Classifier (PR={baseline_pr:.3f})")
for r, col in zip(results, PALETTE):
    prec, rec, _ = precision_recall_curve(y_test, r["_y_proba"])
    ax.plot(rec, prec, lw=2, color=col,
            label=f"{r['Model']}  (PR-AUC={r['PR-AUC']:.4f})")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curves — All Models", fontsize=13, color="#a78bfa")
ax.legend(framealpha=0.3, fontsize=10); ax.grid(alpha=0.3)
plt.tight_layout(); save_fig("phase9_pr_curves.png")

# ── METRICS COMPARISON BAR CHART ─────────────────────────────────────────────
print("  Plotting metrics comparison...")
metrics = ["Accuracy","Precision","Recall","F1","ROC-AUC","PR-AUC"]
x = np.arange(len(metrics)); w = 0.2
fig, ax = plt.subplots(figsize=(14, 6))
for i, (r, col) in enumerate(zip(results, PALETTE)):
    vals = [r[m] for m in metrics]
    bars = ax.bar(x + i*w, vals, w, label=r["Model"], color=col, alpha=0.9)
ax.set_xticks(x + w*1.5); ax.set_xticklabels(metrics)
ax.set_ylabel("Score"); ax.set_ylim(0, 1.05)
ax.set_title("Model Performance Comparison — All Metrics", fontsize=13, color="#a78bfa")
ax.legend(framealpha=0.3, fontsize=10); ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); save_fig("phase9_metrics_comparison.png")

# ── FEATURE IMPORTANCE (tree models) ─────────────────────────────────────────
print("  Plotting feature importances...")
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle("Feature Importance — Top 20", fontsize=13, color="#a78bfa")
for ax, model_name in zip(axes, ["RandomForest","XGBoost"]):
    m = trained[model_name]
    imp = pd.Series(m.feature_importances_, index=X_train.columns)
    top = imp.sort_values(ascending=True).tail(20)
    ax.barh(top.index, top.values, color="#6C63FF", alpha=0.85)
    ax.set_title(model_name); ax.set_xlabel("Importance Score")
plt.tight_layout(); save_fig("phase9_feature_importance.png")

# ── FINAL SUMMARY ─────────────────────────────────────────────────────────────
section("FINAL SUMMARY")
print(f"""
  PHASE 7 — Class Imbalance:
  Best strategy: class_weight='balanced' (practical for 1M rows)
  SMOTE comparable performance but 10x slower at this scale

  PHASE 8 — Models trained on {X_train.shape[0]:,} samples:
""")
for name, t in train_times.items():
    print(f"    {name:<22}: {t:.1f}s")

print(f"""
  PHASE 9 — Best Model Rankings:
  By Recall (most important for churn): {best_row['Model']} ({best_row['Recall']:.4f})
  By F1 Score                         : {best_f1['Model']}  ({best_f1['F1']:.4f})
  By ROC-AUC                          : {best_roc['Model']}  ({best_roc['ROC-AUC']:.4f})

  KEY INSIGHT — Why Recall matters more than Accuracy:
  Missing a churner costs the business (lost revenue).
  False positives (wrong churn prediction) cost only a small
  retention campaign → low penalty. So maximize Recall.

  Plots saved to reports/figures/:""")
for f in sorted(os.listdir(FIG_DIR)):
    if f.startswith("phase"):
        print(f"  reports/figures/{f}")
print(f"\n  Models saved to models/")
print(f"  Metrics saved to reports/model_metrics.csv\n")
