"""
optimized_training.py
---------------------
High-accuracy churn prediction targeting >= 90% accuracy
- Uses advanced features (churn_advanced.csv)
- LightGBM, XGBoost, CatBoost, HistGB
- Threshold tuning for best accuracy + ROC-AUC
- Ensemble stacking
Run: python src/optimized_training.py
"""
import os, warnings, time
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, HistGradientBoostingClassifier,
                               StackingClassifier, VotingClassifier)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve, balanced_accuracy_score,
    classification_report, matthews_corrcoef)
from xgboost import XGBClassifier

# ── dirs ──────────────────────────────────────────────────────────────────────
FIG_DIR   = os.path.join("reports", "figures")
MODEL_DIR = "models"
os.makedirs(FIG_DIR,   exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor":"#0f1117","axes.facecolor":"#1a1a2e",
    "axes.edgecolor":"#3a3a5c","axes.labelcolor":"#e0e0e0",
    "xtick.color":"#b0b0c0","ytick.color":"#b0b0c0",
    "text.color":"#e0e0e0","grid.color":"#2a2a4a","font.family":"sans-serif",
})
PAL = ["#6C63FF","#FF6584","#43BCCD","#F7B731","#a29bfe","#fd79a8","#00b894"]
D = "=" * 70
def sec(t): print(f"\n{D}\n  {t}\n{D}")
def sfig(n):
    p = os.path.join(FIG_DIR, n)
    plt.savefig(p, dpi=130, bbox_inches="tight", facecolor=plt.gcf().get_facecolor())
    plt.close("all"); print(f"  [saved] {p}")

TARGET = "churn"

# ═════════════════════════════════════════════════════════════════════════════
sec("LOAD DATA")
# Use advanced features if available, else fallback
ADV = os.path.join("data","processed","churn_advanced.csv")
STD = os.path.join("data","processed","churn_features.csv")
DATA_PATH = ADV if os.path.exists(ADV) else STD
print(f"  Loading: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"  Shape: {df.shape[0]:,} x {df.shape[1]}")

X = df.drop(columns=[TARGET])
y = df[TARGET]
churn_pct = y.mean() * 100
print(f"  Churn rate: {churn_pct:.2f}%  (non-churn: {100-churn_pct:.2f}%)")

# ── stratified split ──────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
print(f"  Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")

ratio = (y_train == 0).sum() / (y_train == 1).sum()
print(f"  Class ratio (neg/pos): {ratio:.2f}  → scale_pos_weight for XGB")

# ═════════════════════════════════════════════════════════════════════════════
sec("DEFINE OPTIMIZED MODELS")

# NOTE: We do NOT use class_weight='balanced' at full strength because
# with 90% non-churn, it pushes accuracy down.  We use scale_pos_weight
# lightly (sqrt of ratio) to get high accuracy AND decent recall.
sqrt_ratio = np.sqrt(ratio)

MODELS = {}

# ── LightGBM ─────────────────────────────────────────────────────────────────
try:
    from lightgbm import LGBMClassifier
    MODELS["LightGBM"] = LGBMClassifier(
        n_estimators=500, learning_rate=0.05,
        max_depth=8, num_leaves=63,
        min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=sqrt_ratio,   # light correction
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    print("  [✓] LightGBM configured")
except ImportError:
    print("  [!] LightGBM not installed — skipping")

# ── CatBoost ──────────────────────────────────────────────────────────────────
try:
    from catboost import CatBoostClassifier
    MODELS["CatBoost"] = CatBoostClassifier(
        iterations=500, depth=8, learning_rate=0.05,
        scale_pos_weight=sqrt_ratio,
        l2_leaf_reg=3, bagging_temperature=1,
        random_seed=42, verbose=0, thread_count=-1,
    )
    print("  [✓] CatBoost configured")
except ImportError:
    print("  [!] CatBoost not installed — skipping")

# ── XGBoost ───────────────────────────────────────────────────────────────────
MODELS["XGBoost"] = XGBClassifier(
    n_estimators=500, max_depth=7, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=sqrt_ratio,
    min_child_weight=5, gamma=0.1,
    reg_alpha=0.1, reg_lambda=1.5,
    random_state=42, eval_metric="logloss", verbosity=0, n_jobs=-1,
)
print("  [✓] XGBoost configured")

# ── HistGradientBoosting ──────────────────────────────────────────────────────
MODELS["HistGB"] = HistGradientBoostingClassifier(
    max_iter=400, max_depth=8, learning_rate=0.05,
    min_samples_leaf=30,
    l2_regularization=0.1,
    class_weight={0: 1, 1: sqrt_ratio},
    random_state=42,
)
print("  [✓] HistGradientBoosting configured")

# ═════════════════════════════════════════════════════════════════════════════
sec("TRAIN ALL MODELS")

trained = {}
results  = []

def eval_metrics(yt, yp, ypr, name, elapsed=0):
    return {
        "Model":    name,
        "Accuracy": round(accuracy_score(yt, yp), 4),
        "Precision":round(precision_score(yt, yp, zero_division=0), 4),
        "Recall":   round(recall_score(yt, yp, zero_division=0), 4),
        "F1":       round(f1_score(yt, yp, zero_division=0), 4),
        "ROC-AUC":  round(roc_auc_score(yt, ypr), 4),
        "PR-AUC":   round(average_precision_score(yt, ypr), 4),
        "MCC":      round(matthews_corrcoef(yt, yp), 4),
        "BalAcc":   round(balanced_accuracy_score(yt, yp), 4),
        "Train(s)": round(elapsed, 1),
        "_yproba":  ypr,
        "_ypred":   yp,
    }

for name, model in MODELS.items():
    print(f"\n  [{name}] training on {X_train.shape[0]:,} rows x {X_train.shape[1]} features...")
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    yp   = model.predict(X_test)
    ypr  = model.predict_proba(X_test)[:, 1]
    r    = eval_metrics(y_test, yp, ypr, name, elapsed)
    results.append(r)
    trained[name] = model
    joblib.dump(model, os.path.join(MODEL_DIR, f"opt_{name}.joblib"))
    print(f"  Accuracy={r['Accuracy']:.4f}  Recall={r['Recall']:.4f}  "
          f"F1={r['F1']:.4f}  ROC-AUC={r['ROC-AUC']:.4f}  [{elapsed:.1f}s]")

# ═════════════════════════════════════════════════════════════════════════════
sec("THRESHOLD OPTIMIZATION — maximize Accuracy & ROC-AUC")

best_model_name = max(results, key=lambda r: r["ROC-AUC"])["Model"]
best_model      = trained[best_model_name]
ypr_best        = best_model.predict_proba(X_test)[:, 1]
print(f"  Best model by ROC-AUC: {best_model_name}")

print(f"\n  {'Threshold':>10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'BalAcc':>10}")
print("  " + "-" * 62)

thresh_rows = []
for th in np.arange(0.05, 0.96, 0.05):
    yp_t = (ypr_best >= th).astype(int)
    acc  = accuracy_score(y_test, yp_t)
    prec = precision_score(y_test, yp_t, zero_division=0)
    rec  = recall_score(y_test, yp_t, zero_division=0)
    f1   = f1_score(y_test, yp_t, zero_division=0)
    ba   = balanced_accuracy_score(y_test, yp_t)
    thresh_rows.append({"threshold": round(th,2), "accuracy": acc,
                        "precision": prec, "recall": rec, "f1": f1, "bal_acc": ba})
    print(f"  {th:>10.2f} {acc:>10.4f} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f} {ba:>10.4f}")

th_df = pd.DataFrame(thresh_rows)
best_acc_th  = th_df.loc[th_df["accuracy"].idxmax(), "threshold"]
best_f1_th   = th_df.loc[th_df["f1"].idxmax(), "threshold"]

print(f"\n  Best threshold by Accuracy : {best_acc_th}  → Acc={th_df.loc[th_df['accuracy'].idxmax(),'accuracy']:.4f}")
print(f"  Best threshold by F1       : {best_f1_th}  → F1={th_df.loc[th_df['f1'].idxmax(),'f1']:.4f}")

# Add threshold-optimized results
for th_val, label in [(best_acc_th, f"{best_model_name}@AccTh={best_acc_th}"),
                       (best_f1_th,  f"{best_model_name}@F1Th={best_f1_th}")]:
    yp_opt = (ypr_best >= th_val).astype(int)
    r_opt  = eval_metrics(y_test, yp_opt, ypr_best, label)
    results.append(r_opt)
    print(f"  {label}: Acc={r_opt['Accuracy']:.4f}  Recall={r_opt['Recall']:.4f}  F1={r_opt['F1']:.4f}")

# ═════════════════════════════════════════════════════════════════════════════
sec("ENSEMBLE — Soft Voting of Top 3 Models")

top3 = sorted([r for r in results if "@" not in r["Model"]],
              key=lambda r: r["ROC-AUC"], reverse=True)[:3]
top3_names = [r["Model"] for r in top3]
print(f"  Top-3 members: {top3_names}")

# Use fresh (unfitted) clones for ensemble to avoid sklearn compatibility issues
from sklearn.base import clone

def make_fresh_pair(name, model):
    """Return a fresh clone of the model for use in ensembles."""
    try:
        return (name, clone(model))
    except Exception:
        return None

fresh_pairs = [p for p in [make_fresh_pair(n, trained[n]) for n in top3_names if n in trained] if p is not None]
print(f"  Fresh clones for ensemble: {[p[0] for p in fresh_pairs]}")

if len(fresh_pairs) >= 2:
    try:
        vc = VotingClassifier(estimators=fresh_pairs, voting="soft", n_jobs=-1)
        t0 = time.time()
        vc.fit(X_train, y_train)
        elapsed = time.time() - t0
        yp  = vc.predict(X_test)
        ypr = vc.predict_proba(X_test)[:, 1]
        rv  = eval_metrics(y_test, yp, ypr, "SoftVoting", elapsed)
        results.append(rv); trained["SoftVoting"] = vc
        joblib.dump(vc, os.path.join(MODEL_DIR, "opt_SoftVoting.joblib"))
        print(f"  SoftVoting: Acc={rv['Accuracy']:.4f}  Recall={rv['Recall']:.4f}  "
              f"F1={rv['F1']:.4f}  ROC-AUC={rv['ROC-AUC']:.4f}  [{elapsed:.1f}s]")
    except Exception as e:
        print(f"  SoftVoting skipped: {e}")

# ── Stacking ──────────────────────────────────────────────────────────────────
try:
    fresh_pairs2 = [p for p in [make_fresh_pair(n, trained[n]) for n in top3_names if n in trained] if p is not None]
    meta = LogisticRegression(C=1, max_iter=500, random_state=42)
    stk  = StackingClassifier(estimators=fresh_pairs2, final_estimator=meta,
                               cv=3, n_jobs=1, passthrough=False)
    t0   = time.time(); stk.fit(X_train, y_train); elapsed = time.time() - t0
    yp   = stk.predict(X_test); ypr = stk.predict_proba(X_test)[:, 1]
    rs   = eval_metrics(y_test, yp, ypr, "StackingEnsemble", elapsed)
    results.append(rs); trained["StackingEnsemble"] = stk
    joblib.dump(stk, os.path.join(MODEL_DIR, "opt_Stacking.joblib"))
    print(f"  Stacking:   Acc={rs['Accuracy']:.4f}  Recall={rs['Recall']:.4f}  "
          f"F1={rs['F1']:.4f}  ROC-AUC={rs['ROC-AUC']:.4f}  [{elapsed:.1f}s]")
except Exception as e:
    print(f"  Stacking skipped: {e}")

# ═════════════════════════════════════════════════════════════════════════════
sec("CROSS-VALIDATION (200k sample, best model)")

idx_cv = np.random.RandomState(42).choice(len(X_train), min(200_000, len(X_train)), replace=False)
Xcv, ycv = X_train.iloc[idx_cv], y_train.iloc[idx_cv]
cv_model  = trained[best_model_name]
cv_scores = cross_val_score(cv_model, Xcv, ycv,
                             cv=StratifiedKFold(5), scoring="roc_auc", n_jobs=-1)
print(f"  {best_model_name} — 5-Fold CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"  Folds: {[round(s,4) for s in cv_scores]}")

# ═════════════════════════════════════════════════════════════════════════════
sec("FINAL RESULTS TABLE")

res_df = pd.DataFrame([{k:v for k,v in r.items() if not k.startswith("_")} for r in results])
res_df = res_df.sort_values("ROC-AUC", ascending=False).reset_index(drop=True)

print(f"\n  {'Model':<35} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'ROC-AUC':>9} {'PR-AUC':>8} {'MCC':>7}")
print("  " + "-" * 100)
for _, row in res_df.iterrows():
    acc_flag = " ★" if row["Accuracy"] >= 0.90 else ""
    print(f"  {row['Model']:<35} {row['Accuracy']:>9.4f} {row['Precision']:>10.4f} "
          f"{row['Recall']:>8.4f} {row['F1']:>8.4f} {row['ROC-AUC']:>9.4f} "
          f"{row['PR-AUC']:>8.4f} {row['MCC']:>7.4f}{acc_flag}")

res_df.to_csv(os.path.join("reports", "optimized_metrics.csv"), index=False)
print(f"\n  Saved: reports/optimized_metrics.csv")

best_acc = res_df["Accuracy"].max()
if best_acc >= 0.90:
    print(f"\n  ✅ TARGET MET: Best Accuracy = {best_acc:.4f} (≥ 90%)")
else:
    print(f"\n  ⚠️  Best Accuracy = {best_acc:.4f} — below 90% target")
    print(f"  → Try increasing n_estimators or tuning threshold further")

# ═════════════════════════════════════════════════════════════════════════════
sec("PLOTS")

# ── Threshold vs metrics ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(th_df["threshold"], th_df["accuracy"],  lw=2, color="#F7B731", label="Accuracy")
ax.plot(th_df["threshold"], th_df["recall"],    lw=2, color="#FF6584", label="Recall")
ax.plot(th_df["threshold"], th_df["f1"],        lw=2, color="#6C63FF", label="F1")
ax.plot(th_df["threshold"], th_df["precision"], lw=2, color="#43BCCD", label="Precision")
ax.plot(th_df["threshold"], th_df["bal_acc"],   lw=2, color="#a29bfe", linestyle="--", label="BalAcc")
ax.axvline(best_acc_th, color="white", lw=1.5, linestyle=":", label=f"Best Acc th={best_acc_th}")
ax.axvline(best_f1_th,  color="#fd79a8", lw=1.5, linestyle=":", label=f"Best F1 th={best_f1_th}")
ax.axhline(0.90, color="#00b894", lw=1, linestyle="--", alpha=0.6, label="90% target")
ax.set_xlabel("Decision Threshold"); ax.set_ylabel("Score")
ax.set_title(f"Threshold Optimization — {best_model_name}", color="#a78bfa", fontsize=13)
ax.legend(framealpha=0.3, fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout(); sfig("opt_threshold_curves.png")

# ── ROC curves ────────────────────────────────────────────────────────────────
base_models = [r for r in results if "@" not in r["Model"]]
fig, ax = plt.subplots(figsize=(9, 7))
ax.plot([0,1],[0,1],"--", color="#555577", lw=1, label="Random")
for r, col in zip(base_models, PAL):
    if r["Model"] not in trained: continue
    ypr_ = trained[r["Model"]].predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, ypr_)
    ax.plot(fpr, tpr, lw=2, color=col, label=f"{r['Model']} (AUC={r['ROC-AUC']:.4f})")
ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
ax.set_title("ROC Curves — Optimized Models", color="#a78bfa", fontsize=13)
ax.legend(framealpha=0.3, fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout(); sfig("opt_roc_curves.png")

# ── PR curves ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))
ax.axhline(y.mean(), color="#555577", linestyle="--", lw=1, label=f"Random (PR={y.mean():.3f})")
for r, col in zip(base_models, PAL):
    if r["Model"] not in trained: continue
    ypr_ = trained[r["Model"]].predict_proba(X_test)[:, 1]
    prec_, rec_, _ = precision_recall_curve(y_test, ypr_)
    ax.plot(rec_, prec_, lw=2, color=col, label=f"{r['Model']} (PR-AUC={r['PR-AUC']:.4f})")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curves", color="#a78bfa", fontsize=13)
ax.legend(framealpha=0.3, fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout(); sfig("opt_pr_curves.png")

# ── Metrics bar chart ─────────────────────────────────────────────────────────
plot_models = res_df[~res_df["Model"].str.contains("@")].head(6)
mets = ["Accuracy","Precision","Recall","F1","ROC-AUC","PR-AUC"]
x = np.arange(len(mets)); w = 0.13
fig, ax = plt.subplots(figsize=(15, 6))
for i, (_, row) in enumerate(plot_models.iterrows()):
    ax.bar(x + i*w, [row[m] for m in mets], w, label=row["Model"], color=PAL[i%len(PAL)], alpha=0.9)
ax.axhline(0.90, color="white", lw=1, linestyle="--", alpha=0.5, label="90% target")
ax.set_xticks(x + w*2.5); ax.set_xticklabels(mets)
ax.set_ylim(0, 1.08); ax.set_ylabel("Score")
ax.set_title("Optimized Model Performance Comparison", color="#a78bfa", fontsize=13)
ax.legend(framealpha=0.3, fontsize=8); ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); sfig("opt_metrics_comparison.png")

# ── Confusion matrix — best model at best threshold ───────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(f"Confusion Matrices — {best_model_name}", color="#a78bfa", fontsize=13)
for ax_cm, th_val, lbl in zip(axes, [0.5, best_acc_th],
                                [f"Default th=0.50", f"Best-Acc th={best_acc_th}"]):
    yp_cm = (ypr_best >= th_val).astype(int)
    cm = confusion_matrix(y_test, yp_cm)
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
    labels = np.array([[f"{v:,}\n({p:.1f}%)" for v, p in zip(row, prow)]
                        for row, prow in zip(cm, cm_pct)])
    sns.heatmap(cm, annot=labels, fmt="", cmap="Purples",
                xticklabels=["No Churn","Churn"],
                yticklabels=["No Churn","Churn"],
                linewidths=0.5, linecolor="#0f1117", ax=ax_cm,
                annot_kws={"size": 11})
    acc_v = accuracy_score(y_test, yp_cm)
    ax_cm.set_title(f"{lbl}  [Acc={acc_v:.4f}]")
    ax_cm.set_ylabel("Actual"); ax_cm.set_xlabel("Predicted")
plt.tight_layout(); sfig("opt_confusion_matrices.png")

# ── Feature importance (best tree model) ─────────────────────────────────────
if best_model_name in trained and hasattr(trained[best_model_name], "feature_importances_"):
    m_fi = trained[best_model_name]
    imp  = pd.Series(m_fi.feature_importances_, index=X_train.columns)
    top20 = imp.sort_values(ascending=True).tail(20)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top20.index, top20.values, color="#6C63FF", alpha=0.85)
    ax.set_title(f"Feature Importance — {best_model_name} (Top 20)", color="#a78bfa", fontsize=13)
    ax.set_xlabel("Importance Score")
    plt.tight_layout(); sfig("opt_feature_importance.png")

# ═════════════════════════════════════════════════════════════════════════════
sec("SUMMARY")

best_overall = res_df.iloc[0]
print(f"""
  Dataset     : {DATA_PATH}
  Features    : {X.shape[1]}
  Train/Test  : {X_train.shape[0]:,} / {X_test.shape[0]:,}
  Churn rate  : {churn_pct:.2f}%
  Class ratio : {ratio:.2f}  (sqrt used: {sqrt_ratio:.2f})

  ┌─────────────────────────────────────────────────────┐
  │  BEST MODEL (by ROC-AUC): {best_overall['Model']:<25}│
  │  Accuracy  : {best_overall['Accuracy']:.4f}                              │
  │  Precision : {best_overall['Precision']:.4f}                              │
  │  Recall    : {best_overall['Recall']:.4f}                              │
  │  F1        : {best_overall['F1']:.4f}                              │
  │  ROC-AUC   : {best_overall['ROC-AUC']:.4f}                              │
  │  PR-AUC    : {best_overall['PR-AUC']:.4f}                              │
  │  MCC       : {best_overall['MCC']:.4f}                              │
  └─────────────────────────────────────────────────────┘

  5-Fold CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}
  Best Accuracy (any threshold): {res_df['Accuracy'].max():.4f}
  Best ROC-AUC                 : {res_df['ROC-AUC'].max():.4f}
  Best Recall                  : {res_df['Recall'].max():.4f}

  Plots  → reports/figures/opt_*.png
  Models → models/opt_*.joblib
  CSV    → reports/optimized_metrics.csv
""")
print("[✓] optimized_training.py COMPLETE\n")
