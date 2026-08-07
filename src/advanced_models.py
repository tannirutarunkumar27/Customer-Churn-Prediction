"""
advanced_models.py  — Priorities 4-14
LightGBM, CatBoost, XGBoost tuned, Ensemble, Threshold, CV, Risk Segments
Run: python src/advanced_models.py
"""
import os, warnings, time
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, roc_curve,
    precision_recall_curve, matthews_corrcoef, balanced_accuracy_score, classification_report)
from xgboost import XGBClassifier

plt.rcParams.update({"figure.facecolor":"#0f1117","axes.facecolor":"#1a1a2e","axes.edgecolor":"#3a3a5c",
    "axes.labelcolor":"#e0e0e0","xtick.color":"#b0b0c0","ytick.color":"#b0b0c0","text.color":"#e0e0e0",
    "grid.color":"#2a2a4a","font.family":"sans-serif"})
PAL = ["#6C63FF","#FF6584","#43BCCD","#F7B731","#a29bfe","#fd79a8","#00b894"]
FIG = os.path.join("reports","figures"); os.makedirs(FIG, exist_ok=True)
os.makedirs("models", exist_ok=True)

D="="*70
def sec(t): print(f"\n{D}\n  {t}\n{D}")
def sfig(n):
    plt.savefig(os.path.join(FIG,n),dpi=140,bbox_inches="tight",facecolor=plt.gcf().get_facecolor())
    plt.close("all"); print(f"  [saved] reports/figures/{n}")
TARGET="churn"

# ─── LOAD ─────────────────────────────────────────────────────────────────────
sec("LOAD ADVANCED FEATURES")
df = pd.read_csv(os.path.join("data","processed","churn_advanced.csv"))
print(f"  Shape: {df.shape[0]:,} x {df.shape[1]}")
X = df.drop(columns=[TARGET]); y = df[TARGET]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,stratify=y,random_state=42)
ratio = (y_train==0).sum()/(y_train==1).sum()
print(f"  Train={X_train.shape[0]:,} | Test={X_test.shape[0]:,} | pos_weight={ratio:.2f}")

# ─── HELPER ───────────────────────────────────────────────────────────────────
def metrics(yt,yp,yproba,name,t=None):
    return {"Model":name,"Accuracy":round(accuracy_score(yt,yp),4),
            "Precision":round(precision_score(yt,yp,zero_division=0),4),
            "Recall":round(recall_score(yt,yp,zero_division=0),4),
            "F1":round(f1_score(yt,yp,zero_division=0),4),
            "ROC-AUC":round(roc_auc_score(yt,yproba),4),
            "PR-AUC":round(average_precision_score(yt,yproba),4),
            "MCC":round(matthews_corrcoef(yt,yp),4),
            "BalAcc":round(balanced_accuracy_score(yt,yp),4),
            "Train(s)":round(t,1) if t else 0}

results=[]; trained={}

# ─── PRIORITY 5: IMBALANCE COMPARISON (100k sample) ─────────────────────────
sec("PRIORITY 5 — Imbalance Strategy Comparison (100k sample, XGBoost)")
idx=np.random.RandomState(42).choice(len(X_train),100_000,replace=False)
Xs,ys=X_train.iloc[idx],y_train.iloc[idx]

def quick(Xtr,ytr,label):
    m=XGBClassifier(n_estimators=100,scale_pos_weight=(ytr==0).sum()/(ytr==1).sum(),
                    max_depth=5,random_state=42,eval_metric="logloss",verbosity=0)
    m.fit(Xtr,ytr); yp=m.predict(X_test); ypr=m.predict_proba(X_test)[:,1]
    r=metrics(y_test,yp,ypr,label); print(f"  {label:<25} Recall={r['Recall']:.4f} F1={r['F1']:.4f} ROC-AUC={r['ROC-AUC']:.4f}")
    return r

print()
imb=[quick(Xs,ys,"Baseline(XGB)")]

m_cw=XGBClassifier(n_estimators=100,scale_pos_weight=ratio,max_depth=5,random_state=42,eval_metric="logloss",verbosity=0)
m_cw.fit(Xs,ys); yp=m_cw.predict(X_test); ypr=m_cw.predict_proba(X_test)[:,1]
r=metrics(y_test,yp,ypr,"ScalePosWeight"); print(f"  {'ScalePosWeight':<25} Recall={r['Recall']:.4f} F1={r['F1']:.4f} ROC-AUC={r['ROC-AUC']:.4f}")
imb.append(r)

for strat,cls in [("SMOTE","SMOTE"),("BorderlineSMOTE","BorderlineSMOTE"),("ADASYN","ADASYN"),("TomekLinks","TomekLinks")]:
    try:
        if cls=="TomekLinks":
            from imblearn.under_sampling import TomekLinks as CLS
        else:
            from imblearn.over_sampling import SMOTE,BorderlineSMOTE,ADASYN
            CLS=eval(cls)
        samp=CLS(random_state=42)
        Xr,yr=samp.fit_resample(Xs,ys)
        imb.append(quick(Xr,yr,strat))
    except Exception as e:
        print(f"  {strat}: skipped ({e})")

# ─── PRIORITY 10: ADVANCED MODELS (full data) ────────────────────────────────
sec("PRIORITY 10 — Advanced Models (Full 800k Training Set)")

MODELS={}
# LightGBM
try:
    from lightgbm import LGBMClassifier
    MODELS["LightGBM"]=LGBMClassifier(n_estimators=300,scale_pos_weight=ratio,
        max_depth=8,learning_rate=0.05,num_leaves=63,min_child_samples=50,
        random_state=42,n_jobs=-1,verbose=-1)
except ImportError: print("  LightGBM not installed")

# CatBoost
try:
    from catboost import CatBoostClassifier
    MODELS["CatBoost"]=CatBoostClassifier(iterations=300,depth=8,learning_rate=0.05,
        auto_class_weights="Balanced",random_seed=42,verbose=0,thread_count=-1)
except ImportError: print("  CatBoost not installed")

# XGBoost
MODELS["XGBoost"]=XGBClassifier(n_estimators=300,scale_pos_weight=ratio,max_depth=7,
    learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,
    min_child_weight=5,gamma=0.1,random_state=42,eval_metric="logloss",verbosity=0,n_jobs=-1)

# HistGradientBoosting
MODELS["HistGB"]=HistGradientBoostingClassifier(max_iter=300,max_depth=8,
    learning_rate=0.05,class_weight="balanced",random_state=42)

for name,model in MODELS.items():
    print(f"\n  [{name}] training on {X_train.shape[0]:,} samples...")
    t0=time.time(); model.fit(X_train,y_train); elapsed=time.time()-t0
    yp=model.predict(X_test); ypr=model.predict_proba(X_test)[:,1]
    r=metrics(y_test,yp,ypr,name,elapsed); results.append(r); trained[name]=model
    joblib.dump(model,os.path.join("models",f"adv_{name}.joblib"))
    print(f"  Recall={r['Recall']:.4f} F1={r['F1']:.4f} ROC-AUC={r['ROC-AUC']:.4f} PR-AUC={r['PR-AUC']:.4f}  [{elapsed:.1f}s]")

# ─── PRIORITY 7: HYPERPARAMETER TUNING ───────────────────────────────────────
sec("PRIORITY 7 — Hyperparameter Tuning (RandomizedSearchCV, 200k sample, 3-fold)")
idx2=np.random.RandomState(42).choice(len(X_train),min(200_000,len(X_train)),replace=False)
Xt2,yt2=X_train.iloc[idx2],y_train.iloc[idx2]
best_model_name = max(results,key=lambda r:r["ROC-AUC"])["Model"] if results else "XGBoost"
print(f"  Tuning best model so far: {best_model_name}")

if best_model_name=="LightGBM" and "LightGBM" in trained:
    from lightgbm import LGBMClassifier
    param_dist={"n_estimators":[200,300,500],"max_depth":[6,8,10],"num_leaves":[31,63,127],
                "learning_rate":[0.01,0.05,0.1],"min_child_samples":[20,50,100],
                "subsample":[0.7,0.8,0.9],"colsample_bytree":[0.7,0.8,0.9]}
    base=LGBMClassifier(scale_pos_weight=ratio,random_state=42,n_jobs=-1,verbose=-1)
elif best_model_name=="XGBoost":
    param_dist={"n_estimators":[200,300,500],"max_depth":[5,6,7,8],
                "learning_rate":[0.01,0.05,0.1],"subsample":[0.7,0.8,0.9],
                "colsample_bytree":[0.7,0.8,0.9],"min_child_weight":[3,5,10],
                "gamma":[0,0.1,0.3],"reg_alpha":[0,0.1,0.5],"reg_lambda":[1,1.5,2]}
    base=XGBClassifier(scale_pos_weight=ratio,random_state=42,eval_metric="logloss",verbosity=0,n_jobs=-1)
else:
    # HistGradientBoosting
    param_dist={"max_iter":[200,300,400],"max_depth":[5,6,8],"learning_rate":[0.03,0.05,0.1],
                "min_samples_leaf":[20,50,100],"l2_regularization":[0,0.1,0.5]}
    base=HistGradientBoostingClassifier(class_weight="balanced",random_state=42)

rs=RandomizedSearchCV(base,param_dist,n_iter=12,cv=StratifiedKFold(3),
    scoring="roc_auc",random_state=42,n_jobs=1,verbose=0)
t0=time.time(); rs.fit(Xt2,yt2); elapsed=time.time()-t0
best_cv=rs.best_score_ if not np.isnan(rs.best_score_) else 0
print(f"\n  Best params: {rs.best_params_}")
print(f"  Best CV ROC-AUC: {best_cv:.4f}  [{elapsed:.1f}s]")
tuned=rs.best_estimator_; tuned.fit(X_train,y_train)
yp=tuned.predict(X_test); ypr=tuned.predict_proba(X_test)[:,1]
r_tuned=metrics(y_test,yp,ypr,f"Tuned_{best_model_name}")
results.append(r_tuned); trained[f"Tuned_{best_model_name}"]=tuned
joblib.dump(tuned,os.path.join("models","adv_tuned_best.joblib"))
print(f"  Tuned → Recall={r_tuned['Recall']:.4f} F1={r_tuned['F1']:.4f} ROC-AUC={r_tuned['ROC-AUC']:.4f}")

# ─── PRIORITY 11: ENSEMBLE ────────────────────────────────────────────────────
sec("PRIORITY 11 — Ensemble (Voting + Stacking)")
# Use top 3 models by ROC-AUC
top3=sorted(results,key=lambda r:r["ROC-AUC"],reverse=True)[:3]
top3_names=[r["Model"] for r in top3]
print(f"  Ensemble members: {top3_names}")
est=[(n,trained[n]) for n in top3_names if n in trained]

# Only use sklearn-native models in Voting (exclude XGBoost which is not sklearn-native for VotingClassifier)
sklearn_compat = [(n,m) for n,m in est if "XGBoost" not in n]
if len(sklearn_compat)>=2:
    # Soft Voting
    vc=VotingClassifier(estimators=sklearn_compat,voting="soft",n_jobs=-1)
    t0=time.time(); vc.fit(X_train,y_train); elapsed=time.time()-t0
    yp=vc.predict(X_test); ypr=vc.predict_proba(X_test)[:,1]
    rv=metrics(y_test,yp,ypr,"VotingEnsemble",elapsed); results.append(rv); trained["VotingEnsemble"]=vc
    print(f"  VotingEnsemble → Recall={rv['Recall']:.4f} F1={rv['F1']:.4f} ROC-AUC={rv['ROC-AUC']:.4f}")
else:
    print(f"  Skipping VotingClassifier (need >=2 sklearn-native models, have {[n for n,_ in sklearn_compat]})")

# Stacking (works with any estimators)
try:
    meta=LogisticRegression(C=1,class_weight="balanced",max_iter=500,random_state=42)
    stk_est=est[:3]
    sc2=StackingClassifier(estimators=stk_est,final_estimator=meta,cv=3,n_jobs=1,passthrough=False)
    t0=time.time(); sc2.fit(X_train,y_train); elapsed=time.time()-t0
    yp=sc2.predict(X_test); ypr=sc2.predict_proba(X_test)[:,1]
    rs2=metrics(y_test,yp,ypr,"StackingEnsemble",elapsed); results.append(rs2); trained["StackingEnsemble"]=sc2
    print(f"  StackingEnsemble→ Recall={rs2['Recall']:.4f} F1={rs2['F1']:.4f} ROC-AUC={rs2['ROC-AUC']:.4f}")
except Exception as e:
    print(f"  Stacking skipped: {e}")

# ─── PRIORITY 6: THRESHOLD OPTIMIZATION ──────────────────────────────────────
sec("PRIORITY 6 — Threshold Optimization (Best Model)")
best_name=max(results,key=lambda r:r["ROC-AUC"])["Model"]
best_m=trained[best_name]
ypr_best=best_m.predict_proba(X_test)[:,1]
print(f"  Sweeping thresholds on: {best_name}")
print(f"\n  {'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'BalAcc':>10}")
print("  "+"-"*55)
thresh_results=[]
for th in np.arange(0.05,0.65,0.05):
    yp=(ypr_best>=th).astype(int)
    p=precision_score(y_test,yp,zero_division=0)
    r=recall_score(y_test,yp,zero_division=0)
    f=f1_score(y_test,yp,zero_division=0)
    b=balanced_accuracy_score(y_test,yp)
    thresh_results.append({"threshold":round(th,2),"precision":p,"recall":r,"f1":f,"bal_acc":b})
    print(f"  {th:>10.2f} {p:>10.4f} {r:>10.4f} {f:>10.4f} {b:>10.4f}")
th_df=pd.DataFrame(thresh_results)
best_th=th_df.loc[th_df["f1"].idxmax(),"threshold"]
best_th_recall=th_df.loc[th_df["recall"].idxmax(),"threshold"]
print(f"\n  Best threshold by F1   : {best_th}")
print(f"  Best threshold by Recall: {best_th_recall}")

# Apply best threshold
yp_opt=(ypr_best>=best_th).astype(int)
r_opt=metrics(y_test,yp_opt,ypr_best,f"{best_name}@th={best_th}")
results.append(r_opt)
print(f"  {best_name} @ th={best_th} → Recall={r_opt['Recall']:.4f} F1={r_opt['F1']:.4f}")

# Threshold plot
fig,ax=plt.subplots(figsize=(10,5))
ax.plot(th_df["threshold"],th_df["recall"],lw=2,color="#FF6584",label="Recall")
ax.plot(th_df["threshold"],th_df["f1"],lw=2,color="#6C63FF",label="F1")
ax.plot(th_df["threshold"],th_df["precision"],lw=2,color="#43BCCD",label="Precision")
ax.plot(th_df["threshold"],th_df["bal_acc"],lw=2,color="#F7B731",linestyle="--",label="Balanced Acc")
ax.axvline(best_th,color="white",lw=1.5,linestyle=":",label=f"Best F1 th={best_th}")
ax.set_xlabel("Decision Threshold"); ax.set_ylabel("Score")
ax.set_title(f"Threshold Optimization — {best_name}",color="#a78bfa",fontsize=13)
ax.legend(framealpha=0.3); ax.grid(alpha=0.3)
plt.tight_layout(); sfig("adv_threshold_optimization.png")

# ─── PRIORITY 12: STRATIFIED CV ───────────────────────────────────────────────
sec("PRIORITY 12 — Stratified 5-Fold CV (200k sample, best model)")
cv_model=trained[best_name]
cv_scores=cross_val_score(cv_model,Xt2,yt2,cv=StratifiedKFold(5),scoring="roc_auc",n_jobs=-1)
print(f"  {best_name} — 5-Fold CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"  Fold scores: {[round(s,4) for s in cv_scores]}")

# ─── PRIORITY 19: RISK SEGMENTATION ──────────────────────────────────────────
sec("PRIORITY 19 — Risk Segmentation")
proba_df=pd.DataFrame({"churn_prob":ypr_best,"actual":y_test.values})
proba_df["risk_segment"]=pd.cut(proba_df["churn_prob"],
    bins=[0,0.15,0.30,0.50,1.0],labels=["Low","Medium","High","Critical"])
print(f"\n  {'Segment':<12} {'Count':>10} {'Actual Churn%':>15}")
print("  "+"-"*40)
for seg in ["Low","Medium","High","Critical"]:
    sub=proba_df[proba_df["risk_segment"]==seg]
    actual_churn=sub["actual"].mean()*100
    print(f"  {seg:<12} {len(sub):>10,} {actual_churn:>14.2f}%")

# ─── PRIORITY 15: BUSINESS PROFIT METRIC ─────────────────────────────────────
sec("PRIORITY 15 — Business Profit Optimization")
FN_COST=5000; FP_COST=100
print(f"  FN cost (missed churner) = Rs.{FN_COST:,}")
print(f"  FP cost (false alarm)    = Rs.{FP_COST:,}")
print(f"\n  {'Threshold':>10} {'Profit (Rs.)':>15} {'FN':>8} {'FP':>8}")
print("  "+"-"*45)
best_profit=-9e18; best_profit_th=0.5
for th in np.arange(0.05,0.65,0.05):
    yp=(ypr_best>=th).astype(int)
    cm=confusion_matrix(y_test,yp)
    tn,fp,fn,tp=cm.ravel()
    profit=-(fn*FN_COST+fp*FP_COST)
    if profit>best_profit: best_profit=profit; best_profit_th=round(th,2)
    print(f"  {th:>10.2f} {profit:>15,.0f} {fn:>8,} {fp:>8,}")
print(f"\n  Best threshold by profit: {best_profit_th} (Profit=Rs.{best_profit:,.0f})")

# ─── FINAL METRICS TABLE ──────────────────────────────────────────────────────
sec("FINAL MODEL COMPARISON TABLE")
res_df=pd.DataFrame([{k:v for k,v in r.items()} for r in results])
res_df=res_df.sort_values("ROC-AUC",ascending=False).reset_index(drop=True)
print(f"\n  {'Model':<30} {'Acc':>7} {'Prec':>7} {'Recall':>7} {'F1':>7} {'ROC-AUC':>9} {'PR-AUC':>8} {'MCC':>7}")
print("  "+"-"*90)
for _,row in res_df.iterrows():
    print(f"  {row['Model']:<30} {row['Accuracy']:>7.4f} {row['Precision']:>7.4f} "
          f"{row['Recall']:>7.4f} {row['F1']:>7.4f} {row['ROC-AUC']:>9.4f} "
          f"{row['PR-AUC']:>8.4f} {row['MCC']:>7.4f}")
res_df.to_csv(os.path.join("reports","advanced_metrics.csv"),index=False)

# ─── PLOTS ─────────────────────────────────────────────────────────────────────
# ROC curves (all advanced models)
top_models=[r for r in results if "@th=" not in r["Model"] and r["Model"] not in ["VotingEnsemble","StackingEnsemble","Baseline(XGB)","ScalePosWeight"]]
fig,ax=plt.subplots(figsize=(10,8))
ax.plot([0,1],[0,1],"--",color="#555577",lw=1,label="Random")
for r,col in zip(top_models,PAL):
    if r["Model"] not in trained: continue
    ypr_=trained[r["Model"]].predict_proba(X_test)[:,1]
    fpr,tpr,_=roc_curve(y_test,ypr_)
    ax.plot(fpr,tpr,lw=2,color=col,label=f"{r['Model']} (AUC={r['ROC-AUC']:.4f})")
ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
ax.set_title("ROC Curves — Advanced Models",color="#a78bfa",fontsize=13)
ax.legend(framealpha=0.3,fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout(); sfig("adv_roc_curves.png")

# Metrics bar chart
plot_r=res_df[~res_df["Model"].str.contains("@th=")].head(7)
metrics_cols=["Recall","F1","ROC-AUC","PR-AUC"]
x=np.arange(len(metrics_cols)); w=0.12
fig,ax=plt.subplots(figsize=(14,6))
for i,((_,row),col) in enumerate(zip(plot_r.iterrows(),PAL)):
    ax.bar(x+i*w,[row[m] for m in metrics_cols],w,label=row["Model"],color=col,alpha=0.9)
ax.set_xticks(x+w*3); ax.set_xticklabels(metrics_cols)
ax.set_ylim(0,1.05); ax.set_ylabel("Score")
ax.set_title("Advanced Model Comparison",color="#a78bfa",fontsize=13)
ax.legend(framealpha=0.3,fontsize=8); ax.grid(axis="y",alpha=0.3)
plt.tight_layout(); sfig("adv_metrics_comparison.png")

# Confusion matrix best model
fig,axes=plt.subplots(1,2,figsize=(12,5))
fig.suptitle(f"Confusion Matrices — {best_name}",color="#a78bfa",fontsize=13)
for ax,th_,label in zip(axes,[0.5,best_th],[f"Default th=0.5",f"Optimal th={best_th}"]):
    yp_=(ypr_best>=th_).astype(int)
    cm=confusion_matrix(y_test,yp_)
    sns.heatmap(cm,annot=True,fmt=",",cmap="Purples",ax=ax,
                xticklabels=["No Churn","Churn"],yticklabels=["No Churn","Churn"],
                annot_kws={"size":11})
    ax.set_title(label); ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
plt.tight_layout(); sfig("adv_confusion_matrices.png")

print(f"\n  Best model overall: {res_df.iloc[0]['Model']}")
print(f"  Best ROC-AUC: {res_df.iloc[0]['ROC-AUC']:.4f}")
print(f"  Best Recall : {res_df['Recall'].max():.4f} ({res_df.loc[res_df['Recall'].idxmax(),'Model']})")
print(f"  5-Fold CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"\n[✓] advanced_models.py COMPLETE\n")
