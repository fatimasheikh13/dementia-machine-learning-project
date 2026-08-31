#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from datetime import datetime

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression
from xgboost import XGBClassifier
from scipy.stats import uniform, randint

# directories 
RAW_FILE            = "/scratch/c.c2029098/dementia_ml_project/data/processed/ml_data/BDR_AD_control.raw"
EIGENVEC_FILE       = "/scratch/c.c2029098/dementia_ml_project/results/PCA/BDR_AD_control.eigenvec"
USE_FIRST_N_PCS     = 7

OUT_ROOT            = "/scratch/c.c2029098/dementia_ml_project/results/machine_learning/AD_control/xgboost_interactions"
AUC_TXT             = f"{OUT_ROOT}/xgb_auc.txt"
PLOTS_DIR           = f"{OUT_ROOT}/plots"
TABLES_DIR          = f"{OUT_ROOT}/tables"

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Loci of interest (APOE, ECHDC3)
APOE_PREFIX   = "chr19:44908684"
ECHDC3_PREFIX = "chr10:11678309"

def find_feature_cols(columns, prefix):
    return [c for c in columns if str(c).startswith(prefix)]

# Load data
df = pd.read_csv(RAW_FILE, sep=r"\s+")
df.columns = df.columns.str.strip()

# PCs
ev = pd.read_csv(EIGENVEC_FILE, sep=r"\s+", header=None)
m = ev.shape[1] - 2
ev.columns = ["FID","IID"] + [f"PC{i}" for i in range(1, m+1)]
pc_cols = [f"PC{i}" for i in range(1, min(USE_FIRST_N_PCS, m)+1)]
for c in pc_cols:
    ev[c] = pd.to_numeric(ev[c], errors="coerce")

df = df.merge(ev[["FID","IID"] + pc_cols], on=["FID","IID"], how="left")
df.columns = df.columns.str.strip()

# Targets 0/1
y = pd.to_numeric(df["PHENOTYPE"], errors="coerce").replace({1:0, 2:1}).astype(int)

# SNP features only
NON_FEATURES = {"FID","IID","PAT","MAT","SEX","PHENOTYPE","PRS"}
drop_cols = list(NON_FEATURES) + pc_cols
X_snp = df.drop(columns=drop_cols, errors="ignore").apply(pd.to_numeric, errors="coerce").fillna(0.0)

# no leaked PCs in features
leaked = [c for c in X_snp.columns if c.upper().startswith("PC")]
if leaked:
    raise RuntimeError(f"PCs leaked into features: {leaked[:10]}")

# Keep PCs separately for post-hoc adjustment of scores
PCs_all = df[pc_cols].apply(pd.to_numeric, errors="coerce")

# Map target columns
apoe_cols   = find_feature_cols(X_snp.columns, APOE_PREFIX)
echdc3_cols = find_feature_cols(X_snp.columns, ECHDC3_PREFIX)
apoe_col    = apoe_cols[0]   if apoe_cols   else None
echdc3_col  = echdc3_cols[0] if echdc3_cols else None

def pc_adjust_scores(scores: np.ndarray, pcs_df: pd.DataFrame) -> np.ndarray:
    """Regress out PCs from prediction scores (test only)."""
    M = pcs_df.to_numpy(dtype=float, copy=True)
    if M.ndim != 2 or M.shape[0] < 2 or M.shape[1] == 0:
        return scores
    if np.isnan(M).any():
        col_means = np.nanmean(M, axis=0)
        inds = np.where(np.isnan(M))
        M[inds] = np.take(np.where(np.isnan(col_means), 0.0, col_means), inds[1])
    adj = LinearRegression().fit(M, scores)
    return scores - adj.predict(M)

# training 
NUM_REPS = 100         
TEST_SIZE = 0.2
N_ITER    = 100

train_aucs, test_aucs, test_aucs_pcadj = [], [], []

all_shap_rows   = []
all_feature_rows= []
interaction_mats= []   
gain_vectors    = []  

rng_indices = np.arange(len(y))
for rep in range(NUM_REPS):
    tr_idx, te_idx, y_tr, y_te = train_test_split(
        rng_indices, y.values, test_size=TEST_SIZE, random_state=rep, stratify=y
    )
    X_tr = X_snp.iloc[tr_idx]; X_te = X_snp.iloc[te_idx]
    PCs_te = PCs_all.iloc[te_idx]

    # Basic imbalance handling: scale_pos_weight on training fold
    n_pos = (y_tr == 1).sum(); n_neg = (y_tr == 0).sum()
    spw = float(n_neg) / float(max(n_pos, 1))

    base = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        n_jobs=-1,
        random_state=rep,
        tree_method="hist",
        scale_pos_weight=spw,          
        subsample=0.8, colsample_bytree=0.8
    )

    # randoms search
    param_dist = {
        "max_depth": randint(2, 6),
        "min_child_weight": randint(1, 6),
        "gamma": uniform(0.0, 2.0),
        "learning_rate": uniform(0.01, 0.15),
        "n_estimators": randint(200, 601),
        "reg_lambda": uniform(5.0, 20.0), 
        "reg_alpha": uniform(0.0, 5.0),   
    }

    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=rep)
    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_dist,
        n_iter=N_ITER,
        scoring="roc_auc",
        cv=inner_cv,
        n_jobs=-1,
        random_state=rep,
        verbose=0,
        refit=True,
    )
    search.fit(X_tr, y_tr)
    best = search.best_estimator_

    # AUCs
    p_tr  = best.predict_proba(X_tr)[:, 1]
    p_te  = best.predict_proba(X_te)[:, 1]
    auc_tr = roc_auc_score(y_tr, p_tr)
    auc_te = roc_auc_score(y_te, p_te)
    auc_te_adj = roc_auc_score(y_te, pc_adjust_scores(p_te, PCs_te[pc_cols]))

    train_aucs.append(auc_tr); test_aucs.append(auc_te); test_aucs_pcadj.append(auc_te_adj)

    # SHAP main effects on test fold
    explainer = shap.TreeExplainer(best)
    expl = explainer(X_te)
    shap_main = expl.values                       
    all_shap_rows.append(shap_main)
    all_feature_rows.append(X_te.values)

    # SHAP interaction values 
    try:
        shap_inter = explainer.shap_interaction_values(X_te)  
        mean_abs_inter = np.mean(np.abs(shap_inter), axis=0)  
        interaction_mats.append(mean_abs_inter)
    except Exception:
        pass

    # Gain importances aligned to column order
    booster = best.get_booster()
    imp = booster.get_score(importance_type="gain")
    gain_vec = np.array([imp.get(name, 0.0) for name in X_snp.columns], dtype=float)
    gain_vectors.append(gain_vec)

# Aggregate across reps
with open(AUC_TXT, "w") as f:
    for i in range(NUM_REPS):
        f.write(f"Rep {i+1:02d}  Train AUC: {train_aucs[i]:.4f}  "
                f"Test AUC: {test_aucs[i]:.4f}  Test AUC (PC-adj): {test_aucs_pcadj[i]:.4f}\n")
    f.write("\nAverages\n")
    f.write(f"Train AUC: {np.mean(train_aucs):.4f}\n")
    f.write(f"Test AUC: {np.mean(test_aucs):.4f}\n")
    f.write(f"Test AUC (PC-adj): {np.mean(test_aucs_pcadj):.4f}\n")

# Concatenate SHAP rows & features rows for a single, global beeswarm/dependence
SHAP_ALL = np.vstack(all_shap_rows)           
FEAT_ALL = np.vstack(all_feature_rows)       
FEAT_ALL_DF = pd.DataFrame(FEAT_ALL, columns=X_snp.columns)

# One SHAP summary (beeswarm)
plt.figure(figsize=(9, 7))
shap.summary_plot(SHAP_ALL, FEAT_ALL_DF, show=False, max_display=25)
plt.title("SHAP summary (all reps, test folds concatenated)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, f"shap_beeswarm_all_reps_{stamp}.png"), dpi=220)
plt.close()

# Mean SHAP values as table
mean_abs_shap = pd.Series(np.mean(np.abs(SHAP_ALL), axis=0), index=X_snp.columns).sort_values(ascending=False)
mean_abs_shap.to_csv(os.path.join(TABLES_DIR, f"mean_abs_shap_all_reps_{stamp}.tsv"), sep="\t", header=["mean_abs_shap"])

# One dependence plot: APOE colored by chr10 
if (apoe_col is not None) and (echdc3_col is not None):
    apoe_idx   = X_snp.columns.get_loc(apoe_col)
    echdc3_idx = X_snp.columns.get_loc(echdc3_col)
    plt.figure(figsize=(7,5))
    shap.dependence_plot(ind=apoe_idx, shap_values=SHAP_ALL, features=FEAT_ALL_DF,
                         interaction_index=echdc3_idx, show=False)
    plt.title(f"Dependence: {apoe_col} by {echdc3_col} (all reps combined)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"dependence_{apoe_col}_by_{echdc3_col}_{stamp}.png"), dpi=220)
    plt.close()
else:
    print("APOE and/or chr10 feature not found — skipping dependence plot.")

# Top SHAP interactions table (averaged across reps)
if len(interaction_mats) > 0:
    inter_mean = np.mean(np.stack(interaction_mats, axis=0), axis=0)  
    tri_i, tri_j = np.triu_indices(inter_mean.shape[0], k=1)
    vals = inter_mean[tri_i, tri_j]
    order = np.argsort(vals)[::-1]   
    top_k = min(50, len(order))
    top_pairs = []
    for k in range(top_k):
        i = tri_i[order[k]]; j = tri_j[order[k]]
        top_pairs.append({
            "feat_i": X_snp.columns[i],
            "feat_j": X_snp.columns[j],
            "mean_abs_interaction": float(vals[order[k]])
        })
    top_df = pd.DataFrame(top_pairs)
    top_df.to_csv(os.path.join(TABLES_DIR, f"top_shap_interactions_{stamp}.tsv"), sep="\t", index=False)
else:
    print("SHAP interaction values unavailable on this setup — no interaction table written.")

# One bar plot: average XGBoost gain (Top 20) 
if len(gain_vectors) > 0:
    avg_gain = pd.Series(np.mean(np.stack(gain_vectors, axis=0), axis=0), index=X_snp.columns).sort_values(ascending=False)
    plt.figure(figsize=(8,6))
    avg_gain.head(20)[::-1].plot(kind="barh")
    plt.title("Average XGBoost Gain (Top 20) — across reps")
    plt.xlabel("Average gain")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"avg_gain_top20_{stamp}.png"), dpi=220)
    plt.close()
    avg_gain.to_csv(os.path.join(TABLES_DIR, f"avg_gain_full_{stamp}.tsv"), sep="\t", header=["avg_gain"])
else:
    print("No gain vectors collected — skipping gain plot/table.")

print("\nDone. Outputs:")
print(f"- AUC summary:     {AUC_TXT}")
print(f"- SHAP beeswarm:   {os.path.join(PLOTS_DIR, f'shap_beeswarm_all_reps_{stamp}.png')}")
if (apoe_col is not None) and (echdc3_col is not None):
    print(f"- Dependence plot: {os.path.join(PLOTS_DIR, f'dependence_{apoe_col}_by_{echdc3_col}_{stamp}.png')}")
print(f"- Mean|SHAP| TSV:  {os.path.join(TABLES_DIR, f'mean_abs_shap_all_reps_{stamp}.tsv')}")
if len(interaction_mats) > 0:
    print(f"- Top interactions:{os.path.join(TABLES_DIR, f'top_shap_interactions_{stamp}.tsv')}")
print(f"- Avg gain plot:   {os.path.join(PLOTS_DIR, f'avg_gain_top20_{stamp}.png')}")
print(f"- Avg gain TSV:    {os.path.join(TABLES_DIR, f'avg_gain_full_{stamp}.tsv')}")
