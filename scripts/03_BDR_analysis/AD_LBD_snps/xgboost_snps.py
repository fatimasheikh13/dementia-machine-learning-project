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
RAW_FILE            = "/scratch/c.c2029098/dementia_ml_project/data/processed/ml_data/LBD_AD.raw"
EIGENVEC_FILE       = "/scratch/c.c2029098/dementia_ml_project/results/PCA/LBD_AD.eigenvec"
USE_FIRST_N_PCS     = 8

output_auc_file     = "/scratch/c.c2029098/dementia_ml_project/results/machine_learning/AD_LBD_snps/xgboost/AUC/xgboost_auc_nestedcv.txt"
output_params_file  = "/scratch/c.c2029098/dementia_ml_project/results/machine_learning/AD_LBD_snps/xgboost/best_params_nestedcv.txt"
plots_dir           = "/scratch/c.c2029098/dementia_ml_project/results/machine_learning/AD_LBD_snps/xgboost/plots"

for p in [plots_dir, os.path.dirname(output_auc_file), os.path.dirname(output_params_file)]:
    os.makedirs(p, exist_ok=True)

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

#load data 
df = pd.read_csv(RAW_FILE, sep=r"\s+")
df.columns = df.columns.str.strip() 

ev = pd.read_csv(EIGENVEC_FILE, sep=r"\s+", header=None)
m = ev.shape[1] - 2
ev.columns = ["FID","IID"] + [f"PC{i}" for i in range(1, m+1)]
pc_cols = [f"PC{i}" for i in range(1, min(USE_FIRST_N_PCS, m)+1)]
for c in pc_cols:
    ev[c] = pd.to_numeric(ev[c], errors="coerce")

# merge PCs (for test-time adjustment only)
df = df.merge(ev[["FID","IID"] + pc_cols], on=["FID","IID"], how="left")
df.columns = df.columns.str.strip()

# targets 0/1
y = pd.to_numeric(df["PHENOTYPE"], errors="coerce").replace({1:0, 2:1}).astype(int)

# SNP features define
NON_FEATURES = {"FID","IID","PAT","MAT","SEX","PHENOTYPE","PRS"}

# Build by dropping NON_FEATURES and al pc_cols explicitly
drop_cols = list(NON_FEATURES) + pc_cols
X_snp = df.drop(columns=drop_cols, errors="ignore")

# make all numeric
X_snp = X_snp.apply(pd.to_numeric, errors="coerce").fillna(0.0)

# make sure no pc's used in original model 
leaked = [c for c in X_snp.columns if c.upper().startswith("PC")]
if leaked:
    raise RuntimeError(f"PCs leaked into features: {leaked[:10]} (showing first 10)")

# Keep PCs separately for post-hoc adjustment
PCs_all = df[pc_cols].apply(pd.to_numeric, errors="coerce")

# post-hoc PC adjust 
def pc_adjust_scores(scores: np.ndarray, pcs_df: pd.DataFrame) -> np.ndarray:
    M = pcs_df.to_numpy(dtype=float, copy=True)
    if M.ndim != 2 or M.shape[0] < 2 or M.shape[1] == 0:
        return scores
    if np.isnan(M).any():
        col_means = np.nanmean(M, axis=0)
        col_means = np.where(np.isnan(col_means), 0.0, col_means)  # all-NaN col -> 0
        inds = np.where(np.isnan(M))
        M[inds] = np.take(col_means, inds[1])
    adj = LinearRegression().fit(M, scores)
    return scores - adj.predict(M)

# nested CV (SNPs only) 
def run_xgboost_nested_cv_snp_only(X_snp: pd.DataFrame,
                                   y: pd.Series,
                                   PCs_all: pd.DataFrame,
                                   pc_cols: list,
                                   num_reps=50, n_iter=50, test_size=0.2):
    test_aucs, test_aucs_pcadj, train_aucs = [], [], []
    all_best_params = []
    shap_values_list = []
    feature_importances = []

    param_dist = {
        "max_depth": randint(3, 10),
        "learning_rate": uniform(0.01, 0.3),
        "subsample": uniform(0.5, 0.5),
        "colsample_bytree": uniform(0.5, 0.5),
        "n_estimators": randint(100, 600),
        "gamma": uniform(0, 5),
        "min_child_weight": randint(1, 10),
    }

    idx_all = np.arange(len(y))

    for i in range(num_reps):
        print(f"\n--- Repetition {i+1}/{num_reps} ---")

        tr_idx, te_idx, y_tr, y_te = train_test_split(
            idx_all, y.values, test_size=test_size, random_state=i, stratify=y
        )
        X_train = X_snp.iloc[tr_idx]
        X_test  = X_snp.iloc[te_idx]
        PCs_test = PCs_all.iloc[te_idx]

        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=i)

        base = XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            n_jobs=-1,
            random_state=i,
            verbosity=0,
        )

        search = RandomizedSearchCV(
            estimator=base,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring="roc_auc",
            cv=inner_cv,
            n_jobs=-1,
            random_state=i,
            verbose=0,
            class_weight="balanced"
        )
        search.fit(X_train, y_tr)
        best = search.best_estimator_
        all_best_params.append(search.best_params_)

        # predictions
        y_tr_pred = best.predict_proba(X_train)[:, 1]
        y_te_pred = best.predict_proba(X_test)[:, 1]

        # AUCs
        auc_tr = roc_auc_score(y_tr, y_tr_pred)
        auc_te = roc_auc_score(y_te, y_te_pred)
        train_aucs.append(auc_tr)
        test_aucs.append(auc_te)

        # post-hoc PC adjustment (TEST only)
        y_te_pred_adj = pc_adjust_scores(y_te_pred, PCs_test[pc_cols])
        auc_te_adj = roc_auc_score(y_te, y_te_pred_adj)
        test_aucs_pcadj.append(auc_te_adj)

        print(f"Train AUC = {auc_tr:.4f} | Test AUC = {auc_te:.4f} | Test AUC (PC-adj) = {auc_te_adj:.4f}")

        # SHAP on SNPs only
        explainer = shap.Explainer(best)
        shap_vals = explainer(X_test)
        shap_values_list.append(np.abs(shap_vals.values).mean(axis=0))

        # Gain importance (SNPs only)
        booster = best.get_booster()
        feat_names = booster.feature_names  # should match SNP column names
        imp = booster.get_score(importance_type="gain")
        feature_importances.append({f: imp.get(f, 0.0) for f in feat_names})

    # Save AUCs 
    avg_auc_train = float(np.mean(train_aucs))
    avg_auc_test  = float(np.mean(test_aucs))
    avg_auc_test_adj = float(np.mean(test_aucs_pcadj))

    with open(output_auc_file, "w") as f:
        for i in range(num_reps):
            f.write(f"Seed {i} - Train AUC: {train_aucs[i]:.4f} - Test AUC: {test_aucs[i]:.4f} - Test AUC (PC-adj): {test_aucs_pcadj[i]:.4f}\n")
        f.write(f"\nAverage Train AUC: {avg_auc_train:.4f}\n")
        f.write(f"Average Test AUC: {avg_auc_test:.4f}\n")
        f.write(f"Average Test AUC (PC-adj): {avg_auc_test_adj:.4f}\n")

    print(f"\n Final Averages — Train: {avg_auc_train:.4f} | Test: {avg_auc_test:.4f} | Test (PC-adj): {avg_auc_test_adj:.4f}")

    # SHAP values
    shap_mat = np.vstack(shap_values_list)
    avg_shap = pd.Series(shap_mat.mean(axis=0), index=X_snp.columns).sort_values(ascending=False)
    top20_shap = avg_shap.head(20)
    plt.figure(figsize=(8, 6))
    top20_shap[::-1].plot(kind="barh")
    plt.title("Avg SHAP (Top20)")
    plt.xlabel("Mean(SHAP value)")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"avg_shap_importance_nestedcv_top20_SNPsOnly_{stamp}.png"))
    plt.close()

    # importance gain 
    imp_df = pd.DataFrame(feature_importances).reindex(columns=X_snp.columns, fill_value=0.0)
    avg_gain = imp_df.mean(axis=0).sort_values(ascending=False)
    top20_gain = avg_gain.head(20)
    plt.figure(figsize=(8, 6))
    top20_gain[::-1].plot(kind="barh")
    plt.title("Average XGBoost Gain Importance (Nested CV) — Top 20 SNPs")
    plt.xlabel("Average Gain")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"avg_gain_importance_nestedcv_top20_SNPsOnly_{stamp}.png"))
    plt.close()

    # tables
    avg_shap.to_csv(os.path.join(plots_dir, f"avg_shap_importance_full_SNPsOnly_{stamp}.tsv"),
                    sep="\t", header=["mean_abs_shap"])
    avg_gain.to_csv(os.path.join(plots_dir, f"avg_gain_importance_full_SNPsOnly_{stamp}.tsv"),
                    sep="\t", header=["avg_gain"])

    return test_aucs, avg_auc_test, all_best_params

# run model
aucs, avg_auc, all_best_params = run_xgboost_nested_cv_snp_only(
    X_snp=X_snp, y=y, PCs_all=PCs_all, pc_cols=pc_cols, num_reps=50, n_iter=50, test_size=0.2
)

