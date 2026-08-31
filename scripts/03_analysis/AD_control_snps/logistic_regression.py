#!/usr/bin/env python3
# logistic regression using snps 
import os
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

RAW_FILE        = "/scratch/c.c2029098/dementia_ml_project/data/processed/ml_data/BDR_AD_control.raw"
EIGENVEC_FILE   = "/scratch/c.c2029098/dementia_ml_project/results/PCA/BDR_AD_control.eigenvec"
USE_FIRST_N_PCS = 7
N_REPS          = 100
TEST_SIZE       = 0.2
N_FOLDS         = 10
OUT_DIR         = "/scratch/c.c2029098/dementia_ml_project/results/machine_learning/AD_control/logistic_regression/all_snps"
PRINT_EVERY     = 10

os.makedirs(OUT_DIR, exist_ok=True)

def load_raw(path):
    return pd.read_csv(path, sep=r"\s+")

def load_pcs(path, k=None):
    ev = pd.read_csv(path, sep=r"\s+")
    pc_cols = [c for c in ev.columns if str(c).startswith("PC")]
    if k is not None:
        pc_cols = [c for c in pc_cols if int(c[2:]) <= k]
    for c in pc_cols:
        ev[c] = pd.to_numeric(ev[c], errors="coerce")
    return ev[["FID", "IID"] + pc_cols].copy(), pc_cols

def get_snp_cols(df):
    non_feats = {"FID","IID","PAT","MAT","SEX","PHENOTYPE"}
    return [c for c in df.columns if c not in non_feats and not str(c).startswith("PC")]

def pc_adjust(scores, pcs_df):
    X = pcs_df.to_numpy(dtype=float)
    lr = LinearRegression().fit(X, scores)
    return scores - lr.predict(X)

def run(df_raw, ev_df, pc_cols, snps, label):
    df_all = df_raw[["FID","IID","PHENOTYPE"] + snps].copy()
    y_all = pd.to_numeric(df_all["PHENOTYPE"], errors="coerce").replace({1:0, 2:1}).astype(int).to_numpy()
    X_all = df_all[snps].apply(pd.to_numeric, errors="coerce").astype(float)

    rng = np.random.default_rng(42)
    cv_aucs, te_aucs, te_aucs_adj = [], [], []

    tns, fps, fns, tps = [], [], [], []
    accs, precs, recalls, specs, f1s, auc_probs = [], [], [], [], [], []

    for rep in range(N_REPS):
        seed = int(rng.integers(0, 2**31 - 1))
        idx = np.arange(len(df_all))
        tr_idx, te_idx, y_tr, y_te = train_test_split(
            idx, y_all, test_size=TEST_SIZE, stratify=y_all, random_state=seed
        )
        X_tr = X_all.iloc[tr_idx]
        X_te = X_all.iloc[te_idx]

        # 10-fold CV on training 
        kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        fold_aucs = []
        for tr_i, val_i in kf.split(X_tr, y_tr):
            lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
            lr.fit(X_tr.iloc[tr_i], y_tr[tr_i])
            p_val = lr.predict_proba(X_tr.iloc[val_i])[:, 1]
            fold_aucs.append(roc_auc_score(y_tr[val_i], p_val))
        cv_aucs.append(np.mean(fold_aucs))

        # Train on full train, evaluate on test 
        lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
        lr.fit(X_tr, y_tr)
        p_test = lr.predict_proba(X_te)[:, 1]
        te_aucs.append(roc_auc_score(y_te, p_test))

        # PC adjustment of test probabilities
        test_ids = df_all.loc[te_idx, ["FID","IID"]]
        PCs_te = test_ids.merge(ev_df, on=["FID","IID"], how="left")[pc_cols]
        p_test_adj = pc_adjust(p_test, PCs_te)
        te_aucs_adj.append(roc_auc_score(y_te, p_test_adj))

        # Confusion-matrix metrics
        y_pred = (p_test >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_te, y_pred, labels=[0, 1]).ravel()
        accs.append(accuracy_score(y_te, y_pred))
        precs.append(precision_score(y_te, y_pred, zero_division=0))
        recalls.append(recall_score(y_te, y_pred))
        specs.append(tn / (tn + fp))
        f1s.append(f1_score(y_te, y_pred))
        auc_probs.append(roc_auc_score(y_te, p_test))
        tns.append(tn); fps.append(fp); fns.append(fn); tps.append(tp)

        if (rep % PRINT_EVERY) == 0:
            print(f"Rep {rep+1}/{N_REPS}: CV={cv_aucs[-1]:.4f}, Test={te_aucs[-1]:.4f}, Test(PC-adj)={te_aucs_adj[-1]:.4f}")

    # Save per-rep AUCs
    pd.DataFrame({"rep": np.arange(1, N_REPS+1), "cv_auc": cv_aucs}).to_csv(
        os.path.join(OUT_DIR, f"{label}_cv_aucs.tsv"), sep="\t", index=False
    )
    pd.DataFrame({"rep": np.arange(1, N_REPS+1), "test_auc": te_aucs}).to_csv(
        os.path.join(OUT_DIR, f"{label}_test_aucs.tsv"), sep="\t", index=False
    )
    pd.DataFrame({"rep": np.arange(1, N_REPS+1), "test_auc_pc_adjusted": te_aucs_adj}).to_csv(
        os.path.join(OUT_DIR, f"{label}_test_aucs_pc_adjusted.tsv"), sep="\t", index=False
    )

    # Summary (means)
    print(f"\n{label} ({N_REPS} reps)")
    print(f"CV AUC mean: {np.mean(cv_aucs):.4f}")
    print(f"Test AUC mean: {np.mean(te_aucs):.4f}")
    print(f"Test AUC (PC-adj) mean: {np.mean(te_aucs_adj):.4f}  [K={len(pc_cols)} PCs]")

    # Confusion-matrix metric means
    metrics_summary = pd.DataFrame([{
        "TN_mean": np.mean(tns), "FP_mean": np.mean(fps),
        "FN_mean": np.mean(fns), "TP_mean": np.mean(tps),
        "accuracy_mean": np.mean(accs),
        "precision_mean": np.mean(precs),
        "recall_mean": np.mean(recalls),
        "specificity_mean": np.mean(specs),
        "f1_mean": np.mean(f1s),
        "auc_prob_mean": np.mean(auc_probs),
    }])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_path = os.path.join(OUT_DIR, f"{label}_confusionMetrics_summary_{stamp}.tsv")
    metrics_summary.to_csv(metrics_path, sep="\t", index=False)
    print("\nConfusion-matrix metrics (means across reps):")
    print(metrics_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("Saved:", metrics_path)

def main():
    df = load_raw(RAW_FILE)
    ev_df, pc_cols = load_pcs(EIGENVEC_FILE, k=USE_FIRST_N_PCS)
    snps = get_snp_cols(df)
    label = os.path.splitext(os.path.basename(RAW_FILE))[0]
    run(df, ev_df, pc_cols, snps, label)

main()
