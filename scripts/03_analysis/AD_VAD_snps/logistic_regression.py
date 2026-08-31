# Unpenalised logistic regression using SNPs (AD vs VaD)
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

RAW_FILE        = "/scratch/c.c2029098/dementia_ml_project/data/processed/ml_data/VaD_AD.raw"
EIGENVEC_FILE   = "/scratch/c.c2029098/dementia_ml_project/results/PCA/VaD_AD.eigenvec"
USE_FIRST_N_PCS = 10
N_REPS          = 100
TEST_SIZE       = 0.2
N_FOLDS         = 10
OUT_DIR         = "/scratch/c.c2029098/dementia_ml_project/results/machine_learning/AD_VD_snps/logistic_regression/auc"
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
    adj = scores.copy()
    if pcs_df is None or pcs_df.empty:
        return adj
    mask = pcs_df.notna().all(axis=1).values
    if mask.sum() >= 2:
        X = pcs_df.loc[mask].to_numpy(dtype=float)
        lr = LinearRegression().fit(X, scores[mask])
        adj[mask] = scores[mask] - lr.predict(X)
    return adj

def run(df_raw, ev_df, pc_cols, snps, label):
    df_all = df_raw[["FID","IID","PHENOTYPE"] + snps].copy()
    y_all = pd.to_numeric(df_all["PHENOTYPE"], errors="coerce").replace({1:0, 2:1}).astype("Int64")
    X_all = df_all[snps].apply(pd.to_numeric, errors="coerce").astype(float)

    keep = y_all.isin([0,1]) & X_all.notna().all(axis=1)
    df_all = df_all.loc[keep].reset_index(drop=True)
    X_all  = X_all.loc[keep].reset_index(drop=True)
    y_all  = y_all.loc[keep].to_numpy(dtype=int)

    rng = np.random.default_rng(42)
    cv_aucs, te_aucs, te_aucs_adj = [], [], []

    # confusion-matrix aggregates
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

        # 10-fold CV on training (unpenalised)
        kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        fold_aucs = []
        for tr_i, val_i in kf.split(X_tr, y_tr):
            lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
            lr.fit(X_tr.iloc[tr_i], y_tr[tr_i])
            p_val = lr.predict_proba(X_tr.iloc[val_i])[:, 1]
            fold_aucs.append(roc_auc_score(y_tr[val_i], p_val))
        cv_aucs.append(np.mean(fold_aucs))

        # Train full test (unpenalised)
        lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
        lr.fit(X_tr, y_tr)
        p_test = lr.predict_proba(X_te)[:, 1]
        te_aucs.append(roc_auc_score(y_te, p_test))

        # PC adjustment of test probabilities (residualize on PCs)
        test_ids = df_all.loc[te_idx, ["FID","IID"]]
        PCs_te = test_ids.merge(ev_df, on=["FID","IID"], how="left")[pc_cols]
        p_test_adj = pc_adjust(p_test, PCs_te)
        te_aucs_adj.append(roc_auc_score(y_te, p_test_adj))

        # confusion matrix + metrics
        y_prob = p_test
        y_pred = (y_prob >= 0.5).astype(int)
        cm = confusion_matrix(y_te, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        acc  = accuracy_score(y_te, y_pred)
        prec = precision_score(y_te, y_pred, zero_division=0)
        rec  = recall_score(y_te, y_pred)
        spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        f1   = f1_score(y_te, y_pred)
        aucp = roc_auc_score(y_te, y_prob)

        tns.append(tn); fps.append(fp); fns.append(fn); tps.append(tp)
        accs.append(acc); precs.append(prec); recalls.append(rec); specs.append(spec); f1s.append(f1); auc_probs.append(aucp)

        if (rep % PRINT_EVERY) == 0:
            print(f"Rep {rep+1}/{N_REPS}: CV={cv_aucs[-1]:.4f}, Test={te_aucs[-1]:.4f}, Test(PC-adj)={te_aucs_adj[-1]:.4f}")

    # Save AUCs
    pd.DataFrame({"rep": np.arange(1, N_REPS+1), "cv_auc": cv_aucs}).to_csv(
        os.path.join(OUT_DIR, f"{label}_cv_aucs.tsv"), sep="\t", index=False
    )
    pd.DataFrame({"rep": np.arange(1, N_REPS+1), "test_auc": te_aucs}).to_csv(
        os.path.join(OUT_DIR, f"{label}_test_aucs.tsv"), sep="\t", index=False
    )
    pd.DataFrame({"rep": np.arange(1, N_REPS+1), "test_auc_pc_adjusted": te_aucs_adj}).to_csv(
        os.path.join(OUT_DIR, f"{label}_test_aucs_pc_adjusted.tsv"), sep="\t", index=False
    )

    # Summary
    print(f"\n{label} ({N_REPS} reps)")
    print(f"CV AUC mean: {np.mean(cv_aucs):.4f}")
    print(f"Test AUC mean: {np.mean(te_aucs):.4f}")
    print(f"Test AUC (PC-adj) mean: {np.mean(te_aucs_adj):.4f}  [K={len(pc_cols)} PCs]")

    # Confusion-matrix metrics (means)
    metrics_summary = pd.DataFrame([{
        "TN_mean": float(np.mean(tns)), "FP_mean": float(np.mean(fps)),
        "FN_mean": float(np.mean(fns)), "TP_mean": float(np.mean(tps)),
        "accuracy_mean": float(np.mean(accs)),
        "precision_mean": float(np.mean(precs)),
        "recall_mean": float(np.mean(recalls)),
        "specificity_mean": float(np.mean(specs)),
        "f1_mean": float(np.mean(f1s)),
        "auc_prob_mean": float(np.mean(auc_probs)),
    }])
    metrics_path = os.path.join(OUT_DIR, f"{label}_confusionMetrics_summary.tsv")
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