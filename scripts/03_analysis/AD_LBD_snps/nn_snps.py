#!/usr/bin/env python3
import os, warnings, contextlib
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from skorch import NeuralNetClassifier
from skorch.callbacks import GradientNormClipping, EarlyStopping, EpochScoring
from skorch.dataset import ValidSplit
from scipy.stats import randint, reciprocal, uniform
import shap

warnings.filterwarnings("ignore")

@contextlib.contextmanager
def suppress_stdout_stderr():
    import sys
    with open(os.devnull, 'w') as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull
        try: yield
        finally: sys.stdout, sys.stderr = old_stdout, old_stderr

# nn model
class SNPNet(nn.Module):
    def __init__(self, num_features=77, hidden_units=24,
                 dropout_input=0.25, dropout_hidden=0.6, noise_std=0.05):
        super().__init__()
        self.noise_std = noise_std
        self.ln_in = nn.LayerNorm(num_features)
        self.drop_in = nn.Dropout(dropout_input)
        self.fc1 = nn.Linear(num_features, hidden_units)
        self.ln1 = nn.LayerNorm(hidden_units)
        self.drop_h = nn.Dropout(dropout_hidden)
        self.fc_out = nn.Linear(hidden_units, 1)

    def forward(self, x):
        x = self.ln_in(x)
        if self.training and self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        x = self.drop_in(x)
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.drop_h(x)
        return self.fc_out(x).squeeze(1)  # logits

# PC post - hoc 
def load_pcs(eig_path: str, use_first_n: int | None = 10) -> pd.DataFrame:
    ev = pd.read_csv(eig_path, sep=r"\s+", engine="python")
    cols = list(ev.columns)
    if len(cols) >= 2 and (str(cols[0]).upper() != "FID" or str(cols[1]).upper() != "IID"):
        m = len(cols) - 2
        ev.columns = ["FID", "IID"] + [f"PC{i}" for i in range(1, m+1)]
    else:
        ev.rename(columns={cols[0]: "FID", cols[1]: "IID"}, inplace=True)
    pc_cols = [c for c in ev.columns if str(c).startswith("PC")]
    if use_first_n is not None:
        pc_cols = [c for c in pc_cols if int(c[2:]) <= use_first_n]
    key = ev["FID"].astype(str).str.strip() + "||" + ev["IID"].astype(str).str.strip()
    return ev.set_index(key)[pc_cols].apply(pd.to_numeric, errors="coerce")

from sklearn.linear_model import LinearRegression
def pc_adjust_scores(scores: np.ndarray, pcs_df: pd.DataFrame) -> np.ndarray:
    if pcs_df is None or pcs_df.shape[1] == 0:
        return scores
    mask = pcs_df.notna().all(axis=1).values
    adj = scores.copy()
    if mask.sum() >= 2:
        Xpcs = pcs_df.loc[mask].values
        lr = LinearRegression()
        lr.fit(Xpcs, scores[mask])
        adj[mask] = scores[mask] - lr.predict(Xpcs)
    return adj

def run_main_pipeline():
    # directories 
    base_dir = "/scratch/c.c2029098/dementia_ml_project/results/machine_learning/AD_LBD_snps/neural_network"
    plots_dir = os.path.join(base_dir, "plots")
    auc_dir = os.path.join(base_dir, "AUC")
    for d in (plots_dir, auc_dir): os.makedirs(d, exist_ok=True)

    RAW = "/scratch/c.c2029098/dementia_ml_project/data/processed/ml_data/LBD_AD.raw"
    EIG = "/scratch/c.c2029098/dementia_ml_project/results/PCA/LBD_AD.eigenvec"
    PCs_all = load_pcs(EIG, use_first_n=10)

    # load raw files
    df = pd.read_csv(RAW, sep=r"\s+")
    y_all = pd.to_numeric(df["PHENOTYPE"], errors="coerce").replace({1:0, 2:1})
    keep = y_all.isin([0,1])
    df = df.loc[keep].copy()
    y_all = y_all.loc[keep].astype(np.float32).values

    df["KEY"] = df["FID"].astype(str).str.strip() + "||" + df["IID"].astype(str).str.strip()
    df.set_index("KEY", inplace=True)

    NON_FEATURES = {"FID","IID","PAT","MAT","SEX","PHENOTYPE","PRS","KEY"}
    feat_names = [c for c in df.columns if c not in NON_FEATURES and not str(c).startswith("PC")]
    X_all_df = df[feat_names].apply(pd.to_numeric, errors="coerce")  # keep as DF with KEY index

    print(f"SNP features: {len(feat_names)}  |  PCs loaded: {PCs_all.shape[1]}")

    seeds = list(range(30))
    aucs, aucs_pcadj, train_aucs = [], [], []
    shap_values_list, perm_values_list = [], []
    feature_label = "snps_only_with_pc_residualization"

    for seed in seeds:
        print(f"\n Seed {seed}")
        np.random.seed(seed); torch.manual_seed(seed)

        # ----- Outer split (keep KEY index) -----
        X_tr_df, X_te_df, y_tr, y_te = train_test_split(
            X_all_df, y_all, test_size=0.3, random_state=seed, stratify=y_all
        )

        # pos_weight for imbalance 
        n_pos = float((y_tr == 1).sum()); n_neg = float((y_tr == 0).sum())
        pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], dtype=torch.float32)

        # nn pipeline
        net_base = NeuralNetClassifier(
            SNPNet,
            module__num_features=X_tr_df.shape[1],
            optimizer=optim.AdamW,
            criterion=nn.BCEWithLogitsLoss,
            criterion__pos_weight=pos_weight,        
            iterator_train__shuffle=True,
            iterator_train__drop_last=True,
            callbacks=[
                GradientNormClipping(1.0),
                # Track ROC AUC on the internal valid split:
                EpochScoring(scoring='roc_auc', on_train=False,
                             name='valid_roc_auc', lower_is_better=False),
                # Early stop on best valid ROC AUC:
                EarlyStopping(monitor='valid_roc_auc', patience=12,
                              lower_is_better=False, threshold=1e-4)
            ],
            verbose=0,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            train_split=ValidSplit(0.2, stratified=True),  # internal valid split per CV fold
        )

        tofloat32 = FunctionTransformer(lambda X: X.astype(np.float32), accept_sparse=True)
        pipe = Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('sc',  StandardScaler()),
            ('to32', tofloat32),
            ('net', net_base),
        ])

        # random search
        param_dist = {
            'net__module__hidden_units'     : randint(8, 33),        
            'net__module__dropout_input'    : uniform(0.15, 0.35),   
            'net__module__dropout_hidden'   : uniform(0.5, 0.4),   
            'net__module__noise_std'        : uniform(0.00, 0.10),  
            'net__lr'                       : reciprocal(5e-5, 1e-3),
            'net__optimizer__weight_decay'  : reciprocal(1e-4, 5e-2),
            'net__batch_size'               : randint(64, 193),      
            'net__max_epochs'               : randint(60, 161),     
        }

        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        rs = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_dist,
            n_iter=30,
            scoring='roc_auc',
            cv=inner_cv,
            n_jobs=-1,
            random_state=seed,
            verbose=0,
            error_score='raise',
            refit=True,
        )

        with suppress_stdout_stderr():
            rs.fit(X_tr_df, y_tr)

        best_pipe = rs.best_estimator_

        # evaluate on hold out test set 
        p_train = best_pipe.predict_proba(X_tr_df)[:, 1]
        p_test  = best_pipe.predict_proba(X_te_df)[:, 1]

        # Post-hoc PC residualisation on TEST
        PCs_test = PCs_all.reindex(X_te_df.index) if (PCs_all is not None and not PCs_all.empty) else None
        p_test_adj = pc_adjust_scores(p_test, PCs_test)

        auc_train = roc_auc_score(y_tr, p_train)
        auc_test  = roc_auc_score(y_te, p_test)
        auc_test_adj = roc_auc_score(y_te, p_test_adj)

        train_aucs.append(auc_train)
        aucs.append(auc_test)
        aucs_pcadj.append(auc_test_adj)

        print(f"Train AUC = {auc_train:.4f} | Test AUC = {auc_test:.4f} | Test AUC (PC-adj) = {auc_test_adj:.4f}")

        # SHAP
        def model_predict(Xin):
            return best_pipe.predict_proba(Xin)[:, 1]
        bg_n = min(200, X_tr_df.shape[0])
        bg_idx = np.random.choice(X_tr_df.shape[0], bg_n, replace=False)
        background = X_tr_df.iloc[bg_idx].values
        explainer = shap.Explainer(model_predict, background)
        shap_vals = explainer(X_te_df.values)
        shap_values_list.append(np.abs(shap_vals.values).mean(axis=0))

        # Permutation importance 
        perm = permutation_importance(
            best_pipe, X_te_df, y_te,
            scoring='roc_auc', n_repeats=10, random_state=seed
        )
        perm_values_list.append(perm.importances_mean)

    # Save AUCs 
    auc_file = os.path.join(auc_dir, f"{feature_label}_all_auc_seeds.txt")
    with open(auc_file, "w") as f:
        for i, seed in enumerate(seeds):
            f.write(f"Seed {seed} - Train ROC AUC: {train_aucs[i]:.4f} - Test ROC AUC: {aucs[i]:.4f} - Test ROC AUC (PC-adj): {aucs_pcadj[i]:.4f}\n")
        f.write(f"\nAverage Train ROC AUC: {np.nanmean(train_aucs):.4f}\n")
        f.write(f"Average Test ROC AUC: {np.mean(aucs):.4f}\n")
        f.write(f"Average Test ROC AUC (PC-adj): {np.mean(aucs_pcadj):.4f}\n")

    # Aggregate importances 
    mean_shap = np.mean(np.stack(shap_values_list), axis=0)
    mean_perm = np.mean(np.stack(perm_values_list), axis=0)
    imp_df = pd.DataFrame({
        "feature": feat_names,
        "mean_abs_shap": mean_shap,
        "perm_importance": mean_perm
    }).sort_values("mean_abs_shap", ascending=False)

    imp_df.to_csv(os.path.join(plots_dir, f"{feature_label}_feature_importance.tsv"), sep="\t", index=False)

    topk = min(20, len(feat_names))
    top_df = imp_df.head(topk)

    plt.figure(figsize=(10,6))
    plt.barh(top_df["feature"][::-1], top_df["mean_abs_shap"][::-1])
    plt.title("Avg SHAP (Top 20)")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"{feature_label}_avg_shap_top20.png"))
    plt.close()

    top_df2 = imp_df.sort_values("perm_importance", ascending=False).head(topk)
    plt.figure(figsize=(10,6))
    plt.barh(top_df2["feature"][::-1], top_df2["perm_importance"][::-1])
    plt.title("Permutation (Top 20)")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"{feature_label}_avg_permutation_top20.png"))
    plt.close()

run_main_pipeline()
