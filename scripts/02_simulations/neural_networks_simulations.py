#!/usr/bin/env python3
# Core libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib import colors
import seaborn as sns
import os
import warnings

# Stats / modeling
import scipy.stats as sp
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint, uniform

# XGBoost
from xgboost import XGBClassifier

# PyTorch / Skorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from skorch import NeuralNetClassifier
from skorch.callbacks import EpochScoring, EarlyStopping

# Define the Risk model - 

def set_model(model_name, alpha, theta):
    if model_name == 'multiplicative':
        x = np.hstack((np.array([alpha * ((1 + theta) ** 4),
                                 alpha * ((1 + theta) ** 2),
                                 alpha,
                                 alpha * ((1 + theta) ** 2),
                                 alpha * (1 + theta)]),
                       np.repeat(alpha, 4)))
    elif model_name == 'dab1':
        # Custom risk model: additive at locus B (DAB1) when APOE genotype == 2
        # Odds ratios specific to our scenario - this model is used for our epsistatic effect not the above multiplicative 
        x = np.array([
            1, 1, 1,    # APOE = 0 (AA)
            apoe_or, apoe_or, apoe_or,    # APOE = 1 (Aa)
            apoe_or ** 2, (apoe_or ** 2) * dab1_or, (apoe_or ** 2) * (dab1_or ** 2) # APOE = 2 (aa), DAB1 = 1 or 2 has elevated risk
        ])
    else:
        raise ValueError('Model type not recognised')

    return x
    
# The append_missing_counts function checks if all expected genotype keys are present in the data.
# If any are missing, it adds those keys with zero counts.
# This is useful to avoid errors in analyses that expect a complete set of genotype categories.

def append_missing_counts(counts):
    # Define all expected keys
    all_keys = ['00', '10', '01', '11', '02', '20', '12', '21', '22']

    # Check which keys are present in the input Series
    present = [x in counts.index for x in all_keys]

    # If not all keys are present, add the missing ones with count 0
    if sum(present) != len(all_keys):
        missing_keys = [key for key, is_present in zip(all_keys, present) if not is_present]

        # Create a Series for the missing counts, initialized to 0
        extra_counts = pd.Series(np.zeros(len(missing_keys), dtype=int),
                                 index=missing_keys)

        # Use pd.concat to combine the original counts with the new missing counts
        counts = pd.concat([counts, extra_counts])

    return counts
    
# This pprint_vector function displays genotype combination data in a nicely formatted way
def pprint_vector(vector):
    x = pd.DataFrame(np.reshape(vector, (3, 3)),
                     columns=["bb", "Bb", "BB"],
                     index=["AA", "Aa", "aa"])
    print('Risk alleles: A/B\n')
    print(x)


# Set HWE 

# Calculate exact genotype frequencies assuming HWE

def set_hwe_at_locus(maf):
    p = maf
    q = 1 - p

    return np.hstack((p ** 2, 2 * p * q, q ** 2))[::-1]

# Calculates the joint genotype probabilities for two loci under the assumption that the two loci are independent.
def set_control_pgi(hwe_a, hwe_b):
    # creates P(Gi), the vector of probabilities of each two-locus genotype combination
    pgi_controls = np.hstack((
        hwe_a[0] * hwe_b,
        hwe_a[1] * hwe_b,
        hwe_a[2] * hwe_b\
    ))

    return pgi_controls
    
# Adjusts control genotype frequencies by risk multipliers (the model vector) to simulate frequencies in cases.
def set_case_pgi(pgi_controls, model):
    pgi_cases = np.multiply(pgi_controls, model)

    return pgi_cases

# This function samples length number of items randomly but weighted by specified probabilities from the given options 
def weighted_random_choice(options, length, probabilities):
    # lifted from here: https://glowingpython.blogspot.com/2012/09/weighted-random-choice.html
    t = np.cumsum(probabilities)
    s = np.sum(probabilities)
    
    return options[np.searchsorted(t, np.random.rand(length) * s)]

# Sets the random seed so the simulation results are reproducible.
# This ensures every run with the same seed gives the same simulated data.
    
def simulate(maf_a, maf_b, n_cases, n_controls, seed, model):
    print('Current seed: {}'.format(seed))
    np.random.seed(seed)

    # Calculate genotype frequencies at each locus assuming HWE
    hwe_a, hwe_b = [set_hwe_at_locus(m) for m in (maf_a, maf_b)]
    # Calculate joint genotype frequencies in controls
    control_pgi = set_control_pgi(hwe_a, hwe_b)
    # Adjust joint genotype frequencies for cases using the genetic risk model
    case_pgi = set_case_pgi(control_pgi, model)

    # Define the genotype combinations as strings
    alleles = np.array(["aa_bb", "aa_Bb", "aa_BB", "Aa_bb", "Aa_Bb", "Aa_BB", "AA_bb", "AA_Bb", "AA_BB"])[::-1]
    print(alleles)
    # Sample genotype combinations for cases and controls based on probabilities
    interactions = np.hstack((
        weighted_random_choice(alleles, n_cases, case_pgi),
        weighted_random_choice(alleles, n_controls, control_pgi)
    ))

    # Convert string genotypes to numeric genotype format for analysis
    snps = (pd.DataFrame({'snps': interactions})
              .snps.str.split('_', expand=True)
              .replace({'aa': 2, 'aA': 1, 'Aa': 1, 'AA': 0,
                        'bb': 2, 'bB': 1, 'Bb': 1, 'BB': 0}).infer_objects(copy=False)
              .rename(columns={0: 'SNP_A', 1: 'SNP_B'}))
    # Add disease status label    
    snps['Status'] = np.hstack((np.repeat(1, n_cases), np.repeat(0, n_controls)))

    # Return the final simulated dataset
    return snps.loc[:, ['Status', 'SNP_A', 'SNP_B']]

seed = 1
alpha = 3.33 
theta = 1.5
maf_a = 0.36  # APOE
maf_b = 0.04  # DAB1
apoe_or = np.exp(1.2017)  
dab1_or = 2.28
n_cases = 10000
n_controls = 10000
mod = set_model('dab1', alpha, theta)
df = simulate(maf_a, maf_b, n_cases, n_controls, seed=seed, model=mod)

import matplotlib
import matplotlib.pyplot as plt

# imports
import os, sys, json, warnings, contextlib
import numpy as np
import pandas as pd
import scipy.stats as sp
import shap

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.model_selection import StratifiedKFold, train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping, EpochScoring


# Directories
BASE_DIR = "/scratch/c.c2029098/dementia_ml_project/results/simulations/neural_networks"
LOSS_DIR = os.path.join(BASE_DIR, "loss_graphs")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
AUC_DIR   = os.path.join(BASE_DIR, "AUC")
for d in (LOSS_DIR, PLOTS_DIR, AUC_DIR):
    os.makedirs(d, exist_ok=True)

warnings.filterwarnings("ignore")

# Silence stdout/stderr where needed
@contextlib.contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, 'w') as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

# JSON-safe converter (prevents TypeError on dump)
def to_jsonable(obj):
    """Recursively convert objects to JSON-safe values (e.g., classes -> names, numpy scalars -> py)."""
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if hasattr(obj, "__name__"):   # classes/functions like torch.optim.SGD
        return obj.__name__
    return str(obj)

# Network
class SNPNet(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        self.fc1 = nn.Linear(2, 4)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))  # output in [0,1] for BCELoss
        return x.squeeze(1)

# Your simulate() and set_model() functions are assumed available
# Data simulation wrapper
def sim_and_split(maf_a, maf_b, n_cases, n_controls, seed, model, test_size=0.3):
    """
    Uses your simulate(...) -> DataFrame with columns ['Status','SNP_A','SNP_B'].
    SNP_A = APOE dosage, SNP_B = DAB1 dosage (assumed).
    """
    df = simulate(maf_a, maf_b, n_cases, n_controls, seed, model)
    X = df[['SNP_A', 'SNP_B']].values.astype(np.float32)
    y = df['Status'].values.astype(np.float32)
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

# Custom ROC AUC scorer using predict_proba
def roc_auc_from_proba(net, X, y):
    proba = net.predict_proba(X)
    if proba.ndim == 2 and proba.shape[1] > 1:
        proba = proba[:, 1]
    else:
        proba = proba.ravel()
    return float(roc_auc_score(y, proba))

# Main simulation + SHAP aggregation
def neural_network(num_reps=30, n_iter=30, **kwargs):
    aucs = []
    best_params_list = []
    last_best_model = None

    # Track per-run mean |SHAP| for features (we will reorder as [DAB1, APOE])
    shap_values_list = []  # each entry: np.array([|SHAP|_DAB1, |SHAP|_APOE])

    param_dist = {
        'module__dropout_rate': sp.uniform(0.1, 0.3),
        'lr': sp.reciprocal(1e-5, 1e-2),
        'batch_size': sp.randint(48, 128),
        'max_epochs': sp.randint(60, 200),
        'optimizer': [optim.SGD],  # fixed optimizer (class); JSON-safe helper will stringify
    }

    for i in range(num_reps):
        seed = i
        np.random.seed(seed)
        print(f"\nSimulation run {i+1}/{num_reps} (seed={seed})")

        X_train, X_test, y_train, y_test = sim_and_split(seed=seed, **kwargs)

        # Scale
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_test  = scaler.transform(X_test).astype(np.float32)

        # AUC callbacks
        auc_val_cb = EpochScoring(
            scoring=roc_auc_from_proba, name='valid_auc',
            lower_is_better=False, on_train=False
        )
        auc_train_cb = EpochScoring(
            scoring=roc_auc_from_proba, name='train_auc',
            lower_is_better=False, on_train=True
        )

        # Skorch classifier
        net = NeuralNetClassifier(
            SNPNet,
            optimizer=optim.SGD,           # can be overridden by RandomizedSearchCV params
            criterion=nn.BCELoss,          # expects float targets
            max_epochs=200,
            iterator_train__shuffle=True,
            callbacks=[EarlyStopping(patience=10), auc_val_cb, auc_train_cb],
            verbose=0,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        random_search = RandomizedSearchCV(
            estimator=net,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring='roc_auc',
            cv=cv,
            verbose=0,
            n_jobs=-1,
            random_state=seed,
            error_score='raise'
        )

        with suppress_stdout_stderr():
            random_search.fit(X_train, y_train)

        best_model = random_search.best_estimator_
        last_best_model = best_model

        # store best params (JSON-safe)
        best_params_list.append(to_jsonable(random_search.best_params_))

        # Evaluate AUC
        y_test_pred = best_model.predict_proba(X_test)
        if y_test_pred.ndim == 2 and y_test_pred.shape[1] > 1:
            y_test_pred = y_test_pred[:, 1]
        else:
            y_test_pred = y_test_pred.ravel()
        test_auc = roc_auc_score(y_test, y_test_pred)
        aucs.append(test_auc)

        # SHAP per-run (match orientation of your other script)
        # Downsample test points to keep SHAP fast when needed
        explain_n = min(1000, X_test.shape[0])
        te_idx = np.random.choice(X_test.shape[0], explain_n, replace=False)
        X_test_sub = X_test[te_idx]

        # model_predict uses the underlying torch module_
        def model_predict(X_input):
            X_np = np.asarray(X_input, dtype=np.float32)
            X_tensor = torch.from_numpy(X_np).to(best_model.device)
            with torch.no_grad():
                preds = best_model.module_(X_tensor)
            return preds.cpu().numpy()

        # Use TRAIN as background (exactly like your other script)
        explainer = shap.Explainer(model_predict, X_train)
        shap_values = explainer(X_test_sub)

        # Mean absolute SHAP across samples (in feature order of X: ['SNP_A','SNP_B'] = [APOE, DAB1])
        mean_abs_shap_run = np.abs(shap_values.values).mean(axis=0)

        # Reorder to desired label order: [DAB1, APOE]
        # X columns were [SNP_A (APOE), SNP_B (DAB1)] -> indices [0, 1]
        mean_abs_shap_run = mean_abs_shap_run[[1, 0]]
        shap_values_list.append(mean_abs_shap_run)

        # Save per-run curves
        hist = best_model.history
        loss_train = hist[:, 'train_loss']
        loss_val = hist[:, 'valid_loss'] if 'valid_loss' in hist[0] else None
        auc_train = hist[:, 'train_auc'] if 'train_auc' in hist[0] else None
        auc_val = hist[:, 'valid_auc'] if 'valid_auc' in hist[0] else None

        epochs = range(1, len(loss_train)+1)
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4))

        ax0.plot(epochs, loss_train, label='Train')
        if loss_val is not None:
            ax0.plot(epochs, loss_val, label='Val')
        ax0.set_xlabel("Epoch"); ax0.set_ylabel("Loss"); ax0.set_title("Loss curve")
        ax0.legend()

        if auc_train is not None and auc_val is not None:
            min_len = min(len(epochs), len(auc_train), len(auc_val))
            ax1.plot(epochs[:min_len], auc_train[:min_len], label='Train')
            ax1.plot(epochs[:min_len], auc_val[:min_len], label='Val')
            ax1.set_xlabel("Epoch"); ax1.set_ylabel("AUC"); ax1.set_title("AUC curve")
            ax1.legend()
        else:
            ax1.axis('off')

        fig.tight_layout()
        fig.savefig(os.path.join(LOSS_DIR, f"run_{i+1:03d}.png"), dpi=150)
        plt.close(fig)

    # Save AUCs + params
    avg_auc = float(np.mean(aucs)) if len(aucs) else float("nan")
    auc_txt_path = os.path.join(AUC_DIR, "all_auc_seeds.txt")
    with open(auc_txt_path, "w") as f:
        for i, auc in enumerate(aucs, start=1):
            f.write(f"Run {i:03d} AUC: {auc:.4f}\n")
        f.write(f"\nAverage Test ROC AUC over {num_reps} simulations: {avg_auc:.4f}\n")

    params_json_path = os.path.join(AUC_DIR, "best_params_per_run.json")
    with open(params_json_path, "w") as f:
        json.dump(to_jsonable(best_params_list), f, indent=2, ensure_ascii=False)

    summary_path = os.path.join(AUC_DIR, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Num runs: {num_reps}\n")
        f.write(f"Average Test ROC AUC: {avg_auc:.4f}\n")
        f.write(f"AUC file: {auc_txt_path}\n")
        f.write(f"Params file: {params_json_path}\n")
        f.write(f"Loss graphs dir: {LOSS_DIR}\n")

    print(f"\nAverage Test ROC AUC over {num_reps} simulations: {avg_auc:.4f}")
    print(f"AUCs written to: {auc_txt_path}")
    print(f"Best params (per run) written to: {params_json_path}")
    print(f"Summary written to: {summary_path}")
    print(f"Loss graphs saved in: {LOSS_DIR}")

    # Aggregate SHAP across runs and plot (vertical bars, DAB1 then APOE)
    try:
        if len(shap_values_list) == 0:
            print("[INFO] No SHAP values collected; writing zero-importance placeholder.")
            mean_shap = np.zeros(2, dtype=float)  # [DAB1, APOE]
        else:
            mean_shap = np.mean(np.stack(shap_values_list, axis=0), axis=0)  # shape (2,)

        labels = ['DAB1', 'APOE']  # desired order
        os.makedirs(PLOTS_DIR, exist_ok=True)

        # Save TSV
        imp_path = os.path.join(PLOTS_DIR, "feature_importance_shap.tsv")
        pd.DataFrame({"feature": labels, "mean_abs_shap": mean_shap}).to_csv(imp_path, sep="\t", index=False)

        # Vertical bar plot
        plt.figure(figsize=(5, 4))
        plt.bar(range(len(labels)), mean_shap)
        plt.xticks(range(len(labels)), labels)
        plt.ylabel("Average |SHAP| across runs")
        plt.title("Average SHAP Feature Importance (Simulations)")
        plt.tight_layout()
        out_png = os.path.join(PLOTS_DIR, "avg_shap_importance.png")
        plt.savefig(out_png, dpi=150)
        plt.close()

        print(f"SHAP importance TSV: {imp_path}")
        print(f"SHAP plot saved to: {out_png}")
    except Exception as e:
        print(f"[WARN] Failed to aggregate/plot SHAP: {e}")

    return aucs, best_params_list, last_best_model

# risk model
alpha = 3.33
theta = 1.5
mod = set_model('dab1', alpha, theta)

sim_kwargs = {
    'maf_a': 0.36,
    'maf_b': 0.04,
    'n_cases': 10000,
    'n_controls': 10000,
    'model': mod
}

neural_network(num_reps=30, n_iter=30, **sim_kwargs)

