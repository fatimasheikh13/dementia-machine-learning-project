import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping, EpochScoring
from torch.nn.utils import weight_norm
import shap
import contextlib
import sys
import warnings
import scipy.stats as sp

warnings.filterwarnings("ignore")

@contextlib.contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, 'w') as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

class SNPNet(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        self.fc1 = weight_norm(nn.Linear(2, 4))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = weight_norm(nn.Linear(4, 1))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x.squeeze(1)

# Directories and filepaths
def run_main_pipeline():
    base_dir = "/scratch/c.c2029098/dementia_ml_project/results/machine_learning/AD_DAB1_APOE/neural_network"
    loss_graphs_dir = os.path.join(base_dir, "loss_graphs")
    plots_dir = os.path.join(base_dir, "plots")
    auc_dir = os.path.join(base_dir, "AUC")
    os.makedirs(loss_graphs_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(auc_dir, exist_ok=True)

    data_path = "/scratch/c.c2029098/dementia_ml_project/data/processed/ml_data/ml_AD_APOE_DAB1.csv"
    df = pd.read_csv(data_path)
    X_df = df[["DAB1", "APOE"]]
    y_series = df["PHENOTYPE"]
    # Storing values
    seeds = list(range(30))
    aucs = []
    train_aucs = []
    shap_values_list = []
    perm_values_list = []

    for seed in seeds:
        print(f"\n Running seed {seed}")

        np.random.seed(seed)
        torch.manual_seed(seed)

        X_train_df, X_test_df, y_train, y_test = train_test_split(
            X_df, y_series, test_size=0.3, random_state=seed, stratify=y_series)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_df).astype(np.float32)
        X_test = scaler.transform(X_test_df).astype(np.float32)
        y_train_f = y_train.values.astype(np.float32)
        y_test_f = y_test.values.astype(np.float32)

        auc_val_cb = EpochScoring('roc_auc', name='valid_auc', lower_is_better=False, on_train=False)
        auc_train_cb = EpochScoring('roc_auc', name='train_auc', lower_is_better=False, on_train=True)
        # Random search parameters
        param_dist = {
            'module__dropout_rate': sp.uniform(0.1, 0.3),
            'lr': sp.reciprocal(1e-5, 1e-2),
            'batch_size': sp.randint(48, 128),
            'max_epochs': sp.randint(60, 200)
        }
        # Neural network model 
        net = NeuralNetClassifier(
            SNPNet,
            optimizer=optim.Adam,
            optimizer__weight_decay=1e-4,
            criterion=nn.BCELoss,
            iterator_train__shuffle=True,
            callbacks=[EarlyStopping(patience=10), auc_val_cb, auc_train_cb],
            verbose=0,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        # Cross fold validation and random search 
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        random_search = RandomizedSearchCV(
            estimator=net,
            param_distributions=param_dist,
            n_iter=30,
            scoring='roc_auc',
            cv=cv,
            verbose=0,
            n_jobs=-1,
            random_state=seed,
            error_score='raise'
        )
        #Training and testing 
        with suppress_stdout_stderr():
            random_search.fit(X_train, y_train_f)

        best_model = random_search.best_estimator_
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test_f, y_pred_proba)
        aucs.append(auc)
        #Trainging and loss curves
        hist = best_model.history
        loss_train = hist[:, 'train_loss']
        loss_val = hist[:, 'valid_loss'] if 'valid_loss' in hist[0] else None
        auc_train = hist[:, 'train_auc'] if 'train_auc' in hist[0] else None
        auc_val = hist[:, 'valid_auc'] if 'valid_auc' in hist[0] else None

        final_train_auc = auc_train[-1] if auc_train is not None else np.nan
        train_aucs.append(final_train_auc)

        epochs = range(1, len(loss_train) + 1)

        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        axs[0].plot(epochs, loss_train, label="Train Loss")
        if loss_val is not None:
            axs[0].plot(epochs, loss_val, label="Val Loss")
        axs[0].legend(); axs[0].set_title("Loss Curve")

        if auc_train is not None and auc_val is not None:
            min_len = min(len(auc_train), len(auc_val), len(epochs))
            axs[1].plot(epochs[:min_len], auc_train[:min_len], label="Train AUC")
            axs[1].plot(epochs[:min_len], auc_val[:min_len], label="Val AUC")
            axs[1].legend(); axs[1].set_title("AUC Curve")
        else:
            axs[1].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(loss_graphs_dir, f"seed{seed}_curves.png"))
        plt.close()

        # SHAP explainer
        def model_predict(X_input):
            X_tensor = torch.from_numpy(X_input.astype(np.float32)).to(best_model.device)
            with torch.no_grad():
                preds = best_model.module_(X_tensor)
            return preds.cpu().numpy()

        explainer = shap.Explainer(model_predict, X_train)
        shap_values = explainer(X_test)

        # Record SHAP and permutation importance scores
        shap_values_list.append(np.abs(shap_values.values).mean(axis=0))

        perm_result = permutation_importance(
            best_model, X_test, y_test_f,
            scoring='roc_auc', n_repeats=10, random_state=seed)
        perm_values_list.append(perm_result.importances_mean)

    # Save AUCs
    auc_file = os.path.join(auc_dir, "all_auc_seeds.txt")
    with open(auc_file, "w") as f:
        for i in range(len(seeds)):
            f.write(f"Seed {seeds[i]} - Train ROC AUC: {train_aucs[i]:.4f} - Test ROC AUC: {aucs[i]:.4f}\n")
        f.write(f"\nAverage Train ROC AUC: {np.nanmean(train_aucs):.4f}\n")
        f.write(f"Average Test ROC AUC: {np.mean(aucs):.4f}\n")

    # SHAP bargraph
    mean_shap = np.mean(np.stack(shap_values_list), axis=0)
    plt.figure()
    plt.bar(range(len(mean_shap)), mean_shap)
    plt.xticks(range(len(mean_shap)), ['DAB1', 'APOE'])
    plt.title("Average SHAP Feature Importance")
    plt.savefig(os.path.join(plots_dir, "avg_shap_importance.png"))
    plt.close()

    # Permutation bargraph
    mean_perm = np.mean(np.stack(perm_values_list), axis=0)
    plt.figure()
    plt.bar(range(len(mean_perm)), mean_perm)
    plt.xticks(range(len(mean_perm)), ['DAB1', 'APOE'])
    plt.title("Average Permutation Importance")
    plt.savefig(os.path.join(plots_dir, "avg_permutation_importance.png"))
    plt.close()

run_main_pipeline()
