import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from scipy.stats import uniform, randint

# File paths 
BASE_DIR = "/scratch/c.c2029098/dementia_ml_project/results/machine_learning/AD_DAB1_APOE/xgboost_new"
output_auc_file   = os.path.join(BASE_DIR, "AUC", "xgboost_auc_nestedcv.txt")
output_params_file= os.path.join(BASE_DIR, "best_params_nestedcv.txt")
plots_dir         = os.path.join(BASE_DIR, "plots")
inter_dir         = os.path.join(BASE_DIR, "interactions")
os.makedirs(os.path.dirname(output_auc_file), exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(inter_dir, exist_ok=True)

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
MAX_INTERACTION_SAMPLES = 1500  # cap per repetition for interaction calc

# Load data 
file_path = "/scratch/c.c2029098/dementia_ml_project/data/processed/ml_data/ml_AD_APOE_DAB1.csv"
df = pd.read_csv(file_path)

# Require these columns to exist
required_cols = {"APOE", "DAB1", "PHENOTYPE"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

X = df[["DAB1", "APOE"]]  # keep order stable
y = df["PHENOTYPE"]

# Train and test 
def run_xgboost_nested_cv(X, y, num_reps=50, n_iter=50, test_size=0.2):
    test_aucs = []
    train_aucs = []
    all_best_params = []
    shap_values_list = []
    feature_names = X.columns
    feature_importances = []

    # Columns are present and set directly
    apoe_name_global = "APOE"
    dab1_name_global = "DAB1"
    have_pair = True
    # Random search parameters
    param_dist = {
        'max_depth': randint(3, 10),
        'learning_rate': uniform(0.01, 0.3),
        'subsample': uniform(0.5, 0.5),
        'colsample_bytree': uniform(0.5, 0.5),
        'n_estimators': randint(50, 200),
        'gamma': uniform(0, 5),
        'min_child_weight': randint(1, 10)
    }
    # Loop for each repetition 
    for i in range(num_reps):
        print(f"\n Repetition {i+1}/{num_reps}")
        np.random.seed(i)

        # Outer split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=i, stratify=y
        )

        # Inner CV
        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=i)
        # XGBoost model 
        base_model = XGBClassifier(
            objective='binary:logistic',
            use_label_encoder=False,
            eval_metric='auc',
            verbosity=0,
            n_jobs=-1,
            random_state=i
        )

        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring='roc_auc',
            cv=inner_cv,
            verbose=0,
            n_jobs=-1,
            random_state=i
        )

        random_search.fit(X_train, y_train)
        best_model = random_search.best_estimator_
        all_best_params.append(random_search.best_params_)

        # Evaluate
        y_train_pred = best_model.predict_proba(X_train)[:, 1]
        y_test_pred  = best_model.predict_proba(X_test)[:, 1]
        auc_train = roc_auc_score(y_train, y_train_pred)
        auc_test  = roc_auc_score(y_test, y_test_pred)
        train_aucs.append(auc_train)
        test_aucs.append(auc_test)
        print(f"Train AUC = {auc_train:.4f} | Test AUC = {auc_test:.4f}")

        # SHAP (main effects)
        explainer = shap.Explainer(best_model)
        shap_vals = explainer(X_test)
        shap_values_list.append(np.abs(shap_vals.values).mean(axis=0))

        # Feature importance 
        importance = best_model.get_booster().get_score(importance_type='gain')
        full_importance = {feat: importance.get(feat, 0) for feat in feature_names}
        feature_importances.append(full_importance)

        # APOE×DAB1 SHAP interaction + dependence plot 
        if have_pair:
            apoe_col = apoe_name_global
            dab1_col = dab1_name_global
            apoe_idx = list(X_test.columns).index(apoe_col)
            dab1_idx = list(X_test.columns).index(dab1_col)

            # Dependence plot 
            if i == 0:
                X_dep = X_test.copy()
                # tiny jitter on discrete dosages to reduce overplotting
                X_dep[dab1_col] = X_dep[dab1_col].astype(float) + np.random.uniform(-0.03, 0.03, size=len(X_dep))
                shap.dependence_plot(
                    dab1_col, shap_vals.values, X_dep,
                    interaction_index=apoe_col, show=False
                )
                plt.title("SHAP dependence — DAB1 (colored by APOE)")
                plt.xlabel("DAB1 dosage")
                plt.ylabel("SHAP value for DAB1")
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f"shap_dependence_DAB1_by_APOE_{stamp}.png"), dpi=300)
                plt.close()

            # SHAP interaction values 
            tree_expl = shap.TreeExplainer(best_model, feature_perturbation="interventional")
            n_te = len(X_test)
            if n_te > MAX_INTERACTION_SAMPLES:
                rng = np.random.default_rng(i)
                sel = np.sort(rng.choice(n_te, size=MAX_INTERACTION_SAMPLES, replace=False))
            else:
                sel = np.arange(n_te)
            X_int = X_test.iloc[sel]

            S_int = tree_expl.shap_interaction_values(X_int)   
            if isinstance(S_int, list): 
                S_int = S_int[0]

            pair = S_int[:, apoe_idx, dab1_idx]                # per-sample APOE×DAB1 interaction

            # Save per-sample interaction with raw features 
            out_df = pd.DataFrame({
                "idx": sel,
                "APOE": X_int[apoe_col].values,
                "DAB1": X_int[dab1_col].values,
                "SHAP_interaction_APOExDAB1": pair.astype(float)
            })
            out_df.to_csv(
                os.path.join(inter_dir, f"apoe_dab1_interactions_rep{i+1}_{stamp}.tsv"),
                sep="\t", index=False
            )

    # Save summaries 
    avg_auc_train = float(np.mean(train_aucs))
    avg_auc_test  = float(np.mean(test_aucs))
    with open(output_auc_file, 'w') as f:
        for i in range(num_reps):
            f.write(f"Seed {i} - Train AUC: {train_aucs[i]:.4f} - Test AUC: {test_aucs[i]:.4f}\n")
        f.write(f"\nAverage Train AUC: {avg_auc_train:.4f}\n")
        f.write(f"Average Test AUC: {avg_auc_test:.4f}\n")
    print(f"\n Final Average Train AUC: {avg_auc_train:.4f} | Final Average Test AUC: {avg_auc_test:.4f}")

    with open(output_params_file, 'w') as f:
        for i, params in enumerate(all_best_params):
            f.write(f"Seed {i}: {params}\n")

    # Average SHAP bar
    shap_df = pd.DataFrame(shap_values_list, columns=feature_names)
    avg_shap = shap_df.mean().sort_values(ascending=False)
    plt.figure(figsize=(6, 4))
    avg_shap.plot(kind="bar")
    plt.title("Average SHAP Importance (Nested CV)")
    plt.ylabel("Mean(|SHAP value|)")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "avg_shap_importance_nestedcv.png"))
    plt.close()

    # Average XGBoost gain bar
    imp_df = pd.DataFrame(feature_importances)
    avg_imp = imp_df.mean().sort_values(ascending=False)
    plt.figure(figsize=(6, 4))
    avg_imp.plot(kind="bar")
    plt.title("Average XGBoost Gain Importance (Nested CV)")
    plt.ylabel("Average Gain")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "avg_gain_importance_nestedcv.png"))
    plt.close()

    return test_aucs, avg_auc_test, all_best_params

# Run 
aucs, avg_auc, all_best_params = run_xgboost_nested_cv(X, y, num_reps=50, n_iter=50)
