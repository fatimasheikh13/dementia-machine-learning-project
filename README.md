# Dementia ML Epistasis

Machine learning pipeline investigating whether supervised ML models (logistic regression, XGBoost, neural networks) can detect epistatic interactions between Alzheimer's-associated genetic variants (notably APOE and DAB1) and improve discrimination between dementia subtypes (Alzheimer's, vascular dementia, Lewy body dementia) using genotype data from the Brains for Dementia Research (BDR) cohort and simulated data based on UK Biobank (UKBB) effect sizes.

## Project overview

This repository accompanies thesis work on epistasis detection in Alzheimer's disease (AD) using machine learning. Two complementary data sources are used:

1. **Simulated genotype data** (`scripts/02_imulations/`) — reproducing a previously reported APOE–DAB1 interaction under controlled conditions, based on effect sizes from Bracher-Smith et al. (2022).
2. **Real genotype data from the BDR cohort** (`scripts/03_BDR_analysis/`) — a panel of 81 genome-wide significant AD-associated SNPs, used to classify AD vs. controls, AD vs. vascular dementia (VD), and AD vs. Lewy body dementia (LBD).

Three modelling approaches are compared across both data sources:
- **Logistic regression** (unpenalised, `sklearn`)
- **XGBoost** (`xgboost`, tuned via `RandomizedSearchCV`)
- **Feedforward neural networks** (`PyTorch` + `skorch`)

Model interpretability is assessed using SHAP values (main effects and pairwise interactions), and population stratification is accounted for post hoc using principal components.

## Repository structure and pipeline order

Scripts are intended to be run in the following order: **preprocessing → simulations / analysis → results visualisation.**

```
scripts/
├── 01_preprocessing/
├── 02_simulations/
└── 03_BDR_analysis/
    ├── AD_APOE_DAB1/
    ├── AD_control_snps/
    ├── AD_LBD_snps/
    ├── AD_VAD_snps/
└── 04_visualisations\
    ├── PC_vis_AD_VD_LBD.ipynb
    ├── results_vis.ipynb
```

### 1. `01_preprocessing/`

Prepares genotype and phenotype data from the BDR cohort for downstream modelling. Run before any analysis or simulation scripts.

| Script | Purpose |
|---|---|
| `01_extract_apoe_dab1.sh` | Filters the raw BDR genotype file down to AD/control samples, updates the .fam file with sex and diagnosis, then extracts the APOE and DAB1 SNPs via PLINK. Outputs allele frequencies and an additive-dosage (.raw) file for ML |
| `02_extract_AD_snps.sh` | Extracts the 81-SNP AD-associated panel from the BDR dataset built in step 1, applies a MAF ≥ 0.05 filter, computes allele frequencies, generates an additive-dosage `(.raw)` file for ML, computes a PRS from provided SNP weights, and runs genome-wide LD pruning + PCA (10 PCs) for later population-stratification adjustment. |
| `03_extract_vd_lbd_snps.sh` | Independently builds AD-vs-VD and AD-vs-LBD sample subsets directly from the raw BDR dataset, then repeats the same extraction pipeline as step 2 (SNP extraction, MAF filtering, allele frequencies, ML dosage recoding, LD pruning, PCA) for each pair. |
| `04_encode_apoe_genotypes.ipynb` | Derives APOE genotypes (ε2/ε3/ε4 combinations) from rs429358/rs7412 dosages and computes the ε4 allele-count feature used in modelling. |
| `05_compute_prs_betas.ipynb` | Calculates polygenic risk score (PRS) beta coefficients from the training data, used to construct the PRS dataset alongside the Kunkle et al.-derived version generated in step 2. |

### 2. `02_Simulations/`

Generates and models the simulated APOE–DAB1 dataset, used to test whether ML models can recover a known, previously reported interaction under controlled conditions.

| Script | Purpose |
|---|---|
| `simulation_UKBB.ipynb` | Generates the simulated dataset using established APOE ε4 and DAB1 effect sizes from the UKBB dataset and runs logistic regression model on the simulated dataset. |
| `neural_networks_simulations.py` | Trains and evaluates a feedforward neural network on the simulated APOE–DAB1 dataset. |


### 3. `03_BDR_analysis/AD_APOE_DAB1/`

APOE–DAB1 interaction modelling in the real BDR cohort (AD vs. controls), evaluating four specifications: APOE alone, DAB1 alone, APOE + DAB1 additive, and APOE + DAB1 + interaction term.

| Script | Purpose |
|---|---|
| `logistic_regression_APOE_DAB1.ipynb` | Logistic regression models testing APOE, DAB1, and their interaction. |
| `nn_AD_APOE_DAB1.py` / `.sbatch` | Neural network model and SLURM submission script. |
| `xgboost_AD_APOE_DAB1.py` / `.sbatch` | XGBoost model and SLURM submission script. |

### 4. `03_BDR_analysis/AD_control_snps/`

AD-associated SNP panel modelling: AD vs. controls. The most extensively developed comparison, including PRS-based classification and pairwise interaction analysis.

| Script | Purpose |
|---|---|
| `logistic_regression.py` / `.sbatch` | Logistic regression using the AD-associated SNP panel. |
| `logistic_regression_PRS.ipynb` | Logistic regression using PRS as the predictor (both Kunkle et al.-derived and training-derived beta versions). |
| `nn_snps.py` / `.sbatch` | Neural network model on the SNP panel. |
| `xgboost_snps.py` / `.sbatch` | XGBoost model on the SNP panel. |
| `pairwise_interaction.ipynb` | SHAP interaction value analysis — ranks pairwise SNP interactions across repetitions. |

### 5. `03_BDR_analysis/AD_LBD_snps/`

AD-associated SNP panel modelling: AD vs. Lewy body dementia (LBD).

| Script | Purpose |
|---|---|
| `logistic_regression.py` / `.sbatch` | Logistic regression classifier. |
| `nn_snps.py` / `.sbatch` | Neural network classifier. |
| `xgboost_snps.py` / `.sbatch` | XGBoost classifier. |

### 6. `03_BDR_analysis/AD_VAD_snps/`

AD-associated SNP panel modelling: AD vs. vascular dementia (VD).

| Script | Purpose |
|---|---|
| `logistic_regression.py` / `.sbatch` | Logistic regression classifier. |
| `nn_snps.py` / `.sbatch` | Neural network classifier. |
| `xgboost_snps.py` / `.sbatch` | XGBoost classifier. |

### 7. Results visualisation (`04_visualisation/`)

Run after all modelling scripts to summarise and visualise results.

| Script | Purpose |
|---|---|
| `PC_vis_AD_VD_LBD.ipynb` | Visualises principal components used for post-hoc population stratification adjustment across the AD vs. VD/LBD datasets. |
| `results_vis.ipynb` | Compiles and visualises final performance metrics (AUC) results across all models and datasets. |

## File-type key

- `.py` — standalone Python script, typically submitted to the HPC cluster via a matching `.sbatch` file
- `.sbatch` — SLURM job submission script (Cardiff ARCCA HPC), paired with a `.py` script of the same name
- `.ipynb` — Jupyter notebook, run interactively.

## Environment

Scripts were run on Cardiff University's Advanced Research Computing (ARCCA) HPC cluster using SLURM for job submission.

Core dependencies:
- Python ≥ 3.9
- `pandas`, `numpy`, `scikit-learn`
- `xgboost`
- `torch`, `skorch`
- `shap`
- `statsmodels`
- `matplotlib`

An `environment.yml` listing exact pinned versions should be added for full reproducibility.

## Notes on data

Raw genotype data (BDR, UK Biobank) are **not included** in this repository due to data access agreements.

