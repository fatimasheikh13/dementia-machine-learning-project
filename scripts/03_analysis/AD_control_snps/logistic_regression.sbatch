#!/bin/bash

#SBATCH --job-name=ll_ad
#SBATCH --output=/scratch/c.c2029098/dementia_ml_project/results/machine_learning/AD_control/logistic_regression/OUT/lr_%A_%a.out
#SBATCH --error=/scratch/c.c2029098/dementia_ml_project/results/machine_learning/AD_control/logistic_regression/ERR/lr_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --account=scw1557

echo "Job started at $(date)"

# Initialize conda for this shell
eval "$(/apps/languages/anaconda/2023.03/bin/conda shell.bash hook)"

# Activate your conda environment
conda activate /scratch/c.c2029098/ml_envs/ml_env
echo "Conda environment activated"

# Optional: export TMPDIR if needed
export TMPDIR=/scratch/c.c2029098/dementia_ml_project/results/machine_learning/AD_control/logistic_regression/temp

# Run your Python script with SLURM array task ID as argument

python logistic_regression.py