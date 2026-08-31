#!/bin/bash

#SBATCH --job-name=xgb_array
#SBATCH --output=/scratch/c.c2029098/dementia_ml_project/results/machine_learning/AD_LBD_snps/xgboost/OUT/xgb_%A.out
#SBATCH --error=/scratch/c.c2029098/dementia_ml_project/results/machine_learning/AD_LBD_snps/xgboost/ERR/xgb_%A.err
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --account=scw1557

echo "Job $SLURM_ARRAY_TASK_ID started at $(date)"

# Initialize conda for this shell
eval "$(/apps/languages/anaconda/2023.03/bin/conda shell.bash hook)"

# Activate your conda environment
conda activate /scratch/c.c2029098/ml_envs/ml_env
echo "Conda environment activated"

# Optional: export TMPDIR if needed
export TMPDIR=/scratch/c.c2029098/dementia_ml_project/results/machine_learning/AD_LBD_snps/xgboost/temp

# Run your Python script with SLURM array task ID as argument
python xgboost_snps.py

echo "Job $SLURM_ARRAY_TASK_ID finished at $(date)"

