#!/bin/bash
set -euo pipefail

plink=/nfshome/store01/users/c.c2029098/MET583/software/plink
working=/scratch/c.c2029098/dementia_ml_project
BDR_AD=$working/data/processed/genotypes/BDR_AD
snps="$working/data/processed/snps_list/77_snps_extended.txt"   
weights="$working/data/processed/snps_list/prs_77.txt"        

outdir_geno="$working/data/processed/genotypes"
outdir_ml="$working/data/processed/ml_data"
outdir_res="$working/results"
mkdir -p "$outdir_geno" "$outdir_ml" \
         "$outdir_res/PLINK" "$outdir_res/PRS" "$outdir_res/PCA" "$outdir_res/GWAS"

maf_thr=0.05
base_label="BDR_AD_control"  
maf_tag=$(echo "${maf_thr}" | sed 's/^0\.//; s/\.//g') 
maf_suffix="maf${maf_tag}"

echo "Extract the selected SNPs"
"$plink" --bfile "$BDR_AD" \
  --extract "$snps" \
  --make-bed \
  --keep-allele-order \
  --out "$outdir_geno/${base_label}"

echo "Apply MAF filter: keep SNPs with MAF >= ${maf_thr}"
"$plink" --bfile "$outdir_geno/${base_label}" \
  --maf "$maf_thr" \
  --make-bed \
  --keep-allele-order \
  --out "$outdir_geno/${base_label}_${maf_suffix}"

echo "Allele frequencies (MAF-filtered set)"
"$plink" --bfile "$outdir_geno/${base_label}_${maf_suffix}" \
  --freq \
  --out "$outdir_res/PLINK/${base_label}_freq"

echo "Create additive dosages (A format) for ML from MAF-filtered set"
"$plink" --bfile "$outdir_geno/${base_label}_${maf_suffix}" \
  --recode A \
  --out "$outdir_ml/${base_label}"

echo "Compute PRS using provided weights on the MAF-filtered set"
"$plink" --bfile "$outdir_geno/${base_label}_${maf_suffix}" \
  --score "$weights" 2 3 4 header sum \
  --out "$outdir_res/PRS/BDR_AD_PRS"

echo "LD pruning for PCA on BDR_AD (genome-wide)"
"$plink" --bfile "$BDR_AD" \
  --indep-pairwise 200 50 0.1 \
  --out "$outdir_res/PCA/BDR_AD_control_prune"

"$plink" --bfile "$BDR_AD" \
  --extract "$outdir_res/PCA/BDR_AD_control_prune.prune.in" \
  --make-bed \
  --out "$outdir_geno/BDR_AD_control_pruned"

echo "PCA (10 PCs)"
"$plink" --bfile "$outdir_geno/BDR_AD_control_pruned" \
  --pca 10 header tabs \
  --out "$outdir_res/PCA/BDR_AD_control"

echo "Done.
PRS:  $outdir_res/PRS/BDR_AD_PRS.profile
Freq: $outdir_res/PLINK/${base_label}_freq.frq
ML:   $outdir_ml/${base_label}.raw
PCs:  $outdir_res/PCA/BDR_AD_control.eigenvec
