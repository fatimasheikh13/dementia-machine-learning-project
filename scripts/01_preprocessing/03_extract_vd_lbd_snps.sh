#!/bin/bash
set -euo pipefail

plink=/nfshome/store01/users/c.c2029098/MET583/software/plink
working=/scratch/c.c2029098/dementia_ml_project

# Input - raw BDR dataset
BDR="$working/data/raw/BDR_IMP_ALL"

# Pairs created by your Python step
pairs=(
  "VaD_AD"
  "LBD_AD"
)

# SNP list (selected variants; column 2 contains variant IDs)
snps_list="$working/data/processed/snps_list/77_snps_extended.txt"  # input stays the same

# Output directories
outdir_geno="$working/data/processed/genotypes"
outdir_ml="$working/data/processed/ml_data"
outdir_res="$working/results"
mkdir -p "$outdir_geno" "$outdir_ml" \
         "$outdir_res/PLINK" "$outdir_res/PCA"

# MAF threshold
maf_thr=0.05

# Build subset for a pair using keep file; update .fam with Sex/Phenotype from the pair's phenotype file.
build_subset_and_update_fam () {
  local pair="$1"
  local keep_file="$working/data/processed/sample_info/samples_${pair}.txt"
  local pheno_file="$working/data/processed/sample_info/${pair}_phenotypes.txt"
  local out_prefix="$outdir_geno/BDR_${pair}"

  echo "[$pair] Creating subset with --keep: $keep_file"
  "$plink" --bfile "$BDR" \
           --keep "$keep_file" \
           --make-bed \
           --out "$out_prefix"

  # Sort .fam and phenotype by FID for a clean join
  local fam_sorted="$outdir_geno/${pair}_fam_sorted.txt"
  local pheno_sorted="$outdir_geno/${pair}_pheno_sorted.txt"
  local joined="$outdir_geno/${pair}_joined.txt"

  LC_ALL=C sort -k1,1 "$out_prefix.fam" > "$fam_sorted"
  LC_ALL=C sort -k1,1 "$pheno_file"     > "$pheno_sorted"

  # Join on FID; fam columns + appended Sex + Phenotype
  join -1 1 -2 1 "$fam_sorted" "$pheno_sorted" > "$joined"

  # Overwrite .fam with Sex and Phenotype from joined (columns 5 and 6)
  # fam format: FID IID PID MID Sex Phenotype
  awk 'BEGIN{OFS=" "}{print $1,$2,$3,$4,$(NF-1),$NF}' "$joined" > "$out_prefix.fam"

  # Cleanup intermediates
  rm -f "$fam_sorted" "$pheno_sorted" "$joined"

  echo "[$pair] Subset built: $out_prefix.{bed,bim,fam}"
}

# Core per-pair pipeline: extract selected SNPs, MAF filter, freq, ML recode, genome-wide PCA
run_pair_pipeline () {
  local pair="$1"
  local subset_prefix="$outdir_geno/BDR_${pair}"
  local pair_tag="${pair}"

  # 1) Extract the selected SNPs from the pair subset
  echo "[$pair] Extracting selected SNPs (${pair_tag})"
  "$plink" --bfile "$subset_prefix" \
           --extract "$snps_list" \
           --make-bed \
           --keep-allele-order \
           --out "$outdir_geno/${pair_tag}"

  # 1b) Apply MAF filter and use this for downstream
  echo "[$pair] Apply MAF filter: keep SNPs with MAF >= ${maf_thr}"
  "$plink" --bfile "$outdir_geno/${pair_tag}" \
           --maf $maf_thr \
           --make-bed \
           --keep-allele-order \
           --out "$outdir_geno/${pair_tag}_maf05"

  # 2) Allele frequencies (using MAF-filtered set)
  echo "[$pair] Allele frequencies (MAF-filtered)"
  "$plink" --bfile "$outdir_geno/${pair_tag}_maf05" \
           --freq \
           --out "$outdir_res/PLINK/${pair_tag}_freq"

  # 3) Create additive dosages for ML (A coding) using MAF-filtered set
  echo "[$pair] Recode A for ML (MAF-filtered)"
  "$plink" --bfile "$outdir_geno/${pair_tag}_maf05" \
           --recode A \
           --out "$outdir_ml/${pair_tag}"

  # 4) Genome-wide LD pruning for PCA on the pair subset
  echo "[$pair] LD pruning genome-wide"
  "$plink" --bfile "$subset_prefix" \
           --indep-pairwise 200 50 0.1 \
           --out "$outdir_res/PCA/${pair_tag}_prune"

  "$plink" --bfile "$subset_prefix" \
           --extract "$outdir_res/PCA/${pair_tag}_prune.prune.in" \
           --make-bed \
           --out "$outdir_geno/${pair_tag}_pruned"

  # 5) PCA (10 PCs)
  echo "[$pair] PCA (10 PCs)"
  "$plink" --bfile "$outdir_geno/${pair_tag}_pruned" \
           --pca 10 header tabs \
           --out "$outdir_res/PCA/${pair_tag}"

  echo "[$pair] Done."
  echo "  Freq: $outdir_res/PLINK/${pair_tag}_freq.frq"
  echo "  PCs:  $outdir_res/PCA/${pair_tag}.eigenvec"
  echo "  ML A: $outdir_ml/${pair_tag}.raw"
}

# Main loop 
for pair in "${pairs[@]}"; do
  echo "Processing pair: $pair"

  build_subset_and_update_fam "$pair"
  run_pair_pipeline "$pair"
done

echo "All pairs complete."
