 
 
 #!/bin/bash

# Software
plink=/nfshome/store01/users/c.c2029098/MET583/software/plink 

# Working directory base
working=/scratch/c.c2029098/dementia_ml_project

# Paths based on your structure
snps=$working/data/processed/snps_list/apoe_dab1_snps.txt
BDR=$working/data/raw/BDR_IMP_ALL
keep_file=$working/data/processed/sample_info/samples_AD_controls.txt
phenotypes=$working/data/processed/sample_info/AD_phenotypes.txt

# Keep only relevant samples first
$plink --bfile $BDR \
    --keep $keep_file \
    --make-bed \
    --out $working/data/processed/genotypes/BDR_AD

# Sort .fam and phenotype files in same directory as .fam file
sort -k1,1 $working/data/processed/genotypes/BDR_AD.fam > $working/data/processed/genotypes/fam_sorted.fam
sort -k1,1 $phenotypes > $working/data/processed/genotypes/AD_phenotypes_sorted.txt

# Join on FID
join -1 1 -2 1 $working/data/processed/genotypes/fam_sorted.fam $working/data/processed/genotypes/AD_phenotypes_sorted.txt > $working/data/processed/genotypes/joined.txt

# Update .fam file with sex and phenotype info
awk '{print $1, $2, $3, $4, $(NF-1), $NF}' OFS="\t" $working/data/processed/genotypes/joined.txt > $working/data/processed/genotypes/BDR_AD.fam

# Cleanup temp files
rm $working/data/processed/genotypes/fam_sorted.fam
rm $working/data/processed/genotypes/AD_phenotypes_sorted.txt
rm $working/data/processed/genotypes/joined.txt

# Extract APOE and DAB1 SNPs
$plink --bfile $working/data/processed/genotypes/BDR_AD \
    --extract $snps \
    --make-bed \
    --keep-allele-order \
    --out $working/data/processed/genotypes/BDR_AD_APOE_DAB1

# Check allele frequencies
$plink --bfile $working/data/processed/genotypes/BDR_AD_APOE_DAB1 \
    --freq \
    --out $working/results/PLINK/BDR_AD_APOE_DAB1_freq

head $working/results/PLINK/BDR_AD_APOE_DAB1_freq.frq

# Create allele dosage dataset for ML
$plink --bfile $working/data/processed/genotypes/BDR_AD_APOE_DAB1 \
    --recode A \
    --out $working/data/processed/ml_data/BDR_AD_APOE_DAB1


