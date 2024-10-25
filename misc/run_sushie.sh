#!/bin/bash

# specify conda environment
conda activate sushie

# suppose you want to perform sushie on 100 molecular phenotypes
# you may want to have a metadata file that specify the gene name, window start, windown end, chromsome number etc.
start=1
stop=100

for IDX in `seq $start $stop`
do
  # Extract parameters for sushie run
  params=`sed "${IDX}q;d" metadata.tsv`
  set -- junk $params
  shift
  NAME=$1 # gene name
  CHR=$2 # chromsome number
  P0=$3 # genomic window start
  P1=$4 # genomic window end

  echo SUMMARY ${IDX}, ${NAME}, ${CHR}, ${P0}, ${P1}

  # get phenotype data
  pheno_EUR=eur_pheno_file
  pheno_AFR=afr_pheno_file

  # get genotype data
  geno_EUR=eur_geno_file
  geno_AFR=afr_geno_file

  # get covariate data
  covar_EUR=eur_covar_file
  covar_AFR=afr_covar_file

  # for fine-mapping using individual-level data
  sushie finemap \
    --pheno $pheno_EUR $pheno_AFR \
    --covar $covar_EUR $covar_AFR \
    --plink  $geno_EUR $geno_AFR \
    --trait ${NAME} \
    --output ${NAME}.sushie

  gwas_EUR=eur_gwas_file
  gwas_AFR=afr_gwas_file

  ld_EUR=eur_ld_file
  ld_AFR=afr_ld_file

  # for fine-mapping using summary-level data (User specify LD matrix)
  sushie finemap \
    --summary \
    --gwas $gwas_EUR $gwas_AFR \
    --ld $ld_EUR $ld_AFR \
    --sample-size 100 100 \
    --trait ${NAME} \
    --output ${NAME}.sushie

  # for fine-mapping using summary-level data (User specify individual-level reference data)
  sushie finemap \
    --summary \
    --gwas $gwas_EUR $gwas_AFR \
    --plink $geno_EUR $geno_AFR \
    --trait ${NAME} \
    --output ${NAME}.sushie
done
