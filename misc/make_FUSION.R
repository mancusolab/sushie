library(tidyverse)

args <- commandArgs(TRUE)
weight_path <- args[1]
heri_path <- args[2]
cvr2_path <- args[3]
out_path <- args[4]

weight <- read_tsv(weight_path)
heri <- read_tsv(heri_path)
cvr2 <- read_tsv(cvr2_path)

n_pop <- nrow(heri)

for (idx in 1:n_pop){

  # cv.performance
  tmp_cvr2 <- as.numeric(cvr2[idx,2:3])
  cv.performance <- matrix(tmp_cvr2, 2,1)
  rownames(cv.performance) <- c("rsq", "pval")
  colnames(cv.performance) <- "sushie"

  # hsq
  hsq <- c(as.numeric(heri[idx, 3]), NA)

  # hsq.pv
  hsq.pv <- heri[idx,5]

  # N.tot
  N.tot <- as.numeric(cvr2[idx, 4])

  # snps
  weight$cm <- 0
  snps <- weight[c("chrom", "snp", "cm", "pos", "a0", "a1")]
  colnames(snps) <- paste0("V", 1:6)
  snps <- data.frame(snps)

  # wgt.matrix
  wgt.matrix <- matrix(weight[[paste0("ancestry", idx, "_sushie_weight")]])
  rownames(wgt.matrix) <- weight$snp
  colnames(wgt.matrix) <- "sushie"

  save(cv.performance, hsq, hsq.pv, N.tot, snps, wgt.matrix,
    file = paste0(out_path, ".ancestry", idx, ".sushie.fusion.RData"))
}
