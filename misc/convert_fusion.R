library(tidyverse)

args <- commandArgs(TRUE)
pref <- args[1]

pref <- "/Users/zeyunlu/Documents/github/test_sushie/test_res/onepop"
nam_weight <- paste0(pref, ".weights.tsv")
nam_heri <- paste0(pref, ".h2g.tsv")
nam_cvr2 <- paste0(pref, ".cv.r2.tsv")


weight <- read_tsv(nam_weight)
heri <- read_tsv(nam_heri)
cvr2 <- read_tsv(nam_cvr2)

n_feature <- sum(grepl(colnames(weight), pattern = "sushie"))

for (idx in 1:n_feature){
  # cv.performance
  tmp_cvr2 <- as.numeric(cvr2[idx,2:3])
  cv.performance <- matrix(tmp_cvr2, 2,1)
  rownames(cv.performance) <- c("rsq", "pval")
  colnames(cv.performance) <- "sushie"
  
  # hsq
  hsq <- c(as.numeric(heri[idx, 2]), NA)
  
  # hsq.pv
  hsq.pv <- NA
  
  # N.tot
  N.tot <- as.numeric(cvr2[idx, 4])
  
  # snps
  snps <- weight[c("chrom", "snp", paste0("cm_", idx), paste0("pos_", idx), "a0", "a1")]
  colnames(snps) <- paste0("V", 1:6)
  snps <- data.frame(snps)
  
  # wgt.matrix
  wgt.matrix <- matrix(weight[[paste0("feature", idx, "_sushie")]])
  rownames(wgt.matrix) <- weight$snp
  colnames(wgt.matrix) <- "sushie"
  
  save(cv.performance, hsq, hsq.pv, N.tot, snps, wgt.matrix,
    file = paste0(pref, ".feature", idx, ".fusion.RData"))
}
