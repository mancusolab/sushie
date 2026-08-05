# Output files

The output files consist of eight types:

1.  `*.log`
2.  `*.cs.tsv`
3.  `*.alphas.tsv`
4.  `*.weights.tsv`
5.  `*.corr.tsv`
6.  `*.her.tsv`
7.  `*.cv.tsv`
8.  `*.all.results.npy`

Users can output them as compressed files by specifying `--compress`

## Logger

SuShiE by default has a `*.log` file that keep tracks of inference
process.

## Credible set

SuShiE by default outputs a `*.cs.tsv` file that tracks the SNPs in the
credible sets.

If `--meta` and `--mega` are specified (see
[meta and mega SuShiE](manual.md#meta)), it
will output `*.meta.cs.tsv` and `*.mega.cs.tsv`, respectively, to track
the SNPs in the credible sets inferred by meta SuShiE and mega SuShiE.

For `*.meta.cs.tsv`, it will row-bind the output for single-ancestry
SuShiE differed by column `ancestry`.

| Column | Type | Examples | Notes |
|----|----|----|----|
| SNPIndex | Integer | 1, 33, 77 | The SNP unique index identifier. It matches the order of entries in the original genotype data counting start from 0. |
| chrom | Integer | 1, 23 | The chromosome number. |
| snp | String | rs12345 | The SNP unique identifier (e.g., rs ID). It matches the entries in the original genotype data. |
| pos | Integer | 123456 | The SNP position on the chromosome. It matches the entries in the original genotype data. |
| a0 | String | A | The non-counting allele. It matches the entries in the original genotype data. |
| a1 | String | G | The counting allele. If the counting allele is G, and genotype is GG, then it is coded as 2. It matches the entries in the original genotype data. |
| CSIndex | Integer | 1, 2 | The credible set unique index. |
| alpha | Float | 0.8 | The posterior probability of SNPs to be causal in the corresponding credible set ($\alpha_{l,j}$ in the [model](model.md)). |
| c_alpha | Float | 0.95 | The cumulative posterior probability of SNPs to be causal in the descending order. This decides which SNPs are included in the credible sets. |
| pip_all | Float | 0.95 | The posterior inclusion probability ($\text{PIP}_j$ in the [model](model.md)) calculated across $L$ credible sets. For `*.meta.cs.tsv`, it will have an additional column called `meta_pip_all`. |
| pip_cs | Float | 0.95 | The posterior inclusion probability ($\text{PIP}_j$ in the [model](model.md)) calculated across credible sets that are kept after pruning based on purity. For `*.meta.cs.tsv`, it will have an additional column called `meta_pip_cs`. |
| trait | String | GeneABC | The trait, tissue, or gene name. |
| n_snps | Integer | 500 | The number of total SNPs in the inference. |
| ancestry | String | sushie, mega, ancestry_1 | The inference method for this credible set. |

## Full credible set with alphas

By specifying `--alphas`, SuShiE outputs a `*.alphas.tsv` file that
tracks all the SNPs' PIP, $\alpha$ (see the [model](model.md)), and purity across
all $L$.

If `--meta` and `--mega` are specified (see
[meta and mega SuShiE](manual.md#meta)), it
will output `*.meta.alphas.tsv` and `*.mega.alphas.tsv`, respectively,
to track the information inferred by meta SuShiE and mega SuShiE.

For `*.meta.alphas.tsv`, it will row-bind the output for single-ancestry
SuShiE differed by column `ancestry`.

| Column | Type | Examples | Notes |
|----|----|----|----|
| SNPIndex | Integer | 1, 33, 77 | The zero-based SNP index in the original genotype data. |
| chrom | Integer | 1, 23 | The chromosome number. |
| snp | String | rs12345 | The SNP identifier from the original genotype data. |
| pos | Integer | 123456 | The SNP position on the chromosome. |
| a0 | String | A | The non-counting allele. |
| a1 | String | G | The counted allele; an `a1/a1` genotype is coded as 2. |
| alpha_l1 | Float | 0.8 | The posterior causal probability for the first credible set. Additional effects add corresponding columns. |
| in_cs_l1 | Integer | 0, 1 | Whether the SNP belongs to the first credible set. |
| purity_l1 | Float | 0.634 | The sample-size-weighted mean purity across ancestries. |
| kept_l1 | Integer | 0, 1 | Whether the credible set remains after purity filtering. |
| log_bf_l1 | Float | 100.23 | The log Bayes factor for the first credible set. |
| trait | String | GeneABC | The trait, tissue, or gene name. |
| n_snps | Integer | 500 | The number of SNPs in the inference. |
| purity_threshold | Float | 0.5 | The purity threshold used to prune credible sets. |
| ancestry | String | sushie, mega, ancestry_1 | The inference method for this credible set. |

## Prediction weights

SuShiE by default outputs a `*.weights.tsv` file that contains the
prediction weights, PIPs, and whether in CS, across all the fine-mapped
SNPs.

If `--meta` and `--mega` are specified (see
[meta and mega SuShiE](manual.md#meta)), it
will output `*.meta.weights.tsv` and `*.mega.weights.tsv`, respectively.

| Column | Type | Examples | Notes |
|----|----|----|----|
| SNPIndex | Integer | 1, 33, 77 | The SNP unique index identifier. It matches the order of entries in the original genotype data counting start from 0. |
| chrom | Integer | 1, 23 | The chromosome number. |
| snp | String | rs12345 | The SNP unique identifier (e.g., rs ID). It matches the entries in the original genotype data. |
| pos | Integer | 123456 | The SNP position on the chromosome. It matches the entries in the original genotype data. |
| a0 | String | A | The non-counting allele. It matches the entries in the original genotype data. |
| a1 | String | G | The counting allele. If the counting allele is G, and genotype is GG, then it is coded as 2. It matches the entries in the original genotype data. |
| trait | String | GeneABC | The trait, tissue, or gene name. |
| ancestry1_sushie_weight | Float | 1.3 | The ancestry-specific SNP prediction weights inferred by SuShiE. For `*.meta.weights.tsv`, it will have `ancestry1_single_weight` (It will have extra columns depending on the number of ancestries). If `--mega`, it will have `mega_weight` for all ancestries. |
| sushie_pip_all | Float | 0.95 | The posterior inclusion probability ($\text{PIP}_j$ in [model](model.md)) for all the SNPs calculated across $L$ credible sets. (`*.cs.tsv` only contains the PIPs of SNPs that are only in the credible sets). For `*.meta.weights.tsv`, it will have `ancestry1_single_pip`, `meta_pip_all` (It will have extra columns depending on the number of ancestries). For `*.mega.weights.tsv`, it will have `mega_pip_all`. |
| sushie_pip_cs | Float | 0.95 | The posterior inclusion probability ($\text{PIP}_j$ in [model](model.md)) for all the SNPs calculated across credible sets that are kept after purning based on purity. (`*.cs.tsv` only contains the PIPs of SNPs that are only in the credible sets). For `*.meta.weights.tsv`, it will have `ancestry1_single_pip`, `meta_pip_cs` (It will have extra columns depending on the number of ancestries). For `*.mega.weights.tsv`, it will have `mega_pip_cs`. |
| sushie_cs_index | Integer | 0, 1, ..., $L$ | The credible set index where the SNPs fall into. 0 means no credible sets contain this SNP. For `*.meta.weights.tsv`, it will have `ancestry1_cs_index`(It will have extra columns depending on the number of ancestries). For `*.mega.weights.tsv`, it will have `mega_cs_index`. |
| n_snps | Integer | 500 | The number of total SNPs in the inference. |

## Effect-size correlation

SuShiE by default outputs a `*.corr.tsv` file that contains the
estimated effect size covariance matrix for each output credible set
(after pruning for purity). For results of all $L$ credible sets, see
[NumPy results](#numpy-results).

| Column | Type | Examples | Notes |
|----|----|----|----|
| trait | String | GeneABC | The trait, tissue, or gene name. |
| CSIndex | Integer | 1, 2 | The credible set unique index. It depends on `--L` and puring after purity. |
| ancestry1_est_var | Float | 1.34 | The inferred effect size variance (the posterior estimate for $\sigma^2_{i,b}$ in the [model](model.md)) for ancestry 1. It depends on the number of ancestry. One estimate for each credible set. |
| ancestry1_ancestry2_est_covar | Float | 2.56 | The inferred effect size covariance between ancestry 1 and ancestry 2. It depends on the number of pairs of ancestries. One estimate for each credible set. |
| ancestry1_ancestry2_est_corr | Float | 0.8 | The inferred effect size correlation (the posterior estimate for $\rho$ in the [model](model.md)) between ancestry 1 and ancestry 2. It depends on the number of pairs of ancestries. One estimate for each credible set. |

## Heritability

By specifying `--her`, SuShiE outputs a `*.her.tsv` file that tracks the
heritability analysis results for each ancestry.

It contains two rounds of heritability estimation:

1.  Using all the SNPs.
2.  Using the SNPs in the credible set (only if SuShiE outputs non-empty
    credible sets after pruning for purity).

| Column | Type | Examples | Notes |
|----|----|----|----|
| ancestry | Integer | 1, 2 | The ancestry index. |
| genetic_var | Float | 1.32 | The variance of genetic components contributing to the complex traits. `s_genetic_var`, which is estimated only from the SNPs in the credible sets, will be appended if credible sets are not empty after pruning for purity. |
| h2g | Float | 0.23 | The narrow-sense cis-heritability of the traits based on [limix](https://github.com/limix/limix) definition. This includes the variance of the fixed effects. |
| lrt_stats | Float | -123.23 | The likelihood ratio test statistics compared the linear mixed effects model to the fixed effects model (no genetic variance). `s_lrt_stats`, which is estimated only from the SNPs in the credible sets, will be appended if credible sets are not empty after pruning for purity. |
| p_value | Float | -123.23 | The $p$ value for the likelihood ratio test statistics based on chi-square distribution with 1 dof. `s_p_value`, which is estimated only from the SNPs in the credible sets, will be appended if credible sets are not empty after pruning for purity. |
| trait | String | GeneABC | The trait, tissue, or gene name. |

## Cross-validation

By specifying `--cv`, SuShiE outputs a `*.cv.tsv` file that contains the
results from cross validation (see the [user manual](manual.md#cv) for how we
compute the $r^2$).

| Column | Type | Examples | Notes |
|----|----|----|----|
| ancestry | Integer | 1, 2 | The ancestry index. |
| rsq | Float | 0.9 | $r^2$ between predicted and measured expressions from cross-validations. |
| p_value | Float | 0.23 | The $p$ value for the $r^2$. |
| N | Integer | 200 | The sample size for SuShiE inference. |
| trait | String | GeneABC | The trait, tissue, or gene name. |

## NumPy results

By specifying `--numpy`, SuShiE outputs a `*.all.results.npy` file that
contains all the results from inference and snp information. It can only
be read by python numpy package.
