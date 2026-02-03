===
FAQ
===

Frequently asked questions about SuShiE.

General Questions
=================

What is SuShiE?
---------------
SuShiE (Sum of Shared Single Effect) is a Bayesian fine-mapping method designed for multi-ancestry genetic studies. It identifies causal variants while accounting for effect size correlations across ancestries.

When should I use SuShiE vs single-ancestry SuSiE?
--------------------------------------------------
Use SuShiE when you have genetic data from multiple ancestries and want to:

- Leverage shared genetic architecture across populations
- Improve fine-mapping resolution through diverse LD patterns
- Estimate effect size correlations across ancestries

Use single-ancestry SuSiE (via ``--meta`` or ``--mega`` flags) when:

- You only have data from one ancestry
- You want to compare results with multi-ancestry SuShiE

What data formats does SuShiE support?
--------------------------------------
**Genotype data:**

- PLINK (.bed/.bim/.fam)
- VCF (.vcf, .vcf.gz)
- BGEN (.bgen)

**Phenotype/Covariate data:**

- Tab-separated text files

**Summary statistics:**

- Tab-separated GWAS files with Z-scores or effect sizes

Parameters
==========

How do I choose the number of causal variants (L)?
--------------------------------------------------
The ``-L`` parameter specifies the maximum number of causal variants. Recommendations:

- Start with ``-L 10`` (default) for most fine-mapping analyses
- Increase if you expect more causal variants in the region
- The algorithm will only identify credible sets for true signals

What does the ``--rho`` parameter do?
-------------------------------------
The ``--rho`` parameter sets the prior correlation of effect sizes across ancestries:

- ``--rho 0.0``: Assumes independent effects (equivalent to ``--indep``)
- ``--rho 1.0``: Assumes perfectly correlated effects
- Default: Learns correlation from data (recommended)

When should I use ``--no-update``?
----------------------------------
Use ``--no-update`` to disable prior updates during inference:

- When you have strong prior beliefs about effect distributions
- For faster runtime on large datasets
- When empirical Bayes updates cause convergence issues

Output Files
============

What is a credible set?
-----------------------
A credible set is a set of SNPs that contains the causal variant with high probability (default 95%). SuShiE outputs:

- SNPs in each credible set
- Posterior inclusion probabilities (PIPs)
- Purity scores (minimum LD among SNPs in the set)

How do I interpret the PIP?
---------------------------
The Posterior Inclusion Probability (PIP) represents the probability that a SNP is causal given SNPs in the model:

- PIP > 0.95: Strong evidence for causality
- PIP 0.5-0.95: Moderate evidence
- PIP < 0.5: Weak evidence

Higher PIPs in credible sets indicate better fine-mapping resolution.

What does purity mean?
----------------------
Purity is the minimum absolute correlation (r²) between any pair of SNPs in a credible set:

- High purity (>0.5): SNPs are in high LD, harder to distinguish
- Low purity: SNPs are more independent, but set may contain false positives

The ``--purity`` threshold (default 0.5) filters out low-quality credible sets.

Performance
===========

How can I speed up SuShiE?
--------------------------
Several options to improve performance:

1. **Use GPU acceleration:**

   .. code-block:: bash

      export JAX_PLATFORM_NAME=gpu
      sushie finemap ...

2. **Reduce iterations:**

   .. code-block:: bash

      sushie finemap --max-iter 100 ...

3. **Limit SNPs** for purity calculation:

   .. code-block:: bash

      sushie finemap --max-select 500 ...

How much memory does SuShiE need?
---------------------------------
Memory usage depends on:

- Number of samples (N)
- Number of SNPs (P)
- Number of ancestries (K)

Rough estimate: ``N × P × K × 8 bytes`` for genotype storage.

For large datasets, consider:

- Analyzing smaller genomic regions
- Running on a high-memory compute node

Errors
======

"LD matrix is not positive semi-definite"
-----------------------------------------
This error occurs with summary-level data when the LD matrix has numerical issues:

- Ensure LD was computed from a sufficiently large reference panel
- Check for missing or mismatched SNPs
- Try adding a small ridge regularization term

"No credible sets found"
------------------------
This may indicate:

- No significant signal in the region
- Too stringent purity threshold (try ``--purity 0.1``)
- Convergence issues (check ELBO in output)

"Sample sizes don't match"
--------------------------
For summary-level data, ensure:

- ``--sample-size`` matches the number of ancestries
- Sample sizes correspond to the correct GWAS files (in order)
