.. _Files:

============
Output Files
============

The output files consist of eight types:

#. ``*.log``
#. ``*.cs.tsv``
#. ``*.alphas.tsv``
#. ``*.weights.tsv``
#. ``*.corr.tsv``
#. ``*.her.tsv``
#. ``*.cv.tsv``
#. ``*.all.results.npy``

Users can output them as compressed files by specifying ``--compress``

Logger
-------------

SuShiE by default has a ``*.log`` file that keep tracks of inference process.

.. _csfile:

Credible Set
------------

SuShiE by default outputs a ``*.cs.tsv`` file that tracks the SNPs in the credible sets.

If ``--meta`` and ``--mega`` are specified (see definitions in :ref:`meta`), it will output ``*.meta.cs.tsv`` and ``*.mega.cs.tsv``, respectively, to track the SNPs in the credible sets inferred by meta SuShiE and mega SuShiE.

For ``*.meta.cs.tsv``, it will row-bind the output for single-ancestry SuShiE differed by column ``ancestry``.

.. list-table::
   :header-rows: 1

   * - Column
     - Type
     - Examples
     - Notes
   * - SNPIndex
     - Integer
     - 1, 33, 77
     - The SNP unique index identifier. It matches the order of entries in the original genotype data counting start from 0.
   * - chrom
     - Integer
     - 1, 23
     - The chromosome number.
   * - snp
     - String
     - rs12345
     - The SNP unique identifier (e.g., rs ID). It matches the entries in the original genotype data.
   * - pos
     - Integer
     - 123456
     - The SNP position on the chromosome. It matches the entries in the original genotype data.
   * - a0
     - String
     - A
     - The alternative allele. It matches the entries in the original genotype data.
   * - a1
     - String
     - G
     - The reference allele. It matches the entries in the original genotype data.
   * - CSIndex
     - Integer
     - 1, 2
     - The credible set unique index.
   * - alpha
     - Float
     - 0.8
     - The posterior probability of SNPs to be causal in the corresponding credible set (:math:`\alpha_{l,j}` in :ref:`Model`).
   * - c_alpha
     - Float
     - 0.95
     - The cumulative posterior probability of SNPs to be causal in the descending order. This decides which SNPs are included in the credible sets.
   * - pip_all
     - Float
     - 0.95
     - The posterior inclusion probability (:math:`\text{PIP}_j` in :ref:`Model`) calculated across :math:`L` credible sets. For ``*.meta.cs.tsv``, it will have additional column called ``meta_pip_all`` .
   * - pip_cs
     - Float
     - 0.95
     - The posterior inclusion probability (:math:`\text{PIP}_j` in :ref:`Model`) calculated across credible sets that are kept after pruning based on purity. For ``*.meta.cs.tsv``, it will have additional column called ``meta_pip_cs``.
   * - trait
     - String
     - GeneABC
     - The trait, tissue, or gene name.
   * - n_snps
     - Integer
     - 500
     - The number of total SNPs in the inference.
   * - ancestry
     - String
     - sushie, mega, ancestry_1
     - The inference method for this credible set.

.. _alphasfile:

Full Credible Set with Alphas
-----------------------------

By specifying ``--alphas``, SuShiE outputs a ``*.alphas.tsv`` file that tracks all the SNPs' PIP, :math:`\alpha` (see :ref:`Model`), and purity across all :math:`L`.

If ``--meta`` and ``--mega`` are specified (see definitions in :ref:`meta`), it will output ``*.meta.alphas.tsv`` and ``*.mega.alphas.tsv``, respectively, to track the information inferred by meta SuShiE and mega SuShiE.

For ``*.meta.alphas.tsv``, it will row-bind the output for single-ancestry SuShiE differed by column ``ancestry``.


.. list-table::
   :header-rows: 1

   * - Column
     - Type
     - Examples
     - Notes
   * - SNPIndex
     - Integer
     - 1, 33, 77
     - The SNP unique index identifier. It matches the order of entries in the original genotype data counting start from 0.
   * - chrom
     - Integer
     - 1, 23
     - The chromosome number.
   * - snp
     - String
     - rs12345
     - The SNP unique identifier (e.g., rs ID). It matches the entries in the original genotype data.
   * - pos
     - Integer
     - 123456
     - The SNP position on the chromosome. It matches the entries in the original genotype data.
   * - a0
     - String
     - A
     - The alternative allele. It matches the entries in the original genotype data.
   * - a1
     - String
     - G
     - The reference allele. It matches the entries in the original genotype data.
   * - alpha_l1
     - Float
     - 0.8
     - The posterior probability of SNPs to be causal in the first credible set (:math:`\alpha_{l,j}` in :ref:`Model`). Depending on ``--L``, it can have extra columns.
   * - in_cs_l1
     - Integer
     - 0, 1
     - The indicator whether the SNP is in the first credible set. Depending on ``--L``, it can have extra columns.
   * - purity_l1
     - float
     - 0.634
     - The sample-size-weighted average purity across ancestries. To compare with the ``--purity``, it will decide the value in ``in_cs_l1``. Depending on ``--L``, it can have extra columns.
   * - kept_l1
     - Integer
     - 0, 1
     - The indicator whether the credible set is kept after pruning based on purity threshold. Depending on ``--L``, it can have extra columns.
   * - trait
     - String
     - GeneABC
     - The trait, tissue, or gene name.
   * - n_snps
     - Integer
     - 500
     - The number of total SNPs in the inference.
   * - purity_threshold
     - float
     - 0.5
     - The purity threshold to prune the credible sets.
   * - ancestry
     - String
     - sushie, mega, ancestry_1
     - The inference method for this credible set.


.. _weightsfile:
Prediction Weights
------------------

SuShiE by default outputs a ``*.weights.tsv`` file that contains the prediction weights, PIPs, and whether in CS, across all the fine-mapped SNPs.

If ``--meta`` and ``--mega`` are specified (see definitions in :ref:`meta`), it will output ``*.meta.weights.tsv`` and ``*.mega.weights.tsv``, respectively.

.. list-table::
   :header-rows: 1

   * - Column
     - Type
     - Examples
     - Notes
   * - SNPIndex
     - Integer
     - 1, 33, 77
     - The SNP unique index identifier. It matches the order of entries in the original genotype data counting start from 0.
   * - chrom
     - Integer
     - 1, 23
     - The chromosome number.
   * - snp
     - String
     - rs12345
     - The SNP unique identifier (e.g., rs ID). It matches the entries in the original genotype data.
   * - pos
     - Integer
     - 123456
     - The SNP position on the chromosome. It matches the entries in the original genotype data.
   * - a0
     - String
     - A
     - The alternative allele. It matches the entries in the original genotype data.
   * - a1
     - String
     - G
     - The reference allele. It matches the entries in the original genotype data.
   * - trait
     - String
     - GeneABC
     - The trait, tissue, or gene name.
   * - ancestry1_sushie_weight
     - Float
     - 1.3
     - The ancestry-specific SNP prediction weights inferred by SuShiE. For ``*.meta.weights.tsv``, it will have ``ancestry1_single_weight`` (It will have extra columns depending on the number of ancestries). If ``--mega``, it will have ``mega_weight`` for all ancestries.
   * - sushie_pip_all
     - Float
     - 0.95
     - The posterior inclusion probability (:math:`\text{PIP}_j` in :ref:`Model`) for all the SNPs calculated across :math:`L` credible sets. (``*.cs.tsv`` only contains the PIPs of SNPs that are only in the credible sets). For ``*.meta.weights.tsv``, it will have ``ancestry1_single_pip``, ``meta_pip_all`` (It will have extra columns depending on the number of ancestries). For ``*.mega.weights.tsv``, it will have ``mega_pip_all``.
   * - sushie_pip_cs
     - Float
     - 0.95
     - The posterior inclusion probability (:math:`\text{PIP}_j` in :ref:`Model`) for all the SNPs calculated across credible sets that are kept after purning based on purity. (``*.cs.tsv`` only contains the PIPs of SNPs that are only in the credible sets). For ``*.meta.weights.tsv``, it will have ``ancestry1_single_pip``, ``meta_pip_cs`` (It will have extra columns depending on the number of ancestries). For ``*.mega.weights.tsv``, it will have ``mega_pip_cs``.
   * - sushie_cs_index
     - Integer
     - 0, 1, ..., :math:`L`
     - The credible set index where the SNPs fall into. 0 means no credible sets contain this SNP. For ``*.meta.weights.tsv``, it will have ``ancestry1_cs_index``(It will have extra columns depending on the number of ancestries). For ``*.mega.weights.tsv``, it will have ``mega_cs_index``.
   * - n_snps
     - Integer
     - 500
     - The number of total SNPs in the inference.

.. _corrfile:
Effect Size Correlation
-----------------------

SuShiE by default outputs a ``*.corr.tsv`` file that contains the estimated effect size covariance matrix for each output credible set (after pruning for purity). For results of all :math:`L` credible sets, see :ref:`npyfile` file.

.. list-table::
   :header-rows: 1

   * - Column
     - Type
     - Examples
     - Notes
   * - trait
     - String
     - GeneABC
     - The trait, tissue, or gene name.
   * - CSIndex
     - Integer
     - 1, 2
     - The credible set unique index. It depends on ``--L`` and puring after purity.
   * - ancestry1_est_var
     - Float
     - 1.34
     - The inferred effect size variance (the posterior estimate for :math:`\sigma^2_{i,b}` in :ref:`Model`) for ancestry 1. It depends on the number of ancestry. One estimate for each credible set.
   * - ancestry1_ancestry2_est_covar
     - Float
     - 2.56
     - The inferred effect size covariance between ancestry 1 and ancestry 2. It depends on the number of pairs of ancestries. One estimate for each credible set.
   * - ancestry1_ancestry2_est_corr
     - Float
     - 0.8
     - The inferred effect size correlation (the posterior estimate for :math:`\rho` in :ref:`Model`) between ancestry 1 and ancestry 2. It depends on the number of pairs of ancestries. One estimate for each credible set.

.. _herfile:
Heritability Estimation
-----------------------

By specifying ``--her``, SuShiE outputs a ``*.her.tsv`` file that tracks the heritability analysis results for each ancestry.

It contains two rounds of heritability estimation:

#. Using all the SNPs.
#. Using the SNPs in the credible set (only if SuShiE outputs non-empty credible sets after pruning for purity).

.. list-table::
   :header-rows: 1

   * - Column
     - Type
     - Examples
     - Notes
   * - ancestry
     - Integer
     - 1, 2
     - The ancestry index.
   * - genetic_var
     - Flat
     - 1.32
     - The variance of genetic components contributing to the complex traits. ``s_genetic_var``, which is estimated only from the SNPs in the credible sets, will be appended if credible sets are not empty after pruning for purity.
   * - h2g
     - Flat
     - 0.23
     - The narrow-sense cis-heritability of the traits based on `limix <https://github.com/limix/limix>`_ definition. This includes the variance of the fixed effects.
   * - lrt_stats
     - Flat
     - -123.23
     - The likelihood ratio test statistics compared the linear mixed effects model to the fixed effects model (no genetic variance). ``s_lrt_stats``, which is estimated only from the SNPs in the credible sets, will be appended if credible sets are not empty after pruning for purity.
   * - p_value
     - Flat
     - -123.23
     - The :math:`p` value for the likelihood ratio test statistics based on chi-square distribution with 1 dof. ``s_p_value``, which is estimated only from the SNPs in the credible sets, will be appended if credible sets are not empty after pruning for purity.
   * - trait
     - String
     - GeneABC
     - The trait, tissue, or gene name.


.. _cvfile:
Cross Validation
----------------

By specifying ``--cv``, SuShiE outputs a ``*.cv.tsv`` file that contains the results from cross validation (see :ref:`cv` for how we compute the :math:`r^2`).

.. list-table::
   :header-rows: 1

   * - Column
     - Type
     - Examples
     - Notes
   * - ancestry
     - Integer
     - 1, 2
     - The ancestry index.
   * - rsq
     - Flat
     - 0.9
     - :math:`r^2` between predicted and measured expressions from cross-validations.
   * - p_value
     - Flat
     - 0.23
     - The :math:`p` value for the :math:`r^2`.
   * - N
     - Integer
     - 200
     - The sample size for SuShiE inference.
   * - trait
     - String
     - GeneABC
     - The trait, tissue, or gene name.


.. _npyfile:
Everything
----------

By specifying ``--numpy``, SuShiE outputs a ``*.all.results.npy`` file that contains all the results from inference and snp information. It can only be read by python numpy package.
