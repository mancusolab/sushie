.. _manual:

=================
Users Manual
=================

Initialize Environment
======================

SuShiE is a command-line software written in Python. Before installation, we recommend to create a new environment using `conda <https://docs.conda.io/en/latest/>`_ so that it will not affect the software versions of users' other projects.

SuShiE uses `JAX <https://github.com/google/jax>`_ with `Just In Time  <https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html>`_ compilation to achieve high-speed computation. However, there are some `issues <https://github.com/google/jax/issues/5501>`_ for JAX with Mac M1 chip. To solve this, users need to initiate conda using `miniforge <https://github.com/conda-forge/miniforge>`_, and then install SuShiE using ``pip`` in the desired environment.

Installation
============

..
    The easiest way to install is with ``pip``:

    .. code:: bash

    pip install sushie

    Alternatively

Users can download the latest repository and then use ``pip``:

.. code:: bash

    git clone https://github.com/mancusolab/sushie.git
    cd sushie
    pip install .

*We currently only support Python3.8+.*

Data Preparation
================

SuShiE performs SNP fine-mapping on molecular data (e.g., gene expressions). Therefore, it requires at least phenotype and genotype data specified with the option to specify covariates.

Although we highly recommend users to perform high-quality QC on their own genotype, phenotype, and covariate data, we implement following basic QCs in the software:

#. Remove subjects with N/A values from either phenotype or covariates data.
#. Remove SNPs that all subjects have N/A value.
#. Impute SNPs that partial subjects have N/A value based on two times allele frequencies.
#. Only keep subjects who have data in all the genotype, phenotype, and covariate data.
#. Only keep SNPs that are available in all the ancestries.
#. Adjust genotype data across ancestries based on the same reference alleles. Drop non-biallelic SNPs.
#. Remove SNPs that have minor allele frequency (MAF) less than 1% within each ancestry (users can change 1% with ``--maf`).
#. For single ancestry SuSiE, users have the option to perform rank inverse normalization transformation on the phenotype data.

See :func:`sushie.cli.process_raw` for these QCs' source codes.

Testing Data
------------

We provide example data in ``./data/`` folder to test out SuShiE. All the data are in three ancestries: 489 European individuals (EUR), 639 African individuals (AFR), and 481 East Asian individuals (EAS).

The genotype is the high-quality `HapMap <https://www.genome.gov/10001688/international-hapmap-project>`_ SNPs in some random gene 1M base-pair window, which contains 123, 129, and 113 SNPs for EUR, AFR, and EAS respectively in `1000G <https://www.internationalgenome.org/>`_ project. We provide genotype data in `plink 1 <https://www.cog-genomics.org/plink/1.9/input#bed>`_, `vcf <https://en.wikipedia.org/wiki/Variant_Call_Format>`_, and `bgen <https://www.well.ox.ac.uk/~gav/bgen_format/>`_ 1.3 format.

We simulate phenotypes using the approach described in our manuscript.

We randomly simulate null phenotype and covariate data only for testing purpose.

As for the format requirement, see :ref:`Param` for detailed explanations.

Examples
========

SuShiE software is very easy to use, for it only has one command ``finemap``. In this section, we walk through several examples of using SuShiE.

See :ref:`Files` for the detailed explanation of output files.

See :ref:`Param` for the detailed explanation of parameters.

We make a bash script ``./misc/run_sushie.sh`` to show a more general working flow of running SuShiE.

If users still have questions, feel free to contact the developers.

1. Two-Ancestry SuShiE
----------------------

In this example, we perform two-ancestry SuShiE with covariates regressed out from both phenotypes and genotypes while updating the prior effect size covariance matrix during the optimizations.

.. code:: bash

    cd ./data/
    sushie finemap --pheno EUR.pheno.tsv AFR.pheno.tsv --vcf vcf/EUR.vcf vcf/AFR.vcf --covar EUR.covar.tsv AFR.covar.tsv --output ~/test_result


2. :math:`N`-Ancestry SuShiE
----------------------------

In the example below, we perform single-ancestry SuShiE, which is equivalently to the SuSiE model (see :ref:`Reference`).

.. code:: bash

    cd ./data/
    sushie finemap --pheno EUR.pheno.tsv --vcf vcf/EUR.vcf --covar EUR.covar.tsv --output ~/test_result

Or three-ancestry setting:

.. code:: bash

    cd ./data/
    sushie finemap --pheno EUR.pheno.tsv AFR.pheno.tsv EAS.pheno.tsv --vcf vcf/EUR.vcf vcf/AFR.vcf vcf/EAS.vcf --covar EUR.covar.tsv AFR.covar.tsv EAS.covar.tsv --output ~/test_result

3. Can I use other formats of genotypes?
----------------------------------------

Yes! SuShiE can take either `plink 1 <https://www.cog-genomics.org/plink/1.9/input#bed>`_, `vcf <https://en.wikipedia.org/wiki/Variant_Call_Format>`_, or `bgen <https://www.well.ox.ac.uk/~gav/bgen_format/>`_, but not `plink 2 <https://www.cog-genomics.org/plink/2.0/input#pgen>`_.

For plink 1, SuShiE read in the triplet (bed, bim, and fam) prefix.

.. code:: bash

    cd ./data/
    sushie finemap --pheno EUR.pheno.tsv AFR.pheno.tsv --plink plink/EUR plink/AFR --output ~/test_result

For bgen data, users need to make sure that the latter allele shown up in the ``allele ids`` is the reference allele.

.. code:: bash

    cd ./data/
    sushie finemap --pheno EUR.pheno.tsv AFR.pheno.tsv --bgen bgen/EUR.bgen bgen/AFR.bgen --output ~/test_result

.. _index:
4. My data contains all the participants and I do not want to separate them
------------------------------------------------------------------------------

No problem! If all the subjects are in single phenotype, genotype, and covariate files. Users just need to use ``--ancestry-index`` command to specify a file that subject ID on the first column, and the ancestry index on the second column. The ancestry index has to start from 1 continuously to the total number of ancestry.

.. code:: bash

    cd ./data/
    sushie finemap --pheno all.pheno.tsv --plink plink/all --ancestry-index all.ancestry.index.tsv --output ~/test_result

.. _meta:
5. How about mega or meta SuShiE?
---------------------------------

The software employs the function to run meta SuShiE and mega SuShiE by adding the parameter ``--meta`` or ``--mega``.

We define the meta SuShiE as running single-ancestry SuShiE followed by meta analysis of the PIPs:

.. math::
   \text{PIP}_{\text{meta}} = 1 - \prod_{i=1}^k(1 - \text{PIP}_{\text{ancestry i}})

We define the mega SuShiE as running single-ancestry SuShiE on genotype and phenotype data that is row-wise stacked across ancestries.

.. code:: bash

    cd ./data/
    sushie finemap --pheno EUR.pheno.tsv AFR.pheno.tsv --vcf vcf/EUR.vcf vcf/AFR.vcf --meta --output ~/test_result

.. code:: bash

    cd ./data/
    sushie finemap --pheno EUR.pheno.tsv AFR.pheno.tsv --vcf vcf/EUR.vcf vcf/AFR.vcf --mega --output ~/test_result

.. _cv:
6. Let's estimate heritability, run CV, and make FUSION files!
--------------------------------------------------------------

SuShiE incorporates codes in `limix <https://github.com/limix/limix>`_ to estimate the narrow-sense cis-heritability (:math:`h_g^2`) by specifying ``--her``.

SuShiE also has a function (``--cv``) to perform :math:`X`-fold cross-validation (CV; ``--cv-num X``) on the ancestry-specific prediction weights to compute the out-of-sample :math:`r^2` between predicted and measured expressions with its corresponding :math:`p`-value.

Specifically, we randomly (``--seed [YOUR SEED]``) and equally divide the dataset into ``X`` portions. We regard each portion as validation dataset and the rest four portions as training dataset. Then, we perform SuShiE on the training datasets for ``X`` times, and predict the expressions on the corresponding validation dataset. Last, we row-wise stack all ``X`` predicted expressions and compute the :math:`r^2` with row-wise stacked and matched validation dataset.

With these two information (:math:`h_g^2` and CV), we prepare R codes ``./misc/make_fusion.R`` to generate `FUSION <http://gusevlab.org/projects/fusion/>`_-format prediction weights, thus can be used in `TWAS <https://www.nature.com/articles/ng.3506>`_.

.. code:: bash

    cd ./data/
    sushie finemap --pheno EUR.pheno.tsv AFR.pheno.tsv --vcf vcf/EUR.vcf vcf/AFR.vcf --cv --her --output ~/test_result
    Rscript ./misc/make_FUSION.R ~/test_result ~


7. I don't want to scale my phenotype by its standard deviation
---------------------------------------------------------------

Fine-mapping inference sometimes can be sensitive to whether scaling the phenotypes and genotypes. SuShiE by default scales the phenotypes and genotypes by their respective standard deviations. However, if users want to disable it, simply add ``--no-scale`` to the command.


.. code:: bash

    cd ./data/
    sushie finemap --pheno EUR.pheno.tsv AFR.pheno.tsv --vcf vcf/EUR.vcf vcf/AFR.vcf --no-scale --output ~/test_result

8. I have my own initial values for the hyperparameters
-------------------------------------------------------

SuShiE has three hyperparameters (:ref:`Model`): the residual variance (:math:`\sigma^2_e`) prior, the QTL effect size variance (:math:`\sigma^2_{i,b}`) prior, and the ancestral effect size correlation (:math:`\rho`) prior. SuShiE by default initializes them as ``0.001``, ``0.001``, and ``0.8``. If users have their own initial values, simply specify them with ``--resid_var``, ``--effect_var``, and ``--rho``. Make sure the ancestry order has to match the phenotype file order.

.. code:: bash

    cd ./data/
    sushie finemap --pheno EUR.pheno.tsv AFR.pheno.tsv --vcf vcf/EUR.vcf vcf/AFR.vcf --resid-var 2.2 2.2 --effect-var 1.2 3.4 --rho 0.2 --output ~/test_result

By default, SuShiE will update :math:`\sigma^2_{i,b}` and :math:`\rho` during the optimization. If users want to disable it, add ``--no-update`` to the command line.

.. code:: bash

    cd ./data/
    sushie finemap --pheno EUR.pheno.tsv AFR.pheno.tsv --vcf vcf/EUR.vcf vcf/AFR.vcf --resid_var 2.2 2.2 --effect_var 1.2 3.4 --rho 0.2 --no-update --output ~/test_result

In addition, with ``--no-update``, if users only specify ``--effect-var`` but not for ``--rho``, ``--effect-var`` will be fixed during the optimizations while ``--rho`` will get updated, vice versa. In other words, if users want to fix both priors, they have to specify both at the same time or specify neither of them (in the latter case, fixing the default values 0.001 and 0.2 as the priors).

9. What if I assume no correlation across ancestries?
-----------------------------------------------------

SuShiE features that it accounts for ancestral quantitative trait loci (QTL) effect size correlation (:math:`\rho` in :ref:`Model`) in the inference, which is different from other SuSiE-extended multi-ancestry fine-mapping frameworks assuming no ancestral correlation (Joint SuShiE). However, it has the functions to make inference assuming no correlation across ancestries by simply specifying ``--no-update`` on the effect size covariance matrix and fixing the rho equal to zero ``--rho 0``. With this, the effect size variance (:math:`\sigma^2_{i,b}` in :ref:`Model`) will get updated while rho will not.

.. code:: bash

    cd ./data/
    sushie finemap --pheno EUR.pheno.tsv AFR.pheno.tsv --vcf vcf/EUR.vcf vcf/AFR.vcf --no-update --rho 0 --output ~/test_result

10. I want to improvise in post-hoc analysis
-------------------------------------------

We understand :ref:`Files` output by SuShiE may not serve all users' post-hoc analysis. Therefore, we add the option to save all the inference results in ``*.npy`` file by specifying ``--numpy``.

The ``*.npy`` files include SNP information, prior estimators, posterior estimators, credible set, PIPs, and sample size.

.. code:: bash

    cd ./data/
    sushie finemap --pheno EUR.pheno.tsv AFR.pheno.tsv --vcf vcf/EUR.vcf vcf/AFR.vcf --numpy --output ~/test_result


11. I seek to use GPU or TPU to make inference faster
-----------------------------------------------------

SuShiE software uses `JAX <https://github.com/google/jax>`_ with `Just In Time  <https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html>`_ compilation to achieve high-speed computation. Jax can be run on GPU or TPU.

.. code:: bash

    cd ./data/
    sushie finemap --pheno EUR.pheno.tsv AFR.pheno.tsv --vcf vcf/EUR.vcf vcf/AFR.vcf --platform gpu --output ~/test_result

12. I want to use 32-bit precision
----------------------------------

SuShiE uses 64-bit precision to assure an accurate inference. However, if users want to use 32-bit precision, they can specify it by having ``--precision 32``.

Unless necessarily needed, we do not recommend to use 32-bit precision as it may cause non-positive semi-definite effect size covariance prior or decreasing `ELBO <https://en.wikipedia.org/wiki/Evidence_lower_bound>`_, thus concluding the inference earlier.

.. code:: bash

    cd ./data/
    sushie finemap --pheno EUR.pheno.tsv AFR.pheno.tsv --vcf vcf/EUR.vcf vcf/AFR.vcf --precision 32 --output ~/test_result

13. I want to run fine-mapping on certain subjects
--------------------------------------------------

Users can use ``--keep`` command to specify a file that contains the subject IDs. As a result, the following fine-mapping inference only performs on the subjects listed in the file.

.. code:: bash

    cd ./data/
    sushie finemap --pheno EUR.pheno.tsv AFR.pheno.tsv --vcf vcf/EUR.vcf vcf/AFR.vcf --keep keep.subject.tsv --output ~/test_result

.. _Param:

Parameters
==========

.. list-table::
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Example
     - Notes
   * - ``--pheno``
     - String
     - Required, no default
     - ``--pheno EUR.pheno.tsv AFR.pheno.tsv``
     - Phenotype data. It has to be a tsv file that contains at least two columns where the first column is subject ID and the second column is the continuous phenotypic value. It can be a compressed file (e.g., tsv.gz). It is okay to have additional columns, but only the first two columns will be used. **No headers**. Use ``space`` to separate ancestries if more than two. SuShiE currently only fine-maps on **continuous** data.
   * - ``--plink``
     - String
     - None
     - ``--plink plink/EUR plink/AFR``
     - Genotype data in `plink 1 <https://www.cog-genomics.org/plink/1.9/input#bed>`_ format. The plink triplet (bed, bim, and fam) should be in the same folder with the same prefix. Use ``space`` to separate ancestries if more than two. Keep the same ancestry order as phenotype's. SuShiE currently does not take `plink 2 <https://www.cog-genomics.org/plink/2.0/input#pgen>`_ format.
   * - ``--vcf``
     - String
     - None
     - ``--vcf vcf/EUR.vcf vcf/AFR.vcf``
     - Genotype data in `vcf <https://en.wikipedia.org/wiki/Variant_Call_Format>`_ format. Use ``space`` to separate ancestries if more than two. Keep the same ancestry order as phenotype's.
   * - ``--bgen``
     - String
     - None
     - ``--bgen bgen/EUR.bgen bgen/AFR.bgen``
     - Genotype data in `bgen <https://www.well.ox.ac.uk/~gav/bgen_format/>`_ 1.3 format. Use ``space`` to separate ancestries if more than two. Keep the same ancestry order as phenotype's.
   * - ``--ancestry-index``
     - String
     - None
     - ``--ancestry-index all.ancestry.index.tsv``
     - Single file that contains subject ID and their ancestry index. Default is None. It has to be a tsv file that contains at least two columns where the first column is the subject ID and the second column is the ancestry index starting from 1 (e.g., 1, 2, 3 etc.). It can be a compressed file (e.g., tsv.gz). Only the first two columns will be used. No headers. If this file is specified, it assumes that all the phenotypes across ancestries are in one single file, and same thing for genotypes and covariates data. It will produce errors if multiple phenotype, genotype, and covariates are specified.
   * - ``--keep``
     - String
     - None
     - ``--keep keep.subject.tsv``
     - Single file that contains subject ID across all ancestries that are used for fine-mapping. It has to be a tsv file that contains at least one columns where the first column is the subject ID. It can be a compressed file (e.g., tsv.gz). No headers. If this file is specified, all phenotype, genotype, and covariates data will be filtered down to the subjects listed in it.
   * - ``--covar``
     - String
     - None
     - ``--covar EUR.covar.tsv AFR.covar.tsv``
     - Covariates that will be accounted in the fine-mapping. It has to be a tsv file that contains at least two columns where the first column is the subject ID. It can be a compressed file (e.g., tsv.gz). **No headers**. All the columns will be counted. Use ``space`` to separate ancestries if more than two. Keep the same ancestry order as phenotype's. Pre-converting the categorical covariates into dummy variables is required. If the categorical covariate has ``n`` levels, make sure the dummy variables have ``n-1`` columns.
   * - ``--L``
     - Integer
     - 10
     - ``--L 5``
     - Integer number of shared effects pre-specified. Larger number may cause slow inference.
   * - ``--pi``
     - Float
     - 1/p
     - ``--pi 0.1``
     - Prior probability for each SNP to be causal (:math:`\pi` in :ref:`Model`). Default is ``1/p`` where ``p`` is the number of SNPs in the region. It is the fixed across all ancestries.
   * - ``--resid-var``
     - Float
     - 1e-3
     - ``--resid-var 5.18 0.2``
     - Specify the prior for the residual variance (:math:`\sigma^2_e` in :ref:`Model`) for ancestries. Values have to be positive. Use ``space`` to separate ancestries if more than two.
   * - ``--effect-var``
     - Float
     - 1e-3
     - ``--effect-var 5.21 0.99 ``
     - Specify the prior for the causal effect size variance (:math:`\sigma^2_{i,b}` in :ref:`Model`) for ancestries. Values have to be positive. Use ``space`` to separate ancestries if more than two. If ``--no-update`` is specified and ``--rho`` is not, specifying this parameter will only fix ``effect-var`` as prior through optimizations and update ``rho``. If ``--effect-covar``, ``--rho``, and ``--no-update`` all three are specified, both ``--effect-covar`` and ``--rho`` will be fixed as prior through optimizations. If ``--no-update`` is specified, but neither ``--effect-covar`` nor ``--rho``, both ``--effect-covar`` and ``--rho`` will be fixed as default prior value through optimizations.
   * - ``--rho``
     - Float
     - 0.1
     - ``--rho 0.05``
     - Specify the prior for the effect correlation (:math:`\rho` in :ref:`Model`) for ancestries. Default is 0.1 for each pair of ancestries. Use space to separate ancestries if more than two. Each rho has to be a float number between -1 and 1. If there are ``N > 2`` ancestries, ``X = choose(N, 2)`` is required. The rho order has to be ``rho(1,2)``, ..., ``rho(1, N)``, ``rho(2,3)``, ..., ``rho(N-1. N)``. If ``--no-update`` is specified and ``--effect-covar`` is not, specifying this parameter will only fix ``rho`` as prior through optimizations and update ``effect-covar``. If ``--effect-covar``, ``--rho``, and ``--no-update`` all three are specified, both ``--effect-covar`` and ``--rho`` will be fixed as prior through optimizations. If ``--no-update`` is specified, but neither ``--effect-covar`` nor ``--rho``, both ``--effect-covar`` and ``--rho`` will be fixed as default prior value through optimizations.
   * - ``--no-scale``
     - Boolean
     - False
     - ``--no-scale # will store as True``
     - Indicator to scale the genotype and phenotype data by standard deviation. Default is to scale. Specify ``--no-scale`` will store ``True`` value, and may cause different inference.
   * - ``--no-regress``
     - Boolean
     - False
     - ``--no-regress # will store as True``
     - Indicator to regress the covariates on each SNP. Default is to regress. Specify ``--no-regress`` will store ``True`` value. It may slightly slow the inference, but can be more accurate.
   * - ``--no-update``
     - Boolean
     - False
     - ``--no-update # will store as True``
     - Indicator to update effect covariance prior before running single effect regression. Default is to update. Specify ``--no-update`` will store ``True`` value. The updating algorithm is similar to `EM algorithm <https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm>`_ or `Empirical Bayes method <https://en.wikipedia.org/wiki/Empirical_Bayes_method>`_ that computes the prior covariance conditioned on other parameters. See the manuscript for more information."
   * - ``--max-iter``
     - Integer
     - 500
     - ``--max-iter 300``
     - Maximum iterations for the optimization. Larger number may slow the inference while smaller may cause different inference.
   * - ``--min-tol``
     - Float
     - 1e-3
     - ``--min_tol 1e-4``
     - Minimum tolerance for the convergence. Smaller number may slow the inference while larger may cause different inference.
   * - ``--threshold``
     - Float
     - 0.9
     - ``--threshold 0.8``
     - Specify the PIP threshold for SNPs to be included in the credible sets. It has to be a float number between 0 and 1.
   * - ``--purity``
     - Float
     - 0.5
     - ``--purity 0.5``
     - Specify the purity threshold for credible sets to be output. It has to be a float number between 0 and 1.
   * - ``--max-select``
     - Integer
     - 250
     - ``--max-select 100``
     - The maximum selected number of SNPs to calculate the purity. Default is 250. It has to be positive integer number. A larger number can unnecessarily spend much memory.
   * - ``--min-snps``
     - Integer
     - 100
     - ``--min-snps 50``
     - The minimum number of SNPs to fine-map. Default is 100. It has to be positive integer number. A smaller number may produce weird results.
   * - ``--maf``
     - float
     - 0.01
     - ``--maf 0.05``
     - Threshold for minor allele frequency (MAF) to filter out SNPs for each ancestry. It has to be a float between 0 (exclusive) and 0.5 (inclusive).
   * - ``--rint``
     - Boolean
     - False
     - ``--rint``
     - Indicator to perform rank inverse normalization transformation (rint) for each phenotype data. Default is False (do not transform). Specify --rint will store 'True' value. We suggest users to do this QC during data preparation.
   * - ``--no-reorder``
     - Boolean
     - False
     - ``--no-reorder``
     - Indicator to re-order single effects based on Frobenius norm of alpha-weighted posterior mean square. Default is False (to re-order). Specify --no-reorder will store 'True' value.
   * - ``--meta``
     - Boolean
     - False
     - ``--meta # will store as True``
     - Indicator to perform single-ancestry SuShiE followed by meta analysis of the results. Specify ``--meta`` will store ``True`` value and increase running time. Specifying one ancestry in phenotype and genotype parameter will ignore ``--meta``.
   * - ``--mega``
     - Boolean
     - False
     - ``--mega # will store as True``
     - Indicator to perform mega SuShiE that run single-ancestry SuShiE on genotype and phenotype data that is row-wise stacked across ancestries. Specify ``--mega`` will store ``True`` value and increase running time. Specifying one ancestry in phenotype and genotype parameter will ignore ``--mega``.
   * - ``--her``
     - Boolean
     - False
     - ``--her # will store as True``
     - Indicator to perform heritability (:math:`h_g^2`) analysis using limix. Specify ``--her`` will store ``True`` value and increase running time. It estimates :math:`h_g^2` using codes in `limix <https://github.com/limix/limix>`_.
   * - ``--cv``
     - Boolean
     - False
     - ``--cv 0.5 # will store as True``
     - Indicator to perform cross validation (CV) and output CV results (adjusted r-squared and its p-value) for future `FUSION <http://gusevlab.org/projects/fusion/>`_ pipline. Specify ``--cv`` will store ``True`` value and increase running time.
   * - ``--cv-num``
     - Integer
     - 5
     - ``--cv_num 6``
     - The number of fold cross validation. It has to be a positive integer number. Larger number may cause longer running time.
   * - ``--seed``
     - Integer
     - 12345
     - ``--seed 4321``
     - The seed for randomization. It can be used to cut data sets in cross validation. It can also be used to randomly select SNPs in the credible sets to calculate the purity. Default is 12345. It has to be positive integer number.
   * - ``--alphas``
     - Boolean
     - False
     - ``--alphas # will store as True``
     - Indicator to output all the credible set results before pruning for purity including PIPs, :math:`\alpha` (in :ref:`Model`), whether in cs, across all :math:`L`. Default is False. Specify --alphas will store 'True' value and increase running time.
   * - ``--numpy``
     - Boolean
     - False
     - ``--numpy # will store as True``
     - Indicator to output all the results in \*.npy file. Specify ``--numpy`` will store ``True`` and increase running time. \*.npy file contains all the inference results including SNP information, credible sets, pips, priors, posteriors, and sample size for users' own post-hoc analysis.
   * - ``--trait``
     - String
     - "Trait"
     - ``--trait GENE_ABC``
     - Trait, tissue, gene name of the phenotype for better indexing in post-hoc analysis.
   * - ``--quiet``
     - Boolean
     - False
     - ``--quiet # will store as True``
     - Indicator to not print message to console. Specify ``--quiet`` will store ``True`` value.
   * - ``--verbose``
     - Boolean
     - False
     - ``--verbose # will store as True``
     - Indicator to include debug information in the log. Specify ``--verbose`` will store ``True`` value.
   * - ``--compress``
     - Boolean
     - False
     - ``--compress # will store as True``
     - Indicator to compress all output tsv files in 'tsv.gz'. Specify ``--compress`` will store ``True`` value to save disk space. This command will not compress npy files.
   * - ``--platform``
     - String choices in ``["cpu", "gpu", "tpu"]``
     - cpu
     - ``--platform gpu``
     - Indicator for the JAX platform.
   * - ``--jax-precision``
     - Integer choices in ``[32, 64]``
     - 64
     - ``--jax-precision 32``
     - Indicator for the JAX precision: 64-bit or 32-bit. Choose 32-bit may cause 'elbo decreases' warning.
   * - ``--output``
     - String
     - sushie_finemap
     - ``--output folder/trait_name``
     - Prefix for output files.
