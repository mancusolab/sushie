========
SuShiE🍣
========

SuShiE (Sum of Shared Single Effect) is a Python package for multiancestry SNP fine-mapping, estimating effect size correlations across ancestries, and computing ancestry-specific prediction weights using either individual-level or summary-level data for molecular or complex traits.

**We detest usage of our software or scientific outcome to promote racial discrimination.**

SuShiE is described in

`Improved multiancestry fine-mapping identifies cis-regulatory variants underlying molecular traits and disease risk <https://www.nature.com/articles/s41588-025-02262-7>`_ Nature Genetics. July, 2025.


*Zeyun Lu, Xinran Wang, Matthew Carr, Artem Kim, Steven Gazal, Pejman Mohammadi, Lang Wu, James Pirruccello, Linda Kachuri, Alexander Gusev, Nicholas Mancuso*

Contents
========

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   Overview <readme>
   Model <model>
   Users Manual <manual>
   Files <files>
   FAQ <faq>


.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/cli.rst
   api/infer.rst
   api/io.rst
   api/utils.rst

.. toctree::
   :maxdepth: 2
   :caption: Development

   Contributions & Help <contributing>
   Code of Conduct <conduct>
   Version History <version>
   Authors <authors>
   License <license>

Other Software
==============

Feel free to use other software developed by `Mancuso
Lab <https://www.mancusolab.com/>`_:

* `jaxQTL` <https://github.com/mancusolab/jaxqtl>: jaxQTL is a single-cell eQTL mapping tool using highly efficient count-based model (i.e., negative binomial or Poisson).
* `MA-FOCUS <https://github.com/mancusolab/ma-focus>`_: a Bayesian fine-mapping framework using statistics across multiple ancestries to identify the causal genes for complex traits.
* `SuSiE-PCA <https://github.com/mancusolab/susiepca>`_: a scalable Bayesian variable selection technique for sparse principal component analysis
* `twas_sim <https://github.com/mancusolab/twas_sim>`_: a Python software to simulate statistics.
* `FactorGo <https://github.com/mancusolab/factorgo>`_: a scalable variational factor analysis model that learns pleiotropic factors from GWAS summary statistics.
* `HAMSTA <https://github.com/tszfungc/hamsta>`_: a Python software to estimate heritability explained by local ancestry data from admixture mapping summary statistics.
* `Traceax <https://github.com/tszfungc/traceax>`_: a Python library to perform stochastic trace estimation for linear operators.
