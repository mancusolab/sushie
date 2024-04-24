========
SuShiE🍣
========

SuShiE (Sum of Shared Single Effect) is a Python software to fine-map causal SNPs, compute prediction weights, and infer effect size correlation for molecular data (e.g., mRNA levels and protein levels etc.) across multiple ancestries. **The manuscript is in progress.**

.. code:: diff

    - We detest usage of our software or scientific outcome to promote racial discrimination.


SuShiE is described in

.. code::

    `Improved multi-ancestry fine-mapping identifies cis-regulatory variants underlying molecular traits and disease risk <https://www.medrxiv.org/content/10.1101/2024.04.15.24305836v1>`_

    Zeyun Lu, Xinran Wang, Matthew Carr, Artem Kim, Steven Gazal, Pejman Mohammadi, Lang Wu, Alexander Gusev, James Pirruccello, Linda Kachuri, Nicholas Mancuso

Contents
========

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   Overview <readme>
   Model <model>
   Users Manual <manual>
   Files <files>


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

* `MA-FOCUS <https://github.com/mancusolab/ma-focus>`_: a Bayesian
    fine-mapping framework using statistics across multiple ancestries to identify the causal genes for complex traits.
* `SuSiE-PCA <https://github.com/mancusolab/susiepca>`_: a scalable Bayesian variable selection technique for sparse principal component analysis
* `twas_sim <https://github.com/mancusolab/twas_sim>`_: a Python software to simulate statistics.
* `FactorGo <https://github.com/mancusolab/factorgo>`_: a scalable variational factor analysis model that learns pleiotropic factors from GWAS summary statistics.
* `HAMSTA <https://github.com/tszfungc/hamsta>`_: a Python software to estimate heritability explained by local ancestry data from admixture mapping summary statistics.
* `Traceax <https://github.com/tszfungc/traceax>`_: a Python library to perform stochastic trace estimation for linear operators.
