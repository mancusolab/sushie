.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly

    .. image:: https://api.cirrus-ci.com/github/<USER>/sushie.svg?branch=main
        :alt: Built Status
        :target: https://cirrus-ci.com/github/<USER>/sushie
    .. image:: https://readthedocs.org/projects/sushie/badge/?version=latest
        :alt: ReadTheDocs
        :target: https://sushie.readthedocs.io/en/stable/
    .. image:: https://img.shields.io/coveralls/github/<USER>/sushie/main.svg
        :alt: Coveralls
        :target: https://coveralls.io/r/<USER>/sushie
    .. image:: https://img.shields.io/pypi/v/sushie.svg
        :alt: PyPI-Server
        :target: https://pypi.org/project/sushie/
    .. image:: https://img.shields.io/conda/vn/conda-forge/sushie.svg
        :alt: Conda-Forge
        :target: https://anaconda.org/conda-forge/sushie
    .. image:: https://pepy.tech/badge/sushie/month
        :alt: Monthly Downloads
        :target: https://pepy.tech/project/sushie
    .. image:: https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter
        :alt: Twitter
        :target: https://twitter.com/sushie

.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

|

======
SuShiE
======
Python software to fine-map causal SNPs, compute prediction weights of molecular data, and infer effect size correlation across multiple ancestries. The manuscript is in progress.

.. code:: diff

    - We detest usage of our software or scientific outcome to promote racial discrimination.

Check `here <https://mancusolab.github.io/sushie/>`_ for full documentation.


|Model|_ | |Installation|_ | |Example|_ | |Notes|_ | |Version|_ | |Reference|_ | |Support|_

.. _Model:
.. |Model| replace:: **Model**

=================

Model Description
=================
The Sum of SIngle Shared Effect (SuShiE) extends the Sum of SIngle Effect (SuSiE) model by introducing a prior correlation estimator to account for the ancestral quantitative trait loci (QTL) effect size similarity. Specifically, for $i^{\\text{th}}$ of total $k \\in \\mathbb{N}$ ancestries, we model the molecular data $g_i \\in \\mathbb{R}^{n_i \\times 1}$ for $n_i \\in \\mathbb{N}$ individuals as a linear combination of standardized genotype matrix $X_i \\in \\mathbb{R}^{n_i \\times p}$ for $p \\in \\mathbb{N}$ SNPs as

$$g_i=X_i \\beta_i+\\epsilon_i$$

$$\\beta_i=\\sum_{l=1}^{L}\\beta_{i,l}$$

$$\\beta_{i,l} = \\gamma_l \\cdot \b_{i, l}$$

$$\b_{l} = \\begin{bmatrix} \b_{1,l} \\\\ \\vdots \\\\ \b_{k,l} \\end{bmatrix} \\sim \\mathcal{N}(0, C_l) $$

$$C_{i,i',l}= \\begin{cases} \\sigma^2_{i,b,l} & \\text{if } i = i' \\\\ \\rho_{i,i',l} \\cdot \\sigma_{i,b,l} \\cdot \\sigma_{i',b,l} & \\text{otherwise}\\end{cases}$$

$$\\gamma_l \\sim \\text{Multi}(1, \\pi)$$

$$\\epsilon_i \\sim \\mathcal{N}(0, \\sigma^2_{i, e}I_{n_i})$$

where $\\beta_i \\in \\mathbb{R}^{p \\times1}$ is the shared QTL effects, $\\epsilon_i \\in \\mathbb{R}^{n_i \\times 1}$ is the ancestry-specific effects and other environmental noises, $L \\in \\mathbb{R}$ is the number of shared effects, for  $l^{\\text{th}}$  single shared effect,  $b_{i,l} \\in \\mathbb{R}$ is a scaler representing effect size, $C_l \\in \\mathbb{R}^{k \\times k}$ is the prior covariance matrix with $\\sigma^2_{i,b}$ as variance and $\\rho$ as correlation, $\\gamma_l$ is an binary indicator vector specifying which single SNP is the QTL, $\\pi$ is the prior probability for each SNP to be QTL, and $\\sigma^2_e$ is the prior variance for noises.

SuShiE runs `varitional inference <https://en.wikipedia.org/wiki/Variational_Bayesian_methods>`_ to estimate the posterior distribution for $\\beta_l$ and $\\gamma_l$ for each $l^{\\text{th}}$ effect. We can quantify the probability of QTL for each SNP through Posterior Inclusion Probabilities (PIPs). If the posterior distribution of $\\gamma_l$ is $\\text{Multi}(1, \\alpha_l)$, then for each SNP $j$, we have:

$$\\text{PIP}_j = 1 - \\prod_{l=1}^L(1 - \\alpha_{l, j})$$

Suppose $w_{l, j}$ is the posterior estimate of $\\beta_l$ for SNP $j$ and $l^\\text{th}$ effect, we can quantify the QTL prediction weights by summing across $L$ effects:

$$w_j = \\sum_{l=1}^Lw_{l,j}$$


.. _Installation:
.. |Installation| replace:: **Installation**

Installation
==========
The easiest way to install is with ``pip``:

.. code:: bash

    pip install sushie

Alternatively you can download the latest repository and then use ``pip``:

.. code:: bash

    git clone https://github.com/mancusolab/sushie.git
    cd sushie
    pip install

*We currently only support Python3.8+.*

Caveat:
~~~~~~

SuShiE uses `JAX <https://github.com/google/jax>`_ with `Just In Time  <https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html>`_ compliation to achieve high-speed computation. However, there are some `issues <https://github.com/google/jax/issues/5501>`_ for JAX with Mac M1 chip. To solve this, you need to initiate conda using `miniforge <https://github.com/conda-forge/miniforge>`_, and then install SuShiE using ``pip`` in your desired environment. In addition, we are not aware of any issues when running SuShiE in Linux OS.



.. _Example:
.. |Example| replace:: **Example**

Get Started with Example
========================
SuShiE software is very easy to use:

.. code:: bash

    cd ./data/
    sushie finemap --pheno EUR.pheno AFR.pheno --vcf vcf/EUR.vcf vcf/AFR.vcf --covar EUR.covar AFR.covar --output ~/test_result

It can perform:

* narrow-sense cis-heritability estimation
* QTL effect size correlation estimation
* mega-analysis (row-stack the data across ancestries)
* single-ancestry SuSiE followed by meta-analysis
* multi-ancestry SuSiE (correlation prior is set to 0)
* cross-validation for SuShiE prediction weights
* convert prediction results to `FUSION <http://gusevlab.org/projects/fusion/>`_ format

Please see the wiki for more details on how to use SuShiE.

.. _Notes:
.. |Notes| replace:: **Notes**

Notes
=====

SuShiE enable 64-bit precision for more accurate inference by default. If you need to enable 32-bit precision, comment out ``# config.update("jax_enable_x64", True)`` on the ``line 25`` in ``./sushie/cli.py``, and re-install SuShiE using ``pip``.

In addition, the default computation device for JAX is set by environment variables
(see `here <https://jax.readthedocs.io/en/latest/faq.html#faq-data-placement>`_). To change this before calling `sushie`, uncomment and modify ``platform = "cpu"`` and ``config.update("jax_platform_name", platform)`` on ``line 27`` and ``line 28`` in ``./sushie/cli.py``, and re-install SuShiE using ``pip``.

.. _Version:
.. |Version| replace:: **Version**

Version History
===============

.. list-table::
   :header-rows: 1

   * - Version
     - Description
   * - 0.1
     - Initial Release


.. _Reference:
.. |Reference| replace:: **Reference**

Reference
==========
.. [1] Wang, G., Sarkar, A., Carbonetto, P. and Stephens, M. (2020), A simple new approach to variable selection in regression, with application to genetic fine mapping. J. R. Stat. Soc. B, 82: 1273-1300. https://doi.org/10.1111/rssb.12388

.. _Support:
.. |Support| replace:: **Support**

Support
=======
Please report any bugs or feature requests in the `Issue Tracker <https://github.com/mancusolab/sushie/issues>`_. If you have any
questions or comments please contact zeyunlu@usc.edu and nmancuso@usc.edu.

Other Software
=============
Feel free to use other software developed by `Mancuso Lab <https://www.mancusolab.com/>`_:

* `MA-FOCUS <https://github.com/mancusolab/ma-focus>`_: a Bayesian fine-mapping framework using `TWAS <https://www.nature.com/articles/ng.3506>`_ statistics across multiple ancestries to identify the causal genes for complex traits.

* `SuSiE-PCA <https://github.com/mancusolab/susiepca>`_: a scalable Bayesian variable selection technique for sparse principal component analysis

* `twas_sim <https://github.com/mancusolab/twas_sim>`_: a Python software to simulate `TWAS <https://www.nature.com/articles/ng.3506>`_ statistics.


---------------------

.. _pyscaffold-notes:

This project has been set up using PyScaffold 4.1.1. For details and usage
information on PyScaffold see https://pyscaffold.org/.
