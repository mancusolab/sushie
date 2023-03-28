.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly


    .. image:: https://readthedocs.org/projects/sushie/badge/?version=latest
        :alt: ReadTheDocs
        :target: https://sushie.readthedocs.io/en/stable/
    .. image:: https://img.shields.io/coveralls/github/<USER>/sushie/main.svg
        :alt: Coveralls
        :target: https://coveralls.io/r/<USER>/sushie

    .. image:: https://img.shields.io/conda/vn/conda-forge/sushie.svg
        :alt: Conda-Forge
        :target: https://anaconda.org/conda-forge/sushie
    .. image:: https://pepy.tech/badge/sushie/month
        :alt: Monthly Downloads
        :target: https://pepy.tech/project/sushie



.. image:: https://img.shields.io/badge/Docs-Available-brightgreen
        :alt: Documentation-webpage
        :target: https://mancusolab.github.io/sushie/

.. image:: https://img.shields.io/pypi/v/sushie.svg
           :alt: PyPI-Server
           :target: https://pypi.org/project/sushie/

.. image:: https://img.shields.io/github/stars/mancusolab/sushie?style=social
        :alt: Github
        :target: https://github.com/mancusolab/sushie

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
    :alt: License
    :target: https://opensource.org/licenses/MIT

.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/


======
SuShiE
======
Python software to fine-map causal SNPs, compute prediction weights, and infer effect size correlation across multiple ancestries. **The manuscript is in progress.**

.. code:: diff

    - We detest usage of our software or scientific outcome to promote racial discrimination.

Check `here <https://mancusolab.github.io/sushie/>`_ for full documentation.


|Installation|_ | |Example|_ | |Notes|_ | |Version|_ | |Support|_ | |Other Software|_

=================

.. _Installation:
.. |Installation| replace:: **Installation**

Installation
============

..
   The easiest way to install is with ``pip``:
    .. code:: bash
       pip install sushie

    Alternatively,

Users can download the latest repository and then use ``pip``:

.. code:: bash

    git clone https://github.com/mancusolab/sushie.git
    cd sushie
    pip install .

*We currently only support Python3.8+.*

Before installation, we recommend to create a new environment using `conda <https://docs.conda.io/en/latest/>`_ so that it will not affect the software versions of the other projects.

.. _Example:
.. |Example| replace:: **Example**

Get Started with Example
========================
SuShiE software is very easy to use:

.. code:: bash

    cd ./data/
    sushie finemap --pheno EUR.pheno AFR.pheno --vcf vcf/EUR.vcf vcf/AFR.vcf --covar EUR.covar AFR.covar --output ~/test_result

It can perform:

* SuShiE: multi-ancestry fine-mapping accounting for ancestral correlation
* Single-ancestry SuShiE
* Joint SuShiE: multi-ancestry SuShiE without accounting for correlation
* Meta SuShiE: single-ancestry SuShiE followed by meta-analysis
* Mega SuShiE: single-ancestry SuShiE on row-wise stacked data across ancestries
* QTL effect size correlation estimation
* Narrow-sense cis-heritability estimation
* Cross-validation for SuShiE prediction weights
* Convert prediction results to `FUSION <http://gusevlab.org/projects/fusion/>`_ format, thus can be used in `TWAS <https://www.nature.com/articles/ng.3506>`_

See `here <https://mancusolab.github.io/sushie/>`_ for more details on how to use SuShiE.

.. _Notes:
.. |Notes| replace:: **Notes**

Notes
=====

* SuShiE currently only supports **continuous** phenotype fine-mapping.
* SuShiE currently only supports fine-mapping on `autosomes <https://en.wikipedia.org/wiki/Autosome>`_.
* SuShiE uses `JAX <https://github.com/google/jax>`_ with `Just In Time  <https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html>`_ compilation to achieve high-speed computation. However, there are some `issues <https://github.com/google/jax/issues/5501>`_ for JAX with Mac M1 chip. To solve this, users need to initiate conda using `miniforge <https://github.com/conda-forge/miniforge>`_, and then install SuShiE using ``pip`` in the desired environment.

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
   * - 0.11
     - Fix the bug for OLS to compute adjusted r squared.
   * - 0.12
     - Update io.corr function so that report all the correlation results no matter cs is pruned or not. Change the prior correlation parameter from 0.8 to 0.1.

.. _Support:
.. |Support| replace:: **Support**

Support
========

Please report any bugs or feature requests in the `Issue Tracker <https://github.com/mancusolab/sushie/issues>`_. If users have any
questions or comments, please contact Zeyun Lu (zeyunlu@usc.edu) and Nicholas Mancuso (nmancuso@usc.edu).

.. _OtherSoftware:
.. |Other Software| replace:: **Other Software**

Other Software
==============

Feel free to use other software developed by `Mancuso Lab <https://www.mancusolab.com/>`_:

* `MA-FOCUS <https://github.com/mancusolab/ma-focus>`_: a Bayesian fine-mapping framework using `TWAS <https://www.nature.com/articles/ng.3506>`_ statistics across multiple ancestries to identify the causal genes for complex traits.

* `SuSiE-PCA <https://github.com/mancusolab/susiepca>`_: a scalable Bayesian variable selection technique for sparse principal component analysis

* `twas_sim <https://github.com/mancusolab/twas_sim>`_: a Python software to simulate `TWAS <https://www.nature.com/articles/ng.3506>`_ statistics.

---------------------

.. _pyscaffold-notes:

This project has been set up using PyScaffold 4.1.1. For details and usage
information on PyScaffold see https://pyscaffold.org/.
