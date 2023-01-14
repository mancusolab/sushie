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
Software to fine-map causal SNPs, compute prediction weights of molecular data, and infer effect size correlation across multiple ancestries. The manuscript is in progress.

.. code:: diff

    - We detest usage of our software or scientific outcome to promote racial discrimination.

Check `here <https://mancusolab.github.io/sushie/>`_ for full documentation


|Model|_ | |Installation|_ | |Example|_ | |Notes|_ | |References|_ | |Support|_

.. _Model:
.. |Model| replace:: **Model**

Model Description
=================
The Sum of SIngle Shared Effect (SuShiE) extends the Sum of SIngle Effect (SuSiE) model by introducing a prior correlation estimator to account for the ancestral effect size similarity. Specifically, for $i^\text{th}$ of total $k \in \N$ ancestries, we model the molecular data $g_i \in \R^{n_i \times 1}$ for $n_i \in \N$ individuals as a linear combination of standardized genotype matrix $X_i \in \R^{n_i \times p}$ for $p \in \N$ SNPs as

$$g_i=X_i β_i+ϵ_i  $$
β_i=∑_(l=1)^L▒β_(i,l)
β_(i,l)=γ_l∙b_(i,l)
b_l=[█(b_(1,l)@⋮@b_(i,l)@⋮@b_(k,l) )]  ~N(0,C_l)
C_(i,i^',l)={█(σ_(i,b,l)^2                         "if"  i=i^'@ρ_(i,i^',l) σ_(i,b,l) σ_(i^',b,l)     "otherwise" )┤
γ_l="Multi"(1,π)
ϵ_i  ~ N(0,σ_(i,e)^2 I_(n_i ))
$$

We extend the Sum of Single Effects model (i.e. SuSiE) [1]_ to principal component analysis. $Z_{N \\times K}$ is the latent factors

$$X | Z,W \\sim \\mathcal{MN}_{N,P}(ZW, I_N, \\sigma^2 I_P)$$

$$\\mathbf{w}_k = \\sum_{l=1}^L \\mathbf{w}_{kl} $$
$$\\mathbf{w}_{kl} = w_{kl} \\gamma_{kl}$$
$$w_{kl} \\sim \\mathcal{N}(0,\\sigma^2_{0kl})$$
$$\\gamma_{kl} | \\pi \\sim \\text{Multi}(1,\\pi) $$

.. _Installation:
.. |Installation| replace:: **Installation**

Installation
==========
The easiest way to install is with ``pip``:

.. code:: bash

    pip install sushie

Alternatively you can download the latest repo and then use setuptools:

.. code:: bash

    git clone https://github.com/mancusolab/sushie.git
    cd sushie
    pip install

*We currently only support Python3.8+.*

Caveat:
~~~~~~

SuShiE uses `JAX <https://github.com/google/jax>`_ with `Just In Time  <https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html>`_ compliation to achieve high-speed computation. However, there are some `issues <https://github.com/google/jax/issues/5501>`_ for JAX with Mac M1 chip. To solve this, you need to initiate conda using `miniforge <https://github.com/conda-forge/miniforge>`_, and then install SuShiE using ``pip`` in your desired environment.



.. _Example:
.. |Example| replace:: **Example**

Get Started with Example
========================

Please see the wiki for more details on how to use SuShiE to database files.

.. _Notes:
.. |Notes| replace:: **Notes**

Notes
=====

`JAX <https://github.com/google/jax>`_ uses 32-bit precision by default. To enable 64-bit precision before calling
`sushie` add the following code:

.. code:: python

   import jax
   jax.config.update("jax_enable_x64", True)

Similarly, the default computation device for `JAX <https://github.com/google/jax>`_ is set by environment variables
(see `here <https://jax.readthedocs.io/en/latest/faq.html#faq-data-placement>`_). To change this programmatically before
calling `sushie` add the following code:

.. code:: python

   import jax
   platform = "gpu" # "gpu", "cpu", or "tpu"
   jax.config.update("jax_platform_name", platform)

.. _References:
.. |References| replace:: **References**

References
==========
.. [1] Wang, G., Sarkar, A., Carbonetto, P. and Stephens, M. (2020), A simple new approach to variable selection in regression, with application to genetic fine mapping. J. R. Stat. Soc. B, 82: 1273-1300. https://doi.org/10.1111/rssb.12388

.. _Support:
.. |Support| replace:: **Support**

Support
=======
Please report any bugs or feature requests in the `Issue Tracker <https://github.com/mancusolab/sushie/issues>`_. If you have any
questions or comments please contact zeyunlu@usc.edu and/or nmancuso@usc.edu.

Other Software
=============
MA-FOCUS

TWAS Simulator

SuSiE PCA is a scalable Bayesian variable selection technique for sparse principal component analysis

---------------------

.. _pyscaffold-notes:

This project has been set up using PyScaffold 4.1.1. For details and usage
information on PyScaffold see https://pyscaffold.org/.
