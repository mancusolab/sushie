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

.. _Documentation: https://mancusolab.github.io/sushie/
.. |Documentation| replace:: **Documentation**

======
SuShiE
======
Software to compute multi-ancestry functional data prediction weights using SuShiE

|Documentation|_ | |Installation|_ | |Example|_ | |Notes|_ | |References|_

```diff
- We detest usage of our software or scientific outcome to promote racial discrimination.
```

.. _Installation:
.. |Installation| replace:: **Installation**

Installing
==========
The easiest way to install is with pip:

    pip install sushie

Alternatively you can download the latest repo and then use setuptools:

    git clone https://github.com/mancusolab/sushie.git
    cd sushie
    python setup.py install

*We currently only support Python3.8+.*

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

Software and support
====================
If you have any questions or comments please contact nicholas.mancuso@med.usc.edu and zeyunlu@usc.edu

For performing various inferences using summary data from large-scale GWASs please find the following useful software:

1. Association between predicted expression and complex trait/disease [FUSION](https://github.com/gusevlab/fusion_twas) and [PrediXcan](https://github.com/hakyimlab/PrediXcan)
2. Estimating local heritability or genetic correlation [HESS](https://github.com/huwenboshi/hess)
3. Estimating genome-wide heritability or genetic correlation [UNITY](https://github.com/bogdanlab/UNITY)
4. Fine-mapping using summary-data [PAINTOR](https://github.com/gkichaev/PAINTOR_V3.0)
5. Imputing summary statistics using LD [FIZI](https://github.com/bogdanlab/fizi)
6. TWAS simulator (https://github.com/mancusolab/twas_sim)
7. Multi-ancestry TWAS fine-mapping [MA-FOCUS](https://github.com/mancusolab/ma-focus)

---------------------

.. _pyscaffold-notes:

This project has been set up using PyScaffold 4.1.1. For details and usage
information on PyScaffold see https://pyscaffold.org/.
