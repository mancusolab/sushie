[![Documentation-webpage](https://img.shields.io/badge/Docs-Available-brightgreen)](https://mancusolab.github.io/sushie/)
[![Github](https://img.shields.io/github/stars/mancusolab/sushie?style=social)](https://github.com/mancusolab/sushie)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Project generated with PyScaffold](https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold)](https://pyscaffold.org/)

# SuShiE🍣

SuShiE (Sum of Shared Single Effect) is a Python software to fine-map
causal SNPs, compute prediction weights, and infer effect size
correlation for molecular data (e.g., mRNA levels and protein levels
etc.) across multiple ancestries.

``` diff
- We detest usage of our software or scientific outcome to promote racial discrimination.
```

SuShiE is described in
>  [Improved multi-ancestry fine-mapping identifies cis-regulatory variants underlying molecular traits and disease risk](https://www.medrxiv.org/content/10.1101/2024.04.15.24305836v1).
>
> Zeyun Lu,  Xinran Wang,  Matthew Carr,  Artem Kim,  Steven Gazal,  Pejman Mohammadi,  Lang Wu,  Alexander Gusev,  James Pirruccello,  Linda Kachuri,  Nicholas Mancuso.

Check [here](https://mancusolab.github.io/sushie/) for full
documentation.

  [**Installation**](#installation)
  | [**Example**](#get-started-with-example)
  | [**Notes**](#notes)
  | [**Version History**](#version-history)
  | [**Support**](#support)
  | [**Other Software**](#other-software)

## Installation

Users can download the latest repository and then use `pip`:

``` bash
git clone https://github.com/mancusolab/sushie.git
cd sushie
pip install .
```

*We currently only support Python3.8+.*

Before installation, we recommend to create a new environment using
[conda](https://docs.conda.io/en/latest/) so that it will not affect the
software versions of the other projects.

## Get Started with Example

SuShiE software is very easy to use:

For fine-mapping using individual-level data:
``` bash
cd ./data/
sushie finemap --pheno EUR.pheno AFR.pheno --vcf vcf/EUR.vcf vcf/AFR.vcf --covar EUR.covar AFR.covar --output ./test_result
```

For fine-mapping using summary-level data:
``` bash
cd ./data/
sushie finemap --summary --gwas EUR.gwas AFR.gwas --vcf vcf/EUR.vcf vcf/AFR.vcf --sample-size 489 639 --gwas-header chrom snp pos a1 a0 zs --output ./test_result
```

It can perform:

-   SuShiE: multi-ancestry fine-mapping accounting for ancestral
    correlation
-   Single-ancestry SuSiE (Sum of Single Effect)
-   Independent SuShiE: multi-ancestry SuShiE without accounting for
    correlation
-   Meta-SuSiE: single-ancestry SuSiE followed by meta-analysis
-   Mega-SuSiE: single-ancestry SuSiE on row-wise stacked data across
    ancestries (individual-level data only)
-   *cis*-molQTL effect size correlation estimation
-   *cis*-SNP heritability estimation (individual-level data only)
-   Cross-validation for SuShiE prediction weights (individual-level data only)
-   Convert prediction results to
    [FUSION](http://gusevlab.org/projects/fusion/) format, thus can be
    used in [TWAS](https://www.nature.com/articles/ng.3506) (individual-level data only)

See [here](https://mancusolab.github.io/sushie/) for more details on how
to use SuShiE.

If you want to use in-software SuShiE inference function, you can use
following Python code as an example:

``` python
from sushie.infer import infer_sushie
# Xs is for genotype data, and it should be a list of numpy array whose length is the number of ancestry.
# ys is for phenotype data, and it should also be a list of numpy array whose length is the number of ancestry.
infer_sushie(Xs=X, ys=y)
# Or summary-level data
# lds is for LD data, and it should be a list of p by p numpy array whose length is the number of ancestry.
# zs is for GWAS data, and it should be a list of numpy array whose length is the number of ancestry/
infer_sushie_ss(lds=LD, zs=GWAS, ns=np.array([100, 100]))
```

You can customize this function with your own ideas!

## Notes

-   SuShiE currently only supports **continuous** phenotype fine-mapping for individual-level data.
-   SuShiE uses [JAX](https://github.com/google/jax) with [Just In
    Time](https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html)
    compilation to achieve high-speed computation. However, there are
    some [issues](https://github.com/google/jax/issues/5501) for JAX
    with Mac M1 chip. To solve this, users need to initiate conda using
    [miniforge](https://github.com/conda-forge/miniforge), and then
    install SuShiE using `pip` in the desired environment.

## Version History

| Version | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
|---------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 0.1     | Initial Release                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| 0.11    | Fix the bug for OLS to compute adjusted r squared.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| 0.12    | Update io.corr function so that report all the correlation results no matter cs is pruned or not.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| 0.13    | Add `--keep` command to enable user to specify a file that contains the subjects ID SuShiE will perform on. Add `--ancestry_index` command to enable user to specify a file that contains the ancestry index for fine-mapping. With this, user can input single phenotype, genotype, and covariate file that contains all the subjects across ancestries. Implement padding to increase inference time. Record elbo at each iteration and can access it in the `infer.SuShiEResult` object. The alphas table now outputs the average purity and KL divergence for each `L`. Change `--kl_threshold` to `--divergence`. Add `--maf` command to remove SNPs that less than minor allele frequency threshold within each ancestry. Add `--max_select` command to randomly select maximum number of SNPs to compute purity to avoid unnecessary memory spending. Add a QC function to remove duplicated SNPs. |
| 0.14    | Remove KL-Divergence pruning. Enhance command line appearance and improve the output files contents. Fix small bugs on multivariate KL.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| 0.15    | Fix several typos; add a sanity check on reading vcf genotype data by assigning gt_types==Unknown as NA; Add preprint information.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| 0.16  | Implement summary-level data inference. Add option to remove ambiguous SNPs; fix several bugs and enhance codes quality.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |

## Support

For any questions, comments, bug reporting, and feature requests, please contact Zeyun Lu (<zeyunlu@usc.edu>) and
Nicholas Mancuso (<nmancuso@usc.edu>), and open a new thread in the [Issue
Tracker](https://github.com/mancusolab/sushie/issues).

## Other Software

Feel free to use other software developed by [Mancuso
Lab](https://www.mancusolab.com/):

-   [MA-FOCUS](https://github.com/mancusolab/ma-focus): a Bayesian
    fine-mapping framework using
    [TWAS](https://www.nature.com/articles/ng.3506) statistics across
    multiple ancestries to identify the causal genes for complex traits.
-   [SuSiE-PCA](https://github.com/mancusolab/susiepca): a scalable
    Bayesian variable selection technique for sparse principal component
    analysis
-   [twas_sim](https://github.com/mancusolab/twas_sim): a Python
    software to simulate [TWAS](https://www.nature.com/articles/ng.3506)
    statistics.
-   [FactorGo](https://github.com/mancusolab/factorgo): a scalable
    variational factor analysis model that learns pleiotropic factors
    from GWAS summary statistics.
-   [HAMSTA](https://github.com/tszfungc/hamsta): a Python software to
    estimate heritability explained by local ancestry data from
    admixture mapping summary statistics.
-   [Traceax](https://github.com/tszfungc/traceax): a Python library to perform stochastic trace estimation for linear operators.

------------------------------------------------------------------------

This project has been set up using PyScaffold 4.1.1. For details and
usage information on PyScaffold see <https://pyscaffold.org/>.
