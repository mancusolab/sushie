from abc import ABC, abstractmethod
from typing import List, NamedTuple, Optional, Union

import pandas as pd

import jax.numpy as jnp
from jax.tree_util import register_pytree_node, register_pytree_node_class

# prior argument effect_covar, resid_covar, rho, etc.
ListFloatOrNone = Optional[List[float]]
# covar process data, etc.
ListArrayOrNone = Optional[List[jnp.ndarray]]
# effect_covar sushie etc.
ArrayOrFloat = Union[jnp.ndarray, float]
# covar paths
ListStrOrNone = Optional[List[str]]
# covar raw data
PDOrNone = Optional[pd.DataFrame]


class CVData(NamedTuple):
    """Define the raw data object for the future inference.
    Attributes:
        train_geno: genotype data for training SuShiE weights.
        train_pheno: phenotype data for training SuShiE weights.
        valid_geno: genotype data for validating SuShiE weights.
        valid_pheno: phenotype data for validating SuShiE weights.
    """

    train_geno: List[jnp.ndarray]
    train_pheno: List[jnp.ndarray]
    valid_geno: List[jnp.ndarray]
    valid_pheno: List[jnp.ndarray]


class CleanData(NamedTuple):
    """Define the raw data object for the future inference.
    Attributes:
        geno: actual genotype data.
        pheno: phenotype data.
        covar: covariate needed to be adjusted in the inference.
    """

    geno: List[jnp.ndarray]
    pheno: List[jnp.ndarray]
    covar: ListArrayOrNone


class RawData(NamedTuple):
    """Define the raw data object for the future inference.
    Attributes:
        bim: SNP information data.
        fam: individual information data.
        bed: actual genotype data.
        pheno: phenotype data.
        covar: covariate needed to be adjusted in the inference.
    """

    bim: pd.DataFrame
    fam: pd.DataFrame
    bed: jnp.ndarray
    pheno: pd.DataFrame
    covar: PDOrNone


class Prior(NamedTuple):
    """Define the class for the prior parameter of SuShiE model
    Attributes:
        pi: the prior probability for one SNP to be causal
        resid_var: the prior residual variance for all SNPs
        effect_covar: the prior effect sizes covariance matrix for all SNPs
    """

    pi: jnp.ndarray
    resid_var: jnp.ndarray
    effect_covar: jnp.ndarray


class Posterior(NamedTuple):
    """Define the class for the posterior parameter of SuShiE model
    Attributes:
        alpha: the posterior probability for each SNP to be causal (L x p)
        post_mean: the alpha-weighted posterior mean for each SNP (L x p x k)
        post_mean_sq: the alpha-weighted posterior mean square for each SNP (L x p x k x k, a diagonal matrix for k x k)
        weighted_sum_covar: the alpha-weighted sum of posterior effect covariance across SNPs (L x k x k)
        kl: the KL divergence for each L
    """

    alpha: jnp.ndarray
    post_mean: jnp.ndarray
    post_mean_sq: jnp.ndarray
    weighted_sum_covar: jnp.ndarray
    kl: jnp.ndarray


class SushieResult(NamedTuple):
    """Define the class for the SuShiE inference results
    Attributes:
        priors: the final prior parameter for the inference
        posteriors: the final posterior parameter for the inference
        pip: the PIP for each SNP
        cs: the credible sets output after filtering on purity
    """

    priors: Prior
    posteriors: Posterior
    pip: jnp.ndarray
    cs: pd.DataFrame
    elbo: float
    elbo_increase: bool


class PriorAdjustor(NamedTuple):
    times: jnp.ndarray
    plus: jnp.ndarray


@register_pytree_node_class
class AbstractOptFunc(ABC):
    @abstractmethod
    def __call__(
        self,
        beta_hat: jnp.ndarray,
        shat2: ArrayOrFloat,
        priors: Prior,
        posteriors: Posterior,
        prior_adjustor: PriorAdjustor,
        l_iter: int,
    ) -> Prior:
        pass

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        register_pytree_node(cls, cls.tree_flatten, cls.tree_unflatten)

    def tree_flatten(self):
        children = ()
        aux = ()
        return (children, aux)

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls()
