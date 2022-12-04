from abc import ABC, abstractmethod
from typing import List, NamedTuple, Optional, Union

import pandas as pd

import jax.numpy as jnp
from jax.tree_util import register_pytree_node, register_pytree_node_class

# prior argument effect_covar, resid_covar, rho, etc.
ListFloatOrNone = Optional[List[float]]
# covar process data, etc.
ArrayOrNoneList = List[Union[jnp.ndarray, None]]
# effect_covar susie, sushie etc.
ArrayOrFloat = Union[jnp.ndarray, float]
# covar paths
ListStrOrNone = Union[List[str], None]
# covar raw data
PDOrNone = Union[pd.DataFrame, None]


class RawData(NamedTuple):
    bim: pd.DataFrame
    fam: pd.DataFrame
    bed: jnp.ndarray
    pheno: pd.DataFrame
    covar: PDOrNone


class Prior(NamedTuple):
    """
    Define the class for the prior parameter of SuShiE model
    Attributes:
        pi: the prior probability for one SNP to be causal
        resid_var: the prior residual variance for all SNPs
        effect_covar: the prior effect sizes covariance matrix for all SNPs
    """

    pi: jnp.ndarray
    resid_var: jnp.ndarray
    effect_covar: jnp.ndarray


class Posterior(NamedTuple):
    """
    Define the class for the posterior parameter of SuShiE model
    Attributes:
        alpha: the posterior probability for each SNP to be causal (L x p)
        post_mean: the posterior mean for each SNP (L x p x k)
        post_mean_sq: the posterior mean square for each SNP (L x p x k x k, diagonal for kx k)
        post_covar: the posterior effect covariance for each SNP (L x p x k x k)
        kl: the KL divergence for each L
    """

    alpha: jnp.ndarray
    post_mean: jnp.ndarray
    post_mean_sq: jnp.ndarray
    post_covar: jnp.ndarray
    kl: jnp.ndarray


class SushieResult(NamedTuple):
    """
    Define the class for the SuShiE inference results
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


@register_pytree_node_class
class AbstractOptFunc(ABC):
    @abstractmethod
    def __call__(
        self,
        beta_hat: jnp.ndarray,
        shat2: ArrayOrFloat,
        priors: Prior,
        posteriors: Posterior,
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
