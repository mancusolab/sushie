import typing

import jax.numpy as jnp
import pandas as pd

ArrayOrFloat = typing.Union[jnp.ndarray, float]
ListOrNone = typing.Union[typing.List[float], None]
ArrayOrNone = typing.Union[jnp.ndarray, None]
VarUpdate = typing.Callable[
    [jnp.ndarray, ArrayOrFloat, ArrayOrFloat, jnp.ndarray], ArrayOrFloat
]


class CleanData(typing.NamedTuple):
    geno: typing.List[jnp.ndarray]
    pheno: typing.List[jnp.ndarray]
    covar: ListOrNone
    snp: pd.DataFrame
    h2g: ArrayOrNone


class Prior(typing.NamedTuple):
    pi: jnp.ndarray
    resid_var: jnp.ndarray
    effect_covar: jnp.ndarray


class Posterior(typing.NamedTuple):
    alpha: jnp.ndarray
    post_mean: jnp.ndarray
    post_mean_sq: jnp.ndarray
    post_covar: jnp.ndarray


class SushieResult(typing.NamedTuple):
    priors: Prior
    posteriors: Posterior
    pip: jnp.ndarray
    cs: pd.DataFrame
