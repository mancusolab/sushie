import typing

import pandas as pd

import jax.numpy as jnp

# from abc import ABC, abstractmethod


# from jax.tree_util import register_pytree_node, register_pytree_node_class

ArrayOrFloat = typing.Union[jnp.ndarray, float]
ListOrNone = typing.Union[typing.List[float], None]
ArrayOrNone = typing.Union[jnp.ndarray, None]


# VarUpdate = typing.Callable[
#     [jnp.ndarray, ArrayOrFloat, ArrayOrFloat, jnp.ndarray], ArrayOrFloat
# ]
#
# from jax.tree_util import register_pytree_node, register_pytree_node_class
#
#
# @register_pytree_node_class
# class AbstractOptFunc(ABC):
#     @abstractmethod
#     def __call__(
#         self,
#         beta_hat: jnp.ndarray,
#         shat2: ArrayOrFloat,
#         prior_covar_b: ArrayOrFloat,
#         prior_weights: jnp.ndarray,
#     ) -> ArrayOrFloat:
#         pass
#
#     def __init_subclass__(cls, **kwargs):
#         super().__init_subclass__(**kwargs)
#         register_pytree_node(cls, cls.tree_flatten, cls.tree_unflatten)
#
#     def tree_flatten(self):
#         children = None
#         aux = None
#         return (children, aux)
#
#     @classmethod
#     def tree_unflatten(cls, aux, children):
#         return cls()


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
    kl: jnp.ndarray


class SushieResult(typing.NamedTuple):
    priors: Prior
    posteriors: Posterior
    pip: jnp.ndarray
    cs: pd.DataFrame


class _LResult(typing.NamedTuple):
    Xs: typing.List[jnp.ndarray]
    ys: typing.List[jnp.ndarray]
    XtXs: typing.List[jnp.ndarray]
    priors: Prior
    posteriors: Posterior
    # opt_v_func: VarUpdate
    opt_v_func: ArrayOrNone
