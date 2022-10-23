# Copyright Contributors to the SuShiE project.
# SPDX-License-Identifier: Apache-2.0

import typing

import jax.numpy as jnp
import pandas as pd
from jax import random

ArrayOrFloat = typing.Union[jnp.ndarray, float]

class Priors(typing.NamedTuple):
    pi: jnp.ndarray
    resid_var: jnp.ndarray
    effect_covar: jnp.ndarray

class SushieResult(typing.NamedTuple):
    alpha: jnp.ndarray
    b: jnp.ndarray
    bsq: jnp.ndarray
    prior_covar_b: jnp.ndarray
    resid_covar: jnp.ndarray
    pip: jnp.ndarray
    cs: pd.DataFrame

class SERResult(typing.NamedTuple):
    alpha: jnp.ndarray
    post_mean: jnp.ndarray
    post_mean_sq: jnp.ndarray
    post_covar: jnp.ndarray
    prior_covar_b: jnp.ndarray

def get_pip(alpha: jnp.ndarray) -> jnp.ndarray:

    pip = 1 - jnp.prod((1 - alpha), axis=1)

    return pip

def get_cs_sushie(alpha: jnp.ndarray,
           X: jnp.ndarray,
           threshold = 0.9,
           purity_threshold = 0.5,) -> pd.DataFrame:

    P, L = alpha.shape
    tr_alpha = pd.DataFrame(alpha).reset_index()
    cs = pd.DataFrame(columns=["Lidx", "Pidx", "pip", "cpip"])
    n_pop = len(X)
    nX = []
    Xcorr = []
    for idx in range(n_pop):
        tmpNX = X[idx].shape[0]
        nX.append(tmpNX)
        Xcorr.append(jnp.abs(X[idx].T @ X[idx] / tmpNX))

    for idx in range(L):
        # select original index and alpha
        tmp_pd = tr_alpha[["index", idx]].sort_values(idx, ascending=False)
        tmp_pd["csum"]  = tmp_pd[[idx]].cumsum()
        tmp_cs = pd.DataFrame(columns=["Lidx", "Pidx", "pip", "cpip"])

        # for each SNP, add to credible set until the threshold is met
        for jdx in range(P):
            tmp_dict={"Lidx": idx, "Pidx": tmp_pd.iloc[jdx, 0], "pip": tmp_pd.iloc[jdx, 1], "cpip": tmp_pd.iloc[jdx, 2]}
            tmp_cs=pd.concat([tmp_cs, pd.DataFrame([tmp_dict])], ignore_index=True)
            cpip = tmp_pd.iloc[jdx, 2]
            if cpip > threshold:
                break

        # check the impurity
        pidx = (tmp_cs.Pidx.values).astype("int64")
        include_cs = True
        for jdx in range(n_pop):
            if jnp.min(Xcorr[jdx][pidx].T[pidx]) < purity_threshold:
                include_cs = False
        if include_cs:
            cs = pd.concat([cs, tmp_cs], ignore_index=True)

    return cs

def kl_categorical(alpha, pi) -> float:
    """
    KL divergence between two categorical distributions
    KL(alpha || pi)
    """
    return jnp.nansum(alpha * (jnp.log(alpha) - jnp.log(pi)))

def kl_multinormal(m1, s12, m0, s02) -> float:
    """
    KL divergence between multiN(m1, s12) and multiN(m0, s02)
    KL(multiN(m1, s12) || multiN(m0, s02))
    """
    # https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
    p1 = jnp.trace(jnp.einsum("ij,kjm->kim", jnp.linalg.inv(s02), s12), axis1 = 1, axis2 = 2) - len(s02)
    p2 = jnp.einsum("ij,jm,im->i", (m0 - m1), jnp.linalg.inv(s02), (m0 - m1))
    # a more stable way
    # p3 = jnp.log(jnp.linalg.det(s02) / jnp.linalg.det(s12))
    s0, sld0 = jnp.linalg.slogdet(s02)
    s1, sld1 = jnp.linalg.slogdet(s12)
    if s0 != 1 or jnp.any(s1 != 1):
        raise ValueError("Variance matrix is not PSD. Something is wrong.")

    p3 = sld0 - sld1

    return 0.5 * (p1 + p2 + p3)


# SuSiE Supplementarymaterial (B.9)
def erss(X: jnp.ndarray, y: jnp.ndarray, b: jnp.ndarray, bsq: jnp.ndarray) -> ArrayOrFloat:
    mu_li = X @ b
    mu2_li = (X ** 2) @ bsq

    term_1 = jnp.sum((y - jnp.sum(mu_li, axis=1)) ** 2)
    term_2 = jnp.sum(mu2_li - (mu_li ** 2))

    return term_1 + term_2


# SuSiE Supplementarymaterial (B.5)
def eloglike(X: jnp.ndarray, y: jnp.ndarray, b: jnp.ndarray, bsq: jnp.ndarray, resid_var: ArrayOrFloat) -> ArrayOrFloat:
    n, p = X.shape
    norm_term = -(0.5 * n) * jnp.log(2 * jnp.pi * resid_var)
    quad_term = -(0.5 / resid_var) * erss(X, y, b, bsq)

    return norm_term + quad_term

