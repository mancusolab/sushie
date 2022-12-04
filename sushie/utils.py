from typing import Tuple

import limix.her as her
from scipy import stats

import jax.numpy as jnp

from . import core

LOG = "sushie"


def ols(
    X: jnp.ndarray, y: jnp.ndarray
) -> Tuple[jnp.ndarray, core.ArrayOrFloat, jnp.ndarray]:
    """
    Perform an ordinary linear regression.

    param X: n x p matrix for independent variables with no intercept vector.
    param y: n x m matrix for dependent variables. If m > 1, then perform m ordinary regression parallel.

    :return: returns residuals, adjusted r squared, and p values for betas.
    """
    n_samples, n_features = X.shape
    X_inter = jnp.append(jnp.ones((n_samples, 1)), X, axis=1)
    n_features += 1
    XtX_inv = jnp.linalg.inv(X_inter.T @ X_inter)
    betas = XtX_inv @ X_inter.T @ y
    residual = y - X_inter @ betas
    rss = jnp.sum(residual ** 2, axis=0)
    sigma_sq = rss / (n_samples - n_features)
    t_scores = betas / jnp.squeeze(
        jnp.sqrt(jnp.diagonal(XtX_inv)[:, jnp.newaxis] * sigma_sq)
    )
    r_sq = 1 - rss / jnp.sum((y - jnp.mean(y)) ** 2)
    adj_r = 1 - (1 - r_sq) * (n_samples - 1) / (n_samples - n_features)
    p_val = jnp.array(2 * stats.t.sf(abs(t_scores), df=(n_samples - n_features)))

    return residual, adj_r, p_val


def estimate_her(X: jnp.ndarray, y: jnp.ndarray, C: jnp.ndarray = None) -> float:
    """
    Calculate proportion of gene expression variation explained by genotypes (cis-heritability).

    :param X: jnp.ndarray n x p matrix for genotypes.
    :param y: jnp.ndarray n x 1 vector for gene expression.
    :param C: jnp.ndarray n x m vector for covariates.

    :return: float returns cis-heritability using limix.
    """
    n, p = X.shape
    A = jnp.dot(X, X.T) / p
    heritability = her.estimate(y, "normal", A, C, verbose=False)

    return heritability


def _kl_categorical(
    alpha: jnp.ndarray,
    pi: jnp.ndarray,
) -> float:
    """
    KL divergence between two categorical distributions
    KL(alpha || pi)
    """
    return jnp.nansum(alpha * (jnp.log(alpha) - jnp.log(pi)))


def _kl_mvn(
    m1: jnp.ndarray,
    s12: jnp.ndarray,
    m0: float,
    s02: jnp.ndarray,
) -> float:
    """
    KL divergence between multiN(m1, s12) and multiN(m0, s02)
    KL(multiN(m1, s12) || multiN(m0, s02))
    """

    # https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
    k, k = s02.shape

    p1 = (
        jnp.trace(jnp.einsum("ij,kjm->kim", jnp.linalg.inv(s02), s12), axis1=1, axis2=2)
        - k
    )
    p2 = jnp.einsum("ij,jm,im->i", (m0 - m1), jnp.linalg.inv(s02), (m0 - m1))
    # a more stable way
    # p3 = jnp.log(jnp.linalg.det(s02) / jnp.linalg.det(s12))
    s0, sld0 = jnp.linalg.slogdet(s02)
    s1, sld1 = jnp.linalg.slogdet(s12)

    p3 = sld0 - sld1

    return 0.5 * (p1 + p2 + p3)


# SuSiE Supplementary material (B.9)
def _erss(
    X: jnp.ndarray, y: jnp.ndarray, beta: jnp.ndarray, beta_sq: jnp.ndarray
) -> core.ArrayOrFloat:
    mu_li = X @ beta
    mu2_li = (X ** 2) @ beta_sq

    term_1 = jnp.sum((y - jnp.sum(mu_li, axis=1)) ** 2)
    term_2 = jnp.sum(mu2_li - (mu_li ** 2))

    return term_1 + term_2


# SuSiE Supplementary material (B.5)


def _eloglike(
    X: jnp.ndarray,
    y: jnp.ndarray,
    beta: jnp.ndarray,
    beta_sq: jnp.ndarray,
    sigma_sq: core.ArrayOrFloat,
) -> core.ArrayOrFloat:
    n, p = X.shape
    norm_term = -(0.5 * n) * jnp.log(2 * jnp.pi * sigma_sq)
    quad_term = -(0.5 / sigma_sq) * _erss(X, y, beta, beta_sq)

    return norm_term + quad_term
