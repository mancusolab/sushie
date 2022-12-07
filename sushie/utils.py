from typing import Tuple

from glimix_core.lmm import LMM
from numpy_sugar.linalg import economic_qs
from scipy import stats

import jax.numpy as jnp

from . import core


def ols(
    X: jnp.ndarray, y: jnp.ndarray
) -> Tuple[jnp.ndarray, float, core.ArrayOrFloat, jnp.ndarray]:
    """Perform ordinary linear regression.

    Args:
        X: n x p matrix for independent variables with no intercept vector.
        y: n x m matrix for dependent variables. If m > 1, then perform m ordinary regression parallel.

    Returns:
        residual: regression residual   s
        sigma_sq: estimated residual variance
        adj_r: adjusted r squared
        p_val: p values for betas
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

    return residual, sigma_sq, adj_r, p_val


def estimate_her(
    X: jnp.ndarray, y: jnp.ndarray, Covar: jnp.ndarray = None
) -> Tuple[float, float]:
    """Calculate proportion of gene expression variation explained by genotypes (cis-heritability).

    Args:
        X: n x p matrix for independent variables with no intercept vector.
        y: n x 1 vector for gene expression.
        Covar: n x m vector for covariates or None.

    Returns:
        h2g: cis-heritability
        p_value: LRT p-value for cis-heritability
    """
    n, p = X.shape
    GRM = jnp.dot(X, X.T) / p

    # compute the likelihood ratio test
    # compute the null
    if Covar is not None:
        residual, sigma_sq, _, _ = ols(Covar, y)
        like_covar = Covar
    else:
        residual = y
        sigma_sq = jnp.var(y)
        like_covar = jnp.ones(n)

    null_lk = jnp.sum(stats.norm.logpdf(residual, loc=0, scale=jnp.sqrt(sigma_sq)))
    # compute the alternative
    GRM = GRM / jnp.diag(GRM).mean()
    QS = economic_qs(GRM)
    method = LMM(y, like_covar, QS, restricted=True)
    method.fit(verbose=False)
    alt_lk = method.lml()
    lrt_stats = -2 * (null_lk - alt_lk)
    p_value = stats.chi2.sf(lrt_stats, 1)
    g = method.scale * (1 - method.delta)
    e = method.scale * method.delta
    v = jnp.var(method.mean())
    h2g = g / (v + g + e)
    return h2g, p_value
