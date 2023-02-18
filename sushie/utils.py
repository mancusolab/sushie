from typing import List, Optional, Tuple, Union

import pandas as pd
from glimix_core.lmm import LMM
from numpy_sugar.linalg import economic_qs
from scipy import stats

import jax.numpy as jnp
import jax.scipy as jsp

__all__ = [
    "ListFloatOrNone",
    "ols",
    "estimate_her",
    "regress_covar",
]

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


def ols(X: jnp.ndarray, y: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Perform ordinary linear regression using QR Factorization.

    Args:
        X: :math:`n \\times p` matrix for independent variables with no intercept vector.
        y: :math:`n \\times m` matrix for dependent variables. If :math:`m > 1`, then
            perform :math:`m` ordinary regression in parallel.

    Returns:
        :py:obj:`Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]`: A tuple of
            #. contains residuals (:py:obj:`jnp.ndarray`),
            #. adjusted :math:`r^2` (:py:obj:`jnp.ndarray`) for of the regression,
            #. :math:`p` values (:py:obj:`jnp.ndarray`) for the coefficients.

    """

    X_inter = jnp.append(jnp.ones((X.shape[0], 1)), X, axis=1)
    y = jnp.reshape(y, (len(y), -1))
    q_matrix, r_matrix = jnp.linalg.qr(X_inter, mode="reduced")
    qty = q_matrix.T @ y
    beta = jsp.linalg.solve_triangular(r_matrix, qty)
    df = q_matrix.shape[0] - q_matrix.shape[1]
    residual = y - q_matrix @ qty
    rss = jnp.sum(residual ** 2, axis=0)
    sigma = jnp.sqrt(jnp.sum(residual ** 2, axis=0) / df)
    se = (
        jnp.sqrt(
            jnp.diag(
                jsp.linalg.cho_solve((r_matrix, False), jnp.eye(r_matrix.shape[0]))
            )
        )[:, jnp.newaxis]
        @ sigma[jnp.newaxis, :]
    )
    t_scores = beta / se
    p_value = jnp.array(2 * stats.t.sf(abs(t_scores), df=df))

    r_sq = 1 - rss / jnp.sum((y - jnp.mean(y, axis=0)) ** 2, axis=0)
    adj_r = 1 - (1 - r_sq) * (q_matrix.shape[0] - 1) / df

    return residual, adj_r, p_value


def regress_covar(
    X: jnp.ndarray, y: jnp.ndarray, covar: jnp.ndarray, no_regress: bool
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Regress phenotypes and genotypes on covariates and return the residuals.

    Args:
        X: :math:`n \\times p` genotype matrix.
        y: :math:`n \\times 1` phenotype vector.
        covar: :math:`n \\times m` matrix for covariates.
        no_regress: boolean indicator whether to regress genotypes on covariates.

    Returns:
        :py:obj:`Tuple[jnp.ndarray, jnp.ndarray]`: A tuple of
            #. genotype residual matrix after regressing out covariates effects (:py:obj:`jnp.ndarray`),
            #. phenotype residual vector (:py:obj:`jnp.ndarray`) after regressing out covariates effects.

    """

    y, _, _ = ols(covar, y)
    if not no_regress:
        X, _, _ = ols(covar, X)

    return X, y


def estimate_her(
    X: jnp.ndarray, y: jnp.ndarray, covar: jnp.ndarray = None
) -> Tuple[float, jnp.ndarray, float, float, float]:
    """Calculate proportion of expression variation explained by genotypes (cis-heritability; :math:`h_g^2`).

    Args:
        X: :math:`n \\times p` matrix for independent variables with no intercept vector.
        y: :math:`n \\times 1` vector for gene expression.
        covar: :math:`n \\times m` matrix for covariates.

    Returns:
        :py:obj:`Tuple[float, float, float, float, float]`: A tuple of
            #. genetic variance (:py:obj:`float`) of the complex trait,
            #. :math:`h_g^2` (:py:obj:`float`) from `limix <https://github.com/limix/limix>`_ definition,
            #. :math:`h_g^2` (:py:obj:`float`) from `gcta <https://yanglab.westlake.edu.cn/software/gcta/>`_ definition,
            #. LRT test statistics (:py:obj:`float`) for :math:`h_g^2`,
            #. LRT :math:`p` value (:py:obj:`float`) for :math:`h_g^2`.

    """
    n, p = X.shape

    if covar is None:
        covar = jnp.ones(n)

    GRM = jnp.dot(X, X.T) / p
    GRM = GRM / jnp.diag(GRM).mean()
    QS = economic_qs(GRM)
    method = LMM(y, covar, QS, restricted=True)
    method.fit(verbose=False)

    g = method.scale * (1 - method.delta)
    e = method.scale * method.delta
    v = jnp.var(method.mean())
    h2g_w_v = g / (v + g + e)
    h2g_wo_v = g / (g + e)
    alt_lk = method.lml()
    method.delta = 1
    method.fix("delta")
    method.fit(verbose=False)
    null_lk = method.lml()
    lrt_stats = -2 * (null_lk - alt_lk)
    p_value = stats.chi2.sf(lrt_stats, 1) / 2

    return g, h2g_w_v, h2g_wo_v, lrt_stats, p_value
