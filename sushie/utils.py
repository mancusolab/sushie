from typing import List, Optional, Tuple

import pandas as pd
from glimix_core.lmm import LMM
from numpy_sugar.linalg import economic_qs
from scipy import stats

import jax.numpy as jnp
import jax.scipy as jsp
from jax import Array
from jax.typing import ArrayLike

__all__ = [
    "make_pip",
    "rint",
    "ListFloatOrNone",
    "IntOrNone",
    "ols",
    "estimate_her",
    "regress_covar",
]

# prior argument effect_covar, resid_covar, rho, etc.
ListFloatOrNone = Optional[List[float]]
# covar process data, etc.
ListArrayOrNone = Optional[List[ArrayLike]]
# effect_covar sushie etc.
ArrayOrFloat = ArrayLike
# covar paths
ListStrOrNone = Optional[List[str]]
# covar raw data
PDOrNone = Optional[pd.DataFrame]
# int or none
IntOrNone = Optional[int]


def make_pip(alpha: ArrayLike) -> Array:
    """The function to calculate posterior inclusion probability (PIP).

    Args:
        alpha: :math:`L \\times p` matrix that contains posterior probability for SNP to be causal
            (i.e., :math:`\\alpha` in :ref:`Model`).

    Returns:
        :py:obj:`Array`: :math:`p \\times 1` vector for the posterior inclusion probability.

    """

    pip = -jnp.expm1(jnp.sum(jnp.log1p(-alpha), axis=0))

    return pip


def rint(y_val: ArrayLike) -> Array:
    """Perform rank inverse normalization transformation.

    Args:
        y_val: :math:`n \\times 1` vector for dependent variables.

    Returns:
        :py:obj:`Array`: A array of transformed value.

    """

    n_pt = y_val.shape[0]
    r_y = stats.rankdata(y_val)
    q_y = stats.norm.ppf(r_y / (n_pt + 1))

    return q_y


def ols(X: ArrayLike, y: ArrayLike) -> Tuple[Array, Array, Array]:
    """Perform ordinary linear regression using QR Factorization.

    Args:
        X: :math:`n \\times p` matrix for independent variables with no intercept vector.
        y: :math:`n \\times m` matrix for dependent variables. If :math:`m > 1`, then
            perform :math:`m` ordinary regression in parallel.

    Returns:
        :py:obj:`Tuple[Array, Array, Array]`: A tuple of
            #. contains residuals (:py:obj:`Array`),
            #. adjusted :math:`r^2` (:py:obj:`Array`) for of the regression,
            #. :math:`p` values (:py:obj:`Array`) for the coefficients.

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
    X: ArrayLike, y: ArrayLike, covar: ArrayLike, no_regress: bool
) -> Tuple[Array, Array]:
    """Regress phenotypes and genotypes on covariates and return the residuals.

    Args:
        X: :math:`n \\times p` genotype matrix.
        y: :math:`n \\times 1` phenotype vector.
        covar: :math:`n \\times m` matrix for covariates.
        no_regress: boolean indicator whether to regress genotypes on covariates.

    Returns:
        :py:obj:`Tuple[Array, Array]`: A tuple of
            #. genotype residual matrix after regressing out covariates effects (:py:obj:`Array`),
            #. phenotype residual vector (:py:obj:`Array`) after regressing out covariates effects.

    """

    y, _, _ = ols(covar, y)
    if not no_regress:
        X, _, _ = ols(covar, X)

    return X, y


def estimate_her(
    X: ArrayLike,
    y: ArrayLike,
    covar: ArrayLike = None,
    normalize: bool = True,
) -> Tuple[float, Array, float, float]:
    """Calculate proportion of expression variation explained by genotypes (cis-heritability; :math:`h_g^2`).

    Args:
        X: :math:`n \\times p` matrix for independent variables with no intercept vector.
        y: :math:`n \\times 1` vector for gene expression.
        covar: :math:`n \\times m` matrix for covariates.
        normalize: Boolean value to indicate whether normalize X and y

    Returns:
        :py:obj:`Tuple[float, float, float, float, float]`: A tuple of
            #. genetic variance (:py:obj:`float`) of the complex trait,
            #. :math:`h_g^2` (:py:obj:`float`) from `limix <https://github.com/limix/limix>`_ definition,
            #. LRT test statistics (:py:obj:`float`) for :math:`h_g^2`,
            #. LRT :math:`p` value (:py:obj:`float`) for :math:`h_g^2`.

    """
    n, p = X.shape

    if normalize:
        X -= jnp.mean(X, axis=0)
        X /= jnp.std(X, axis=0)
        y -= jnp.mean(y)
        y /= jnp.std(y)

    if covar is None:
        covar = jnp.ones(n)

    GRM = jnp.dot(X, X.T) / p
    # normalize the covariance matrix as suggested by Limix
    # https://horta-limix.readthedocs.io/en/api/_modules/limix/her/_estimate.html#estimate
    # and https://horta-limix.readthedocs.io/en/api/_modules/limix/qc/kinship.html#normalise_covariance
    # here, we calculate GRM using p, instead of p-1, so jnp.diag.mean should be equivalent to jnp.trace/(p-1)
    GRM /= jnp.diag(GRM).mean()
    QS = economic_qs(GRM)
    method = LMM(y, covar, QS, restricted=True)
    method.fit(verbose=False)  # alternative

    g = method.scale * (1 - method.delta)
    e = method.scale * method.delta
    v = jnp.var(method.mean())
    h2g = g / (v + g + e)
    alt_lk = method.lml()
    method.delta = 1
    method.fix("delta")
    method.fit(verbose=False)  # null
    null_lk = method.lml()
    lrt_stats = -2 * (null_lk - alt_lk)
    # https://en.wikipedia.org/wiki/Wilks%27_theorem
    p_value = stats.chi2.sf(lrt_stats, 1) / 2

    return g, h2g, lrt_stats, p_value
