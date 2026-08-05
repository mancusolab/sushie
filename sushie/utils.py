# pattern: Functional Core

import numpy as np
import polars as pl

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
ListFloatOrNone = list[float] | None
# covar process data, etc.
ListArrayOrNone = list[Array] | None
# effect_covar sushie etc.
ArrayOrFloat = ArrayLike
# covar paths
ListStrOrNone = list[str] | None
# covar raw data
PDOrNone = pl.DataFrame | None
# int or none
IntOrNone = int | None


def make_pip(alpha: ArrayLike) -> Array:
    """The function to calculate posterior inclusion probability (PIP).

    Args:
        alpha: $L \\times p$ matrix that contains posterior probability for SNP to be causal
            (i.e., $\\alpha$ in [model description](../model.md)).

    Returns:
        `Array`: $p \\times 1$ vector for the posterior inclusion probability.

    """

    alpha = jnp.asarray(alpha)
    pip = -jnp.expm1(jnp.sum(jnp.log1p(-alpha), axis=0))

    return pip


def rint(y_val: ArrayLike) -> Array:
    """Perform rank inverse normalization transformation.

    Args:
        y_val: $n \\times 1$ vector for dependent variables.

    Returns:
        `Array`: A array of transformed value.

    """

    y_val = jnp.asarray(y_val)
    n_pt = y_val.shape[0]
    r_y = stats.rankdata(np.asarray(y_val))
    q_y = stats.norm.ppf(r_y / (n_pt + 1))

    return jnp.asarray(q_y)


def ols(X: ArrayLike, y: ArrayLike) -> tuple[Array, Array, Array]:
    """Perform ordinary linear regression using QR Factorization.

    Args:
        X: $n \\times p$ matrix for independent variables with no intercept vector.
        y: $n \\times m$ matrix for dependent variables. If $m > 1$, then
            perform $m$ ordinary regression in parallel.

    Returns:
        `Tuple[Array, Array, Array]`: A tuple of
            - contains residuals (`Array`),
            - adjusted $r^2$ (`Array`) for of the regression,
            - $p$ values (`Array`) for the coefficients.

    """

    X = jnp.asarray(X)
    y = jnp.asarray(y)
    X_inter = jnp.concatenate([jnp.ones((X.shape[0], 1)), X], axis=1)
    y = jnp.reshape(y, (y.shape[0], -1))
    q_matrix, r_matrix = jnp.linalg.qr(X_inter, mode="reduced")
    qty = q_matrix.T @ y
    beta = jsp.linalg.solve_triangular(r_matrix, qty)
    df = q_matrix.shape[0] - q_matrix.shape[1]
    residual = y - q_matrix @ qty
    rss = jnp.sum(residual**2, axis=0)
    sigma = jnp.sqrt(jnp.sum(residual**2, axis=0) / df)
    se = (
        jnp.sqrt(jnp.diag(jsp.linalg.cho_solve((r_matrix, False), jnp.eye(r_matrix.shape[0]))))[:, jnp.newaxis]
        @ sigma[jnp.newaxis, :]
    )
    t_scores = beta / se
    p_value = jnp.asarray(2 * stats.t.sf(np.asarray(jnp.abs(t_scores)), df=df))

    r_sq = 1 - rss / jnp.sum((y - jnp.mean(y, axis=0)) ** 2, axis=0)
    adj_r = 1 - (1 - r_sq) * (q_matrix.shape[0] - 1) / df

    return residual, adj_r, p_value


def regress_covar(X: ArrayLike, y: ArrayLike, covar: ArrayLike, no_regress: bool) -> tuple[Array, Array]:
    """Regress phenotypes and genotypes on covariates and return the residuals.

    Args:
        X: $n \\times p$ genotype matrix.
        y: $n \\times 1$ phenotype vector.
        covar: $n \\times m$ matrix for covariates.
        no_regress: boolean indicator whether to regress genotypes on covariates.

    Returns:
        `Tuple[Array, Array]`: A tuple of
            - genotype residual matrix after regressing out covariates effects (`Array`),
            - phenotype residual vector (`Array`) after regressing out covariates effects.

    """

    X = jnp.asarray(X)
    y = jnp.asarray(y)
    covar = jnp.asarray(covar)

    y, _, _ = ols(covar, y)
    if not no_regress:
        X, _, _ = ols(covar, X)

    return X, y


def estimate_her(
    X: ArrayLike,
    y: ArrayLike,
    covar: ArrayLike | None = None,
    normalize: bool = True,
) -> tuple[float, Array, float, float]:
    """Calculate proportion of expression variation explained by genotypes (cis-heritability; $h_g^2$).

    Args:
        X: $n \\times p$ matrix for independent variables with no intercept vector.
        y: $n \\times 1$ vector for gene expression.
        covar: $n \\times m$ matrix for covariates.
        normalize: Boolean value to indicate whether normalize X and y

    Returns:
        `Tuple[float, Array, float, float]`: A tuple of
            - genetic variance (`float`) of the complex trait,
            - $h_g^2$ (`Array`) from [limix](https://github.com/limix/limix) definition,
            - LRT test statistics (`float`) for $h_g^2$,
            - LRT $p$ value (`float`) for $h_g^2$.

    Example:
        Estimate cis-heritability for a gene:

        ```python
        import numpy as np
        from sushie.utils import estimate_her

        # Genotype matrix (100 samples, 500 SNPs)
        X = np.random.randn(100, 500)
        # Gene expression
        y = np.random.randn(100)

        # Estimate heritability
        g, h2g, lrt_stat, p_value = estimate_her(X, y)
        print(f"Heritability: {h2g:.3f}, p-value: {p_value:.4f}")
        ```

    """
    X = jnp.asarray(X)
    y = jnp.asarray(y)

    n, p = X.shape

    if normalize:
        X -= jnp.mean(X, axis=0)
        X /= jnp.std(X, axis=0)
        y -= jnp.mean(y)
        y /= jnp.std(y)

    if covar is None:
        covar = jnp.ones(n)
    else:
        covar = jnp.asarray(covar)

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
