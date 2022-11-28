import logging
import typing

import limix.her as her
import pandas as pd
from scipy import stats

import jax.numpy as jnp

from . import core

LOG = "sushie"


def _get_pip(alpha: jnp.ndarray) -> jnp.ndarray:

    pip = 1 - jnp.prod((1 - alpha), axis=0)

    return pip


def _get_cs(
    alpha: jnp.ndarray,
    Xs: typing.List[jnp.ndarray],
    threshold: float = 0.9,
    purity: float = 0.5,
) -> pd.DataFrame:
    n_l, _ = alpha.shape
    t_alpha = pd.DataFrame(alpha.T).reset_index()
    cs = pd.DataFrame(columns=["CSIndex", "SNPIndex", "pip", "cpip"])
    # ld is always pxp, so it can be converted to jnp.array
    ld = jnp.array([x.T @ x / x.shape[0] for x in Xs])

    for idx in range(n_l):
        # select original index and alpha
        tmp_pd = (
            t_alpha[["index", idx]]
            .sort_values(idx, ascending=False)
            .reset_index(drop=True)
        )
        tmp_pd["csum"] = tmp_pd[[idx]].cumsum()
        n_row = tmp_pd[tmp_pd.csum < threshold].shape[0]

        # if all rows less than threshold + 1 is what we want to select
        if n_row == tmp_pd.shape[0]:
            select_idx = jnp.arange(n_row)
        else:
            select_idx = jnp.arange(n_row + 1)
        tmp_pd = (
            tmp_pd.iloc[select_idx, :]
            .assign(CSIndex=idx + 1)
            .rename(columns={"csum": "cpip", "index": "SNPIndex", idx: "pip"})
        )

        # check the impurity
        snp_idx = tmp_pd.SNPIndex.values.astype("int64")

        min_corr = jnp.min(jnp.abs(ld[:, snp_idx][:, :, snp_idx]))
        if min_corr > purity:
            cs = pd.concat([cs, tmp_pd], ignore_index=True)

    return cs


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


def _drop_na_inf(df: pd.DataFrame, nam: str, idx: int) -> pd.DataFrame:
    log = logging.getLogger(LOG)
    old_row = df.shape[0]
    df.replace([jnp.inf, -jnp.inf], jnp.nan)
    df.dropna(inplace=True)
    diff = old_row - df.shape[0]
    if diff != 0:
        log.info(
            f"Ancestry {idx + 1}: Drop {diff} rows from {nam} table due to INF value or NAN value."
        )
    return df


def ols(
    X: jnp.ndarray, y: jnp.ndarray
) -> typing.Tuple[jnp.ndarray, core.ArrayOrFloat, jnp.ndarray]:
    """
    Perform an ordinary linear regression.

    :param X: n x p matrix for independent variables with no intercept vector.
    :param y: n x m matrix for dependent variables. If m > 1, then perform m ordinary regression parallel.

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
    t_scores = betas / jnp.sqrt(jnp.diagonal(XtX_inv)[:, jnp.newaxis] * sigma_sq)
    r_sq = 1 - rss / jnp.sum((y - jnp.mean(y)) ** 2)
    adj_r = 1 - (1 - r_sq) * (n_samples - 1) / (n_samples - n_features)
    p_val = jnp.array(2 * stats.t.sf(abs(t_scores), df=(n_samples - n_features)))

    return residual, adj_r, p_val


def allele_check(
    base0: pd.Series,
    base1: pd.Series,
    compare0: pd.Series,
    compare1: pd.Series,
    idx: int,
) -> typing.Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Check whether SNPs alleles match across ancestries.

    :param base0: A0 for the first ancestry (baseline).
    :param base1: A1 for the first ancestry (baseline).
    :param compare0: A0 for the compared ancestry.
    :param compare1: A1 for the compared ancestry.
    :param idx: Ancestry Index.

    :return: indices for correct SNPs, flipped SNPs, and wrong SNPs (SNPs cannot be flipped).
    """

    log = logging.getLogger(LOG)
    correct = jnp.logical_and(base0 == compare0, base1 == compare1)
    flipped = jnp.logical_and(base0 == compare1, base1 == compare0)
    correct_idx = jnp.where(correct)[0]
    flipped_idx = jnp.where(flipped)[0]
    wrong_idx = jnp.where(jnp.logical_not(jnp.logical_or(correct, flipped)))[0]

    if len(flipped_idx) != 0:
        log.warning(
            f"Ancestry{idx + 1} has {len(flipped_idx)} flipped alleles from ancestry 1. Will flip these SNPs."
        )

    if len(wrong_idx):
        log.warning(
            f"Ancestry{idx + 1} has {len(wrong_idx)} alleles that couldn't be flipped. Will remove these SNPs."
        )

    return correct_idx, flipped_idx, wrong_idx


def _estimate_her(X: jnp.ndarray, y: jnp.ndarray, C: jnp.ndarray = None) -> float:
    """
    Perform an ordinary linear regression.

    :param X: jnp.ndarray n x p matrix for independent variables with no intercept vector.
    :param y: jnp.ndarray n x m matrix for dependent variables. If m > 1, then perform m ordinary regression parallel.

    :return: typing.Tuple[jnp.ndarray, core.ArrayOrFloat, core.ArrayOrFloat] returns residuals, adjusted r squared,
            and p values for betas.
    """
    n, p = X.shape
    A = jnp.dot(X, X.T) / p
    h2g = her.estimate(y, "normal", A, C, verbose=False)

    return h2g
