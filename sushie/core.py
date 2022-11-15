import typing

import jax.numpy as jnp
import pandas as pd

ArrayOrFloat = typing.Union[jnp.ndarray, float]
ListOrNone = typing.Union[typing.List[float], None]
ArrayOrNone = typing.Union[jnp.ndarray, None]


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


def get_pip(alpha: jnp.ndarray) -> jnp.ndarray:
    pip = 1 - jnp.prod((1 - alpha), axis=1)

    return pip


def get_cs(
    alpha: jnp.ndarray,
    Xs: typing.List[jnp.ndarray],
    threshold: float = 0.9,
    purity: float = 0.5,
) -> pd.DataFrame:
    _, n_l = alpha.shape
    t_alpha = pd.DataFrame(alpha).reset_index()
    cs = pd.DataFrame(columns=["CSIndex", "SNPIndex", "pip", "cpip"])
    n_pop = len(Xs)
    ld = [x.T @ x / x.shape[0] for x in Xs]

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
        include_cs = True
        for jdx in range(n_pop):
            if jnp.min(ld[jdx][snp_idx].T[snp_idx]) < purity:
                include_cs = False
        if include_cs:
            cs = pd.concat([cs, tmp_pd], ignore_index=True)

    return cs


def kl_categorical(
    alpha: jnp.ndarray,
    pi: jnp.ndarray,
) -> float:
    """
    KL divergence between two categorical distributions
    KL(alpha || pi)
    """
    return jnp.nansum(alpha * (jnp.log(alpha) - jnp.log(pi)))


def kl_mvn(
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
    if s0 != 1:
        raise ValueError("Prior effect covariance matrix is not valid.")
    elif jnp.any(s1 != 1):
        raise ValueError(
            "One of the posterior effect covariance matrices is not valid."
        )

    p3 = sld0 - sld1

    return 0.5 * (p1 + p2 + p3)


# SuSiE Supplementary material (B.9)
def erss(
    X: jnp.ndarray, y: jnp.ndarray, beta: jnp.ndarray, beta_sq: jnp.ndarray
) -> ArrayOrFloat:
    mu_li = X @ beta
    mu2_li = (X ** 2) @ beta_sq

    term_1 = jnp.sum((y - jnp.sum(mu_li, axis=1)) ** 2)
    term_2 = jnp.sum(mu2_li - (mu_li ** 2))

    return term_1 + term_2


# SuSiE Supplementary material (B.5)


def eloglike(
    X: jnp.ndarray,
    y: jnp.ndarray,
    beta: jnp.ndarray,
    beta_sq: jnp.ndarray,
    sigma_sq: ArrayOrFloat,
) -> ArrayOrFloat:
    n, p = X.shape
    norm_term = -(0.5 * n) * jnp.log(2 * jnp.pi * sigma_sq)
    quad_term = -(0.5 / sigma_sq) * erss(X, y, beta, beta_sq)

    return norm_term + quad_term
