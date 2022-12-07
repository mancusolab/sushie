import math
from typing import List, NamedTuple, Tuple

import pandas as pd

import jax.numpy as jnp
import jax.scipy.stats as stats
from jax import jit, lax, nn

from . import core, log, utils


def run_sushie(
    Xs: List[jnp.ndarray],
    ys: List[jnp.ndarray],
    covar: core.ArrayOrNoneList,
    L: int = 5,
    no_scale: bool = False,
    no_regress: bool = False,
    no_update: bool = False,
    pi: jnp.ndarray = None,
    resid_var: core.ListFloatOrNone = None,
    effect_var: core.ListFloatOrNone = None,
    rho: core.ListFloatOrNone = None,
    max_iter: int = 500,
    min_tol: float = 1e-5,
    threshold: float = 0.9,
    purity: float = 0.5,
) -> core.SushieResult:
    """The main inference function for running SuShiE.
    Attributes:
        Xs: genotype data for multiple ancestries.
        ys: phenotype data for multiple ancestries.
        covar: covariate data for multiple ancestries.
        L: inferred number of eQTLs for the gene, default is five.
        no_scale: do not scale the genotype and phenotype, default is to scale.
        no_regress: do not regress covariates on genotypes, default is to regress.
        no_update: do not update the effect size prior, default is to update using em.
        pi: the probability prior for an SNP to be an eQTL, default is 1/p (None) where p is the number of SNPs.
        resid_var: prior residual variance, default is 1e-3 (None).
        effect_var: prior effect size variance, default is 1e-3 (None).
        rho: prior effect size correlation, default is 0.1 (None).
        max_iter: the maximum iteration for optimization, default is 500.
        min_tol: the convergence tolerance, default is 1e-5.
        threshold: the credible set threshold, default is 0.9.
        purity: default is 0.5.

    Returns:
        SuShiE result object that contains prior, posterior, cs, and pip
    """
    if len(Xs) == len(ys):
        n_pop = len(Xs)
    else:
        raise ValueError(
            "The number of geno and pheno data does not match. Check your input."
        )

    # check x and y have the same sample size
    for idx in range(n_pop):
        if Xs[idx].shape[0] != ys[idx].shape[0]:
            raise ValueError(
                f"Ancestry {idx + 1}: The samples of geno and pheno data does not match. Check your input."
            )

    # check each ancestry has the same number of SNPs
    for idx in range(1, n_pop):
        if Xs[idx - 1].shape[1] != Xs[idx].shape[1]:
            raise ValueError(
                f"Ancestry {idx} and ancestry {idx} do not have the same number of SNPs."
            )

    if L <= 0:
        raise ValueError("Inferred L is invalid, choose a positive L.")

    if min_tol > 0.1:
        log.logger.warning("Minimum intolerance is low. Inference may not be accurate.")

    if not 0 < threshold < 1:
        raise ValueError("CS threshold is not between 0 and 1. Specify a valid one.")

    if not 0 < purity < 1:
        raise ValueError("Purity is not between 0 and 1. Specify a valid one.")

    if pi is not None and (pi >= 1 or pi <= 0):
        raise ValueError(
            "Pi prior is not a probability (0-1). Specify a valid pi prior."
        )

    _, n_snps = Xs[0].shape

    if n_snps < L:
        raise ValueError(
            "The number of common SNPs across ancestries is less than inferred L."
            + "Please choose a smaller L or expand the genomic window."
        )

    if no_update:
        log.logger.info(
            "No updates on the effect size prior. Inference may be slow and different."
        )

    # first regress out covariates if there are any, then scale the genotype and phenotype
    if covar[0] is not None:
        for idx in range(n_pop):
            ys[idx], _, _, _ = utils.ols(covar[idx], ys[idx])

            # regress covar on each SNP
            if not no_regress:
                (
                    Xs[idx],
                    _,
                    _,
                    _,
                ) = utils.ols(covar[idx], Xs[idx])

    # center data
    for idx in range(n_pop):
        Xs[idx] -= jnp.mean(Xs[idx], axis=0)
        ys[idx] -= jnp.mean(ys[idx])
        # scale data if specified
        if not no_scale:
            Xs[idx] /= jnp.std(Xs[idx], axis=0)
            ys[idx] /= jnp.std(ys[idx])

        ys[idx] = jnp.squeeze(ys[idx])

    if resid_var is None:
        resid_var = []
        for idx in range(n_pop):
            resid_var.append(jnp.var(ys[idx], ddof=1))
    else:
        if len(resid_var) != n_pop:
            raise ValueError(
                "Number of specified residual prior does not match ancestry number."
            )
        resid_var = [float(i) for i in resid_var]
        if jnp.any(jnp.array(resid_var) <= 0):
            raise ValueError(
                "The input of residual prior is invalid (<0). Check your input."
            )

    if effect_var is None:
        effect_var = [1e-3] * n_pop
    else:
        if len(effect_var) != n_pop:
            raise ValueError(
                "Number of specified effect prior does not match ancestry number."
            )
        effect_var = [float(i) for i in effect_var]
        if jnp.any(jnp.array(effect_var) <= 0):
            raise ValueError("The input of effect size prior is invalid (<0).")

    exp_num_rho = math.comb(n_pop, 2)
    if rho is None:
        rho = [0.1] * exp_num_rho
    else:
        if n_pop == 1:
            log.logger.info("Running single-ancestry SuShiE. No need to specify rho.")
        if len(rho) != exp_num_rho:
            raise ValueError(
                f"Number of specified rho ({len(rho)}) does not match expected"
                + f"number {exp_num_rho}.",
            )
        rho = [float(i) for i in rho]
        # double-check the if it's invalid rho
        if jnp.any(jnp.abs(jnp.array(rho)) >= 1):
            raise ValueError(
                "The input of rho is invalid (>=1 or <=-1). Check your input."
            )

    effect_covar = jnp.diag(jnp.array(effect_var))
    ct = 0
    for row in range(1, n_pop):
        for col in range(n_pop):
            if col < row:
                _cov = jnp.sqrt(effect_var[row] * effect_var[col])
                effect_covar = effect_covar.at[row, col].set(rho[ct] * _cov)
                effect_covar = effect_covar.at[col, row].set(rho[ct] * _cov)
                ct += 1

    priors = core.Prior(
        pi=jnp.ones(n_snps) / float(n_snps) if pi is None else pi,
        resid_var=jnp.array(resid_var),
        # L x k x k
        effect_covar=jnp.array([effect_covar] * L),
    )

    posteriors = core.Posterior(
        alpha=jnp.zeros((L, n_snps)),
        post_mean=jnp.zeros((L, n_snps, n_pop)),
        post_mean_sq=jnp.zeros((L, n_snps, n_pop, n_pop)),
        post_covar=jnp.zeros((L, n_pop, n_pop)),
        kl=jnp.zeros((L,)),
    )

    opt_v_func = EMOptFunc() if not no_update else NoopOptFunc()

    XtXs = []
    for idx in range(n_pop):
        XtXs.append(jnp.sum(Xs[idx] ** 2, axis=0))

    elbo_last = -jnp.inf
    elbo_cur = -jnp.inf
    elbo_increase = True
    for o_iter in range(max_iter):
        priors, posteriors, elbo_cur = _update_effects(
            Xs,
            ys,
            XtXs,
            priors,
            posteriors,
            opt_v_func,
        )
        if elbo_cur < elbo_last and (not jnp.isclose(elbo_cur, elbo_last, atol=1e-8)):
            elbo_increase = False

        if jnp.abs(elbo_cur - elbo_last) < min_tol:
            log.logger.info(
                f"Reach minimum tolerance threshold {min_tol} after {o_iter + 1}"
                + " iterations, stop optimization.",
            )
            break

        if o_iter + 1 == max_iter:
            log.logger.info(f"Reach maximum iteration threshold {max_iter}.")
        elbo_last = elbo_cur

    if elbo_increase:
        log.logger.info(f"Optimization finished. Final ELBO score: {elbo_cur}.")
    else:
        log.logger.warning(
            f"ELBO does not non-decrease. Final ELBO score: {elbo_cur}."
            + " Double check your genotype, phenotype, and covariate data."
            + " Contact developer if this error message remains.",
        )

    pip = _get_pip(posteriors.alpha)
    cs = _get_cs(posteriors.alpha, Xs, threshold, purity)

    return core.SushieResult(priors, posteriors, pip, cs)


class _LResult(NamedTuple):
    Xs: List[jnp.ndarray]
    ys: List[jnp.ndarray]
    XtXs: List[jnp.ndarray]
    priors: core.Prior
    posteriors: core.Posterior
    opt_v_func: core.AbstractOptFunc


@jit
def _update_effects(
    Xs: List[jnp.ndarray],
    ys: List[jnp.ndarray],
    XtXs: List[jnp.ndarray],
    priors: core.Prior,
    posteriors: core.Posterior,
    opt_v_func: core.AbstractOptFunc,
) -> Tuple[core.Prior, core.Posterior, float]:
    l_dim, n_snps, n_pop = posteriors.post_mean.shape
    ns = [X.shape[0] for X in Xs]
    residual = []

    post_mean_lsum = jnp.sum(posteriors.post_mean, axis=0)
    for idx in range(n_pop):
        residual.append(ys[idx] - Xs[idx] @ post_mean_lsum[:, idx])

    init_l_result = _LResult(
        Xs=Xs,
        ys=residual,
        XtXs=XtXs,
        priors=priors,
        posteriors=posteriors,
        opt_v_func=opt_v_func,
    )

    l_result = lax.fori_loop(0, l_dim, _update_l, init_l_result)

    _, _, _, priors, posteriors, _ = l_result

    erss_list = []
    exp_ll = 0.0
    tr_b_s = posteriors.post_mean.T
    tr_bsq_s = jnp.einsum("nmij,ij->nmi", posteriors.post_mean_sq, jnp.eye(n_pop)).T
    for idx in range(n_pop):
        tr_b = tr_b_s[idx]
        tr_bsq = tr_bsq_s[idx]
        tmp_erss = _erss(Xs[idx], ys[idx], tr_b, tr_bsq) / ns[idx]
        erss_list.append(tmp_erss)
        exp_ll += _eloglike(Xs[idx], ys[idx], tr_b, tr_bsq, tmp_erss)

    priors = priors._replace(resid_var=jnp.array(erss_list))
    kl_divs = jnp.sum(posteriors.kl)
    elbo_score = exp_ll - kl_divs

    return priors, posteriors, elbo_score


def _update_l(l_iter: int, param: _LResult) -> _LResult:
    Xs, residual, XtXs, priors, posteriors, opt_v_func = param
    n_pop = len(Xs)
    residual_l = []

    for idx in range(n_pop):
        residual_l.append(
            residual[idx] + Xs[idx] @ posteriors.post_mean[l_iter, :, idx]
        )

    priors, posteriors = _ssr(
        Xs,
        residual_l,
        XtXs,
        priors,
        posteriors,
        l_iter,
        opt_v_func,
    )

    for idx in range(n_pop):
        residual[idx] = residual_l[idx] - Xs[idx] @ posteriors.post_mean[l_iter, :, idx]

    update_param = param._replace(
        ys=residual,
        priors=priors,
        posteriors=posteriors,
    )

    return update_param


def _ssr(
    Xs: List[jnp.ndarray],
    ys: List[jnp.ndarray],
    XtXs: List[jnp.ndarray],
    priors: core.Prior,
    posteriors: core.Posterior,
    l_iter: int,
    optimize_v: core.AbstractOptFunc,
) -> Tuple[core.Prior, core.Posterior]:
    n_pop = len(Xs)
    _, n_snps = Xs[0].shape

    beta_hat = jnp.zeros((n_snps, n_pop))
    shat2 = jnp.zeros((n_snps, n_pop))

    for idx in range(n_pop):
        Xty = Xs[idx].T @ ys[idx]
        beta_hat = beta_hat.at[:, idx].set(Xty / XtXs[idx])
        shat2 = shat2.at[:, idx].set(priors.resid_var[idx] / XtXs[idx])

    shat2 = jnp.eye(n_pop) * (shat2[:, jnp.newaxis])

    priors = optimize_v(beta_hat, shat2, priors, posteriors, l_iter)

    _, posteriors = _compute_posterior(beta_hat, shat2, priors, posteriors, l_iter)

    return priors, posteriors


def _compute_posterior(
    beta_hat: jnp.ndarray,
    shat2: core.ArrayOrFloat,
    priors: core.Prior,
    posteriors: core.Posterior,
    l_iter: int,
) -> Tuple[core.Prior, core.Posterior]:
    n_snps, n_pop = beta_hat.shape
    # quick way to calculate the inverse instead of using linalg.inv
    inv_shat2 = jnp.eye(n_pop) * (
        1 / jnp.diagonal(shat2, axis1=1, axis2=2)[:, jnp.newaxis]
    )

    prior_covar = priors.effect_covar[l_iter]
    post_covar = jnp.linalg.inv(inv_shat2 + jnp.linalg.inv(priors.effect_covar[l_iter]))
    rTZDinv = beta_hat / jnp.diagonal(shat2, axis1=1, axis2=2)

    post_mean = jnp.einsum("ijk,ik->ij", post_covar, rTZDinv)
    post_mean_sq = post_covar + jnp.einsum("ij,im->ijm", post_mean, post_mean)
    alpha = nn.softmax(
        jnp.log(priors.pi)
        - stats.multivariate_normal.logpdf(
            jnp.zeros((n_snps, n_pop)), post_mean, post_covar
        )
    )
    weighted_post_mean = post_mean * alpha[:, jnp.newaxis]
    weighted_post_mean_sq = post_mean_sq * alpha[:, jnp.newaxis, jnp.newaxis]
    # this is also the prior in our E step
    weighted_post_covar = jnp.einsum("j,jmn->mn", alpha, post_mean_sq)

    kl_alpha = _kl_categorical(alpha, priors.pi)
    kl_betas = alpha.T @ _kl_mvn(post_mean, post_covar, 0.0, prior_covar)

    priors = priors._replace(
        effect_covar=priors.effect_covar.at[l_iter].set(weighted_post_covar)
    )

    posteriors = posteriors._replace(
        alpha=posteriors.alpha.at[l_iter].set(alpha),
        post_mean=posteriors.post_mean.at[l_iter].set(weighted_post_mean),
        post_mean_sq=posteriors.post_mean_sq.at[l_iter].set(weighted_post_mean_sq),
        post_covar=posteriors.post_covar.at[l_iter].set(weighted_post_covar),
        kl=posteriors.kl.at[l_iter].set(kl_alpha + kl_betas),
    )

    return priors, posteriors


class EMOptFunc(core.AbstractOptFunc):
    def __call__(
        self,
        beta_hat: jnp.ndarray,
        shat2: core.ArrayOrFloat,
        priors: core.Prior,
        posteriors: core.Posterior,
        l_iter: int,
    ) -> core.Prior:
        priors, _ = _compute_posterior(beta_hat, shat2, priors, posteriors, l_iter)

        return priors


class NoopOptFunc(core.AbstractOptFunc):
    def __call__(
        self,
        beta_hat: jnp.ndarray,
        shat2: core.ArrayOrFloat,
        priors: core.Prior,
        posteriors: core.Posterior,
        l_iter: int,
    ) -> core.Prior:
        return priors


def _get_pip(alpha: jnp.ndarray) -> jnp.ndarray:
    pip = 1 - jnp.prod((1 - alpha), axis=0)

    return pip


def _get_cs(
    alpha: jnp.ndarray,
    Xs: List[jnp.ndarray],
    threshold: float = 0.9,
    purity: float = 0.5,
) -> pd.DataFrame:
    n_l, _ = alpha.shape
    t_alpha = pd.DataFrame(alpha.T).reset_index()
    cs = pd.DataFrame(columns=["CSIndex", "SNPIndex", "alpha", "c_alpha"])
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
            .rename(columns={"csum": "c_alpha", "index": "SNPIndex", idx: "alpha"})
        )

        # check the impurity
        snp_idx = tmp_pd.SNPIndex.values.astype("int64")

        min_corr = jnp.min(jnp.abs(ld[:, snp_idx][:, :, snp_idx]))
        if min_corr > purity:
            cs = pd.concat([cs, tmp_pd], ignore_index=True)

    return cs


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


def _kl_categorical(
    alpha: jnp.ndarray,
    pi: jnp.ndarray,
) -> float:
    return jnp.nansum(alpha * (jnp.log(alpha) - jnp.log(pi)))


def _kl_mvn(
    m1: jnp.ndarray,
    s12: jnp.ndarray,
    m0: float,
    s02: jnp.ndarray,
) -> float:
    # https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
    k, k = s02.shape

    p1 = (
        jnp.trace(jnp.einsum("ij,kjm->kim", jnp.linalg.inv(s02), s12), axis1=1, axis2=2)
        - k
    )
    p2 = jnp.einsum("ij,jm,im->i", (m0 - m1), jnp.linalg.inv(s02), (m0 - m1))

    s0, sld0 = jnp.linalg.slogdet(s02)
    s1, sld1 = jnp.linalg.slogdet(s12)

    p3 = sld0 - sld1

    return 0.5 * (p1 + p2 + p3)


def _erss(
    X: jnp.ndarray, y: jnp.ndarray, beta: jnp.ndarray, beta_sq: jnp.ndarray
) -> core.ArrayOrFloat:
    mu_li = X @ beta
    mu2_li = (X ** 2) @ beta_sq

    term_1 = jnp.sum((y - jnp.sum(mu_li, axis=1)) ** 2)
    term_2 = jnp.sum(mu2_li - (mu_li ** 2))

    return term_1 + term_2
