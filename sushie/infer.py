import logging
import math
import typing

import jax.numpy as jnp
import jax.scipy.stats as stats
from jax import jit, lax, nn

from . import core, utils

LOG = "sushie"


class _LResult(typing.NamedTuple):
    """ """

    Xs: typing.List[jnp.ndarray]
    ys: typing.List[jnp.ndarray]
    XtXs: typing.List[jnp.ndarray]
    priors: core.Prior
    posteriors: core.Posterior
    opt_v_func: core.AbstractOptFunc
    # opt_v_func: core.ArrayOrNone


def run_sushie(
    Xs: typing.List[jnp.ndarray],
    ys: typing.List[jnp.ndarray],
    L: int,
    norm_X: bool = True,
    norm_y: bool = False,
    pi: jnp.ndarray = None,
    resid_var: jnp.ndarray = None,
    effect_var: jnp.ndarray = None,
    rho: jnp.ndarray = None,
    max_iter: int = 500,
    min_tol: float = 1e-5,
    opt_mode: str = "em",
    threshold: float = 0.9,
    purity: float = 0.5,
) -> core.SushieResult:
    """
    The main inference function for SuSiE PCA.
    Attributes:
        mu_z: mean parameter for factor Z

    """

    log = logging.getLogger(LOG)
    # check if number of the ancestry are the same
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

    _, n_snps = Xs[0].shape

    if n_snps < L:
        raise ValueError(
            "The number of common SNPs across ancestries is less than inferred L."
            + "Please choose a smaller L or expand the genomic window."
        )

    if min_tol > 0.1:
        log.warning("Minimum intolerance is low. Inference may not be accurate.")

    if not 0 < threshold < 1:
        raise ValueError("CS threshold is not between 0 and 1. Specify a valid one.")

    if not 0 < purity < 1:
        raise ValueError("Purity is not between 0 and 1. Specify a valid one.")

    for idx in range(n_pop):
        if norm_X:
            Xs[idx] = (Xs[idx] - jnp.mean(Xs[idx], axis=0)) / jnp.std(Xs[idx], axis=0)
        if norm_y:
            ys[idx] = (ys[idx] - jnp.mean(ys[idx])) / jnp.std(ys[idx])

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

    if opt_mode == "noop":
        if resid_var is None or effect_var is None or rho is None:
            raise ValueError(
                "'noop' is specified. rho, resid_var, and effect_var cannot be None."
            )

    if pi is not None and (pi >= 1 or pi <= 0):
        raise ValueError(
            "Pi prior is not a probability (0-1). Specify a valid pi prior."
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
        post_covar=jnp.zeros((L, n_snps, n_pop, n_pop)),
        kl=jnp.zeros((L,)),
    )

    # hard code for now only use "em", instead of "noop"
    opt_v_func = _construct_optimize_v(opt_mode)
    # opt_v_func = None

    XtXs = []
    for idx in range(n_pop):
        XtXs.append(jnp.sum(Xs[idx] ** 2, axis=0))

    elbo = [-jnp.inf]
    for o_iter in range(max_iter):
        priors, posteriors, elbo_score = _update_effects(
            Xs,
            ys,
            XtXs,
            priors,
            posteriors,
            opt_v_func,
        )
        elbo.append(elbo_score)

        if jnp.abs(elbo[o_iter + 1] - elbo[o_iter]) < min_tol:
            log.warning(
                f"Reach minimum tolerance threshold {min_tol} after {o_iter + 1}"
                + " iterations, stop optimization.",
            )
            break

        if o_iter + 1 == max_iter:
            log.warning(
                f"Reach maximum iteration threshold {max_iter} after {o_iter + 1}."
            )

    if all(i <= j or jnp.isclose(i, j, atol=1e-8) for i, j in zip(elbo, elbo[1:])):
        log.info(f"Optimization finished. Final ELBO score: {elbo[-1]}.")
    else:
        raise ValueError(
            f"ELBO does not non-decrease. Final ELBO score: {elbo[-1]}."
            + " Double check your genotype, phenotype, and covariate data."
            + " Contact developer if this error message remains.",
        )

    pip = utils._get_pip(posteriors.alpha)
    cs = utils._get_cs(posteriors.alpha, Xs, threshold, purity)

    return core.SushieResult(priors, posteriors, pip, cs)


@jit
def _update_effects(
    Xs: typing.List[jnp.ndarray],
    ys: typing.List[jnp.ndarray],
    XtXs: typing.List[jnp.ndarray],
    priors: core.Prior,
    posteriors: core.Posterior,
    opt_v_func: core.AbstractOptFunc,
    # opt_v_func: core.ArrayOrNone,
) -> typing.Tuple[core.Prior, core.Posterior, float]:
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
        tmp_erss = utils._erss(Xs[idx], ys[idx], tr_b, tr_bsq) / ns[idx]
        erss_list.append(tmp_erss)
        exp_ll += utils._eloglike(Xs[idx], ys[idx], tr_b, tr_bsq, tmp_erss)

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
    Xs: typing.List[jnp.ndarray],
    ys: typing.List[jnp.ndarray],
    XtXs: typing.List[jnp.ndarray],
    priors: core.Prior,
    posteriors: core.Posterior,
    l_iter: int,
    optimize_v: core.AbstractOptFunc,
) -> typing.Tuple[core.Prior, core.Posterior]:
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
) -> typing.Tuple[core.Prior, core.Posterior]:
    n_snps, n_pop = beta_hat.shape
    # quick way to calculate the inverse instead of using linalg.inv
    inv_shat2 = jnp.eye(n_pop) * (
        1 / jnp.diagonal(shat2, axis1=1, axis2=2)[:, jnp.newaxis]
    )

    prior_covar = priors.effect_covar[l_iter]
    # post_var = jnp.reciprocal(1 / shat2 + 1 / prior_var_b)  # A.1
    post_covar = jnp.linalg.inv(inv_shat2 + jnp.linalg.inv(priors.effect_covar[l_iter]))
    # post_mean = (post_var / shat2) * beta_hat  # A.2
    # we want to use shat2 to calculate rTZDinv
    # shat2 is pxkxk, use jnp.diagonal to make it pxk
    rTZDinv = beta_hat / jnp.diagonal(shat2, axis1=1, axis2=2)

    # rTZDinv is pxk, post_covar is pxkxk
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

    # third term for residual elbo (e.q. B.15)
    kl_alpha = utils._kl_categorical(alpha, priors.pi)
    kl_betas = alpha.T @ utils._kl_mvn(post_mean, post_covar, 0.0, prior_covar)

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


def _construct_optimize_v(mode: str = "em") -> core.AbstractOptFunc:

    if mode == "em":
        return EMOptFunc()
    elif mode == "noop":
        return NoopOptFunc()
    else:
        raise ValueError("invalid optimize mode")
