import logging
import typing
import math

import jax.numpy as jnp
import jax.scipy.stats as stats
from jax import nn, jit

import sushie
from sushie import core


def run_sushie(
        Xs: typing.List[jnp.ndarray],
        ys: typing.List[jnp.ndarray],
        L: int,
        norm_X: bool = True,
        norm_y: bool = False,
        pi: jnp.ndarray = None,
        resid_var: jnp.ndarray = None,
        effect_covar: jnp.ndarray = None,
        rho: jnp.ndarray = None,
        max_iter: int = 500,
        min_tol: float = 1e-5,
        opt_mode: str = "em",
        threshold: float = 0.9,
        purity: float = 0.5,
) -> core.SushieResult:
    log = logging.getLogger(sushie.LOG)

    # check if number of the ancestry are the same
    if len(Xs) == len(ys):
        n_pop = len(Xs)
        log.info(f"Detecting {n_pop} features for SuShiE fine-mapping.")
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
            (
                "The number of common SNPs across ancestries is less than inferred L.",
                "Please choose a smaller L or expand the genomic window.",
            )
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

    if effect_covar is None:
        effect_covar = [1e-3] * n_pop
    else:
        if len(effect_covar) != n_pop:
            raise ValueError(
                "Number of specified effect prior does not match ancestry number."
            )
        effect_covar = [float(i) for i in effect_covar]
        if jnp.any(jnp.array(effect_covar) <= 0):
            raise ValueError("The input of effect size prior is invalid (<0).")

    if rho is None:
        rho = [0.1] * n_pop
    else:
        exp_num_rho = math.comb(n_pop, 2)
        if len(rho) != exp_num_rho:
            raise ValueError(
                (
                    f"Number of specified rho ({len(rho)}) does not match expected",
                    f"number {exp_num_rho}.",
                )
            )
        rho = [float(i) for i in rho]
        # double-check the if it's invalid rho
        if jnp.any(jnp.abs(jnp.array(rho)) >= 1):
            raise ValueError(
                "The input of rho is invalid (>=1 or <=-1). Check your input."
            )

    if opt_mode == "noop":
        if resid_var is None or effect_covar is None or rho is None:
            raise ValueError("'noop' is specified. rho, resid_var, and effect_covar cannot be None.")

        log.warning(
            (
                "No updates on the effect size prior because of 'noop' setting for"
                " opt_mode. Inference may be inaccurate.",
            )
        )

    if pi is not None and (pi >= 1 or pi <= 0):
        raise ValueError(
            "Pi prior is not a probability (0-1). Specify a valid pi prior."
        )

    effect_covar = jnp.diag(jnp.array(effect_covar))
    ct = 0
    for row in range(1, n_pop):
        for col in range(n_pop):
            if col < row:
                _cov = jnp.sqrt(effect_covar[row] * effect_covar[col])
                effect_covar = effect_covar.at[row, col].set(rho[ct] * _cov)
                effect_covar = effect_covar.at[col, row].set(rho[ct] * _cov)
                ct += 1

    priors = core.Prior(
        pi=jnp.ones(n_snps) / float(n_snps) if pi is None else pi,
        resid_var=jnp.array(resid_var),
        # L x k x k
        effect_covar=jnp.array([effect_covar] * L),
    )

    opt_v_func = _construct_optimize_v(opt_mode)

    sushie_res = _inner_sushie(
        Xs,
        ys,
        L,
        priors,
        opt_v_func,
        max_iter,
        min_tol,
        threshold,
        purity,
    )

    return sushie_res


def _inner_sushie(
        Xs: typing.List[jnp.ndarray],
        ys: typing.List[jnp.ndarray],
        L: int,
        priors: core.Prior,
        opt_v_func: typing.Callable,
        max_iter: int,
        min_tol: float,
        threshold: float,
        purity: float,
) -> core.SushieResult:
    log = logging.getLogger(sushie.LOG)

    n_pop = len(Xs)
    _, n_snps = Xs[0].shape

    posteriors = core.Posterior(
        alpha=jnp.zeros((n_snps, L)),
        post_mean=jnp.zeros((L, n_snps, n_pop)),
        post_mean_sq=jnp.zeros((L, n_snps, n_pop, n_pop)),
        post_covar=jnp.zeros((L, n_snps, n_pop, n_pop)),
    )

    # use lists for debugging, later on we'll drop to just scalar terms
    elbo = [-jnp.inf]
    for o_iter in range(max_iter):
        priors, posteriors, elbo_score = _update_effects(
            Xs,
            ys,
            priors,
            posteriors,
            opt_v_func,
        )
        elbo.append(elbo_score)

        # print(f"eloglike  = {exp_ll} | kldiv = {kl_divs} | elbo: {elbo_score} |")
        if jnp.abs(elbo[o_iter + 1] - elbo[o_iter]) < min_tol:
            log.warning(
                (
                    f"Reach minimum tolerance threshold {min_tol} after {o_iter + 1}",
                    " iterations, stop optimization.",
                ),
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
            (
                f"ELBO does not non-decrease. Final ELBO score: {elbo[-1]}.",
                " Double check your genotype, phenotype, and covariate data.",
                " Contact developer if this error message remains.",
            )
        )

    pip = core.get_pip(posteriors.alpha)
    cs = core.get_cs(
        posteriors.alpha, Xs, threshold=threshold, purity=purity
    )

    sushie_res = core.SushieResult(
        priors=priors,
        posteriors=posteriors,
        pip=pip,
        cs=cs,
    )

    return sushie_res


@jit
def _update_effects(
        Xs: typing.List[jnp.ndarray],
        ys: typing.List[jnp.ndarray],
        priors: core.Prior,
        posteriors: core.Posterior,
        opt_v_func: object,
) -> typing.Tuple[core.Prior, core.Posterior, float]:
    l_dim, n_snps, n_pop = posteriors.post_mean.shape
    ns = [X.shape[0] for X in Xs]

    kl = jnp.zeros((l_dim,))
    residual = []
    # bv is Lxpxk, sum across all Ls
    post_mean_lsum = jnp.sum(posteriors.post_mean, axis=0)
    for idx in range(n_pop):
        residual.append(ys[idx] - Xs[idx] @ post_mean_lsum[:, idx])

    for l_iter in range(l_dim):
        residual_l = []
        for idx in range(n_pop):
            residual_l.append(residual[idx] + Xs[idx] @ posteriors.post_mean[l_iter, :, idx])

        res_posterior, res_prior_covar = single_shared_effect_regression(
            Xs, residual_l, priors.effect_covar[l_iter], priors, opt_v_func
        )
        posteriors: core.Posterior = posteriors._replace(
            alpha=posteriors.alpha.at[l_iter].set(res_posterior.alpha),
            post_mean=posteriors.post_mean.at[l_iter].set(
                res_posterior.post_mean * res_posterior.alpha[:, jnp.newaxis]
            ),
            post_mean_sq=posteriors.post_mean_sq.at[l_iter].set(
                res_posterior.post_mean_sq * res_posterior.alpha[:, jnp.newaxis, jnp.newaxis]
            ),
            post_covar=posteriors.post_covar.at[l_iter].set(
                jnp.einsum("j,jmn->mn", res_posterior.alpha, res_posterior.post_mean_sq)
            )
        )
        priors: core.Prior = priors._replace(
            effect_covar=priors.effect_covar.at[l_iter].set(res_prior_covar)
        )

        # third term for residual elbo (e.q. B.15)
        kl_alpha = core.kl_categorical(res_posterior.alpha, priors.pi)
        kl_b = res_posterior.alpha.T @ core.kl_mvn(
            res_posterior.post_mean, res_posterior.post_covar, 0.0, res_prior_covar
        )
        kl = kl.at[l_iter].set(kl_alpha + kl_b)
        for idx in range(n_pop):
            residual[idx] = residual_l[idx] - Xs[idx] @ posteriors.post_mean[l_iter, :, idx]

    erss_list = []
    exp_ll = 0.0
    tr_b_s = jnp.transpose(posteriors.post_mean)
    tr_bsq_s = jnp.transpose(jnp.einsum("nmij,ij->nmi", posteriors.post_mean_sq, jnp.eye(n_pop)))
    for idx in range(n_pop):
        tr_b = tr_b_s[idx]
        tr_bsq = tr_bsq_s[idx]
        tmp_erss = core.erss(Xs[idx], ys[idx], tr_b, tr_bsq) / ns[idx]
        erss_list.append(tmp_erss)
        exp_ll += core.eloglike(Xs[idx], ys[idx], tr_b, tr_bsq, tmp_erss)

    priors: core.Prior = priors._replace(resid_var=jnp.array(erss_list))
    kl_divs = jnp.sum(kl)
    elbo_score = exp_ll - kl_divs

    return priors, posteriors, elbo_score


@jit
def single_shared_effect_regression(
        Xs: typing.List[jnp.ndarray],
        ys: typing.List[jnp.ndarray],
        prior_covar_b: jnp.ndarray,
        priors: core.Prior,
        optimize_v,
) -> typing.Tuple[core.Posterior, jnp.ndarray]:
    n_pop = len(Xs)
    _, n_snps = Xs[0].shape

    # we eventually want beta hat to be pxk
    tr_beta_hat = jnp.zeros((n_pop, n_snps))
    shat2 = jnp.zeros((n_pop, n_snps))

    for idx in range(n_pop):
        XtX = jnp.sum(Xs[idx] ** 2, axis=0)[:, jnp.newaxis]
        Xty = Xs[idx].T @ ys[idx][:, jnp.newaxis]
        tr_beta_hat = tr_beta_hat.at[idx].set(Xty / XtX)
        shat2 = shat2.at[idx].set(priors.resid_var[idx] / XtX)

    beta_hat = jnp.transpose(tr_beta_hat)
    tr_shat2 = jnp.transpose(shat2)

    shat2 = jnp.eye(n_pop) * (tr_shat2[:, jnp.newaxis])
    update_covar_b: core.ArrayOrFloat = optimize_v(beta_hat, shat2, prior_covar_b, priors.pi)

    posterior_res: core.Posterior = _compute_posterior(beta_hat, shat2, update_covar_b, priors.pi)

    return posterior_res, update_covar_b


@jit
def _compute_posterior(
        beta_hat: jnp.ndarray,
        shat2: core.ArrayOrFloat,
        prior_covar_b: core.ArrayOrFloat,
        prior_weights: jnp.ndarray,
) -> core.Posterior:
    n_snps, n_pop = beta_hat.shape
    # quick way to calculate the inverse instead of using linalg.inv
    inv_shat2 = jnp.eye(n_pop) * (
            1 / jnp.diagonal(shat2, axis1=1, axis2=2)[:, jnp.newaxis]
    )

    # post_var = jnp.reciprocal(1 / shat2 + 1 / prior_var_b)  # A.1
    post_covar = jnp.linalg.inv(inv_shat2 + jnp.linalg.inv(prior_covar_b))
    # post_mean = (post_var / shat2) * beta_hat  # A.2
    # we want to use shat2 to calculate rTZDinv
    # shat2 is pxkxk, use jnp.diagonal to make it pxk
    rTZDinv = beta_hat / jnp.diagonal(shat2, axis1=1, axis2=2)

    # rTZDinv is pxk, post_covar is pxkxk
    post_mean = jnp.einsum("ijk,ik->ij", post_covar, rTZDinv)
    post_mean_sq = post_covar + jnp.einsum("ij,im->ijm", post_mean, post_mean)
    alpha = nn.softmax(
        jnp.log(prior_weights)
        - stats.multivariate_normal.logpdf(
            jnp.zeros((n_snps, n_pop)), post_mean, post_covar
        )
    )

    res = core.Posterior(
        alpha=alpha,
        post_mean=post_mean,
        post_mean_sq=post_mean_sq,
        post_covar=post_covar,
    )

    return res


def _optimize_v_em(
        beta_hat: jnp.ndarray,
        shat2: core.ArrayOrFloat,
        prior_covar_b: core.ArrayOrFloat,
        prior_weights: jnp.ndarray,
) -> core.ArrayOrFloat:
    res = _compute_posterior(beta_hat, shat2, prior_covar_b, prior_weights)

    new_prior_var_b = jnp.einsum(
        "j,jmn->mn", res.alpha, res.post_mean_sq
    )

    return new_prior_var_b


def _optimize_v_noop(
        beta_hat: jnp.ndarray,
        shat2: core.ArrayOrFloat,
        prior_covar_b: core.ArrayOrFloat,
        prior_weights: jnp.ndarray,
) -> core.ArrayOrFloat:
    return prior_covar_b


def _construct_optimize_v(mode: str = "em") -> typing.Callable:
    if mode == "em":
        opt_fun = _optimize_v_em
    elif mode == "noop":
        opt_fun = _optimize_v_noop
    else:
        raise ValueError("invalid optimize mode")

    return opt_fun
