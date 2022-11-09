import logging
import typing

import jax.numpy as jnp
import jax.scipy.stats as stats
from jax import nn

import sushie
from sushie import core


def _construct_optimize_v(mode: str = "em") -> typing.Callable:
    if mode == "em":
        opt_fun = _optimize_v_em
    elif mode == "noop":
        opt_fun = _optimize_v_noop
    else:
        raise ValueError("invalid optimize mode")

    return opt_fun


def _optimize_v_em(
    betahat: jnp.ndarray,
    shat2: core.ArrayOrFloat,
    prior_var_b: core.ArrayOrFloat,
    prior_weights: jnp.ndarray,
) -> core.ArrayOrFloat:
    p1, n_pop = betahat.shape
    # quick way to calcualte the inverse instead of using linalg.inv
    inv_shat2 = jnp.eye(n_pop) * (
        1 / jnp.diagonal(shat2, axis1=1, axis2=2)[:, jnp.newaxis]
    )
    post_covar = jnp.linalg.inv(inv_shat2 + jnp.linalg.inv(prior_var_b))
    rTZDinv = betahat / jnp.diagonal(shat2, axis1=1, axis2=2)

    # if n_pop = 2
    # rTZDinv is px2, post_covar is px2x2
    # make rTZDinv px1x2 first and then use @, get px1x2 and then transpose get px2x1
    post_mean = jnp.transpose(rTZDinv[:, jnp.newaxis] @ post_covar, axes=(0, 2, 1))
    # reshape to get px2
    post_mean = post_mean.reshape(post_mean.shape[0:2])
    post_mean_sq = post_covar + jnp.einsum("ij,im->ijm", post_mean, post_mean)
    alpha = nn.softmax(
        jnp.log(prior_weights)
        - stats.multivariate_normal.logpdf(
            jnp.zeros((p1, n_pop)), post_mean, post_covar
        )
    )
    # post_mean_sq is px2x2, reshape alpha from px1 to 1xp
    new_prior_var_b = jnp.einsum(
        "ij,jmn->mn", alpha.reshape(1, len(alpha)), post_mean_sq
    )

    return new_prior_var_b


def _optimize_v_noop(
    betahat: jnp.ndarray,
    shat2: core.ArrayOrFloat,
    prior_var_b: core.ArrayOrFloat,
    prior_weights: jnp.ndarray,
) -> core.ArrayOrFloat:
    return prior_var_b


def run_sushie(
    Xs: typing.List[jnp.ndarray],
    ys: typing.List[jnp.ndarray],
    L: int,
    pi: jnp.ndarray = None,
    resid_var: jnp.ndarray = None,
    effect_var: jnp.ndarray = None,
    rho: jnp.ndarray = None,
    max_iter: int = 100,
    min_tol: float = 1e-3,
    opt_mode: str = "noop",
    threshold: float = 0.9,
    purity: float = 0.5,
) -> core.SushieResult:
    """
    Vanilla SuSiE model
    """
    # log = logging.getLogger(sushie.LOG)

    # todo: error checking on input
    # check len is same, p is same
    # L > 0, max_iter > 0, etc...

    n_pop = len(Xs)
    n1, p1 = Xs[0].shape

    # todo: make center/standardization an option
    for idx in range(n_pop):
        Xs[idx] = (Xs[idx] - jnp.mean(Xs[idx], axis=0)) / jnp.std(Xs[idx], axis=0)
        ys[idx] = jnp.squeeze(ys[idx])

    if resid_var is None:
        resid_var = []
        for idx in range(n_pop):
            resid_var.append(jnp.var(ys[idx], ddof=1))

    if effect_var is None:
        effect_var = [1e-3] * n_pop

    if rho is None:
        rho = [0.1] * n_pop

    effect_covar = jnp.diag(jnp.array(effect_var))
    ct = 0
    for row in range(1, n_pop):
        for col in range(n_pop):
            if col < row:
                _cov = jnp.sqrt(effect_var[row] * effect_var[col])
                effect_covar = effect_covar.at[row, col].set(rho[ct] * _cov)
                effect_covar = effect_covar.at[col, row].set(rho[ct] * _cov)
                ct += 1

    priors = core.Priors(
        pi=jnp.ones(p1) / float(p1) if pi is None else pi,
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
    priors: core.Priors,
    opt_v_func: typing.Callable,
    max_iter: int,
    min_tol: float,
    threshold: float,
    purity: float,
):
    log = logging.getLogger(sushie.LOG)

    n_pop = len(Xs)
    n1, p1 = Xs[0].shape

    n = []
    for idx in range(n_pop):
        n.append(Xs[idx].shape[0])

    params = core.SushieParams(
        alpha=jnp.zeros((p1, L)),
        b=jnp.zeros((L, p1, n_pop)),
        bsq=jnp.zeros((L, p1, n_pop, n_pop)),
    )

    # use lists for debugging, later on we'll drop to just scalar terms
    elbo = [-jnp.inf]
    for o_iter in range(max_iter):

        # internal loop
        params, elbo_score, priors = _update_effects(
            Xs,
            ys,
            params,
            priors,
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

        if o_iter == max_iter - 1:
            log.warning(
                f"Reach maximum iteration threshold {max_iter} after {o_iter + 1}."
            )

    if all(i <= j or jnp.isclose(i, j, atol=1e-8) for i, j in zip(elbo, elbo[1:])):
        log.info(f"Optimization finished. Final ELBO score: {elbo_score}.")
    else:
        raise ValueError(
            (
                f"ELBO does not non-decrease. Final ELBO score: {elbo_score}.",
                " Double check your genotype, phenotype, and covariate data.",
                " Contact developer if this error message remains.",
            )
        )

    pip = core.get_pip(params.alpha)
    cs = core.get_cs_sushie(
        params.alpha, Xs, threshold=threshold, purity_threshold=purity
    )

    sushie_res = core.SushieResult(
        params=params,
        priors=priors,
        pip=pip,
        cs=cs,
    )

    return sushie_res


def _update_effects(Xs, ys, params, priors, opt_v_func):
    # todo: documentation
    # try out JIT

    l_dim, p, n_pop = params.b.shape
    ns = [X.shape[0] for X in Xs]

    KL = jnp.zeros((l_dim,))
    residu = [0.0] * n_pop
    for idx in range(n_pop):
        residu[idx] = ys[idx] - Xs[idx] @ jnp.sum(params.b[:, idx], axis=0)

    r_l = [0.0] * n_pop
    for l_iter in range(l_dim):
        for idx in range(n_pop):
            r_l[idx] = residu[idx] + Xs[idx] @ params.b[l_iter, :, idx]

        res = single_shared_effect_regression(
            Xs, r_l, priors.prior_covar_b[l_iter], priors, opt_v_func
        )
        params = params._replace(
            alpha=params.alpha.at[:, l_iter].set(res.alpha),
            b=params.b.at[l_iter].set(res.post_mean * res.alpha.reshape(p, 1)),
            bsq=params.bsq.at[l_iter].set(
                res.post_mean_sq * res.alpha.reshape(p, 1, 1)
            ),
        )
        priors = priors._replace(
            prior_covar_b=priors.prior_covar_b.at[l_iter].set(res.prior_covar_b)
        )

        # third term for residual elbo (e.q. B.15)
        kl_alpha = core.kl_categorical(res.alpha, priors.pi)
        kl_b = res.alpha.T @ core.kl_multinormal(
            res.post_mean, res.post_covar, 0.0, res.prior_covar_b
        )
        KL = KL.at[l_iter].set(kl_alpha + kl_b)
        for idx in range(n_pop):
            residu[idx] = r_l[idx] - Xs[idx] @ params.b[l_iter, :, idx]

    erss_list = []
    exp_ll = 0.0
    tmpbsqs = jnp.transpose(jnp.einsum("nmij,ij->nmi", params.bsq, jnp.eye(n_pop)))
    for idx in range(n_pop):
        tmpb = jnp.transpose(params.b)[idx]
        # bsq is Lxpxkxk, where k is n_pop
        tmpbsq = tmpbsqs[idx]
        tmperss = core.erss(Xs[idx], ys[idx], tmpb, tmpbsq) / ns[idx]
        erss_list.append(tmperss)
        exp_ll += core.eloglike(Xs[idx], ys[idx], tmpb, tmpbsq, tmperss)

    priors = priors._replace(resid_var=jnp.array(erss_list))
    kl_divs = jnp.sum(KL)
    elbo_score = exp_ll - kl_divs

    return params, elbo_score, priors


def single_shared_effect_regression(
    X: typing.List[jnp.ndarray],
    y: typing.List[jnp.ndarray],
    prior_covar_b: jnp.ndarray,
    priors: core.Priors,
    optimize_v,
) -> core.SERResult:
    # note: for vanilla SuSiE XtX never changes.
    n_pop = len(X)
    n1, p1 = X[0].shape

    XtX = []
    Xty = []
    betahat = []
    s2e = []
    shat2 = []

    for idx in range(n_pop):
        XtX.append(jnp.sum(X[idx] ** 2, axis=0).reshape(p1, 1))
        Xty.append(X[idx].T @ y[idx].reshape((len(y[idx]), 1)))
        betahat.append(Xty[idx] / XtX[idx])
        s2e.append(priors.resid_var[idx])
        shat2.append(s2e[idx] / XtX[idx])

    betahat = jnp.array(betahat)
    betahat = jnp.transpose(betahat.reshape(betahat.shape[0:2]))
    tmp_shat2 = jnp.array(shat2)
    tmp_shat2 = jnp.transpose(tmp_shat2.reshape(tmp_shat2.shape[0:2]))

    shat2 = jnp.eye(n_pop) * (tmp_shat2[:, jnp.newaxis])
    inv_shat2 = jnp.eye(n_pop) * (1 / tmp_shat2[:, jnp.newaxis])
    prior_covar_b = optimize_v(betahat, shat2, prior_covar_b, priors.pi)

    # moments for beta given causal term
    # post_var = jnp.reciprocal(1 / shat2 + 1 / prior_var_b)  # A.1
    post_covar = jnp.linalg.inv(inv_shat2 + jnp.linalg.inv(prior_covar_b))

    # post_mean = (post_var / shat2) * betahat  # A.2
    # we want to use shat2 to calculate rTZDinv
    # shat2 is px2x2, use jnp.diagnoal to make it px2
    rTZDinv = betahat / jnp.diagonal(shat2, axis1=1, axis2=2)

    # rTZDinv is px2, post_covar is px2x2
    # make rTZDinv px1x2 first and then use @, get px1x2 and then transpose get px2x1
    post_mean = jnp.transpose(rTZDinv[:, jnp.newaxis] @ post_covar, axes=(0, 2, 1))
    # reshape to get px2
    post_mean = post_mean.reshape(post_mean.shape[0:2])

    post_mean_sq = post_covar + jnp.einsum("ij,im->ijm", post_mean, post_mean)

    alpha = nn.softmax(
        jnp.log(priors.pi)
        - stats.multivariate_normal.logpdf(
            jnp.zeros((p1, n_pop)), post_mean, post_covar
        )
    )

    res = core.SERResult(
        alpha=alpha,
        post_mean=post_mean,
        post_mean_sq=post_mean_sq,
        post_covar=post_covar,
        prior_covar_b=prior_covar_b,
    )

    return res