import math
from typing import NamedTuple, Tuple

import equinox as eqx

import jax.numpy as jnp
from jax import Array, lax
from jax.typing import ArrayLike

from . import infer, log, utils

__all__ = [
    "infer_sushie_ss",
]


class _LResult_ss(NamedTuple):
    Xtys: Array
    XtXs: Array
    priors: infer.Prior
    posteriors: infer.Posterior
    prior_adjustor: infer._PriorAdjustor
    opt_v_func: infer._AbstractOptFunc


def infer_sushie_ss(
    lds: ArrayLike,
    ns: ArrayLike,
    zs: ArrayLike,
    L: int = 10,
    no_update: bool = False,
    pi: ArrayLike = None,
    resid_var: utils.ListFloatOrNone = None,
    effect_var: utils.ListFloatOrNone = None,
    rho: utils.ListFloatOrNone = None,
    max_iter: int = 500,
    min_tol: float = 1e-4,
    threshold: float = 0.95,
    purity: float = 0.5,
    purity_method: str = "weighted",
    max_select: int = 250,
    min_snps: int = 100,
    no_reorder: bool = False,
    seed: int = 12345,
) -> infer.SushieResult:
    """The main inference function for running SuShiE.

    Args:
        lds: LD matrix for multiple ancestries.
        zs: molQTL scan z scores for multiple ancestries.
        ns: Sample size for each ancestry.
        L: Inferred number of eQTLs for the gene.
        no_update: Do not update the effect size prior. Default is to update.
        pi: The probability prior for one SNP to be causal (:math:`\\pi` in :ref:`Model`). Default is :math:`1` over
            the number of SNPs by specifying it as ``None``.
        resid_var: Prior residual variance (:math:`\\sigma^2_e` in :ref:`Model`).
            Default is :math:`0.001` by specifying it as ``None``.
        effect_var: Prior causal effect size variance (:math:`\\sigma^2_{i,b}` in :ref:`Model`).
            Default is :math:`0.001` by specifying it as ``None``.
        rho: Prior effect size correlation (:math:`\\rho` in :ref:`Model`).
            Default is :math:`0.1` by specifying it as ``None``.
        max_iter: The maximum iteration for optimization. Default is :math:`500`.
        min_tol: The convergence tolerance. Default is :math:`10^{-4}`.
        threshold: The credible set threshold. Default is :math:`0.95`.
        purity: The minimum pairwise correlation across SNPs to be eligible as output credible set.
            Default is :math:`0.5`.
        purity_method: The method to compute purity across ancestries. Default is ``weighted``.
        max_select: The maximum number of selected SNPs to compute purity. Default is :math:`250`.
        min_snps: The minimum number of SNPs to fine-map. Default is :math:`100`.
        no_reorder: Do not re-order single effects based on Frobenius norm of effect size covariance prior.
            Default is to re-order.
        seed: The randomization seed for selecting SNPs in the credible set to compute purity. Default is :math:`12345`.

    Returns:
        :py:obj:`SushieResult`: A SuShiE result object that contains prior (:py:obj:`Prior`),
        posterior (:py:obj:`Posterior`), ``cs``, ``pip``, ``elbo``, and ``elbo_increase``.

    """
    n_pop = ns.shape[0]

    if len(lds) != n_pop:
        raise ValueError(
            f"The number of LD matrices ({len(lds)}) does not match the number of ancestries ({n_pop})."
        )

    if not all(ld.shape == lds[0].shape for ld in lds):
        raise ValueError("LD matrices do not have the same shape. Check your input.")

    if lds[0].shape[0] != lds[0].shape[1]:
        raise ValueError("LD matrices are not square matrices. Check your input.")

    if zs is None:
        raise ValueError("Z scores are not provided. Check your input.")
    else:
        if len(zs) != n_pop:
            raise ValueError(
                f"The number of Z scores ({len(zs)}) does not match the number of ancestries ({n_pop})."
            )

        if not all(z.shape == zs[0].shape for z in zs):
            raise ValueError(
                "Z scores across ancestries do not have the same shape. Check your input."
            )

        if zs[0].shape[0] != lds[0].shape[0]:
            raise ValueError(
                "Z scores do not have the same number of SNPs as the LD matrices. Check your input."
            )

    if L <= 0:
        raise ValueError(f"Inferred L ({L}) is invalid, choose a positive L.")

    if min_tol > 0.1:
        log.logger.warning(
            f"Minimum intolerance ({min_tol}) is greater than 0.1. Inference may not be accurate."
        )

    if not 0 < threshold < 1:
        raise ValueError(
            f"Credible set PIP threshold ({threshold}) must be greater than 0 and less than 1."
            + " Specify a valid value using '--threshold' for command-line usage"
            + " or 'threshold=' for in-Python function calls."
        )

    if not 0 < purity < 1:
        raise ValueError(
            f"Purity threshold ({purity}) must be greater than or equal to 0 and less than 1. "
            + " Specify a valid value using '--purity' for command-line usage"
            + " or 'purity=' for in-Python function calls."
        )

    if max_select <= 0:
        raise ValueError(
            "The maximum selected number of SNPs for purity must be greater than 0."
            + " Specify a valid value using '--max-select' for command-line usage"
            + " or 'max_select=' for in-Python function calls."
        )

    if min_snps <= 0:
        raise ValueError(
            "The minimum number of SNPs to fine-map is invalid. Choose a positive integer."
        )

    n_snps = lds[0].shape[0]

    if pi is None:
        pi = jnp.ones(n_snps) / float(n_snps)
    else:
        if not (pi > 0).all():
            raise ValueError("Prior probability/weights must be all positive values.")

        if pi.shape[0] != lds[0].shape[1]:
            raise ValueError(
                f"Prior probability/weights ({pi.shape[0]}) does not match the number of SNPs ({lds[0].shape[1]})."
            )

        if jnp.sum(pi) != 1:
            log.logger.debug(
                "Prior probability/weights sum is not equal to 1. Will normalize to sum to 1."
            )
            pi = pi.astype(float) / jnp.sum(pi)

    if resid_var is None:
        resid_var = []
        for idx in range(n_pop):
            resid_var.append(1)
    else:
        if len(resid_var) != n_pop:
            raise ValueError(
                f"Number of specified residual prior ({len(resid_var)}) does not match ancestry number ({n_pop})."
            )
        resid_var = [float(i) for i in resid_var]
        if jnp.any(jnp.array(resid_var) <= 0):
            raise ValueError(
                f"The input of residual prior ({resid_var}) is invalid (<0). Check your input."
            )

    if min_snps < L:
        raise ValueError(
            f"The number of minimum common SNPs across ancestries ({min_snps}) is less than inferred L ({L})."
            + " Specify a larger value using '--min-snps' for command-line usage"
            + " or 'min_snps=' for in-Python function calls."
        )

    if n_snps < min_snps:
        raise ValueError(
            f"The number of common SNPs across ancestries ({n_snps}) is less than minimum common"
            + " number of SNPs (100) specified."
            + " Users can specify a smaller value using '--min-snps' for command-line usage"
            + " or 'min_snps=' for in-Python function calls."
        )

    param_effect_var = effect_var
    if effect_var is None:
        effect_var = [1e-3] * n_pop
    else:
        if len(effect_var) != n_pop:
            raise ValueError(
                f"Number of specified effect prior ({len(effect_var)}) does not match ancestry number ({n_pop})."
            )
        effect_var = [float(i) for i in effect_var]
        if jnp.any(jnp.array(effect_var) <= 0):
            raise ValueError(
                f"The effect size prior variance ({effect_var}) must be positive."
            )

    exp_num_rho = math.comb(n_pop, 2)
    param_rho = rho
    if rho is None:
        rho = [0.1] * exp_num_rho
    else:
        if n_pop == 1:
            log.logger.debug(
                "Running single-ancestry SuShiE. The '--rho' parameter is specified but will be ignored."
            )

        if (len(rho) != exp_num_rho) and n_pop != 1:
            raise ValueError(
                f"Number of specified rho ({len(rho)}) does not match expected"
                + f" number {exp_num_rho}.",
            )
        rho = [float(i) for i in rho]
        # double-check the if it's invalid rho
        if jnp.any(jnp.abs(jnp.array(rho)) > 1):
            raise ValueError(
                f"Effect size prior correlation ({rho}) must be between -1 and 1 (inclusive)."
            )

    effect_covar = jnp.diag(jnp.array(effect_var))
    ct = 0
    for col in range(n_pop):
        for row in range(1, n_pop):
            if col < row:
                _two_sd = jnp.sqrt(effect_var[row] * effect_var[col])
                effect_covar = effect_covar.at[row, col].set(rho[ct] * _two_sd)
                effect_covar = effect_covar.at[col, row].set(rho[ct] * _two_sd)
                ct += 1

    if no_update:
        # if we specify no_update and rho, we want to keep rho through iterations and update variance
        if param_effect_var is None and param_rho is not None and n_pop != 1:
            prior_adjustor = infer._PriorAdjustor(
                times=jnp.eye(n_pop),
                plus=effect_covar - jnp.diag(jnp.diag(effect_covar)),
            )

            log.logger.info(
                "No updates on the prior effect correlation rho while updating prior effect variance."
            )
        # if we specify no_update and effect_covar, we want to keep variance through iterations, and update rho
        elif param_effect_var is not None and param_rho is None and n_pop != 1:
            prior_adjustor = infer._PriorAdjustor(
                times=jnp.ones((n_pop, n_pop)) - jnp.eye(n_pop),
                plus=effect_covar * jnp.eye(n_pop),
            )
            log.logger.info(
                "No updates on the prior effect variance while updating prior effect correlation rho."
            )
        # if we (do not specify effect_covar and rho) or (specify both effect_covar and rho)
        # nothing is updated through iterations
        else:
            prior_adjustor = infer._PriorAdjustor(
                times=jnp.zeros((n_pop, n_pop)), plus=effect_covar
            )
            log.logger.info(
                "No updates on the prior effect size variance/covariance matrix."
            )
    else:
        prior_adjustor = infer._PriorAdjustor(
            times=jnp.ones((n_pop, n_pop)), plus=jnp.zeros((n_pop, n_pop))
        )

    # define:
    # k is ancestry
    # n is sample size
    # p is SNP
    # l is the number of effects

    priors = infer.Prior(
        # p x 1
        pi=pi,
        # k x 1
        resid_var=jnp.array(resid_var)[:, jnp.newaxis],
        # l x k x k
        effect_covar=jnp.array([effect_covar] * L),
    )

    posteriors = infer.Posterior(
        # l x p
        alpha=jnp.zeros((L, n_snps)),
        # l x p x k
        post_mean=jnp.zeros((L, n_snps, n_pop)),
        # l x p x k x k
        post_mean_sq=jnp.zeros((L, n_snps, n_pop, n_pop)),
        # l x n x n
        weighted_sum_covar=jnp.zeros((L, n_pop, n_pop)),
        kl=jnp.zeros((L,)),
    )

    # since we use prior adjustor, this is really no need
    # opt_v_func = NoopOptFunc() would work
    opt_v_func = infer._EMOptFunc() if not no_update else infer._NoopOptFunc()

    # get XtXs and Xtys
    zs = jnp.array(zs)
    lds = jnp.array(lds)
    sigma2 = ns / (ns + zs ** 2)
    Xtys = jnp.sqrt(ns) * jnp.sqrt(sigma2) * zs
    XtXs = ns[:, :, jnp.newaxis] * lds

    elbo_tracker = jnp.array([-jnp.inf])
    elbo_increase = True
    for o_iter in range(max_iter):
        log.logger.debug(f"Starting optimization iteration {o_iter + 1}.")
        prev_priors = priors
        prev_posteriors = posteriors

        priors, posteriors, elbo_cur = _update_effects_ss(
            Xtys,
            XtXs,
            ns,
            priors,
            posteriors,
            prior_adjustor,
            opt_v_func,
        )
        elbo_last = elbo_tracker[o_iter]
        elbo_tracker = jnp.append(elbo_tracker, elbo_cur)
        elbo_increase = elbo_cur >= elbo_last or jnp.isclose(
            elbo_cur, elbo_last, atol=1e-8
        )

        log.logger.debug(f"Iteration {o_iter + 1} finished.")

        if not elbo_increase:
            log.logger.warning(
                f"Optimization finished after {o_iter + 1} iterations."
                + f" ELBO decreased. Final ELBO score: {elbo_cur}. Return last iteration's results."
                + " It can be precision issue,"
                + " and adding 'import jax; jax.config.update('jax_enable_x64', True)' may fix it."
                + " If this issue keeps rising for many genes, contact the developers."
            )
            priors = prev_priors
            posteriors = prev_posteriors
            break

        decimal_digit = len(str(min_tol)) - str(min_tol).find(".") - 1

        if jnp.abs(elbo_cur - elbo_last) < min_tol:
            log.logger.info(
                f"Optimization concludes after {o_iter + 1} iterations. Final ELBO score: {elbo_cur:.{decimal_digit}f}."
                + f" Reach minimum tolerance threshold {min_tol}.",
            )
            break

        if o_iter + 1 == max_iter:
            log.logger.info(
                f"Optimization concludes after {o_iter + 1} iterations. Final ELBO score: {elbo_cur:.{decimal_digit}f}."
                + f" Reach maximum iteration threshold {max_iter}.",
            )

    l_order = jnp.arange(L)
    if not no_reorder:
        log.logger.debug(
            "Reordering effects based on Frobenius norm of effect size covariance prior."
        )
        priors, posteriors, l_order = infer._reorder_l(priors, posteriors)

    log.logger.debug("Computing credible sets.")

    cs, full_alphas, pip_all, pip_cs = infer.make_cs(
        posteriors.alpha,
        ns,
        None,
        lds,
        threshold,
        purity,
        purity_method,
        max_select,
        seed,
    )

    log.logger.debug(
        "Inference and credible set computation complete. Beginning to write results."
    )

    return infer.SushieResult(
        priors,
        posteriors,
        pip_all,
        pip_cs,
        cs,
        full_alphas,
        ns,
        elbo_tracker,
        elbo_increase,
        l_order,
    )


@eqx.filter_jit
def _update_effects_ss(
    Xtys: ArrayLike,
    XtXs: ArrayLike,
    ns: ArrayLike,
    priors: infer.Prior,
    posteriors: infer.Posterior,
    prior_adjustor: infer._PriorAdjustor,
    opt_v_func: infer._AbstractOptFunc,
) -> Tuple[infer.Prior, infer.Posterior, Array]:
    l_dim, n_snps, n_pop = posteriors.post_mean.shape

    # reduce from lxpxk to pxk
    post_mean_lsum = jnp.sum(posteriors.post_mean, axis=0)

    residual = Xtys - jnp.einsum("kpq,kq->kp", XtXs, post_mean_lsum.T)

    init_l_result = _LResult_ss(
        Xtys=residual,
        XtXs=XtXs,
        priors=priors,
        posteriors=posteriors,
        prior_adjustor=prior_adjustor,
        opt_v_func=opt_v_func,
    )

    l_result = lax.fori_loop(0, l_dim, _update_l, init_l_result)

    _, _, priors, posteriors, _, _ = l_result

    # from lxpxk to kxpxl
    tr_b_s = posteriors.post_mean.T
    # from lxpxkxk to lxpxk (get the diagonal), then become kxpxl
    tr_bsq_s = jnp.diagonal(posteriors.post_mean_sq, axis1=2, axis2=3).T

    exp_ll = jnp.sum(_eloglike_ss(Xtys, XtXs, ns, tr_b_s, tr_bsq_s, priors.resid_var))
    sigma2 = _erss_ss(Xtys, XtXs, ns, tr_b_s, tr_bsq_s)[:, jnp.newaxis] / ns
    kl_divs = jnp.sum(posteriors.kl)
    elbo_score = exp_ll - kl_divs
    priors = priors._replace(resid_var=sigma2)

    return priors, posteriors, elbo_score


def _update_l(l_iter: int, param: _LResult_ss) -> _LResult_ss:
    residual, XtXs, priors, posteriors, prior_adjustor, opt_v_func = param

    residual_l = residual + jnp.einsum(
        "kpq,kq->kp", XtXs, posteriors.post_mean[l_iter].T
    )

    priors, posteriors = _ssr_ss(
        residual_l,
        XtXs,
        priors,
        posteriors,
        prior_adjustor,
        l_iter,
        opt_v_func,
    )

    residual = residual_l - jnp.einsum(
        "kpq,kq->kp", XtXs, posteriors.post_mean[l_iter].T
    )

    update_param = param._replace(
        Xtys=residual,
        priors=priors,
        posteriors=posteriors,
    )

    return update_param


def _ssr_ss(
    Xtys: ArrayLike,
    XtXs: ArrayLike,
    priors: infer.Prior,
    posteriors: infer.Posterior,
    prior_adjustor: infer._PriorAdjustor,
    l_iter: int,
    opt_v_func: infer._AbstractOptFunc,
) -> Tuple[infer.Prior, infer.Posterior]:

    n_pop, n_snps = Xtys.shape
    # Xtys is kxp, priors.resid_var is kx1, and the result is kxp, transpose is pxk
    rTZDinv = (Xtys / priors.resid_var).T

    # priors.resid_var is kx1, XtXs is kxpxp, and the result is kxp, and inverse is pxk
    inv_shat2 = (jnp.diagonal(XtXs, axis1=1, axis2=2) / priors.resid_var).T

    # expand it to diag matrix, so they're pxkxk
    inv_shat2 = jnp.eye(n_pop) * inv_shat2[:, jnp.newaxis]

    priors = opt_v_func(rTZDinv, inv_shat2, priors, posteriors, prior_adjustor, l_iter)

    _, posteriors = infer._compute_posterior(
        rTZDinv, inv_shat2, priors, posteriors, l_iter
    )

    return priors, posteriors


def _eloglike_ss(
    Xtys: ArrayLike,
    XtXs: ArrayLike,
    ns: ArrayLike,
    beta: ArrayLike,
    beta_sq: ArrayLike,
    sigma_sq: ArrayLike,
) -> Array:
    norm_term = -(0.5 * ns) * jnp.log(2 * jnp.pi * sigma_sq)
    quad_term = (
        -(0.5 / sigma_sq) * _erss_ss(Xtys, XtXs, ns, beta, beta_sq)[:, jnp.newaxis]
    )
    return norm_term + quad_term


def _erss_ss(
    Xtys: ArrayLike, XtXs: ArrayLike, ns: ArrayLike, beta: ArrayLike, beta_sq: ArrayLike
) -> Array:

    # beta is kxpxl
    ebeta = jnp.sum(beta, axis=2)
    # N, ns is k by 1
    term_1 = jnp.squeeze(ns)
    # -2 * E(beta)(Xty)
    # ebeta is kxp, and Xtys is kxp, multiply them still kxp, and sum over p, so it's k
    term_2 = -2 * jnp.sum(ebeta * Xtys, axis=1)
    # eb * XtXs * eb
    term_3 = jnp.einsum("kp,kpq,kq->k", ebeta, XtXs, ebeta)
    # beta was kxpxl, convert it to kxlxp
    tr_beta = jnp.transpose(beta, axes=(0, 2, 1))
    tr_beta_sq = jnp.transpose(beta_sq, axes=(0, 2, 1))

    term_4 = -1 * jnp.einsum("klp,kpq,klq->k", tr_beta, XtXs, tr_beta)
    term_5 = jnp.einsum("kp,klp->k", jnp.diagonal(XtXs, axis1=1, axis2=2), tr_beta_sq)

    return term_1 + term_2 + term_3 + term_4 + term_5
