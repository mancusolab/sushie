import math
from abc import ABCMeta, abstractmethod
from typing import List, NamedTuple, Tuple

import equinox as eqx
import pandas as pd

import jax.numpy as jnp
import jax.scipy.stats as stats
from jax import Array, lax, nn, random
from jax.typing import ArrayLike

from . import log, utils

__all__ = [
    "Prior",
    "Posterior",
    "SushieResult",
    "infer_sushie",
    "make_cs",
    "_PriorAdjustor",
    "_AbstractOptFunc",
    "_NoopOptFunc",
    "_EMOptFunc",
    "_compute_posterior",
    "_reorder_l",
]


class Prior(NamedTuple):
    """Define the class for the prior parameter of SuShiE model.

    Attributes:
        pi: The prior probability for one SNP to be causal.
        resid_var: The prior residual variance for all SNPs.
        effect_covar: The prior effect sizes covariance matrix for all SNPs.

    """

    pi: Array
    resid_var: Array
    effect_covar: Array


class Posterior(NamedTuple):
    """Define the class for the posterior parameter of SuShiE model.

    Attributes:
        alpha: Posterior probability for SNP to be causal (i.e., :math:`\\alpha` in :ref:`Model`; :math:`L \\times p`).
        post_mean: The alpha-weighted posterior mean for each SNP (:math:`L \\times p \\times k`).
        post_mean_sq: The alpha-weighted posterior mean square for each SNP (:math:`L \\times p \\times k \\times k`
            , a diagonal matrix for :math:`k \\times k`).
        weighted_sum_covar: The alpha-weighted sum of posterior effect covariance across SNPs
            (:math:`L \\times k \\times k`).
        kl: The Kullback–Leibler (KL) divergence for each :math:`L`.
        log_bf: The log Bayes factor for each SNP (:math:`L \\times p`).
    """

    alpha: Array
    post_mean: Array
    post_mean_sq: Array
    weighted_sum_covar: Array
    kl: Array
    log_bf: Array


class SushieResult(NamedTuple):
    """Define the class for the SuShiE inference results.

    Attributes:
        priors: The final prior parameter for the inference.
        posteriors: The final posterior parameter for the inference.
        pip_all: The PIP for each SNP across :math:`L` credible sets.
        pip_cs:  The PIP across credible sets that are not pruned.
        cs: The credible sets output after filtering on purity.
        alphas: The full credible sets before filtering on purity.
        sample_size: The sample size for each ancestry in the inference.
        elbo: The final ELBO.
        elbo_increase: A boolean to indicate whether ELBO increases during the optimizations.
        l_order: The original order that SuShiE infers. For example, if L=3 and it is 0,2,1,
            then the original SuShiE's second effect (0-based index 1) is now third,
            and the original SuShiE's third effect (0-based index 2) is now second
            after sorting use Frobenius norm.

    """

    priors: Prior
    posteriors: Posterior
    pip_all: Array
    pip_cs: Array
    cs: pd.DataFrame
    alphas: pd.DataFrame
    sample_size: Array
    elbo: Array
    elbo_increase: bool
    l_order: Array


class _PriorAdjustor(NamedTuple):
    times: Array
    plus: Array


class _AbstractOptFunc(eqx.Module, metaclass=ABCMeta):
    @abstractmethod
    def __call__(
        self,
        rTZDinv: ArrayLike,
        inv_shat2: ArrayLike,
        priors: Prior,
        posteriors: Posterior,
        prior_adjustor: _PriorAdjustor,
        l_iter: int,
    ) -> Prior:
        pass


class _LResult(NamedTuple):
    Xs: Array
    ys: Array
    XtXs: Array
    priors: Prior
    posteriors: Posterior
    prior_adjustor: _PriorAdjustor
    opt_v_func: _AbstractOptFunc


class _EMOptFunc(_AbstractOptFunc):
    def __call__(
        self,
        rTZDinv: ArrayLike,
        inv_shat2: ArrayLike,
        priors: Prior,
        posteriors: Posterior,
        prior_adjustor: _PriorAdjustor,
        l_iter: int,
    ) -> Prior:
        priors, _ = _compute_posterior(rTZDinv, inv_shat2, priors, posteriors, l_iter)

        return priors


class _NoopOptFunc(_AbstractOptFunc):
    def __call__(
        self,
        rTZDinv: ArrayLike,
        inv_shat2: ArrayLike,
        priors: Prior,
        posteriors: Posterior,
        prior_adjustor: _PriorAdjustor,
        l_iter: int,
    ) -> Prior:
        priors, _ = _compute_posterior(rTZDinv, inv_shat2, priors, posteriors, l_iter)
        priors = priors._replace(
            effect_covar=priors.effect_covar.at[l_iter].set(
                priors.effect_covar[l_iter] * prior_adjustor.times + prior_adjustor.plus
            )
        )
        return priors


def infer_sushie(
    Xs: List[ArrayLike],
    ys: List[ArrayLike],
    covar: utils.ListArrayOrNone = None,
    L: int = 10,
    no_scale: bool = False,
    no_regress: bool = False,
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
) -> SushieResult:
    """The main inference function for running SuShiE.

    Args:
        Xs: Genotype data for multiple ancestries.
        ys: Phenotype data for multiple ancestries.
        covar: Covariate data for multiple ancestries.
        L: Inferred number of eQTLs for the gene.
        no_scale: Do not scale the genotype and phenotype. Default is to scale.
        no_regress: Do not regress covariates on genotypes. Default is to regress.
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

    Example:
        Basic usage with two-ancestry data::

            import numpy as np
            from sushie.infer import infer_sushie

            # Generate example data for 2 ancestries
            # Ancestry 1: 100 samples, 500 SNPs
            X1 = np.random.randn(100, 500)
            y1 = np.random.randn(100)

            # Ancestry 2: 150 samples, 500 SNPs
            X2 = np.random.randn(150, 500)
            y2 = np.random.randn(150)

            # Run SuShiE fine-mapping
            result = infer_sushie(Xs=[X1, X2], ys=[y1, y2], L=5)

            # Access results
            print(result.pip)       # Posterior inclusion probabilities
            print(result.cs)        # Credible sets

    """
    if len(Xs) == len(ys):
        n_pop = len(Xs)
    else:
        raise ValueError(
            f"The number of geno ({len(Xs)}) and pheno ({len(ys)}) data does not match. Check your input."
        )

    # check x and y have the same sample size
    for idx in range(n_pop):
        if Xs[idx].shape[0] != ys[idx].shape[0]:
            raise ValueError(
                f"Ancestry {idx + 1}: The sample size of geno ({Xs[idx].shape[0]}) "
                + f"and pheno ({ys[idx].shape[0]}) data does not match. Check your input."
            )

    # check each ancestry has the same number of SNPs
    for idx in range(1, n_pop):
        if Xs[idx - 1].shape[1] != Xs[idx].shape[1]:
            raise ValueError(
                f"Ancestry {idx} and ancestry {idx} do not have "
                + f"the same number of SNPs ({Xs[idx - 1].shape[1]} vs {Xs[idx].shape[1]})."
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
            "The minimum number of SNPs to fine-map must be greater than 0."
            + " Specify a valid value using '--min-snps' for command-line usage"
            + " or 'min_snps=' for in-Python function calls."
        )

    _, n_snps = Xs[0].shape

    if pi is None:
        pi = jnp.ones(n_snps) / float(n_snps)
    else:
        if not (pi > 0).all():
            raise ValueError("Prior probability/weights must be all positive values.")

        if pi.shape[0] != Xs[0].shape[1]:
            raise ValueError(
                f"Prior probability/weights ({pi.shape[0]}) does not match the number of SNPs ({Xs[0].shape[1]})."
            )

        if jnp.sum(pi) > 1:
            log.logger.debug(
                "Prior probability/weights sum is not equal to 1. Will normalize to sum to 1."
            )
            pi = pi.astype(float) / jnp.sum(pi)

    # first regress out covariates if there are any, then scale the genotype and phenotype
    if covar is not None:
        for idx in range(n_pop):
            Xs[idx], ys[idx] = utils.regress_covar(
                Xs[idx], ys[idx], covar[idx], no_regress
            )

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
        if jnp.any(jnp.abs(jnp.array(rho)) >= 1):
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
            prior_adjustor = _PriorAdjustor(
                times=jnp.eye(n_pop),
                plus=effect_covar - jnp.diag(jnp.diag(effect_covar)),
            )

            log.logger.info(
                "No updates on the prior effect correlation rho while updating prior effect variance."
            )
        # if we specify no_update and effect_covar, we want to keep variance through iterations, and update rho
        elif param_effect_var is not None and param_rho is None and n_pop != 1:
            prior_adjustor = _PriorAdjustor(
                times=jnp.ones((n_pop, n_pop)) - jnp.eye(n_pop),
                plus=effect_covar * jnp.eye(n_pop),
            )
            log.logger.info(
                "No updates on the prior effect variance while updating prior effect correlation rho."
            )
        # if we (do not specify effect_covar and rho) or (specify both effect_covar and rho)
        # nothing is updated through iterations
        else:
            prior_adjustor = _PriorAdjustor(
                times=jnp.zeros((n_pop, n_pop)), plus=effect_covar
            )
            log.logger.info(
                "No updates on the prior effect size variance/covariance matrix."
            )
    else:
        prior_adjustor = _PriorAdjustor(
            times=jnp.ones((n_pop, n_pop)), plus=jnp.zeros((n_pop, n_pop))
        )

    # define:
    # k is ancestry
    # n is sample size
    # p is SNP
    # l is the number of effects

    priors = Prior(
        # p x 1
        pi=pi,
        # k x 1
        resid_var=jnp.array(resid_var)[:, jnp.newaxis],
        # l x k x k
        effect_covar=jnp.array([effect_covar] * L),
    )

    posteriors = Posterior(
        # l x p
        alpha=jnp.zeros((L, n_snps)),
        # l x p x k
        post_mean=jnp.zeros((L, n_snps, n_pop)),
        # l x p x k x k
        post_mean_sq=jnp.zeros((L, n_snps, n_pop, n_pop)),
        # l x n x n
        weighted_sum_covar=jnp.zeros((L, n_pop, n_pop)),
        # l
        kl=jnp.zeros((L,)),
        # l x p
        log_bf=jnp.zeros((L, n_snps)),
    )

    # since we use prior adjustor, this is really no need
    # opt_v_func = NoopOptFunc() would work
    opt_v_func = _EMOptFunc() if not no_update else _NoopOptFunc()

    # padding
    ns = jnp.array([X.shape[0] for X in Xs])[:, jnp.newaxis]
    p_max = jnp.max(ns)
    for idx in range(n_pop):
        # ((a,b), (c,d)) where a means top, b means bottom, c means left, and d means right
        Xs[idx] = jnp.pad(
            Xs[idx], ((0, p_max - jnp.squeeze(ns[idx])), (0, 0)), "constant"
        )
        ys[idx] = jnp.pad(ys[idx], (0, p_max - jnp.squeeze(ns[idx])), "constant")
    # k x n x p
    Xs = jnp.array(Xs)
    # k x n
    ys = jnp.array(ys)
    XtXs = jnp.sum(Xs ** 2, axis=1)

    elbo_tracker = jnp.array([-jnp.inf])
    elbo_increase = True
    for o_iter in range(max_iter):
        log.logger.debug(f"Starting optimization iteration {o_iter + 1}.")

        prev_priors = priors
        prev_posteriors = posteriors

        priors, posteriors, elbo_cur = _update_effects(
            Xs,
            ys,
            XtXs,
            ns,
            priors,
            posteriors,
            prior_adjustor,
            opt_v_func,
        )

        log.logger.debug(f"Iteration {o_iter + 1} finished.")

        elbo_last = elbo_tracker[o_iter]
        elbo_tracker = jnp.append(elbo_tracker, elbo_cur)
        elbo_increase = elbo_cur >= elbo_last or jnp.isclose(
            elbo_cur, elbo_last, atol=1e-8
        )

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
        priors, posteriors, l_order = _reorder_l(priors, posteriors)

    log.logger.debug("Computing credible sets.")

    cs, full_alphas, pip_all, pip_cs = make_cs(
        posteriors.alpha,
        posteriors.log_bf,
        ns,
        Xs,
        None,
        threshold,
        purity,
        purity_method,
        max_select,
        seed,
    )

    log.logger.debug(
        "Inference and credible set computation complete. Beginning to write results."
    )

    return SushieResult(
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
def _update_effects(
    Xs: ArrayLike,
    ys: ArrayLike,
    XtXs: ArrayLike,
    ns: ArrayLike,
    priors: Prior,
    posteriors: Posterior,
    prior_adjustor: _PriorAdjustor,
    opt_v_func: _AbstractOptFunc,
) -> Tuple[Prior, Posterior, Array]:
    l_dim, n_snps, n_pop = posteriors.post_mean.shape

    # reduce from lxpxk to pxk
    post_mean_lsum = jnp.sum(posteriors.post_mean, axis=0)

    residual = ys - jnp.einsum("knp,pk->kn", Xs, post_mean_lsum)

    init_l_result = _LResult(
        Xs=Xs,
        ys=residual,
        XtXs=XtXs,
        priors=priors,
        posteriors=posteriors,
        prior_adjustor=prior_adjustor,
        opt_v_func=opt_v_func,
    )

    l_result = lax.fori_loop(0, l_dim, _update_l, init_l_result)

    _, _, _, priors, posteriors, _, _ = l_result

    # from lxpxk to kxpxl
    tr_b_s = posteriors.post_mean.T
    # from lxpxkxk to lxpxk (get the diagonal), then become kxpxl
    tr_bsq_s = jnp.diagonal(posteriors.post_mean_sq, axis1=2, axis2=3).T

    exp_ll = jnp.sum(_eloglike(Xs, ys, ns, tr_b_s, tr_bsq_s, priors.resid_var))
    sigma2 = _erss(Xs, ys, tr_b_s, tr_bsq_s)[:, jnp.newaxis] / ns

    kl_divs = jnp.sum(posteriors.kl)
    elbo_score = exp_ll - kl_divs

    priors = priors._replace(resid_var=sigma2)

    return priors, posteriors, elbo_score


def _update_l(l_iter: int, param: _LResult) -> _LResult:
    Xs, residual, XtXs, priors, posteriors, prior_adjustor, opt_v_func = param
    residual_l = residual + jnp.einsum("knp,kp->kn", Xs, posteriors.post_mean[l_iter].T)

    priors, posteriors = _ssr(
        Xs,
        residual_l,
        XtXs,
        priors,
        posteriors,
        prior_adjustor,
        l_iter,
        opt_v_func,
    )

    residual = residual_l - jnp.einsum("knp,kp->kn", Xs, posteriors.post_mean[l_iter].T)

    update_param = param._replace(
        ys=residual,
        priors=priors,
        posteriors=posteriors,
    )

    return update_param


def _ssr(
    Xs: ArrayLike,
    ys: ArrayLike,
    XtXs: ArrayLike,
    priors: Prior,
    posteriors: Posterior,
    prior_adjustor: _PriorAdjustor,
    l_iter: int,
    opt_v_func: _AbstractOptFunc,
) -> Tuple[Prior, Posterior]:
    n_pop, _, n_snps = Xs.shape

    Xty = jnp.einsum("knp,kn->kp", Xs, ys)
    rTZDinv = (Xty / priors.resid_var).T

    # priors.resid_var is kx1, XtXs is kxp, and the result is kxp, and inverse is pxk
    inv_shat2 = (XtXs / priors.resid_var).T

    # expand it to diag matrix, so they're pxkxk
    inv_shat2 = jnp.eye(n_pop) * inv_shat2[:, jnp.newaxis]

    priors = opt_v_func(rTZDinv, inv_shat2, priors, posteriors, prior_adjustor, l_iter)

    _, posteriors = _compute_posterior(rTZDinv, inv_shat2, priors, posteriors, l_iter)

    return priors, posteriors


def _compute_posterior(
    rTZDinv: ArrayLike,
    inv_shat2: ArrayLike,
    priors: Prior,
    posteriors: Posterior,
    l_iter: int,
) -> Tuple[Prior, Posterior]:
    n_snps, n_pop, _ = inv_shat2.shape

    # prior_covar is kxk
    prior_covar = priors.effect_covar[l_iter]
    # post_covar is pxkxk
    post_covar = jnp.linalg.inv(inv_shat2 + jnp.linalg.inv(prior_covar))

    # dim m = dim k for the next two lines
    post_mean = jnp.einsum("pkm,pm->pk", post_covar, rTZDinv)
    post_mean_sq = post_covar + jnp.einsum("pk,pm->pkm", post_mean, post_mean)

    # compute the ABF in the original susie paper
    # which is equivalent to the inverse posterior density at 0
    log_bf = -1 * stats.multivariate_normal.logpdf(
        # origianlly it was logpdf(jnp.zeros((n_snps, n_pop)), post_mean, post_covar)
        # but to match what our math derivation in the paper, we change it to following
        # it will not change the result as N(0; 1, 1) is the same as N(1; 0, 1) in terms of density value
        post_mean,
        jnp.zeros((n_snps, n_pop)),
        post_covar,
    )

    alpha = nn.softmax(jnp.log(priors.pi) + log_bf)

    weighted_post_mean = post_mean * alpha[:, jnp.newaxis]
    weighted_post_mean_sq = post_mean_sq * alpha[:, jnp.newaxis, jnp.newaxis]
    # this is also the prior in our E step
    weighted_sum_covar = jnp.sum(weighted_post_mean_sq, axis=0)
    kl_alpha = _kl_categorical(alpha, priors.pi)
    kl_betas = alpha @ _kl_mvn(post_mean, post_covar, 0.0, prior_covar)

    priors = priors._replace(
        effect_covar=priors.effect_covar.at[l_iter].set(weighted_sum_covar)
    )

    posteriors = posteriors._replace(
        alpha=posteriors.alpha.at[l_iter].set(alpha),
        post_mean=posteriors.post_mean.at[l_iter].set(weighted_post_mean),
        post_mean_sq=posteriors.post_mean_sq.at[l_iter].set(weighted_post_mean_sq),
        weighted_sum_covar=posteriors.weighted_sum_covar.at[l_iter].set(
            weighted_sum_covar
        ),
        kl=posteriors.kl.at[l_iter].set(kl_alpha + kl_betas),
        log_bf=posteriors.log_bf.at[l_iter].set(log_bf),
    )

    return priors, posteriors


def _kl_categorical(
    alpha: ArrayLike,
    pi: ArrayLike,
) -> Array:
    return jnp.nansum(alpha * (jnp.log(alpha) - jnp.log(pi)))


def _kl_mvn(
    m0: ArrayLike,
    sigma0: ArrayLike,
    m1: ArrayLike,
    sigma1: ArrayLike,
) -> float:
    # https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions
    k, _ = sigma1.shape

    sigma1_inv = jnp.linalg.inv(sigma1)
    # m1 and sigma1 is the prior
    # m0 and sigma0 is the posterior
    # m0 is pxk and m1 is 0, so diff is pxk
    diff = m1 - m0

    # sigma1_inv is prior, so it's kxk, sigma0 is pxkxk
    # we want to multiply within each p
    # to calculate trace using einsum, e.g.: jnp.einsum("ii",A) where A is a 2x2 matrix
    p1 = jnp.einsum("km,pmk->p", sigma1_inv, sigma0) - k

    # diff is pxk, sigma1_inv is kxk, we want to multiply within each p
    p2 = jnp.einsum("pk,km,pm->p", diff, sigma1_inv, diff)

    _, sld1 = jnp.linalg.slogdet(sigma1)
    _, sld0 = jnp.linalg.slogdet(sigma0)

    p3 = sld1 - sld0

    return 0.5 * (p1 + p2 + p3)


def _eloglike(
    X: ArrayLike,
    y: ArrayLike,
    ns: ArrayLike,
    beta: ArrayLike,
    beta_sq: ArrayLike,
    sigma_sq: ArrayLike,
) -> Array:
    norm_term = -(0.5 * ns) * jnp.log(2 * jnp.pi * sigma_sq)
    quad_term = -(0.5 / sigma_sq) * _erss(X, y, beta, beta_sq)[:, jnp.newaxis]
    return norm_term + quad_term


def _erss(X: ArrayLike, y: ArrayLike, beta: ArrayLike, beta_sq: ArrayLike) -> Array:
    mu_li = jnp.einsum("knp,kpl->knl", X, beta)
    mu2_li = jnp.einsum("knp,kpl->knl", X ** 2, beta_sq)

    # jnp.sum(mu_li, axis=2) sum across l, get kxn, then sum across n, term_1 is kx1
    term_1 = jnp.sum((y - jnp.sum(mu_li, axis=2)) ** 2, axis=1)
    # sum across n and l, then term_2 is kx1
    term_2 = jnp.sum(mu2_li - (mu_li ** 2), axis=(1, 2))

    return term_1 + term_2


def _reorder_l(priors: Prior, posteriors: Posterior) -> Tuple[Prior, Posterior, Array]:

    frob_norm = jnp.sum(
        jnp.linalg.svd(posteriors.weighted_sum_covar, compute_uv=False), axis=1
    )

    # we want to reorder them based on the Frobenius norm
    l_order = jnp.argsort(-frob_norm)

    # priors effect_covar
    priors = priors._replace(effect_covar=priors.effect_covar[l_order])

    posteriors = posteriors._replace(
        alpha=posteriors.alpha[l_order],
        post_mean=posteriors.post_mean[l_order],
        post_mean_sq=posteriors.post_mean_sq[l_order],
        weighted_sum_covar=posteriors.weighted_sum_covar[l_order],
        kl=posteriors.kl[l_order],
        log_bf=posteriors.log_bf[l_order],
    )

    return priors, posteriors, l_order


def make_cs(
    alpha: ArrayLike,
    log_bf: ArrayLike,
    ns: ArrayLike,
    Xs: ArrayLike = None,
    lds: ArrayLike = None,
    threshold: float = 0.9,
    purity: float = 0.5,
    purity_method: str = "weighted",
    max_select: int = 500,
    seed: int = 12345,
) -> Tuple[pd.DataFrame, pd.DataFrame, Array, Array]:
    """The function to compute the credible sets.

    Args:
        alpha: :math:`L \\times p` matrix that contains posterior probability for SNP to be causal
            (i.e., :math:`\\alpha` in :ref:`Model`).
        log_bf: :math:`L \\times p` matrix that contains log Bayes factor for each SNP in each effect.
        Xs: Genotype data for multiple ancestries. It cannot be None if lds is None.
        lds: LD matrix for multiple ancestries. It cannot be None if Xs is None.
        ns: Sample size for each ancestry.
        threshold: The credible set threshold.
        purity: The minimum pairwise correlation across SNPs to be eligible as output credible set.
        purity_method: The method to compute purity across ancestries.
        max_select: The maximum number of selected SNPs to compute purity.
        seed: The randomization seed for selecting SNPs in the credible set to compute purity.

    Returns:
        :py:obj:`Tuple[pd.DataFrame, pd.DataFrame, Array, Array]`: A tuple of
            #. credible set (:py:obj:`pd.DataFrame`) after pruning for purity,
            #. full credible set (:py:obj:`pd.DataFrame`) before pruning for purity,
            #. PIPs (:py:obj:`Array`) across :math:`L` credible sets,
            #. PIPs (:py:obj:`Array`) across credible sets that are not pruned. An array of zero if all credible sets
                are pruned.

    Example:
        Compute credible sets from SuShiE posterior::

            from sushie.infer import infer_sushie, make_cs

            # Run SuShiE first
            result = infer_sushie(Xs=[X1, X2], ys=[y1, y2], L=5)

            # Extract credible sets with custom threshold
            cs, full_cs, pip_all, pip_cs = make_cs(
                alpha=result.posteriors.alpha,
                log_bf=result.posteriors.log_bf,
                ns=np.array([X1.shape[0], X2.shape[0]]),
                Xs=[X1, X2],
                threshold=0.9,
                purity=0.5
            )

    """
    if Xs is None and lds is None:
        raise ValueError(
            "Both Xs and lds are None. Please specify at least one of them."
        )

    rng_key = random.PRNGKey(seed)
    n_l, _ = alpha.shape
    t_alpha = pd.DataFrame(alpha.T).reset_index()

    cs = pd.DataFrame(columns=["CSIndex", "SNPIndex", "alpha", "c_alpha"])
    full_alphas = t_alpha[["index"]]

    for ldx in range(n_l):
        # select original index and alpha
        tmp_pd = (
            t_alpha[["index", ldx]]
            .sort_values(ldx, ascending=False)
            .reset_index(drop=True)
        )
        tmp_pd["csum"] = tmp_pd[[ldx]].cumsum()
        n_row = tmp_pd[tmp_pd.csum < threshold].shape[0]

        # if select rows less than total rows, n_row + 1
        if n_row == tmp_pd.shape[0]:
            select_idx = jnp.arange(n_row)
        else:
            select_idx = jnp.arange(n_row + 1)

        # output CS Index is 1-based
        tmp_cs = (
            tmp_pd.iloc[select_idx, :]
            .assign(CSIndex=(ldx + 1))
            .rename(columns={"csum": "c_alpha", "index": "SNPIndex", ldx: "alpha"})
        )

        tmp_pd["in_cs"] = (tmp_pd.index.values <= jnp.max(select_idx)) * 1

        # prepare alphas table's entries
        tmp_pd = tmp_pd.drop(["csum"], axis=1).rename(
            columns={
                "in_cs": f"in_cs_l{ldx + 1}",
                ldx: f"alpha_l{ldx + 1}",
            }
        )

        full_alphas = full_alphas.merge(tmp_pd, how="left", on="index")

        # check the purity
        snp_idx = tmp_cs.SNPIndex.values.astype("int64")

        # randomly select `max_select` SNPs
        if len(snp_idx) > max_select:
            snp_idx = random.choice(
                rng_key, snp_idx, shape=(max_select,), replace=False
            )

        # update the genotype data and LD
        if Xs is not None:
            ld_Xs = Xs[:, :, snp_idx]
            ld = jnp.einsum("ijk,ijm->ikm", ld_Xs, ld_Xs) / ns[:, jnp.newaxis]
        elif lds is not None:
            ld = lds[:, snp_idx, :][:, :, snp_idx]

        if purity_method == "weighted":
            ss_weight = ns / jnp.sum(ns)

            avg_corr = jnp.sum(
                jnp.min(jnp.abs(ld), axis=(1, 2))[:, jnp.newaxis] * ss_weight
            )
        elif purity_method == "max":
            avg_corr = jnp.max(jnp.min(jnp.abs(ld), axis=(1, 2)))
        elif purity_method == "min":
            avg_corr = jnp.min(jnp.min(jnp.abs(ld), axis=(1, 2)))
        else:
            raise ValueError(
                f"Invalid purity method {purity_method}. Choose from 'weighted', 'max', or 'min'."
            )

        full_alphas[f"purity_l{ldx + 1}"] = avg_corr

        if avg_corr > purity:
            cs = pd.concat([cs, tmp_cs], ignore_index=True)
            full_alphas[f"kept_l{ldx + 1}"] = 1
        else:
            full_alphas[f"kept_l{ldx + 1}"] = 0

        # add log bayes factor
        full_alphas[f"log_bf_l{ldx + 1}"] = log_bf[ldx, :]

    pip_all = utils.make_pip(alpha)

    # CSIndex is now 1-based
    pip_cs = utils.make_pip(
        alpha[
            (cs.CSIndex.unique().astype(int) - 1),
        ]
    )

    n_snp_cs = cs.SNPIndex.values.astype(int)
    n_snp_cs_unique = jnp.unique(cs.SNPIndex.values.astype(int))

    if len(n_snp_cs) != len(n_snp_cs_unique):
        log.logger.warning(
            "Same SNPs appear in different credible set, which is very unusual."
            + " You may want to check this gene in details."
        )

    cs["pip_all"] = jnp.array([pip_all[idx] for idx in cs.SNPIndex.values.astype(int)])
    cs["pip_cs"] = jnp.array([pip_cs[idx] for idx in cs.SNPIndex.values.astype(int)])

    full_alphas["pip_all"] = pip_all
    full_alphas["pip_cs"] = pip_cs
    full_alphas = full_alphas.rename(columns={"index": "SNPIndex"})

    log.logger.info(
        f"{len(cs.CSIndex.unique())} out of {n_l} credible sets remain after pruning based on purity ({purity})."
        + " For detailed results, specify --alphas."
    )

    return cs, full_alphas, pip_all, pip_cs
