import copy

import pandas as pd

from jax import random
import jax.numpy as jnp

from . import infer, utils


def _output_cs(args, result, clean_data):
    cs = (
        pd.merge(clean_data.snp, result.cs, how="inner", on=["SNPIndex"])
        .drop(columns=["SNPIndex"])
        .assign(trait=args.trait)
        .sort_values(by=["CSIndex", "pip", "cpip"], ascending=[True, False, True])
    )
    cs.to_csv(f"{args.output}.cs.tsv", sep="\t", index=None)

    return None


def _output_corr(args, result):
    n_pop = len(result.priors.resid_var)

    CSIndex = jnp.unique(result.cs.CSIndex.values.astype(int))
    # only output after-purity CS
    corr_cs_only = jnp.transpose(result.priors.effect_covar[CSIndex - 1])
    corr = pd.DataFrame(data={"trait": args.trait, "CSIndex": CSIndex})
    for idx in range(n_pop):
        _var = corr_cs_only[idx, idx]
        tmp_pd = pd.DataFrame(data={f"ancestry{idx + 1}_est_var": _var})
        corr = pd.concat([corr, tmp_pd], axis=1)
        for jdx in range(idx + 1, n_pop):
            _covar = corr_cs_only[idx, jdx]
            _var1 = corr_cs_only[idx, idx]
            _var2 = corr_cs_only[jdx, jdx]
            _corr = _covar / jnp.sqrt(_var1 * _var2)
            tmp_pd_covar = pd.DataFrame(
                data={f"ancestry{idx + 1}_ancestry{jdx + 1}_est_covar": _covar}
            )
            tmp_pd_corr = pd.DataFrame(
                data={f"ancestry{idx + 1}_ancestry{jdx + 1}_est_corr": _corr}
            )
            corr = pd.concat([corr, tmp_pd_covar, tmp_pd_corr], axis=1)
    corr.to_csv(f"{args.output}.corr.tsv", sep="\t", index=False)

    return None


def _output_weights(args, result, clean_data):
    n_pop = len(clean_data.geno)

    snp_copy = copy.deepcopy(clean_data.snp).assign(trait=args.trait)
    tmp_weights = pd.DataFrame(
        data=jnp.sum(result.posteriors.post_mean, axis=0),
        columns=[f"ancestry{idx + 1}_sushie" for idx in range(n_pop)],
    )
    weights = pd.concat([snp_copy, tmp_weights], axis=1)
    weights.to_csv(f"{args.output}.weights.tsv", sep="\t", index=False)

    return None


def _output_h2g(args, result, clean_data):
    n_pop = len(clean_data.geno)

    est_her = pd.DataFrame(
        data=clean_data.h2g,
        index=[f"ancestry{idx + 1}" for idx in range(n_pop)],
        columns=["h2g"],
    ).assign(trait=args.trait)
    # only output h2g that has credible sets
    SNPIndex = result.cs.SNPIndex.values.astype(int)
    if len(SNPIndex) != 0:
        shared_h2g = jnp.zeros(n_pop)
        for idx in range(n_pop):
            tmp_shared_h2g = utils._estimate_her(
                clean_data.geno[idx][:, SNPIndex],
                clean_data.pheno[idx],
                clean_data.covar[idx],
            )
            shared_h2g = shared_h2g.at[idx].set(tmp_shared_h2g)
        shared_pd = pd.DataFrame(
            data=shared_h2g,
            index=[f"ancestry{idx + 1}" for idx in range(n_pop)],
            columns=["shared_h2g"],
        )
        est_her = pd.concat([est_her, shared_pd], axis=1).reset_index()
    est_her.to_csv(f"{args.output}.h2g.tsv", sep="\t", index=False)

    return None


def _output_cv(args, clean_data, resid_var, effect_var, rho):
    rng_key = random.PRNGKey(args.seed)
    cv_geno = copy.deepcopy(clean_data.geno)
    cv_pheno = copy.deepcopy(clean_data.pheno)
    n_pop = len(clean_data.geno)

    # shuffle the data first
    for idx in range(n_pop):
        rng_key, c_key = random.split(rng_key, 2)
        tmp_n = cv_geno[idx].shape[0]
        shuffled_index = random.choice(c_key, tmp_n, (tmp_n,), replace=False)
        cv_pheno[idx] = cv_pheno[idx][shuffled_index]
        cv_geno[idx] = cv_geno[idx][shuffled_index]

    # create a list to store future estimated y value
    est_y = [jnp.array([])] * n_pop
    for cv in range(args.cv_num):
        test_X = []
        train_X = []
        train_y = []
        # make the training and test for each population separately
        # because sample size may be different
        for idx in range(n_pop):
            tmp_n = cv_geno[idx].shape[0]
            increment = int(tmp_n / args.cv_num)
            start = cv * increment
            end = (cv + 1) * increment
            # if it is the last fold, take all the rest of the data.
            if cv == args.cv_num - 1:
                end = max(tmp_n, (cv + 1) * increment)
            test_X.append(cv_geno[idx][start:end])
            train_X.append(cv_geno[idx][jnp.r_[:start, end:tmp_n]])
            train_y.append(cv_pheno[idx][jnp.r_[:start, end:tmp_n]])

        cv_result = infer.run_sushie(
            train_X,
            train_y,
            L=args.L,
            pi=args.pi,
            resid_var=resid_var,
            effect_var=effect_var,
            rho=rho,
            max_iter=args.max_iter,
            min_tol=args.min_tol,
            opt_mode=args.opt_mode,
            threshold=args.threshold,
            purity=args.purity,
        )

        total_weight = jnp.sum(cv_result.posteriors.post_mean, axis=0)
        for idx in range(n_pop):
            tmp_cv_weight = total_weight[:, idx]
            est_y[idx] = jnp.append(est_y[idx], test_X[idx] @ tmp_cv_weight)

    cv_res = []
    for idx in range(n_pop):
        _, adj_r2, pval = utils._ols(est_y[idx][:, jnp.newaxis], cv_pheno[idx])
        cv_res.append([adj_r2, pval[1]])

    sample_size = [i.shape[0] for i in clean_data.geno]
    cv_r2 = (
        pd.DataFrame(
            data=cv_res,
            index=[f"feature{idx + 1}" for idx in range(n_pop)],
            columns=["rsq", "pval"],
        )
        .reset_index()
        .assign(N=sample_size)
    )
    cv_r2.to_csv(f"{args.output}.cv.tsv", sep="\t", index=False)

    return None
