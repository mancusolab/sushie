#!/usr/bin/env python

from __future__ import division

import argparse
import copy
import logging
import math
import os
import sys
import typing
import warnings
from importlib import metadata

import limix.her as her
import pandas as pd
from jax import random
from pandas_plink import read_plink
from scipy import stats

from . import LOG, core, infer

warnings.filterwarnings("ignore")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import jax.numpy as jnp


def get_command_string(args):
    """
    Format focus call and options into a string for logging/printing
    :return: string containing formatted arguments to focus
    """

    base = "sushie {}{}".format(args[0], os.linesep)
    rest = args[1:]
    rest_strs = []
    needs_tab = True
    for cmd in rest:
        if "-" == cmd[0]:
            if cmd in ["--quiet", "-q", "--verbose", "-v"]:
                rest_strs.append("\t{}{}".format(cmd, os.linesep))
                needs_tab = True
            else:
                rest_strs.append("\t{}".format(cmd))
                needs_tab = False
        else:
            if needs_tab:
                rest_strs.append("\t{}{}".format(cmd, os.linesep))
                needs_tab = True
            else:
                rest_strs.append(" {}{}".format(cmd, os.linesep))
                needs_tab = True

    return base + "".join(rest_strs) + os.linesep


def drop_na_inf(df: pd.DataFrame, nam: str, idx: int) -> pd.DataFrame:
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


def regress_resid(X, y):
    """
    Perform a marginal linear regression for each snp on the phenotype.

    :param X: numpy.ndarray n x p genotype matrix to regress over
    :param y: numpy.ndarray phenotype vector

    :return: pandas.DataFrame containing estimated beta and standard error
    """
    n_samples, n_snps = X.shape
    X_intercept = jnp.append(jnp.ones((n_samples, 1)), X, axis=1)
    XtX_inv = jnp.linalg.inv(jnp.transpose(X_intercept) @ X_intercept)
    betas = XtX_inv @ jnp.transpose(X_intercept) @ y
    residual = y - X_intercept @ betas
    rss = jnp.sum(residual ** 2)
    sigma_sq = rss / (n_samples - n_snps - 1)
    t_scores = betas / jnp.sqrt(jnp.diagonal(XtX_inv) * sigma_sq)
    p_val = stats.t.sf(abs(t_scores), df=(n_samples - n_snps - 1))
    r_squared = 1 - rss / jnp.sum((y - jnp.mean(y)) ** 2)
    adj_r = 1 - (1 - r_squared) * (n_samples - 1) / (n_samples - n_snps - 1)

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

    :param base0: A0 for the first ancestry (baseline)
    :param base1: A1 for the first ancestry (baseline)
    :param compare0: A0 for the compared ancestry
    :param compare1: A1 for the compared ancestry
    :param idx: Ancestry Index

    :return: pandas.DataFrame containing estimated beta and standard error
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


def estimate_her(X: jnp.ndarray, y: jnp.ndarray, C: jnp.ndarray = None) -> float:
    n, p = X.shape
    A = jnp.dot(X, X.T) / p
    h2g = her.estimate(y, "normal", A, C, verbose=False)

    return h2g


def parameter_check(
    args,
) -> typing.Tuple[core.ListOrNone, core.ListOrNone, core.ListOrNone]:
    log = logging.getLogger(LOG)

    if len(args.geno) == len(args.pheno):
        n_pop = len(args.geno)
        log.info(f"Detecting {n_pop} ancestries for SuShiE fine-mapping.")
    else:
        raise ValueError(
            "The number of geno and pheno data does not match. Check your input."
        )

    if args.covar is not None:
        if len(args.covar) != n_pop:
            raise ValueError("The number of covariate data does not match geno data.")

    if args.resid_var is not None:
        if len(args.resid_var) != n_pop:
            raise ValueError(
                "Number of specified residual prior does not match geno data."
            )

        resid_var = [float(i) for i in args.resid_var]

        if jnp.any(jnp.array(resid_var) <= 0):
            raise ValueError(
                "The input of residual prior is invalid (<0). Check your input."
            )
    else:
        resid_var = None

    if args.effect_covar is not None:
        effect_covar = args.effect_covar
        if len(effect_covar) != n_pop:
            raise ValueError(
                "Number of specified effect prior does not match feature number."
            )
        effect_covar = [float(i) for i in effect_covar]
        if jnp.any(jnp.array(effect_covar) <= 0):
            raise ValueError("The input of effect size prior is invalid (<0).")
    else:
        effect_covar = None

    if n_pop != 1 and args.rho is not None:
        rho = args.rho
        exp_num_rho = math.comb(n_pop, 2)
        if len(rho) != exp_num_rho:
            raise ValueError(
                (
                    f"Number of specified rho ({len(rho)}) does not match expected",
                    f"number {exp_num_rho}.",
                )
            )
        rho = [float(i) for i in rho]
        # double-check if it's invalid rho
        if jnp.any(jnp.abs(jnp.array(rho)) >= 1):
            raise ValueError(
                "The input of rho is invalid (>=1 or <=-1). Check your input."
            )
    else:
        rho = None

    if args.pi is not None and (args.pi > 1 or args.pi < 0):
        raise ValueError(
            "Pi prior is not a probability (0-1). Specify a valid pi prior."
        )

    if args.L <= 0:
        raise ValueError("Inferred L is invalid, choose a positive L.")

    if args.min_tol > 0.1:
        log.warning("Minimum intolerance is low. Inference may not be accurate.")

    if not 0 < args.threshold < 1:
        raise ValueError("CS threshold is not between 0 and 1. Specify a valid one.")

    if not 0 < args.purity < 1:
        raise ValueError("Purity is not between 0 and 1. Specify a valid one.")

    if args.opt_mode == "noop":
        if resid_var is None or effect_covar is None or rho is None:
            raise ValueError(
                "'noop' is specified. --resid_var, --effect_covar, and --rho cannot be None."
            )

        log.warning(
            (
                "No updates on the effect size prior because of 'noop' setting for"
                " opt_mode. Inference may be inaccurate.",
            )
        )

    if args.cv:
        if args.cv_num <= 1:
            raise ValueError(
                (
                    "The number of folds in cross validation is invalid.",
                    " Choose some number greater than 1.",
                )
            )
        elif args.cv_num > 5:
            log.warning(
                (
                    "The number of folds in cross validation is too large.",
                    " It may cause longer running time.",
                )
            )

        if args.seed <= 0:
            raise ValueError(
                "The seed specified for CV is invalid. Please choose a positive integer."
            )

    return resid_var, effect_covar, rho


def process_raw(
    geno_paths: typing.List[str],
    pheno_paths: typing.List[str],
    covar_paths: typing.List[str] = None,
    norm_X: bool = True,
    norm_y: bool = False,
    h2g: bool = False,
    regress: bool = True,
) -> core.CleanData:
    log = logging.getLogger(LOG)

    n_pop = len(geno_paths)

    bim = []
    fam = []
    bed = []
    pheno = []

    # read in geno and pheno
    for idx in range(n_pop):
        log.info(f"Ancestry {idx + 1}: Reading in genotype data and phenotype data.")
        tmp_bim, tmp_fam, tmp_bed = read_plink(geno_paths[idx], verbose=False)

        # read pheno data
        tmp_pheno = pd.read_csv(
            pheno_paths[idx], sep="\t", header=None, dtype={0: object}
        )

        # drop all the nan inf values
        tmp_bim = drop_na_inf(tmp_bim, "bim", idx)
        tmp_fam = drop_na_inf(tmp_fam, "fam", idx)
        tmp_pheno = drop_na_inf(tmp_pheno, "pheno", idx)

        # rename the columns with pop index for better processing in the future
        tmp_bim = tmp_bim.rename(
            columns={
                "i": f"bimIDX_{idx + 1}",
                "a0": f"a0_{idx + 1}",
                "a1": f"a1_{idx + 1}",
                "cm": f"cm_{idx + 1}",
                "pos": f"pos_{idx + 1}",
            }
        )
        tmp_fam = tmp_fam[["fid", "iid", "i"]].rename(
            columns={"i": f"famIDX_{idx + 1}"}
        )
        tmp_pheno = tmp_pheno.reset_index()
        # keep track of pheno index for future matching the bed file if bed
        # files are shuffled due to merging
        tmp_pheno = tmp_pheno.rename(
            columns={"index": f"phenoIDX_{idx + 1}", 0: "fid", 1: "iid"}
        )

        if len(tmp_bim) == 0:
            raise ValueError(
                (
                    f"Ancestry {idx + 1}: No genotype data found for ancestry at {geno_paths[idx]}."
                )
            )

        tmp_bed_value = tmp_bed.compute()
        bim.append(tmp_bim)
        fam.append(tmp_fam)
        bed.append(tmp_bed_value)
        pheno.append(tmp_pheno)

    # read in covar
    if covar_paths is not None:
        covar = []
        for idx in range(n_pop):
            tmp_covar = pd.read_csv(
                covar_paths[idx], sep="\t", header=None, dtype={0: object}
            )
            tmp_covar = drop_na_inf(tmp_covar, "covar", idx)
            tmp_covar = tmp_covar.reset_index()
            # keep track of covar index for future matching the bed file if
            # bed files are shuffled due to merging
            tmp_covar = tmp_covar.rename(
                columns={"index": f"covarIDX_{idx + 1}", 0: "fid", 1: "iid"}
            )
            covar.append(tmp_covar)
    else:
        covar = [None] * n_pop
        log.warning("No covariates detected for this analysis.")

    # find common snps across ancestries
    if n_pop > 1:
        snps = pd.merge(bim[0], bim[1], how="inner", on=["chrom", "snp"])
        for idx in range(n_pop - 2):
            snps = pd.merge(snps, bim[idx + 2], how="inner", on=["chrom", "snp"])
        # report how many snps we removed due to independent SNPs
        for idx in range(n_pop):
            snps_num_diff = bim[idx].shape[0] - snps.shape[0]
            log.warning(
                (
                    f"Ancestry{idx + 1} has {snps_num_diff} independent SNPs and "
                    f"{snps.shape[0]} common SNPs. Will remove these independent SNPs.",
                )
            )
    else:
        snps = bim[0]

    # find flipped reference alleles across ancestries
    flip_idx = []
    if n_pop > 1:
        for idx in range(1, n_pop):
            correct_idx, tmp_flip_idx, wrong_idx = allele_check(
                snps["a0_1"].values,
                snps["a1_1"].values,
                snps[f"a0_{idx + 1}"].values,
                snps[f"a1_{idx + 1}"].values,
                idx,
            )

            # save the index for future swapping
            flip_idx.append(tmp_flip_idx)

            if len(wrong_idx) != 0:
                snps = snps.drop(index=wrong_idx)

            # drop unused columns
            snps = snps.drop(columns=[f"a0_{idx + 1}", f"a1_{idx + 1}"])
        # rename columns for better indexing in the future
    snps = snps.reset_index().rename(
        columns={"index": "SNPIndex", "a0_1": "a0", "a1_1": "a1"}
    )

    # find common individuals across geno, pheno, and covar within an ancestry
    common_fam = []
    for idx in range(n_pop):
        # match fam id and pheno id
        tmp_common_fam = pd.merge(
            fam[idx],
            pheno[idx][[f"phenoIDX_{idx + 1}", "fid", "iid"]],
            how="inner",
            on=["fid", "iid"],
        )
        if covar[idx] is not None:
            # match fam id and covar id
            tmp_common_fam = pd.merge(
                tmp_common_fam,
                covar[idx][[f"covarIDX_{idx + 1}", "fid", "iid"]],
                how="inner",
                on=["fid", "iid"],
            )
        common_fam.append(tmp_common_fam)
        n_common = tmp_common_fam.shape[0]
        if n_common == 0:
            raise ValueError(
                (
                    f"Ancestry {idx + 1}: No common individuals between phenotype and "
                    "genotype found. Please double check source data.",
                )
            )
        else:
            log.info(
                (
                    f"Ancestry {idx + 1}: Found {n_common} common individuals",
                    " between phenotype and genotype.",
                )
            )
            # sanity check how many we lose
            fam_drop_ind = fam[idx].shape[0] - n_common
            pheno_drop_ind = pheno[idx].shape[0] - n_common
            covar_drop_ind = 0
            if covar[idx] is not None:
                covar_drop_ind = covar[idx].shape[0] - n_common

            if (fam_drop_ind != 0 or pheno_drop_ind != 0) or covar_drop_ind != 0:
                log.warning(
                    f"Ancestry {idx + 1}: Delete {fam_drop_ind} from fam file. "
                )
                log.warning(
                    f"Ancestry {idx + 1}: Delete {pheno_drop_ind} from phenotype file."
                )
                log.warning(
                    f"Ancestry {idx + 1}: Delete {covar_drop_ind} from covar file."
                )

    # filter on geno, pheno, and covar
    for idx in range(n_pop):
        # get individual and snp id
        common_ind_id = common_fam[idx][f"famIDX_{idx + 1}"].values
        common_snp_id = snps[f"bimIDX_{idx + 1}"].values

        # filter on individuals who have both geno, pheno, and covar (if applicable)
        # filter on shared snps across ancestries
        tmp_bed = bed[idx][common_snp_id, common_ind_id]

        # flip genotypes for bed files starting second ancestry
        if idx > 0:
            # flip_idx count starts at the second ancestry
            tmp_bed[flip_idx[idx - 1]] = 2 - tmp_bed[flip_idx[idx - 1]]
        tmp_bed = tmp_bed.T

        if norm_X:
            tmp_bed -= jnp.mean(tmp_bed, axis=0)
            tmp_bed /= jnp.std(tmp_bed, axis=0)

        bed[idx] = tmp_bed

        snps = snps.drop(columns=[f"bimIDX_{idx + 1}"])

        # swap pheno and covar rows order to match fam/bed file, and then select the
        # values for future fine-mapping
        common_pheno_id = common_fam[idx][f"phenoIDX_{idx + 1}"].values
        tmp_pheno = pheno[idx].iloc[common_pheno_id, 3].values

        if norm_y:
            tmp_pheno -= jnp.mean(tmp_pheno)
            tmp_pheno /= jnp.std(tmp_pheno)

        pheno[idx] = tmp_pheno

        if covar[idx] is not None:
            common_covar_id = common_fam[idx][f"covarIDX_{idx + 1}"].values
            n_covar = covar[idx].shape[1]
            covar[idx] = covar[idx].iloc[common_covar_id, 3:n_covar].values

    # estimate heritability
    if h2g:
        est_h2g = jnp.zeros(n_pop)
        for idx in range(n_pop):
            tmp_h2g = estimate_her(bed[idx], pheno[idx], covar[idx])
            est_h2g = est_h2g.at[idx].set(tmp_h2g)
    else:
        est_h2g = None

    # regress covar on y
    for idx in range(n_pop):
        if covar[idx] is not None:
            pheno[idx] = regress_resid(covar[idx], pheno[idx])
            # regress covar on each SNP, it might be slow, the default is True
            if regress:
                tmp_n, tmp_p = bed[idx].shape
                for snp in range(tmp_p):
                    # seems jnp array doesn't work in sm, so use np.array
                    bed[idx] = (
                        bed[idx]
                        .at[:, snp]
                        .set(regress_resid(covar[idx], bed[idx][:, snp]))
                    )
    log.info(
        (
            f"Successfully prepare genotype ({bed[0].shape[1]} SNPs) and phenotype data for {n_pop}",
            " ancestries, and start fine-mapping using SuShiE.",
        )
    )
    result = core.CleanData(
        geno=bed,
        pheno=pheno,
        covar=covar,
        snp=snps,
        h2g=est_h2g,
    )
    return result


def _output_cv(args, clean_data, resid_var, effect_covar, rho):
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
            increment = int(tmp_n / args.crossval_num)
            start = cv * increment
            end = (cv + 1) * increment
            # if it is the last fold, take all the rest of the data.
            if cv == args.crossval_num - 1:
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
            effect_covar=effect_covar,
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
        _, adj_r2, pval = regress_resid(est_y[idx], cv_pheno[idx])
        cv_res.append([adj_r2, pval])

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
            tmp_shared_h2g = estimate_her(
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


def _output_cs(args, result, clean_data):
    cs = (
        pd.merge(clean_data.snp, result.cs, how="inner", on=["SNPIndex"])
        .drop(columns=["SNPIndex"])
        .assign(trait=args.trait)
        .sort_values(by=["CSIndex", "pip", "cpip"], ascending=[True, False, True])
    )
    cs.to_csv(f"{args.output}.cs.tsv", sep="\t", index=None)

    return None


def run_finemap(args):
    log = logging.getLogger(LOG)

    try:
        resid_var, effect_covar, rho = parameter_check(args)

        clean_data = process_raw(
            args.geno, args.pheno, args.covar, args.norm_X, args.norm_y, args.regress
        )

        result = infer.run_sushie(
            clean_data.geno,
            clean_data.pheno,
            L=args.L,
            norm_X=args.norm_X,
            norm_y=args.norm_y,
            pi=args.pi,
            resid_var=resid_var,
            effect_covar=effect_covar,
            rho=rho,
            max_iter=args.max_iter,
            min_tol=args.min_tol,
            opt_mode=args.opt_mode,
            threshold=args.threshold,
            purity=args.purity,
        )

        # output credible set
        _output_cs(args, result, clean_data)

        if args.h2g:
            _output_h2g(args, result, clean_data)

        if args.weights:
            _output_weights(args, result, clean_data)

        if args.corr:
            _output_corr(args, result)

        if args.cv:
            log.info(f"Starting {args.cv_num}-fold cross validation.")
            _output_cv(args, clean_data, resid_var, effect_covar, rho)

    except Exception as err:
        log.error(err)
    finally:
        log.info(
            (
                "Finished SuShiE fine-mapping. Thanks for using our software.\n",
                "For bug reporting, suggestions, and comments, please go to https://github.com/mancusolab/sushie.",
            )
        )
    return 0


def build_finemap_parser(subp):
    # add imputation parser
    finemap = subp.add_parser(
        "finemap",
        description="Perform fine-mapping on individual genotype and phenotype data using SuShiE.",
    )

    # main arguments
    finemap.add_argument(
        "--geno",
        nargs="+",
        type=str,
        help="Genotype data in plink format. Use space to separate ancestries if more than two.",
    )

    finemap.add_argument(
        "--pheno",
        nargs="+",
        type=str,
        help=(
            "Phenotype data. It has to be a tsv file that contains at least three",
            " columns where the first two columns are FID and IID as in plink fam file.",
            " No headers. Use space to separate ancestries if more than two.",
        ),
    )

    # fine-map general options
    finemap.add_argument(
        "--covar",
        nargs="+",
        default=None,
        type=str,
        help=(
            "Covariates that will be accounted in the fine-mapping. Default is None."
            " It has to be a tsv file that contains at least three columns where the",
            " first two columns are FID and IID as in plink fam file. You need to",
            " pre-process the character covariates into dummy variables. No headers.",
            " Use space to separate ancestries if more than two.",
        ),
    )

    finemap.add_argument(
        "--L",
        default=10,
        type=int,
        help=(
            "Integer number of shared effects pre-specified.",
            " Default is 10. Larger number may cause slow inference.",
        ),
    )

    finemap.add_argument(
        "--norm_X",
        default=True,
        type=bool,
        help=(
            "Indicator to standardize genotype data by centering around mean and scaling by standard deviation.",
            " Default is True. False may causal different inference.",
        ),
    )

    finemap.add_argument(
        "--norm_y",
        default=False,
        type=bool,
        help=(
            "Indicator to standardize phenotype data by centering around mean and scaling by standard deviation.",
            " Default is False. True may causal different inference.",
        ),
    )

    finemap.add_argument(
        "--regress",
        default=True,
        type=bool,
        help=(
            "Indicator to regress the covariates on each SNP. Default is True.",
            " True can slow the inference, but can be more accurate.",
        ),
    )

    finemap.add_argument(
        "--pi",
        default=None,
        type=float,
        help=(
            "Prior probability for each SNP to be causal.",
            " Default is 1/p where p is the number of SNPs in the region.",
        ),
    )

    finemap.add_argument(
        "--resid_var",
        nargs="+",
        default=None,
        type=float,
        help=(
            "Specify the prior for the residual variance for ancestries. Default is None.",
            " Values have to be positive. Use space to separate ancestries if more than two.",
        ),
    )

    finemap.add_argument(
        "--effect_covar",
        nargs="+",
        default=None,
        type=float,
        help=(
            "Specify the prior for the causal variance for ancestries. Default is None.",
            " Values have to be positive. Use space to separate ancestries if more than two.",
        ),
    )

    finemap.add_argument(
        "--rho",
        nargs="+",
        default=None,
        type=float,
        help=(
            "Specify the prior for the effect correlation for ancestries. Default is None.",
            " Use space to separate ancestries if more than two. Each rho has to be a float number between -1 and 1.",
            " If more than two ancestries (N), choose(N, 2) is required.",
            " The rho order has to be rho(1,2), rho(1,3), ..., rho(1, N), rho(2,3), ..., rho(N-1. N).",
        ),
    )

    finemap.add_argument(
        "--max_iter",
        default=500,
        type=int,
        help=(
            "Maximum iterations for the optimization. Default is 500.",
            " Larger number can slow the inference while smaller can cause inaccurate estimate.",
        ),
    )

    finemap.add_argument(
        "--min_tol",
        default=1e-5,
        type=float,
        help=(
            "Minimum tolerance for the convergence. Default is 1e-5.",
            " Smaller number can slow the inference while larger can cause inaccurate estimate.",
        ),
    )

    finemap.add_argument(
        "--opt_mode",
        choices=["em", "noop"],
        type=str,
        default="em",
        help=(
            "Optimization method to update prior effect covariance matrix. Default is 'em'.",
            " Other option is 'noop'.",
            " 'em' is similar to Expectation–maximization algorithm that computes the prior covariance conditioned",
            " on other parameters."
            " 'noop' does not update prior effect covariance matrix at each iteration. The inference may be inaccurate",
            " If 'noop' is specified, it requires input for --resid_var, --effect_covar, and --rho.",
        ),
    )

    finemap.add_argument(
        "--threshold",
        default=0.9,
        type=float,
        help=(
            "Specify the PIP threshold for SNPs to be included in the credible sets. Default is 0.9.",
            " It has to be a float number between 0 and 1.",
        ),
    )

    finemap.add_argument(
        "--purity",
        default=0.5,
        type=float,
        help=(
            "Specify the purity threshold for credible sets to be output. Default is 0.5.",
            " It has to be a float number between 0 and 1.",
        ),
    )

    finemap.add_argument(
        "--h2g",
        default=False,
        type=bool,
        help=(
            "Indicator to perform heritability analysis using limix and output *.h2g.tsv file.",
            " Default is False. True will cause longer running time.",
            " *.h2g.tsv file contains two estimated h2g using all genotypes and using only SNPs in the credible sets.",
        ),
    )

    finemap.add_argument(
        "--weights",
        default=False,
        type=bool,
        help=(
            "Indicator to output *.weights.tsv file for prediction weights of all SNPs.",
            " Default is False. True will cause longer running time.",
        ),
    )

    finemap.add_argument(
        "--corr",
        default=False,
        type=bool,
        help=(
            "Indicator to output *.corr.tsv file.",
            " Default is False. True will cause longer running time.",
            " *.cv.tsv file contains estimated variance and covariance across ancestries.",
        ),
    )

    finemap.add_argument(
        "--cv",
        default=False,
        type=bool,
        help=(
            "Indicator to perform cross validation (CV) and output CV results for future FUSION pipline.",
            " Default is False. True will cause longer running time.",
            " CV results are *.cv.tsv file that includes the CV adjusted r-squared and corresponding p-value.",
        ),
    )

    finemap.add_argument(
        "--cv_num",
        default=5,
        type=int,
        help=(
            "The number of fold cross validation. Default is 5.",
            " It has to be a positive integer number. Larger number may cause longer running time.",
        ),
    )

    finemap.add_argument(
        "--seed",
        default=12345,
        type=int,
        help=(
            "The seed to randomly cut data sets in cross validation. Default is 12345.",
            " It has to be positive integer number",
        ),
    )

    finemap.add_argument(
        "--trait",
        default="Trait",
        help=(
            "Trait, tissue, gene name of the phenotype for better indexing in downstream analysis. Default is 'Trait'.",
        ),
    )

    # misc options
    finemap.add_argument(
        "--quiet",
        default=False,
        action="store_true",
        help="Do not print anything to stdout. Default is False.",
    )
    finemap.add_argument(
        "--verbose",
        default=False,
        action="store_true",
        help="Verbose logging. Includes debug info. Default is False.",
    )
    finemap.add_argument(
        "--output",
        default="sushie_finemap",
        help=(
            "Prefix for output data. Default is 'sushie_finemap'.",
            " The software by default will output one file: *.cs.tsv that contains the credible sets.",
        ),
    )

    return finemap


def _main(argsv):
    # setup main parser
    argp = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subp = argp.add_subparsers(
        help="Subcommands: finemap to perform genotype fine-mapping using SuShiE"
    )

    # add subparsers for focus commands
    finemap = build_finemap_parser(subp)
    finemap.set_defaults(func=run_finemap)

    # parse arguments
    args = argp.parse_args(argsv)

    # hack to check that at least one sub-command was selected in 3.6
    # 3.7 -might- have fixed this bug
    if not hasattr(args, "func"):
        argp.print_help()
        return 2  # command-line error

    cmd_str = get_command_string(argsv)

    version = metadata.version("sushie")

    masthead = "===================================" + os.linesep
    masthead += "             SuShiE v{}             ".format(version) + os.linesep
    masthead += "===================================" + os.linesep

    # setup logging
    log_format = "[%(asctime)s - %(levelname)s] %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    log = logging.getLogger(LOG)
    if args.verbose:
        log.setLevel(logging.DEBUG)
    else:
        log.setLevel(logging.INFO)
    fmt = logging.Formatter(fmt=log_format, datefmt=date_format)

    # write to stdout unless quiet is set
    if not args.quiet:
        sys.stdout.write(masthead)
        sys.stdout.write(cmd_str)
        sys.stdout.write("Starting log..." + os.linesep)
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(fmt)
        log.addHandler(stdout_handler)

    # setup log file, but write PLINK-style command first
    disk_log_stream = open(f"{args.output}.log", "w")
    disk_log_stream.write(masthead)
    disk_log_stream.write(cmd_str)
    disk_log_stream.write("Starting log..." + os.linesep)

    disk_handler = logging.StreamHandler(disk_log_stream)
    disk_handler.setFormatter(fmt)
    log.addHandler(disk_handler)

    # launch finemap
    args.func(args)

    return 0


def run_cli():
    return _main(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
