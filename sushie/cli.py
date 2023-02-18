#!/usr/bin/env python

from __future__ import division

import argparse
import copy
import logging
import os
import sys
import warnings
from importlib import metadata
from typing import Callable, List, Optional, Tuple

import pandas as pd

from jax import config, random

from . import infer, io, log, utils

warnings.filterwarnings("ignore")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import jax.numpy as jnp

__all__ = [
    "parameter_check",
    "process_raw",
    "sushie_wrapper",
    "run_finemap",
]


def parameter_check(
    args,
) -> Tuple[List[str], Callable]:
    """The function to process raw phenotype, genotype, covariate data across ancestries.

    Args:
        args: The command line parameter input.

    Returns:
        :py:obj:`Tuple[List[str], Callable]`:
            A tuple of
                #. a list of genotype data paths (:py:obj:`List[str]`),
                #. genotype read-in function (:py:obj:`Callable`).

    """

    n_pop = len(args.pheno)
    log.logger.info(
        f"Detecting phenotypes for {n_pop} ancestries for SuShiE fine-mapping."
    )

    n_geno = (
        int(args.plink is not None)
        + int(args.vcf is not None)
        + int(args.bgen is not None)
    )

    if n_geno > 1:
        log.logger.warning(
            f"Detecting {n_geno} genotypes, will only use one genotypes in the order of 'plink, vcf, and bgen'"
        )

    # decide genotype data
    if args.plink is not None:
        if len(args.plink) != n_pop:
            raise ValueError(
                "The numbers of ancestries in plink geno and pheno data does not match. Check your input."
            )
        else:
            log.logger.info("Detecting genotype data in plink format.")
            geno_path = args.plink
            geno_func = io.read_triplet
    elif args.vcf is not None:
        if len(args.vcf) != n_pop:
            raise ValueError(
                "The numbers of ancestries in vcf geno and pheno data does not match. Check your input."
            )
        else:
            log.logger.info("Detecting genotype data in vcf format.")
            geno_path = args.vcf
            geno_func = io.read_vcf
    elif args.bgen is not None:
        if len(args.bgen) != n_pop:
            raise ValueError(
                "The numbers of ancestries in bgen geno and pheno data does not match. Check your input."
            )
        else:
            log.logger.info("Detecting genotype data in bgen format.")
            geno_path = args.bgen
            geno_func = io.read_bgen
    else:
        raise ValueError(
            "No genotype data specified in either plink, vcf, or bgen format. Check your input."
        )

    if args.covar is not None:
        if len(args.covar) != n_pop:
            raise ValueError("The number of covariate data does not match geno data.")
    else:
        log.logger.info("No covariates detected for this analysis.")

    if args.cv:
        if args.cv_num <= 1:
            raise ValueError(
                "The number of folds in cross validation is invalid."
                + " Choose some number greater than 1.",
            )
        elif args.cv_num > 5:
            log.logger.warning(
                "The number of folds in cross validation is too large."
                + " It may increase running time.",
            )

        if args.seed <= 0:
            raise ValueError(
                "The seed specified for CV is invalid. Please choose a positive integer."
            )

    if (args.meta or args.mega) and n_pop == 1:
        log.logger.warning(
            "The number of ancestry is 1, but --meta or --mega is specified. Will skip meta or mega SuShiE."
        )

    return geno_path, geno_func


def process_raw(
    rawData: List[io.RawData],
    no_regress: bool,
    mega: bool,
    cv: bool,
    cv_num: int,
    seed: int,
) -> Tuple[
    pd.DataFrame,
    io.CleanData,
    Optional[io.CleanData],
    Optional[List[io.CVData]],
]:
    """The function to process raw phenotype, genotype, covariate data across ancestries.

    Args:
        rawData: Raw data for phenotypes, genotypes, covariates across ancestries.
        no_regress: The indicator whether to regress genotypes on covariates.
        mega: The indicator whether to prepare datasets for mega SuShiE.
        cv: The indicator whether to prepare datasets for cross-validation.
        cv_num: The number for :math:`X`-fold cross-validation.
        seed: The random seed for row-wise shuffling the datasets for cross validation.


    Returns:
        :py:obj:`Tuple[pd.DataFrame, io.CleanData, Optional[io.CleanData], Optional[List[io.CVData]]]`:
        A tuple of
            #. SNP information (:py:obj:`pd.DataFrame`),
            #. dataset for running SuShiE (:py:obj:`io.CleanData`),
            #. dataset for mega SuShiE (:py:obj:`Optional[io.CleanData]`),
            #. dataset for cross-validation (:py:obj:`Optional[List[io.CVData]]`).

    """

    n_pop = len(rawData)

    for idx in range(n_pop):

        # remove NA/inf value for subjects
        old_subject_num = rawData[idx].fam.shape[0]
        rawData[idx], del_num = _drop_na_subjects(rawData[idx])

        if del_num != 0:
            log.logger.warning(
                f"Ancestry {idx + 1}: Drop {del_num} out of {old_subject_num} subjects because of INF or NAN value"
                + " in either phenotype or covariate data."
            )

        old_snp_num = rawData[idx].bim.shape[0]
        # impute genotype data even though we suggest users to impute the genotypes beforehand
        rawData[idx], del_num, imp_num = _impute_geno(rawData[idx])

        if del_num != 0:
            log.logger.warning(
                f"Ancestry {idx + 1}: Drop {del_num} out of {old_snp_num} SNPs because all subjects have NAN value"
                + " in genotype data."
            )

        if imp_num != 0:
            log.logger.warning(
                f"Ancestry {idx + 1}: Impute {imp_num} out of {old_snp_num} SNPs with NAN value based on allele"
                + " frequency."
            )

        # reset index and add index column to all dataset for future inter-ancestry or inter-dataset processing
        rawData[idx] = _reset_idx(rawData[idx], idx)

        # find common individuals across geno, pheno, and covar within an ancestry
        rawData[idx] = _filter_common_ind(rawData[idx], idx)

        if rawData[idx].fam.shape[0] == 0:
            raise ValueError(
                f"Ancestry {idx + 1}: No common individuals across phenotype, covariates,"
                + " genotype found. Please double check source data.",
            )
        else:
            log.logger.info(
                f"Ancestry {idx + 1}: Found {rawData[idx].fam.shape[0]} common individuals"
                + " across phenotype, covariate, and genotype.",
            )

    # find common snps across ancestries
    if n_pop > 1:
        snps = pd.merge(
            rawData[0].bim, rawData[1].bim, how="inner", on=["chrom", "snp"]
        )
        for idx in range(n_pop - 2):
            snps = pd.merge(
                snps, rawData[idx + 2].bim, how="inner", on=["chrom", "snp"]
            )

        # report how many snps we removed due to independent SNPs
        for idx in range(n_pop):
            snps_num_diff = rawData[idx].bim.shape[0] - snps.shape[0]
            log.logger.warning(
                f"Ancestry{idx + 1} has {snps_num_diff} independent SNPs and {snps.shape[0]}"
                + " common SNPs. Will remove these independent SNPs.",
            )
    else:
        snps = rawData[0].bim

    # find flipped reference alleles across ancestries
    flip_idx = []
    if n_pop > 1:
        for idx in range(1, n_pop):
            correct_idx, tmp_flip_idx, wrong_idx = _allele_check(
                snps["a0_1"].values,
                snps["a1_1"].values,
                snps[f"a0_{idx + 1}"].values,
                snps[f"a1_{idx + 1}"].values,
            )

            if len(tmp_flip_idx) != 0:
                log.logger.warning(
                    f"Ancestry{idx + 1} has {len(tmp_flip_idx)} flipped alleles from ancestry 1. Will flip these SNPs."
                )

            # save the index for future swapping
            flip_idx.append(tmp_flip_idx)

            if len(wrong_idx) != 0:
                snps = snps.drop(index=wrong_idx)
                log.logger.warning(
                    f"Ancestry{idx + 1} has {len(wrong_idx)} alleles that couldn't be flipped. Will remove these SNPs."
                )

            if snps.shape[0] == 0:
                raise ValueError(
                    f"Ancestry {idx + 1} has none of correct or flippable SNPs from ancestry 1. Check your source.",
                )
            # drop unused columns
            snps = snps.drop(
                columns=[f"a0_{idx + 1}", f"a1_{idx + 1}", f"pos_{idx + 1}"]
            )
        # rename columns for better indexing in the future
    snps = snps.reset_index().rename(
        columns={"index": "SNPIndex", "a0_1": "a0", "a1_1": "a1", "pos_1": "pos"}
    )

    geno = []
    pheno = []
    covar = []
    # filter on geno, pheno, and covar
    for idx in range(n_pop):
        _, tmp_fam, tmp_geno, tmp_pheno, tmp_covar = rawData[idx]

        # get common individual and snp id
        common_ind_id = tmp_fam[f"famIDX_{idx + 1}"].values
        common_snp_id = snps[f"bimIDX_{idx + 1}"].values
        snps = snps.drop(columns=[f"bimIDX_{idx + 1}"])

        # filter on individuals who have both geno, pheno, and covar (if applicable)
        # filter on shared snps across ancestries
        tmp_geno = tmp_geno[common_ind_id, :][:, common_snp_id]

        # flip genotypes for bed files starting second ancestry
        if idx > 0 and len(flip_idx[idx - 1]) != 0:
            tmp_geno[:, flip_idx[idx - 1]] = 2 - tmp_geno[:, flip_idx[idx - 1]]

        # swap pheno and covar rows order to match fam/bed file, and then select the
        # values for future fine-mapping
        common_pheno_id = tmp_fam[f"phenoIDX_{idx + 1}"].values
        tmp_pheno = tmp_pheno["pheno"].values[common_pheno_id]

        geno.append(tmp_geno)
        pheno.append(tmp_pheno)

        if tmp_covar is not None:
            # select the common individual for covar
            common_covar_id = tmp_fam[f"covarIDX_{idx + 1}"].values
            n_covar = tmp_covar.shape[1]
            tmp_covar = tmp_covar.iloc[common_covar_id, 2:n_covar].values
            covar.append(tmp_covar)

    if len(covar) == 0:
        data_covar = None
    else:
        data_covar = covar

    regular_data = io.CleanData(geno=geno, pheno=pheno, covar=data_covar)

    log.logger.info(
        f"Successfully prepare covariate and phenotype data with {geno[0].shape[1]} SNPs for {n_pop}"
        + " ancestries, and start fine-mapping using SuShiE.",
    )

    mega_data = None
    cv_data = None
    # when doing mega or cross validation, we need to regress out covariates first
    if mega or cv:
        cv_geno = copy.deepcopy(geno)
        cv_pheno = copy.deepcopy(pheno)
        if data_covar is not None:
            for idx in range(n_pop):
                cv_geno[idx], cv_pheno[idx] = utils.regress_covar(
                    geno[idx], pheno[idx], data_covar[idx], no_regress
                )

        if cv:
            cv_data = _prepare_cv(cv_geno, cv_pheno, cv_num, seed)

        # prepare mega dataset
        # it's possible that different ancestries have different number of covariates,
        # so we need to regress out first
        if mega:
            mega_geno = cv_geno[0]
            mega_pheno = cv_pheno[0]
            for idx in range(1, n_pop):
                mega_geno = jnp.append(mega_geno, cv_geno[idx], axis=0)
                mega_pheno = jnp.append(mega_pheno, cv_pheno[idx], axis=0)

            mega_data = io.CleanData(
                geno=[mega_geno],
                pheno=[mega_pheno],
                covar=None,
            )

    return snps, regular_data, mega_data, cv_data


def sushie_wrapper(
    data: io.CleanData,
    cv_data: Optional[List[io.CVData]],
    args: argparse.Namespace,
    snps: pd.DataFrame,
    meta: bool = False,
    mega: bool = False,
) -> None:
    """The wrapper function to run SuShiE in regular, meta, or mega.

    Args:
        data: The clean data for SuShiE inference.
        cv_data: The cross-validation dataset.
        args: The command line parameter input.
        snps: The SNP information.
        meta: The indicator whether to prepare datasets for meta SuShiE.
        mega: The indicator whether to prepare datasets for mega SuShiE.

    """

    n_pop = len(data.geno)

    if meta:
        output = f"{args.output}.meta"
    elif mega:
        output = f"{args.output}.mega"
    else:
        output = f"{args.output}.sushie"

    resid_var = None if mega is True else args.resid_var
    effect_var = None if mega is True else args.effect_var
    rho = None if mega is True else args.rho

    # keeps track of single-ancestry PIP to get meta-PIP
    pips = jnp.zeros((snps.shape[0], 1)) if meta else None
    result = []
    if meta:
        # if this is meta, run it ancestry by ancestry
        for idx in range(n_pop):
            if args.resid_var is None:
                resid_var = None
            else:
                resid_var = [args.resid_var[idx]]

            if args.effect_var is None:
                effect_var = None
            else:
                effect_var = [args.effect_var[idx]]

            if data.covar is None:
                covar = None
            else:
                covar = [data.covar[idx]]

            log.logger.info(f"Running Meta SuShiE on ancestry {idx + 1}.")

            tmp_result = infer.infer_sushie(
                [data.geno[idx]],
                [data.pheno[idx]],
                covar,
                L=args.L,
                no_scale=args.no_scale,
                no_regress=args.no_regress,
                no_update=args.no_update,
                pi=args.pi,
                resid_var=resid_var,
                effect_var=effect_var,
                rho=None,
                max_iter=args.max_iter,
                min_tol=args.min_tol,
                threshold=args.threshold,
                purity=args.purity,
                no_kl=args.no_kl,
                kl_threshold=args.kl_threshold,
            )

            pips = jnp.append(pips, tmp_result.pip[:, jnp.newaxis], axis=1)
            result.append(tmp_result)

        pips = jnp.delete(pips, 0, 1)
        pips = 1 - jnp.prod(1 - pips, axis=1)
    else:
        if mega:
            log.logger.info("Running Mega SuShiE.")

        tmp_result = infer.infer_sushie(
            data.geno,
            data.pheno,
            data.covar,
            L=args.L,
            no_scale=args.no_scale,
            no_regress=args.no_regress,
            no_update=args.no_update,
            pi=args.pi,
            resid_var=resid_var,
            effect_var=effect_var,
            rho=rho,
            max_iter=args.max_iter,
            min_tol=args.min_tol,
            threshold=args.threshold,
            purity=args.purity,
            no_kl=args.no_kl,
            kl_threshold=args.kl_threshold,
        )
        result.append(tmp_result)

    io.output_cs(
        result, pips, snps, output, args.trait, args.compress, meta=meta, mega=mega
    )
    io.output_weights(
        result, pips, snps, output, args.trait, args.compress, meta=meta, mega=mega
    )

    if args.numpy:
        io.output_numpy(result, snps, output)

    if args.alphas:
        io.output_alphas(
            result, snps, output, args.trait, args.compress, meta=meta, mega=mega
        )

    if not (mega or meta):
        io.output_corr(result, output, args.trait, args.compress)

        if args.her:
            io.output_her(result, data, output, args.trait, args.compress)

        if args.cv:
            log.logger.info(f"Running {args.cv_num}-fold cross validation.")
            cv_res = _run_cv(args, cv_data)
            sample_size = [idx.shape[0] for idx in data.geno]
            io.output_cv(cv_res, sample_size, output, args.trait, args.compress)

    return None


def run_finemap(args):
    """The umbrella function to run SuShiE.

    Args:
        args: The command line parameter input.

    """

    try:
        if args.jax_precision == 64:
            config.update("jax_enable_x64", True)

        config.update("jax_platform_name", args.platform)

        geno_path, geno_func = parameter_check(args)

        rawData = io.read_data(args.pheno, args.covar, geno_path, geno_func)

        snps, regular_data, mega_data, cv_data = process_raw(
            rawData, args.no_regress, args.mega, args.cv, args.cv_num, args.seed
        )

        sushie_wrapper(regular_data, cv_data, args, snps, meta=False, mega=False)

        # if only one ancestry, need to run mega or meta
        n_pop = len(regular_data.geno)
        if n_pop != 1:
            if args.meta:
                sushie_wrapper(regular_data, None, args, snps, meta=True, mega=False)

            if args.mega:
                sushie_wrapper(mega_data, None, args, snps, meta=False, mega=True)

    except Exception as err:
        import traceback

        print(
            "".join(
                traceback.format_exception(
                    etype=type(err), value=err, tb=err.__traceback__
                )
            )
        )
        log.logger.error(err)

    finally:
        log.logger.info(
            "Finished SuShiE fine-mapping. Thanks for using our software."
            + " For bug reporting, suggestions, and comments, please go to https://github.com/mancusolab/sushie.",
        )
    return 0


def _get_command_string(args):
    base = f"sushie {args[0]}{os.linesep}"
    rest = args[1:]
    rest_strs = []
    needs_tab = True
    for cmd in rest:
        if "-" == cmd[0]:
            if cmd in ["--quiet", "-q", "--verbose", "-v"]:
                rest_strs.append(f"\t{cmd}{os.linesep}")
                needs_tab = True
            else:
                rest_strs.append(f"\t{cmd}")
                needs_tab = False
        else:
            if needs_tab:
                rest_strs.append(f"\t{cmd}{os.linesep}")
                needs_tab = True
            else:
                rest_strs.append(f" {cmd}{os.linesep}")
                needs_tab = True

    return base + "".join(rest_strs) + os.linesep


def _drop_na_subjects(rawData: io.RawData) -> Tuple[io.RawData, int]:
    _, fam, bed, pheno, covar = rawData

    del_idx = jnp.array([], dtype=int)

    val = jnp.array(pheno["pheno"])
    del_idx = jnp.append(del_idx, jnp.where(jnp.isnan(val))[0])
    del_idx = jnp.append(del_idx, jnp.where(jnp.isinf(val))[0])

    if covar is not None:
        val = jnp.array(covar.drop(columns="iid"))
        del_idx = jnp.append(del_idx, jnp.where(jnp.isinf(val).any(axis=1))[0])
        del_idx = jnp.append(del_idx, jnp.where(jnp.isnan(val).any(axis=1))[0])

    fam = fam.drop(del_idx)
    pheno = pheno.drop(del_idx)
    bed = jnp.delete(bed, del_idx, 0)

    if covar is not None:
        covar = covar.drop(del_idx)

    rawData = rawData._replace(
        fam=fam,
        pheno=pheno,
        bed=bed,
        covar=covar,
    )

    return rawData, len(del_idx)


def _impute_geno(rawData: io.RawData) -> Tuple[io.RawData, int, int]:
    bim, _, bed, _, _ = rawData

    # if we observe SNPs have nan value for all participants (although not likely), drop them
    del_idx = jnp.array([], dtype=int)
    del_idx = jnp.append(del_idx, jnp.where(jnp.isnan(bed).all(axis=0))[0])

    bim = bim.drop(del_idx)
    bed = jnp.delete(bed, del_idx, 1)

    # if we observe SNPs that partially have nan value, impute them with column mean
    col_mean = jnp.nanmean(bed, axis=0)
    imp_idx = jnp.where(jnp.isnan(bed))

    # column is the SNP index
    if len(imp_idx[1]) != 0:
        bed = bed.at[imp_idx].set(jnp.take(col_mean, imp_idx[1]))

    rawData = rawData._replace(
        bim=bim,
        bed=bed,
    )

    return rawData, len(del_idx), len(jnp.unique(imp_idx[1]))


def _reset_idx(rawData: io.RawData, idx: int) -> io.RawData:
    bim, fam, _, pheno, covar = rawData

    bim = (
        bim.reset_index(drop=True)
        .reset_index()
        .rename(
            columns={
                "index": f"bimIDX_{idx + 1}",
                "pos": f"pos_{idx + 1}",
                "a0": f"a0_{idx + 1}",
                "a1": f"a1_{idx + 1}",
            }
        )
    )

    fam = (
        fam.reset_index(drop=True)
        .reset_index()
        .rename(columns={"index": f"famIDX_{idx + 1}"})
    )
    pheno = (
        pheno.reset_index(drop=True)
        .reset_index()
        .rename(columns={"index": f"phenoIDX_{idx + 1}"})
    )
    if covar is not None:
        covar = (
            covar.reset_index(drop=True)
            .reset_index()
            .rename(columns={"index": f"covarIDX_{idx + 1}"})
        )

    rawData = rawData._replace(
        bim=bim,
        fam=fam,
        pheno=pheno,
        covar=covar,
    )

    return rawData


def _filter_common_ind(rawData: io.RawData, idx: int) -> io.RawData:
    _, fam, _, pheno, covar = rawData

    common_fam = pd.merge(
        fam, pheno[[f"phenoIDX_{idx + 1}", "iid"]], how="inner", on=["iid"]
    )
    if covar is not None:
        # match fam id and covar id
        common_fam = pd.merge(
            common_fam, covar[[f"covarIDX_{idx + 1}", "iid"]], how="inner", on=["iid"]
        )
    rawData = rawData._replace(fam=common_fam)

    return rawData


def _allele_check(
    baseA0: pd.Series,
    baseA1: pd.Series,
    compareA0: pd.Series,
    compareA1: pd.Series,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    correct = jnp.array(
        ((baseA0 == compareA0) * 1) * ((baseA1 == compareA1) * 1), dtype=int
    )
    flipped = jnp.array(
        ((baseA0 == compareA1) * 1) * ((baseA1 == compareA0) * 1), dtype=int
    )
    correct_idx = jnp.where(correct == 1)[0]
    flipped_idx = jnp.where(flipped == 1)[0]
    wrong_idx = jnp.where((correct + flipped) == 0)[0]

    return correct_idx, flipped_idx, wrong_idx


def _prepare_cv(
    geno: List[jnp.ndarray], pheno: List[jnp.ndarray], cv_num: int, seed: int
) -> List[io.CVData]:
    rng_key = random.PRNGKey(seed)
    n_pop = len(geno)

    geno_split = []
    pheno_split = []
    # shuffle the data first
    for idx in range(n_pop):
        tmp_n = geno[idx].shape[0]
        rng_key, c_key = random.split(rng_key, 2)
        shuffled_index = random.choice(c_key, tmp_n, (tmp_n,), replace=False)
        geno[idx] = geno[idx][shuffled_index]
        pheno[idx] = pheno[idx][shuffled_index]
        geno_split.append(jnp.array_split(geno[idx], cv_num))
        pheno_split.append(jnp.array_split(pheno[idx], cv_num))

    cv_data = []
    for cv in range(cv_num):
        train_geno = []
        train_pheno = []
        valid_geno = []
        valid_pheno = []
        train_index = jnp.delete(jnp.arange(5), cv).tolist()

        # make the training and test for each population separately
        # because sample size may be different
        for idx in range(n_pop):
            valid_geno.append(geno_split[idx][cv])
            valid_pheno.append(pheno_split[idx][cv])
            train_geno.append(
                jnp.concatenate([geno_split[idx][jdx] for jdx in train_index])
            )
            train_pheno.append(
                jnp.concatenate([pheno_split[idx][jdx] for jdx in train_index])
            )

        tmp_cv_data = io.CVData(
            train_geno=train_geno,
            train_pheno=train_pheno,
            valid_geno=valid_geno,
            valid_pheno=valid_pheno,
        )

        cv_data.append(tmp_cv_data)

    return cv_data


def _run_cv(args, cv_data) -> List[List[jnp.ndarray]]:
    n_pop = len(cv_data[0].train_geno)
    # create a list to store future estimated y value
    est_y = [jnp.array([])] * n_pop
    ori_y = [jnp.array([])] * n_pop
    for jdx in range(args.cv_num):
        cv_result = infer.infer_sushie(
            cv_data[jdx].train_geno,
            cv_data[jdx].train_pheno,
            None,
            L=args.L,
            no_scale=args.no_scale,
            no_regress=args.no_regress,
            no_update=args.no_update,
            pi=args.pi,
            resid_var=args.resid_var,
            effect_var=args.effect_var,
            rho=args.rho,
            max_iter=args.max_iter,
            min_tol=args.min_tol,
            threshold=args.threshold,
            purity=args.purity,
        )

        total_weight = jnp.sum(cv_result.posteriors.post_mean, axis=0)
        for idx in range(n_pop):
            tmp_cv_weight = total_weight[:, idx]
            est_y[idx] = jnp.append(
                est_y[idx], cv_data[jdx].valid_geno[idx] @ tmp_cv_weight
            )
            ori_y[idx] = jnp.append(ori_y[idx], cv_data[jdx].valid_pheno[idx])

    cv_res = []
    for idx in range(n_pop):
        _, adj_r2, p_value = utils.ols(
            est_y[idx][:, jnp.newaxis], ori_y[idx][:, jnp.newaxis]
        )
        cv_res.append([adj_r2[0], p_value[1][0]])

    return cv_res


def build_finemap_parser(subp):
    # add imputation parser
    finemap = subp.add_parser(
        "finemap",
        description=(
            "Perform SNP fine-mapping of gene expression on",
            " individual genotype and phenotype data using SuShiE.",
        ),
    )
    finemap.add_argument(
        "--pheno",
        nargs="+",
        type=str,
        required=True,
        help=(
            "Phenotype data. It has to be a tsv file that contains at least two",
            " columns where the first column is subject ID and",
            " the second column is the continuous phenotypic value. It can be a compressed file (e.g., tsv.gz).",
            " It is okay to have additional columns, but only the first two columns will be used."
            " No headers. Use 'space' to separate ancestries if more than two.",
            " SuShiE currently only fine-maps on continuous data.",
        ),
    )

    # main arguments
    finemap.add_argument(
        "--plink",
        nargs="+",
        type=str,
        default=None,
        help=(
            "Genotype data in plink 1 format. The plink triplet (bed, bim, and fam) should be",
            " in the same folder with the same prefix.",
            " Use 'space' to separate ancestries if more than two.",
            " Keep the same ancestry order as phenotype's.",
            " SuShiE currently does not take plink 2 format.",
        ),
    )

    finemap.add_argument(
        "--vcf",
        nargs="+",
        type=str,
        default=None,
        help=(
            "Genotype data in vcf format. Use 'space' to separate ancestries if more than two.",
            " Keep the same ancestry order as phenotype's.",
        ),
    )

    finemap.add_argument(
        "--bgen",
        nargs="+",
        type=str,
        default=None,
        help=(
            "Genotype data in bgen 1.3 format. Use 'space' to separate ancestries if more than two.",
            " Keep the same ancestry order as phenotype's.",
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
            " It has to be a tsv file that contains at least two columns where the",
            " first column is the subject ID. It can be a compressed file (e.g., tsv.gz)",
            " All the columns will be counted. Use 'space' to separate ancestries if more than two.",
            " Keep the same ancestry order as phenotype's.",
            " Pre-converting the categorical covariates into dummy variables is required. No headers.",
            "If your categorical covariate has n levels, make sure the dummy variables have n-1 columns.",
        ),
    )

    finemap.add_argument(
        "--L",
        default=5,
        type=int,
        help=(
            "Integer number of shared effects pre-specified.",
            " Default is 5. Larger number may cause slow inference.",
        ),
    )

    # fine-map prior options
    finemap.add_argument(
        "--pi",
        default=None,
        type=float,
        help=(
            "Prior probability for each SNP to be causal.",
            " Default is 1/p where p is the number of SNPs in the region.",
            " It is the fixed across all ancestries.",
        ),
    )

    finemap.add_argument(
        "--resid_var",
        nargs="+",
        default=None,
        type=float,
        help=(
            "Specify the prior for the residual variance for ancestries. Default is 1e-3 for each ancestry.",
            " Values have to be positive. Use 'space' to separate ancestries if more than two.",
        ),
    )

    finemap.add_argument(
        "--effect_var",
        nargs="+",
        default=None,
        type=float,
        help=(
            "Specify the prior for the causal effect size variance for ancestries. Default is 1e-3 for each ancestry.",
            " Values have to be positive. Use 'space' to separate ancestries if more than two.",
            " If --no_update is specified and --rho is not, specifying this parameter will",
            " only keep effect_var as prior through optimizations and update rho.",
            " If --effect_covar, --rho, and --no_update all three are specified, both --effect_covar and --rho",
            " will be fixed as prior through optimizations.",
            " If --no_update is specified, but neither --effect_covar nor --rho,",
            " both --effect_covar and --rho will be fixed as default prior value through optimizations.",
        ),
    )

    finemap.add_argument(
        "--rho",
        nargs="+",
        default=None,
        type=float,
        help=(
            "Specify the prior for the effect correlation for ancestries. Default is 0.8 for each pair of ancestries.",
            " Use 'space' to separate ancestries if more than two. Each rho has to be a float number between -1 and 1.",
            " If there are N > 2 ancestries, X = choose(N, 2) is required.",
            " The rho order has to be rho(1,2), ..., rho(1, N), rho(2,3), ..., rho(N-1. N).",
            " If --no_update is specified and --effect_covar is not, specifying this parameter will",
            " only fix rho as prior through optimizations and update effect_covar.",
            " If --effect_covar, --rho, and --no_update all three are specified, both --effect_covar and --rho",
            " will be fixed as prior through optimizations.",
            " If --no_update is specified, but neither --effect_covar nor --rho,",
            " both --effect_covar and --rho will be fixed as default prior value through optimizations.",
        ),
    )

    # fine-map inference options
    finemap.add_argument(
        "--no_scale",
        default=False,
        action="store_true",
        help=(
            "Indicator to scale the genotype and phenotype data by standard deviation.",
            " Default is False (to scale)."
            " Specify --no_scale will store 'True' value, and may cause different inference.",
        ),
    )

    finemap.add_argument(
        "--no_regress",
        default=False,
        action="store_true",
        help=(
            "Indicator to regress the covariates on each SNP. Default is False (to regress).",
            " Specify --no_regress will store 'True' value.",
            " It may slightly slow the inference, but can be more accurate.",
        ),
    )

    finemap.add_argument(
        "--no_update",
        default=False,
        action="store_true",
        help=(
            "Indicator to update effect covariance prior before running single effect regression.",
            " Default is False (to update).",
            " Specify --no_update will store 'True' value. The updating algorithm is similar to EM algorithm",
            " that computes the prior covariance conditioned on other parameters.",
            " See the manuscript for more information.",
        ),
    )

    finemap.add_argument(
        "--max_iter",
        default=500,
        type=int,
        help=(
            "Maximum iterations for the optimization. Default is 500.",
            " Larger number may slow the inference while smaller may cause different inference.",
        ),
    )

    finemap.add_argument(
        "--min_tol",
        default=1e-4,
        type=float,
        help=(
            "Minimum tolerance for the convergence. Default is 1e-5.",
            " Smaller number may slow the inference while larger may cause different inference.",
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
        "--no_kl",
        default=False,
        action="store_true",
        help=(
            "Indicator to use KL divergence as alternative credible set pruning threshold in addition to purity.",
            " Default is False. Specify --no_kl will store 'True' value and will not use KL divergence as",
            " extra threshold.",
        ),
    )

    finemap.add_argument(
        "--kl_threshold",
        default=5.0,
        type=float,
        help=(
            "Specify the KL divergence threshold for credible sets to be output. Default is 5.",
            " It has to be a positive number.",
        ),
    )

    # I/O option
    finemap.add_argument(
        "--meta",
        default=False,
        action="store_true",
        help=(
            "Indicator to perform single-ancestry SuShiE followed by meta analysis of the results.",
            " Default is False. Specify --meta will store 'True' value and increase running time.",
            " Specifying one ancestry in phenotype and genotype parameter will ignore --meta.",
        ),
    )

    finemap.add_argument(
        "--mega",
        default=False,
        action="store_true",
        help=(
            "Indicator to perform mega SuShiE that run single-ancestry SuShiE on",
            " genotype and phenotype data that is row-wise stacked across ancestries.",
            " Default is False. Specify --mega will store 'True' value and increase running time.",
            " Specifying one ancestry in phenotype and genotype parameter will ignore --mega.",
        ),
    )

    finemap.add_argument(
        "--her",
        default=False,
        action="store_true",
        help=(
            "Indicator to perform heritability (h2g) analysis using limix. Default is False.",
            " Specify --her will store 'True' value and increase running time.",
            " It estimates h2g with two definitions. One is with variance of fixed terms (original limix definition),",
            " and the other is without variance of fixed terms (gcta definition).",
            " It also estimates these two definitions' h2g using using all genotypes,",
            " and using only SNPs in the credible sets.",
        ),
    )

    finemap.add_argument(
        "--cv",
        default=False,
        action="store_true",
        help=(
            "Indicator to perform cross validation (CV) and output CV results (adjusted r-squared and its p-value)",
            " for future FUSION pipline. Default is False. ",
            " Specify --cv will store 'True' value and increase running time.",
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
            " It has to be positive integer number.",
        ),
    )

    finemap.add_argument(
        "--alphas",
        default=False,
        action="store_true",
        help=(
            "Indicator to output all the cs (alphas) results before pruning for purity",
            " including PIPs, alphas, whether in cs, across all L.",
            " Default is False. Specify --alphas will store 'True' value and increase running time.",
        ),
    )

    finemap.add_argument(
        "--numpy",
        default=False,
        action="store_true",
        help=(
            "Indicator to output all the results in *.npy file.",
            " Default is False. Specify --numpy will store 'True' value and increase running time.",
            " *.npy file contains all the inference results including credible sets, pips, priors and posteriors",
            " for your own post-hoc analysis.",
        ),
    )

    finemap.add_argument(
        "--trait",
        default="Trait",
        help=(
            "Trait, tissue, gene name of the phenotype for better indexing in post-hoc analysis. Default is 'Trait'.",
        ),
    )

    # misc options
    finemap.add_argument(
        "--quiet",
        default=False,
        action="store_true",
        help="Indicator to not print message to console. Default is False. Specify --numpy will store 'True' value.",
    )
    finemap.add_argument(
        "--verbose",
        default=False,
        action="store_true",
        help=(
            "Indicator to include debug information in the log. Default is False.",
            " Specify --numpy will store 'True' value.",
        ),
    )

    finemap.add_argument(
        "--compress",
        default=False,
        action="store_true",
        help=(
            "Indicator to compress all output tsv files in tsv.gz.",
            " Default is False. Specify --compress will store 'True' value to save disk space.",
            " This command will not compress *.npy files.",
        ),
    )

    finemap.add_argument(
        "--platform",
        default="cpu",
        type=str,
        choices=["cpu", "gpu", "tpu"],
        help=(
            "Indicator for the JAX platform. It has to be 'cpu', 'gpu', or 'tpu'. Default is cpu.",
        ),
    )

    finemap.add_argument(
        "--jax_precision",
        default=64,
        type=int,
        choices=[32, 64],
        help=(
            "Indicator for the JAX precision: 64-bit or 32-bit.",
            " Default is 64-bit. Choose 32-bit may cause 'elbo decreases' warning.",
        ),
    )

    finemap.add_argument(
        "--output",
        default="sushie_finemap",
        help=("Prefix for output files. Default is 'sushie_finemap'.",),
    )

    return finemap


def _main(argsv):
    # setup main parser
    argp = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subp = argp.add_subparsers(
        help="Subcommands: finemap to perform gene expression fine-mapping using SuShiE"
    )

    finemap = build_finemap_parser(subp)
    finemap.set_defaults(func=run_finemap)

    # parse arguments
    args = argp.parse_args(argsv)

    cmd_str = _get_command_string(argsv)

    version = metadata.version("sushie")

    masthead = "===================================" + os.linesep
    masthead += f"             SuShiE v{version}             " + os.linesep
    masthead += "===================================" + os.linesep

    # setup logging
    log_format = "[%(asctime)s - %(levelname)s] %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    if args.verbose:
        log.logger.setLevel(logging.DEBUG)
    else:
        log.logger.setLevel(logging.INFO)
    fmt = logging.Formatter(fmt=log_format, datefmt=date_format)
    log.logger.propagate = False

    # write to stdout unless quiet is set
    if not args.quiet:
        sys.stdout.write(masthead)
        sys.stdout.write(cmd_str)
        sys.stdout.write("Starting log..." + os.linesep)
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(fmt)
        log.logger.addHandler(stdout_handler)

    # setup log file, but write PLINK-style command first
    disk_log_stream = open(f"{args.output}.log", "w")
    disk_log_stream.write(masthead)
    disk_log_stream.write(cmd_str)
    disk_log_stream.write("Starting log..." + os.linesep)

    disk_handler = logging.StreamHandler(disk_log_stream)
    disk_handler.setFormatter(fmt)
    log.logger.addHandler(disk_handler)

    # launch finemap
    args.func(args)

    return 0


def run_cli():
    return _main(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
