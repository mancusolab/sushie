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


def _filter_maf(rawData: io.RawData, maf: float) -> Tuple[io.RawData, int]:
    bim, _, bed, _, _ = rawData

    old_num = bim.shape[0]

    # calculate maf
    snp_maf = jnp.mean(bed, axis=0) / 2
    snp_maf = jnp.where(snp_maf > 0.5, 1 - snp_maf, snp_maf)

    (sel_idx,) = jnp.where(snp_maf >= maf)

    bim = bim.iloc[sel_idx, :]
    bed = bed[:, sel_idx]

    rawData = rawData._replace(
        bim=bim,
        bed=bed,
    )

    del_num = old_num - len(sel_idx)

    return rawData, del_num


def _keep_file_subjects(
    rawData: io.RawData, keep_subject: List[str]
) -> Tuple[io.RawData, int, int]:
    _, fam, bed, pheno, covar = rawData

    old_fam_num = fam.shape[0]
    old_pheno_num = pheno.shape[0]

    bed = bed[fam.iid.isin(keep_subject).values, :]
    fam = fam.loc[fam.iid.isin(keep_subject)]
    pheno = pheno.loc[pheno.iid.isin(keep_subject)]

    if covar is not None:
        covar = covar.loc[covar.iid.isin(keep_subject)]

    del_fam_num = old_fam_num - fam.shape[0]
    del_pheno_num = old_pheno_num - pheno.shape[0]

    rawData = rawData._replace(
        fam=fam,
        pheno=pheno,
        bed=bed,
        covar=covar,
    )

    return rawData, del_fam_num, del_pheno_num


def _drop_na_subjects(rawData: io.RawData) -> Tuple[io.RawData, int]:
    _, fam, bed, pheno, covar = rawData

    val = jnp.array(pheno["pheno"])
    del_idx = jnp.logical_or(jnp.isnan(val), jnp.isinf(val))

    if covar is not None:
        val = jnp.array(covar.drop(columns="iid"))
        covar_del = jnp.logical_or(
            jnp.any(jnp.isinf(val), axis=1), jnp.any(jnp.isnan(val), axis=1)
        )
        del_idx = jnp.logical_or(del_idx, covar_del)

    (drop_idx,) = jnp.where(del_idx)
    fam = fam.drop(drop_idx)
    pheno = pheno.drop(drop_idx)
    bed = jnp.delete(bed, drop_idx, 0)

    if covar is not None:
        covar = covar.drop(drop_idx)

    rawData = rawData._replace(
        fam=fam,
        pheno=pheno,
        bed=bed,
        covar=covar,
    )

    return rawData, len(drop_idx)


def _remove_dup_geno(rawData: io.RawData) -> Tuple[io.RawData, int]:
    bim, _, bed, _, _ = rawData

    (dup_idx,) = jnp.where(bim.snp.duplicated().values)

    bim = bim.drop(dup_idx)
    bed = jnp.delete(bed, dup_idx, 1)

    rawData = rawData._replace(
        bim=bim,
        bed=bed,
    )

    return rawData, len(dup_idx)


def _impute_geno(rawData: io.RawData) -> Tuple[io.RawData, int, int]:
    bim, _, bed, _, _ = rawData

    # if we observe SNPs have nan value for all participants (although not likely), drop them
    (del_idx,) = jnp.where(jnp.all(jnp.isnan(bed), axis=0))

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
    (correct_idx,) = jnp.where(correct == 1)
    (flipped_idx,) = jnp.where(flipped == 1)
    (wrong_idx,) = jnp.where((correct + flipped) == 0)

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
            train_geno.append(
                jnp.concatenate([geno_split[idx][jdx] for jdx in train_index])
            )
            valid_pheno_split = utils.rint(pheno_split[idx][cv])
            valid_pheno.append(valid_pheno_split)
            train_pheno_split = utils.rint(
                jnp.concatenate([pheno_split[idx][jdx] for jdx in train_index])
            )
            train_pheno.append(train_pheno_split)

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
            max_select=args.max_select,
            seed=args.seed,
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


def parameter_check(
    args,
) -> Tuple[int, pd.DataFrame, List[str], List[str], Callable]:
    """The function to process raw phenotype, genotype, covariates data across ancestries.

    Args:
        args: The command line parameter input.

    Returns:
        :py:obj:`Tuple[int, pd.DataFrame, List[str], Callable]`:
            A tuple of
                #. an integer to indicate how many ancestries,
                #. a DataFrame that contains ancestry index (can be none),
                #. a list that contains subject ID that fine-mapping performs on.
                #. a list of genotype data paths (:py:obj:`List[str]`),
                #. genotype read-in function (:py:obj:`Callable`).

    """
    if args.ancestry_index is not None:
        ancestry_index = pd.read_csv(args.ancestry_index[0], header=None, sep="\t")
        old_pt = ancestry_index.shape[0]
        ancestry_index = ancestry_index.drop_duplicates()

        if old_pt != ancestry_index.shape[0]:
            log.logger.debug(
                f"Index file has {old_pt - ancestry_index.shape[0]} duplicated subjects."
            )

        if ancestry_index[0].duplicated().sum() != 0:
            raise ValueError(
                "The ancestry index file contains subjects with multiple ancestry index. Check the source."
            )

        n_pop = len(ancestry_index[1].unique())
        index_check = jnp.all(
            jnp.array(ancestry_index[1].unique()).sort() == (jnp.arange(n_pop) + 1)
        )

        if not index_check:
            raise ValueError(
                "The ancestry index doesn't start from 1 continuously to the total number of ancestry."
                + f" Check {args.ancestry_index}."
            )

        if len(args.pheno) > 1:
            raise ValueError(
                "Multiple phenotype files are detected. Expectation is one when --ancestry_index is specified."
            )

        log.logger.debug(
            "Detect ancestry index file, so it expects to have single phenotype, genotype, and covariates files"
        )

    else:
        ancestry_index = pd.DataFrame()
        n_pop = len(args.pheno)

    name_ancestry = "ancestry" if n_pop == 1 else "ancestries"

    log.logger.info(f"Detect phenotypes for {args.trait} from {n_pop} {name_ancestry}.")

    n_geno = (
        int(args.plink is not None)
        + int(args.vcf is not None)
        + int(args.bgen is not None)
    )

    if n_geno > 1:
        log.logger.info(
            f"Detect {n_geno} genotypes, will only use one genotypes in the order of 'plink, vcf, and bgen'"
        )

    # decide genotype data
    if args.plink is not None:
        if args.ancestry_index is not None:
            if len(args.plink) > 1:
                raise ValueError(
                    "Multiple plink files are detected. Expectation is one when --ancestry_index is specified."
                )
        else:
            if len(args.plink) != n_pop:
                raise ValueError(
                    "The numbers of ancestries in plink geno and pheno data does not match. Check the source."
                )

        log.logger.info("Detect genotype data in plink format.")
        geno_path = args.plink
        geno_func = io.read_triplet
    elif args.vcf is not None:
        if args.ancestry_index is not None:
            if len(args.vcf) > 1:
                raise ValueError(
                    "Multiple vcf files are detected. Expectation is one when --ancestry_index is specified."
                )
        else:
            if len(args.vcf) != n_pop:
                raise ValueError(
                    "The numbers of ancestries in vcf geno and pheno data does not match. Check the source."
                )
        log.logger.info("Detect genotype data in vcf format.")
        geno_path = args.vcf
        geno_func = io.read_vcf
    elif args.bgen is not None:
        if args.ancestry_index is not None:
            if len(args.bgen) > 1:
                raise ValueError(
                    "Multiple bgen files are detected. Expectation is one when --ancestry_index is specified."
                )
        else:
            if len(args.bgen) != n_pop:
                raise ValueError(
                    "The numbers of ancestries in bgen geno and pheno data does not match. Check the source."
                )

        log.logger.info("Detect genotype data in bgen format.")
        geno_path = args.bgen
        geno_func = io.read_bgen
    else:
        raise ValueError(
            "No genotype data specified in either plink, vcf, or bgen format. Check the source."
        )

    if args.covar is not None:
        if args.ancestry_index is not None:
            if len(args.covar) > 1:
                raise ValueError(
                    "Multiple covariates files are detected. Expectation is one when --ancestry_index is specified."
                )
        else:
            if len(args.covar) != n_pop:
                raise ValueError(
                    "The number of covariates data does not match geno data."
                )
        log.logger.info("Detect covariates data.")
    else:
        log.logger.info("No covariates detected for this analysis.")

    keep_subject = []
    if args.keep is not None:
        log.logger.info(
            "Detect keep subject file. The inference only performs on the subjects listed in the file."
        )
        df_keep = pd.read_csv(args.keep[0], header=None, sep="\t")[[0]]
        if df_keep.shape[0] == 0:
            raise ValueError(
                "No subjects are listed in the keep subject file. Check the source."
            )
        old_pt = df_keep.shape[0]
        df_keep = df_keep.drop_duplicates()

        if old_pt != df_keep.shape[0]:
            log.logger.debug(
                f"The keep subject file has {old_pt - df_keep.shape[0]} duplicated subjects."
            )
        keep_subject = df_keep[0].values.tolist()

    if args.seed <= 0:
        raise ValueError(
            "The seed specified for randomization is invalid. Choose a positive integer."
        )

    if args.cv:
        if args.cv_num <= 1:
            raise ValueError(
                "The number of folds in cross validation is invalid."
                + " Choose some number greater than 1.",
            )

    if args.maf <= 0 or args.maf > 0.5:
        raise ValueError(
            "The minor allele frequency (MAF) has to be between 0 (exclusive) and 0.5 (inclusive)."
            + " Choose a valid float."
        )

    if (args.meta or args.mega) and n_pop == 1:
        log.logger.info(
            "The number of ancestry is 1, but --meta or --mega is specified. Will skip meta or mega SuSiE."
        )

    return n_pop, ancestry_index, keep_subject, geno_path, geno_func


def process_raw(
    rawData: List[io.RawData],
    keep_subject: List[str],
    maf: float,
    rint: bool,
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
    """The function to process raw phenotype, genotype, covariates data across ancestries.

    Args:
        rawData: Raw data for phenotypes, genotypes, covariates across ancestries.
        keep_subject: The DataFrame that contains subject ID that fine-mapping performs on.
        maf: The minor allele frequency threshold to filter the genotypes.
        rint: The indicator whether to perform rank inverse normalization on each phenotype data.
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

        # remove subjects that are not in the keep file
        if len(keep_subject) != 0:
            old_fam_num = rawData[idx].fam.shape[0]
            old_pheno_num = rawData[idx].pheno.shape[0]

            (
                rawData[idx],
                del_fam_num,
                del_pheno_num,
            ) = _keep_file_subjects(rawData[idx], keep_subject)

            if len(rawData[idx].fam) == 0:
                raise ValueError(
                    f"Ancestry {idx + 1}: No subjects in the genotype data are listed in the keep file."
                    + " Check the source."
                )

            if len(rawData[idx].pheno) == 0:
                raise ValueError(
                    f"Ancestry {idx + 1}: No subjects in the pheno data are listed in the keep file."
                    + " Check the source."
                )

            if del_fam_num != 0:
                log.logger.debug(
                    f"Ancestry {idx + 1}: Drop {del_fam_num} out of {old_fam_num} subjects in the genotype data"
                    + " because these subjects are not listed in the subject keep file."
                )

            if del_pheno_num != 0:
                log.logger.debug(
                    f"Ancestry {idx + 1}: Drop {del_pheno_num} out of {old_pheno_num} subjects in the phenotype data"
                    + " because these subjects are not listed in the subject keep file."
                )

        # remove NA/inf value for subjects across phenotype or covariates data
        old_subject_num = rawData[idx].fam.shape[0]
        rawData[idx], del_num = _drop_na_subjects(rawData[idx])

        if del_num != 0:
            log.logger.debug(
                f"Ancestry {idx + 1}: Drop {del_num} out of {old_subject_num} subjects because of INF or NAN value"
                + " in either phenotype or covariates data."
            )

        if del_num == old_subject_num:
            raise ValueError(
                f"Ancestry {idx + 1}: All subjects have INF or NAN value in either phenotype or covariates data."
                + " Check the source."
            )

        old_snp_num = rawData[idx].bim.shape[0]
        # remove duplicates SNPs based on rsid even though we suggest users to do some QC on this
        rawData[idx], del_num = _remove_dup_geno(rawData[idx])

        if del_num != 0:
            log.logger.debug(
                f"Ancestry {idx + 1}: Drop {del_num} out of {old_snp_num} SNPs because of duplicates in the rs ID"
                + " in genotype data."
            )

        old_snp_num = rawData[idx].bim.shape[0]
        # impute genotype data even though we suggest users to impute the genotypes beforehand
        rawData[idx], del_num, imp_num = _impute_geno(rawData[idx])

        if del_num == old_snp_num:
            raise ValueError(
                f"Ancestry {idx + 1}: All SNPs have INF or NAN value in genotype data. Check the source."
            )

        if del_num != 0:
            log.logger.debug(
                f"Ancestry {idx + 1}: Drop {del_num} out of {old_snp_num} SNPs because all subjects have NAN value"
                + " in genotype data."
            )

        if imp_num != 0:
            log.logger.debug(
                f"Ancestry {idx + 1}: Impute {imp_num} out of {old_snp_num} SNPs with NAN value based on allele"
                + " frequency."
            )

        old_snp_num = rawData[idx].bim.shape[0]
        # remove SNPs that cannot pass MAF threshold
        rawData[idx], del_num = _filter_maf(rawData[idx], maf)

        if del_num == old_snp_num:
            raise ValueError(
                f"Ancestry {idx + 1}: All SNPs cannot pass the MAF threshold at {maf}."
            )

        if del_num != 0:
            log.logger.debug(
                f"Ancestry {idx + 1}: Drop {del_num} out of {old_snp_num} SNPs because of maf threshold at {maf}."
            )

        if rawData[idx].bim.shape[0] == 0:
            raise ValueError(
                f"Ancestry {idx + 1}: no SNPs left after QC. Check the source."
            )

        # reset index and add index column to all dataset for future inter-ancestry or inter-dataset processing
        rawData[idx] = _reset_idx(rawData[idx], idx)

        # find common individuals across geno, pheno, and covar within an ancestry
        rawData[idx] = _filter_common_ind(rawData[idx], idx)

        if rawData[idx].fam.shape[0] == 0:
            raise ValueError(
                f"Ancestry {idx + 1}: No common individuals across phenotype, covariates,"
                + " genotype found. Check the source.",
            )
        else:
            log.logger.debug(
                f"Ancestry {idx + 1}: Found {rawData[idx].fam.shape[0]} common individuals"
                + " across phenotype, covariates, and genotype.",
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
        if snps.shape[0] == 0:
            raise ValueError("Ancestries have no common SNPs. Check the source.")
        # report how many snps we removed due to independent SNPs
        for idx in range(n_pop):
            snps_num_diff = rawData[idx].bim.shape[0] - snps.shape[0]
            log.logger.debug(
                f"Ancestry{idx + 1} has {snps_num_diff} independent SNPs and {snps.shape[0]}"
                + " common SNPs. Inference only performs on common SNPs.",
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
                log.logger.debug(
                    f"Ancestry{idx + 1} has {len(tmp_flip_idx)} flipped alleles from ancestry 1. Will flip these SNPs."
                )

            # save the index for future swapping
            flip_idx.append(tmp_flip_idx)

            if len(wrong_idx) != 0:
                snps = snps.drop(index=wrong_idx)
                log.logger.debug(
                    f"Ancestry{idx + 1} has {len(wrong_idx)} alleles that couldn't be flipped. Will remove these SNPs."
                )

            if snps.shape[0] == 0:
                raise ValueError(
                    f"Ancestry {idx + 1} has none of correct or flippable SNPs from ancestry 1. Check the source.",
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
    total_ind = 0
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
        total_ind += tmp_pheno.shape[0]
        geno.append(tmp_geno)

        if rint:
            tmp_pheno = utils.rint(tmp_pheno)

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

    name_ancestry = "ancestry" if n_pop == 1 else "ancestries"

    log.logger.info(
        f"Prepare {geno[0].shape[1]} SNPs for {total_ind} individuals from {n_pop} {name_ancestry} after"
        + " data cleaning. Specify --verbose for details.",
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

            # because it row-binds the phenotype data for each ancestry, we want to rint again
            mega_pheno = utils.rint(mega_pheno)
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
        method_type = "meta"
    elif mega:
        output = f"{args.output}.mega"
        method_type = "mega"
    else:
        output = f"{args.output}.sushie"
        method_type = "sushie"

    resid_var = None if mega is True else args.resid_var
    effect_var = None if mega is True else args.effect_var
    rho = None if mega is True else args.rho

    # padding will change the original data, make a copy for heritability
    heri_data = copy.deepcopy(data)

    # keeps track of single-ancestry PIP to get meta-PIP
    pips_all = []
    pips_cs = []
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

            log.logger.info(
                f"Start fine-mapping using SuSiE on ancestry {idx + 1} with {args.L} effects"
                + " because --meta is specified."
            )

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
                max_select=args.max_select,
                min_snps=args.min_snps,
                no_reorder=args.no_reorder,
                seed=args.seed,
            )
            pips_all.append(tmp_result.pip_all[:, jnp.newaxis])
            pips_cs.append(tmp_result.pip_cs[:, jnp.newaxis])
            result.append(tmp_result)

        pips_all = utils.make_pip(jnp.concatenate(pips_all, axis=1).T)
        pips_cs = utils.make_pip(jnp.concatenate(pips_cs, axis=1).T)
    else:
        # normal sushie and mega sushie can use the same wrapper function
        if mega:
            log.logger.info(
                f"Start fine-mapping using Mega SuSiE with {args.L} effects because --mega is specified."
            )
        else:
            log.logger.info(f"Start fine-mapping using SuShiE with {args.L} effects.")

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
            max_select=args.max_select,
            min_snps=args.min_snps,
            no_reorder=args.no_reorder,
            seed=args.seed,
        )
        result.append(tmp_result)

    pips = [pips_all, pips_cs] if meta else None

    io.output_cs(result, pips, snps, output, args.trait, args.compress, method_type)
    io.output_weights(
        result, pips, snps, output, args.trait, args.compress, method_type
    )

    if args.numpy:
        log.logger.info(
            "Save all the inference results in numpy file because --numpy is specified "
        )
        io.output_numpy(result, snps, output)

    if args.alphas:
        log.logger.info(
            "Save all credible set results before pruning as --alphas is specified "
        )

        io.output_alphas(
            result,
            snps,
            output,
            args.trait,
            args.compress,
            method_type,
            args.purity,
        )

    if not (mega or meta):
        io.output_corr(result, output, args.trait, args.compress)

        if args.her:
            log.logger.info("Save heritability analysis results as --her is specified")
            io.output_her(heri_data, output, args.trait, args.compress)

        if args.cv:
            log.logger.info(
                f"Start {args.cv_num}-fold cross validation as --cv is specified "
            )
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
            config.update("jax_default_matmul_precision", "highest")

        config.update("jax_platform_name", args.platform)

        n_pop, ancestry_index, keep_subject, geno_path, geno_func = parameter_check(
            args
        )

        rawData = io.read_data(
            n_pop,
            ancestry_index,
            args.pheno,
            args.covar,
            geno_path,
            geno_func,
        )

        snps, regular_data, mega_data, cv_data = process_raw(
            rawData,
            keep_subject,
            args.maf,
            args.rint,
            args.no_regress,
            args.mega,
            args.cv,
            args.cv_num,
            args.seed,
        )

        normal_data = copy.deepcopy(regular_data)
        sushie_wrapper(normal_data, cv_data, args, snps, meta=False, mega=False)

        # if only one ancestry, need to run mega or meta
        n_pop = len(regular_data.geno)
        if n_pop != 1:
            if args.meta:
                meta_data = copy.deepcopy(regular_data)
                sushie_wrapper(meta_data, None, args, snps, meta=True, mega=False)

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
            f"Fine-mapping finishes for {args.trait}, and thanks for using our software."
            + " For bug reporting, suggestions, and comments, please go to https://github.com/mancusolab/sushie.",
        )
    return 0


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
        "--ancestry-index",
        nargs=1,
        default=None,
        type=str,
        help=(
            "Single file that contains subject ID and their ancestry index. Default is None."
            " It has to be a tsv file that contains at least two columns where the",
            " first column is the subject ID and the second column is the ancestry index",
            " starting from 1 (e.g., 1, 2, 3 etc.). It can be a compressed file (e.g., tsv.gz).",
            " Only the first two columns will be used. No headers.",
            " If this file is specified, it assumes that all the phenotypes across ancestries are in one single file,",
            " and same thing for genotypes and covariates data.",
            " It will produce errors if multiple phenotype, genotype, and covariates are specified.",
        ),
    )

    finemap.add_argument(
        "--keep",
        nargs=1,
        default=None,
        type=str,
        help=(
            "Single file that contains subject ID across all ancestries that are used for fine-mapping."
            " It has to be a tsv file that contains at least one columns where the",
            " first column is the subject ID. It can be a compressed file (e.g., tsv.gz). No headers.",
            " If this file is specified, all phenotype, genotype, and covariates data will be filtered down to",
            " the subjects listed in it.",
        ),
    )

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
            "If your categorical covariates have n levels, make sure the dummy variables have n-1 columns.",
        ),
    )

    finemap.add_argument(
        "--L",
        default=10,
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
        "--resid-var",
        nargs="+",
        default=None,
        type=float,
        help=(
            "Specify the prior for the residual variance for ancestries. Default is 1e-3 for each ancestry.",
            " Values have to be positive. Use 'space' to separate ancestries if more than two.",
        ),
    )

    finemap.add_argument(
        "--effect-var",
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
            "Specify the prior for the effect correlation for ancestries. Default is 0.1 for each pair of ancestries.",
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
        "--no-scale",
        default=False,
        action="store_true",
        help=(
            "Indicator to scale the genotype and phenotype data by standard deviation.",
            " Default is False (to scale)."
            " Specify --no_scale will store 'True' value, and may cause different inference.",
        ),
    )

    finemap.add_argument(
        "--no-regress",
        default=False,
        action="store_true",
        help=(
            "Indicator to regress the covariates on each SNP. Default is False (to regress).",
            " Specify --no_regress will store 'True' value.",
            " It may slightly slow the inference, but can be more accurate.",
        ),
    )

    finemap.add_argument(
        "--no-update",
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
        "--max-iter",
        default=500,
        type=int,
        help=(
            "Maximum iterations for the optimization. Default is 500.",
            " Larger number may slow the inference while smaller may cause different inference.",
        ),
    )

    finemap.add_argument(
        "--min-tol",
        default=1e-3,
        type=float,
        help=(
            "Minimum tolerance for the convergence. Default is 1e-3.",
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
        "--max-select",
        default=250,
        type=int,
        help=(
            "The maximum selected number of SNPs to calculate the purity. Default is 250.",
            " It has to be positive integer number. A larger number can unnecessarily spend much memory.",
        ),
    )

    finemap.add_argument(
        "--min-snps",
        default=100,
        type=int,
        help=(
            "The minimum number of SNPs to fine-map. Default is 100.",
            " It has to be positive integer number.",
        ),
    )

    finemap.add_argument(
        "--maf",
        default=0.01,
        type=float,
        help=(
            "Threshold for minor allele frequency (MAF) to filter out SNPs for each ancestry.",
            " It has to be a float between 0 (exclusive) and 0.5 (inclusive).",
        ),
    )

    finemap.add_argument(
        "--rint",
        default=False,
        action="store_true",
        help=(
            "Indicator to perform rank inverse normalization transformation (rint) for each phenotype data.",
            " Default is False (do not transform).",
            " Specify --rint will store 'True' value.",
            " We suggest users to do this QC during data preparation.",
        ),
    )

    finemap.add_argument(
        "--no-reorder",
        default=False,
        action="store_true",
        help=(
            "Indicator to re-order single effects based on Frobenius norm of alpha-weighted",
            " posterior mean square.",
            " Default is False (to re-order).",
            " Specify --no-reorder will store 'True' value.",
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
            " After row-binding phenotype data, it will perform rank inverse normalization transformation.",
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
        "--cv-num",
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
            "The seed for randomization. It can be used to cut data sets in cross validation. ",
            " It can also be used to randomly select SNPs in the credible sets to calculate the purity."
            " Default is 12345. It has to be positive integer number.",
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
        "--jax-precision",
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


def _get_command_string(args):
    base = f"sushie {args[0]}{os.linesep}"
    rest = args[1:]
    rest_strs = []
    needs_tab = True
    lead_prompt = None
    for cmd in rest:
        if "-" == cmd[0]:
            lead_prompt = cmd
            if cmd in [
                "--quiet",
                "--verbose",
                "--compress",
                "--numpy",
                "--no-scale",
                "--no-regress",
                "--no-update",
                "--meta",
                "--mega",
                "--her",
                "--cv",
                "--alphas",
                "--rint",
            ]:
                rest_strs.append(f"\t{cmd}{os.linesep}")
                needs_tab = True
            else:
                rest_strs.append(f"\t{cmd}")
                needs_tab = False
        else:
            if needs_tab:
                rest_strs.append(f"\t{' '*len(lead_prompt)} {cmd}{os.linesep}")
                needs_tab = True
            else:
                rest_strs.append(f" {cmd}{os.linesep}")
                needs_tab = True

    return base + "".join(rest_strs) + os.linesep


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
