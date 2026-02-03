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
from scipy.stats import norm

from jax import config, random

from . import infer, infer_ss, io, log, utils

# Filter ABSL and JAX warnings that clutter output (must be before JAX import)
warnings.filterwarnings("ignore", module="absl")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="jax")
import jax.numpy as jnp  # noqa: E402

__all__ = [
    "parameter_check",
    "process_raw",
    "sushie_wrapper",
    "run_finemap",
]


def _keep_file_subjects(
    rawData: io.RawData, keep_subject: List[str], idx: int
) -> io.RawData:
    _, _, _, pheno, _ = rawData

    log.logger.debug(
        f"Remove individuals based on the keep file for ancestry {idx + 1}."
    )

    # we just need to filter out the subjects in phenotype file
    # because later pheno and covar will be inner merged with fam file
    old_pheno_num = pheno.shape[0]
    pheno = pheno[pheno.iid.isin(keep_subject)].reset_index(drop=True)
    new_pheno_num = pheno.shape[0]
    del_num = old_pheno_num - new_pheno_num
    if del_num != 0:
        log.logger.debug(
            f"Ancestry {idx + 1}:Drop {del_num} out of {old_pheno_num} subjects because of the keep subject file."
        )
    if new_pheno_num == 0:
        raise ValueError(
            f"Ancestry {idx + 1}: All subjects are removed because of the keep subject file. Check the source."
        )

    rawData = rawData._replace(
        pheno=pheno,
    )

    return rawData


def _drop_na_subjects(rawData: io.RawData, idx: int) -> io.RawData:
    _, _, _, pheno, covar = rawData

    old_pheno_num = pheno.shape[0]
    pheno = pheno.dropna().reset_index(drop=True)
    new_pheno_num = pheno.shape[0]
    del_pheno_num = old_pheno_num - new_pheno_num

    log.logger.debug(f"Remove individuals based on NA value for ancestry {idx + 1}.")

    if del_pheno_num != 0:
        log.logger.debug(
            f"Ancestry {idx + 1}: Drop {del_pheno_num} out of {old_pheno_num} subjects"
            + " because of INF or NAN value in phenotype data."
        )

    if new_pheno_num == 0:
        raise ValueError(
            f"Ancestry {idx + 1}: All subjects have INF or NAN value in phenotype data."
            + " Check the source."
        )

    if covar is not None:
        old_covar_num = covar.shape[0]
        covar = covar.dropna().reset_index(drop=True)
        new_covar_num = covar.shape[0]
        del_covar_num = old_covar_num - new_covar_num

        if del_covar_num != 0:
            log.logger.debug(
                f"Ancestry {idx + 1}: Drop {del_covar_num} out of {old_covar_num} subjects"
                + " because of INF or NAN value in covariates data."
            )

        if new_covar_num == 0:
            raise ValueError(
                f"Ancestry {idx + 1}: All subjects have INF or NAN value in covariates data."
                + " Check the source."
            )

    rawData = rawData._replace(
        pheno=pheno,
        covar=covar,
    )

    return rawData


def _impute_geno(rawData: io.RawData, idx: int) -> io.RawData:
    bim, _, bed, _, _ = rawData
    old_bim_num = bim.shape[0]

    log.logger.debug(f"Impute genotypes for ancestry {idx + 1}.")

    # to make sure that the bim index is continuous
    bim = bim.reset_index(drop=True)
    # if we observe SNPs have nan value for all participants (although not likely), drop them
    (del_idx,) = jnp.where(jnp.all(jnp.isnan(bed), axis=0))
    # this is the first time we modify the bim and bed file
    # it's okay just to directly drop them
    bim = bim.drop(del_idx).reset_index(drop=True)
    bed = jnp.delete(bed, del_idx, 1)

    if len(del_idx) == bim.shape[0]:
        raise ValueError(
            f"Ancestry {idx + 1}: All SNPs have INF or NAN value in genotype data. Check the source."
        )

    if len(del_idx) != 0:
        log.logger.debug(
            f"Ancestry {idx + 1}: Drop {len(del_idx)} out of {old_bim_num} SNPs because all subjects have NAN value"
            + " in genotype data."
        )

    # if we observe SNPs that partially have nan value, impute them with column mean
    col_mean = jnp.nanmean(bed, axis=0)
    # it gives the dimension index of the nan value
    imp_idx = jnp.where(jnp.isnan(bed))

    old_bim_num = bim.shape[0]
    # column is the SNP index
    if len(imp_idx[1]) != 0:
        # based on the column index of imp_idx, we used jnp.take to get the (multiple) value
        bed = bed.at[imp_idx].set(jnp.take(col_mean, imp_idx[1]))
        log.logger.debug(
            f"Ancestry {idx + 1}: Impute {len(imp_idx[1])} out of {old_bim_num} SNPs with NAN value based on allele"
            + " frequency."
        )

    rawData = rawData._replace(
        bim=bim,
        bed=bed,
    )

    return rawData


def _filter_maf(rawData: io.RawData, maf: float, idx: int) -> io.RawData:
    bim, _, bed, _, _ = rawData

    old_bim_num = bim.shape[0]

    log.logger.debug(f"Filter genotypes based on MAF for ancestry {idx + 1}.")

    # calculate maf
    snp_maf = jnp.mean(bed, axis=0) / 2
    snp_maf = jnp.where(snp_maf > 0.5, 1 - snp_maf, snp_maf)

    (sel_idx,) = jnp.where(snp_maf >= maf)

    bim = bim.iloc[sel_idx, :].reset_index(drop=True)
    bed = bed[:, sel_idx]

    rawData = rawData._replace(
        bim=bim,
        bed=bed,
    )

    del_num = old_bim_num - len(sel_idx)

    if del_num == old_bim_num:
        raise ValueError(
            f"Ancestry {idx + 1}: All SNPs cannot pass the MAF threshold at {maf}."
        )

    if del_num != 0:
        log.logger.debug(
            f"Ancestry {idx + 1}: Drop {del_num} out of {old_bim_num} SNPs because of maf threshold at {maf}."
        )

    return rawData


def _remove_dup_geno(rawData: io.RawData, idx: int) -> io.RawData:
    bim, _, bed, _, _ = rawData
    old_bim_num = bim.shape[0]
    # to make sure that the bim index is continuous
    bim = bim.reset_index(drop=True)

    log.logger.debug(
        f"Remove duplicated individuals based on genotype data for ancestry {idx + 1}."
    )

    (dup_idx,) = jnp.where(bim.snp.duplicated().values)

    bim = bim.drop(dup_idx).reset_index(drop=True)
    bed = jnp.delete(bed, dup_idx, 1)
    del_num = len(dup_idx)

    if del_num != 0:
        log.logger.debug(
            f"Ancestry {idx + 1}: Drop {del_num} out of {old_bim_num} SNPs because of duplicates in the rsID"
            + " in genotype data."
        )

    rawData = rawData._replace(
        bim=bim,
        bed=bed,
    )

    return rawData


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

    log.logger.debug(
        f"Keep common individuals based on genotype and phenotype data for ancestry {idx + 1}."
    )

    common_fam = fam.merge(
        pheno[[f"phenoIDX_{idx + 1}", "iid"]], how="inner", on=["iid"]
    )

    if covar is not None:
        # match fam id and covar id
        common_fam = common_fam.merge(
            covar[[f"covarIDX_{idx + 1}", "iid"]], how="inner", on=["iid"]
        )

    if common_fam.shape[0] == 0:
        raise ValueError(
            f"Ancestry {idx + 1}: No common individuals across phenotype, covariates,"
            + " genotype found. Check the source.",
        )
    else:
        log.logger.debug(
            f"Ancestry {idx + 1}: Found {common_fam.shape[0]} common individuals"
            + " across phenotype, covariates, and genotype.",
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
    geno: List[jnp.ndarray],
    pheno: List[jnp.ndarray],
    cv_num: int,
    seed: int,
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
        train_index = jnp.delete(jnp.arange(cv_num), cv).tolist()

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


def _run_cv(args, cv_data, pi) -> List[List[jnp.ndarray]]:
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
            pi=pi,
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
    args: argparse.Namespace,
) -> Tuple[int, pd.DataFrame, List[str], pd.DataFrame, List[str], Callable]:
    """The function to process raw phenotype, genotype, covariates data across ancestries
        for individual-level data fine-mapping.

    Args:
        args: The command line parameter input.

    Returns:
        :py:obj:`Tuple[int, pd.DataFrame, List[str], Callable]`:
            A tuple of
                #. an integer to indicate how many ancestries,
                #. a DataFrame that contains ancestry index (can be none),
                #. a list that contains subject ID that fine-mapping performs on.
                #. a DataFrame that contains prior probability for each SNP to be causal.
                #. a list of genotype data paths (:py:obj:`List[str]`),
                #. genotype read-in function (:py:obj:`Callable`).

    """
    if args.pheno is None:
        raise ValueError(
            "No phenotype file specified. Specify --summary if summary-level fine-mapping is wanted."
        )

    if args.ancestry_index is not None:
        log.logger.debug("Read in ancestry index file.")
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

    log.logger.info(f"Detect phenotypes for {args.trait} for {n_pop} {name_ancestry}.")

    n_geno = (
        int(args.plink is not None)
        + int(args.vcf is not None)
        + int(args.bgen is not None)
    )

    if n_geno > 1:
        log.logger.info(
            f"Detect {n_geno} genotypes, will only use one type of genotypes in the order of 'plink, vcf, and bgen'"
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

        log.logger.info(
            f"Detect genotype data in plink format for {n_pop} {name_ancestry}."
        )
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
        log.logger.info(
            f"Detect genotype data in vcf format for {n_pop} {name_ancestry}."
        )
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

        log.logger.info(
            f"Detect genotype data in bgen format for {n_pop} {name_ancestry}."
        )
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

    if args.pi != "uniform":
        log.logger.info(
            "Detect file that contains prior weights for each SNP to be causal."
        )
        pi = pd.read_csv(args.pi, header=None, sep="\t")
        if pi.shape[0] == 0:
            raise ValueError(
                "No prior weights are listed in the prior file. Check the source."
            )

        if pi.shape[1] < 2:
            raise ValueError(
                "The prior file has less than 2 columns. It has to be at least two columns."
                + " The first column is the SNP ID, and the second column the prior probability."
            )

        if pi.shape[1] > 2:
            log.logger.debug(
                "The prior file has more than 2 columns. Will only use the first two columns."
            )

        pi = pi.iloc[:, 0:2]
        pi.columns = ["snp", "pi"]
    else:
        pi = pd.DataFrame()

    if args.seed <= 0:
        raise ValueError(
            "The seed specified for randomization must be greater than 0. Choose a positive integer using --seed."
        )

    if args.cv:
        if args.cv_num <= 1:
            raise ValueError(
                "The number of folds in cross validation must be greater than 1."
                + " Update with --cv-num.",
            )

    if args.maf <= 0 or args.maf > 0.5:
        raise ValueError(
            "The minor allele frequency (MAF) has to be between 0 (exclusive) and 0.5 (inclusive)."
            + " Choose a valid frequency using --maf."
        )

    if (args.meta or args.mega) and n_pop == 1:
        log.logger.debug(
            "The number of ancestry is 1, but --meta or --mega is specified. Will skip meta or mega SuSiE."
        )

    if args.chrom is None and args.start is None and args.end is None:
        log.logger.debug(
            "No region is specified. Will use all SNPs available in the data."
        )

    elif args.chrom is not None and args.start is not None and args.end is not None:
        if args.start <= 0:
            raise ValueError(
                "The start position for the region must be greater than 0. Update with --start."
            )

        if args.end <= 0:
            raise ValueError(
                "The end position for the region must be greater than 0. Update with --end."
            )

        if args.end <= args.start:
            raise ValueError(
                "The end position for the region must be greater than --start. Update with"
                + " --start or --end."
            )

        log.logger.info(
            f"Detect region (chrom{args.chrom}:{args.start}:{args.end}) to be fine-mapped."
            + " Will only use SNPs within the region."
        )
    else:
        raise ValueError(
            "The region is not specified correctly. Please provide --chrom, --start, and --end together,"
            + " or omit all of them."
        )

    log.logger.debug("Finish parameter check for individual-level fine-mapping.")

    return n_pop, ancestry_index, keep_subject, pi, geno_path, geno_func


def parameter_check_ss(
    args: argparse.Namespace,
) -> Tuple[int, pd.DataFrame, List[str], Callable, bool]:
    """The function to process raw phenotype, genotype, covariates data across ancestries
        for summary-level fine-mapping.

    Args:
        args: The command line parameter input.

    Returns:
        :py:obj:`Tuple[int, pd.DataFrame, List[str], Callable]`:
            A tuple of
                #. an integer to indicate how many ancestries,
                #. a DataFrame that contains prior probability for each SNP to be causal.
                #. a list of genotype data paths (:py:obj:`List[str]`),
                #. genotype read-in function (:py:obj:`Callable`).
                #. a boolean to indicate whether the genotype data is in LD format.

    """
    if args.gwas is None:
        raise ValueError("No GWAS summary statistics file specified. Check the source.")
    else:
        n_pop = len(args.gwas)

    name_ancestry = "ancestry" if n_pop == 1 else "ancestries"

    log.logger.info(f"Detect GWAS files for {args.trait} for {n_pop} {name_ancestry}.")

    n_geno = (
        int(args.plink is not None)
        + int(args.vcf is not None)
        + int(args.bgen is not None)
        + int(args.ld is not None)
    )

    if n_geno > 1:
        log.logger.info(
            f"Detect {n_geno} genotype or LD files,"
            + " will only use one type of file in the order of 'plink, vcf, bgen, and ld'",
        )

    # decide genotype data
    ld_file = False
    if args.plink is not None:
        if len(args.plink) != n_pop:
            raise ValueError(
                "The numbers of ancestries in plink geno and GWAS data does not match. Check the source."
            )

        log.logger.info(
            f"Detect genotype data in plink format for {n_pop} {name_ancestry}."
        )
        geno_path = args.plink
        geno_func = io.read_triplet
    elif args.vcf is not None:
        if len(args.vcf) != n_pop:
            raise ValueError(
                "The numbers of ancestries in vcf geno and GWAS data does not match. Check the source."
            )
        log.logger.info(
            f"Detect genotype data in vcf format for {n_pop} {name_ancestry}."
        )
        geno_path = args.vcf
        geno_func = io.read_vcf
    elif args.bgen is not None:
        if len(args.bgen) != n_pop:
            raise ValueError(
                "The numbers of ancestries in bgen geno and GWAS data does not match. Check the source."
            )

        log.logger.info(
            f"Detect genotype data in bgen format for {n_pop} {name_ancestry}."
        )
        geno_path = args.bgen
        geno_func = io.read_bgen
    elif args.ld is not None:
        if len(args.ld) != n_pop:
            raise ValueError(
                "The numbers of ancestries in ld geno and pheno data does not match. Check the source."
            )

        log.logger.info(f"Detect LD data in tsv files for {n_pop} {name_ancestry}.")
        geno_path = args.ld
        geno_func = io.read_ld
        ld_file = True
    else:
        raise ValueError(
            "No genotype/LD data specified in either plink, vcf, bgen, or LD files. Check the source."
        )

    if args.sample_size is not None:
        if len(args.sample_size) != n_pop:
            raise ValueError(
                "The numbers of ancestries in sample size and GWAS data does not match. Check the source"
                + " and update --sample-size."
            )

        if not all(sample_size > 0 for sample_size in args.sample_size):
            raise ValueError(
                "The sample size specified for summary-level fine-mapping is invalid. Choose a positive integer"
                + " using --sample-size."
            )

        log.logger.info(f"Detect sample sizes for {n_pop} {name_ancestry}.")
    else:
        raise ValueError(
            "No sample size specified for summary-level fine-mapping. Check the source."
        )

    if args.pi != "uniform":
        log.logger.info(
            "Detect file that contains prior weights for each SNP to be causal."
        )
        pi = pd.read_csv(args.pi, header=None, sep="\t")

        # remove dupliate rows
        pi = pi.drop_duplicates(subset=pi.columns[0])

        if pi.shape[0] == 0:
            raise ValueError(
                "No prior weights are listed in the prior file. Check the source."
            )

        if pi.shape[1] < 2:
            raise ValueError(
                "The prior file has less than 2 columns. It has to be at least two columns."
                + " The first column is the SNP ID, and the second column the prior probability."
            )

        if pi.shape[1] > 2:
            log.logger.debug(
                "The prior file has more than 2 columns. Will only use the first two columns."
            )

        pi = pi.iloc[:, 0:2]
        pi.columns = ["snp", "pi"]
    else:
        pi = pd.DataFrame()

    if args.seed <= 0:
        raise ValueError(
            "The seed specified for randomization is invalid. Choose a positive integer using --seed."
        )

    if args.maf <= 0 or args.maf > 0.5:
        raise ValueError(
            "The minor allele frequency (MAF) has to be between 0 (exclusive) and 0.5 (inclusive)."
            + " Choose a valid frequency using --maf."
        )

    if args.meta and n_pop == 1:
        log.logger.debug(
            "The number of ancestry is 1, but --meta is specified. Will skip meta or mega SuSiE."
        )

    if args.chrom is None and args.start is None and args.end is None:
        log.logger.debug(
            "No region is specified. Will use all SNPs available in the data."
        )
    elif args.chrom is not None and args.start is not None and args.end is not None:
        if args.start <= 0:
            raise ValueError(
                "The start position for the region must be greater than 0. Update with --start."
            )

        if args.end <= 0:
            raise ValueError(
                "The end position for the region must be greater than 0. Update with --end."
            )

        if args.end <= args.start:
            raise ValueError(
                "The end position for the region must be greater than --start. Update with"
                + " --start or --end."
            )

        log.logger.info(
            f"Detect region (chrom{args.chrom}:{args.start}:{args.end}) to be fine-mapped."
            + " Will only use SNPs within the region."
        )
    else:
        raise ValueError(
            "The region is not specified correctly. Please provide --chrom, --start, and --end together,"
            + " or omit all of them."
        )

    if args.gwas_sig <= 0 or args.gwas_sig > 1:
        raise ValueError(
            "The significance threshold for P-values in GWAS summary statistics must be greater than 0"
            + " and less than or equal to 1. Choose a valid number using --gwas-sig."
        )

    if args.ld_adjust > 0.1 or args.ld_adjust < 0:
        raise ValueError(
            "The LD adjustment parameter has to be greater than 0 and less than 0.1."
            + " Choose a valid number using --ld-adjust."
        )

    if args.cv:
        log.logger.debug(
            "Cross-validation is not supported for summary-level fine-mapping. This flag (--cv) will be ignored."
        )

    if args.mega:
        log.logger.debug(
            "Mega SuShiE is not supported for summary-level fine-mapping. This flag (--mega) will be ignored."
        )

    if args.her:
        log.logger.debug(
            "Heritability estimation is not supported for summary-level fine-mapping."
            + " This flag (--her) will be ignored."
        )

    log.logger.debug("Finish parameter check for summary-level fine-mapping.")

    return n_pop, pi, geno_path, geno_func, ld_file


def process_raw(
    rawData: List[io.RawData],
    keep_subject: List[str],
    pi: pd.DataFrame,
    keep_ambiguous: bool,
    maf: float,
    rint: bool,
    no_regress: bool,
    mega: bool,
    cv: bool,
    cv_num: int,
    seed: int,
    chrom: utils.IntOrNone,
    start: utils.IntOrNone,
    end: utils.IntOrNone,
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
        pi: The DataFrame that contains prior weights for each SNP to be causal.
        keep_ambiguous: The indicator whether to keep ambiguous SNPs.
        maf: The minor allele frequency threshold to filter the genotypes.
        rint: The indicator whether to perform rank inverse normalization on each phenotype data.
        no_regress: The indicator whether to regress genotypes on covariates.
        mega: The indicator whether to prepare datasets for mega SuShiE.
        cv: The indicator whether to prepare datasets for cross-validation.
        cv_num: The number for :math:`X`-fold cross-validation.
        seed: The random seed for row-wise shuffling the datasets for cross validation.
        chrom: The chromosome to filter SNPs.
        start: The start position to filter SNPs.
        end: The end position to filter SNPs.


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
        # keep subjects that are listed in the keep subject file
        if len(keep_subject) != 0:
            rawData[idx] = _keep_file_subjects(rawData[idx], keep_subject, idx)

        # remove NA/inf value for subjects across phenotype or covariates data
        rawData[idx] = _drop_na_subjects(rawData[idx], idx)

        # impute genotype data even though we suggest users to impute the genotypes beforehand
        rawData[idx] = _impute_geno(rawData[idx], idx)

        # remove SNPs that cannot pass MAF threshold
        rawData[idx] = _filter_maf(rawData[idx], maf, idx)

        # remove duplicates SNPs based on rsid even though we suggest users to do some QC on this
        rawData[idx] = _remove_dup_geno(rawData[idx], idx)

        # reset index and add index column to all dataset for future inter-ancestry or inter-dataset processing
        rawData[idx] = _reset_idx(rawData[idx], idx)

        # find common individuals across geno, pheno, and covar within an ancestry
        rawData[idx] = _filter_common_ind(rawData[idx], idx)

    # find common snps across ancestries
    log.logger.debug("Fine common SNPs across ancestries.")

    if n_pop > 1:
        snps = (
            rawData[0]
            .bim.merge(rawData[1].bim, how="inner", on=["chrom", "snp"])
            .reset_index(drop=True)
        )

        for idx in range(n_pop - 2):
            snps = snps.merge(
                rawData[idx + 2].bim, how="inner", on=["chrom", "snp"]
            ).reset_index(drop=True)
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

    # remove biallelic SNPs dont match across ancestries
    # e.g., A/T for EUR but A/C for AFR
    if n_pop > 1:
        log.logger.debug("Remove SNPs that do not have same alleles across ancestries.")
        for idx in range(1, n_pop):
            _, _, remove_idx = _allele_check(
                snps["a0_1"].values,
                snps["a1_1"].values,
                snps[f"a0_{idx + 1}"].values,
                snps[f"a1_{idx + 1}"].values,
            )

            if len(remove_idx) != 0:
                snps = snps.drop(remove_idx).reset_index(drop=True)
                log.logger.debug(
                    f"Ancestry{idx + 1} has {len(remove_idx)} alleles that"
                    + "couldn't match to ancestry 1 and couldn't be flipped. Will remove these SNPs."
                )

            if snps.shape[0] == 0:
                raise ValueError(
                    f"Ancestry {idx + 1} has none of correct or flippable SNPs matching to ancestry 1."
                    + "Check the source.",
                )

    # remove ambiguous SNPs (i.e., A/T, T/A, C/G, G/C pairs) in genotype data
    if not keep_ambiguous:
        log.logger.debug("Remove ambiguous SNPs.")

        ambiguous_snps = ["AT", "TA", "CG", "GC"]
        if_ambig = (snps.a0_1 + snps.a1_1).isin(ambiguous_snps)
        del_num = if_ambig.sum()
        snps = snps[~if_ambig].reset_index(drop=True)

        if snps.shape[0] == 0:
            raise ValueError(
                "All SNPs are ambiguous in genotype data. Check the source."
            )

        if del_num != 0:
            log.logger.debug(f"Drop {del_num} ambiguous SNPs in genotype data.")

    log.logger.debug(
        "Filter SNPs based on Chrom, Start, and End using coordinates of first ancestry."
    )
    snps["chrom"] = snps["chrom"].astype("int64")

    if chrom is not None:
        old_num = snps.shape[0]
        snps = snps[snps.chrom == chrom]
        del_num = old_num - snps.shape[0]

        if snps.shape[0] == 0:
            raise ValueError(f"No SNPs remain after filtering on chromosome {chrom}.")

        if del_num != 0:
            log.logger.debug(f"Drop {del_num} SNPs that are not on chromosome {chrom}.")

        old_num = snps.shape[0]
        snps = snps[snps.pos_1 >= start]
        del_num = old_num - snps.shape[0]

        if snps.shape[0] == 0:
            raise ValueError(
                f"No SNPs are located after position {start} on chromosome {chrom}."
            )

        if del_num != 0:
            log.logger.debug(
                f"Drop {del_num} SNPs that are located before position {start} on chromosome {chrom}."
            )

        old_num = snps.shape[0]
        snps = snps[snps.pos_1 <= end]
        del_num = old_num - snps.shape[0]

        if snps.shape[0] == 0:
            raise ValueError(
                f"No SNPs are located before position {end} on chromosome {chrom}."
            )

        if del_num != 0:
            log.logger.debug(
                f"Drop {del_num} SNPs that are located after position {end} on chromosome {chrom}."
            )

        snps = snps.reset_index(drop=True)

    # find flipped reference alleles across ancestries
    flip_idx = []
    if n_pop > 1:
        log.logger.debug(
            "Flip the alleles of subsequent ancestries to match those of the first ancestry."
        )
        for idx in range(1, n_pop):
            _, tmp_flip_idx, _ = _allele_check(
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

            # drop unused columns
            snps = snps.drop(
                columns=[f"a0_{idx + 1}", f"a1_{idx + 1}", f"pos_{idx + 1}"]
            )

    # rename columns for better indexing in the future
    snps = snps.reset_index().rename(
        columns={"index": "SNPIndex", "a0_1": "a0", "a1_1": "a1", "pos_1": "pos"}
    )

    if pi.shape[0] != 0:
        # append prior weights to the snps
        log.logger.debug("Process prior weight file.")
        snps = pd.merge(snps, pi, how="left", on="snp")
        nan_count = snps["pi"].isna().sum()
        if nan_count > 0:
            log.logger.debug(
                f"{nan_count} SNP(s) have missing prior weights. Will replace them with the mean value of the rest."
            )
        # if the column pi has nan value, replace it with the mean value of the rest of the column
        snps["pi"] = snps["pi"].fillna(snps["pi"].mean())
        pi = jnp.array(snps["pi"].values)
    else:
        snps["pi"] = jnp.ones(snps.shape[0]) / float(snps.shape[0])
        pi = None

    geno = []
    pheno = []
    covar = []
    total_ind = 0
    # filter on geno, pheno, and covar
    for idx in range(n_pop):
        log.logger.debug(f"Final process the rawdata for ancestry {idx + 1}.")
        _, tmp_fam, tmp_geno, tmp_pheno, tmp_covar = rawData[idx]

        # get common individual and snp id
        common_ind_id = tmp_fam[f"famIDX_{idx + 1}"].values
        common_snp_id = snps[f"bimIDX_{idx + 1}"].values
        snps = snps.drop(columns=[f"bimIDX_{idx + 1}"])

        # filter on individuals who have both geno, pheno, and covar (if applicable)
        # filter on shared snps across ancestries
        tmp_geno = tmp_geno[common_ind_id, :][:, common_snp_id]

        # flip genotypes for bed files starting second ancestry
        # flip index is the positional index based on snps data frame, so we have to subset genotype
        # data based on the common snps (i.e., snps data frame).
        if idx > 0 and len(flip_idx[idx - 1]) != 0:
            tmp_geno = tmp_geno.at[:, flip_idx[idx - 1]].set(
                2 - tmp_geno[:, flip_idx[idx - 1]]
            )

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

    regular_data = io.CleanData(geno=geno, pheno=pheno, covar=data_covar, pi=pi)

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
                pi=pi,
            )

    log.logger.debug(
        "Finish preparing data for cross-validation and mega fine-mapping."
    )

    return snps, regular_data, mega_data, cv_data


def process_raw_ss(
    geno_path: List[str],
    geno_func: Callable,
    ld_file: bool,
    pi: pd.DataFrame,
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, io.ssData]:
    """The function to process raw phenotype, genotype, covariates data across ancestries.

    Args:
        geno_paths: The path for genotype data across ancestries.
        geno_func: The function to read in genotypes depending on the format.
        ld_file: The indicator whether LD matrix in provided.
        pi: The DataFrame that contains prior weights for each SNP to be causal.
        args: The command line parameter input.

    Returns:
        :py:obj:`Tuple[pd.DataFrame, io.ssData]`:
        A tuple of
            #. SNP information (:py:obj:`pd.DataFrame`),
            #. dataset for running summary-level SuShiE (:py:obj:`io.ssData`),

    """

    n_pop = len(geno_path)

    ld_geno_list = []
    gwas_list = []
    for idx in range(n_pop):
        # read in GWAS data
        log.logger.debug(f"Read in GWAS data for ancestry {idx + 1}.")

        df_gwas = io.read_gwas(
            args.gwas[idx], args.gwas_header, args.chrom, args.start, args.end
        )

        df_gwas = df_gwas.rename(
            columns={
                "pos": f"pos_{idx + 1}",
                "a0": f"a0_{idx + 1}",
                "a1": f"a1_{idx + 1}",
                "z": f"z_{idx + 1}",
            }
        )

        # read in genotype data or LD data
        if ld_file:
            log.logger.debug(f"Read in LD file for ancestry {idx + 1}.")
            df_ld = geno_func(geno_path[idx])
            if not (df_ld.values == df_ld.values.T).all():
                if (df_ld.round(4).values == df_ld.round(4).values.T).all():
                    df_ld = df_ld.round(4)
                    log.logger.debug(
                        f"Ancestry {idx + 1}: The LD matrix is not symmetric. Will symmetrize it"
                        + " by rounding to 4 digits."
                    )
                else:
                    raise ValueError(
                        f"Ancestry {idx + 1}: The LD matrix is still not symmetric after rounding 4 digits."
                        + " Check the source."
                    )

            if df_ld.shape[0] == 0:
                raise ValueError(
                    f"Ancestry {idx + 1}: No SNPs in the LD data. Check the source."
                )

            if df_gwas["snp"].isin(df_ld.columns).sum() == 0:
                raise ValueError(
                    f"Ancestry {idx + 1}: No common SNPs between GWAS and LD data. Check the source."
                )

            # make sure the diagonal of the LD matrix is 1
            # and add a small value to the diagonal to avoid singular matrix
            # default is 0
            df_ld.values[range(df_ld.shape[0]), range(df_ld.shape[0])] = (
                1 + args.ld_adjust
            )

            ld_geno_list.append(df_ld)
        else:
            log.logger.debug(
                f"Read in genotype file to compute LD for ancestry {idx + 1}."
            )
            bim, fam, bed = geno_func(geno_path[idx])
            bim.chrom = bim.chrom.astype(int)

            if bed.shape[0] == 0:
                raise ValueError(
                    f"Ancestry {idx + 1}: No SNPs in the genotype data. Check the source."
                )

            tmp_rawData = io.RawData(
                bim=bim, fam=fam, bed=bed, pheno=pd.DataFrame([]), covar=None
            )

            # impute genotype data even though we suggest users to impute the genotypes beforehand
            tmp_rawData = _impute_geno(tmp_rawData, idx)

            # remove SNPs that cannot pass MAF threshold
            tmp_rawData = _filter_maf(tmp_rawData, args.maf, idx)

            # remove duplicates SNPs based on rsid even though we suggest users to do some QC on this
            tmp_rawData = _remove_dup_geno(tmp_rawData, idx)

            # find overlap between GWAS snps and genotype snps
            tmp_bim = (
                tmp_rawData.bim.reset_index(drop=True)
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

            if df_gwas["snp"].isin(tmp_bim["snp"]).sum() == 0:
                raise ValueError(
                    f"Ancestry {idx + 1}: No common SNPs between GWAS and genotype data. Check the source."
                )

            tmp_rawData = tmp_rawData._replace(bim=tmp_bim)
            ld_geno_list.append(tmp_rawData)

        gwas_list.append(df_gwas)

    # find common snps across ancestries
    if n_pop > 1:
        log.logger.debug("Find common GWAS SNPs across ancestries.")
        snps_gwas = (
            gwas_list[0]
            .merge(gwas_list[1], how="inner", on=["chrom", "snp"])
            .reset_index(drop=True)
        )
        for idx in range(n_pop - 2):
            snps_gwas = snps_gwas.merge(
                gwas_list[idx + 2], how="inner", on=["chrom", "snp"]
            ).reset_index(drop=True)

        if snps_gwas.shape[0] == 0:
            raise ValueError(
                "GWAS data have no common SNPs across ancestries. Check the source."
            )

        # report how many snps we removed due to independent SNPs
        for idx in range(n_pop):
            snps_num_diff = gwas_list[idx].shape[0] - snps_gwas.shape[0]
            log.logger.debug(
                f"Ancestry{idx + 1} has {snps_num_diff} independent SNPs and {snps_gwas.shape[0]}"
                + " common SNPs. Inference only performs on common SNPs.",
            )

        if ld_file:
            snps_ld = pd.DataFrame({"snps": ld_geno_list[0].columns})
            snps_ld = snps_ld[snps_ld.snps.isin(ld_geno_list[1].columns)]

            for idx in range(n_pop - 2):
                snps_ld = snps_ld[snps_ld.snps.isin(ld_geno_list[idx + 2].columns)]

            if snps_ld.shape[0] == 0:
                raise ValueError(
                    "LD data have no common SNPs across ancestries. Check the source."
                )

            for idx in range(n_pop):
                snps_num_diff = ld_geno_list[idx].shape[0] - snps_ld.shape[0]
                log.logger.debug(
                    f"Ancestry{idx + 1} has {snps_num_diff} independent SNPs and {snps_ld.shape[0]}"
                    + " common SNPs. Inference only performs on common SNPs.",
                )
        else:
            snps_bim = (
                ld_geno_list[0]
                .bim.merge(ld_geno_list[1].bim, how="inner", on=["chrom", "snp"])
                .reset_index(drop=True)
            )
            for idx in range(n_pop - 2):
                snps_bim = snps_bim.merge(
                    ld_geno_list[idx + 2].bim, how="inner", on=["chrom", "snp"]
                ).reset_index(drop=True)

            if snps_bim.shape[0] == 0:
                raise ValueError(
                    "Genotype data have no common SNPs across ancestries. Check the source."
                )

            for idx in range(n_pop):
                snps_num_diff = ld_geno_list[idx].bim.shape[0] - snps_bim.shape[0]
                log.logger.debug(
                    f"Ancestry{idx + 1} has {snps_num_diff} independent SNPs and {snps_bim.shape[0]}"
                    + " common SNPs. Inference only performs on common SNPs.",
                )
    else:
        snps_gwas = gwas_list[0].reset_index(drop=True)
        if ld_file:
            snps_ld = pd.DataFrame({"snps": ld_geno_list[0].columns})
        else:
            snps_bim = ld_geno_list[0].bim.reset_index(drop=True)

    # filter GWAS SNPs based on signficiant threshold
    z_threshold = norm.ppf(1 - args.gwas_sig / 2)
    # Select columns with names starting with "z_"
    z_cols = snps_gwas.filter(regex="^z_")
    old_num = snps_gwas.shape[0]
    if args.gwas_sig_type == "at-least":
        sel_snps = z_cols.abs().gt(z_threshold).any(axis=1)
    else:
        sel_snps = z_cols.abs().gt(z_threshold).all(axis=1)

    snps_gwas = snps_gwas[sel_snps].copy().reset_index(drop=True)
    new_num = snps_gwas.shape[0]

    log.logger.debug(
        f"Drop {old_num - new_num} SNPs with GWAS P value less than {args.gwas_sig}"
        + f" based on {args.gwas_sig_type} method."
    )

    # remove non-biallelic SNPs across ancestries
    if n_pop > 1:
        log.logger.debug("Remove SNPs that do not have same alleles across ancestries.")
        for idx in range(1, n_pop):
            _, _, remove_idx = _allele_check(
                snps_gwas["a0_1"].values,
                snps_gwas["a1_1"].values,
                snps_gwas[f"a0_{idx + 1}"].values,
                snps_gwas[f"a1_{idx + 1}"].values,
            )

            if len(remove_idx) != 0:
                snps_gwas = snps_gwas.drop(index=remove_idx).reset_index(drop=True)
                log.logger.debug(
                    f"Ancestry{idx + 1} GWAS has {len(remove_idx)} alleles that"
                    + "couldn't match to ancestry 1 and couldn't be flipped. Will remove these SNPs."
                )

            if snps_gwas.shape[0] == 0:
                raise ValueError(
                    f"Ancestry {idx + 1} has none of correct or flippable SNPs matching to ancestry 1."
                    + "Check the source.",
                )

            if not ld_file:
                _, _, remove_idx = _allele_check(
                    snps_bim["a0_1"].values,
                    snps_bim["a1_1"].values,
                    snps_bim[f"a0_{idx + 1}"].values,
                    snps_bim[f"a1_{idx + 1}"].values,
                )

                if len(remove_idx) != 0:
                    snps_bim = snps_bim.drop(index=remove_idx).reset_index(drop=True)
                    log.logger.debug(
                        f"Ancestry{idx + 1} Genotype data has {len(remove_idx)} alleles that"
                        + "couldn't match to ancestry 1 and couldn't be flipped. Will remove these SNPs."
                    )

                if snps_bim.shape[0] == 0:
                    raise ValueError(
                        f"Ancestry {idx + 1} genotype data has none of correct or flippable SNPs matching"
                        + "to ancestry 1. Check the source."
                    )

    # remove ambiguous SNPs (i.e., A/T, T/A, C/G, G/C pairs) in genotype data
    if not args.keep_ambiguous:
        log.logger.debug("Remove ambiguous SNPs from GWAS data.")
        ambiguous_snps = ["AT", "TA", "CG", "GC"]
        if_ambig = (snps_gwas.a0_1 + snps_gwas.a1_1).isin(ambiguous_snps)
        del_num = if_ambig.sum()
        snps_gwas = snps_gwas[~if_ambig].reset_index(drop=True)

        if snps_gwas.shape[0] == 0:
            raise ValueError("All SNPs are ambiguous in GWAS data. Check the source.")

        if del_num != 0:
            log.logger.debug(f"Drop {del_num} ambiguous SNPs in GWAS data.")

    # find flipped reference alleles across ancestries
    if n_pop > 1:
        log.logger.debug(
            "Flip the alleles of subsequent ancestries to match those of the first ancestry."
        )
        for idx in range(1, n_pop):
            _, tmp_flip_idx, _ = _allele_check(
                snps_gwas["a0_1"].values,
                snps_gwas["a1_1"].values,
                snps_gwas[f"a0_{idx + 1}"].values,
                snps_gwas[f"a1_{idx + 1}"].values,
            )

            if len(tmp_flip_idx) != 0:
                log.logger.debug(
                    f"Ancestry{idx + 1} has {len(tmp_flip_idx)} flipped alleles in GWAS from ancestry 1."
                    + " Will flip these SNPs."
                )

                snps_gwas.loc[tmp_flip_idx, f"z_{idx + 1}"] *= -1

            # drop unused columns
            snps_gwas = snps_gwas.drop(
                columns=[f"a0_{idx + 1}", f"a1_{idx + 1}", f"pos_{idx + 1}"]
            )

            if not ld_file:
                _, tmp_flip_idx, _ = _allele_check(
                    snps_bim["a0_1"].values,
                    snps_bim["a1_1"].values,
                    snps_bim[f"a0_{idx + 1}"].values,
                    snps_bim[f"a1_{idx + 1}"].values,
                )

                if len(tmp_flip_idx) != 0:
                    log.logger.debug(
                        f"Ancestry{idx + 1} has {len(tmp_flip_idx)} flipped alleles in genotype from ancestry 1."
                        + " Will flip these SNPs."
                    )

                _, _, tmp_geno, _, _ = ld_geno_list[idx]

                common_snp_id = snps_bim[f"bimIDX_{idx + 1}"].values
                tmp_geno = tmp_geno[:, common_snp_id]

                # flip genotypes for bed files starting second ancestry
                # flip index is the positional index based on snps data frame, so we have to subset genotype
                # data based on the common snps (i.e., snps data frame).
                if len(tmp_flip_idx) != 0:
                    tmp_geno = tmp_geno.at[:, tmp_flip_idx].set(
                        2 - tmp_geno[:, tmp_flip_idx]
                    )

                ld_geno_list[idx] = ld_geno_list[idx]._replace(bed=tmp_geno)

                # drop unused columns
                snps_bim = snps_bim.drop(
                    columns=[
                        f"a0_{idx + 1}",
                        f"a1_{idx + 1}",
                        f"pos_{idx + 1}",
                        f"bimIDX_{idx + 1}",
                    ]
                )

    snps_gwas = snps_gwas.rename(columns={"pos_1": "pos", "a0_1": "a0", "a1_1": "a1"})

    if not ld_file:
        # in the above codes, we only subset for idx = 1 to n_pop, but we haven't subset for idx = 0
        _, _, tmp_geno, _, _ = ld_geno_list[0]
        common_snp_id = snps_bim["bimIDX_1"].values
        tmp_geno = tmp_geno[:, common_snp_id]
        ld_geno_list[0] = ld_geno_list[0]._replace(bed=tmp_geno)
        snps_bim = snps_bim.rename(columns={"pos_1": "pos", "a0_1": "a0", "a1_1": "a1"})
        snps_bim = snps_bim.drop(columns=["bimIDX_1"])

    # merge gwas with LD or bim data
    gwas_list = []
    ld_list = []
    log.logger.debug("Merge GWAS and LD data across ancestries.")
    if ld_file:
        # users have to ensure that the counting allele is the same across GWAS and LD data
        overlap_snps = snps_gwas["snp"][snps_gwas["snp"].isin(snps_ld.snps)]

        if overlap_snps.shape[0] == 0:
            raise ValueError(
                "No common SNPs between GWAS and LD data. Check the source."
            )

        df_gwas = (
            snps_gwas.set_index("snp", drop=False)
            .loc[overlap_snps]
            .reset_index(drop=True)
        )
        for idx in range(n_pop):
            gwas_list.append(jnp.array(df_gwas[f"z_{idx + 1}"].values))
            tmp_ld = ld_geno_list[idx]
            tmp_ld = tmp_ld.loc[overlap_snps, overlap_snps]
            if (tmp_ld.values == tmp_ld.values.T).all():
                ld_list.append(jnp.array(tmp_ld))
            else:
                raise ValueError(
                    f"Ancestry {idx + 1}: The LD matrix becomes asymmetric during QC. Contact devleoper."
                )
    else:
        snps_bim = snps_bim.rename(columns={"a0": "a0_bim", "a1": "a1_bim"})

        # merge between GWAS and bim files
        all_snps = snps_gwas.merge(
            snps_bim[["chrom", "snp", "a0_bim", "a1_bim"]],
            how="inner",
            on=["chrom", "snp"],
        ).reset_index(drop=True)

        _, _, tmp_wrong_idx = _allele_check(
            all_snps["a0"].values,
            all_snps["a1"].values,
            all_snps["a0_bim"].values,
            all_snps["a1_bim"].values,
        )

        if len(tmp_wrong_idx) != 0:
            all_snps = all_snps.drop(index=tmp_wrong_idx).reset_index(drop=True)
            log.logger.debug(
                f"Drop {len(tmp_wrong_idx)} SNPs with wrong alleles between GWAS and genotype data."
            )

        overlap_snps = all_snps["snp"]

        if overlap_snps.shape[0] == 0:
            raise ValueError(
                "No common SNPs between GWAS and genotype data. Check the source."
            )

        # re-order bim files based on the order of all_snps
        snps_bim = (
            snps_bim.reset_index(names="SNPIndex")
            .set_index("snp", drop=False)
            .loc[overlap_snps]
            .reset_index(drop=True)
        )

        # here it's just easier to flip GWAS z scores
        _, tmp_flip_idx, _ = _allele_check(
            all_snps["a0_bim"].values,
            all_snps["a1_bim"].values,
            all_snps["a0"].values,
            all_snps["a1"].values,
        )

        if len(tmp_flip_idx) != 0:
            for idx in range(n_pop):
                all_snps.loc[tmp_flip_idx, f"z_{idx + 1}"] *= -1
                log.logger.debug(
                    f"Flip {len(tmp_flip_idx)} SNPs alleles in GWAS data to match genotype data."
                )

        df_gwas = all_snps.drop(columns=["a0", "a1"]).rename(
            columns={"a0_bim": "a0", "a1_bim": "a1"}
        )[
            ["chrom", "snp", "pos", "a0", "a1"]
            + [f"z_{idx + 1}" for idx in range(n_pop)]
        ]

        for idx in range(n_pop):

            gwas_list.append(jnp.array(df_gwas[f"z_{idx + 1}"].values))
            _, _, tmp_geno, _, _ = ld_geno_list[idx]
            tmp_geno = tmp_geno[:, snps_bim["SNPIndex"].values]
            tmp_geno -= tmp_geno.mean(axis=0)
            tmp_geno /= tmp_geno.std(axis=0)
            tmp_ld = tmp_geno.T @ tmp_geno / tmp_geno.shape[0]
            tmp_ld = tmp_ld + jnp.eye(tmp_ld.shape[0]) * args.ld_adjust
            ld_list.append(tmp_ld)

    snps = (
        df_gwas[["chrom", "snp", "pos", "a0", "a1"]]
        .reset_index(drop=True)
        .reset_index(names="SNPIndex")
        .copy()
    )

    if pi.shape[0] != 0:
        # append prior weights to the snps
        snps = pd.merge(snps, pi, how="left", on="snp")
        nan_count = snps["pi"].isna().sum()
        if nan_count > 0:
            log.logger.debug(
                f"{nan_count} SNP(s) have missing prior weights. Will replace them with the mean value of the rest."
            )
        # if the column pi has nan value, replace it with the mean value of the rest of the column
        snps["pi"] = snps["pi"].fillna(snps["pi"].mean())
        pi = jnp.array(snps["pi"].values)
    else:
        snps["pi"] = jnp.ones(snps.shape[0]) / float(snps.shape[0])
        pi = None

    regular_data = io.ssData(
        zs=gwas_list, lds=ld_list, ns=jnp.array(args.sample_size)[:, jnp.newaxis], pi=pi
    )

    name_ancestry = "ancestry" if n_pop == 1 else "ancestries"

    log.logger.info(
        f"Prepare {snps.shape[0]} SNPs from {n_pop} {name_ancestry} after"
        + " data cleaning. Specify --verbose for details.",
    )

    return snps, regular_data


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
                pi=data.pi,
                resid_var=resid_var,
                effect_var=effect_var,
                rho=None,
                max_iter=args.max_iter,
                min_tol=args.min_tol,
                threshold=args.threshold,
                purity=args.purity,
                purity_method=args.purity_method,
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
            pi=data.pi,
            resid_var=resid_var,
            effect_var=effect_var,
            rho=rho,
            max_iter=args.max_iter,
            min_tol=args.min_tol,
            threshold=args.threshold,
            purity=args.purity,
            purity_method=args.purity_method,
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
            cv_res = _run_cv(args, cv_data, data.pi)
            sample_size = jnp.squeeze(tmp_result.sample_size)
            io.output_cv(cv_res, sample_size, output, args.trait, args.compress)

    return None


def sushie_wrapper_ss(
    data: io.ssData,
    args: argparse.Namespace,
    snps: pd.DataFrame,
    meta: bool = False,
) -> None:
    """The wrapper function to run SuShiE in regular, meta, or mega.

    Args:
        data: The clean summary data for SuShiE inference.
        args: The command line parameter input.
        snps: The SNP information.
        meta: The indicator whether to prepare datasets for meta SuShiE.

    """

    n_pop = len(data.lds)

    if meta:
        output = f"{args.output}.meta"
        method_type = "meta"
    else:
        output = f"{args.output}.sushie"
        method_type = "sushie"

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

            log.logger.info(
                f"Start fine-mapping using SuSiE on ancestry {idx + 1} with {args.L} effects"
                + " because --meta is specified."
            )

            tmp_result = infer_ss.infer_sushie_ss(
                lds=[data.lds[idx]],
                ns=data.ns[idx][:, jnp.newaxis],
                zs=[data.zs[idx]],
                L=args.L,
                no_update=args.no_update,
                pi=data.pi,
                resid_var=resid_var,
                effect_var=effect_var,
                rho=None,
                max_iter=args.max_iter,
                min_tol=args.min_tol,
                threshold=args.threshold,
                purity=args.purity,
                purity_method=args.purity_method,
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
        log.logger.info(f"Start fine-mapping using SuShiE with {args.L} effects.")
        tmp_result = infer_ss.infer_sushie_ss(
            lds=data.lds,
            ns=data.ns,
            zs=data.zs,
            L=args.L,
            no_update=args.no_update,
            pi=data.pi,
            resid_var=args.resid_var,
            effect_var=args.effect_var,
            rho=args.rho,
            max_iter=args.max_iter,
            min_tol=args.min_tol,
            threshold=args.threshold,
            purity=args.purity,
            purity_method=args.purity_method,
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

    if not meta:
        io.output_corr(result, output, args.trait, args.compress)

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
        if args.summary is True:
            log.logger.info("Start fine-mapping using SuShiE on summary-level data.")

            n_pop, pi, geno_path, geno_func, ld_file = parameter_check_ss(args)

            snps, ss_data = process_raw_ss(
                geno_path,
                geno_func,
                ld_file,
                pi,
                args,
            )

            normal_data = copy.deepcopy(ss_data)
            sushie_wrapper_ss(normal_data, args, snps, meta=False)

            # if only one ancestry, no need to run mega or meta
            if n_pop != 1:
                if args.meta:
                    meta_data = copy.deepcopy(ss_data)
                    sushie_wrapper_ss(meta_data, args, snps, meta=True)

        else:
            log.logger.info("Start fine-mapping using SuShiE on individual-level data.")

            (
                n_pop,
                ancestry_index,
                keep_subject,
                pi,
                geno_path,
                geno_func,
            ) = parameter_check(args)

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
                pi,
                args.keep_ambiguous,
                args.maf,
                args.rint,
                args.no_regress,
                args.mega,
                args.cv,
                args.cv_num,
                args.seed,
                args.chrom,
                args.start,
                args.end,
            )

            normal_data = copy.deepcopy(regular_data)
            sushie_wrapper(normal_data, cv_data, args, snps, meta=False, mega=False)

            # if only one ancestry, no need to run mega or meta
            if n_pop != 1:
                if args.meta:
                    meta_data = copy.deepcopy(regular_data)
                    sushie_wrapper(meta_data, None, args, snps, meta=True, mega=False)

                if args.mega:
                    sushie_wrapper(mega_data, None, args, snps, meta=False, mega=True)

    except Exception as err:
        import traceback

        print("".join(traceback.format_exception(type(err), err, err.__traceback__)))
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
        "--summary",
        action="store_true",
        default=False,
        help=(
            "Indicator whether to run fine-mapping on summary statistics.",
            " Default is False.",
            " If True, the software will need GWAS files as input data by specifying --gwas",
            " and need LD matrix by specifying either --ld or one of the --plink, --vcf, or --bgen.",
            " If False, the software will need phenotype data by specifying --pheno",
            " and genotype data by specifying either --plink, --vcf, or --bgen.",
        ),
    )

    finemap.add_argument(
        "--pheno",
        nargs="+",
        type=str,
        help=(
            "Phenotype data. It has to be a tsv file that contains at least two",
            " columns where the first column is subject ID and",
            " the second column is the continuous phenotypic value. It can be a compressed file (e.g., tsv.gz).",
            " It is okay to have additional columns, but only the first two columns will be used."
            " No headers. Use 'space' to separate ancestries if more than two.",
            " For individual-level data fine-mapping, SuShiE currently only fine-maps on continuous data.",
        ),
    )

    finemap.add_argument(
        "--gwas",
        nargs="+",
        type=str,
        help=(
            "GWAS data. It has to be a tsv file that contains at least six columns.",
            " The six columns are chromsome number, SNP ID, base-pair position, counting allele,",
            " non-counting allele, and the z-score. Users can can specify column names using --gwas-header.",
            " The chromsome number needs to be integer number instead of 'chr1' or 'chrom1'.",
            " By default, the software assumes the header names are 'chrom', 'snp', 'pos', 'a1', 'a0', 'z'.",
            " It can be a compressed file (e.g., tsv.gz).",
            " It is okay to have additional columns, but only the mentioned columns will be used."
            " Use 'space' to separate ancestries if more than two.",
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
            " Data has to only contain bialleic variant.",
            " If used in summary-level fine-mapping, the SNP ID has to match the GWAS data in --gwas.",
            " The software will flip the alleles if the counting allele in GWAS data is different from the plink data.",
        ),
    )

    finemap.add_argument(
        "--vcf",
        nargs="+",
        type=str,
        default=None,
        help=(
            "Genotype data in vcf format. Use 'space' to separate ancestries if more than two.",
            " Keep the same ancestry order as phenotype's. The software will count RFE allele.",
            " If gt_types is UNKNOWN, it will be coded as NA, and be imputed by allele frequency.",
            " Data has to only contain bialleic variant.",
            " If used in summary-level fine-mapping, the SNP ID has to match the GWAS data in --gwas.",
            " The software will flip the alleles if the counting allele in GWAS data is different from the plink data.",
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
            " Data has to only contain bialleic variant.",
            " If used in summary-level fine-mapping, the SNP ID has to match the GWAS data in --gwas.",
            " The software will flip the alleles if the counting allele in GWAS data is different from the plink data.",
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
        "--ld",
        nargs="+",
        default=None,
        type=str,
        help=(
            "LD files that will be used in the fine-mapping. Default is None."
            " Keep the same ancestry order as GWAS files.",
            " It has to be a tsv or comparessed file (e.g., tsv.gz).",
            " The header has to be the SNP name matching the GWAS data in --gwas",
            " It can have less or more SNPs than the GWAS data,",
            " and the software will find the overlap SNPs.",
            " Users must ensure that the LD and GWAS z statistics are computed using the same counting alleles.",
        ),
    )

    finemap.add_argument(
        "--chrom",
        default=None,
        type=int,
        choices=list(range(1, 23)),
        help=(
            "Chromsome number to subset SNPs in the fine-mapping. Default is None."
            " Value has to be an integer number between 1 and 22.",
            " The SNP position information from the first ancestry will be used for filtering.",
            " If this flag is specified, --start and --end must also be provided.",
        ),
    )

    finemap.add_argument(
        "--start",
        default=None,
        type=int,
        help=(
            "Base-pair start position to subset SNPs in the fine-mapping. Default is None."
            " Value has to be a positive integer number.",
            " The SNP position information from the first ancestry will be used for filtering.",
            " If this flag is specified, --chrom and --end must also be provided.",
        ),
    )

    finemap.add_argument(
        "--end",
        default=None,
        type=int,
        help=(
            "Base-pair end position to subset SNPs in the fine-mapping. Default is None."
            " Value has to be a positive integer number and larger than --start.",
            " The SNP position information from the first ancestry will be used for filtering.",
            " If this flag is specified, --chrom and --start must also be provided.",
        ),
    )

    finemap.add_argument(
        "--sample-size",
        default=None,
        nargs="+",
        type=int,
        help=(
            "GWAS sample size of each ancestry. Default is None."
            " Values have to be positive integer. Use 'space' to separate ancestries if more than two.",
            " The order has to be the same as the GWAS data in --gwas.",
        ),
    )

    finemap.add_argument(
        "--gwas-header",
        nargs="+",
        default=["chrom", "snp", "pos", "a1", "a0", "z"],
        type=str,
        help=(
            "GWAS file header names. Default is ['chrom', 'snp', 'pos', 'a1', 'a0', 'z'].",
            " Users can specify the header names for the GWAS data in this order.",
        ),
    )

    finemap.add_argument(
        "--gwas-sig",
        default=1.0,
        type=float,
        help=(
            "The significance threshold for SNPs to be included in the fine-mapping.",
            " Default is 1.0. Only SNPs with P value less than this threshold will be included.",
            " It has to be a float number between 0 and 1.",
        ),
    )

    finemap.add_argument(
        "--gwas-sig-type",
        default="at-least",
        choices=["at-least", "all"],
        help=(
            "The cases how to include significant SNPs in the fine-mapping across ancestries.",
            " If it is 'at-least', the software will include SNPs that are significant in at least one ancestry.",
            " If it is 'all', the software will include SNPs that are significant in all ancestries.",
            " Default is 'at-least'.",
            " The significant threshold is specified by --gwas-sig.",
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

    # fine-map prior options
    finemap.add_argument(
        "--pi",
        default="uniform",
        type=str,
        help=(
            "Prior probability for each SNP to be causal.",
            " Default is uniform (i.e., 1/p where p is the number of SNPs in the region.",
            " It is the fixed across all ancestries.",
            " Alternatively, users can specify the file path that contains the prior weights for each SNP.",
            " The weights have to be positive value.",
            " The weights will be normalized to sum to 1 before inference.",
            " The file has to be a tsv file that contains two columns where the",
            " first column is the SNP ID and the second column is the prior weights.",
            " Additional columns will be ignored.",
            " For SNPs do not have prior weights in the file, it will be assigned the average value of the rest.",
            " It can be a compressed file (e.g., tsv.gz). No headers.",
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
        default=0.95,
        type=float,
        help=(
            "Specify the PIP threshold for SNPs to be included in the credible sets. Default is 0.95.",
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
        "--purity-method",
        default="weighted",
        type=str,
        choices=["weighted", "max", "min"],
        help=(
            "Specify the method to compute purity across ancestries.",
            " Users choose 'weighted', 'max', or 'min'.",
            " `weighted` is the sum of the purity of each ancestry weighted by the sample size.",
            " `max` is the maximum purity value across ancestries.",
            " `min` is the minimum purity value across ancestries.",
            " Default is weighted.",
        ),
    )

    finemap.add_argument(
        "--ld-adjust",
        default=0,
        type=float,
        help=(
            "The adjusting number to LD diagonal to ensure the positive definiteness.",
            " It has to be positive integer number between 0 and 0.1. Default is 0.",
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
            "Indicator to re-order single effects based on Frobenius norm of effect size covariance prior."
            " Default is False (to re-order).",
            " Specify --no-reorder will store 'True' value.",
        ),
    )

    finemap.add_argument(
        "--keep-ambiguous",
        default=False,
        action="store_true",
        help=(
            "Indicator to keep ambiguous SNPs (i.e., A/T, T/A, C/G, or G/C pairs) from the genotypes.",
            " Recommend to remove these SNPs if each ancestry data is from different studies",
            " or plan to use the inference results for downstream analysis with other datasets."
            " Default is False (not to keep).",
            " Specify --keep-ambiguous will store 'True' value.",
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
            " for future FUSION pipeline. Default is False. ",
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
        help="Indicator to not print message to console. Default is False. Specify --quiet will store 'True' value.",
    )

    finemap.add_argument(
        "--verbose",
        default=False,
        action="store_true",
        help=(
            "Indicator to include debug information in the log. Default is False.",
            " Specify --verbose will store 'True' value.",
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
                "--keep-ambiguous",
                "--summary",
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
