#!/usr/bin/env python

from __future__ import division

import argparse
import logging
import os
import sys
import warnings
from importlib import metadata
from typing import Callable, List, Tuple

import pandas as pd

from . import core, infer, io

# from .log import LOG

LOG = "sushie"
warnings.filterwarnings("ignore")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import jax.numpy as jnp


def _get_command_string(args):
    """
    Format sushie call and options into a string for logging/printing
    :return: string containing formatted arguments to sushie
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


def _parameter_check(
    args,
) -> Tuple[List[str], Callable]:
    log = logging.getLogger(LOG)

    n_pop = len(args.pheno)
    log.info(f"Detecting phenotypes for {n_pop} ancestries for SuShiE fine-mapping.")

    n_geno = (
        int(args.plink is not None)
        + int(args.vcf is not None)
        + int(args.bgen is not None)
    )

    if n_geno > 1:
        log.warning(
            f"Detecting {n_geno} genotypes, will use the genotypes in the order of 'plink, vcf, and bgen'"
        )

    # decide genotype data
    if args.plink is not None:
        if len(args.plink) != n_pop:
            raise ValueError(
                "The numbers of ancestries in plink geno and pheno data does not match. Check your input."
            )
        else:
            log.info("Detecting genotype data in plink format.")
            geno_path = args.plink
            geno_func = io._construct_read_geno("plink")
    elif args.vcf is not None:
        if len(args.vcf) != n_pop:
            raise ValueError(
                "The numbers of ancestries in vcf geno and pheno data does not match. Check your input."
            )
        else:
            log.info("Detecting genotype data in vcf format.")
            geno_path = args.vcf
            geno_func = io._construct_read_geno("vcf")
    elif args.bgen is not None:
        if len(args.bgen) != n_pop:
            raise ValueError(
                "The numbers of ancestries in bgen geno and pheno data does not match. Check your input."
            )
        else:
            log.info("Detecting genotype data in bgen format.")
            geno_path = args.bgen
            geno_func = io._construct_read_geno("bgen")
    else:
        raise ValueError(
            "No genotype data specified in either plink, vcf, or bgen format. Check your input."
        )

    if args.covar is not None:
        if len(args.covar) != n_pop:
            raise ValueError("The number of covariate data does not match geno data.")
    else:
        log.info("No covariates detected for this analysis.")

    if args.cv:
        if args.cv_num <= 1:
            raise ValueError(
                "The number of folds in cross validation is invalid."
                + " Choose some number greater than 1.",
            )
        elif args.cv_num > 5:
            log.warning(
                "The number of folds in cross validation is too large."
                + " It may increase running time.",
            )

        if args.seed <= 0:
            raise ValueError(
                "The seed specified for CV is invalid. Please choose a positive integer."
            )

    if args.meta and n_pop == 1:
        log.warning(
            "Running meta SuSiE, but the number of ancestry is 1, which is redundant."
        )

    if args.mega and n_pop == 1:
        log.warning(
            "Running mega SuSiE, but the number of ancestry is 1, which is redundant."
        )

    return geno_path, geno_func


def _drop_nainf(rawData: core.RawData) -> Tuple[core.RawData, int]:
    _, fam, bed, pheno, covar = rawData

    del_idx = jnp.array([], dtype=int)
    del_idx = jnp.append(del_idx, jnp.where(jnp.isnan(bed).any(axis=1))[0])
    del_idx = jnp.append(del_idx, jnp.where(jnp.isinf(bed).any(axis=1))[0])

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


def _reset_idx(rawData: core.RawData, idx: int) -> core.RawData:
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


def _filter_common_ind(rawData: core.RawData, idx: int) -> core.RawData:
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
    correct = jnp.logical_and(baseA0 == compareA0, baseA1 == compareA1)
    flipped = jnp.logical_and(baseA0 == compareA1, baseA1 == compareA0)
    correct_idx = jnp.where(correct)[0]
    flipped_idx = jnp.where(flipped)[0]
    wrong_idx = jnp.where(jnp.logical_not(jnp.logical_or(correct, flipped)))[0]

    return correct_idx, flipped_idx, wrong_idx


def _process_raw(
    rawData: List[core.RawData],
) -> Tuple[List[jnp.ndarray], List[jnp.ndarray], core.ArrayOrNoneList, pd.DataFrame]:
    """
    Perform an ordinary linear regression.

    :param X: jnp.ndarray n x p matrix for independent variables with no intercept vector.
    :param y: jnp.ndarray n x m matrix for dependent variables. If m > 1, then perform m ordinary regression parallel.

    :return: typing.Tuple[jnp.ndarray, core.ArrayOrFloat, core.ArrayOrFloat] returns residuals, adjusted r squared,
            and p values for betas.
    """
    log = logging.getLogger(LOG)
    n_pop = len(rawData)

    for idx in range(n_pop):
        # remove NA/inf value
        old_num = rawData[idx].fam.shape[0]
        rawData[idx], del_num = _drop_nainf(rawData[idx])

        if del_num != 0:
            log.warning(
                f"Ancestry {idx + 1}: Drop {del_num} out of {old_num} subjects because of INF or NAN value"
                + "in either genotype, phenotype, or covariate data."
            )

        # reset index and add index column to all dataset for future inter-ancestry or inter-dataset processing
        rawData[idx] = _reset_idx(rawData[idx], idx)

        # find common individuals across geno, pheno, and covar within an ancestry
        rawData[idx] = _filter_common_ind(rawData[idx], idx)

        if rawData[idx].fam.shape[0] == 0:
            raise ValueError(
                f"Ancestry {idx + 1}: No common individuals across phenotype, covariates, "
                + "genotype found. Please double check source data.",
            )
        else:
            log.info(
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
            log.warning(
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
                log.warning(
                    f"Ancestry{idx + 1} has {len(tmp_flip_idx)} flipped alleles from ancestry 1. Will flip these SNPs."
                )

            # save the index for future swapping
            flip_idx.append(tmp_flip_idx)

            if len(wrong_idx) != 0:
                snps = snps.drop(index=wrong_idx)
                log.warning(
                    f"Ancestry{idx + 1} has {len(wrong_idx)} alleles that couldn't be flipped. Will remove these SNPs."
                )

            if snps.shape[0] == 0:
                raise ValueError(
                    f"Ancestry {idx + 1} has none of correct or flippable SNPs from ancestry 1. Check your source.",
                )
            # drop unused columns
            snps = snps.drop(columns=[f"a0_{idx + 1}", f"a1_{idx + 1}", f"pos_{idx+1}"])
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

        if tmp_covar is not None:
            # select the common individual for covar
            common_covar_id = tmp_fam[f"covarIDX_{idx + 1}"].values
            n_covar = tmp_covar.shape[1]
            tmp_covar = tmp_covar.iloc[common_covar_id, 2:n_covar].values

        geno.append(tmp_geno)
        pheno.append(tmp_pheno)
        covar.append(tmp_covar)

    log.info(
        f"Successfully prepare genotype ({geno[0].shape[1]} SNPs) and phenotype data for {n_pop}"
        + " ancestries, and start fine-mapping using SuShiE.",
    )

    return geno, pheno, covar, snps


def run_meta(
    geno: List[jnp.ndarray],
    pheno: List[jnp.ndarray],
    covar: core.ArrayOrNoneList,
    snps: pd.DataFrame,
    args: argparse.Namespace,
    save_file: bool = True,
) -> pd.DataFrame:
    n_pop = len(geno)

    resid_var = None
    effect_var = None

    pips = jnp.zeros((snps.shape[0], 1))
    cs_table = []
    for idx in range(n_pop):
        if args.resid_var is not None:
            resid_var = [float(args.resid_var[idx])]

        if args.effect_var is not None:
            effect_var = [float(args.effect_var[idx])]

        tmp_result = infer.run_sushie(
            [geno[idx]],
            [pheno[idx]],
            [covar[idx]],
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
        )

        pips = jnp.append(pips, tmp_result.pip[:, jnp.newaxis], axis=1)
        tmp_cs_table = io.output_cs(
            tmp_result, snps, args.output, args.trait, save_file=False
        )
        tmp_cs_table["ancestry"] = idx + 1
        cs_table.append(tmp_cs_table)

    pips = jnp.delete(pips, 0, 1)
    pips = 1 - jnp.prod(1 - pips, axis=1)

    final_cs = pd.DataFrame(columns=cs_table[0].columns.values)

    for idx in range(n_pop):
        tmp_table: pd.DataFrame = cs_table[idx]
        tmp_table["meta_pip"] = pips[tmp_table.SNPIndex.values.astype(int)]
        final_cs = pd.concat([final_cs, tmp_table], axis=0)

    if save_file:
        final_cs.to_csv(f"{args.output}.meta.tsv", sep="\t", index=False)

    return final_cs


def run_mega(
    geno: List[jnp.ndarray],
    pheno: List[jnp.ndarray],
    covar: core.ArrayOrNoneList,
    snps: pd.DataFrame,
    args: argparse.Namespace,
    save_file: bool = True,
) -> pd.DataFrame:
    n_pop = len(geno)

    resid_var = args.resid_var if n_pop == 1 else None
    effect_var = args.effect_var if n_pop == 1 else None
    rho = args.rho if n_pop == 1 else None

    mega_geno = geno[0]
    mega_pheno = pheno[0]
    mega_covar = None

    if covar[0] is not None:
        mega_covar = covar[0]

    for idx in range(1, n_pop):
        mega_geno = jnp.append(mega_geno, geno[idx], axis=0)
        mega_pheno = jnp.append(mega_pheno, pheno[idx], axis=0)
        if covar[idx] is not None:
            mega_covar = jnp.append(mega_covar, covar[idx], axis=0)

    mega_result = infer.run_sushie(
        [mega_geno],
        [mega_pheno],
        [mega_covar],
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
    )

    mega_table = io.output_cs(
        mega_result, snps, f"{args.output}.mega", args.trait, save_file=save_file
    )

    return mega_table


def run_finemap(args):
    log = logging.getLogger(LOG)

    try:
        geno_path, geno_func = _parameter_check(args)

        rawData = io.read_data(args.pheno, args.covar, geno_path, geno_func)

        geno, pheno, covar, snps = _process_raw(rawData)

        result = infer.run_sushie(
            geno,
            pheno,
            covar,
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

        # output credible set
        io.output_cs(result, snps, args.output, args.trait)

        if args.her:
            io.output_her(result, geno, pheno, covar, args.output, args.trait)

        if args.weight:
            io.output_weight(result, snps, args.output, args.trait)

        if args.corr:
            io.output_corr(result, args.output, args.trait)

        if args.numpy:
            io.output_numpy(result, args.output)

        if args.cv:
            log.info(f"Starting {args.cv_num}-fold cross validation.")
            io.output_cv(geno, pheno, covar, args)

        if args.meta:
            log.info("Starting running meta SuSiE.")
            run_meta(geno, pheno, covar, snps, args)

        if args.mega:
            log.info("Starting running mega SuSiE.")
            run_mega(geno, pheno, covar, snps, args)

    except Exception as err:
        import traceback

        print(
            "".join(
                traceback.format_exception(
                    etype=type(err), value=err, tb=err.__traceback__
                )
            )
        )
        log.error(err)

    finally:
        log.info(
            "Finished SuShiE fine-mapping. Thanks for using our software."
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
            " columns where the first column is subject ID as the IID in plink fam file and",
            " the second column is the continuous phenotypic value. Only the first two columns will be used."
            " No headers. Use space to separate ancestries if more than two.",
        ),
    )

    # main arguments
    finemap.add_argument(
        "--plink",
        nargs="+",
        type=str,
        default=None,
        help=(
            "Genotype data in plink format. Use space to separate ancestries if more than two.",
            " Keep the same ancestry order as phenotype's.",
        ),
    )

    finemap.add_argument(
        "--vcf",
        nargs="+",
        type=str,
        default=None,
        help=(
            "Genotype data in vcf format. Use space to separate ancestries if more than two.",
            " Keep the same ancestry order as phenotype's.",
        ),
    )

    finemap.add_argument(
        "--bgen",
        nargs="+",
        type=str,
        default=None,
        help=(
            "Genotype data in bgen 1.3 format. Use space to separate ancestries if more than two.",
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
            " first column is the subject ID as the IID in plink fam file.",
            " Pre-converting the character covariates into dummy variables is required. No headers.",
            " All the columns will be used. Use space to separate ancestries if more than two.",
            " Keep the same ancestry order as phenotype's.",
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
        "--effect_var",
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

    # fine-map inference options
    finemap.add_argument(
        "--no_scale",
        default=False,
        action="store_true",
        help=(
            "Indicator to scale the genotype and phenotype data by standard deviation.",
            " Default is False (scaling)."
            " Specify --no_scale will store True value, and may have different inference.",
        ),
    )

    finemap.add_argument(
        "--no_regress",
        default=False,
        action="store_true",
        help=(
            "Indicator to regress the covariates on each SNP. Default is False (regressing).",
            " Specify --no_regress will store True value, and can slow the inference, but can be more accurate.",
        ),
    )

    finemap.add_argument(
        "--no_update",
        default=False,
        action="store_true",
        help=(
            "Indicator to update effect covariance prior before running single effect regression.",
            " Default is False (updating).",
            " Specify --no_update will store True value; it can slow the inference or have different one.",
            " The updating algorithm is similar to EM algorithm that computes the prior covariance conditioned",
            " on other parameters.",
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

    # I/O option
    finemap.add_argument(
        "--meta",
        default=False,
        action="store_true",
        help=(
            "Indicator to perform meta analysis of single-ancestry SuSiE and output *.pip.meta.tsv file.",
            " Default is False. Specify --meta will store True value and increase running time.",
        ),
    )

    finemap.add_argument(
        "--mega",
        default=False,
        action="store_true",
        help=(
            "Indicator to perform mega SuSiE by stacking the genotype and phenotype data across ancestries.",
            " It will output *.pip.mega.tsv file.",
            " Default is False. Specify --mega will store True value and increase running time.",
        ),
    )

    finemap.add_argument(
        "--her",
        default=False,
        action="store_true",
        help=(
            "Indicator to perform heritability analysis using limix and output *.h2g.tsv file.",
            " Default is False. Specify --her will store True value and increase running time.",
            " *.h2g.tsv file contains two estimated h2g using all genotypes and using only SNPs in the credible sets.",
        ),
    )

    finemap.add_argument(
        "--weight",
        default=False,
        action="store_true",
        help=(
            "Indicator to output *.weights.tsv file for prediction weights of all SNPs.",
            " Default is False. Specify --weight will store True value and increase running time.",
        ),
    )

    finemap.add_argument(
        "--corr",
        default=False,
        action="store_true",
        help=(
            "Indicator to output *.corr.tsv file.",
            " Default is False. Specify --corr will store True value and increase running time.",
            " *.cv.tsv file contains estimated variance and covariance across ancestries.",
        ),
    )

    finemap.add_argument(
        "--numpy",
        default=False,
        action="store_true",
        help=(
            "Indicator to output *.npy file.",
            " Default is False. Specify --numpy will store True and increase running time.",
            " *.npy file contains all the inference results including credible sets, pips, priors and posteriors",
            " for your own wanted analysis.",
        ),
    )

    finemap.add_argument(
        "--cv",
        default=False,
        action="store_true",
        help=(
            "Indicator to perform cross validation (CV) and output CV results for future FUSION pipline.",
            " Default is False. Specify --cv will store True value and increase running time.",
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
            "Trait, tissue, gene name of the phenotype for better indexing in post-hoc analysis. Default is 'Trait'.",
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
        help="Subcommands: finemap to perform gene expression fine-mapping using SuShiE"
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

    cmd_str = _get_command_string(argsv)

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
        log.propagate = False
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
