#!/usr/bin/env python

from __future__ import division

import argparse
import logging
import math
import os
import sys
import typing
import warnings
from importlib import metadata

import pandas as pd
from pandas_plink import read_plink

from . import core, infer, io, utils

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

    if args.effect_var is not None:
        effect_var = args.effect_var
        if len(effect_var) != n_pop:
            raise ValueError(
                "Number of specified effect prior does not match feature number."
            )
        effect_var = [float(i) for i in effect_var]
        if jnp.any(jnp.array(effect_var) <= 0):
            raise ValueError("The input of effect size prior is invalid (<0).")
    else:
        effect_var = None

    if args.rho is not None:
        rho = args.rho
        exp_num_rho = math.comb(n_pop, 2)
        if len(rho) != exp_num_rho:
            raise ValueError(
                f"Number of specified rho ({len(rho)}) does not match expected"
                + f"number {exp_num_rho}.",
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
        if resid_var is None or effect_var is None or rho is None:
            raise ValueError(
                "'noop' is specified. --resid_var, --effect_var, and --rho cannot be None."
            )

        log.warning(
            "No updates on the effect size prior because of 'noop' setting for"
            + " opt_mode. Inference may be slow.",
        )

        if jnp.any(jnp.array(rho) == 0):
            log.warning(
                "'noop' setting for opt_mode and '0' setting for rho, the effect correlation."
                + " Inference may be inaccurate as it does not account for ancestral correlation.",
            )

    if args.cv:
        if args.cv_num <= 1:
            raise ValueError(
                "The number of folds in cross validation is invalid."
                + " Choose some number greater than 1.",
            )
        elif args.cv_num > 5:
            log.warning(
                "The number of folds in cross validation is too large."
                + " It may cause longer running time.",
            )

        if args.seed <= 0:
            raise ValueError(
                "The seed specified for CV is invalid. Please choose a positive integer."
            )

    return resid_var, effect_var, rho


def _process_raw(
    geno_paths: typing.List[str],
    pheno_paths: typing.List[str],
    covar_paths: typing.List[str] = None,
    norm_X: bool = True,
    norm_y: bool = False,
    h2g: bool = False,
    regress: bool = True,
) -> core.CleanData:
    """
    Perform an ordinary linear regression.

    :param X: jnp.ndarray n x p matrix for independent variables with no intercept vector.
    :param y: jnp.ndarray n x m matrix for dependent variables. If m > 1, then perform m ordinary regression parallel.

    :return: typing.Tuple[jnp.ndarray, core.ArrayOrFloat, core.ArrayOrFloat] returns residuals, adjusted r squared,
            and p values for betas.
    """
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
        tmp_bim = utils._drop_na_inf(tmp_bim, "bim", idx)
        tmp_fam = utils._drop_na_inf(tmp_fam, "fam", idx)
        tmp_pheno = utils._drop_na_inf(tmp_pheno, "pheno", idx)

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
        # keep track of pheno index for future matching the bed file if bed
        # files are shuffled due to merging
        tmp_pheno = tmp_pheno.reset_index().rename(
            columns={"index": f"phenoIDX_{idx + 1}", 0: "fid", 1: "iid"}
        )

        if len(tmp_bim) == 0:
            raise ValueError(
                f"Ancestry {idx + 1}: No genotype data found for ancestry at {geno_paths[idx]}."
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
            tmp_covar = utils._drop_na_inf(tmp_covar, "covar", idx)
            # keep track of covar index for future matching the bed file if
            # bed files are shuffled due to merging
            tmp_covar = tmp_covar.reset_index().rename(
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
                f"Ancestry{idx + 1} has {snps_num_diff} independent SNPs and {snps.shape[0]}"
                + " common SNPs. Will remove these independent SNPs.",
            )
    else:
        snps = bim[0]

    # find flipped reference alleles across ancestries
    flip_idx = []
    if n_pop > 1:
        for idx in range(1, n_pop):
            correct_idx, tmp_flip_idx, wrong_idx = utils.allele_check(
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
                f"Ancestry {idx + 1}: No common individuals between phenotype and "
                + "genotype found. Please double check source data.",
            )
        else:
            log.info(
                f"Ancestry {idx + 1}: Found {n_common} common individuals"
                + " between phenotype and genotype.",
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
        tmp_bed = bed[idx][:, common_ind_id][common_snp_id, :]

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
            tmp_h2g = utils._estimate_her(bed[idx], pheno[idx], covar[idx])
            est_h2g = est_h2g.at[idx].set(tmp_h2g)
    else:
        est_h2g = None

    # regress covar on y
    for idx in range(n_pop):
        if covar[idx] is not None:
            pheno_resid, _, _ = utils.ols(covar[idx], pheno[idx])
            pheno[idx] = pheno_resid
            # regress covar on each SNP, it might be slow, the default is True
            if regress:
                (
                    geno_resid,
                    _,
                    _,
                ) = utils.ols(covar[idx], bed[idx])
                bed[idx] = geno_resid
    log.info(
        f"Successfully prepare genotype ({bed[0].shape[1]} SNPs) and phenotype data for {n_pop}"
        + " ancestries, and start fine-mapping using SuShiE.",
    )

    return core.CleanData(bed, pheno, snps, est_h2g)


def run_finemap(args):
    log = logging.getLogger(LOG)

    try:
        resid_var, effect_var, rho = _parameter_check(args)

        clean_data = utils._process_raw(
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
            effect_var=effect_var,
            rho=rho,
            max_iter=args.max_iter,
            min_tol=args.min_tol,
            opt_mode=args.opt_mode,
            threshold=args.threshold,
            purity=args.purity,
        )

        # output credible set
        io.output_cs(args, result, clean_data)

        if args.h2g:
            io.output_h2g(args, result, clean_data)

        if args.weights:
            io.output_weights(args, result, clean_data)

        if args.corr:
            io.output_corr(args, result)

        if args.numpy:
            io.output_numpy(args, result)

        if args.cv:
            log.info(f"Starting {args.cv_num}-fold cross validation.")
            io.output_cv(args, clean_data, resid_var, effect_var, rho)

    except Exception as err:
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
        "--geno",
        nargs="+",
        type=str,
        help="Genotype data in plink format. Use space to separate ancestries if more than two.",
    )

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
        required=True,
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
            " 'noop' does not update prior effect covariance matrix at each iteration. The inference may be slow",
            " If 'noop' is specified, it requires input for --resid_var, --effect_var, and --rho.",
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
        "--numpy",
        default=False,
        type=bool,
        help=(
            "Indicator to output *.npy file.",
            " Default is False. True will cause longer running time.",
            " *.npy file contains all the inference results including credible sets, pips, priors and posteriors",
            " for your own wanted analysis.",
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
