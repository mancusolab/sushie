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

from . import compute, core, infer, io

# from .log import LOG

LOG = "sushie"
warnings.filterwarnings("ignore")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import jax.numpy as jnp


def get_command_string(args):
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
            + " opt_mode. Inference may be inaccurate.",
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


def run_finemap(args):
    log = logging.getLogger(LOG)

    try:
        resid_var, effect_var, rho = parameter_check(args)

        clean_data = compute.process_raw(
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
        io._output_cs(args, result, clean_data)

        if args.h2g:
            io._output_h2g(args, result, clean_data)

        if args.weights:
            io._output_weights(args, result, clean_data)

        if args.corr:
            io._output_corr(args, result)

        if args.cv:
            log.info(f"Starting {args.cv_num}-fold cross validation.")
            io._output_cv(args, clean_data, resid_var, effect_var, rho)

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
            " 'noop' does not update prior effect covariance matrix at each iteration. The inference may be inaccurate",
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
