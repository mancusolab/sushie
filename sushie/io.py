import argparse
import copy
import warnings
from typing import Callable, List, Tuple

import pandas as pd

from jax import random

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from pandas_plink import read_plink
    from cyvcf2 import VCF
    from bgen_reader import open_bgen
    import jax.numpy as jnp

from . import core, infer, log, utils


def read_data(
    pheno_paths: List[str],
    covar_paths: core.ListStrOrNone,
    geno_paths: List[str],
    geno_func: Callable,
) -> List[core.RawData]:
    """Read in pheno, covar, and genotype data and convert it to raw data object."""
    n_pop = len(pheno_paths)

    rawData = []

    for idx in range(n_pop):
        log.logger.info(f"Ancestry {idx + 1}: Reading in genotype data.")

        tmp_bim, tmp_fam, tmp_bed = geno_func(geno_paths[idx])

        if len(tmp_bim) == 0:
            raise ValueError(
                f"Ancestry {idx + 1}: No genotype data found for ancestry at {geno_paths[idx]}."
            )
        if len(tmp_fam) == 0:
            raise ValueError(
                f"Ancestry {idx + 1}: No fam data found for ancestry at {geno_paths[idx]}."
            )

        tmp_pheno = (
            pd.read_csv(pheno_paths[idx], sep="\t", header=None, dtype={0: object})
            .rename(columns={0: "iid", 1: "pheno"})
            .reset_index(drop=True)
        )

        if len(tmp_pheno) == 0:
            raise ValueError(
                f"Ancestry {idx + 1}: No pheno data found for ancestry at {pheno_paths[idx]}."
            )

        if covar_paths is not None:
            tmp_covar = (
                pd.read_csv(covar_paths[idx], sep="\t", header=None, dtype={0: object})
                .rename(columns={0: "iid"})
                .reset_index(drop=True)
            )
        else:
            tmp_covar = None

        rawData.append(
            core.RawData(
                bim=tmp_bim, fam=tmp_fam, bed=tmp_bed, pheno=tmp_pheno, covar=tmp_covar
            )
        )

    return rawData


def read_triplet(path: str) -> Tuple[pd.DataFrame, pd.DataFrame, jnp.ndarray]:
    """Read in genotype data in plink format."""
    bim, fam, bed = read_plink(path, verbose=False)
    bim = bim[["chrom", "snp", "pos", "a0", "a1"]]
    fam = fam[["iid"]]
    # we want bed file to be nxp
    bed = bed.compute().T
    return bim, fam, bed


def read_vcf(path: str) -> Tuple[pd.DataFrame, pd.DataFrame, jnp.ndarray]:
    """Read in genotype data in vcf format."""
    vcf = VCF(path)
    fam = pd.DataFrame(vcf.samples).rename(columns={0: "iid"})
    bim = pd.DataFrame(columns=["chrom", "snp", "pos", "a0", "a1"])
    # add a placeholder
    bed = jnp.ones((1, fam.shape[0]))
    for var in vcf:
        tmp_bim = pd.DataFrame(
            data={
                "chrom": var.CHROM,
                "snp": var.ID,
                "pos": var.POS,
                "a0": var.ALT,
                "a1": var.REF,
            }
        )
        bim = pd.concat([bim, tmp_bim])
        raw_bed = pd.DataFrame(var.gt_bases)[0].str.split("/", expand=True)
        tmp_bed = jnp.array((raw_bed[0] == var.REF) * 1 + (raw_bed[1] == var.REF) * 1)
        bed = jnp.append(
            bed,
            tmp_bed[
                jnp.newaxis,
            ],
            axis=0,
        )
    # delete place holder, and transpose it
    bed = jnp.delete(bed, 0, 0).T
    bim = bim.reset_index(drop=True)

    return bim, fam, bed


def read_bgen(path: str) -> Tuple[pd.DataFrame, pd.DataFrame, jnp.ndarray]:
    """Read in genotype data in bgen format."""
    bgen = open_bgen(path, verbose=False)
    fam = pd.DataFrame(bgen.samples).rename(columns={0: "iid"})
    bim = pd.DataFrame(
        data={"chrom": bgen.chromosomes, "snp": bgen.rsids, "pos": bgen.positions}
    )
    allele = (
        pd.DataFrame(bgen.allele_ids)[0]
        .str.split(",", expand=True)
        .rename(columns={0: "a0", 1: "a1"})
    )
    bim = pd.concat([bim, allele], axis=1).reset_index(drop=True)[
        ["chrom", "snp", "pos", "a0", "a1"]
    ]
    bed = jnp.einsum("ijk,k->ij", bgen.read(), jnp.array([0, 1, 2]))

    return bim, fam, bed


# output functions
def output_cs(
    result: core.SushieResult,
    snps: pd.DataFrame,
    output: str,
    trait: str,
    save_file: bool = True,
) -> pd.DataFrame:
    """Output credible set in tsv file."""
    cs = result.cs
    cs["pip"] = result.pip[result.cs.SNPIndex.values.astype(int)]
    cs = (
        pd.merge(snps, result.cs, how="inner", on=["SNPIndex"])
        # .drop(columns=["SNPIndex"])
        .assign(trait=trait).sort_values(
            by=["CSIndex", "alpha", "c_alpha"], ascending=[True, False, True]
        )
    )

    # add a placeholder better for post-hoc analysis
    if cs.shape[0] == 0:
        cs = cs.append({"trait": trait}, ignore_index=True)

    if save_file:
        cs.to_csv(f"{output}.cs.tsv", sep="\t", index=False)

    return cs


def output_her(
    result: core.SushieResult,
    geno: List[jnp.ndarray],
    pheno: List[jnp.ndarray],
    covar: core.ArrayOrNoneList,
    output: str,
    trait: str,
    save_file: bool = True,
) -> pd.DataFrame:
    """Output heritability results in tsv file."""
    n_pop = len(result.priors.resid_var)
    est_h2g = jnp.zeros(n_pop)
    p_value = jnp.zeros(n_pop)
    for idx in range(n_pop):
        tmp_h2g, tmp_p_value = utils.estimate_her(geno[idx], pheno[idx], covar[idx])
        est_h2g = est_h2g.at[idx].set(tmp_h2g)
        p_value = p_value.at[idx].set(tmp_p_value)
    est_her = (
        pd.DataFrame(
            data={"h2g": est_h2g, "h2g_p": p_value},
            index=[idx for idx in range(n_pop)],
        )
        .assign(trait=trait)
        .reset_index()
        .rename(columns={"index": "ancestry"})
    )

    # only output h2g that has credible sets
    SNPIndex = result.cs.SNPIndex.values.astype(int)
    if len(SNPIndex) != 0:
        shared_h2g = jnp.zeros(n_pop)
        p_value = jnp.zeros(n_pop)
        for idx in range(n_pop):
            tmp_shared_h2g, tmp_p_value = utils.estimate_her(
                geno[idx][:, SNPIndex], pheno[idx], covar[idx]
            )
            shared_h2g = shared_h2g.at[idx].set(tmp_shared_h2g)
            p_value = p_value.at[idx].set(tmp_p_value)

        est_her["shared_h2g"] = shared_h2g
        est_her["shared_h2g_p"] = p_value

    if est_her.shape[0] == 0:
        est_her = est_her.append({"trait": trait}, ignore_index=True)

    if save_file:
        est_her.to_csv(f"{output}.h2g.tsv", sep="\t", index=False)

    return est_her


def output_weight(
    result: core.SushieResult,
    snps: pd.DataFrame,
    output: str,
    trait: str,
    save_file: bool = True,
) -> pd.DataFrame:
    """Output prediction weights in tsv file."""
    n_pop = len(result.priors.resid_var)

    snp_copy = copy.deepcopy(snps).assign(trait=trait)
    tmp_weights = pd.DataFrame(
        data=jnp.sum(result.posteriors.post_mean, axis=0),
        columns=[f"ancestry{idx + 1}_sushie" for idx in range(n_pop)],
    )
    weights = pd.concat([snp_copy, tmp_weights], axis=1)

    if weights.shape[0] == 0:
        weights = weights.append({"trait": trait}, ignore_index=True)

    if save_file:
        weights.to_csv(f"{output}.weights.tsv", sep="\t", index=False)

    return weights


def output_corr(
    result: core.SushieResult,
    output: str,
    trait: str,
    save_file: bool = True,
) -> pd.DataFrame:
    """Output correlation results in tsv file."""
    n_pop = len(result.priors.resid_var)

    CSIndex = jnp.unique(result.cs.CSIndex.values.astype(int))
    # only output after-purity CS
    corr_cs_only = jnp.transpose(result.posteriors.post_covar[CSIndex - 1])
    corr = pd.DataFrame(data={"trait": trait, "CSIndex": CSIndex})
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

    if corr.shape[0] == 0:
        corr = corr.append({"trait": trait}, ignore_index=True)

    if save_file:
        corr.to_csv(f"{output}.corr.tsv", sep="\t", index=False)

    return corr


def output_cv(
    geno: List[jnp.ndarray],
    pheno: List[jnp.ndarray],
    covar: core.ArrayOrNoneList,
    args: argparse.Namespace,
    save_file: bool = True,
) -> pd.DataFrame:
    """Output cross validation results in tsv file."""
    rng_key = random.PRNGKey(args.seed)
    cv_geno = copy.deepcopy(geno)
    cv_pheno = copy.deepcopy(pheno)
    cv_covar = copy.deepcopy(covar)
    n_pop = len(geno)

    # shuffle the data first
    for idx in range(n_pop):
        rng_key, c_key = random.split(rng_key, 2)
        tmp_n = cv_geno[idx].shape[0]
        shuffled_index = random.choice(c_key, tmp_n, (tmp_n,), replace=False)
        cv_pheno[idx] = cv_pheno[idx][shuffled_index]
        cv_geno[idx] = cv_geno[idx][shuffled_index]
        tmp_covar = cv_covar[idx]

        if tmp_covar is not None:
            cv_covar[idx] = tmp_covar[shuffled_index]

    # create a list to store future estimated y value
    est_y = [jnp.array([])] * n_pop
    for cv in range(args.cv_num):
        test_X = []
        train_X = []
        train_y = []
        train_covar = [None] * n_pop
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
            tmp_covar = cv_covar[idx]
            if tmp_covar is not None:
                train_covar.append(tmp_covar[jnp.r_[:start, end:tmp_n]])

        cv_result = infer.run_sushie(
            train_X,
            train_y,
            train_covar,
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
            est_y[idx] = jnp.append(est_y[idx], test_X[idx] @ tmp_cv_weight)

    cv_res = []
    for idx in range(n_pop):
        _, _, adj_r2, p_value = utils.ols(est_y[idx][:, jnp.newaxis], cv_pheno[idx])
        cv_res.append([adj_r2, p_value[1]])

    sample_size = [i.shape[0] for i in geno]
    cv_r2 = (
        pd.DataFrame(
            data=cv_res,
            index=[idx for idx in range(n_pop)],
            columns=["rsq", "p_value"],
        )
        .reset_index()
        .rename(columns={"index": "ancestry"})
        .assign(N=sample_size)
    )

    if cv_r2.shape[0] == 0:
        cv_r2 = cv_r2.append({"trait": args.trait}, ignore_index=True)

    if save_file:
        cv_r2.to_csv(f"{args.output}.cv.tsv", sep="\t", index=False)

    return cv_r2


def output_numpy(result: core.SushieResult, output: str) -> None:
    """Output all results in npy file."""
    jnp.save(f"{output}.all.results.npy", result)

    return None


def run_meta(
    geno: List[jnp.ndarray],
    pheno: List[jnp.ndarray],
    covar: core.ArrayOrNoneList,
    snps: pd.DataFrame,
    args: argparse.Namespace,
    save_file: bool = True,
) -> pd.DataFrame:
    """Run single ancestry SuShiE and calculate the meta-analyzed PIP."""
    n_pop = len(geno)

    resid_var = None
    effect_var = None

    pips = jnp.zeros((snps.shape[0], 1))
    cs_table = []
    # args.resid_var is either None or a list of float, so we have to do so
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
        tmp_cs_table = output_cs(
            tmp_result, snps, args.output, args.trait, save_file=False
        )
        tmp_cs_table["ancestry"] = idx + 1
        cs_table.append(tmp_cs_table)

    pips = jnp.delete(pips, 0, 1)
    pips = 1 - jnp.prod(1 - pips, axis=1)

    final_cs = pd.DataFrame(columns=cs_table[0].columns.values)

    for idx in range(n_pop):
        tmp_table = cs_table[idx]

        if tmp_table.shape[0] == 1 and tmp_table.snp.isna().values.any():
            tmp_table["meta_pip"] = jnp.nan
        else:
            tmp_table["meta_pip"] = pips[tmp_table.SNPIndex.values.astype(int)]

        final_cs = pd.concat([final_cs, tmp_table], axis=0)

    if save_file:
        final_cs.to_csv(f"{args.output}.meta.cs.tsv", sep="\t", index=False)

    return final_cs


def run_mega(
    geno: List[jnp.ndarray],
    pheno: List[jnp.ndarray],
    covar: core.ArrayOrNoneList,
    snps: pd.DataFrame,
    args: argparse.Namespace,
    save_file: bool = True,
) -> pd.DataFrame:
    """Run "mega" SuShiE that row binds the geno and pheno data across ancestry and perform single ancestry SuShiE."""
    n_pop = len(geno)

    resid_var = args.resid_var if n_pop == 1 else None
    effect_var = args.effect_var if n_pop == 1 else None
    rho = args.rho if n_pop == 1 else None

    # it's possible that different ancestries have different number of covariates,
    # so we need to regress out first
    if covar[0] is not None:
        for idx in range(n_pop):
            pheno[idx], _, _, _ = utils.ols(covar[idx], pheno[idx])

            if not args.no_regress:
                (
                    geno[idx],
                    _,
                    _,
                    _,
                ) = utils.ols(covar[idx], geno[idx])

    mega_geno = geno[0]
    mega_pheno = pheno[0]
    for idx in range(1, n_pop):
        mega_geno = jnp.append(mega_geno, geno[idx], axis=0)
        mega_pheno = jnp.append(mega_pheno, pheno[idx], axis=0)

    mega_result = infer.run_sushie(
        [mega_geno],
        [mega_pheno],
        [None],
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

    mega_table = output_cs(
        mega_result, snps, f"{args.output}.mega", args.trait, save_file=save_file
    )

    return mega_table
