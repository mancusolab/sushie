import copy
import warnings
from typing import Callable, List, Optional, Tuple

import pandas as pd

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from pandas_plink import read_plink
    from cyvcf2 import VCF
    from bgen_reader import open_bgen
    import jax.numpy as jnp

from . import core, log, utils


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
    bim_list = []
    bed_list = []
    for var in vcf:
        bim_list.append([var.CHROM, var.ID, var.POS, var.ALT[0], var.REF])
        raw_bed = pd.DataFrame(var.gt_bases)[0].str.split("/", expand=True)
        tmp_bed = jnp.array((raw_bed[0] == var.REF) * 1 + (raw_bed[1] == var.REF) * 1)
        bed_list.append(tmp_bed)

    bim = pd.DataFrame(bim_list, columns=["chrom", "snp", "pos", "a0", "a1"])
    bed = jnp.array(bed_list).T

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
    result: List[core.SushieResult],
    meta_pip: Optional[jnp.ndarray],
    snps: pd.DataFrame,
    output: str,
    trait: str,
    no_compress: bool,
    meta: bool,
    mega: bool,
) -> pd.DataFrame:
    """Output credible set in tsv file."""
    cs = pd.DataFrame()

    for idx in range(len(result)):
        tmp_cs = (
            pd.merge(snps, result[idx].cs, how="inner", on=["SNPIndex"])
            .assign(trait=trait, n_snps=snps.shape[0])
            .sort_values(
                by=["CSIndex", "alpha", "c_alpha"], ascending=[True, False, True]
            )
        )

        if meta_pip is not None:
            tmp_cs["meta_pip"] = meta_pip[tmp_cs.SNPIndex.values.astype(int)]

        if meta:
            ancestry_idx = f"ancestry_{idx + 1}"
        elif mega:
            ancestry_idx = "mega"
        else:
            ancestry_idx = "sushie"

        tmp_cs["ancestry"] = ancestry_idx
        cs = pd.concat([cs, tmp_cs], axis=0)

    # add a placeholder better for post-hoc analysis
    if cs.shape[0] == 0:
        cs = cs.append({"trait": trait}, ignore_index=True)

    file_name = f"{output}.cs.tsv.gz" if not no_compress else f"{output}.cs.tsv"

    cs.to_csv(file_name, sep="\t", index=False)

    return cs


def output_weight_pip(
    result: List[core.SushieResult],
    meta_pip: Optional[jnp.ndarray],
    snps: pd.DataFrame,
    output: str,
    trait: str,
    no_compress: bool,
    meta: bool,
    mega: bool,
) -> pd.DataFrame:
    """Output prediction weights in tsv file."""
    n_pop = len(result[0].priors.resid_var)
    n_l = result[0].posteriors.alpha.shape[0]
    weights = copy.deepcopy(snps).assign(trait=trait)

    for idx in range(len(result)):
        if meta:
            cname_idx = [f"ancestry{idx + 1}_single_weight"]
            cname_pip = f"ancestry{idx + 1}_single_pip"
            cname_cs = f"ancestry{idx + 1}_in_cs"
            cname_alpha = [f"ancestry{idx + 1}_l{ldx +1}_alpha" for ldx in range(n_l)]
        elif mega:
            cname_idx = ["mega_weight"]
            cname_pip = "mega_pip"
            cname_cs = "mega_in_cs"
            cname_alpha = [f"mega_l{ldx + 1}_alpha" for ldx in range(n_l)]
        else:
            cname_idx = [f"ancestry{jdx + 1}_sushie_weight" for jdx in range(n_pop)]
            cname_pip = "sushie_pip"
            cname_cs = "sushie_in_cs"
            cname_alpha = [f"sushie_l{ldx + 1}_alpha" for ldx in range(n_l)]

        tmp_weights = pd.DataFrame(
            data=jnp.sum(result[idx].posteriors.post_mean, axis=0),
            columns=cname_idx,
        )

        tmp_weights[cname_pip] = result[idx].pip
        tmp_weights[cname_alpha] = jnp.transpose(result[idx].posteriors.alpha)
        weights = pd.concat([weights, tmp_weights], axis=1)
        weights[cname_cs] = (
            weights["SNPIndex"].isin(result[idx].cs["SNPIndex"].tolist()).astype(int)
        )

    if meta_pip is not None:
        weights["meta_pip"] = meta_pip
        tmp_cs = (weights["ancestry1_in_cs"] == 0) * 1
        for idx in range(1, len(result)):
            tmp_cs = tmp_cs * (weights["ancestry1_in_cs"] == 0) * 1
        weights["meta_in_cs"] = 1 - tmp_cs

    if weights.shape[0] == 0:
        weights = weights.append({"trait": trait}, ignore_index=True)

    file_name = (
        f"{output}.weights.tsv.gz" if not no_compress else f"{output}.weights.tsv"
    )

    weights.to_csv(file_name, sep="\t", index=False)

    return weights


def output_her(
    result: List[core.SushieResult],
    data: core.CleanData,
    output: str,
    trait: str,
    no_compress: bool,
) -> pd.DataFrame:
    """Output heritability results in tsv file."""
    n_pop = len(data.geno)

    her_result = []
    for idx in range(n_pop):
        if data.covar is None:
            tmp_covar = None
        else:
            tmp_covar = data.covar[idx]
        tmp_her_result = utils.estimate_her(data.geno[idx], data.pheno[idx], tmp_covar)
        her_result.append(tmp_her_result)

    est_her = pd.DataFrame(
        data=her_result,
        columns=["genetic_var", "h2g_w_v", "h2g_wo_v", "lrt_stats", "p_value"],
        index=[idx + 1 for idx in range(n_pop)],
    ).reset_index(names="ancestry")

    # only output h2g that has credible sets
    SNPIndex = result[0].cs.SNPIndex.values.astype(int)
    shared_col = [
        "s_genetic_var",
        "s_h2g_w_v",
        "s_h2g_wo_v",
        "s_lrt_stats",
        "s_p_value",
    ]
    est_shared_her = pd.DataFrame(columns=shared_col)
    if len(SNPIndex) != 0:
        shared_her = []
        for idx in range(n_pop):
            if data.covar is None:
                tmp_covar = None
            else:
                tmp_covar = data.covar[idx]

            tmp_shared_her = utils.estimate_her(
                data.geno[idx][:, SNPIndex], data.pheno[idx], tmp_covar
            )
            shared_her.append(tmp_shared_her)

        est_shared_her = pd.DataFrame(
            data=shared_her, columns=shared_col, index=[idx + 1 for idx in range(n_pop)]
        ).reset_index(names="ancestry")
    est_her = pd.concat([est_her, est_shared_her], axis=1).assign(trait=trait)

    if est_her.shape[0] == 0:
        est_her = est_her.append({"trait": trait}, ignore_index=True)

    file_name = f"{output}.h2g.tsv.gz" if not no_compress else f"{output}.h2g.tsv"

    est_her.to_csv(file_name, sep="\t", index=False)

    return est_her


def output_corr(
    result: List[core.SushieResult],
    output: str,
    trait: str,
    no_compress: bool,
) -> pd.DataFrame:
    """Output correlation results in tsv file."""
    n_pop = len(result[0].priors.resid_var)

    CSIndex = jnp.unique(result[0].cs.CSIndex.values.astype(int))
    # only output after-purity CS
    corr_cs_only = jnp.transpose(result[0].posteriors.weighted_sum_covar[CSIndex - 1])
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

    file_name = f"{output}.corr.tsv.gz" if not no_compress else f"{output}.corr.tsv"

    corr.to_csv(file_name, sep="\t", index=False)

    return corr


def output_cv(
    cv_res: List,
    sample_size: List[int],
    output: str,
    trait: str,
    no_compress: bool,
) -> pd.DataFrame:
    """Output cross validation results in tsv file."""
    cv_r2 = (
        pd.DataFrame(
            data=cv_res,
            index=[idx + 1 for idx in range(len(sample_size))],
            columns=["rsq", "p_value"],
        )
        .reset_index(names="ancestry")
        .assign(N=sample_size, trait=trait)
    )

    if cv_r2.shape[0] == 0:
        cv_r2 = cv_r2.append({"trait": trait}, ignore_index=True)

    file_name = f"{output}.cv.tsv.gz" if not no_compress else f"{output}.cv.tsv"

    cv_r2.to_csv(file_name, sep="\t", index=False)

    return cv_r2


def output_numpy(result: List[core.SushieResult], output: str) -> None:
    """Output all results in npy file."""

    jnp.save(f"{output}.all.results.npy", result)

    return None
