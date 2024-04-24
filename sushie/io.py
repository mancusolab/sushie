import copy
import warnings
from typing import Callable, List, NamedTuple, Optional, Tuple

import pandas as pd

from jax import Array

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from pandas_plink import read_plink
    from cyvcf2 import VCF
    from bgen_reader import open_bgen
    import jax.numpy as jnp

from . import infer, utils

__all__ = [
    "CVData",
    "CleanData",
    "RawData",
    "read_data",
    "read_triplet",
    "read_bgen",
    "read_vcf",
    "output_cs",
    "output_alphas",
    "output_weights",
    "output_her",
    "output_corr",
    "output_cv",
    "output_numpy",
]


class CVData(NamedTuple):
    """Define the raw data object for the future inference.

    Attributes:
        train_geno: genotype data for training SuShiE weights.
        train_pheno: phenotype data for training SuShiE weights.
        valid_geno: genotype data for validating SuShiE weights.
        valid_pheno: phenotype data for validating SuShiE weights.

    """

    train_geno: List[Array]
    train_pheno: List[Array]
    valid_geno: List[Array]
    valid_pheno: List[Array]


class CleanData(NamedTuple):
    """Define the raw data object for the future inference.

    Attributes:
        geno: actual genotype data.
        pheno: phenotype data.
        covar: covariate needed to be adjusted in the inference.

    """

    geno: List[Array]
    pheno: List[Array]
    covar: utils.ListArrayOrNone


class RawData(NamedTuple):
    """Define the raw data object for the future inference.

    Attributes:
        bim: SNP information data.
        fam: individual information data.
        bed: actual genotype data.
        pheno: phenotype data.
        covar: covariate needed to be adjusted in the inference.

    """

    bim: pd.DataFrame
    fam: pd.DataFrame
    bed: Array
    pheno: pd.DataFrame
    covar: utils.PDOrNone


def read_data(
    n_pop: int,
    ancestry_index: pd.DataFrame,
    pheno_paths: List[str],
    covar_paths: utils.ListStrOrNone,
    geno_paths: List[str],
    geno_func: Callable,
) -> List[RawData]:
    """Read in pheno, covar, and genotype data and convert it to raw data object.

    Args:
        n_pop: The int to indicate the number of ancestries.
        ancestry_index: The DataFrame that contains ancestry index.
        pheno_paths: The path for phenotype data across ancestries.
        covar_paths: The path for covariates data across ancestries.
        geno_paths: The path for genotype data across ancestries.
        geno_func: The function to read in genotypes depending on the format.

    Returns:
        :py:obj:`List[RawData]`: A list of Raw data object (:py:obj:`RawData`).

    """

    index_file = True if ancestry_index.shape[0] != 0 else False
    rawData = []
    for idx in range(n_pop):
        # if there is no index file, we read in the data ancestry by ancestry
        # if there is index file, we just need to read in the data once at first
        if (not index_file) or (index_file and idx == 0):
            bim, fam, bed = geno_func(geno_paths[idx])

            pheno = (
                pd.read_csv(pheno_paths[idx], sep="\t", header=None, dtype={0: object})
                .rename(columns={0: "iid", 1: "pheno"})
                .reset_index(drop=True)
            )

            if covar_paths is not None:
                covar = (
                    pd.read_csv(
                        covar_paths[idx], sep="\t", header=None, dtype={0: object}
                    )
                    .rename(columns={0: "iid"})
                    .reset_index(drop=True)
                )
            else:
                covar = None

        # it has some pycharm warnings. It's okay to ingore them.
        # I couldn't think of a way to remove these warnings other than pre-specify them before for loops
        # but the codes will look silly
        tmp_bim = bim
        tmp_bed = bed
        tmp_fam = fam
        tmp_pheno = pheno
        tmp_covar = covar
        if index_file:
            tmp_pt = ancestry_index.loc[ancestry_index[1] == (idx + 1)][0]
            tmp_fam = fam.loc[fam.iid.isin(tmp_pt)]
            tmp_bed = bed[fam.iid.isin(tmp_pt).values, :]
            tmp_pheno = pheno.loc[pheno.iid.isin(tmp_pt)]

            if covar_paths is not None:
                tmp_covar = covar.loc[covar.iid.isin(tmp_pt)]
            else:
                tmp_covar = None

        if len(tmp_bim) == 0:
            raise ValueError(f"Ancestry {idx + 1}: No genotype data found.")

        if len(tmp_fam) == 0:
            raise ValueError(f"Ancestry {idx + 1}: No fam data found.")

        if len(tmp_pheno) == 0:
            raise ValueError(f"Ancestry {idx + 1}: No pheno data found.")

        if covar_paths is not None and len(tmp_covar) == 0:
            raise ValueError(f"Ancestry {idx + 1}: No covar data found.")

        rawData.append(
            RawData(
                bim=tmp_bim, fam=tmp_fam, bed=tmp_bed, pheno=tmp_pheno, covar=tmp_covar
            )
        )

    return rawData


def read_triplet(path: str) -> Tuple[pd.DataFrame, pd.DataFrame, Array]:
    """Read in genotype data in `plink 1 <https://www.cog-genomics.org/plink/1.9/input#bed>`_ format.
        `pandas_plink <https://pandas-plink.readthedocs.io/>`_ package is used to read in the plink file.

    Args:
        path: The path for plink genotype data (suffix only).

    Returns:
        :py:obj:`Tuple[pd.DataFrame, pd.DataFrame, Array]`: A tuple of
            #. SNP information (bim; :py:obj:`pd.DataFrame`),
            #. individuals information (fam; :py:obj:`pd.DataFrame`),
            #. genotype matrix (bed; :py:obj:`Array`).

    """

    bim, fam, bed = read_plink(path, verbose=False)
    bim = bim[["chrom", "snp", "pos", "a0", "a1"]]
    fam = fam[["iid"]]
    # we want bed file to be nxp
    bed = jnp.array(bed.compute().T, dtype="float64")
    return bim, fam, bed


def read_vcf(path: str) -> Tuple[pd.DataFrame, pd.DataFrame, Array]:
    """Read in genotype data in `vcf <https://en.wikipedia.org/wiki/Variant_Call_Format>`_ format.
        `cyvcf2 <https://brentp.github.io/cyvcf2/>`_ package is used to read in the vcf file.
        gt_types are used to determine the genotype matrix. It it is UNKNOWN, it will be coded as NA.

    Args:
        path: The path for vcf genotype data (full file name). It will count REF allele.

    Returns:
        :py:obj:`Tuple[pd.DataFrame, pd.DataFrame, Array]`: A tuple of
            #. SNP information (bim; :py:obj:`pd.DataFrame`),
            #. participants information (fam; :py:obj:`pd.DataFrame`),
            #. genotype matrix (bed; :py:obj:`Array`).

    """

    vcf = VCF(path, gts012=True)
    fam = pd.DataFrame(vcf.samples).rename(columns={0: "iid"})
    bim_list = []
    bed_list = []
    for var in vcf:
        # var.ALT is a list of alternative allele
        bim_list.append([var.CHROM, var.ID, var.POS, var.ALT[0], var.REF])
        var.gt_types = jnp.where(var.gt_types == 3, jnp.nan, var.gt_types)
        tmp_bed = 2 - var.gt_types
        bed_list.append(tmp_bed)

    bim = pd.DataFrame(bim_list, columns=["chrom", "snp", "pos", "a0", "a1"])
    bed = jnp.array(bed_list, dtype="float64").T

    return bim, fam, bed


def read_bgen(path: str) -> Tuple[pd.DataFrame, pd.DataFrame, Array]:
    """Read in genotype data in `bgen <https://www.well.ox.ac.uk/~gav/bgen_format/>`_ 1.3 format.
     `bgen-reader <https://pypi.org/project/bgen-reader/>`_ package is used to read in the bgen file.

    Args:
        path: The path for bgen genotype data (full file name).

    Returns:
        :py:obj:`Tuple[pd.DataFrame, pd.DataFrame, Array]`: A tuple of
            #. SNP information (bim; :py:obj:`pd.DataFrame`),
            #. individuals information (fam; :py:obj:`pd.DataFrame`),
            #. genotype matrix (bed; :py:obj:`Array`).

    """

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
    bed = jnp.array(
        jnp.einsum("ijk,k->ij", bgen.read(), jnp.array([0, 1, 2])), dtype="float64"
    )

    return bim, fam, bed


# output functions
def output_cs(
    result: List[infer.SushieResult],
    meta_pip: Optional[List[Array]],
    snps: pd.DataFrame,
    output: str,
    trait: str,
    compress: bool,
    method_type: str,
) -> pd.DataFrame:
    """Output credible set (after pruning for purity) file ``*cs.tsv`` (see :ref:`csfile`).

    Args:
        result: The sushie inference result.
        meta_pip: The meta-analyzed PIPs from Meta SuShiE.
        snps: The SNP information table.
        output: The output file prefix.
        trait: The trait name better for post-hoc analysis index.
        compress: The indicator whether to compress the output files.
        method_type: Which method the result belongs to: sushie, mega, or meta.

    Returns:
        :py:obj:`pd.DataFrame`: A data frame that outputs to the ``*cs.tsv`` file (:py:obj:`pd.DataFrame`).

    """
    cs = []

    for idx in range(len(result)):
        tmp_cs = (
            snps.merge(result[idx].cs, how="inner", on=["SNPIndex"])
            .assign(trait=trait, n_snps=snps.shape[0])
            .sort_values(
                by=["CSIndex", "alpha", "c_alpha"], ascending=[True, False, True]
            )
        )

        if meta_pip is not None:
            tmp_cs["meta_pip_all"] = meta_pip[0][tmp_cs.SNPIndex.values.astype(int)]
            tmp_cs["meta_pip_cs"] = meta_pip[1][tmp_cs.SNPIndex.values.astype(int)]

        if method_type == "meta":
            ancestry_idx = f"ancestry_{idx + 1}"
        elif method_type == "mega":
            ancestry_idx = "mega"
        else:
            ancestry_idx = "sushie"

        tmp_cs["ancestry"] = ancestry_idx
        cs.append(tmp_cs)
    cs = pd.concat(cs)

    # add a placeholder better for post-hoc analysis
    if cs.shape[0] == 0:
        cs = cs.append({"trait": trait}, ignore_index=True)

    file_name = f"{output}.cs.tsv.gz" if compress else f"{output}.cs.tsv"

    cs.to_csv(file_name, sep="\t", index=False)

    return cs


def output_weights(
    result: List[infer.SushieResult],
    meta_pip: Optional[List[Array]],
    snps: pd.DataFrame,
    output: str,
    trait: str,
    compress: bool,
    method_type: str,
) -> pd.DataFrame:
    """Output prediction weights file ``*weights.tsv`` (see :ref:`weightsfile`).

    Args:
        result: The sushie inference result.
        meta_pip: The meta-analyzed PIPs from Meta SuShiE.
        snps: The SNP information table.
        output: The output file prefix.
        trait: The trait name better for post-hoc analysis index.
        compress: The indicator whether to compress the output files.
        method_type: Which method the result belongs to: sushie, mega, or meta.

    Returns:
        :py:obj:`pd.DataFrame`: A data frame that outputs to the ``*weights.tsv`` file (:py:obj:`pd.DataFrame`).

    """

    n_pop = len(result[0].priors.resid_var)
    weights = copy.deepcopy(snps).assign(trait=trait, n_snps=snps.shape[0])

    for idx in range(len(result)):
        if method_type == "meta":
            cname_idx = [f"ancestry{idx + 1}_single_weight"]
            cname_pip_all = f"ancestry{idx + 1}_single_pip_all"
            cname_pip_cs = f"ancestry{idx + 1}_single_pip_cs"
            cname_cs = f"ancestry{idx + 1}_cs_index"
        elif method_type == "mega":
            cname_idx = ["mega_weight"]
            cname_pip_all = "mega_pip_all"
            cname_pip_cs = "mega_pip_cs"
            cname_cs = "mega_cs_index"
        else:
            cname_idx = [f"ancestry{jdx + 1}_sushie_weight" for jdx in range(n_pop)]
            cname_pip_all = "sushie_pip_all"
            cname_pip_cs = "sushie_pip_cs"
            cname_cs = "sushie_cs_index"

        tmp_weights = pd.DataFrame(
            data=jnp.sum(result[idx].posteriors.post_mean, axis=0),
            columns=cname_idx,
        )

        tmp_weights[cname_pip_all] = result[idx].pip_all
        tmp_weights[cname_pip_cs] = result[idx].pip_cs
        weights = pd.concat([weights, tmp_weights], axis=1)

        df_cs = (
            result[idx]
            .cs[["SNPIndex", "CSIndex"]]
            .groupby("SNPIndex")["CSIndex"]
            .agg(lambda x: ",".join(x.astype(str)))
            .reset_index()
        )

        # although for super rare cases, we have the same snp in more credible sets
        # to record this situation in the weights file (we introduce WARNING in the inference function),
        # we just concatenate the CS index with comma by creating this tmp_cs pandas data frame
        tmp_cs = (
            weights[["SNPIndex"]]
            .merge(df_cs, on="SNPIndex", how="left")
            .fillna("No CS")
        )

        weights = weights.merge(
            tmp_cs.rename(columns={"CSIndex": cname_cs}), on="SNPIndex"
        )

    if meta_pip is not None:
        weights["meta_pip_all"] = meta_pip[0]
        weights["meta_pip_cs"] = meta_pip[1]

    file_name = f"{output}.weights.tsv.gz" if compress else f"{output}.weights.tsv"

    weights.to_csv(file_name, sep="\t", index=False)

    return weights


def output_alphas(
    result: List[infer.SushieResult],
    snps: pd.DataFrame,
    output: str,
    trait: str,
    compress: bool,
    method_type: str,
    purity: float,
) -> pd.DataFrame:
    """Output full credible set (before pruning for purity) file ``*alphas.tsv`` (see :ref:`alphasfile`).

    Args:
        result: The sushie inference result.
        snps: The SNP information table.
        output: The output file prefix.
        trait: The trait name better for post-hoc analysis index.
        compress: The indicator whether to compress the output files.
        method_type: Which method the result belongs to: sushie, mega, or meta.
        purity: The purity threshold.

    Returns:
        :py:obj:`pd.DataFrame`: A data frame that outputs to the ``*alphas.tsv`` file (:py:obj:`pd.DataFrame`).

    """
    alphas = []
    for idx in range(len(result)):
        tmp_alphas = snps.merge(
            result[idx].alphas, how="inner", on=["SNPIndex"]
        ).assign(
            trait=trait,
            n_snps=snps.shape[0],
            purity_threshold=purity,
        )

        if method_type == "meta":
            ancestry_idx = f"ancestry_{idx + 1}"
        elif method_type == "mega":
            ancestry_idx = "mega"
        else:
            ancestry_idx = "sushie"

        tmp_alphas["ancestry"] = ancestry_idx
        alphas.append(tmp_alphas)

    alphas = pd.concat(alphas, axis=0)

    file_name = f"{output}.alphas.tsv.gz" if compress else f"{output}.alphas.tsv"

    alphas.to_csv(file_name, sep="\t", index=False)

    return alphas


def output_her(
    data: CleanData,
    output: str,
    trait: str,
    compress: bool,
) -> pd.DataFrame:
    """Output heritability estimation file ``*her.tsv`` (see :ref:`herfile`).

    Args:
        data: The clean data that are used to estimate traits' heritability.
        output: The output file prefix.
        trait: The trait name better for post-hoc analysis index.
        compress: The indicator whether to compress the output files.

    Returns:
        :py:obj:`pd.DataFrame`: A data frame that outputs to the ``*her.tsv`` file (:py:obj:`pd.DataFrame`).

    """

    n_pop = len(data.geno)

    her_result = []
    for idx in range(n_pop):
        if data.covar is None:
            tmp_covar = None
        else:
            tmp_covar = data.covar[idx]
        tmp_her_result = utils.estimate_her(data.geno[idx], data.pheno[idx], tmp_covar)
        her_result.append(tmp_her_result)

    est_her = (
        pd.DataFrame(
            data=her_result,
            columns=["genetic_var", "h2g", "lrt_stats", "p_value"],
            index=[idx + 1 for idx in range(n_pop)],
        )
        .reset_index(names="ancestry")
        .assign(trait=trait)
    )

    if est_her.shape[0] == 0:
        est_her = est_her.append({"trait": trait}, ignore_index=True)

    file_name = f"{output}.her.tsv.gz" if compress else f"{output}.her.tsv"

    est_her.to_csv(file_name, sep="\t", index=False)

    return est_her


def output_corr(
    result: List[infer.SushieResult],
    output: str,
    trait: str,
    compress: bool,
) -> pd.DataFrame:
    """Output effect size correlation file ``*corr.tsv`` (see :ref:`corrfile`).

    Args:
        result: The sushie inference result.
        output: The output file prefix.
        trait: The trait name better for post-hoc analysis index.
        compress: The indicator whether to compress the output files.

    Returns:
        :py:obj:`pd.DataFrame`: A data frame that outputs to the ``*corr.tsv`` file (:py:obj:`pd.DataFrame`).

    """

    n_pop = len(result[0].priors.resid_var)
    raw_corr = result[0].posteriors.weighted_sum_covar
    n_l = len(raw_corr)
    tmp_corr = jnp.transpose(raw_corr)
    corr = pd.DataFrame(data={"trait": trait, "CSIndex": (jnp.arange(n_l) + 1)})

    for idx in range(n_pop):
        _var = tmp_corr[idx, idx]
        tmp_pd = pd.DataFrame(data={f"ancestry{idx + 1}_est_var": _var})
        corr = pd.concat([corr, tmp_pd], axis=1)
        for jdx in range(idx + 1, n_pop):
            _covar = tmp_corr[idx, jdx]
            _var1 = tmp_corr[idx, idx]
            _var2 = tmp_corr[jdx, jdx]
            _corr = _covar / jnp.sqrt(_var1 * _var2)
            tmp_pd_covar = pd.DataFrame(
                data={f"ancestry{idx + 1}_ancestry{jdx + 1}_est_covar": _covar}
            )
            tmp_pd_corr = pd.DataFrame(
                data={f"ancestry{idx + 1}_ancestry{jdx + 1}_est_corr": _corr}
            )
            corr = pd.concat([corr, tmp_pd_covar, tmp_pd_corr], axis=1)

    file_name = f"{output}.corr.tsv.gz" if compress else f"{output}.corr.tsv"

    corr.to_csv(file_name, sep="\t", index=False)

    return corr


def output_cv(
    cv_res: List,
    sample_size: List[int],
    output: str,
    trait: str,
    compress: bool,
) -> pd.DataFrame:
    """Output cross validation file ``*cv.tsv`` for
        future `FUSION <http://gusevlab.org/projects/fusion/>`_ pipline (see :ref:`cvfile`).

    Args:
        cv_res: The cross-validation result (adjusted :math:`r^2` and corresponding :math:`p` values).
        sample_size: The sample size for the SuShiE inference.
        output: The output file prefix.
        trait: The trait name better for post-hoc analysis index.
        compress: The indicator whether to compress the output files.

    Returns:
        :py:obj:`pd.DataFrame`: A data frame that outputs to the ``*cv.tsv`` file (:py:obj:`pd.DataFrame`).

    """

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

    file_name = f"{output}.cv.tsv.gz" if compress else f"{output}.cv.tsv"

    cv_r2.to_csv(file_name, sep="\t", index=False)

    return cv_r2


def output_numpy(
    result: List[infer.SushieResult], snps: pd.DataFrame, output: str
) -> None:
    """Output all results in ``*.npy`` file (no compress option) (see :ref:`npyfile`).

    Args:
        result: The sushie inference result.
        snps: The SNP information
        output: The output file prefix.

    Returns:
        :py:obj:`None`: This function returns nothing (:py:obj:`None`:).

    """
    jnp.save(f"{output}.all.results.npy", [snps, result])

    return None
