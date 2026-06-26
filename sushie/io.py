# pattern: Mixed (needs refactoring)

from collections.abc import Callable
from typing import Any, NamedTuple

import genoio as _genoio
import numpy as np
import polars as pl

import jax.numpy as jnp

from jax import Array

from . import infer, log, utils


genoio: Any = _genoio

__all__ = [
    "CVData",
    "CleanData",
    "RawData",
    "read_data",
    "read_triplet",
    "read_bgen",
    "read_vcf",
    "read_gwas",
    "read_ld",
    "output_cs",
    "output_alphas",
    "output_weights",
    "output_her",
    "output_corr",
    "output_cv",
    "output_numpy",
]


class CVData(NamedTuple):
    """Define the cross validation data object.

    Attributes:
        train_geno: genotype data for training SuShiE weights.
        train_pheno: phenotype data for training SuShiE weights.
        valid_geno: genotype data for validating SuShiE weights.
        valid_pheno: phenotype data for validating SuShiE weights.

    """

    train_geno: list[Array]
    train_pheno: list[Array]
    valid_geno: list[Array]
    valid_pheno: list[Array]


class CleanData(NamedTuple):
    """Define the clean data object ready for the future inference.

    Attributes:
        geno: actual genotype data.
        pheno: phenotype data.
        covar: covariate needed to be adjusted in the inference.
        pi: prior weights for each SNP to be causal.

    """

    geno: list[Array]
    pheno: list[Array]
    covar: utils.ListArrayOrNone
    pi: Array | None


class ssData(NamedTuple):
    """Define the summary data object ready for the future inference.

    Attributes:
        gwas: GWAS data.
        lds: LD data.
        ssize: GWAS sample size
        pi: prior weights for each SNP to be causal.

    """

    zs: list[Array]
    lds: list[Array]
    ns: Array
    pi: Array | None


class RawData(NamedTuple):
    """Define the raw data object for the future data cleaning.

    Attributes:
        bim: SNP information data.
        fam: individual information data.
        bed: actual genotype data.
        pheno: phenotype data.
        covar: covariate needed to be adjusted in the inference.

    """

    bim: pl.DataFrame
    fam: pl.DataFrame
    bed: Array
    pheno: pl.DataFrame
    covar: utils.PDOrNone


def read_data(
    n_pop: int,
    ancestry_index: pl.DataFrame,
    pheno_paths: list[str],
    covar_paths: utils.ListStrOrNone,
    geno_paths: list[str],
    geno_func: Callable,
) -> list[RawData]:
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
            log.logger.debug(f"Read in genotype data for ancestry {idx + 1}.")

            bim, fam, bed = geno_func(geno_paths[idx])

            log.logger.debug(f"Read in phenotype data for ancestry {idx + 1}.")

            pheno = pl.read_csv(
                pheno_paths[idx],
                separator="\t",
                has_header=False,
                schema_overrides={"column_1": pl.String},
            ).rename({"column_1": "iid", "column_2": "pheno"})

            log.logger.debug(f"Read in covariate data for ancestry {idx + 1}.")

            if covar_paths is not None:
                covar = pl.read_csv(
                    covar_paths[idx],
                    separator="\t",
                    has_header=False,
                    schema_overrides={"column_1": pl.String},
                ).rename({"column_1": "iid"})
            else:
                covar = None

        tmp_bim = bim
        tmp_bed = bed
        tmp_fam = fam
        tmp_pheno = pheno
        tmp_covar = covar
        if index_file:
            tmp_pt = ancestry_index.filter(pl.col("column_2") == (idx + 1)).get_column("column_1").to_list()
            fam_mask = fam["iid"].is_in(tmp_pt).to_numpy()
            tmp_fam = fam.filter(pl.col("iid").is_in(tmp_pt))
            tmp_bed = bed[fam_mask, :]
            tmp_pheno = pheno.filter(pl.col("iid").is_in(tmp_pt))

            if covar is not None:
                tmp_covar = covar.filter(pl.col("iid").is_in(tmp_pt))
            else:
                tmp_covar = None

        if len(tmp_bim) == 0:
            raise ValueError(f"Ancestry {idx + 1}: No genotype data found.")

        if len(tmp_fam) == 0:
            raise ValueError(f"Ancestry {idx + 1}: No fam data found.")

        if len(tmp_pheno) == 0:
            raise ValueError(f"Ancestry {idx + 1}: No pheno data found.")

        if tmp_covar is not None and len(tmp_covar) == 0:
            raise ValueError(f"Ancestry {idx + 1}: No covar data found.")

        rawData.append(RawData(bim=tmp_bim, fam=tmp_fam, bed=tmp_bed, pheno=tmp_pheno, covar=tmp_covar))

    log.logger.debug("Finish read in data for all ancestries.")

    return rawData


def _read_genoio_dataset(dataset, **read_options) -> tuple[pl.DataFrame, pl.DataFrame, Array]:
    """Read a genoio dataset into SuShiE's canonical metadata and array types."""
    bed, fam, bim = dataset.read(
        missing="nan",
        dtype="float64",
        return_samples=True,
        return_variants=True,
        **read_options,
    )

    bim = bim.rename({"id": "snp"}).select(["chrom", "snp", "pos", "a0", "a1"])
    fam = fam.select(["iid"])
    bed = jnp.asarray(bed)

    return bim, fam, bed


def read_triplet(path: str) -> tuple[pl.DataFrame, pl.DataFrame, Array]:
    """Read in genotype data in `plink 1 <https://www.cog-genomics.org/plink/1.9/input#bed>`_ format.
        `genoio <https://github.com/mancusolab/genoio>`_ package is used to read in the plink file.

    Args:
        path: The path for plink genotype data (suffix only).

    Returns:
        :py:obj:`Tuple[pl.DataFrame, pl.DataFrame, Array]`: A tuple of
            #. SNP information (bim; :py:obj:`pl.DataFrame`),
            #. individuals information (fam; :py:obj:`pl.DataFrame`),
            #. genotype matrix (bed; :py:obj:`Array`).

    """

    return _read_genoio_dataset(genoio.bfile(path))


def read_vcf(path: str) -> tuple[pl.DataFrame, pl.DataFrame, Array]:
    """Read in genotype data in `vcf <https://en.wikipedia.org/wiki/Variant_Call_Format>`_ format.
        `genoio <https://github.com/mancusolab/genoio>`_ package is used to read in the vcf file.
        Missing genotypes are coded as NA.

    Args:
        path: The path for vcf genotype data (full file name). It will count REF allele.

    Returns:
        :py:obj:`Tuple[pl.DataFrame, pl.DataFrame, Array]`: A tuple of
            #. SNP information (bim; :py:obj:`pl.DataFrame`),
            #. participants information (fam; :py:obj:`pl.DataFrame`),
            #. genotype matrix (bed; :py:obj:`Array`).

    """

    bim, fam, bed = _read_genoio_dataset(genoio.vcf(path))
    bim = bim.select(
        "chrom",
        "snp",
        "pos",
        pl.col("a1").alias("a0"),
        pl.col("a0").alias("a1"),
    )
    bed = 2 - bed

    return bim, fam, bed


def read_bgen(path: str) -> tuple[pl.DataFrame, pl.DataFrame, Array]:
    """Read in genotype data in `bgen <https://www.well.ox.ac.uk/~gav/bgen_format/>`_ 1.3 format.
     `genoio <https://github.com/mancusolab/genoio>`_ package is used to read in the bgen file.

    Args:
        path: The path for bgen genotype data (full file name).

    Returns:
        :py:obj:`Tuple[pl.DataFrame, pl.DataFrame, Array]`: A tuple of
            #. SNP information (bim; :py:obj:`pl.DataFrame`),
            #. individuals information (fam; :py:obj:`pl.DataFrame`),
            #. genotype matrix (bed; :py:obj:`Array`).

    """

    return _read_genoio_dataset(genoio.bgen(path), dosage="dosage")


def read_gwas(
    path: str,
    header: list[str],
    chrom: utils.IntOrNone,
    start: utils.IntOrNone,
    end: utils.IntOrNone,
) -> pl.DataFrame:
    """Read in GWAS data in tsv file.

    Args:
        path: The path for GWAS data (full file name).
        header: The header for GWAS data.
        chrom: The chromosome number.
        start: The start position.
        end: The end position.

    Returns:
        :py:obj:`pl.DataFrame`

    """

    df_gwas = pl.read_csv(path, separator="\t").drop_nulls()

    if not all(col in df_gwas.columns for col in header):
        raise ValueError("The specified GWAS columns are not in the GWAS data.")

    df_gwas = (
        df_gwas.select(header)
        .rename(
            {
                header[0]: "chrom",
                header[1]: "snp",
                header[2]: "pos",
                header[3]: "a1",
                header[4]: "a0",
                header[5]: "z",
            }
        )
        .with_columns(
            pl.col("chrom").cast(pl.Int64),
            pl.col("pos").cast(pl.Int64),
            pl.when(pl.col("z").is_infinite()).then(None).otherwise(pl.col("z")).alias("z"),
        )
        .drop_nulls()
    )

    log.logger.debug("Filter GWAS data based on Chrom, Start, and End.")
    if chrom is not None:
        old_num = df_gwas.shape[0]
        df_gwas = df_gwas.filter(pl.col("chrom") == chrom)
        del_num = old_num - df_gwas.shape[0]

        if df_gwas.shape[0] == 0:
            raise ValueError(f"No SNPs remain after filtering on chromosome {chrom} for GWAS data from {path}.")

        if del_num != 0:
            log.logger.debug(f"Drop {del_num} SNPs that are not on chromosome {chrom} for GWAS data from {path}.")

        old_num = df_gwas.shape[0]
        df_gwas = df_gwas.filter(pl.col("pos") >= start)
        del_num = old_num - df_gwas.shape[0]

        if df_gwas.shape[0] == 0:
            raise ValueError(
                f"No SNPs are located after position {start} on chromosome {chrom} for GWAS data from {path}."
            )

        if del_num != 0:
            log.logger.debug(
                f"Drop {del_num} SNPs that are located before position {start} on chromosome {chrom}."
                + " for GWAS data from {path}."
            )

        old_num = df_gwas.shape[0]
        df_gwas = df_gwas.filter(pl.col("pos") <= end)
        del_num = old_num - df_gwas.shape[0]

        if df_gwas.shape[0] == 0:
            raise ValueError(
                f"No SNPs are located before position {end} on chromosome {chrom} for GWAS data from {path}."
            )

        if del_num != 0:
            log.logger.debug(
                f"Drop {del_num} SNPs that are located after position {end} on chromosome {chrom}."
                + " for GWAS data from {path}."
            )

    return df_gwas


def read_ld(path: str) -> pl.DataFrame:
    """Read in LD (linkage disequilibrium) matrix from a TSV file.

    The LD matrix should be a symmetric correlation matrix where rows and columns
    represent SNPs. The file should be tab-separated with SNP IDs as column headers.
    Rows with infinite or NaN values are automatically removed.

    Args:
        path: The path to the LD matrix file (tab-separated, .tsv format).

    Returns:
        :py:obj:`pl.DataFrame`: LD correlation matrix with SNP IDs as index and columns.

    Example:
        Read LD matrix for fine-mapping::

            from sushie.io import read_ld

            # Read LD matrix
            ld = read_ld("path/to/ld_matrix.tsv")
            print(ld.shape)  # Should be (n_snps, n_snps)

    Note:
        The LD matrix must be computed using the same reference alleles as the
        GWAS summary statistics for correct fine-mapping results.

    """

    ld = pl.read_csv(path, separator="\t")
    ld = ld.with_columns(pl.when(pl.all().is_infinite()).then(None).otherwise(pl.all()).name.keep())

    keep_rows = ld.select(pl.any_horizontal(pl.all().is_null()).not_().alias("keep")).get_column("keep")
    keep_idx = keep_rows.arg_true().to_list()
    keep_cols = [ld.columns[idx] for idx in keep_idx if idx < ld.width]

    ld = ld.filter(keep_rows).select(keep_cols)

    return ld


def _write_table(frame: pl.DataFrame, file_name: str, compress: bool) -> None:
    frame.write_csv(
        file_name,
        separator="\t",
        compression="gzip" if compress else "uncompressed",
    )


# output functions
def output_cs(
    result: list[infer.SushieResult],
    meta_pip: list[Array] | None,
    snps: pl.DataFrame,
    output: str,
    trait: str,
    compress: bool,
    method_type: str,
) -> pl.DataFrame:
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
        :py:obj:`pl.DataFrame`: A data frame that outputs to the ``*cs.tsv`` file (:py:obj:`pl.DataFrame`).

    """
    cs = []

    for idx in range(len(result)):
        tmp_cs = (
            snps.join(
                result[idx].cs,
                how="inner",
                on=["SNPIndex"],
                maintain_order="left",
            )
            .with_columns(
                pl.lit(trait).alias("trait"),
                pl.lit(snps.shape[0]).alias("n_snps"),
            )
            .sort(["CSIndex", "alpha", "c_alpha"], descending=[False, True, False])
        )

        if meta_pip is not None:
            snp_idx = tmp_cs["SNPIndex"].to_numpy().astype(int)
            tmp_cs = tmp_cs.with_columns(
                pl.Series("meta_pip_all", np.asarray(meta_pip[0][snp_idx])),
                pl.Series("meta_pip_cs", np.asarray(meta_pip[1][snp_idx])),
            )

        if method_type == "meta":
            ancestry_idx = f"ancestry_{idx + 1}"
        elif method_type == "mega":
            ancestry_idx = "mega"
        else:
            ancestry_idx = "sushie"

        tmp_cs = tmp_cs.with_columns(pl.lit(ancestry_idx).alias("ancestry"))
        cs.append(tmp_cs)
    cs = pl.concat(cs) if len(cs) != 0 else pl.DataFrame()

    # add a placeholder better for post-hoc analysis
    if cs.shape[0] == 0:
        cs = pl.DataFrame({"trait": [trait]})

    file_name = f"{output}.cs.tsv.gz" if compress else f"{output}.cs.tsv"

    _write_table(cs, file_name, compress)

    return cs


def output_weights(
    result: list[infer.SushieResult],
    meta_pip: list[Array] | None,
    snps: pl.DataFrame,
    output: str,
    trait: str,
    compress: bool,
    method_type: str,
) -> pl.DataFrame:
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
        :py:obj:`pl.DataFrame`: A data frame that outputs to the ``*weights.tsv`` file (:py:obj:`pl.DataFrame`).

    """

    n_pop = len(result[0].priors.resid_var)
    weights = snps.with_columns(
        pl.lit(trait).alias("trait"),
        pl.lit(snps.shape[0]).alias("n_snps"),
    )

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

        tmp_weights = pl.DataFrame(
            np.asarray(jnp.sum(result[idx].posteriors.post_mean, axis=0)),
            schema=cname_idx,
        )

        tmp_weights = tmp_weights.with_columns(
            pl.Series(cname_pip_all, np.asarray(result[idx].pip_all)),
            pl.Series(cname_pip_cs, np.asarray(result[idx].pip_cs)),
        )
        weights = pl.concat([weights, tmp_weights], how="horizontal")

        df_cs = (
            result[idx]
            .cs.select(["SNPIndex", "CSIndex"])
            .group_by("SNPIndex")
            .agg(pl.col("CSIndex").cast(pl.String).str.join(",").alias("CSIndex"))
        )

        # although for super rare cases, we have the same snp in more credible sets
        # to record this situation in the weights file (we introduce WARNING in the inference function).
        tmp_cs = (
            weights.select(["SNPIndex"])
            .join(df_cs, on="SNPIndex", how="left", maintain_order="left")
            .with_columns(pl.col("CSIndex").fill_null("No CS"))
        )

        weights = weights.join(
            tmp_cs.rename({"CSIndex": cname_cs}),
            on="SNPIndex",
            how="left",
            maintain_order="left",
        )

    if meta_pip is not None:
        weights = weights.with_columns(
            pl.Series("meta_pip_all", np.asarray(meta_pip[0])),
            pl.Series("meta_pip_cs", np.asarray(meta_pip[1])),
        )

    file_name = f"{output}.weights.tsv.gz" if compress else f"{output}.weights.tsv"

    _write_table(weights, file_name, compress)

    return weights


def output_alphas(
    result: list[infer.SushieResult],
    snps: pl.DataFrame,
    output: str,
    trait: str,
    compress: bool,
    method_type: str,
    purity: float,
) -> pl.DataFrame:
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
        :py:obj:`pl.DataFrame`: A data frame that outputs to the ``*alphas.tsv`` file (:py:obj:`pl.DataFrame`).

    """
    alphas = []
    for idx in range(len(result)):
        tmp_alphas = snps.join(
            result[idx].alphas,
            how="inner",
            on=["SNPIndex"],
            maintain_order="left",
        ).with_columns(
            pl.lit(trait).alias("trait"),
            pl.lit(snps.shape[0]).alias("n_snps"),
            pl.lit(purity).alias("purity_threshold"),
        )

        if method_type == "meta":
            ancestry_idx = f"ancestry_{idx + 1}"
        elif method_type == "mega":
            ancestry_idx = "mega"
        else:
            ancestry_idx = "sushie"

        tmp_alphas = tmp_alphas.with_columns(pl.lit(ancestry_idx).alias("ancestry"))
        alphas.append(tmp_alphas)

    alphas = pl.concat(alphas) if len(alphas) != 0 else pl.DataFrame()

    file_name = f"{output}.alphas.tsv.gz" if compress else f"{output}.alphas.tsv"

    _write_table(alphas, file_name, compress)

    return alphas


def output_her(
    data: CleanData,
    output: str,
    trait: str,
    compress: bool,
) -> pl.DataFrame:
    """Output heritability estimation file ``*her.tsv`` (see :ref:`herfile`).

    Args:
        data: The clean data that are used to estimate traits' heritability.
        output: The output file prefix.
        trait: The trait name better for post-hoc analysis index.
        compress: The indicator whether to compress the output files.

    Returns:
        :py:obj:`pl.DataFrame`: A data frame that outputs to the ``*her.tsv`` file (:py:obj:`pl.DataFrame`).

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
        pl.DataFrame(
            her_result,
            schema=["genetic_var", "h2g", "lrt_stats", "p_value"],
            orient="row",
        )
        .with_row_index("ancestry", offset=1)
        .with_columns(pl.lit(trait).alias("trait"))
    )

    if est_her.shape[0] == 0:
        est_her = pl.DataFrame({"trait": [trait]})

    file_name = f"{output}.her.tsv.gz" if compress else f"{output}.her.tsv"

    _write_table(est_her, file_name, compress)

    return est_her


def output_corr(
    result: list[infer.SushieResult],
    output: str,
    trait: str,
    compress: bool,
) -> pl.DataFrame:
    """Output effect size correlation file ``*corr.tsv`` (see :ref:`corrfile`).

    Args:
        result: The sushie inference result.
        output: The output file prefix.
        trait: The trait name better for post-hoc analysis index.
        compress: The indicator whether to compress the output files.

    Returns:
        :py:obj:`pl.DataFrame`: A data frame that outputs to the ``*corr.tsv`` file (:py:obj:`pl.DataFrame`).

    """

    n_pop = len(result[0].priors.resid_var)
    raw_corr = result[0].posteriors.weighted_sum_covar
    n_l = len(raw_corr)
    tmp_corr = jnp.transpose(raw_corr)
    corr = pl.DataFrame({"trait": [trait] * n_l, "CSIndex": np.asarray(jnp.arange(n_l) + 1)})

    for idx in range(n_pop):
        _var = tmp_corr[idx, idx]
        corr = corr.with_columns(pl.Series(f"ancestry{idx + 1}_est_var", np.asarray(_var)))
        for jdx in range(idx + 1, n_pop):
            _covar = tmp_corr[idx, jdx]
            _var1 = tmp_corr[idx, idx]
            _var2 = tmp_corr[jdx, jdx]
            _corr = _covar / jnp.sqrt(_var1 * _var2)
            corr = corr.with_columns(
                pl.Series(
                    f"ancestry{idx + 1}_ancestry{jdx + 1}_est_covar",
                    np.asarray(_covar),
                ),
                pl.Series(
                    f"ancestry{idx + 1}_ancestry{jdx + 1}_est_corr",
                    np.asarray(_corr),
                ),
            )

    file_name = f"{output}.corr.tsv.gz" if compress else f"{output}.corr.tsv"

    _write_table(corr, file_name, compress)

    return corr


def output_cv(
    cv_res: list,
    sample_size: list[int],
    output: str,
    trait: str,
    compress: bool,
) -> pl.DataFrame:
    """Output cross validation file ``*cv.tsv`` for
        future `FUSION <http://gusevlab.org/projects/fusion/>`_ pipeline (see :ref:`cvfile`).

    Args:
        cv_res: The cross-validation result (adjusted :math:`r^2` and corresponding :math:`p` values).
        sample_size: The sample size for the SuShiE inference.
        output: The output file prefix.
        trait: The trait name better for post-hoc analysis index.
        compress: The indicator whether to compress the output files.

    Returns:
        :py:obj:`pl.DataFrame`: A data frame that outputs to the ``*cv.tsv`` file (:py:obj:`pl.DataFrame`).

    """

    cv_r2 = (
        pl.DataFrame(
            cv_res,
            schema=["rsq", "p_value"],
            orient="row",
        )
        .with_row_index("ancestry", offset=1)
        .with_columns(
            pl.Series("N", np.asarray(sample_size)),
            pl.lit(trait).alias("trait"),
        )
    )

    if cv_r2.shape[0] == 0:
        cv_r2 = pl.DataFrame({"trait": [trait]})

    file_name = f"{output}.cv.tsv.gz" if compress else f"{output}.cv.tsv"

    _write_table(cv_r2, file_name, compress)

    return cv_r2


def output_numpy(result: list[infer.SushieResult], snps: pl.DataFrame, output: str) -> None:
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
