import logging
import typing

import jax.numpy as jnp
import limix.her as her
import pandas as pd
from pandas_plink import read_plink
from scipy import stats

from . import core

# from .log import LOG
LOG = "sushie"


def _get_pip(alpha: jnp.ndarray) -> jnp.ndarray:
    pip = 1 - jnp.prod((1 - alpha), axis=0)

    return pip


def _get_cs(
    alpha: jnp.ndarray,
    Xs: typing.List[jnp.ndarray],
    threshold: float = 0.9,
    purity: float = 0.5,
) -> pd.DataFrame:
    n_l, _ = alpha.shape
    t_alpha = pd.DataFrame(alpha.T).reset_index()
    cs = pd.DataFrame(columns=["CSIndex", "SNPIndex", "pip", "cpip"])
    # ld is always pxp, so it can be converted to jnp.array
    ld = jnp.array([x.T @ x / x.shape[0] for x in Xs])

    for idx in range(n_l):
        # select original index and alpha
        tmp_pd = (
            t_alpha[["index", idx]]
            .sort_values(idx, ascending=False)
            .reset_index(drop=True)
        )
        tmp_pd["csum"] = tmp_pd[[idx]].cumsum()
        n_row = tmp_pd[tmp_pd.csum < threshold].shape[0]

        # if all rows less than threshold + 1 is what we want to select
        if n_row == tmp_pd.shape[0]:
            select_idx = jnp.arange(n_row)
        else:
            select_idx = jnp.arange(n_row + 1)
        tmp_pd = (
            tmp_pd.iloc[select_idx, :]
            .assign(CSIndex=idx + 1)
            .rename(columns={"csum": "cpip", "index": "SNPIndex", idx: "pip"})
        )

        # check the impurity
        snp_idx = tmp_pd.SNPIndex.values.astype("int64")

        min_corr = jnp.min(ld[:, snp_idx][:, :, snp_idx])
        if min_corr > purity:
            cs = pd.concat([cs, tmp_pd], ignore_index=True)

    return cs


def _kl_categorical(
    alpha: jnp.ndarray,
    pi: jnp.ndarray,
) -> float:
    """
    KL divergence between two categorical distributions
    KL(alpha || pi)
    """
    return jnp.nansum(alpha * (jnp.log(alpha) - jnp.log(pi)))


def _kl_mvn(
    m1: jnp.ndarray,
    s12: jnp.ndarray,
    m0: float,
    s02: jnp.ndarray,
) -> float:
    """
    KL divergence between multiN(m1, s12) and multiN(m0, s02)
    KL(multiN(m1, s12) || multiN(m0, s02))
    """

    # https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
    k, k = s02.shape

    p1 = (
        jnp.trace(jnp.einsum("ij,kjm->kim", jnp.linalg.inv(s02), s12), axis1=1, axis2=2)
        - k
    )
    p2 = jnp.einsum("ij,jm,im->i", (m0 - m1), jnp.linalg.inv(s02), (m0 - m1))
    # a more stable way
    # p3 = jnp.log(jnp.linalg.det(s02) / jnp.linalg.det(s12))
    s0, sld0 = jnp.linalg.slogdet(s02)
    s1, sld1 = jnp.linalg.slogdet(s12)

    p3 = sld0 - sld1

    return 0.5 * (p1 + p2 + p3)


# SuSiE Supplementary material (B.9)
def _erss(
    X: jnp.ndarray, y: jnp.ndarray, beta: jnp.ndarray, beta_sq: jnp.ndarray
) -> core.ArrayOrFloat:
    mu_li = X @ beta
    mu2_li = (X ** 2) @ beta_sq

    term_1 = jnp.sum((y - jnp.sum(mu_li, axis=1)) ** 2)
    term_2 = jnp.sum(mu2_li - (mu_li ** 2))

    return term_1 + term_2


# SuSiE Supplementary material (B.5)


def _eloglike(
    X: jnp.ndarray,
    y: jnp.ndarray,
    beta: jnp.ndarray,
    beta_sq: jnp.ndarray,
    sigma_sq: core.ArrayOrFloat,
) -> core.ArrayOrFloat:
    n, p = X.shape
    norm_term = -(0.5 * n) * jnp.log(2 * jnp.pi * sigma_sq)
    quad_term = -(0.5 / sigma_sq) * _erss(X, y, beta, beta_sq)

    return norm_term + quad_term


def _drop_na_inf(df: pd.DataFrame, nam: str, idx: int) -> pd.DataFrame:
    log = logging.getLogger(LOG)
    old_row = df.shape[0]
    df.replace([jnp.inf, -jnp.inf], jnp.nan)
    df.dropna(inplace=True)
    diff = old_row - df.shape[0]
    if diff != 0:
        log.info(
            f"Ancestry {idx + 1}: Drop {diff} rows from {nam} table due to INF value or NAN value."
        )
    return df


def _ols(X, y):
    """
    Perform a marginal linear regression for each snp on the phenotype.

    :param X: numpy.ndarray n x p genotype matrix to regress over
    :param y: numpy.ndarray phenotype vector

    :return: pandas.DataFrame containing estimated beta and standard error
    """
    n_samples, n_features = X.shape
    X_inter = jnp.append(jnp.ones((n_samples, 1)), X, axis=1)
    n_features += 1
    XtX_inv = jnp.linalg.inv(X_inter.T @ X_inter)
    betas = XtX_inv @ X_inter.T @ y
    residual = y - X_inter @ betas
    rss = jnp.sum(residual ** 2, axis=0)
    sigma_sq = rss / (n_samples - n_features)
    t_scores = betas / jnp.sqrt(jnp.diagonal(XtX_inv)[:, jnp.newaxis] * sigma_sq)
    p_val = 2 * stats.t.sf(abs(t_scores), df=(n_samples - n_features))
    r_sq = 1 - rss / jnp.sum((y - jnp.mean(y)) ** 2)
    adj_r = 1 - (1 - r_sq) * (n_samples - 1) / (n_samples - n_features)

    return residual, adj_r, p_val


def _allele_check(
    base0: pd.Series,
    base1: pd.Series,
    compare0: pd.Series,
    compare1: pd.Series,
    idx: int,
) -> typing.Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Check whether SNPs alleles match across ancestries.

    :param base0: A0 for the first ancestry (baseline)
    :param base1: A1 for the first ancestry (baseline)
    :param compare0: A0 for the compared ancestry
    :param compare1: A1 for the compared ancestry
    :param idx: Ancestry Index

    :return: pandas.DataFrame containing estimated beta and standard error
    """

    log = logging.getLogger(LOG)
    correct = jnp.logical_and(base0 == compare0, base1 == compare1)
    flipped = jnp.logical_and(base0 == compare1, base1 == compare0)
    correct_idx = jnp.where(correct)[0]
    flipped_idx = jnp.where(flipped)[0]
    wrong_idx = jnp.where(jnp.logical_not(jnp.logical_or(correct, flipped)))[0]

    if len(flipped_idx) != 0:
        log.warning(
            f"Ancestry{idx + 1} has {len(flipped_idx)} flipped alleles from ancestry 1. Will flip these SNPs."
        )

    if len(wrong_idx):
        log.warning(
            f"Ancestry{idx + 1} has {len(wrong_idx)} alleles that couldn't be flipped. Will remove these SNPs."
        )

    return correct_idx, flipped_idx, wrong_idx


def _estimate_her(X: jnp.ndarray, y: jnp.ndarray, C: jnp.ndarray = None) -> float:
    n, p = X.shape
    A = jnp.dot(X, X.T) / p
    h2g = her.estimate(y, "normal", A, C, verbose=False)

    return h2g


def _process_raw(
    geno_paths: typing.List[str],
    pheno_paths: typing.List[str],
    covar_paths: typing.List[str] = None,
    norm_X: bool = True,
    norm_y: bool = False,
    h2g: bool = False,
    regress: bool = True,
) -> core.CleanData:
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
        tmp_bim = _drop_na_inf(tmp_bim, "bim", idx)
        tmp_fam = _drop_na_inf(tmp_fam, "fam", idx)
        tmp_pheno = _drop_na_inf(tmp_pheno, "pheno", idx)

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
        tmp_pheno = tmp_pheno.reset_index()
        # keep track of pheno index for future matching the bed file if bed
        # files are shuffled due to merging
        tmp_pheno = tmp_pheno.rename(
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
            tmp_covar = _drop_na_inf(tmp_covar, "covar", idx)
            tmp_covar = tmp_covar.reset_index()
            # keep track of covar index for future matching the bed file if
            # bed files are shuffled due to merging
            tmp_covar = tmp_covar.rename(
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
                f"Ancestry{idx + 1} has {snps_num_diff} independent SNPs and "
                f"{snps.shape[0]} common SNPs. Will remove these independent SNPs.",
            )
    else:
        snps = bim[0]

    # find flipped reference alleles across ancestries
    flip_idx = []
    if n_pop > 1:
        for idx in range(1, n_pop):
            correct_idx, tmp_flip_idx, wrong_idx = _allele_check(
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
            tmp_h2g = _estimate_her(bed[idx], pheno[idx], covar[idx])
            est_h2g = est_h2g.at[idx].set(tmp_h2g)
    else:
        est_h2g = None

    # regress covar on y
    for idx in range(n_pop):
        if covar[idx] is not None:
            pheno_resid, _, _ = _ols(covar[idx], pheno[idx])
            pheno[idx] = pheno_resid
            # regress covar on each SNP, it might be slow, the default is True
            if regress:
                (
                    geno_resid,
                    _,
                    _,
                ) = _ols(covar[idx], bed[idx])
                bed[idx] = geno_resid
    log.info(
        f"Successfully prepare genotype ({bed[0].shape[1]} SNPs) and phenotype data for {n_pop}"
        + " ancestries, and start fine-mapping using SuShiE.",
    )
    result = core.CleanData(
        geno=bed,
        pheno=pheno,
        covar=covar,
        snp=snps,
        h2g=est_h2g,
    )
    return result
