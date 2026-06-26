# pattern: Imperative Shell

import genoio
import numpy as np
import polars as pl

from scipy import stats

import jax.numpy as jnp

from jax import config, random


# set key
rng_key = random.PRNGKey(1234)
config.update("jax_enable_x64", True)


def regress(Z, pheno):
    betas = []
    ses = []
    pvals = []
    zs = []
    for snp in Z.T:
        beta, inter, rval, pval, se = stats.linregress(snp, pheno)
        betas.append(beta)
        ses.append(se)
        pvals.append(pval)
        zs.append(beta / se)

    res = pl.DataFrame({"beta": betas, "se": ses, "pval": pvals, "zs": zs})

    return res


def _metadata_to_polars(frame) -> pl.DataFrame:
    if isinstance(frame, pl.DataFrame):
        return frame
    if hasattr(frame, "to_dict"):
        return pl.DataFrame(frame.to_dict(as_series=False))
    if hasattr(frame, "to_arrow"):
        converted = pl.from_arrow(frame.to_arrow())
        return converted.to_frame() if isinstance(converted, pl.Series) else converted
    return pl.DataFrame(frame)


def _compute_ld(G):
    n, p = [float(x) for x in G.shape]
    mafs = jnp.mean(G, axis=0) / 2
    G -= mafs * 2
    G /= jnp.std(G, axis=0)

    # regularize so that LD is PSD
    LD = jnp.dot(G.T, G) / n
    return LD


# flip allele if necessary
def _allele_check(baseA0, baseA1, compareA0, compareA1):
    # no snps that have more than 2 alleles
    # e.g., G and T for EUR and G and A for AFR
    correct = jnp.array(((baseA0 == compareA0) * 1) * ((baseA1 == compareA1) * 1), dtype=int)
    flipped = jnp.array(((baseA0 == compareA1) * 1) * ((baseA1 == compareA0) * 1), dtype=int)
    correct_idx = jnp.where(correct == 1)[0]
    flipped_idx = jnp.where(flipped == 1)[0]

    return correct_idx, flipped_idx


def _column_array(frame: pl.DataFrame, column: str):
    return frame.get_column(column).to_numpy()


def _inner_snp_join(left: pl.DataFrame, right: pl.DataFrame) -> pl.DataFrame:
    return left.join(
        right,
        how="inner",
        on=["chrom", "snp"],
        maintain_order="left",
    )


def _write_no_header(frame: pl.DataFrame, path: str) -> None:
    frame.write_csv(path, include_header=False, separator="\t")


# read plink1.9 triplet
n_pop = 3
pop = ["EUR", "AFR", "EAS"]
bim = []
fam = []
bed = []
# create ancestry index
fam_index = []

for idx in range(n_pop):
    tmp_bed, tmp_fam, tmp_bim = genoio.bfile(f"./plink/{pop[idx]}").read(
        missing="nan",
        dtype="float64",
        return_samples=True,
        return_variants=True,
    )
    tmp_bim = (
        _metadata_to_polars(tmp_bim)
        .rename({"id": "snp"})
        .select(["chrom", "snp", "pos", "a0", "a1"])
        .with_row_index(f"bimIDX_{idx}")
        .rename(
            {
                "pos": f"pos_{idx}",
                "a0": f"a0_{idx}",
                "a1": f"a1_{idx}",
            }
        )
    )
    tmp_fam = _metadata_to_polars(tmp_fam).select(["iid"])
    bim.append(tmp_bim)
    fam.append(tmp_fam)
    bed.append(jnp.asarray(tmp_bed))
    fam_index.append(tmp_fam.with_columns(pl.lit(idx + 1).alias("index")))

snps = _inner_snp_join(bim[0], bim[1])
snps = _inner_snp_join(snps, bim[2])

flip_idx = []
if n_pop > 1:
    for idx in range(1, n_pop):
        # keep track of miss match alleles
        correct_idx, tmp_flip_idx = _allele_check(
            _column_array(snps, "a0_0"),
            _column_array(snps, "a1_0"),
            _column_array(snps, f"a0_{idx}"),
            _column_array(snps, f"a1_{idx}"),
        )
        flip_idx.append(tmp_flip_idx)
        snps = snps.drop([f"a0_{idx}", f"a1_{idx}"])
    snps = snps.rename({"a0_0": "a0", "a1_0": "a1", "pos_0": "pos"})

lds = []
for idx in range(n_pop):
    # subset genotype file to common snps
    bed[idx] = bed[idx][:, _column_array(snps, f"bimIDX_{idx}")]
    # flip the mismatched allele
    if idx > 0:
        flip = flip_idx[idx - 1]
        bed[idx] = bed[idx].at[:, flip].set(2 - bed[idx][:, flip])
    lds.append(_compute_ld(bed[idx]))

for idx in range(n_pop):
    tmp_ld = pl.DataFrame(
        np.asarray(_compute_ld(bed[idx])),
        schema=snps.get_column("snp").to_list(),
    )
    tmp_ld.with_columns(pl.all().round(4)).write_csv(f"{pop[idx]}.ld", separator="\t")

# We assume 2 causal SNPs and h2g = 0.5 for demonstration.
# We also assume qtl effect size correlations are 0.8 for all ancestry pairs.
# The per-snp variance is 0.5 / 2, and the covariance is
# 0.8 * jnp.sqrt((0.5 / 2) ** 2) = 0.2.
b_covar = jnp.array(
    [
        [0.25, 0.2, 0.2],
        [0.2, 0.25, 0.2],
        [0.2, 0.2, 0.25],
    ]
)
L = 2

zero_mean = jnp.zeros((n_pop,))
bvec = random.multivariate_normal(rng_key, zero_mean, b_covar, shape=(L,))

# random select 2 snps as causal
rng_key, gamma_key = random.split(rng_key, 2)
gamma = random.choice(gamma_key, snps.shape[0], shape=(L,), replace=False)

print(snps.gather(np.asarray(gamma)).select(["chrom", "snp", "a0", "a1"]))
#    chrom         snp a0 a1
# 31     1  rs10914958  G  A
# 72     1   rs1886340  G  A

all_pheno = []
all_covar = []
all_gwas = []
for idx in range(n_pop):
    # make some random noises
    rng_key, y_key, sex_key, other_key = random.split(rng_key, 4)
    n_individuals = fam[idx].shape[0]
    tmp_bed = bed[idx][:, gamma]
    tmp_bed = tmp_bed - jnp.mean(tmp_bed, axis=0)
    tmp_bed = tmp_bed / jnp.std(tmp_bed, axis=0)
    tmp_g = tmp_bed @ bvec.T[idx]
    tmp_s2g = jnp.var(tmp_g)
    tmp_s2e = ((1 / 0.5) - 1) * tmp_s2g
    noise = random.normal(y_key, shape=(n_individuals,))
    tmp_y = tmp_g + jnp.sqrt(tmp_s2e) * noise
    tmp_pheno = fam[idx].with_columns(pl.Series("pheno", np.asarray(tmp_y)))
    _write_no_header(tmp_pheno, f"./{pop[idx]}.pheno")
    all_pheno.append(tmp_pheno)

    # make some random covariates
    covar_sex = 1 * random.bernoulli(sex_key, p=0.5, shape=(n_individuals,))
    covar_other = random.normal(other_key, shape=(n_individuals,))
    tmp_covar = fam[idx].with_columns(
        pl.Series("sex", np.asarray(covar_sex)),
        pl.Series("other", np.asarray(covar_other)),
    )
    all_covar.append(tmp_covar)
    _write_no_header(tmp_covar, f"./{pop[idx]}.covar")
    # run gwas
    gwas_bed = bed[idx]
    gwas_bed -= jnp.mean(gwas_bed, axis=0)
    gwas_bed /= jnp.std(gwas_bed, axis=0)
    tmp_y -= jnp.mean(tmp_y)
    tmp_y /= jnp.std(tmp_y)
    df_gwas = regress(gwas_bed, tmp_y)
    all_gwas.append(df_gwas)

_write_no_header(pl.concat(all_pheno), "./all.pheno")
_write_no_header(pl.concat(all_covar), "./all.covar")
all_fam_index = pl.concat(fam_index)
_write_no_header(all_fam_index, "./all.ancestry.index")

for idx in range(n_pop):
    pl.concat(
        [snps.select(["chrom", "snp", "pos", "a0", "a1"]), all_gwas[idx]],
        how="horizontal",
    ).write_csv(f"./{pop[idx]}.gwas", separator="\t")

# create keep.subject file, randomly 1500 from 1609 individuals
rng_key, pt_key = random.split(rng_key, 2)
sel_pt = random.randint(pt_key, shape=(1500,), minval=0, maxval=all_fam_index.shape[0] - 1)
all_fam_index.select(["iid"]).gather(np.asarray(sel_pt)).write_csv(
    "./keep.subject", include_header=False, separator="\t"
)

rng_key, unif_key = random.split(rng_key, 2)
unif = random.uniform(unif_key, shape=(snps.shape[0],), minval=5, maxval=200)
snps = snps.with_columns(pl.Series("unif", np.asarray(unif)))
snps.select(["snp", "unif"]).write_csv("./prior_weights", include_header=False, separator="\t")
