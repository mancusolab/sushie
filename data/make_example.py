import pandas as pd
from pandas_plink import read_plink
from scipy import stats

import jax.numpy as jnp
from jax import random
from jax.config import config

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

    res = pd.DataFrame({"beta": betas, "se": ses, "pval": pvals, "zs": zs})

    return res


def _compute_ld(G):
    G = G.T
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
    correct = jnp.array(
        ((baseA0 == compareA0) * 1) * ((baseA1 == compareA1) * 1), dtype=int
    )
    flipped = jnp.array(
        ((baseA0 == compareA1) * 1) * ((baseA1 == compareA0) * 1), dtype=int
    )
    correct_idx = jnp.where(correct == 1)[0]
    flipped_idx = jnp.where(flipped == 1)[0]

    return correct_idx, flipped_idx


# read plink1.9 triplet
n_pop = 3
pop = ["EUR", "AFR", "EAS"]
bim = []
fam = []
bed = []
# create ancestry index
fam_index = []

for idx in range(n_pop):
    tmp_bim, tmp_fam, tmp_bed = read_plink(f"./plink/{pop[idx]}", verbose=False)
    tmp_bim = tmp_bim[["chrom", "snp", "a0", "a1", "i"]].rename(
        columns={"i": f"bimIDX_{idx}", "a0": f"a0_{idx}", "a1": f"a1_{idx}"}
    )
    bim.append(tmp_bim)
    fam.append(tmp_fam)
    bed.append(tmp_bed)
    new_fam = tmp_fam[["iid"]].copy()
    new_fam["index"] = idx + 1
    fam_index.append(new_fam)

snps = pd.merge(bim[0], bim[1], how="inner", on=["chrom", "snp"])
snps = pd.merge(snps, bim[2], how="inner", on=["chrom", "snp"])

flip_idx = []
if n_pop > 1:
    for idx in range(1, n_pop):
        # keep track of miss match alleles
        correct_idx, tmp_flip_idx = _allele_check(
            snps["a0_0"].values,
            snps["a1_0"].values,
            snps[f"a0_{idx}"].values,
            snps[f"a1_{idx}"].values,
        )
        flip_idx.append(tmp_flip_idx)
        snps = snps.drop(columns=[f"a0_{idx}", f"a1_{idx}"])
    snps = snps.rename(columns={"a0_0": "a0", "a1_0": "a1"})

lds = []
for idx in range(n_pop):
    # subset genotype file to common snps
    bed[idx] = bed[idx][snps[f"bimIDX_{idx}"].values, :].compute()
    # flip the mismatched allele
    if idx > 0:
        bed[idx][flip_idx[idx - 1]] = 2 - bed[idx][flip_idx[idx - 1]]
    lds.append(_compute_ld(bed[idx]))

pd.DataFrame(_compute_ld(bed[0])).to_csv("EUR.ld", sep="\t", index=False)
pd.DataFrame(_compute_ld(bed[1])).to_csv("AFR.ld", sep="\t", index=False)
pd.DataFrame(_compute_ld(bed[2])).to_csv("HIS.ld", sep="\t", index=False)

# we assume there exists 2 causal snps, and the heritability is 0.5 (we make it large as demonstrating purpose)
# we also assume the qtl effect size correlations are 0.8 for all ancestry pairs
# as a result, the per-snp variance is 0.5/2, and the covariance is 0.8*jnp.sqrt((0.5/2)**2) = 0.2
b_covar = jnp.array([[0.25, 0.2, 0.2], [0.2, 0.25, 0.2], [0.2, 0.2, 0.25]])
L = 2

bvec = random.multivariate_normal(rng_key, jnp.zeros((n_pop,)), b_covar, shape=(L,))

# random select 2 snps as causal
rng_key, gamma_key = random.split(rng_key, 2)
gamma = random.choice(gamma_key, snps.shape[0], shape=(L,), replace=False)

print(
    snps.iloc[
        gamma,
    ][["chrom", "snp", "a0", "a1"]]
)
#    chrom         snp a0 a1
# 31     1  rs10914958  G  A
# 72     1   rs1886340  G  A

all_pheno = []
all_covar = []
all_gwas = []
for idx in range(n_pop):
    tmp_pheno = fam[idx][["iid"]].copy()
    # make some random noises
    rng_key, y_key, sex_key, other_key = random.split(rng_key, 4)
    tmp_bed = bed[idx].T[:, gamma]
    tmp_bed = tmp_bed - jnp.mean(tmp_bed, axis=0)
    tmp_bed = tmp_bed / jnp.std(tmp_bed, axis=0)
    tmp_g = tmp_bed @ bvec.T[idx]
    tmp_s2g = jnp.var(tmp_g)
    tmp_s2e = ((1 / 0.5) - 1) * tmp_s2g
    tmp_y = tmp_g + jnp.sqrt(tmp_s2e) * random.normal(
        y_key, shape=(tmp_pheno.shape[0],)
    )
    tmp_pheno["pheno"] = tmp_y
    tmp_pheno.to_csv(f"./{pop[idx]}.pheno", index=False, header=None, sep="\t")
    all_pheno.append(tmp_pheno)

    # make some random covariates
    tmp_covar = fam[idx][["iid"]].copy()
    covar_sex = 1 * (random.bernoulli(sex_key, p=0.5, shape=(tmp_covar.shape[0],)))
    covar_other = random.normal(other_key, shape=(tmp_covar.shape[0],))
    tmp_covar["sex"] = covar_sex
    tmp_covar["other"] = covar_other
    all_covar.append(tmp_covar)
    tmp_covar.to_csv(f"./{pop[idx]}.covar", index=False, header=None, sep="\t")
    # run gwas
    gwas_bed = bed[idx].T
    gwas_bed -= jnp.mean(gwas_bed, axis=0)
    gwas_bed /= jnp.std(gwas_bed, axis=0)
    tmp_y -= jnp.mean(tmp_y)
    tmp_y /= jnp.std(tmp_y)
    df_gwas = regress(gwas_bed, tmp_y)
    all_gwas.append(df_gwas)

pd.concat(all_pheno).to_csv("./all.pheno", index=False, header=None, sep="\t")
pd.concat(all_covar).to_csv("./all.covar", index=False, header=None, sep="\t")
pd.concat(fam_index).to_csv("./all.ancestry.index", index=False, header=None, sep="\t")

all_gwas[0].to_csv("./EUR.gwas", index=False, header=True, sep="\t")
all_gwas[1].to_csv("./AFR.gwas", index=False, header=True, sep="\t")
all_gwas[2].to_csv("./HIS.gwas", index=False, header=True, sep="\t")

# create keep.subject file, randomly 1500 from 1609 individuals
rng_key, pt_key = random.split(rng_key, 2)
sel_pt = random.randint(
    pt_key, shape=(1500,), minval=0, maxval=pd.concat(fam_index).shape[0] - 1
)
pd.concat(fam_index)[["iid"]].iloc[sel_pt, :].to_csv(
    "./keep.subject", index=False, header=None, sep="\t"
)

rng_key, unif_key = random.split(rng_key, 2)
unif = random.uniform(unif_key, shape=(snps.shape[0],), minval=5, maxval=200)
snps["unif"] = unif
snps[["snp", "unif"]].to_csv("./prior_weights", index=False, header=None, sep="\t")
