import pytest

import jax.numpy as jnp
import jax.numpy.linalg as jnla
import jax.random as rdm
import jax.scipy.linalg as jsla
from jax.config import config

import sushie

config.update("jax_enable_x64", True)


@pytest.mark.parametrize("N,K", [(50, 2), (100, 1)])
def test_infer_sushie_simple(N: int, K: int, seed: int = 0):
    key = rdm.PRNGKey(seed)

    P = 4
    L = 2

    key, g_key, b_key, s_key, y_key = rdm.split(key, 5)

    h2g = 0.1

    g_covar_block = jnp.array([[1.0, 0.8], [0.8, 1.0]])
    g_covar = jsla.block_diag(g_covar_block, g_covar_block)
    L_g = jnla.cholesky(g_covar)

    X = rdm.normal(g_key, shape=(K, N, P))
    X = jnp.einsum("knp,pj->knj", X, L_g)

    snps = jnp.array([0, 2])
    beta = jnp.ones((L, K))

    G = jnp.einsum("knl,lk->kn", X[:, :, snps], beta)

    s2gs = jnp.std(G, axis=-1)
    s2es = ((1 / h2g) - 1) * s2gs
    y = G + rdm.normal(y_key, shape=(K, N)) * jnp.sqrt(s2es[:, jnp.newaxis])

    Xs = []
    ys = []
    for k in range(K):
        Xs.append(X[k, :, :])
        ys.append(y[k, :])

    # this really is just sanity check that it doesn't crash...
    res = sushie.infer.infer_sushie(Xs, ys, L=L)

    assert res is not None


@pytest.mark.parametrize("N,P,K,L", [(50, 100, 2, 2), (100, 50, 3, 2)])
def test_infer_sushie(N: int, P: int, K: int, L: int, seed: int = 0):
    key = rdm.PRNGKey(seed)

    key, g_key, b_key, s_key, y_key = rdm.split(key, 5)

    h2g = 0.1
    rho = 0.8 * h2g
    covar = (
        jnp.diag(h2g * jnp.ones(K))
        + rho * jnp.ones((K, K))
        - jnp.diag(rho * jnp.ones(K))
    )

    X = rdm.normal(g_key, shape=(K, N, P))
    snps = rdm.choice(s_key, P, shape=(L,), replace=False)
    beta = rdm.multivariate_normal(b_key, mean=jnp.zeros(K), cov=covar, shape=(L,))

    G = jnp.einsum("knl,lk->kn", X[:, :, snps], beta)

    s2gs = jnp.std(G, axis=-1)
    s2es = ((1 / h2g) - 1) * s2gs
    y = G + rdm.normal(y_key, shape=(K, N)) * jnp.sqrt(s2es[:, jnp.newaxis])

    Xs = []
    ys = []
    for k in range(K):
        Xs.append(X[k, :, :])
        ys.append(y[k, :])

    # this really is just sanity check that it doesn't crash...
    res = sushie.infer.infer_sushie(Xs, ys, L=L)
    assert res is not None
