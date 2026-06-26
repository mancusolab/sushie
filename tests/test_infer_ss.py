import polars as pl
import pytest

import jax
import jax.numpy as jnp

from sushie.infer_ss import infer_sushie_ss


jax.config.update("jax_enable_x64", True)


def _summary_stat_inputs():
    lds = [jnp.eye(4), jnp.eye(4)]
    zs = [
        jnp.array([5.0, 0.2, -0.1, 0.0]),
        jnp.array([4.6, 0.1, -0.2, 0.0]),
    ]
    return lds, zs


def test_infer_sushie_ss_accepts_vector_sample_sizes():
    lds, zs = _summary_stat_inputs()

    result = infer_sushie_ss(
        lds=lds,
        zs=zs,
        ns=jnp.array([200.0, 220.0]),
        L=1,
        min_snps=4,
        max_iter=5,
        no_update=True,
    )

    assert result.pip_all.shape == (4,)
    assert result.sample_size.shape == (2, 1)
    assert isinstance(result.cs, pl.DataFrame)
    assert isinstance(result.alphas, pl.DataFrame)


def test_infer_sushie_ss_rejects_z_score_count_mismatch():
    lds, zs = _summary_stat_inputs()

    with pytest.raises(ValueError, match="number of Z scores"):
        infer_sushie_ss(
            lds=lds,
            zs=zs[:1],
            ns=jnp.array([200.0, 220.0]),
            L=1,
            min_snps=4,
        )


def test_infer_sushie_ss_rejects_multi_column_sample_sizes():
    lds, zs = _summary_stat_inputs()

    with pytest.raises(ValueError, match="Sample sizes"):
        infer_sushie_ss(
            lds=lds,
            zs=zs,
            ns=jnp.array([[200.0, 201.0], [220.0, 221.0]]),
            L=1,
            min_snps=4,
        )


def test_infer_sushie_ss_rejects_ld_z_score_shape_mismatch():
    lds, zs = _summary_stat_inputs()

    with pytest.raises(ValueError, match="same number of SNPs"):
        infer_sushie_ss(
            lds=lds,
            zs=[zs[0][:-1], zs[1][:-1]],
            ns=jnp.array([200.0, 220.0]),
            L=1,
            min_snps=4,
        )
