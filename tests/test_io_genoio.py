import numpy as np
import polars as pl

from sushie import io


class _FakeDataset:
    def __init__(self):
        self.kwargs = None
        self.matrix = np.array([[0.0, np.nan], [1.0, 2.0]], dtype=np.float64)
        self.samples = pl.DataFrame(
            {
                "fid": ["F1", "F2"],
                "iid": ["S1", "S2"],
                "father": ["0", "0"],
                "mother": ["0", "0"],
                "sex": [0, 0],
                "phenotype": [-9, -9],
            }
        )
        self.variants = pl.DataFrame(
            {
                "chrom": ["1", "1"],
                "pos": [101, 202],
                "id": ["rs1", "rs2"],
                "a0": ["A", "C"],
                "a1": ["G", "T"],
            }
        )

    def read(self, **kwargs):
        self.kwargs = kwargs
        return self.matrix, self.samples, self.variants


class _FakeGenoio:
    def __init__(self):
        self.dataset = _FakeDataset()
        self.calls = []

    def bfile(self, path):
        self.calls.append(("bfile", path))
        return self.dataset

    def vcf(self, path):
        self.calls.append(("vcf", path))
        return self.dataset

    def bgen(self, path):
        self.calls.append(("bgen", path))
        return self.dataset


def test_read_triplet_uses_genoio_bfile_and_preserves_contract(monkeypatch):
    fake_genoio = _FakeGenoio()
    monkeypatch.setattr(io, "genoio", fake_genoio, raising=False)

    bim, fam, bed = io.read_triplet("plink-prefix")

    assert fake_genoio.calls == [("bfile", "plink-prefix")]
    assert fake_genoio.dataset.kwargs == {
        "missing": "nan",
        "dtype": "float64",
        "return_samples": True,
        "return_variants": True,
    }
    assert isinstance(bim, pl.DataFrame)
    assert isinstance(fam, pl.DataFrame)
    assert bim.to_dict(as_series=False) == {
        "chrom": ["1", "1"],
        "snp": ["rs1", "rs2"],
        "pos": [101, 202],
        "a0": ["A", "C"],
        "a1": ["G", "T"],
    }
    assert fam.to_dict(as_series=False) == {"iid": ["S1", "S2"]}
    np.testing.assert_allclose(np.asarray(bed), fake_genoio.dataset.matrix, equal_nan=True)


def test_read_vcf_uses_genoio_vcf_and_preserves_sushie_contract(monkeypatch):
    fake_genoio = _FakeGenoio()
    monkeypatch.setattr(io, "genoio", fake_genoio, raising=False)

    bim, fam, bed = io.read_vcf("data.vcf")

    assert fake_genoio.calls == [("vcf", "data.vcf")]
    assert isinstance(bim, pl.DataFrame)
    assert isinstance(fam, pl.DataFrame)
    assert bim.to_dict(as_series=False) == {
        "chrom": ["1", "1"],
        "snp": ["rs1", "rs2"],
        "pos": [101, 202],
        "a0": ["G", "T"],
        "a1": ["A", "C"],
    }
    assert fam.to_dict(as_series=False) == {"iid": ["S1", "S2"]}
    np.testing.assert_allclose(
        np.asarray(bed),
        2.0 - fake_genoio.dataset.matrix,
        equal_nan=True,
    )


def test_read_bgen_uses_genoio_bgen_and_preserves_sushie_contract(monkeypatch):
    fake_genoio = _FakeGenoio()
    monkeypatch.setattr(io, "genoio", fake_genoio, raising=False)

    bim, fam, bed = io.read_bgen("data.bgen")

    assert fake_genoio.calls == [("bgen", "data.bgen")]
    assert fake_genoio.dataset.kwargs == {
        "dosage": "dosage",
        "missing": "nan",
        "dtype": "float64",
        "return_samples": True,
        "return_variants": True,
    }
    assert isinstance(bim, pl.DataFrame)
    assert isinstance(fam, pl.DataFrame)
    assert list(bim.columns) == ["chrom", "snp", "pos", "a0", "a1"]
    assert fam.to_dict(as_series=False) == {"iid": ["S1", "S2"]}
    np.testing.assert_allclose(np.asarray(bed), fake_genoio.dataset.matrix, equal_nan=True)
