# pattern: Imperative Shell

import argparse

from functools import partial

import sushie.cli as cli
import sushie.io as io


def _individual_args(**overrides):
    args = {
        "pheno": ["pheno.tsv"],
        "ancestry_index": None,
        "trait": "trait",
        "plink": None,
        "plink2": None,
        "plink2_dosage": None,
        "vcf": None,
        "bgen": None,
        "covar": None,
        "keep": None,
        "pi": "uniform",
        "seed": 1,
        "cv": False,
        "cv_num": 5,
        "maf": 0.01,
        "meta": False,
        "mega": False,
        "chrom": None,
        "start": None,
        "end": None,
    }
    args.update(overrides)
    return argparse.Namespace(**args)


def _summary_args(**overrides):
    args = {
        "gwas": ["gwas.tsv"],
        "trait": "trait",
        "plink": None,
        "plink2": None,
        "plink2_dosage": None,
        "vcf": None,
        "bgen": None,
        "ld": None,
        "sample_size": [100],
        "pi": "uniform",
        "seed": 1,
        "maf": 0.01,
        "meta": False,
        "mega": False,
        "her": False,
        "cv": False,
        "chrom": None,
        "start": None,
        "end": None,
        "gwas_sig": 1.0,
        "ld_adjust": 0.0,
    }
    args.update(overrides)
    return argparse.Namespace(**args)


def test_parameter_check_selects_plink2_reader_for_individual_data():
    _, _, _, _, geno_path, geno_func = cli.parameter_check(_individual_args(plink2=["cohort"]))

    assert geno_path == ["cohort"]
    assert geno_func is io.read_pfile


def test_parameter_check_selects_plink2_dosage_reader_for_individual_data():
    _, _, _, _, geno_path, geno_func = cli.parameter_check(_individual_args(plink2_dosage=["cohort"]))

    assert geno_path == ["cohort"]
    assert isinstance(geno_func, partial)
    assert geno_func.func is io.read_pfile
    assert geno_func.keywords == {"dosage": True}


def test_parameter_check_ss_selects_plink2_reader_for_summary_data():
    _, _, geno_path, geno_func, ld_file = cli.parameter_check_ss(_summary_args(plink2=["cohort"]))

    assert geno_path == ["cohort"]
    assert geno_func is io.read_pfile
    assert ld_file is False


def test_parameter_check_ss_selects_plink2_dosage_reader_for_summary_data():
    _, _, geno_path, geno_func, ld_file = cli.parameter_check_ss(_summary_args(plink2_dosage=["cohort"]))

    assert geno_path == ["cohort"]
    assert isinstance(geno_func, partial)
    assert geno_func.func is io.read_pfile
    assert geno_func.keywords == {"dosage": True}
    assert ld_file is False


def test_main_parses_plink2_flag(monkeypatch, tmp_path):
    captured_args: argparse.Namespace | None = None

    def capture_finemap(args):
        nonlocal captured_args
        captured_args = args
        return 0

    monkeypatch.setattr(cli, "run_finemap", capture_finemap)

    status = cli._main(
        [
            "finemap",
            "--pheno",
            "unused.pheno",
            "--plink2",
            "unused-pgen-prefix",
            "--output",
            str(tmp_path / "plink2"),
            "--quiet",
        ]
    )

    assert status == 0
    assert captured_args is not None
    assert captured_args.plink2 == ["unused-pgen-prefix"]
    assert captured_args.plink2_dosage is None


def test_main_parses_plink2_dosage_flag(monkeypatch, tmp_path):
    captured_args: argparse.Namespace | None = None

    def capture_finemap(args):
        nonlocal captured_args
        captured_args = args
        return 0

    monkeypatch.setattr(cli, "run_finemap", capture_finemap)

    status = cli._main(
        [
            "finemap",
            "--pheno",
            "unused.pheno",
            "--plink2-dosage",
            "unused-pgen-prefix",
            "--output",
            str(tmp_path / "plink2-dosage"),
            "--quiet",
        ]
    )

    assert status == 0
    assert captured_args is not None
    assert captured_args.plink2 is None
    assert captured_args.plink2_dosage == ["unused-pgen-prefix"]


def test_vcf_help_describes_alt_as_counting_allele():
    parser = argparse.ArgumentParser()
    finemap = cli.build_finemap_parser(parser.add_subparsers())

    help_text = " ".join(finemap.format_help().split())

    assert "count ALT allele" in help_text
    assert "count RFE allele" not in help_text
