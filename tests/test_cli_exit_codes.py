# pattern: Imperative Shell

import argparse
import traceback

import sushie.cli as cli


def test_run_finemap_returns_nonzero_when_finemap_fails(monkeypatch):
    monkeypatch.setattr(traceback, "format_exception", lambda *args: [])
    args = argparse.Namespace(
        jax_precision=64,
        platform="cpu",
        summary=False,
        pheno=None,
        trait="trait",
    )

    assert cli.run_finemap(args) == 1


def test_main_returns_subcommand_exit_code(monkeypatch, tmp_path):
    def fail_finemap(args):
        return 3

    monkeypatch.setattr(cli, "run_finemap", fail_finemap)

    status = cli._main(
        [
            "finemap",
            "--pheno",
            "unused.pheno",
            "--vcf",
            "unused.vcf",
            "--output",
            str(tmp_path / "sushie-exit-code"),
            "--quiet",
        ]
    )

    assert status == 3
