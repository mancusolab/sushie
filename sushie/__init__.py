# Copyright Contributors to the SuSiE-py project.
# SPDX-License-Identifier: Apache-2.0

import logging

from sushie.infer import run_sushie
from sushie.version import __version__


# next 4 lines taken from https://github.com/pyro-ppl/numpyro/blob/master/numpyro/__init__.py
# Copyright Pyro devs
# filter out this annoying warning, which raises even when we install CPU-only jaxlib
def _filter_absl_cpu_warning(record):
    return not record.getMessage().startswith("No GPU/TPU found, falling back to CPU.")


logging.getLogger("absl").addFilter(_filter_absl_cpu_warning)


__all__ = [
    "__version__",
    "run_sushie",
]

VERSION = __version__
LOG = "MAIN"