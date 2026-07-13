"""pyrox: Equinox-to-NumPyro bridge primitives and ensemble inference.

The GP building blocks live in the ``pyrox-gp`` package (``pyrox_gp``)
and the Bayesian NN layers in ``pyrox-nn`` (``pyrox_nn``); both build
on the primitives exported here.
"""

from pyrox import _core, inference


__version__ = "0.1.0"  # x-release-please-version
__all__ = [
    "__version__",
    "_core",
    "inference",
]
