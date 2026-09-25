"""The deprecated ``pyrox_gp._src.kernels`` shim warns and re-exports kernellib.

The kernel-function tests themselves moved to kernellib with the functions
(``tests/test_functional_kernels.py`` there, ported verbatim).
"""

import importlib
import sys

import kernellib.functional as F
import pytest


def test_import_warns_and_reexports_kernellib_functions():
    sys.modules.pop("pyrox_gp._src.kernels", None)
    with pytest.warns(DeprecationWarning, match="kernellib.functional"):
        mod = importlib.import_module("pyrox_gp._src.kernels")
    names = [
        "constant_kernel",
        "cosine_kernel",
        "kernel_add",
        "kernel_mul",
        "linear_kernel",
        "matern_kernel",
        "periodic_kernel",
        "polynomial_kernel",
        "rational_quadratic_kernel",
        "rbf_kernel",
        "white_kernel",
    ]
    for name in names:
        assert getattr(mod, name) is getattr(F, name), name


def test_kernel_protocol_is_kernellib_abstract_kernel():
    import kernellib
    import pyrox_gp

    assert pyrox_gp.Kernel is kernellib.AbstractKernel
    assert isinstance(pyrox_gp.RBF(), kernellib.AbstractKernel)
