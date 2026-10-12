"""The ``pyrox_gp._src.kernels`` deprecation shim is gone; use kernellib."""

import importlib

import pytest


def test_src_kernels_shim_removed():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("pyrox_gp._src.kernels")


def test_kernellib_functional_is_the_import_path():
    from kernellib.functional import rbf_kernel

    assert callable(rbf_kernel)
