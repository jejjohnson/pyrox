"""pyrox_ssm's public surface: ``__all__`` is the contract."""

from __future__ import annotations

import subprocess
import sys

import pyrox_ssm
import pytest


def test_version():
    assert isinstance(pyrox_ssm.__version__, str)
    assert pyrox_ssm.__version__


def test_all_names_exist():
    for name in pyrox_ssm.__all__:
        assert hasattr(pyrox_ssm, name), name


def test_all_has_no_duplicates():
    assert len(pyrox_ssm.__all__) == len(set(pyrox_ssm.__all__))


# The names pyrox#481 changed, old → new.
RENAMED = {
    "SparseConditionedMarkovGP": "ConditionedSparseMarkovGP",
    "NonGaussConditionedMarkovGP": "ConditionedNonGaussMarkovGP",
    "SharedPlusSiteMarkovGP": "SharedPlusSiteMarkovGPPrior",
}


@pytest.mark.parametrize(("old", "new"), RENAMED.items())
def test_a_renamed_class_warns_and_is_the_new_class(old, new):
    with pytest.warns(DeprecationWarning, match=rf"use pyrox_ssm\.{new}"):
        alias = getattr(pyrox_ssm, old)
    assert alias is getattr(pyrox_ssm, new)


@pytest.mark.parametrize(("old", "new"), RENAMED.items())
def test_all_lists_the_new_names_and_not_the_old(old, new):
    assert new in pyrox_ssm.__all__
    assert old not in pyrox_ssm.__all__


def test_from_import_of_an_old_name_still_works():
    with pytest.warns(DeprecationWarning):
        from pyrox_ssm import SparseConditionedMarkovGP
    assert SparseConditionedMarkovGP is pyrox_ssm.ConditionedSparseMarkovGP


def test_an_unknown_name_raises_attribute_error():
    with pytest.raises(AttributeError, match="no attribute 'NoSuchPrior'"):
        _ = pyrox_ssm.NoSuchPrior


def test_importing_pyrox_ssm_emits_no_warning():
    # A fresh interpreter, so an import cached by this session cannot hide one.
    code = "import warnings; warnings.simplefilter('error'); import pyrox_ssm"
    subprocess.run([sys.executable, "-c", code], check=True)
