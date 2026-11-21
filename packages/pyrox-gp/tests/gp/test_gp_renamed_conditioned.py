"""The two conditioned results renamed by pyrox#516 keep their old names as aliases."""

from __future__ import annotations

import subprocess
import sys

import pyrox_gp
import pytest


# old → new
RENAMED = {
    "NonGaussConditionedGP": "ConditionedNonGaussGP",
    "MultiOutputConditionedGP": "ConditionedMultiOutputGP",
}


@pytest.mark.parametrize(("old", "new"), RENAMED.items())
def test_an_old_name_warns_and_is_the_new_class(old, new):
    with pytest.warns(DeprecationWarning, match=rf"use pyrox_gp\.{new}"):
        alias = getattr(pyrox_gp, old)
    assert alias is getattr(pyrox_gp, new)
    assert new in pyrox_gp.__all__
    assert old not in pyrox_gp.__all__


def test_from_import_of_an_old_name_still_works():
    with pytest.warns(DeprecationWarning):
        from pyrox_gp import NonGaussConditionedGP
    assert NonGaussConditionedGP is pyrox_gp.ConditionedNonGaussGP


def test_importing_pyrox_gp_emits_no_deprecation_warning():
    code = (
        "import warnings; "
        "warnings.simplefilter('error', DeprecationWarning); "
        "import pyrox_gp"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
