"""Run every ``Examples:`` block in pyrox_lgm, so the docs stay executable."""

from __future__ import annotations

import doctest
import importlib
import pkgutil

import pyrox_lgm
import pytest


MODULES = sorted(
    m.name for m in pkgutil.walk_packages(pyrox_lgm.__path__, prefix="pyrox_lgm.")
)


@pytest.mark.parametrize("name", MODULES)
def test_doctests(name):
    result = doctest.testmod(
        importlib.import_module(name), optionflags=doctest.ELLIPSIS
    )
    assert result.failed == 0
