"""pyrox_lgm's public surface: ``__all__`` is the contract."""

from __future__ import annotations

import pyrox_lgm


def test_version():
    assert isinstance(pyrox_lgm.__version__, str)
    assert pyrox_lgm.__version__


def test_all_names_exist():
    for name in pyrox_lgm.__all__:
        assert hasattr(pyrox_lgm, name), name


def test_all_has_no_duplicates():
    assert len(pyrox_lgm.__all__) == len(set(pyrox_lgm.__all__))
