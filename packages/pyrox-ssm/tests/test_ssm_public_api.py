"""pyrox_ssm's public surface: ``__all__`` is the contract."""

from __future__ import annotations

import pyrox_ssm


def test_version():
    assert isinstance(pyrox_ssm.__version__, str)
    assert pyrox_ssm.__version__


def test_all_names_exist():
    for name in pyrox_ssm.__all__:
        assert hasattr(pyrox_ssm, name), name


def test_all_has_no_duplicates():
    assert len(pyrox_ssm.__all__) == len(set(pyrox_ssm.__all__))
