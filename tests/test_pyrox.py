"""Package-level smoke tests — import surface and Core-wave exports.

Behavior of individual _core primitives is covered in
``test_core_pyrox_module.py``, ``test_core_parameterized.py``, and
``test_core_numpyro_integration.py``. This file only verifies that the
advertised public surface is importable and shaped as documented.
"""

from __future__ import annotations

import dataclasses

import pyrox
import pyrox._core
import pyrox.gp
import pyrox.nn


def test_top_level_exports_version():
    assert isinstance(pyrox.__version__, str)
    assert pyrox.__version__


def test_subpackages_importable():
    assert pyrox._core is not None
    assert pyrox.gp is not None
    assert pyrox.nn is not None


def test_core_public_surface_matches_contract():
    """Wave 1 contract: `_core` re-exports the documented bridge primitives."""
    from pyrox._core import (
        Parameterized,
        PyroxModule,
        PyroxParam,
        PyroxSample,
        pyrox_method,
    )

    # Sanity: classes and callable.
    assert isinstance(PyroxModule, type)
    assert isinstance(Parameterized, type)
    assert issubclass(Parameterized, PyroxModule)
    assert callable(pyrox_method)
    # Descriptors are value-object containers, not trainable modules:
    # PyroxParam is a NamedTuple (tuple subclass with _fields), PyroxSample
    # is a plain dataclass.
    assert issubclass(PyroxParam, tuple)
    assert hasattr(PyroxParam, "_fields")
    assert dataclasses.is_dataclass(PyroxSample)
