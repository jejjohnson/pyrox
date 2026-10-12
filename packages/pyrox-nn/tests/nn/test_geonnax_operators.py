"""The geonnax neural operators are re-exported from ``pyrox_nn`` by identity."""

import geonnax
import geonnax.layers
import pyrox_nn
import pytest


@pytest.mark.parametrize(
    ("name", "source"),
    [
        ("GeoFNO", geonnax),
        ("LearnedDeformation", geonnax),
        ("ManifoldNO", geonnax),
        ("BasisTransform", geonnax.layers),
    ],
)
def test_operator_reexport_identity(name, source):
    assert getattr(pyrox_nn, name) is getattr(source, name)
    assert name in pyrox_nn.__all__
