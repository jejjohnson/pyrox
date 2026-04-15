import pyrox
import pyrox._core
import pyrox.gp
import pyrox.nn


def test_import():
    assert pyrox is not None


def test_subpackages_importable():
    assert pyrox._core is not None
    assert pyrox.gp is not None
    assert pyrox.nn is not None
