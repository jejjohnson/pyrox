"""Fixtures for the pyrox-gauss tests, which moved here from gaussx with the code.

``getkey`` is gaussx's per-test seeded key (a CRC of the node id), so the moved
tests draw the same models they drew in gaussx and a failure reproduces by
rerunning it; ``EQX_GETKEY_SEED`` overrides the seed. x64 is on from the
workspace's root ``conftest.py``.
"""

from __future__ import annotations

import os
import zlib

import equinox.internal as eqxi
import pytest


@pytest.fixture
def getkey(request):
    """A ``GetKey`` seeded per test from a CRC of its node id."""
    override = os.environ.get("EQX_GETKEY_SEED")
    seed = (
        int(override)
        if override is not None
        else zlib.crc32(request.node.nodeid.encode())
    )
    return eqxi.GetKey(seed=seed)
