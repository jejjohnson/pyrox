"""pyrox_ssm recurses in time and must not depend on its sibling packages.

filterax, the ``ensemble`` extra, is imported by ``pyrox_ssm.ensemble`` only,
so ``import pyrox_ssm`` works without it.
"""

from __future__ import annotations

import ast
import pkgutil
import subprocess
import sys
from pathlib import Path

import pyrox_ssm


SRC = Path(pyrox_ssm.__file__).parent
FORBIDDEN = ("pyrox_gp", "pyrox_lgm", "pyrox_nn")
ENSEMBLE = "ensemble.py"


def _imported_modules(path: Path) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.add(node.module)
    return names


def test_no_source_file_imports_a_sibling_package():
    offenders = {
        str(path.relative_to(SRC)): sorted(
            m for m in _imported_modules(path) if m.split(".")[0] in FORBIDDEN
        )
        for path in SRC.rglob("*.py")
    }
    assert not {k: v for k, v in offenders.items() if v}


def test_only_the_ensemble_module_imports_filterax():
    importers = sorted(
        str(path.relative_to(SRC))
        for path in SRC.rglob("*.py")
        if any(m.split(".")[0] == "filterax" for m in _imported_modules(path))
    )
    assert importers == [ENSEMBLE]


def test_importing_every_module_does_not_load_a_sibling_package():
    modules = [m.name for m in pkgutil.walk_packages([str(SRC)], prefix="pyrox_ssm.")]
    code = (
        "import importlib, sys\n"
        "importlib.import_module('pyrox_ssm')\n"
        f"for m in {modules!r}: importlib.import_module(m)\n"
        f"loaded = [n for n in {FORBIDDEN!r} if n in sys.modules]\n"
        "assert not loaded, f'{loaded} was imported'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_pyrox_ssm_imports_without_filterax():
    # sys.modules["filterax"] = None makes every import of filterax raise
    # ImportError, as if the ensemble extra were not installed.
    modules = [
        m.name
        for m in pkgutil.walk_packages([str(SRC)], prefix="pyrox_ssm.")
        if m.name != "pyrox_ssm.ensemble"
    ]
    code = (
        "import importlib, sys\n"
        "sys.modules['filterax'] = None\n"
        "importlib.import_module('pyrox_ssm')\n"
        f"for m in {modules!r}: importlib.import_module(m)\n"
        "try:\n"
        "    import pyrox_ssm.ensemble\n"
        "except ImportError as exc:\n"
        "    assert \"pip install 'pyrox-ssm[ensemble]'\" in str(exc), exc\n"
        "else:\n"
        "    raise AssertionError('pyrox_ssm.ensemble imported without filterax')\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
