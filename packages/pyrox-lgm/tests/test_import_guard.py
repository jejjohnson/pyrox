"""pyrox_lgm is precision-form and must not depend on pyrox_gp."""

from __future__ import annotations

import ast
import pkgutil
import subprocess
import sys
from pathlib import Path

import pyrox_lgm


SRC = Path(pyrox_lgm.__file__).parent


def _imported_modules(path: Path) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.add(node.module)
    return names


def test_no_source_file_imports_pyrox_gp():
    offenders = {
        str(path.relative_to(SRC)): sorted(
            m for m in _imported_modules(path) if m.split(".")[0] == "pyrox_gp"
        )
        for path in SRC.rglob("*.py")
    }
    assert not {k: v for k, v in offenders.items() if v}


def test_importing_every_module_does_not_load_pyrox_gp():
    modules = [m.name for m in pkgutil.walk_packages([str(SRC)], prefix="pyrox_lgm.")]
    code = (
        "import importlib, sys\n"
        f"for m in {modules!r}: importlib.import_module(m)\n"
        "assert 'pyrox_gp' not in sys.modules, 'pyrox_gp was imported'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
