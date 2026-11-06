"""pyrox_nn and pyrox_gp are siblings over pyrox: pyrox_nn must not import pyrox_gp."""

from __future__ import annotations

import ast
import pkgutil
import subprocess
import sys
from pathlib import Path

import pyrox_nn


SRC = Path(pyrox_nn.__file__).parent


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


def test_every_module_imports_with_pyrox_gp_uninstalled():
    modules = [m.name for m in pkgutil.walk_packages([str(SRC)], prefix="pyrox_nn.")]
    # None in sys.modules makes `import pyrox_gp` raise, as if it were not installed.
    code = (
        "import importlib, sys\n"
        "sys.modules['pyrox_gp'] = None\n"
        "import pyrox_nn\n"
        f"for m in {modules!r}: importlib.import_module(m)\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_project_does_not_declare_pyrox_gp():
    pyproject = SRC.parents[1] / "pyproject.toml"
    head = pyproject.read_text().split("[project.optional-dependencies]")[0]
    assert '"pyrox-gp"' not in head
