"""The pyrox core is the Equinox-NumPyro bridge, with no Gaussian linear algebra."""

from __future__ import annotations

import ast
import pkgutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pyrox
import pyrox.inference
import pytest


SRC = Path(pyrox.__file__).parent
FORBIDDEN = {"gaussx", "lineax", "kernellib", "geonnax"}


def _imported_roots(path: Path) -> set[str]:
    roots = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            roots.add(node.module.split(".")[0])
    return roots


def test_no_source_file_imports_a_forbidden_package():
    offenders = {
        str(path.relative_to(SRC)): sorted(_imported_roots(path) & FORBIDDEN)
        for path in SRC.rglob("*.py")
    }
    assert not {name: found for name, found in offenders.items() if found}


@pytest.mark.parametrize("forbidden", sorted(FORBIDDEN))
def test_every_module_imports_with_the_package_uninstalled(forbidden):
    modules = [m.name for m in pkgutil.walk_packages([str(SRC)], prefix="pyrox.")]
    # None in sys.modules makes the import raise, as if it were not installed.
    code = (
        "import importlib, sys\n"
        f"sys.modules[{forbidden!r}] = None\n"
        f"for m in {modules!r}: importlib.import_module(m)\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_project_does_not_declare_a_forbidden_package():
    meta = tomllib.loads((SRC.parents[1] / "pyproject.toml").read_text())
    project = meta["project"]
    extras = project.get("optional-dependencies", {}).values()
    requirements = [*project["dependencies"], *(r for v in extras for r in v)]
    declared = {r.split(">")[0].split("[")[0].split(" ")[0] for r in requirements}
    assert not declared & FORBIDDEN
    assert "laplace" not in project.get("optional-dependencies", {})


@pytest.mark.parametrize("name", ["LastLayerLaplace", "last_layer_laplace"])
def test_old_laplace_import_warns_and_returns_the_pyrox_nn_object(name):
    pyrox_nn = pytest.importorskip("pyrox_nn")
    with pytest.warns(DeprecationWarning, match=f"from pyrox_nn import {name}"):
        got = getattr(pyrox.inference, name)
    assert got is getattr(pyrox_nn, name)


@pytest.mark.parametrize("name", ["LastLayerLaplace", "last_layer_laplace"])
def test_old_laplace_import_names_pyrox_nn_when_it_is_not_installed(name):
    # None in sys.modules makes the import raise, as if it were not installed.
    code = (
        "import sys\n"
        "sys.modules['pyrox_nn'] = None\n"
        "try:\n"
        f"    from pyrox.inference import {name}\n"
        "except ImportError as error:\n"
        "    assert 'install pyrox-nn' in str(error), error\n"
        "else:\n"
        "    raise AssertionError('expected an ImportError')\n"
    )
    subprocess.run([sys.executable, "-W", "ignore", "-c", code], check=True)


def test_importing_the_core_does_not_load_pyrox_nn():
    modules = [m.name for m in pkgutil.walk_packages([str(SRC)], prefix="pyrox.")]
    code = (
        "import importlib, sys\n"
        f"for m in {modules!r}: importlib.import_module(m)\n"
        "assert 'pyrox_nn' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_inference_still_rejects_an_unknown_name():
    with pytest.raises(AttributeError, match="no_such_name"):
        pyrox.inference.no_such_name  # noqa: B018
