"""Guard that every public symbol appears on exactly one API reference page.

The pages under ``docs/api/`` document curated ``members:`` lists rather than
dumping whole modules, which keeps the rendered table of contents organised
but means a newly exported symbol is silently undocumented unless someone
remembers to add it. These tests turn that into a failure.
"""

from __future__ import annotations

import collections
import importlib
import pathlib
import re

import pytest


DOCS_API = pathlib.Path(__file__).resolve().parents[1] / "docs" / "api"

# Modules with a dedicated reference page, and the sub-modules they re-export
# (documented on their own pages, so they are not members of the parent).
DOCUMENTED_MODULES = [
    "xtremax.distributions",
    "xtremax.primitives",
    "xtremax.extraction",
    "xtremax.simulations",
    "xtremax.point_processes",
    "xtremax.point_processes.primitives",
    "xtremax.point_processes.operators",
    "xtremax.point_processes.distributions",
]
RE_EXPORTED_SUBMODULES = {
    "xtremax.point_processes": {"primitives", "operators", "distributions"},
}

_BLOCK = re.compile(r"^::: ([\w.]+)\n((?:^[ \t].*\n|^\n)*)", re.M)
_MEMBERS = re.compile(r"members:\s*\n((?:\s+- .*\n)+)")


def _documented() -> dict[str, list[str]]:
    """Map module path -> members documented across all API pages."""
    found: dict[str, list[str]] = collections.defaultdict(list)
    for page in sorted(DOCS_API.glob("*.md")):
        for block in _BLOCK.finditer(page.read_text()):
            module, body = block.group(1), block.group(2)
            members = _MEMBERS.search(body)
            if members is None:
                # A whole-module dump documents everything in that module.
                found[module].append("*")
                continue
            for line in members.group(1).strip().splitlines():
                found[module].append(line.strip().removeprefix("- ").strip())
    return found


def _expected(module_path: str) -> set[str]:
    module = importlib.import_module(module_path)
    skip = RE_EXPORTED_SUBMODULES.get(module_path, set())
    return {name for name in module.__all__ if name not in skip}


def _actual(module_path: str, documented: dict[str, list[str]]) -> list[str]:
    names = list(documented.get(module_path, []))
    # Symbols documented from a sub-path page, e.g. conditionally exported
    # members shown via ``xtremax.extraction.quantile_regression``.
    for other, members in documented.items():
        if other.startswith(f"{module_path}.") and other not in DOCUMENTED_MODULES:
            names += members
    return names


@pytest.mark.parametrize("module_path", DOCUMENTED_MODULES)
def test_every_public_symbol_is_documented(module_path: str) -> None:
    documented = _documented()
    names = _actual(module_path, documented)
    if "*" in names:
        pytest.skip(f"{module_path} is documented as a whole-module dump")

    missing = _expected(module_path) - set(names)
    assert not missing, (
        f"{module_path}: not on any docs/api/ page: {sorted(missing)}. "
        "Add each symbol to the members: list of the section it belongs to."
    )


@pytest.mark.parametrize("module_path", DOCUMENTED_MODULES)
def test_no_symbol_is_documented_twice(module_path: str) -> None:
    names = [n for n in _actual(module_path, _documented()) if n != "*"]
    duplicated = sorted(
        n for n, count in collections.Counter(names).items() if count > 1
    )
    assert not duplicated, (
        f"{module_path}: documented in more than one section: {duplicated}. "
        "Each symbol belongs to exactly one section."
    )


@pytest.mark.parametrize("module_path", DOCUMENTED_MODULES)
def test_no_documented_symbol_is_missing_from_the_package(module_path: str) -> None:
    """A members: entry that no longer exists renders as an empty section."""
    module = importlib.import_module(module_path)
    documented = _documented()
    # Only check names listed against this module directly; sub-path pages may
    # legitimately document conditional exports absent from ``__all__``.
    for name in documented.get(module_path, []):
        if name == "*":
            continue
        assert hasattr(module, name), (
            f"{module_path}: docs/api/ lists '{name}', which the module does not "
            "export. Remove it from the members: list or restore the export."
        )
