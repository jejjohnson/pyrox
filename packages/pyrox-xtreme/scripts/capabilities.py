"""Generate, or check, the capability index: every public name in xtremax.

``docs/api/capabilities.md`` lists each public object of xtremax's public
modules once, at its subpackage home, grouped by the module that defines it,
with the first sentence of its docstring. Agents and people search it before
writing a helper ("Reuse before you write" in ``AGENTS.md``).

Usage::

    make capabilities                                            # rewrite
    uv run --extra threshold python scripts/capabilities.py --check  # stale?

``tests/test_capabilities.py`` runs the check. The index lists
``XarrayQuantileRegressor`` and ``quantile_regression_threshold``, which
``xtremax.extraction`` exports only when scikit-learn (the ``[threshold]``
extra) is installed, so the script refuses to run without it; ``make
capabilities`` runs it with ``uv run --extra threshold``.

``--check`` also fails when two xtremax homes export the same name for
different objects, unless ``ALLOWED_SHARED_NAMES`` gives the reason (or the
name follows the deliberate pattern in ``_allowed_by_layout``).
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import re
import sys
import types
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "docs" / "api" / "capabilities.md"

# Public homes, in order. Each name is listed once, under the first home that
# exports it: the subpackages are the canonical homes (one API page each), and
# the top-level ``xtremax`` namespace only re-exports from two of them, so it
# comes last and gets a section only if it ever exports something new.
MODULES: list[str] = [
    "xtremax.distributions",
    "xtremax.primitives",
    "xtremax.extraction",
    "xtremax.point_processes",
    "xtremax.point_processes.primitives",
    "xtremax.point_processes.operators",
    "xtremax.point_processes.distributions",
    "xtremax.simulations",
    "xtremax",
]

# A module with more public names than this is split into one table per
# defining module (gev, gpd, hawkes, ...), provided the groups hold two names
# on average; otherwise (one class per module, as in the operators) it is a
# single table.
_GROUP_ABOVE = 10

# Names two homes may share for different objects, and why. The deliberate
# pattern (``_allowed_by_layout``) needs no entry here.
ALLOWED_SHARED_NAMES: dict[str, str] = {}

_ROLE = re.compile(r":(\w+):`(~?)([^`]+)`")
# A full stop that ends a sentence (not "e.g." / "i.e." / "etc.").
_SENTENCE_END = re.compile(r"(?<!e\.g)(?<!i\.e)(?<!etc)\.\s+(?=[A-Z])")


def _role(match: re.Match[str]) -> str:
    """Turn a Sphinx role into Markdown: ``:math:`x``` → ``$x$``, others code."""
    role, tilde, target = match.groups()
    if role == "math":
        # A bare | would split the Markdown table row.
        return "$" + target.replace("|", r"\vert ") + "$"
    return f"`{target.rsplit('.', 1)[-1] if tilde else target}`"


def _first_line(text: str) -> str:
    """The docstring's summary: its first sentence, on one line."""
    paragraph = text.strip().split("\n\n")[0]
    line = " ".join(part.strip() for part in paragraph.splitlines())
    line = _ROLE.sub(_role, line)
    sentence = _SENTENCE_END.split(line, maxsplit=1)[0]
    if sentence != line:
        line = sentence + "."
    if len(line) > 160:
        line = line[:157].rstrip() + "..."
    return line.replace("|", "\\|")


def _kind(obj: object) -> str:
    if inspect.isclass(obj):
        return "class"
    if callable(obj):
        return "function"
    return "constant"


def _origin(obj: object) -> str:
    module = getattr(obj, "__module__", None)
    if not isinstance(module, str) or not (callable(obj) or inspect.isclass(obj)):
        module = type(obj).__module__
    return module


def _summary(obj: object) -> str:
    if not _origin(obj).startswith("xtremax"):
        return f"Re-exported from `{_origin(obj).split('.')[0]}`."
    if not callable(obj):
        value = repr(obj)
        return f"`{value if len(value) <= 60 else value[:57] + '...'}`".replace(
            "|", "\\|"
        )
    return _first_line(inspect.getdoc(obj) or "")


def _public(module: types.ModuleType) -> list[str]:
    names = getattr(module, "__all__", None)
    if names is None:
        names = [n for n in dir(module) if not n.startswith("_")]
    return [
        n
        for n in names
        if not n.startswith("__")
        and not isinstance(getattr(module, n, None), types.ModuleType)
    ]


def _module_title(name: str) -> str:
    module = sys.modules.get(name) or importlib.import_module(name)
    title = _first_line(module.__doc__ or "").rstrip(".")
    return f"{title} (`{name}`)" if title else f"`{name}`"


def _allowed_by_layout(homes: dict[str, object]) -> bool:
    """The deliberate same-name pairs.

    Each NumPyro wrapper in ``xtremax.point_processes.distributions`` is named
    after the equinox operator it wraps in
    ``xtremax.point_processes.operators`` (``HomogeneousPoissonProcess``, ...).
    """
    layers = sorted(p.rsplit(".", 1)[0] for p in homes)
    return layers == [
        "xtremax.point_processes.distributions",
        "xtremax.point_processes.operators",
    ]


def _require_threshold_extra() -> None:
    if importlib.util.find_spec("sklearn") is None:
        raise SystemExit(
            "scripts/capabilities.py needs scikit-learn (the [threshold] extra), "
            "which xtremax.extraction needs to export its quantile-regression "
            "names: run it as `uv run --extra threshold python "
            "scripts/capabilities.py` (or `make capabilities`)."
        )


def collect() -> tuple[dict[str, list[tuple]], list[str]]:
    """``{home: [(name, kind, summary, defining module)]}`` and the clashes."""
    _require_threshold_extra()
    index: dict[str, list[tuple]] = defaultdict(list)
    owners: dict[str, dict[str, object]] = defaultdict(dict)
    seen: set[int] = set()
    for module_name in MODULES:
        module = importlib.import_module(module_name)
        for name in _public(module):
            obj = getattr(module, name)
            owners[name][f"{module_name}.{name}"] = obj
            if id(obj) in seen and not isinstance(obj, int | float | str):
                continue
            seen.add(id(obj))
            index[module_name].append((name, _kind(obj), _summary(obj), _origin(obj)))
    clashes = []
    for name, homes in sorted(owners.items()):
        if name in ALLOWED_SHARED_NAMES or _allowed_by_layout(homes):
            continue
        if len({id(o) for o in homes.values()}) > 1:
            clashes.append(f"{name}: {', '.join(sorted(homes))}")
    return index, clashes


def _table(rows: list[tuple]) -> list[str]:
    out = ["| Name | Kind | What it does |", "|---|---|---|"]
    out += [f"| `{r[0]}` | {r[1]} | {r[2]} |" for r in rows]
    return [*out, ""]


def _body() -> tuple[list[str], int]:
    index, _ = collect()
    out: list[str] = []
    total = 0
    for module_name in MODULES:
        rows = index[module_name]
        if not rows:
            continue
        total += len(rows)
        out += [f"## `{module_name}`", ""]
        groups: dict[str, list[tuple]] = defaultdict(list)
        for row in rows:
            groups[row[3]].append(row)
        if len(rows) <= _GROUP_ABOVE or len(groups) > len(rows) // 2:
            out += _table(sorted(rows))
            continue
        for origin in sorted(groups):
            out += [f"### {_module_title(origin)}", "", *_table(sorted(groups[origin]))]
    return out, total


def render() -> str:
    body, total = _body()
    out = [
        "# Capability index",
        "",
        "<!-- Generated by scripts/capabilities.py — do not edit by hand. -->",
        "",
        f"Every public name in xtremax ({total} of them), each listed once at its",
        "subpackage home and grouped by the module that defines it, with the first",
        "sentence of its docstring. `import xtremax` re-exports the five EVT",
        "distribution classes and every name in `xtremax.primitives`; the",
        "point-process wrappers in `xtremax.point_processes.distributions` share",
        "their names with the operators they wrap. Search this page before writing",
        "a helper: if what you need is here, compose it; if it is almost here,",
        "extend it where it lives. Regenerate with `make capabilities` after",
        "changing a public API (`tests/test_capabilities.py` checks it is",
        "current).",
        "",
        *body,
    ]
    return "\n".join(out).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if stale")
    args = parser.parse_args()
    text = render()
    status = 0
    _, clashes = collect()
    if clashes:
        print("Names bound to different objects in two homes:")
        print("\n".join(f"  {c}" for c in clashes))
        print("Rename one, or add it to ALLOWED_SHARED_NAMES with a reason.")
        status = 1
    if args.check:
        current = INDEX.read_text(encoding="utf-8") if INDEX.exists() else ""
        if current != text:
            print(f"{INDEX.relative_to(ROOT)} is stale; run `make capabilities`")
            status = 1
    else:
        INDEX.write_text(text, encoding="utf-8")
        print(f"wrote {INDEX.relative_to(ROOT)}")
    return status


if __name__ == "__main__":
    sys.exit(main())
