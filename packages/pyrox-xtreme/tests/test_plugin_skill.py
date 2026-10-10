"""Keep the Claude Code plugin's guidance runnable and current.

``plugins/xtremax/skills/extreme-value-analysis-with-xtremax/SKILL.md`` is
what agents in downstream projects read before writing extreme value code
on xtremax, so a stale name or a broken example there teaches them the
wrong API. These tests run its worked example (slow: two NUTS fits) and
check that every ``xtremax.X`` / ``xtx.X`` (``xtremax.extraction``) path it
and the plugin's reviewer name still exists.
"""

from __future__ import annotations

import importlib
import re
import types
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "plugins" / "xtremax"
SKILL = PLUGIN / "skills" / "extreme-value-analysis-with-xtremax" / "SKILL.md"
AGENT = PLUGIN / "agents" / "xtremax-reuse-reviewer.md"
if not SKILL.is_file():
    # A built distribution ships tests/ without the repository root.
    pytest.skip("the plugin is not present", allow_module_level=True)

_ALIASES = {"xtx": "xtremax.extraction"}
# Dotted paths, not preceded by a word character, dot or slash (so the
# repository URL ``github.com/jejjohnson/xtremax.git`` is not a name).
_NAME = re.compile(r"(?<![\w./])((?:xtx|xtremax)(?:\.[A-Za-z_]\w*)+)")
_BLOCKS = re.compile(r"```python\n(.*?)```", re.S)


def _exists(dotted: str) -> bool:
    head, *rest = dotted.split(".")
    parts = [*_ALIASES.get(head, head).split("."), *rest]
    for i in range(len(parts), 0, -1):  # the longest importable module prefix
        try:
            obj = importlib.import_module(".".join(parts[:i]))
        except ImportError:
            continue
        break
    for attr in parts[i:]:
        if isinstance(obj, types.ModuleType) and attr not in getattr(
            obj, "__all__", dir(obj)
        ):
            return False
        if not hasattr(obj, attr):
            return False
        obj = getattr(obj, attr)
    return True


@pytest.mark.slow
def test_worked_example_runs_and_its_claims_hold():
    import scipy.stats as st

    (block,) = [
        b for b in _BLOCKS.findall(SKILL.read_text()) if "temporal_block_maxima" in b
    ]
    ns: dict = {}
    exec(block, ns)

    # Daily values are iid N(25, 3²), so the annual maximum's CDF is
    # Φ((z - 25)/3)^365 and the 100-year level solves it = 0.99 (≈ 37.10).
    truth = 25.0 + 3.0 * st.norm.ppf(0.99 ** (1 / 365))
    z_bm = np.asarray(ns["z100_bm"])
    z_pot = np.asarray(ns["z100_pot"])
    # The posterior spread is each estimate's own uncertainty: a calibrated
    # posterior puts the truth within 3 posterior sd (fixed seeds, so this is
    # deterministic; the prose quotes 37.0 ± 0.9 and 36.8 ± 0.5).
    for z in (z_bm, z_pot):
        assert abs(np.median(z) - truth) < 3 * z.std()
    # ~430 declustered peaks pin the level down better than 60 maxima.
    assert z_pot.std() < z_bm.std()
    # Normal maxima are in the Gumbel domain; at n = 365 the effective shape
    # is slightly negative (-0.04 and -0.11 here), well inside 0.2.
    assert abs(float(np.median(ns["post"]["xi"]))) < 0.2
    assert abs(float(np.median(ns["post_pot"]["xi"]))) < 0.2
    # Runs declustering (r = 1) of iid days merges only adjacent exceedances,
    # about 2 % of them at the 98th percentile: θ ≈ 0.98.
    assert 0.95 < ns["theta"] <= 1.0
    # 2 % of 365 days is 7.3 exceedances a year, less the merged runs (≈ 7.2).
    assert 6.0 < float(np.median(ns["post_pot"]["rate"])) < 8.0


@pytest.mark.parametrize("path", [SKILL, AGENT], ids=lambda p: p.name)
def test_named_api_is_current(path: Path):
    # The quantile-regression names are exported only with scikit-learn.
    pytest.importorskip("sklearn")
    stale = sorted({n for n in _NAME.findall(path.read_text()) if not _exists(n)})
    assert not stale, f"{path.name} names objects that do not exist: {stale}"
