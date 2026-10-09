"""Keep the Claude Code plugin's guidance runnable and current.

``plugins/pyrox/skills/bayesian-models-with-pyrox/SKILL.md`` is what agents
in downstream projects read before writing models on pyrox, so a stale name
or a broken example there teaches them the wrong API. These tests run its
worked example and check that every ``pyrox_gp.X`` / ``pyrox_nn.X`` /
``pyrox_lgm.X`` / ``pyrox.inference.X`` / ``pyrox._core.X`` it and the
plugin's reviewer name is a current public name.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]
PLUGIN = ROOT / "plugins" / "pyrox"
SKILL = PLUGIN / "skills" / "bayesian-models-with-pyrox" / "SKILL.md"
AGENT = PLUGIN / "agents" / "pyrox-reuse-reviewer.md"
if not SKILL.is_file():
    # A built distribution ships tests/ without the workspace root.
    pytest.skip("the plugin is not present", allow_module_level=True)

_NAME = re.compile(
    r"\b(pyrox_gp|pyrox_nn|pyrox_lgm|pyrox\.inference|pyrox\._core)\.([A-Za-z_]\w*)"
)
_BLOCKS = re.compile(r"```python\n(.*?)```", re.S)


@pytest.mark.slow
def test_worked_example_runs_and_recovers_the_slope():
    (block,) = [b for b in _BLOCKS.findall(SKILL.read_text()) if "BayesianLinear" in b]
    ns: dict = {}
    exec(block, ns)
    assert ns["sites"] == ["linear.weight", "linear.bias", "obs"]
    # y = 2x + noise of sd 0.1 on 50 points: the posterior sd of the slope is
    # about 0.1 / √(Σx²) ≈ 0.024, so 0.15 is several sd and still catches a
    # broken model.
    slope = float(np.squeeze(ns["svi_result"].params["linear.weight_auto_loc"]))
    assert abs(slope - 2.0) < 0.15
    assert ns["draws"].shape == (100, 50)
    assert {
        "RBF.lengthscale_auto_loc",
        "RBF.variance_auto_loc",
        "noise_auto_loc",
    } <= set(ns["gp_result"].params)


@pytest.mark.parametrize("path", [SKILL, AGENT], ids=lambda p: p.name)
def test_named_api_is_current(path: Path):
    stale = []
    for module_name, attr in set(_NAME.findall(path.read_text())):
        module = importlib.import_module(module_name)
        if attr in getattr(module, "__all__", dir(module)):
            continue
        try:  # a submodule path such as ``pyrox_nn.api``
            importlib.import_module(f"{module_name}.{attr}")
        except ImportError:
            stale.append(f"{module_name}.{attr}")
    assert not stale, f"{path.name} names objects that do not exist: {sorted(stale)}"
