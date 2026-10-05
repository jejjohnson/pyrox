"""inla() against golden R-INLA fixtures (jejjohnson/pyrox#254).

``fixtures/rinla/make_fixtures.R`` simulates (or loads) each case, fits it in
R-INLA with pyrox-lgm's default priors and writes data and summaries to
JSON; these tests refit the same data with ``lgm.inla`` and compare. R-INLA
runs ``strategy="gaussian"`` with its VB mean correction, which is
pyrox-lgm's default ``"vb"``.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import kernellib as kl
import numpy as np
import pyrox_lgm as lgm
import pytest


jax.config.update("jax_enable_x64", True)

FIXTURES = Path(__file__).parent / "fixtures" / "rinla"


def _load(name):
    return json.loads((FIXTURES / f"{name}.json").read_text())


def _case(name):
    """``(model, data, {R hyperparameter name: (our key, index)}, fixture)``."""
    fx = _load(name)
    d = fx["data"]
    y = np.asarray(d["y"], dtype=float)
    lik = {"Precision for the Gaussian observations": ("lik.prec", None)}
    if name == "rw2_gaussian":
        model = lgm.LGM(
            (lgm.RW2(len(y), name="t"),), lgm.FixedEffects(("intercept",)), lgm.Gaussian()
        )
        data = {"y": y, "t": np.asarray(d["t"])}
        hyper = lik | {"Precision for t": ("t.tau", None)}
    elif name == "ar1_gaussian":
        model = lgm.LGM(
            (lgm.AR1(len(y), name="t"),), lgm.FixedEffects(("intercept",)), lgm.Gaussian()
        )
        data = {"y": y, "t": np.asarray(d["t"])}
        hyper = lik | {"Precision for t": ("t.tau", None), "Rho for t": ("t.rho", None)}
    elif name == "scotland_bym2":
        e = np.asarray(d["edges"], dtype=int)
        graph = kl.graph_from_edges(e[:, 0], e[:, 1], int(d["n_nodes"]))
        model = lgm.LGM(
            (lgm.BYM2(graph, name="region"),),
            lgm.FixedEffects(("intercept", "x")),
            lgm.Poisson(),
        )
        data = {
            "y": y,
            "offset": np.asarray(d["offset"]),
            "x": np.asarray(d["x"]),
            "region": np.asarray(d["region"]),
        }
        hyper = {
            "Precision for region": ("region.tau", None),
            "Phi for region": ("region.phi", None),
        }
    elif name == "spde_poisson":
        spde = lgm.SPDE(
            mesh=(np.asarray(d["vertices"]), np.asarray(d["triangles"], dtype=int)),
            prior=lgm.PCMatern(float(d["range0"]), 0.5, 1.0, 0.01),
            name="s",
        )
        model = lgm.LGM((spde,), lgm.FixedEffects(("intercept",)), lgm.Poisson())
        data = {"y": y, "s": np.asarray(d["loc"])}
        hyper = {"Range for s": ("s.range_sigma", 0), "Stdev for s": ("s.range_sigma", 1)}
    elif name == "pod_bernoulli":
        model = lgm.LGM(
            (lgm.RW2(int(d["n_bins"]), name="size"),),
            lgm.FixedEffects(("intercept", "wind")),
            lgm.Bernoulli(),
        )
        data = {"y": y, "size": np.asarray(d["size"]), "wind": np.asarray(d["wind"])}
        hyper = {"Precision for size": ("size.tau", None)}
    else:  # pragma: no cover
        raise KeyError(name)
    return model, data, hyper, fx


def _fixed_key(r_name):
    return {"(Intercept)": "intercept"}.get(r_name, r_name)
