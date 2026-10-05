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
            (lgm.RW2(len(y), name="t"),),
            lgm.FixedEffects(("intercept",)),
            lgm.Gaussian(),
        )
        data = {"y": y, "t": np.asarray(d["t"])}
        hyper = lik | {"Precision for t": ("t.tau", None)}
    elif name == "ar1_gaussian":
        model = lgm.LGM(
            (lgm.AR1(len(y), name="t"),),
            lgm.FixedEffects(("intercept",)),
            lgm.Gaussian(),
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
        hyper = {
            "Range for s": ("s.range_sigma", 0),
            "Stdev for s": ("s.range_sigma", 1),
        }
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


def _intrinsic_offset(name, model):
    """Our log marginal likelihood minus R-INLA's, from the intrinsic priors.

    An intrinsic prior's normalising constant is a convention. pyrox-lgm
    keeps the exact one of the density on the constrained space,
    ``0.5 log|S|* - 0.5 rank log(2 pi)`` for a structure ``S``; R-INLA
    drops ``0.5 log|S|*`` for ``rw2`` and both terms for ``bym2``'s ICAR
    part (measured at fixed hyperparameters, where the two agree to 0.02
    once this is removed). Proper models (AR(1), SPDE) need no offset.
    """
    if name in ("ar1_gaussian", "spde_poisson"):
        return 0.0
    comp = model.components[0]
    if isinstance(comp, lgm.RW2):
        S = float(comp.scale) * np.asarray(comp.structure.as_matrix())
    else:
        S = np.asarray(comp.structure.as_matrix())
    ev = np.linalg.eigvalsh(S)
    ev = ev[ev > 1e-8 * ev.max()]
    half_logpdet = 0.5 * np.sum(np.log(ev))
    if name == "scotland_bym2":
        return half_logpdet - 0.5 * ev.size * np.log(2 * np.pi)
    return half_logpdet


CASES = [
    "rw2_gaussian",
    "ar1_gaussian",
    "scotland_bym2",
    "spde_poisson",
    "pod_bernoulli",
]

# Per case, measured against R-INLA 26.8.7 and bounded with a margin (every
# interval contains 1, so an exact match always passes), all in
# R-INLA's own posterior sds:
#   random_z: max |latent mean difference| / sd;  random_sd / fixed_sd: the
#   range of sd ratios (ours / R-INLA's).
# The per-theta fits agree with R-INLA to 1e-6 at fixed hyperparameters; the
# gaps are in the integration over theta (design and hyperparameter
# marginals), largest where the hyperposterior is flat and skewed (AR(1)
# noise precision, POD tau: log-scale posterior sd ~1.8). Narrowing them is
# tracked separately.
TOLERANCES = {
    "rw2_gaussian": {
        "random_z": 0.03,
        "random_sd": (0.99, 1.02),
        "fixed_sd": (0.99, 1.02),
    },  # measured 0.012, [1.002, 1.007], 1.003
    "ar1_gaussian": {
        "random_z": 0.4,
        "random_sd": (0.85, 1.02),
        "fixed_sd": (0.98, 1.03),
    },  # measured 0.31, [0.872, 0.957], 1.001
    "scotland_bym2": {
        "random_z": 0.15,
        "random_sd": (0.89, 1.02),
        "fixed_sd": (0.96, 1.02),
    },  # measured 0.078, [0.908, 1.001], [0.985, 0.988]
    "spde_poisson": {
        "random_z": 0.15,
        "random_sd": (0.92, 1.02),
        "fixed_sd": (0.92, 1.02),
    },  # measured 0.076, [0.939, 0.961], 0.948
    "pod_bernoulli": {
        "random_z": 0.25,
        "random_sd": (0.89, 1.02),
        "fixed_sd": (0.96, 1.02),
    },  # measured 0.148, [0.912, 0.996], [0.986, 1.000]
}


@pytest.mark.parametrize("name", CASES)
def test_fixture_hyperparameters_are_all_mapped(name):
    model, _, hyper, fx = _case(name)
    assert set(hyper) == set(fx["vb"]["hyperpar"])
    assert len(fx["vb"]["theta_mode"]) == len(hyper)
    # Every mapped key exists, and indexed ones (SPDE's pair) are in range.
    shapes = {key: shape for key, _, shape in model._sizes()}
    for key, idx in hyper.values():
        assert key in shapes, key
        assert (idx is None) == (shapes[key] == ()), key
        if idx is not None:
            assert 0 <= idx < shapes[key][0], key


@pytest.fixture(scope="module")
def fits():
    out = {}

    def get(name):
        if name not in out:
            model, data, hyper, fx = _case(name)
            out[name] = (model, hyper, fx, lgm.inla(model, data))
        return out[name]

    return get


@pytest.mark.slow
@pytest.mark.parametrize("name", CASES)
def test_inla_matches_r_inla(fits, name):
    model, hyper, fx, res = fits(name)
    ref, tol = fx["vb"], TOLERANCES[name]

    # Fixed effects: means within 0.1 sd (measured <= 0.063).
    for r_name, s in ref["fixed"].items():
        ours = res.fixed[_fixed_key(r_name)]
        assert abs(float(ours.mean) - s["mean"]) < 0.1 * s["sd"], r_name
        lo, hi = tol["fixed_sd"]
        assert lo < float(ours.sd) / s["sd"] < hi, r_name

    # The latent field, node by node.
    ((_, s),) = ref["random"].items()
    ours = res.random[model.components[0].name]
    sd_r = np.asarray(s["sd"])
    z = np.abs(np.asarray(ours.mean) - np.asarray(s["mean"])) / sd_r
    assert np.max(z) < tol["random_z"]
    ratio = np.asarray(ours.sd) / sd_r
    lo, hi = tol["random_sd"]
    assert lo < ratio.min() and ratio.max() < hi

    # Hyperparameter medians within 0.75 posterior sd on R-INLA's internal
    # scale, which is pyrox-lgm's unconstrained u (measured <= 0.61, the flat
    # AR(1) noise precision; RW2's and BYM2's precisions within 0.05).
    spec = model.theta_spec()
    sd_internal = np.sqrt(np.diag(np.asarray(ref["theta_cov"])))
    for r_name, (key, idx) in hyper.items():
        inverse = spec[key][1].inv
        ours_u = np.asarray(inverse(jnp.asarray(res.hyperpar[key].q50)))
        if idx is None:
            r_u = float(inverse(jnp.asarray(ref["hyperpar"][r_name]["q50"])))
        else:
            pair = jnp.asarray(
                [ref["hyperpar"][n]["q50"] for n, (k, _) in hyper.items() if k == key]
            )
            r_u = float(np.asarray(inverse(pair))[idx])
            ours_u = ours_u[idx]
        k = _internal_index(ref["theta_mode"], r_name)
        assert abs(float(ours_u) - r_u) < 0.75 * sd_internal[k], r_name

    # Log marginal likelihood, after the intrinsic-normaliser convention:
    # within 1.0 of R-INLA's integration estimate (measured <= 0.80; R-INLA's
    # own "integration" and "Gaussian" estimates differ by up to 1.3 here).
    ours_ml = float(res.log_marginal_likelihood) - _intrinsic_offset(name, model)
    assert abs(ours_ml - ref["mlik_integration"]) < 1.0


def _internal_index(theta_mode, r_name):
    """Position of an R hyperparameter in R-INLA's internal theta."""
    what, _, effect = r_name.partition(" for ")
    key = {
        "Precision": "precision",
        "Rho": "rho",
        "Phi": "phi",
        "Range": "range",
        "Stdev": "stdev",
    }[what.split()[0]]
    for k, internal in enumerate(theta_mode):
        name = internal.lower()
        if key in name and effect.lower() in name:
            return k
    raise KeyError(r_name)  # pragma: no cover
