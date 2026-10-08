"""inla() against golden R-INLA fixtures (jejjohnson/pyrox#254).

``fixtures/rinla/make_fixtures.R`` simulates (or loads) each case, fits it in
R-INLA with pyrox-lgm's default priors and writes data and summaries to
JSON; these tests refit the same data with ``lgm.inla`` and compare. R-INLA
runs ``strategy="gaussian"`` with its VB mean correction, which is
pyrox-lgm's default ``"vb"``, and ``int.strategy="auto"``, which is
pyrox-lgm's default ``integration="auto"``.
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


def _convention_offset(name, model):
    """Our log marginal likelihood minus R-INLA's, from normalising conventions.

    - Intrinsic priors: pyrox-lgm keeps the exact normalising constant of
      the density on the constrained space, ``0.5 log|S|* - 0.5 rank
      log(2 pi)`` for a structure ``S``; R-INLA drops ``0.5 log|S|*`` for
      ``rw2`` and both terms for ``bym2``'s ICAR part.
    - ``pc.cor0``: R-INLA's C prior (``priorfunc_pc_cor0``) is the density
      of ``|rho|``, which integrates to 2 over ``(-1, 1)``; pyrox-lgm's
      `PCAR1Rho` integrates to 1.
    - BYM2's ``phi``: R-INLA tabulates the PC prior on ``logit(phi) <= 12``
      and renormalises it there (``inla.pc.bym.phi``); pyrox-lgm's
      `PCBYM2Phi` is the exact prior, which puts about 12 % of its mass
      beyond (``d(phi)`` grows like ``sqrt(-log(1 - phi))``).

    At fixed hyperparameters the likelihood parts agree to 1e-5 (AR(1)) and
    0.01 (BYM2) once these are removed.
    """
    if name == "ar1_gaussian":
        return -np.log(2.0)
    if name == "spde_poisson":
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
        phi_prior = model.theta_spec()["region.phi"][0]
        in_table = float(phi_prior.cdf(jnp.asarray(1.0 / (1.0 + np.exp(-12.0)))))
        return half_logpdet - 0.5 * ev.size * np.log(2 * np.pi) + np.log(in_table)
    return half_logpdet


def _gaussian_log_ml(model, data, u):
    """The Gaussian approximation over theta at ``u``, as R-INLA's ``mlik``
    second row: ``log pi(u | y) + m/2 log 2 pi - 1/2 log|-Hessian|``."""
    from pyrox_lgm._inla import _hessian, _log_post

    A = model.projector(data)
    d = {k: jnp.asarray(v) for k, v in data.items()}
    neg_h = -np.asarray(_hessian(model, A, d, 50, u))
    lp = float(_log_post(model, A, d, 50, u))
    m = u.shape[0]
    return lp + 0.5 * m * np.log(2 * np.pi) - 0.5 * np.linalg.slogdet(neg_h)[1]


CASES = [
    "rw2_gaussian",
    "ar1_gaussian",
    "scotland_bym2",
    "spde_poisson",
    "pod_bernoulli",
]

# Per case, measured against R-INLA 26.8.7 and bounded with a margin (every
# interval contains 1, so an exact match always passes):
#   random_rel: max |latent mean difference| / max |R-INLA's latent mean|;
#   random_z: the same difference in R-INLA's posterior sds;
#   random_sd / fixed_sd: the range of sd ratios (ours / R-INLA's);
#   median_rel: max relative difference of the hyperparameter medians.
# `integration="auto"` is R-INLA's own design (its fixed grids, the CCD, the
# skewness corrections, early stop and pruning), the VB correction moves
# R-INLA's nodes and the summaries are of the mixture on its mean +- 5 sds,
# as R-INLA's combined marginals are; with those, the cases agree to
# the precision below. AR(1)'s noise precision is the flattest direction
# (posterior sd 1.9 in log): its median is 8 % off, 0.04 sd, from R-INLA's
# finite-difference Hessian and skewness corrections.
TOLERANCES = {
    "rw2_gaussian": {
        "random_rel": 1e-4,
        "random_z": 0.002,
        "random_sd": (0.998, 1.002),
        "fixed_sd": (0.998, 1.002),
        "median_rel": 0.005,
    },  # measured 1.4e-5, 0.0002, [0.9997, 0.9997], 0.9997, 0.20 %
    "ar1_gaussian": {
        "random_rel": 2e-3,
        "random_z": 0.015,
        "random_sd": (0.99, 1.005),
        "fixed_sd": (0.99, 1.005),
        "median_rel": {"Precision for the Gaussian observations": 0.1, None: 0.01},
    },  # measured 1.0e-3, 0.0068, [0.997, 0.997], 0.999, 7.9 % (lik.prec), 0.35 %
    "scotland_bym2": {
        "random_rel": 1e-3,
        "random_z": 0.006,
        "random_sd": (0.995, 1.003),
        "fixed_sd": (0.995, 1.003),
        "median_rel": 0.005,
    },  # measured 4.9e-4, 0.0028, [0.9975, 1.0005], 0.9995, 0.07 %
    "spde_poisson": {
        "random_rel": 2e-3,
        "random_z": 0.003,
        "random_sd": (0.99, 1.003),
        "fixed_sd": (0.985, 1.003),
        "median_rel": 0.01,
    },  # measured 8.9e-4, 0.0012, [0.994, 0.998], 0.993, 0.50 %
    "pod_bernoulli": {
        "random_rel": 1e-3,
        "random_z": 0.005,
        "random_sd": (0.98, 1.003),
        "fixed_sd": (0.995, 1.003),
        "median_rel": 0.005,
    },  # measured 2.8e-4, 0.0024, [0.987, 1.000], 1.000, 0.11 %
}

# R-INLA's two fits of the AR(1) case (``"vb"`` and ``"sla"``, identical for
# a Gaussian likelihood) stopped its mode search at different points in the
# flat noise-precision direction (log 2.83 and 3.01, posterior sd 1.9);
# ours converges to 3.007, the "sla" fit's, so that fit is the reference.
REFERENCE_RUN = {"ar1_gaussian": "sla"}


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
    # One R name per (key, index), and a vector key's indices cover it.
    assert len(set(hyper.values())) == len(hyper)
    for key, shape in shapes.items():
        if shape:
            idxs = sorted(i for k, i in hyper.values() if k == key)
            assert idxs == list(range(shape[0])), key


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
    ref, tol = fx[REFERENCE_RUN.get(name, "vb")], TOLERANCES[name]

    # Fixed effects: means within 0.01 sd (measured <= 0.002).
    for r_name, s in ref["fixed"].items():
        ours = res.fixed[_fixed_key(r_name)]
        assert abs(float(ours.mean) - s["mean"]) < 0.01 * s["sd"], r_name
        lo, hi = tol["fixed_sd"]
        assert lo < float(ours.sd) / s["sd"] < hi, r_name

    # The latent field, node by node.
    ((_, s),) = ref["random"].items()
    ours = res.random[model.components[0].name]
    r_mean, sd_r = np.asarray(s["mean"]), np.asarray(s["sd"])
    diff = np.abs(np.asarray(ours.mean) - r_mean)
    assert np.max(diff) / np.max(np.abs(r_mean)) < tol["random_rel"]
    assert np.max(diff / sd_r) < tol["random_z"]
    ratio = np.asarray(ours.sd) / sd_r
    lo, hi = tol["random_sd"]
    assert lo < ratio.min() and ratio.max() < hi

    # Hyperparameter medians: relative on the user scale, and within 0.05
    # posterior sd on R-INLA's internal scale, which is pyrox-lgm's
    # unconstrained u (measured <= 0.042, AR(1)'s noise precision).
    spec = model.theta_spec()
    sd_internal = np.sqrt(np.diag(np.asarray(ref["theta_cov"])))
    for r_name, (key, idx) in hyper.items():
        q50 = np.ravel(np.asarray(res.hyperpar[key].q50))[idx or 0]
        r_q50 = ref["hyperpar"][r_name]["q50"]
        rel = tol["median_rel"]
        rel = rel.get(r_name, rel[None]) if isinstance(rel, dict) else rel
        assert abs(q50 / r_q50 - 1.0) < rel, r_name
        inverse = spec[key][1].inv
        ours_u = np.asarray(inverse(jnp.asarray(res.hyperpar[key].q50)))
        if idx is None:
            r_u = float(inverse(jnp.asarray(r_q50)))
        else:
            pair = jnp.asarray(
                [ref["hyperpar"][n]["q50"] for n, (k, _) in hyper.items() if k == key]
            )
            r_u = float(np.asarray(inverse(pair))[idx])
            ours_u = ours_u[idx]
        k = _internal_index(ref["theta_mode"], r_name)
        assert abs(float(ours_u) - r_u) < 0.05 * sd_internal[k], r_name

    # Log marginal likelihood, after the normalising conventions. The
    # Gaussian approximation over theta (R-INLA's second ``mlik`` row) within
    # 0.1 (measured <= 0.033). R-INLA's integrated estimate (first row) is
    # biased by its design's unnormalised weights (`_rinla_mlik_bias`); with
    # that removed, ours within 0.2 (measured <= 0.106, BYM2 and POD, whose
    # skewed posteriors R-INLA integrates without the Jacobian of its
    # stretched points).
    offset = _convention_offset(name, model)
    gauss = _gaussian_log_ml(model, _case(name)[1], res.theta_mode) - offset
    assert abs(gauss - ref["mlik_gaussian"]) < 0.1
    ours_ml = float(res.log_marginal_likelihood) - offset
    r_ml = ref["mlik_integration"] - _rinla_mlik_bias(model.n_theta)
    assert abs(ours_ml - r_ml) < 0.2


def _rinla_mlik_bias(m):
    """R-INLA's integrated log ML minus the exact one, for a Gaussian posterior.

    R-INLA (``GMRFLib_ai_INLA_experimental``) estimates the evidence as
    ``log(0.75 sum_k w_k pi_k / max_k(w_k pi_k)) + log pi(theta*) -
    1/2 log|H|``: its design weights ``w_k`` are not normalised, the 0.75
    is the old grid step, and the sum is scaled by its largest term rather
    than the mode's. For a Gaussian posterior that is the exact evidence
    plus this constant: +0.39 for one hyperparameter, -0.44 for two, -1.29
    for three. (Its skewness-stretched points without their Jacobian add a
    further bias of order 0.1 under a skewed posterior.)
    """
    from pyrox_lgm._inla import _rinla_design

    x, log_w = _rinla_design(m)
    ld = log_w - 0.5 * np.sum(x**2, axis=1)
    lse = ld.max() + np.log(np.sum(np.exp(ld - ld.max())))
    return lse - ld.max() + np.log(0.75) - 0.5 * m * np.log(2 * np.pi)


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
        # Match the whole effect name: "for t" must not match "for the
        # gaussian observations".
        if key in name and name.endswith(f" for {effect.lower()}"):
            return k
    raise KeyError(r_name)  # pragma: no cover
