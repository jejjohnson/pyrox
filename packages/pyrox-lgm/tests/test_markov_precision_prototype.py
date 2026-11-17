"""The SDE-to-precision bridge a ``MarkovGP`` component will stand on.

A Matérn-1/2 kernel on a regular grid is exactly a stationary AR(1): with
variance $\\sigma^2$, lengthscale $\\ell$ and step $\\Delta$, the transition is
$\\rho = e^{-\\Delta/\\ell}$ and the marginal precision is $\\tau = 1/\\sigma^2$. So
the precision ``gaussx.ssm_to_naturals`` builds from the discretised kernel must
equal `gaussx.ar1_precision` entry by entry, and ``inla()`` run on it must
reproduce the `AR1` fit and the R-INLA ``ar1`` fixture.

The kernel covers $\rho \in (0, 1)$ only, while `AR1`'s $\rho$ ranges over
$(-1, 1)$: a Matérn-1/2 ``MarkovGP`` is the positively correlated half of AR(1).
"""

from __future__ import annotations

import json
from pathlib import Path

import equinox as eqx
import gaussx as gx
import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pyrox_lgm as lgm
import pytest
from jaxtyping import Array, Float
from numpyro.distributions.transforms import Transform
from pyrox_lgm._components._base import AbstractComponent, default_transform


jax.config.update("jax_enable_x64", True)

FIXTURE = Path(__file__).parent / "fixtures" / "rinla" / "ar1_gaussian.json"


def _markov_precision(sde: gx.SDEKernel, times: Float[Array, " N"]) -> gx.BlockTriDiag:
    """$\\Lambda$ of the discretised SDE on ``times``, ``(N d) x (N d)``.

    ``ssm_to_naturals`` returns $-\\tfrac12\\Lambda$, built from the
    ``(N - 1, d, d)`` transitions and model-error covariances and the
    stationary initial covariance.
    """
    A, Q = sde.discretise_sequence(jnp.diff(times))
    d = sde.state_dim
    _, neg_half_precision = gx.ssm_to_naturals(
        A, Q, jnp.zeros(d), sde.initial_covariance()
    )
    return -2.0 * neg_half_precision


def _ar1_sde(rho, tau, step: float) -> gx.MaternSDE:
    """The Matérn-1/2 kernel whose grid of spacing ``step`` is AR(1) with
    ``(rho, tau)``: $\\sigma^2 = 1/\\tau$ and $\\ell = -\\Delta / \\log\\rho$."""
    return gx.MaternSDE(1.0 / tau, -step / jnp.log(rho), order=0)


class _MarkovAR1(AbstractComponent):
    """Throwaway component: `AR1`'s hyperparameters and priors, its precision
    built through `_markov_precision` instead of `gaussx.ar1_precision`.

    ``inla()``'s mode search starts at $u = 0$, which is $\rho = 0$, and on
    the fixture steps as far as $\rho = -0.31$ before turning back (8 of its
    49 evaluations have $\rho \le 0$); the kernel has no $\rho \le 0$, so
    $\rho$ is clamped at $10^{-6}$ there. The posterior has no mass below
    $\rho = 0.3$, so the clamp moves the search path, not the fit.
    """

    n: int = eqx.field(static=True)
    name: str = eqx.field(static=True)
    tau_prior: dist.Distribution
    rho_prior: dist.Distribution

    def __init__(self, n: int, *, name: str) -> None:
        reference = lgm.AR1(n, name=name)
        self.n = n
        self.name = name
        self.tau_prior = reference.tau_prior
        self.rho_prior = reference.rho_prior

    @property
    def n_nodes(self) -> int:
        return self.n

    def theta_spec(self) -> dict[str, tuple[dist.Distribution, Transform]]:
        return {
            "tau": (self.tau_prior, default_transform(self.tau_prior)),
            "rho": (self.rho_prior, default_transform(self.rho_prior)),
        }

    def prior(self, theta, *, constraint="hard", soft_constraint_scale=1e-3):
        del constraint, soft_constraint_scale  # proper
        rho = jnp.maximum(theta["rho"], 1e-6)
        sde = _ar1_sde(rho, theta["tau"], 1.0)
        precision = _markov_precision(sde, jnp.arange(self.n, dtype=float))
        return gx.GaussianMRF(jnp.zeros(self.n), precision)


@pytest.mark.parametrize("rho", [0.05, 0.5, 0.9, 0.999])
@pytest.mark.parametrize("tau", [0.1, 1.0, 25.0])
@pytest.mark.parametrize("step", [0.3, 1.0])
def test_matern12_precision_is_the_ar1_precision(rho, tau, step):
    n = 30
    precision = _markov_precision(_ar1_sde(rho, tau, step), step * jnp.arange(n))
    assert isinstance(precision, gx.BlockTriDiag)
    assert precision.diagonal.shape == (n, 1, 1)
    assert precision.sub_diagonal.shape == (n - 1, 1, 1)
    ours = np.asarray(precision.as_matrix())
    expected = np.asarray(gx.ar1_precision(n, rho, tau).as_matrix())
    scale = np.max(np.abs(expected))
    # Round-off only: measured at most 1.9e-13 of the largest entry over this
    # grid (at rho = 0.999, where 1 - rho^2 cancels), 1e-15 elsewhere.
    np.testing.assert_allclose(ours, expected, rtol=1e-12, atol=1e-12 * scale)


def test_matern12_marginal_variances_are_the_kernel_variance():
    sde = _ar1_sde(0.8, 4.0, 1.0)
    gmrf = gx.GaussianMRF(jnp.zeros(20), _markov_precision(sde, jnp.arange(20.0)))
    # Round-off only: measured 1.7e-15 relative.
    np.testing.assert_allclose(gmrf.marginal_variances(), 0.25, rtol=1e-12)


@pytest.mark.slow
def test_inla_through_the_bridge_reproduces_ar1_and_r_inla():
    fx = json.loads(FIXTURE.read_text())
    y = np.asarray(fx["data"]["y"], dtype=float)
    data = {"y": y, "t": np.asarray(fx["data"]["t"])}
    fixed, lik = lgm.FixedEffects(("intercept",)), lgm.Gaussian()
    ar1 = lgm.inla(lgm.LGM((lgm.AR1(len(y), name="t"),), fixed, lik), data)
    bridge = lgm.inla(lgm.LGM((_MarkovAR1(len(y), name="t"),), fixed, lik), data)

    # The two fits see the same precision wherever rho > 0, so they agree to
    # the mode search's tolerance: measured 4.5e-7 in theta (the flat noise
    # precision), 6.4e-8 in the latent means, 8.7e-8 relative in the sds,
    # 4.3e-7 relative in the medians and 7e-8 in the log evidence.
    np.testing.assert_allclose(bridge.theta_mode, ar1.theta_mode, atol=1e-5)
    np.testing.assert_allclose(
        bridge.random["t"].mean, ar1.random["t"].mean, rtol=0, atol=1e-6
    )
    np.testing.assert_allclose(bridge.random["t"].sd, ar1.random["t"].sd, rtol=1e-6)
    for key in ("t.tau", "t.rho", "lik.prec"):
        np.testing.assert_allclose(
            bridge.hyperpar[key].q50, ar1.hyperpar[key].q50, rtol=1e-5
        )
    np.testing.assert_allclose(
        float(bridge.log_marginal_likelihood),
        float(ar1.log_marginal_likelihood),
        atol=1e-6,
    )

    # And the bridge meets R-INLA at the tolerances test_rinla_fixtures.py
    # holds AR1 to (its "sla" run is the reference there).
    ref = fx["sla"]
    ((_, r_t),) = ref["random"].items()
    r_mean, r_sd = np.asarray(r_t["mean"]), np.asarray(r_t["sd"])
    diff = np.abs(np.asarray(bridge.random["t"].mean) - r_mean)
    assert np.max(diff) / np.max(np.abs(r_mean)) < 2e-3
    assert np.max(diff / r_sd) < 0.015
    ratio = np.asarray(bridge.random["t"].sd) / r_sd
    assert ratio.min() > 0.99 and ratio.max() < 1.005
    medians = {
        "Precision for t": ("t.tau", 0.01),
        "Rho for t": ("t.rho", 0.01),
        "Precision for the Gaussian observations": ("lik.prec", 0.1),
    }
    for r_name, (key, rel) in medians.items():
        q50 = float(bridge.hyperpar[key].q50)
        assert abs(q50 / ref["hyperpar"][r_name]["q50"] - 1.0) < rel, r_name
