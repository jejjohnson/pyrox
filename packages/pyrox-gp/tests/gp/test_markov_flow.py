"""Tests for the Normalizing Kalman Filter model surface.

Load-bearing facts pinned here: ``warp=None`` reduces *exactly* to a
direct ``gaussx.kalman_filter`` call (the test that catches a mis-wired
warp default); the warped marginal equals both ``gauss_flows``'
``normalizing_kalman_filter`` density and the dense ``(TM, TM)`` joint
change-of-variables; masking agrees with the exact marginal over
observed entries; the NumPyro/factor/EnsembleMAP integration works with
no flow dependencies; and ``predict`` returns the quadrature pushforward
``E[G(z)]`` rather than ``G(E[z])``.

Tests that construct a warp need ``gauss_flows`` (a non-``None`` warp
requires the ``flows`` extra even for flowjax-only bijections) and skip
when it is absent; the rest run on the base install.

The 1e-13 agreements against dense references need float64 to be on
*before* ``gaussx`` is first imported, since its Gaussian log-density
constant is evaluated at import time; the root ``conftest.py`` is what
guarantees that, and setting the flag here would be too late whenever
another module is collected first.
"""

from __future__ import annotations

import sys

import gaussx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
import pytest
from gaussx import LGSSM, MaskedLGSSM
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from pyrox_gp import NormalizingKalmanPrior, normalizing_kalman_factor


T, M = 8, 2


def _base(T: int = T, M: int = M) -> LGSSM:
    """A small, well-conditioned LGSSM with non-diagonal H."""
    A = 0.85 * jnp.eye(M) + 0.05
    H = jnp.eye(M) + 0.1
    Q = 0.15 * jnp.eye(M)
    R = 0.2 * jnp.eye(M)
    return LGSSM(A, H, Q, R, jnp.zeros(M), jnp.eye(M), n_steps=T)


def _exp_warp(m: int = M):
    """A non-affine elementwise warp (requires the flows extra)."""
    pytest.importorskip("gauss_flows")
    from flowjax.bijections import Exp

    return Exp(shape=(m,))


def _dense_joint_moments(base: LGSSM):
    """Dense ``(TM,)`` mean and ``(TM, TM)`` covariance of the base."""
    A, H, Q, R = base.A, base.H, base.Q, base.R
    n_steps, n_channels = base.event_shape
    means, covs = [], []
    m, P = base.m0, base.P0
    for _ in range(n_steps):
        m = A @ m
        P = A @ P @ A.T + Q
        means.append(H @ m)
        covs.append(P)
    mu = jnp.concatenate(means)  # (TM,)
    sigma = jnp.zeros((n_steps * n_channels, n_steps * n_channels))
    for s in range(n_steps):
        for t in range(s, n_steps):
            # Cov(x_s, x_t) = P_s (A^{t-s})^T for t >= s.
            cross = covs[s] @ jnp.linalg.matrix_power(A, t - s).T
            block = H @ cross @ H.T + (R if s == t else 0.0)
            rows = slice(s * n_channels, (s + 1) * n_channels)
            cols = slice(t * n_channels, (t + 1) * n_channels)
            sigma = sigma.at[rows, cols].set(block)
            if t != s:
                sigma = sigma.at[cols, rows].set(block.T)
    return mu, sigma


def _dense_log_marginal(base, y, warp=None, mask=None):
    """Dense joint change-of-variables reference for ``log_marginal``.

    Inverse-warps ``y`` per step, scores the flattened series under the
    dense ``(TM, TM)`` Gaussian joint (restricted to observed entries
    when ``mask`` is given), and adds the per-entry inverse log-dets of
    the observed entries.
    """
    if warp is None:
        z = y
        log_det = jnp.zeros(y.shape)
    else:
        z = jax.vmap(warp.inverse)(y)

        def per_entry_log_det(y_t):
            # Elementwise warp: J is diagonal, so J @ 1 = diag(J).
            _, diag = jax.jvp(warp.inverse, (y_t,), (jnp.ones_like(y_t),))
            return jnp.log(jnp.abs(diag))

        log_det = jax.vmap(per_entry_log_det)(y)  # (T, M)
    keep = (
        jnp.ones(y.size, dtype=bool) if mask is None else jnp.asarray(mask).reshape(-1)
    )
    idx = jnp.nonzero(keep)[0]
    mu, sigma = _dense_joint_moments(base)
    gauss = dist.MultivariateNormal(mu[idx], covariance_matrix=sigma[jnp.ix_(idx, idx)])
    return gauss.log_prob(z.reshape(-1)[idx]) + jnp.sum(log_det.reshape(-1)[idx])


# --- exact reduction ----------------------------------------------------


def test_warp_none_reduces_to_direct_kalman_filter():
    """``warp=None`` must equal a direct ``gaussx.kalman_filter`` call
    exactly — this is the test that catches a mis-wired warp default."""
    base = _base()

    def mean_fn(times):
        return 0.3 * jnp.stack([times, -times], axis=-1)

    prior = NormalizingKalmanPrior(base, mean_fn=mean_fn)
    y = base.sample(jr.key(0)) + mean_fn(jnp.arange(T, dtype=jnp.float64))

    direct = gaussx.kalman_filter(
        base.A,
        base.H,
        base.Q,
        base.R,
        y - mean_fn(jnp.arange(T, dtype=jnp.float64)),
        base.m0,
        base.P0,
    ).log_likelihood
    assert prior.log_marginal(y) == direct


def test_warp_none_log_marginal_matches_dense_joint():
    """Unwarped marginal equals the dense ``(TM, TM)`` joint density."""
    base = _base()
    y = base.sample(jr.key(1))
    got = NormalizingKalmanPrior(base).log_marginal(y)
    want = _dense_log_marginal(base, y)
    assert jnp.abs(got - want) < 1e-13


# --- warped marginal ----------------------------------------------------


def test_log_marginal_matches_gauss_flows_nkf():
    """Warped marginal equals ``gauss_flows.normalizing_kalman_filter``."""
    gauss_flows = pytest.importorskip("gauss_flows")
    base = _base()
    warp = _exp_warp()
    nkf = gauss_flows.normalizing_kalman_filter(
        gauss_flows.NumpyroBase(dist=base), warp
    )
    y = nkf.sample(jr.key(2))
    got = NormalizingKalmanPrior(base, warp=warp).log_marginal(y)
    assert jnp.abs(got - nkf.log_prob(y)) < 1e-13


def test_log_marginal_matches_dense_joint_change_of_variables():
    """Warped marginal equals the dense joint change-of-variables:
    ``KF(G^{-1}(y)) + sum log-det`` is exact, not an approximation."""
    pytest.importorskip("gauss_flows")
    base = _base()
    warp = _exp_warp()
    z = base.sample(jr.key(3))
    y = jax.vmap(warp.transform)(z)
    got = NormalizingKalmanPrior(base, warp=warp).log_marginal(y)
    want = _dense_log_marginal(base, y, warp=warp)
    assert jnp.abs(got - want) < 1e-13


# --- masking ------------------------------------------------------------


def test_masked_log_marginal_matches_dense_marginal_unwarped():
    """Unwarped: a ``(T, M)`` mask gives the exact marginal over the
    observed entries, and masked entries are never read (NaN-safe)."""
    base = _base()
    y = base.sample(jr.key(4))
    mask = jr.bernoulli(jr.key(5), 0.7, (T, M)).at[0, 0].set(True)
    y_nan = jnp.where(mask, y, jnp.nan)
    got = NormalizingKalmanPrior(base).log_marginal(y_nan, mask)
    want = _dense_log_marginal(base, y, mask=mask)
    assert jnp.abs(got - want) < 1e-13


def test_masked_log_marginal_matches_dense_marginal_warped():
    """Warped: mask agrees with the dense marginal over observed entries
    — the log-det is summed over observed channels only."""
    pytest.importorskip("gauss_flows")
    base = _base()
    warp = _exp_warp()
    y = jax.vmap(warp.transform)(base.sample(jr.key(6)))
    mask = jr.bernoulli(jr.key(7), 0.7, (T, M)).at[0, 0].set(True)
    prior = NormalizingKalmanPrior(base, warp=warp)
    y_nan = jnp.where(mask, y, jnp.nan)
    got = prior.log_marginal(y_nan, mask)
    want = _dense_log_marginal(base, y, warp=warp, mask=mask)
    assert jnp.abs(got - want) < 1e-13

    # A MaskedLGSSM base contributes its own mask when none is passed.
    masked_base = MaskedLGSSM(
        base.A, base.H, base.Q, base.R, base.m0, base.P0, T, obs_mask=mask
    )
    via_base = NormalizingKalmanPrior(masked_base, warp=warp).log_marginal(y_nan)
    assert jnp.abs(via_base - got) < 1e-13


# --- NumPyro integration ------------------------------------------------


def _model(y):
    n_steps, n_channels = y.shape
    log_q = numpyro.sample(
        "log_q", dist.Normal(0.0, 1.0).expand([n_channels]).to_event(1)
    )
    log_r = numpyro.sample(
        "log_r", dist.Normal(0.0, 1.0).expand([n_channels]).to_event(1)
    )
    base = LGSSM(
        0.85 * jnp.eye(n_channels),
        jnp.eye(n_channels),
        jnp.diag(jnp.exp(jnp.asarray(log_q))),
        jnp.diag(jnp.exp(jnp.asarray(log_r))),
        jnp.zeros(n_channels),
        jnp.eye(n_channels),
        n_steps=n_steps,
    )
    normalizing_kalman_factor("nkf", NormalizingKalmanPrior(base), y)


def test_numpyro_trace_sites_and_svi_loss_decreases():
    """The example model exposes the expected sample sites plus one
    factor site, and 50 SVI steps decrease the loss."""
    y = _base().sample(jr.key(8))
    trace = numpyro.handlers.trace(numpyro.handlers.seed(_model, jr.key(0))).get_trace(
        y
    )
    assert trace["log_q"]["type"] == "sample"
    assert trace["log_r"]["type"] == "sample"
    # numpyro implements factor as a unit-Uniform sample site carrying
    # the factor value through ``log_density``.
    factor_sites = [
        name
        for name, site in trace.items()
        if site["type"] == "sample" and name not in ("log_q", "log_r")
    ]
    assert factor_sites == ["nkf"]

    svi = SVI(_model, AutoNormal(_model), numpyro.optim.Adam(0.05), Trace_ELBO())
    result = svi.run(jr.key(9), 50, y, progress_bar=False)
    assert jnp.all(jnp.isfinite(result.losses))
    assert result.losses[-10:].mean() < result.losses[:10].mean()


def test_ensemble_map_over_seeds_is_finite():
    """``EnsembleMAP`` with 4 members fits the collapsed marginal and
    returns finite parameters — the ensembling path the issue unlocks."""
    optax = pytest.importorskip("optax")
    from pyrox.inference import EnsembleMAP

    y = _base().sample(jr.key(10))

    def log_joint(params, x_batch, y_batch):
        del x_batch
        base = LGSSM(
            0.85 * jnp.eye(M),
            jnp.eye(M),
            jnp.diag(jnp.exp(params[:M])),
            jnp.diag(jnp.exp(params[M:])),
            jnp.zeros(M),
            jnp.eye(M),
            n_steps=T,
        )
        loglik = NormalizingKalmanPrior(base).log_marginal(y_batch)
        logprior = dist.Normal(0.0, 1.0).log_prob(params).sum()
        return loglik, logprior

    runner = EnsembleMAP(
        log_joint=log_joint,
        init_fn=lambda key: 0.1 * jr.normal(key, (2 * M,)),
        optimizer=optax.adam(5e-2),
        ensemble_size=4,
    )
    result = runner.run(jr.key(11), 50, y, y)
    assert result.params.shape == (4, 2 * M)
    assert jnp.all(jnp.isfinite(result.params))
    assert jnp.all(jnp.isfinite(result.losses))


# --- predict ------------------------------------------------------------


def _rts_reference(base, z):
    """Warped-space observation moments straight from gaussx."""
    state = gaussx.kalman_filter(base.A, base.H, base.Q, base.R, z, base.m0, base.P0)
    m_smooth, P_smooth = gaussx.rts_smoother(state, base.A, base.Q)
    mz = m_smooth @ base.H.T
    vz = jax.vmap(lambda P: jnp.diag(base.H @ P @ base.H.T))(P_smooth) + jnp.diag(
        base.R
    )
    return mz, vz


def test_predict_identity_warp_equals_rts_moments():
    """An identity warp reproduces the RTS smoother moments exactly —
    the quadrature pushforward of the identity is the identity."""
    pytest.importorskip("gauss_flows")
    from flowjax.bijections import Identity

    base = _base()
    y = base.sample(jr.key(12))
    mz, vz = _rts_reference(base, y)

    mean_none, var_none = NormalizingKalmanPrior(base).predict(y)
    assert jnp.max(jnp.abs(mean_none - mz)) < 1e-13
    assert jnp.max(jnp.abs(var_none - vz)) < 1e-13

    prior = NormalizingKalmanPrior(base, warp=Identity(shape=(M,)))
    mean_id, var_id = prior.predict(y)
    assert jnp.max(jnp.abs(mean_id - mz)) < 1e-12
    assert jnp.max(jnp.abs(var_id - vz)) < 1e-12


def test_predict_is_pushforward_mean_not_warp_of_mean():
    """``predict`` returns ``E[G(z)]``, not ``G(E[z])``: the two differ
    for a non-affine warp, and quadrature matches Monte Carlo."""
    pytest.importorskip("gauss_flows")
    base = _base()
    warp = _exp_warp()
    y = jax.vmap(warp.transform)(base.sample(jr.key(13)))
    prior = NormalizingKalmanPrior(base, warp=warp)

    mean, var = prior.predict(y)
    mz, vz = _rts_reference(base, jax.vmap(warp.inverse)(y))
    warp_of_mean = jax.vmap(warp.transform)(mz)

    # E[G(z)] > G(E[z]) strictly for the convex exp warp.
    assert jnp.all(mean > warp_of_mean)

    # Monte-Carlo pushforward of the same warped-space moments.
    n_samples = 400_000
    eps = jr.normal(jr.key(14), (n_samples, T, M))
    samples = jax.vmap(jax.vmap(warp.transform))(mz + jnp.sqrt(vz) * eps)
    mc_mean = samples.mean(axis=0)
    mc_se = samples.std(axis=0) / jnp.sqrt(n_samples)
    assert jnp.all(jnp.abs(mean - mc_mean) < 6.0 * mc_se)
    assert jnp.all(var > 0)


# --- optional dependency boundary --------------------------------------


def test_unwarped_path_works_without_gauss_flows(monkeypatch):
    """With ``gauss_flows`` absent, the unwarped model constructs and
    fits normally; only passing a warp raises with the install hint.

    Runs on the base install too: the warp used for the raise check is a
    plain flowjax ``Affine`` (flowjax is a hard dependency), and
    ``gauss_flows`` is blocked via ``sys.modules`` either way.
    """
    from flowjax.bijections import Affine

    warp = Affine(loc=jnp.zeros(M), scale=jnp.ones(M))
    base = _base()
    y = base.sample(jr.key(15))

    for name in [n for n in sys.modules if n.split(".")[0] == "gauss_flows"]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "gauss_flows", None)

    prior = NormalizingKalmanPrior(base)  # constructs fine
    assert jnp.isfinite(prior.log_marginal(y))
    mean, var = prior.predict(y, n_ahead=2)
    assert mean.shape == (T + 2, M)
    assert jnp.all(jnp.isfinite(mean)) and jnp.all(jnp.isfinite(var))

    with pytest.raises(ImportError, match=r"pyrox-gp\[flows\]"):
        NormalizingKalmanPrior(base, warp=warp)
