"""Tests for `LinearGaussianSSMPrior` and its Markov-GP special case.

The contracts:

* ``MarkovGPPrior.as_lgssm`` is exactly the model ``MarkovGPPrior`` filters
  (its fields are the documented sequences, and the log marginal and
  smoothed moments agree to ``rtol=1e-12``, mean function and mask included);
* the log marginal equals the dense joint Gaussian of the stacked
  observations, for a local-level model and for M = 2 outputs with entries
  masked (the dense reference keeps the observed entries only);
* gradients match finite differences, and every matrix field is
  differentiable under ``jit``, inside NUTS, and with ``NaN`` placeholders at
  masked entries.

The dense references are exact in exact arithmetic; under x64 the Kalman
recursion and the dense Cholesky round differently by ~1e-14 on these
sizes, so they are compared at ``rtol=1e-10``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
import pytest
from jax.scipy.stats import multivariate_normal
from numpyro import handlers
from numpyro.infer import MCMC, NUTS
from pyrox_ssm import (
    ConditionedLinearGaussianSSM,
    LinearGaussianSSMPrior,
    MarkovGPPrior,
    MaternSDE,
    lgssm_factor,
)


jax.config.update("jax_enable_x64", True)

T = 12
NOISE = jnp.asarray(0.05)


# --- helpers -------------------------------------------------------------


def _local_level(q, r, *, T: int = T, m0=0.3, p0=2.0) -> LinearGaussianSSMPrior:
    """``x_k = x_{k-1} + eta_k``, ``y_k = x_k + eps_k``, scalar ``q`` and ``r``."""
    ones = jnp.ones((T, 1, 1))
    return LinearGaussianSSMPrior(
        transition=ones,
        obs_model=jnp.eye(1),
        process_noise=q * ones,
        obs_noise=r * jnp.eye(1),
        init_mean=jnp.full((1,), m0),
        init_cov=p0 * jnp.eye(1),
    )


def _local_level_dense(q, r, *, T: int = T, m0=0.3, p0=2.0):
    """Mean and covariance of ``y_{1:T}`` under the local-level model.

    ``x_k = x_0 + eta_1 + ... + eta_k`` for ``k = 1..T`` (the filter predicts
    before the first update), so ``Cov(y_i, y_j) = p0 + q min(i, j) + r [i = j]``.
    """
    k = jnp.arange(1, T + 1, dtype=jnp.float64)
    cov = p0 + q * jnp.minimum(k[:, None], k[None, :]) + r * jnp.eye(T)
    return jnp.full((T,), m0), cov


def _two_output_model(key) -> LinearGaussianSSMPrior:
    """D = 2, M = 2, time-varying transition and observation matrices."""
    k1, k2, k3 = jr.split(key, 3)
    base = jnp.array([[0.9, 0.2], [-0.1, 0.8]])
    A = base + 0.05 * jr.normal(k1, (T, 2, 2))
    H = jnp.array([[1.0, 0.5], [-0.3, 1.0]]) + 0.1 * jr.normal(k2, (T, 2, 2))
    Q = jnp.broadcast_to(jnp.array([[0.2, 0.05], [0.05, 0.1]]), (T, 2, 2))
    R = jnp.array([[0.3, 0.1], [0.1, 0.4]])
    offset = 0.2 * jr.normal(k3, (T, 2))
    return LinearGaussianSSMPrior(
        A,
        H,
        Q,
        R,
        jnp.array([0.5, -0.2]),
        jnp.array([[1.0, 0.3], [0.3, 0.7]]),
        obs_offset=offset,
    )


def _dense_moments(prior: LinearGaussianSSMPrior):
    """Mean ``(T M,)`` and covariance ``(T M, T M)`` of the stacked ``y_{1:T}``.

    Built by unrolling ``x_k = A_k x_{k-1} + eta_k`` for ``k = 1..T``: with
    ``Phi(i, j) = A_i ... A_{j+1}``, ``Cov(x_i, x_j) = Phi(i, j) P_j`` for
    ``i >= j``, and ``Cov(y_i, y_j) = H_i Cov(x_i, x_j) H_j^T + R [i = j]``.
    """
    A, Q = prior.transition, prior.process_noise
    n, M = prior.n_steps, prior.obs_dim
    H = jnp.broadcast_to(prior.obs_model, (n, M, prior.state_dim))
    R = jnp.broadcast_to(prior.obs_noise, (n, M, M))
    b = jnp.zeros((n, M)) if prior.obs_offset is None else prior.obs_offset
    means, covs = [], []
    m, P = prior.init_mean, prior.init_cov
    for k in range(n):
        m = A[k] @ m
        P = A[k] @ P @ A[k].T + Q[k]
        means.append(m)
        covs.append(P)
    rows = []
    for i in range(n):
        row = []
        for j in range(n):
            if i >= j:
                phi = jnp.eye(prior.state_dim)
                for k in range(j + 1, i + 1):
                    phi = A[k] @ phi
                C = phi @ covs[j]
            else:
                phi = jnp.eye(prior.state_dim)
                for k in range(i + 1, j + 1):
                    phi = A[k] @ phi
                C = (phi @ covs[i]).T
            block = H[i] @ C @ H[j].T + (R[i] if i == j else 0.0)
            row.append(block)
        rows.append(jnp.concatenate(row, axis=1))
    mean = jnp.concatenate([H[k] @ means[k] + b[k] for k in range(n)])
    return mean, jnp.concatenate(rows, axis=0)


def _markov_prior(*, with_mean: bool) -> MarkovGPPrior:
    times = jnp.sort(jr.uniform(jr.key(0), (T,), maxval=5.0))
    return MarkovGPPrior(
        MaternSDE(1.3, 0.6, order=2),
        times,
        mean_fn=(lambda t: 0.4 * t - 0.1) if with_mean else None,
        obs_noise_floor=1e-6,
    )


# --- MarkovGPPrior.as_lgssm ------------------------------------------------


def test_as_lgssm_fields_are_the_documented_sequences() -> None:
    prior = _markov_prior(with_mean=True)
    lgssm = prior.as_lgssm(NOISE)
    d = prior.state_dim
    assert lgssm.transition.shape == (T, d, d)
    assert lgssm.obs_model.shape == (1, d)
    # Leading zero step: A_1 = I and Q_1 = 0, so y_1 observes x_0 ~ N(0, P_inf).
    assert jnp.allclose(lgssm.transition[0], jnp.eye(d), rtol=0, atol=1e-14)
    assert jnp.allclose(lgssm.process_noise[0], 0.0, rtol=0, atol=1e-14)
    assert jnp.array_equal(lgssm.init_mean, jnp.zeros(d))
    assert jnp.array_equal(lgssm.init_cov, prior.initial_covariance())
    assert jnp.array_equal(lgssm.obs_noise[:, 0, 0], jnp.full(T, NOISE + 1e-6))
    assert lgssm.obs_offset is not None
    assert jnp.array_equal(lgssm.obs_offset[:, 0], prior.mean(prior.times))


@pytest.mark.parametrize("with_mean", [False, True], ids=["zero-mean", "mean-fn"])
@pytest.mark.parametrize("masked", [False, True], ids=["unmasked", "masked"])
def test_as_lgssm_matches_markov_gp(with_mean: bool, masked: bool) -> None:
    prior = _markov_prior(with_mean=with_mean)
    times = prior.times
    y = jnp.sin(2.0 * times) + 0.3 * times
    mask = (jnp.arange(T) % 3 != 1) if masked else None
    if masked:
        y = jnp.where(mask, y, jnp.nan)
    lgssm = prior.as_lgssm(NOISE)

    # Same model, same arithmetic: rtol 1e-12 is the brief's tolerance
    # (the routing makes them bit-identical in practice).
    lm_markov = prior.log_marginal(y, NOISE, mask=mask)
    assert jnp.allclose(lgssm.log_marginal(y, mask=mask), lm_markov, rtol=1e-12)
    m_ref, P_ref, lm_ref = prior.smooth(y, NOISE, mask=mask)
    m, P, lm = lgssm.smooth(y[:, None], mask=mask)
    assert jnp.allclose(m, m_ref, rtol=1e-12, atol=1e-14)
    assert jnp.allclose(P, P_ref, rtol=1e-12, atol=1e-14)
    assert jnp.allclose(lm, lm_ref, rtol=1e-12)
    assert jnp.isfinite(lm)


# --- dense references ----------------------------------------------------


def test_local_level_log_marginal_matches_dense_reference() -> None:
    q, r = 0.15, 0.4
    y = jnp.cumsum(jr.normal(jr.key(1), (T,))) * 0.4
    mean, cov = _local_level_dense(q, r)
    expected = multivariate_normal.logpdf(y, mean, cov)
    got = _local_level(q, r).log_marginal(y[:, None])
    assert jnp.allclose(got, expected, rtol=1e-10)


def test_local_level_accepts_a_flat_series_when_m_is_one() -> None:
    prior = _local_level(0.15, 0.4)
    y = jr.normal(jr.key(2), (T,))
    assert jnp.array_equal(prior.log_marginal(y), prior.log_marginal(y[:, None]))


@pytest.mark.parametrize("per_entry", [True, False], ids=["T-M-mask", "T-mask"])
def test_two_outputs_with_masked_entries_match_dense_reference(
    per_entry: bool,
) -> None:
    prior = _two_output_model(jr.key(3))
    y = jr.normal(jr.key(4), (T, 2))
    if per_entry:
        # Output 1 is missing at three steps; output 0 at one other.
        mask = jnp.ones((T, 2), dtype=bool)
        mask = mask.at[jnp.array([2, 5, 9]), 1].set(False).at[7, 0].set(False)
        keep = mask.reshape(-1)
    else:
        mask = jnp.arange(T) % 4 != 2
        keep = jnp.repeat(mask, 2)
    mean, cov = _dense_moments(prior)
    idx = jnp.flatnonzero(keep)
    expected = multivariate_normal.logpdf(
        y.reshape(-1)[idx], mean[idx], cov[jnp.ix_(idx, idx)]
    )
    got = prior.log_marginal(
        jnp.where(mask if per_entry else mask[:, None], y, jnp.nan), mask=mask
    )
    assert jnp.allclose(got, expected, rtol=1e-10)


def test_two_outputs_unmasked_match_dense_reference() -> None:
    prior = _two_output_model(jr.key(5))
    y = jr.normal(jr.key(6), (T, 2))
    mean, cov = _dense_moments(prior)
    expected = multivariate_normal.logpdf(y.reshape(-1), mean, cov)
    assert jnp.allclose(prior.log_marginal(y), expected, rtol=1e-10)


# --- gradients, jit, NUTS --------------------------------------------------


def test_local_level_gradient_in_q_matches_finite_differences() -> None:
    y = jr.normal(jr.key(7), (T, 1))

    def f(q):
        return _local_level(q, 0.4).log_marginal(y)

    q0, h = 0.2, 1e-6
    fd = (f(q0 + h) - f(q0 - h)) / (2.0 * h)
    # Central differences have O(h^2) truncation and O(eps / h) round-off
    # error, ~1e-10 relative here.
    assert jnp.allclose(jax.grad(f)(q0), fd, rtol=1e-7)


def test_jit_and_grad_with_respect_to_every_matrix_field() -> None:
    prior = _two_output_model(jr.key(8))
    y = jr.normal(jr.key(9), (T, 2))
    mask = jnp.ones((T, 2), dtype=bool).at[3, 1].set(False)

    def lm(p):
        return p.log_marginal(y, mask=mask)

    assert jnp.allclose(jax.jit(lm)(prior), lm(prior), rtol=1e-12)
    grads = jax.jit(jax.grad(lm))(prior)
    for name in (
        "transition",
        "obs_model",
        "process_noise",
        "obs_noise",
        "init_mean",
        "init_cov",
        "obs_offset",
    ):
        g = getattr(grads, name)
        assert g.shape == getattr(prior, name).shape, name
        assert jnp.all(jnp.isfinite(g)), name
        assert jnp.any(g != 0.0), name


def test_nan_at_masked_entries_keeps_value_and_gradient_finite() -> None:
    prior = _two_output_model(jr.key(10))
    y = jr.normal(jr.key(11), (T, 2))
    mask = jnp.ones((T, 2), dtype=bool).at[jnp.array([1, 4]), 0].set(False)
    y_nan = jnp.where(mask, y, jnp.nan)
    y_zero = jnp.where(mask, y, 0.0)

    value = prior.log_marginal(y_nan, mask=mask)
    assert jnp.isfinite(value)
    assert jnp.array_equal(value, prior.log_marginal(y_zero, mask=mask))
    grads = jax.grad(lambda p, yy: p.log_marginal(yy, mask=mask), argnums=(0, 1))(
        prior, y_nan
    )
    for leaf in jax.tree.leaves(grads):
        assert jnp.all(jnp.isfinite(leaf))
    # The masked entries of y carry no information, so no gradient either.
    assert jnp.array_equal(grads[1][~mask], jnp.zeros(2))


def _local_level_model(y, mask=None):
    q = numpyro.sample("q", dist.LogNormal(-2.0, 1.0))
    r = numpyro.sample("r", dist.LogNormal(-1.0, 1.0))
    lgssm_factor("y", _local_level(q, r, T=y.shape[0]), y, mask=mask)


def test_lgssm_factor_registers_the_log_marginal() -> None:
    y = jr.normal(jr.key(12), (T, 1))
    mask = jnp.arange(T) != 4
    trace = handlers.trace(handlers.seed(_local_level_model, 0)).get_trace(y, mask)
    assert set(trace) == {"q", "r", "y"}
    q, r = trace["q"]["value"], trace["r"]["value"]
    expected = _local_level(q, r).log_marginal(y, mask=mask)
    assert jnp.array_equal(trace["y"]["fn"].log_factor, expected)


def test_lgssm_factor_runs_under_nuts_smoke() -> None:
    y = jnp.cumsum(jr.normal(jr.key(13), (T, 1)), axis=0) * 0.3
    mask = jnp.arange(T) != 6
    mcmc = MCMC(
        NUTS(_local_level_model), num_warmup=5, num_samples=5, progress_bar=False
    )
    mcmc.run(jr.key(0), jnp.where(mask[:, None], y, jnp.nan), mask)
    samples = mcmc.get_samples()
    assert samples["q"].shape == (5,) and samples["r"].shape == (5,)
    assert jnp.all(jnp.isfinite(samples["q"])) and jnp.all(jnp.isfinite(samples["r"]))


# --- construction and conditioning ---------------------------------------


def test_condition_holds_the_smoothed_posterior() -> None:
    prior = _two_output_model(jr.key(14))
    y = jr.normal(jr.key(15), (T, 2))
    mask = jnp.arange(T) != 3
    cond = prior.condition(jnp.where(mask[:, None], y, jnp.nan), mask=mask)
    assert isinstance(cond, ConditionedLinearGaussianSSM)
    m, P, lm = prior.smooth(y, mask=mask)
    assert jnp.array_equal(cond.smoothed_means, m)
    assert jnp.array_equal(cond.smoothed_covs, P)
    assert jnp.array_equal(cond.log_marginal, lm)
    assert jnp.array_equal(cond.y[3], jnp.zeros(2))
    assert cond.mask is not None and cond.mask.shape == (T,)


def test_integer_matrices_are_promoted_to_float() -> None:
    ones = jnp.ones((3, 1, 1), dtype=jnp.int32)
    prior = LinearGaussianSSMPrior(
        ones,
        jnp.eye(1, dtype=jnp.int32),
        ones,
        jnp.eye(1, dtype=jnp.int32),
        jnp.zeros(1, dtype=jnp.int32),
        jnp.eye(1, dtype=jnp.int32),
    )
    assert jnp.issubdtype(prior.transition.dtype, jnp.floating)
    assert jnp.isfinite(prior.log_marginal(jnp.array([1, 2, 3])))


def test_float32_inputs_stay_float32() -> None:
    ones = jnp.ones((3, 1, 1), dtype=jnp.float32)
    eye = jnp.eye(1, dtype=jnp.float32)
    prior = LinearGaussianSSMPrior(
        ones, eye, 0.1 * ones, eye, jnp.zeros(1, dtype=jnp.float32), eye
    )
    assert prior.transition.dtype == jnp.float32
    y = jnp.array([0.1, 0.2, 0.3], dtype=jnp.float32)
    assert prior.log_marginal(y).dtype == jnp.float32


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("transition", jnp.ones((4, 2, 3)), "transition must be a"),
        ("obs_model", jnp.ones((2, 3)), "obs_model must have shape"),
        ("obs_model", jnp.ones((5, 2, 2)), "obs_model must have shape"),
        ("process_noise", jnp.ones((3, 2, 2)), "process_noise must have shape"),
        ("obs_noise", jnp.ones((3, 3)), "obs_noise must have shape"),
        ("init_mean", jnp.ones(3), "init_mean must have shape"),
        ("init_cov", jnp.ones((2, 3)), "init_cov must have shape"),
        ("obs_offset", jnp.ones((4, 3)), "obs_offset must have shape"),
    ],
)
def test_bad_shapes_raise_at_construction(field, value, match) -> None:
    kwargs = {
        "transition": jnp.ones((4, 2, 2)),
        "obs_model": jnp.ones((2, 2)),
        "process_noise": jnp.ones((4, 2, 2)),
        "obs_noise": jnp.eye(2),
        "init_mean": jnp.zeros(2),
        "init_cov": jnp.eye(2),
        "obs_offset": None,
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match=match):
        LinearGaussianSSMPrior(**kwargs)


def test_bad_observation_and_mask_shapes_raise() -> None:
    prior = _two_output_model(jr.key(16))
    with pytest.raises(ValueError, match=r"y must have shape \(12, 2\)"):
        prior.log_marginal(jnp.ones(T))
    with pytest.raises(ValueError, match="mask has shape"):
        prior.log_marginal(jnp.ones((T, 2)), mask=jnp.ones((T, 3), dtype=bool))
