"""`MarkovGPSitesPrior` and `ConditionedMarkovGPSites`: padded sites as one prior.

The references are per-site `MarkovGPPrior`s on the same padded rows (and on
each site's own unpadded series); both run the same masked Kalman filter and
RTS smoother under ``jax.vmap`` or one site at a time, so they agree to
``rtol=1e-10`` under x64.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from numpyro.infer.util import log_density
from pyrox_ssm import (
    ConditionedMarkovGPSites,
    MarkovGPPrior,
    MarkovGPSitesPrior,
    MaternSDE,
    markov_gp_sites_factor,
    pad_sites,
)


jax.config.update("jax_enable_x64", True)

ELLS = jnp.array([0.6, 1.3, 2.0])
NOISES = jnp.array([0.05, 0.1, 0.02])
T_STAR = jnp.array([0.0, 0.7, 2.5, 4.6])


def _ragged():
    times = [
        jnp.linspace(0.0, 4.0, 9),
        jnp.array([0.3, 1.0, 2.2, 2.9]),
        jnp.array([1.5]),
    ]
    ys = [jnp.sin(t) + 0.2 * t for t in times]
    return times, ys


def _kernel():
    return MaternSDE(0.7, ELLS, order=1)


def _site_prior(s, t_s):
    return MarkovGPPrior(MaternSDE(0.7, ELLS[s], order=1), t_s)


@pytest.mark.parametrize("per_site_noise", [False, True], ids=["shared", "per-site"])
def test_log_marginal_is_the_sum_of_site_log_marginals_and_the_factor(
    per_site_noise,
):
    times, ys = _ragged()
    t, y, mask = pad_sites(times, ys)
    noise = NOISES if per_site_noise else jnp.asarray(0.05)
    prior = MarkovGPSitesPrior(_kernel(), t, mask=mask)
    got = prior.log_marginal(y, noise)
    noises = NOISES if per_site_noise else jnp.full(3, 0.05)
    expected = sum(
        _site_prior(s, times[s]).log_marginal(ys[s], noises[s]) for s in range(3)
    )
    assert jnp.allclose(got, expected, rtol=1e-10)

    def model():
        markov_gp_sites_factor("y", _kernel(), t, y, noise, mask=mask)

    assert jnp.allclose(log_density(model, (), {}, {})[0], got, rtol=1e-12)


@pytest.mark.parametrize("include_noise", [False, True], ids=["latent", "noisy"])
def test_condition_predict_equals_each_site_markov_gp(include_noise):
    times, ys = _ragged()
    t, y, mask = pad_sites(times, ys)
    cond = MarkovGPSitesPrior(_kernel(), t, mask=mask).condition(y, NOISES)
    assert isinstance(cond, ConditionedMarkovGPSites)
    mean, var = cond.predict(T_STAR, include_noise=include_noise)
    assert mean.shape == var.shape == (3, T_STAR.shape[0])
    for s in range(3):
        # The site on its own unpadded series: padding adds predict-only steps.
        ref = _site_prior(s, times[s]).condition(ys[s], NOISES[s])
        m_ref, v_ref = ref.predict(T_STAR, include_noise=include_noise)
        assert jnp.allclose(mean[s], m_ref, rtol=1e-10, atol=1e-12)
        assert jnp.allclose(var[s], v_ref, rtol=1e-10, atol=1e-12)
    assert jnp.allclose(
        cond.log_marginal,
        MarkovGPSitesPrior(_kernel(), t, mask=mask).log_marginal(y, NOISES),
        rtol=1e-12,
    )


def test_padding_further_changes_neither_the_log_marginal_nor_the_predictions():
    times, ys = _ragged()
    t, y, mask = pad_sites(times, ys)
    t_long, y_long, mask_long = pad_sites(times, ys, length=t.shape[1] + 5)
    short = MarkovGPSitesPrior(_kernel(), t, mask=mask)
    long = MarkovGPSitesPrior(_kernel(), t_long, mask=mask_long)
    assert jnp.allclose(
        short.log_marginal(y, NOISES), long.log_marginal(y_long, NOISES), rtol=1e-10
    )
    # Predict at each site's observed times, one row per site.
    t_obs = jnp.stack([times[0][:4], times[1], jnp.full(4, times[2][0])])
    m_s, v_s = short.condition(y, NOISES).predict(t_obs)
    m_l, v_l = long.condition(y_long, NOISES).predict(t_obs)
    assert jnp.allclose(m_s, m_l, rtol=1e-10, atol=1e-12)
    assert jnp.allclose(v_s, v_l, rtol=1e-10, atol=1e-12)


def test_predict_takes_one_row_of_test_times_per_site():
    times, ys = _ragged()
    t, y, mask = pad_sites(times, ys)
    cond = MarkovGPSitesPrior(_kernel(), t, mask=mask).condition(y, NOISES)
    rows = jnp.stack([T_STAR, T_STAR + 1.0, T_STAR - 0.5])
    mean, _ = cond.predict(rows)
    for s in range(3):
        m_one, _ = cond.predict(rows[s])
        assert jnp.allclose(mean[s], m_one[s], rtol=1e-12, atol=1e-14)


def test_an_unmasked_prior_with_a_mean_function():
    t = jnp.stack([jnp.linspace(0.0, 3.0, 6), jnp.linspace(0.5, 3.5, 6)])
    y = jnp.sin(t)
    mean_fn = lambda tt: 0.3 * tt
    prior = MarkovGPSitesPrior(MaternSDE(1.0, 1.0, order=1), t, mean_fn=mean_fn)
    mean, _ = prior.condition(y, jnp.asarray(0.1)).predict(T_STAR)
    for s in range(2):
        ref = MarkovGPPrior(MaternSDE(1.0, 1.0, order=1), t[s], mean_fn=mean_fn)
        m_ref, _ = ref.condition(y[s], jnp.asarray(0.1)).predict(T_STAR)
        assert jnp.allclose(mean[s], m_ref, rtol=1e-10, atol=1e-12)


class _Linear(eqx.Module):
    """``a + b t``, with scalar or per-site ``(S,)`` coefficients."""

    a: jax.Array
    b: jax.Array

    def __call__(self, t):
        return self.a + self.b * t


@pytest.mark.parametrize("per_site", [False, True], ids=["shared", "per-site"])
def test_a_mean_module_is_shared_or_one_per_site(per_site):
    times, ys = _ragged()
    t, y, mask = pad_sites(times, ys)
    a = jnp.array([0.5, -1.0, 2.0]) if per_site else jnp.asarray(0.5)
    b = jnp.array([0.1, 0.0, -0.3]) if per_site else jnp.asarray(0.1)
    prior = MarkovGPSitesPrior(_kernel(), t, mask=mask, mean_fn=_Linear(a, b))
    mean, _ = prior.condition(y, NOISES).predict(T_STAR)
    total = 0.0
    for s in range(3):
        a_s, b_s = (a[s], b[s]) if per_site else (a, b)
        ref = MarkovGPPrior(
            MaternSDE(0.7, ELLS[s], order=1), times[s], mean_fn=_Linear(a_s, b_s)
        )
        m_ref, _ = ref.condition(ys[s], NOISES[s]).predict(T_STAR)
        assert jnp.allclose(mean[s], m_ref, rtol=1e-10, atol=1e-12)
        total = total + ref.log_marginal(ys[s], NOISES[s])
    assert jnp.allclose(prior.log_marginal(y, NOISES), total, rtol=1e-10)


def test_shapes_are_checked():
    times, ys = _ragged()
    t, y, mask = pad_sites(times, ys)
    with pytest.raises(ValueError, match=r"times must be \(S, N\)"):
        MarkovGPSitesPrior(_kernel(), t[0])
    with pytest.raises(ValueError, match="mask has shape"):
        MarkovGPSitesPrior(_kernel(), t, mask=mask[:, :-1])
    with pytest.raises(ValueError, match="leading axis of length S"):
        MarkovGPSitesPrior(MaternSDE(0.7, ELLS[:2], order=1), t, mask=mask)
    with pytest.raises(ValueError, match="leading axis of length S"):
        MarkovGPSitesPrior(
            _kernel(), t, mask=mask, mean_fn=_Linear(jnp.zeros(2), jnp.zeros(2))
        )
    prior = MarkovGPSitesPrior(_kernel(), t, mask=mask)
    with pytest.raises(ValueError, match="y has shape"):
        prior.log_marginal(y[:, :-1], NOISES)
    with pytest.raises(ValueError, match="noise_var"):
        prior.condition(y, NOISES[:2])
    with pytest.raises(ValueError, match="t_star must be"):
        prior.condition(y, NOISES).predict(jnp.ones((2, 3)))
