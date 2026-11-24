"""INLAResult.predict_moments against the mixture inla() keeps."""

from __future__ import annotations

import jax
import numpy as np
import pyrox_lgm as lgm
import pytest


N = 12


def _fit(integration):
    rng = np.random.default_rng(0)
    index = np.arange(20) % (N - 1)  # the last node is never observed
    data = {
        "y": rng.poisson(np.exp(1.0 + np.sin(index / 2.0))).astype(float),
        "t": index,
        "z": rng.normal(size=20),
        "offset": 0.1 * rng.normal(size=20),
    }
    model = lgm.LGM(
        (lgm.RW1(N, name="t"),), lgm.FixedEffects(("intercept", "z")), lgm.Poisson()
    )
    return lgm.inla(model, data, integration=integration), data


def _check(res, data):
    # At the training rows: the mixture of the moments inla() keeps there,
    # its variances from the selected inverse and ours from solves.
    w = np.asarray(res.theta_weights)
    m, v = np.asarray(res.predictor_means), np.asarray(res.predictor_variances)
    ref = w @ m
    mean, var = res.predict_moments({k: data[k] for k in ("t", "z", "offset")})
    assert np.allclose(mean, ref, rtol=1e-8)
    assert np.allclose(var, w @ (v + (m - ref) ** 2), rtol=1e-7)


def test_predict_moments_is_the_mixture_at_the_training_rows():
    res, data = _fit("eb")  # one design point: the smoke test of the fast tier
    _check(res, data)
    none = {k: data[k][:0] for k in ("t", "z", "offset")}
    assert [a.shape for a in res.predict_moments(none)] == [(0,), (0,)]


@pytest.mark.slow
def test_predict_moments_over_the_whole_design():
    res, data = _fit("auto")
    assert res.theta_weights.shape[0] > 1
    _check(res, data)
    new = {"t": np.array([N - 1, 0]), "z": np.array([0.5, 2.0])}
    mean, var = (np.asarray(a) for a in res.predict_moments(new))
    n = 4000
    sampled = res.predict(new, jax.random.key(0), n)
    # Five standard errors of the mean and of the sd; the mixture is close
    # enough to a Gaussian for the latter's sqrt(1 / 2n), doubled.
    sd = np.sqrt(var)
    assert np.all(np.abs(sampled.mean - mean) < 5.0 * sd / np.sqrt(n))
    assert np.all(np.abs(sampled.sd - sd) < 10.0 * sd / np.sqrt(2 * n))
