"""Tests for `pyrox.inference.last_layer_laplace` (#322).

The reference is closed-form Bayesian linear regression (BLR) on frozen
features: for a last layer linear in its weights the GGN Laplace posterior
is exact, so mean, variance and evidence must match BLR to float64
round-off. The suite runs with x64 on (root ``conftest.py``); ``1e-9``
relative tolerances are many orders above the round-off of the
``(P, P) = (9, 9)`` solves involved and far below any modelling error.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from pyrox.inference import LastLayerLaplace, last_layer_laplace


SIGMA = 0.3
LAM = 2.0
N, D, H = 40, 2, 8


def _predict(params, x):
    feats = jnp.tanh(x @ params["features"])
    return feats @ params["output"]["W"] + params["output"]["b"]


def _is_output(path, _leaf):
    return "output" in jax.tree_util.keystr(path)


def _phi(params, x):
    """The frozen design matrix ``[tanh(x W1), 1]``."""
    feats = jnp.tanh(x @ params["features"])
    return jnp.concatenate([feats, jnp.ones((x.shape[0], 1))], axis=1)


@pytest.fixture
def blr():
    """Frozen random features, BLR posterior and its closed forms."""
    k1, k2, k3, k4 = jr.split(jr.key(0), 4)
    x = jr.normal(k1, (N, D))
    x_new = jr.normal(k4, (7, D))
    params = {
        "features": jr.normal(k2, (D, H)),
        "output": {"W": jnp.zeros(H), "b": jnp.zeros(())},
    }
    phi = _phi(params, x)
    y = jnp.sin(x[:, 0]) + SIGMA * jr.normal(k3, (N,))
    # Closed-form BLR (dense jnp.linalg is fine in tests, per AGENTS.md).
    a = phi.T @ phi / SIGMA**2 + LAM * jnp.eye(H + 1)
    cov = jnp.linalg.inv(a)
    m = cov @ phi.T @ y / SIGMA**2
    params = {
        "features": params["features"],
        "output": {"W": m[:H], "b": m[H]},
    }
    phi_new = _phi(params, x_new)
    marg_cov = SIGMA**2 * jnp.eye(N) + phi @ phi.T / LAM
    evidence = jax.scipy.stats.multivariate_normal.logpdf(y, jnp.zeros(N), marg_cov)
    return {
        "x": x,
        "y": y,
        "x_new": x_new,
        "params": params,
        "mean": phi_new @ m,
        "var": jnp.einsum("mp,pq,mq->m", phi_new, cov, phi_new),
        "evidence": evidence,
        "a": a,
    }


def _fit(p, **kw):
    kw = {"prior_precision": LAM, "jitter": 0.0, **kw}
    return last_layer_laplace(
        _predict,
        p["params"],
        p["x"],
        p["y"],
        sigma=SIGMA,
        param_filter=_is_output,
        **kw,
    )


# -- Closed-form equality ------------------------------------------------------


def test_linear_last_layer_equals_bayesian_linear_regression(blr):
    post = _fit(blr)
    assert isinstance(post, LastLayerLaplace)
    assert post.w_map.shape == (H + 1,)
    mean, var = post.predict(blr["x_new"])
    assert mean.shape == var.shape == (7,)
    assert jnp.allclose(mean, blr["mean"], rtol=1e-9, atol=1e-12)
    assert jnp.allclose(var, blr["var"], rtol=1e-9, atol=1e-12)
    # ravel order is (W, b), the order of _phi's columns.
    assert jnp.allclose(post.precision().as_matrix(), blr["a"], rtol=1e-9)
    assert jnp.allclose(
        post.log_marginal_likelihood(), blr["evidence"], rtol=1e-9, atol=1e-9
    )


def test_output_shape_follows_predict_fn(blr):
    """A ``(N, 1)`` output is flattened for J and reshaped back."""

    def predict_col(params, x):
        return _predict(params, x)[:, None]

    post = last_layer_laplace(
        predict_col,
        blr["params"],
        blr["x"],
        blr["y"][:, None],
        sigma=SIGMA,
        param_filter=_is_output,
        prior_precision=LAM,
        jitter=0.0,
    )
    mean, var = post.predict(blr["x_new"])
    assert mean.shape == var.shape == (7, 1)
    assert jnp.allclose(var[:, 0], blr["var"], rtol=1e-9)


# -- full vs diag ---------------------------------------------------------------


def test_diag_keeps_the_diagonal_of_the_full_ggn(blr):
    full = _fit(blr, structure="full")
    diag = _fit(blr, structure="diag")
    assert full.ggn.shape == (H + 1, H + 1)
    assert diag.ggn.shape == (H + 1,)
    assert jnp.allclose(diag.ggn, jnp.diag(full.ggn), rtol=1e-12)
    # Diagonal precision ⇒ Var = Σ_p J_p² / h_p, with J = φ for a linear layer.
    phi_new = _phi(blr["params"], blr["x_new"])
    _, var = diag.predict(blr["x_new"])
    expected = jnp.sum(phi_new**2 / (diag.ggn + LAM), axis=1)
    assert jnp.allclose(var, expected, rtol=1e-12)
    # Dropping the (positive-definite) off-diagonal coupling changes the answer.
    _, var_full = full.predict(blr["x_new"])
    assert not jnp.allclose(var, var_full, rtol=1e-3)


def test_diag_equals_full_for_orthogonal_features():
    """With orthogonal Jacobian columns ``JᵀJ`` is diagonal: the two agree."""
    q, _ = jnp.linalg.qr(jr.normal(jr.key(1), (30, 4)))
    x = jnp.arange(30)
    y = jr.normal(jr.key(2), (30,))
    params = {"w": jnp.ones(4)}

    def predict(p, idx):
        return q[idx] @ p["w"]

    def keep_all(_path, _leaf):
        return True

    kw = {"sigma": 0.5, "param_filter": keep_all, "prior_precision": 0.7}
    full = last_layer_laplace(predict, params, x, y, structure="full", **kw)
    diag = last_layer_laplace(predict, params, x, y, structure="diag", **kw)
    for a, b in zip(full.predict(x[:9]), diag.predict(x[:9]), strict=True):
        assert jnp.allclose(a, b, rtol=1e-9)
    assert jnp.allclose(
        full.log_marginal_likelihood(), diag.log_marginal_likelihood(), rtol=1e-9
    )


# -- prior-precision tuning ----------------------------------------------------


def _blr_mean(phi, y, lam):
    a = phi.T @ phi / SIGMA**2 + lam * jnp.eye(phi.shape[1])
    return jnp.linalg.solve(a, phi.T @ y / SIGMA**2)


def _blr_evidence(phi, y, lam):
    n = y.shape[0]
    cov = SIGMA**2 * jnp.eye(n) + phi @ phi.T / lam
    return jax.scipy.stats.multivariate_normal.logpdf(y, jnp.zeros(n), cov)


@pytest.fixture
def stale(blr):
    """The BLR problem with w* fitted at λ = 50, not at the λ under test."""
    phi = _phi(blr["params"], blr["x"])
    m = _blr_mean(phi, blr["y"], 50.0)
    params = {"features": blr["params"]["features"], "output": {"W": m[:H], "b": m[H]}}
    return {**blr, "params": params, "phi": phi}


def test_evidence_and_mode_are_exact_blr_away_from_the_fitted_lambda(stale):
    """The quadratic model re-solves the mode, so a stale w* does not matter."""
    post = _fit(stale, prior_precision=50.0)
    for lam in (0.1, 2.0, 50.0, 300.0):
        assert jnp.allclose(
            post.log_marginal_likelihood(lam),
            _blr_evidence(stale["phi"], stale["y"], lam),
            rtol=1e-9,
        )
        assert jnp.allclose(
            post.mode(lam), _blr_mean(stale["phi"], stale["y"], lam), rtol=1e-9
        )


def test_optimize_prior_precision_maximizes_exact_blr_evidence(stale):
    post = _fit(stale, prior_precision=50.0)
    tuned = post.optimize_prior_precision()
    # Closed-form BLR evidence on a dense grid; spacing 12 / 4000 = 0.003.
    grid = jnp.linspace(-6.0, 6.0, 4001)
    evs = jax.vmap(lambda g: _blr_evidence(stale["phi"], stale["y"], jnp.exp(g)))(grid)
    spacing = grid[1] - grid[0]
    log_lam = jnp.log(tuned.prior_precision)
    assert abs(log_lam - grid[jnp.argmax(evs)]) <= spacing
    # The refined optimum is no worse than the best grid point. The slack is
    # float64 round-off of a logpdf of size ~1e2 (≈1e-14 relative), with
    # room to spare.
    tuned_ev = _blr_evidence(stale["phi"], stale["y"], tuned.prior_precision)
    assert tuned_ev >= jnp.max(evs) - 1e-10
    assert tuned_ev > _blr_evidence(stale["phi"], stale["y"], 50.0)
    # The method's evidence and the returned mean are the BLR ones at λ̂.
    assert jnp.allclose(tuned.log_marginal_likelihood(), tuned_ev, rtol=1e-9)
    lam_hat = tuned.prior_precision
    w_hat = _blr_mean(stale["phi"], stale["y"], lam_hat)
    assert jnp.allclose(tuned.w_mean, w_hat, rtol=1e-9)
    mean, _ = tuned.predict(stale["x_new"])
    phi_new = _phi(stale["params"], stale["x_new"])
    assert jnp.allclose(mean, phi_new @ w_hat, rtol=1e-9)


def test_optimize_prior_precision_diag_is_stationary(stale):
    """No closed form for "diag": check the optimum against its own evidence."""
    post = _fit(stale, structure="diag", prior_precision=50.0)
    tuned = post.optimize_prior_precision()
    assert tuned.log_marginal_likelihood() > post.log_marginal_likelihood()

    def ev(log_lam):
        return tuned.log_marginal_likelihood(jnp.exp(log_lam))

    log_lam = jnp.log(tuned.prior_precision)
    # Golden-section leaves the bracket ~1e-21 wide in log λ, so the slope
    # there is float64 round-off of an O(1e2) evidence (~1e-12); 1e-6 is far
    # above that and far below the O(1) slope one decade away.
    assert abs(jax.grad(ev)(log_lam)) < 1e-6
    grid = jnp.linspace(log_lam - 9.0, log_lam + 9.0, 401)
    # Same round-off slack as in the full test above.
    assert tuned.log_marginal_likelihood() >= jnp.max(jax.vmap(ev)(grid)) - 1e-10


@pytest.mark.parametrize("structure", ["full", "diag"])
@pytest.mark.parametrize("zero_targets", [False, True])
def test_zero_weights_give_finite_lambda(blr, structure, zero_targets):
    """w* = 0 used to send λ to inf; with y = f = 0 too, b = 0 and the
    evidence increases in λ, so λ stops at the finite end of the grid."""
    params = {
        "features": blr["params"]["features"],
        "output": {"W": jnp.zeros(H), "b": jnp.zeros(())},
    }
    y = jnp.zeros_like(blr["y"]) if zero_targets else blr["y"]
    post = last_layer_laplace(
        _predict,
        params,
        blr["x"],
        y,
        sigma=SIGMA,
        param_filter=_is_output,
        structure=structure,
    ).optimize_prior_precision()
    assert jnp.isfinite(post.prior_precision)
    assert jnp.isfinite(post.log_marginal_likelihood())
    assert jnp.all(jnp.isfinite(post.predict(blr["x_new"])[1]))


def test_evidence_argument_matches_rebuilt_posterior(blr):
    post = _fit(blr)
    other = _fit(blr, prior_precision=0.3)
    assert jnp.allclose(
        post.log_marginal_likelihood(0.3), other.log_marginal_likelihood()
    )


# -- transforms and validation ---------------------------------------------------


def test_jit_and_grad(blr):
    post = _fit(blr)

    @eqx.filter_jit
    def predict(post, x):
        return post.predict(x)

    _, var = predict(post, blr["x_new"])
    assert jnp.allclose(var, blr["var"], rtol=1e-9)
    g = jax.grad(lambda lam: post.log_marginal_likelihood(lam))(1.0)
    assert jnp.isfinite(g)


def test_works_on_an_equinox_module():
    """Params may be a module with non-array leaves (activation functions)."""
    mlp = eqx.nn.MLP(1, 1, 8, 2, activation=jnp.sin, key=jr.key(3))
    x = jnp.linspace(-1, 1, 20)[:, None]
    y = jnp.sin(3 * x[:, 0])

    def predict(m, x):
        return jax.vmap(m)(x)[:, 0]

    def last(path, _leaf):
        return "layers[2]" in jax.tree_util.keystr(path)

    post = last_layer_laplace(predict, mlp, x, y, sigma=0.1, param_filter=last)
    assert post.w_map.shape == (8 + 1,)
    mean, var = post.predict(x)
    assert jnp.allclose(mean, predict(mlp, x))
    assert jnp.all(var > 0)


def test_rejects_unknown_structure(blr):
    with pytest.raises(ValueError, match="structure"):
        _fit(blr, structure="kfac")


def test_rejects_empty_selection(blr):
    with pytest.raises(ValueError, match="selected no"):
        last_layer_laplace(
            _predict,
            blr["params"],
            blr["x"],
            blr["y"],
            sigma=SIGMA,
            param_filter=lambda _p, _l: False,
        )


# -- 1-D SIREN with a held-out gap (slow) -------------------------------------


_SIREN_H, _OMEGA, _NOISE = 32, 3.0, 0.05


def _siren_init(key):
    k0, k1, k2 = jr.split(key, 3)
    lim = jnp.sqrt(6.0 / _SIREN_H) / _OMEGA  # SIREN hidden-layer init
    return {
        "hidden": {
            "W0": jr.uniform(k0, (1, _SIREN_H), minval=-1.0, maxval=1.0),
            "b0": jnp.zeros(_SIREN_H),
            "W1": jr.uniform(k1, (_SIREN_H, _SIREN_H), minval=-lim, maxval=lim),
            "b1": jnp.zeros(_SIREN_H),
        },
        "output": {
            "W": jr.normal(k2, (_SIREN_H,)) / jnp.sqrt(_SIREN_H),
            "b": jnp.zeros(()),
        },
    }


def _siren(p, x):
    h = jnp.sin(_OMEGA * (x[:, None] @ p["hidden"]["W0"] + p["hidden"]["b0"]))
    h = jnp.sin(_OMEGA * (h @ p["hidden"]["W1"] + p["hidden"]["b1"]))
    return h @ p["output"]["W"] + p["output"]["b"]


def _truth(x):
    return jnp.sin(3.0 * x) + 0.3 * x


@pytest.mark.slow
def test_siren_gap_inflates_variance_and_stays_calibrated_on_data():
    """MAP fit with `ensemble_map` (one member), then last-layer Laplace.

    Training data cover [-1, -0.25] and [0.25, 1]; the gap (-0.25, 0.25) is
    held out. Each band is mean ± 2 sd of the predictive (latent variance
    plus σ²), scored on 200 fresh noisy targets, so its nominal coverage is
    95.4 % with a binomial SE of √(0.954 · 0.046 / 200) ≈ 0.015. The bound
    0.954 - 4 · 0.015 = 0.894 covers noise resampling only; the MAP bias it
    must also absorb was measured on five training seeds (data seed s, init
    seed fold_in(key(1), s), s = 0 the seed below) when this was written:

    * in-data coverage 0.955-0.975 (smallest margin over 0.894: 0.06);
    * gap coverage 0.925-0.99 (smallest margin: 0.03);
    * gap / in-data mean latent sd 1.64-6.14, asserted at >= 1.25.

    The gap check can fail: last-layer Laplace is not calibrated in gaps in
    general (Foong et al., 2019, "'In-between' uncertainty in Bayesian
    neural networks"); it holds here because this SIREN's error in the gap
    stays within the inflated band on every seed tried.
    """
    import optax
    from pyrox.inference import ensemble_map

    kx, kn, kt = jr.split(jr.key(0), 3)
    x = jnp.concatenate(
        [
            jr.uniform(kx, (60,), minval=-1.0, maxval=-0.25),
            jr.uniform(jr.fold_in(kx, 1), (60,), minval=0.25, maxval=1.0),
        ]
    )
    y = _truth(x) + _NOISE * jr.normal(kn, x.shape)

    def log_joint(p, xb, yb):
        ll = -0.5 * jnp.sum((yb - _siren(p, xb)) ** 2) / _NOISE**2
        lp = -0.5 * sum(jnp.sum(leaf**2) for leaf in jax.tree.leaves(p))
        return ll, lp

    stacked, _ = ensemble_map(
        log_joint,
        _siren_init,
        ensemble_size=1,
        num_epochs=3000,
        data=(x, y),
        seed=jr.key(1),
        optimizer=optax.adam(3e-3),
    )
    params = jax.tree.map(lambda a: a[0], stacked)

    post = last_layer_laplace(
        _siren, params, x, y, sigma=_NOISE, param_filter=_is_output
    ).optimize_prior_precision()

    k_in, k_gap = jr.split(kt)
    x_in = jnp.concatenate(
        [
            jr.uniform(k_in, (100,), minval=-1.0, maxval=-0.25),
            jr.uniform(jr.fold_in(k_in, 1), (100,), minval=0.25, maxval=1.0),
        ]
    )
    x_gap = jr.uniform(k_gap, (200,), minval=-0.25, maxval=0.25)

    def covered(xs, key):
        mean, var = post.predict(xs)
        y_new = _truth(xs) + _NOISE * jr.normal(key, xs.shape)
        sd = jnp.sqrt(var + _NOISE**2)
        return jnp.mean(jnp.abs(y_new - mean) <= 2.0 * sd), jnp.mean(jnp.sqrt(var))

    cov_in, sd_in = covered(x_in, jr.key(10))
    cov_gap, sd_gap = covered(x_gap, jr.key(11))

    assert cov_in >= 0.954 - 4 * 0.015
    assert cov_gap >= 0.954 - 4 * 0.015
    assert sd_gap >= 1.25 * sd_in


def test_docstring_example_runs():
    """The `last_layer_laplace` example is a doctest (Codex review on #324:
    the old ``"layers[-1]"`` selector matched no key path)."""
    import doctest

    from pyrox.inference import _laplace

    finder = doctest.DocTestFinder()
    runner = doctest.DocTestRunner()
    for test in finder.find(_laplace.last_layer_laplace, "last_layer_laplace"):
        runner.run(test)
    assert runner.failures == 0
    assert runner.tries > 0
