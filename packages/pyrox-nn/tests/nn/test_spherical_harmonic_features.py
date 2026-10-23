"""Tests for `pyrox_nn.SphericalHarmonicFeatures` (weight-space GP on S²)."""

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro import handlers
from numpyro.infer import SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from pyrox_gp import RBF, Matern
from pyrox_nn import SphericalHarmonicFeatures


def _unit_points(key: jax.Array, n: int) -> jax.Array:
    """``n`` points uniform on S² as unit vectors ``(n, 3)``."""
    x = jr.normal(key, (n, 3))
    return x / jnp.linalg.norm(x, axis=1, keepdims=True)


def _feature_matrix(layer: SphericalHarmonicFeatures, x: jax.Array) -> jax.Array:
    r"""$B = \partial f / \partial \alpha$, i.e. columns $\sqrt{a_l} Y_{lm}(x)$.

    The layer is linear in ``alpha``, so the Jacobian is exact and
    $B B^\top$ is the covariance of the layer's prior: the truncated kernel.
    """
    name = f"{layer.pyrox_name or type(layer).__name__}.alpha"

    def f(alpha):
        return handlers.substitute(layer, data={name: alpha})(x)

    return jax.jit(jax.jacfwd(f))(jnp.zeros(layer.num_basis))  # (N, M)


# --- construction and validation -------------------------------------------


def test_output_shape_xyz():
    layer = SphericalHarmonicFeatures.init(l_max=3, kernel=Matern(nu=1.5))
    with handlers.seed(rng_seed=0):
        f = layer(_unit_points(jr.key(0), 11))
    assert f.shape == (11,)
    assert layer.num_basis == 16


@pytest.mark.parametrize("unit", ["degrees", "radians"])
def test_lonlat_matches_xyz(unit):
    """``lonlat`` input is the same function as the matching unit vectors."""
    kern = Matern(nu=1.5, init_lengthscale=0.5)
    lonlat_deg = jnp.array([[0.0, 0.0], [90.0, 45.0], [-120.0, -60.0], [10.0, 89.0]])
    lonlat = lonlat_deg if unit == "degrees" else jnp.deg2rad(lonlat_deg)
    lon, lat = jnp.deg2rad(lonlat_deg[:, 0]), jnp.deg2rad(lonlat_deg[:, 1])
    xyz = jnp.stack(
        [jnp.cos(lat) * jnp.cos(lon), jnp.cos(lat) * jnp.sin(lon), jnp.sin(lat)],
        axis=-1,
    )
    ll = SphericalHarmonicFeatures.init(
        l_max=5, kernel=kern, input_mode="lonlat", input_unit=unit
    )
    xy = SphericalHarmonicFeatures.init(l_max=5, kernel=kern)
    with handlers.seed(rng_seed=3):
        f_ll = ll(lonlat)
    with handlers.seed(rng_seed=3):
        f_xy = xy(xyz)
    # float64 round-off of the trig in the coordinate transform.
    np.testing.assert_allclose(f_ll, f_xy, atol=1e-12)


def test_init_validation():
    kern = Matern(nu=1.5)
    with pytest.raises(ValueError, match="l_max"):
        SphericalHarmonicFeatures.init(l_max=-1, kernel=kern)
    with pytest.raises(ValueError, match="num_quadrature"):
        SphericalHarmonicFeatures.init(l_max=2, kernel=kern, num_quadrature=0)
    with pytest.raises(ValueError, match="input_mode"):
        SphericalHarmonicFeatures.init(l_max=2, kernel=kern, input_mode="latlon")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="explicit input_unit"):
        SphericalHarmonicFeatures.init(l_max=2, kernel=kern, input_mode="lonlat")
    with pytest.raises(ValueError, match="explicit input_unit"):
        SphericalHarmonicFeatures.init(
            l_max=2,
            kernel=kern,
            input_mode="lonlat",
            input_unit="deg",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="lonlat' only"):
        SphericalHarmonicFeatures.init(l_max=2, kernel=kern, input_unit="degrees")


def test_rejects_wrong_input_shape():
    layer = SphericalHarmonicFeatures.init(l_max=2, kernel=Matern(nu=1.5))
    with handlers.seed(rng_seed=0), pytest.raises(ValueError, match=r"\(N, 3\)"):
        layer(jnp.zeros((4, 2)))


def test_rejects_ard_kernel():
    """An ARD lengthscale is not zonal, as in SphericalHarmonicInducingFeatures."""
    layer = SphericalHarmonicFeatures.init(l_max=4, kernel=Matern(nu=1.5, input_dim=3))
    with (
        handlers.seed(rng_seed=0),
        pytest.raises(NotImplementedError, match="isotropic"),
    ):
        layer(_unit_points(jr.key(0), 3))


# --- sites -------------------------------------------------------------------


def test_registers_alpha_site():
    layer = SphericalHarmonicFeatures.init(
        l_max=3, kernel=Matern(nu=1.5), pyrox_name="shf"
    )
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        layer(_unit_points(jr.key(0), 5))
    samples = {n: s for n, s in tr.items() if s["type"] == "sample"}
    assert set(samples) == {"shf.alpha"}
    assert samples["shf.alpha"]["value"].shape == (16,)


def test_default_scope_name():
    layer = SphericalHarmonicFeatures.init(l_max=1, kernel=RBF())
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        layer(_unit_points(jr.key(0), 2))
    assert "SphericalHarmonicFeatures.alpha" in tr


def test_kernel_hyperprior_registers_once():
    """Kernel priors register once, under the kernel's scope."""
    kern = Matern(nu=1.5, init_lengthscale=0.5)
    kern.set_prior("lengthscale", dist.LogNormal(jnp.log(0.5), 0.1))
    layer = SphericalHarmonicFeatures.init(l_max=4, kernel=kern, pyrox_name="shf")
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        layer(_unit_points(jr.key(0), 5))
    samples = {n for n, s in tr.items() if s["type"] == "sample"}
    assert samples == {"shf.alpha", "Matern.lengthscale"}


def test_two_instances_distinct_names():
    def model(x):
        # Each layer's kernel is its own module too, so it needs its own name.
        ka, kb = RBF(pyrox_name="ka"), RBF(pyrox_name="kb")
        a = SphericalHarmonicFeatures.init(l_max=2, kernel=ka, pyrox_name="a")
        b = SphericalHarmonicFeatures.init(l_max=2, kernel=kb, pyrox_name="b")
        return a(x) + b(x)

    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        model(_unit_points(jr.key(0), 3))
    assert {"a.alpha", "b.alpha"} <= set(tr)


# --- transforms and inference --------------------------------------------------


def test_jit_and_grad():
    layer = SphericalHarmonicFeatures.init(l_max=6, kernel=Matern(nu=1.5))
    x = _unit_points(jr.key(0), 8)

    @jax.jit
    def fwd(x):
        with handlers.seed(rng_seed=0):
            return layer(x)

    f = fwd(x)
    with handlers.seed(rng_seed=0):
        f_eager = layer(x)
    np.testing.assert_allclose(f, f_eager, rtol=1e-12)
    g = jax.grad(lambda x: fwd(x).sum())(x)
    assert g.shape == x.shape
    assert jnp.all(jnp.isfinite(g))


@pytest.mark.parametrize(
    "lengthscale",
    [
        pytest.param(1.0, marks=pytest.mark.slow),
        2.0,
        pytest.param(5.0, marks=pytest.mark.slow),
    ],
)
def test_float32_lengthscale_gradient_is_finite(lengthscale):
    """Regression: negative tail coefficients must not give a NaN gradient.

    In float32 the Funk-Hecke quadrature returns a few slightly negative
    a_l at l_max=40 for these RBF lengthscales; clipping them with
    ``sqrt(maximum(a, 0))`` has gradient 0 * inf = NaN there.
    """
    with jax.enable_x64(False):
        kern = RBF(init_lengthscale=lengthscale)
        layer = SphericalHarmonicFeatures.init(l_max=40, kernel=kern)
        x = _unit_points(jr.key(0), 4).astype(jnp.float32)
        alpha = jnp.ones(layer.num_basis, dtype=jnp.float32)

        def f(ls):
            data = {"SphericalHarmonicFeatures.alpha": alpha, "RBF.lengthscale": ls}
            return handlers.substitute(layer, data=data)(x).sum()

        g = jax.grad(f)(jnp.float32(lengthscale))
        assert jnp.isfinite(g)


def test_output_dtype_follows_input():
    """A float32 input gives a float32 output even with x64 on."""
    layer = SphericalHarmonicFeatures.init(l_max=2, kernel=Matern(nu=1.5))
    with handlers.seed(rng_seed=0):
        f = layer(_unit_points(jr.key(0), 3).astype(jnp.float32))
    assert f.dtype == jnp.float32


def test_svi_and_predictive_smoke():
    layer = SphericalHarmonicFeatures.init(
        l_max=1, kernel=Matern(nu=1.5), pyrox_name="shf"
    )
    x = _unit_points(jr.key(0), 20)
    y = x[:, 2]

    def model(x, y=None):
        f = layer(x)
        numpyro.sample("y", dist.Normal(f, 0.1).to_event(1), obs=y)

    optax = pytest.importorskip("optax")
    guide = AutoNormal(model)
    svi = SVI(model, guide, optax.adam(1e-2), Trace_ELBO())
    result = svi.run(jr.key(1), 20, x, y, progress_bar=False)
    assert jnp.all(jnp.isfinite(result.losses))
    pred = Predictive(model, guide=guide, params=result.params, num_samples=4)
    draws = pred(jr.key(2), x)
    assert draws["y"].shape == (4, 20)


# --- covariance ------------------------------------------------------------------


def test_monte_carlo_covariance_matches_truncated_kernel():
    r"""The empirical covariance of $f$ over prior draws is the truncated kernel.

    $f = B\alpha$ with $\alpha \sim \mathcal{N}(0, I)$, so $f \sim
    \mathcal{N}(0, K_L)$ with $K_L = B B^\top$ exactly. With $S$ draws and the
    known zero mean, $\hat C_{ij} = \frac{1}{S}\sum_s f_i^{(s)} f_j^{(s)}$ is
    unbiased and, by Isserlis' theorem,

    $$
    \mathrm{Var}[\hat C_{ij}] = \frac{K_{ii} K_{jj} + K_{ij}^2}{S}.
    $$

    For $S = 4000$ the CLT makes $\hat C_{ij}$ close to normal, so each entry
    is checked at 5 standard errors (two-sided $p \approx 5.7\times10^{-7}$);
    over the 15 distinct entries of a 5x5 matrix the family-wise false-failure
    rate is below $10^{-5}$ (Bonferroni). The keys are fixed, so the test is
    deterministic.

    $K_L$ here is built from the layer's own features, so this checks the
    prior sampling ($\alpha \sim \mathcal{N}(0, I)$ scaled by
    $\sqrt{a_l}$), not the kernel; the comparison against the kernel is
    `test_truncation_error_decreases_with_l_max`.
    """
    layer = SphericalHarmonicFeatures.init(
        l_max=4, kernel=Matern(nu=1.5, init_lengthscale=0.5)
    )
    x = _unit_points(jr.key(0), 5)
    n_draws = 4000
    keys = jr.split(jr.key(1), n_draws)
    F = jax.jit(jax.vmap(lambda k: handlers.seed(layer, rng_seed=k)(x)))(keys)
    # F: (S, 5)
    C_hat = F.T @ F / n_draws
    K_L = _feature_matrix(layer, x)
    K_L = K_L @ K_L.T
    d = jnp.diag(K_L)
    se = jnp.sqrt((d[:, None] * d[None, :] + K_L**2) / n_draws)
    assert jnp.all(jnp.abs(C_hat - K_L) <= 5.0 * se)


@pytest.mark.parametrize(
    ("kern", "l_maxes"),
    [
        pytest.param(Matern(nu=1.5, init_lengthscale=0.5), (2, 4, 8), id="matern32"),
        pytest.param(
            Matern(nu=1.5, init_lengthscale=0.5),
            (2, 4, 8, 16),
            id="matern32-L16",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            RBF(init_lengthscale=0.5),
            (2, 4, 8, 16),
            id="rbf-L16",
            marks=pytest.mark.slow,
        ),
    ],
)
def test_truncation_error_decreases_with_l_max(kern, l_maxes):
    r"""$k_L \to k$ as $L$ grows.

    With $c_l = a_l (2l+1)/(4\pi) \ge 0$, $k(n,n') = \sum_l c_l
    P_l(n\cdot n')$ and $|P_l| \le 1$, so

    $$
    \sup_{n,n'} |k - k_L| = e_L := \sum_{l>L} c_l = k(n,n) - k_L(n,n),
    $$

    attained on the diagonal. $e_L$ is non-increasing in $L$; the test checks
    it strictly decreases over ``l_maxes`` and that the off-diagonal
    error stays below it. The slack ``1e-9`` covers the Gauss-Legendre error
    of the Funk-Hecke coefficients (about 1e-12 for these kernels with 256
    nodes) and float64 round-off.
    """
    x = _unit_points(jr.key(0), 30)
    K = kern(x, x)
    diag_err = []
    for l_max in l_maxes:
        layer = SphericalHarmonicFeatures.init(l_max=l_max, kernel=kern)
        B = _feature_matrix(layer, x)
        K_L = B @ B.T
        e_diag = jnp.max(jnp.abs(jnp.diag(K) - jnp.diag(K_L)))
        assert jnp.max(jnp.abs(K - K_L)) <= e_diag + 1e-9
        diag_err.append(float(e_diag))
    assert all(a > b for a, b in itertools.pairwise(diag_err)), diag_err
    if isinstance(kern, RBF) and l_maxes[-1] >= 16:
        # The RBF coefficients decay super-exponentially: L = 16 is exact
        # to quadrature accuracy.
        assert diag_err[-1] < 1e-9


# --- posterior --------------------------------------------------------------------


@pytest.mark.slow
def test_ensemble_map_matches_exact_gp_mean():
    r"""EnsembleMAP on the layer recovers the exact GP posterior mean.

    The weight-space model $y = B\alpha + \varepsilon$, $\alpha \sim
    \mathcal{N}(0, I)$, $\varepsilon \sim \mathcal{N}(0, \sigma^2 I)$ has the
    strictly convex negative log-posterior
    $\mathcal{L}(\alpha) = \lVert y - B\alpha\rVert^2 / 2\sigma^2 +
    \lVert\alpha\rVert^2/2$, Hessian $H = B^\top B/\sigma^2 + I$, and its MAP
    $\alpha^\star$ gives $B\alpha^\star = K_L(K_L + \sigma^2 I)^{-1} y$: the
    exact GP mean under the truncated kernel $K_L = BB^\top$. Two bounds make
    the tolerance:

    1. Optimisation. For an iterate $\hat\alpha$ with gradient
       $g = H(\hat\alpha - \alpha^\star)$, $\lVert B(\hat\alpha -
       \alpha^\star)\rVert_2 = \lVert B H^{-1} g\rVert_2 \le
       \tfrac{\sigma}{2}\lVert g\rVert_2$, since each singular value $s$ of
       $B$ gives $s / (s^2/\sigma^2 + 1) \le \sigma/2$.
    2. Truncation. $\mu - \mu_L = \sigma^2 (K + \sigma^2 I)^{-1}(K - K_L)
       v_L$ with $v_L = (K_L + \sigma^2 I)^{-1} y$, and
       $\lVert\sigma^2(K + \sigma^2 I)^{-1}\rVert_2 \le 1$, so
       $\lVert\mu - \mu_L\rVert_2 \le \lVert(K - K_L) v_L\rVert_2$.

    Both bounds are computed, not tuned; the test also checks that their sum
    is far below the noise level (RMS under $0.01\sigma$), so the match is
    not vacuous. ``1e-8 * ||y||`` covers float64 round-off in the dense
    solves. 1,000 points (not the issue's 2,000) keep the dense reference
    affordable; the check does not depend on N.
    """
    optax = pytest.importorskip("optax")
    from pyrox.inference import EnsembleMAP

    n, sigma, n_steps = 1000, 0.1, 6000
    kern = Matern(nu=2.5, init_lengthscale=0.5)
    layer = SphericalHarmonicFeatures.init(l_max=30, kernel=kern, pyrox_name="shf")
    k_x, k_eps = jr.split(jr.key(0))
    x = _unit_points(k_x, n)
    lon, lat = jnp.arctan2(x[:, 1], x[:, 0]), jnp.arcsin(x[:, 2])
    y = jnp.sin(2 * lon) * jnp.cos(lat) + 0.5 * x[:, 2] ** 2
    y = y + sigma * jr.normal(k_eps, (n,))

    def log_joint(alpha, xb, yb):
        with handlers.trace() as tr, handlers.substitute(data={"shf.alpha": alpha}):
            f = layer(xb)
        site = tr["shf.alpha"]
        loglik = dist.Normal(f, sigma).log_prob(yb).sum()
        return loglik, site["fn"].log_prob(site["value"]).sum()

    runner = EnsembleMAP(
        log_joint=log_joint,
        init_fn=lambda key: 0.1 * jr.normal(key, (layer.num_basis,)),
        optimizer=optax.adam(optax.exponential_decay(3e-2, n_steps, 1e-3)),
        ensemble_size=2,
    )
    result = jax.jit(lambda: runner.run(jr.key(1), n_steps, x, y))()

    B = _feature_matrix(layer, x)  # (N, M)
    K_L = B @ B.T
    K = kern(x, x)
    eye = jnp.eye(n)
    v_L = jnp.linalg.solve(K_L + sigma**2 * eye, y)
    mu_L = K_L @ v_L
    mu = K @ jnp.linalg.solve(K + sigma**2 * eye, y)
    trunc_bound = jnp.linalg.norm((K - K_L) @ v_L)
    slack = 1e-8 * jnp.linalg.norm(y)

    neg_log_post = jax.grad(lambda a: -sum(log_joint(a, x, y)))
    for alpha in result.params:
        mu_map = B @ alpha
        opt_bound = 0.5 * sigma * jnp.linalg.norm(neg_log_post(alpha))
        assert jnp.linalg.norm(mu_map - mu_L) <= opt_bound + slack
        assert jnp.linalg.norm(mu_map - mu) <= opt_bound + trunc_bound + slack
        assert (opt_bound + trunc_bound) / jnp.sqrt(n) < 0.01 * sigma
