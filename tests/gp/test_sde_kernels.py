"""Tests for state-space (SDE) kernel representations in pyrox.gp.

The contract is small: each :class:`SDEKernel` produces a closed-form
``(F, L, H, Q_c, P_inf)`` tuple satisfying the Lyapunov equation, and a
``discretise(dt)`` map that returns PSD process-noise covariances. The
SDE autocovariance ``H exp(F tau) P_inf H^T`` must reproduce the dense
kernel value on the same scalar grid.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import pytest

from pyrox.gp import (
    ConstantSDE,
    CosineSDE,
    MaternSDE,
    PeriodicSDE,
    ProductSDE,
    QuasiPeriodicSDE,
    SumSDE,
)
from pyrox.gp._sde_kernels import _scaled_bessel_i_seq
from pyrox.gp._src.kernels import matern_kernel, periodic_kernel


# --- structural shape contract -------------------------------------------


@pytest.mark.parametrize(("order", "expected_dim"), [(0, 1), (1, 2), (2, 3)])
def test_state_dim_matches_order(order: int, expected_dim: int) -> None:
    sde = MaternSDE(variance=1.0, lengthscale=1.0, order=order)
    assert sde.state_dim == expected_dim
    assert sde.nu == order + 0.5


@pytest.mark.parametrize("order", [0, 1, 2])
def test_sde_params_shapes(order: int) -> None:
    sde = MaternSDE(variance=0.7, lengthscale=0.4, order=order)
    F, L, H, Q_c, P_inf = sde.sde_params()
    d = order + 1
    assert F.shape == (d, d)
    assert L.shape == (d, 1)
    assert H.shape == (1, d)
    assert Q_c.shape == (1, 1)
    assert P_inf.shape == (d, d)


def test_invalid_order_raises() -> None:
    with pytest.raises(ValueError, match="order in"):
        MaternSDE(variance=1.0, lengthscale=1.0, order=3)


def test_invalid_variance_raises() -> None:
    with pytest.raises(ValueError, match="variance must be positive"):
        MaternSDE(variance=-1.0, lengthscale=1.0, order=1)
    with pytest.raises(ValueError, match="variance must be positive"):
        MaternSDE(variance=0.0, lengthscale=1.0, order=1)


def test_invalid_lengthscale_raises() -> None:
    with pytest.raises(ValueError, match="lengthscale must be positive"):
        MaternSDE(variance=1.0, lengthscale=-0.5, order=1)
    with pytest.raises(ValueError, match="lengthscale must be positive"):
        MaternSDE(variance=1.0, lengthscale=0.0, order=1)


def test_integer_inputs_coerced_to_float() -> None:
    """Integer ``variance`` / ``lengthscale`` must not propagate as integer dtype."""
    sde = MaternSDE(variance=1, lengthscale=1, order=1)
    assert jnp.issubdtype(sde.variance.dtype, jnp.floating)
    assert jnp.issubdtype(sde.lengthscale.dtype, jnp.floating)
    F, _L, _H, _Q_c, P_inf = sde.sde_params()
    assert jnp.issubdtype(F.dtype, jnp.floating)
    assert jnp.issubdtype(P_inf.dtype, jnp.floating)


def test_discretise_q_is_symmetric() -> None:
    """``Q_k`` returned from the default ``discretise`` must be symmetric."""
    sde = MaternSDE(variance=1.0, lengthscale=0.3, order=2)
    _A, Q = sde.discretise(jnp.array([0.05, 0.1, 0.5, 2.0]))
    # Each Q[i] should equal its transpose to machine precision.
    for q in Q:
        assert jnp.allclose(q, q.T, atol=1e-7)


# --- stationary-covariance / Lyapunov consistency ------------------------


@pytest.mark.parametrize("order", [0, 1, 2])
def test_h_p_inf_recovers_variance(order: int) -> None:
    """``H P_inf H^T`` must equal the kernel variance."""
    sde = MaternSDE(variance=0.7, lengthscale=0.4, order=order)
    _F, _L, H, _Q_c, P_inf = sde.sde_params()
    var_recovered = (H @ P_inf @ H.T).squeeze()
    assert jnp.allclose(var_recovered, jnp.asarray(0.7), atol=1e-6)


@pytest.mark.parametrize("order", [0, 1, 2])
def test_lyapunov_equation_holds(order: int) -> None:
    """``F P_inf + P_inf F^T + L Q_c L^T = 0`` to numerical precision."""
    sde = MaternSDE(variance=1.3, lengthscale=0.6, order=order)
    F, L, _H, Q_c, P_inf = sde.sde_params()
    residual = F @ P_inf + P_inf @ F.T + L @ Q_c @ L.T
    # Float32 roundoff scales with ``lambda ** (2*order+1)``; the order=2
    # entries of ``F P_inf`` reach magnitudes near :math:`\lambda^5 \approx 700`.
    scale = jnp.maximum(jnp.abs(F @ P_inf).max(), 1.0)
    assert jnp.allclose(residual, jnp.zeros_like(residual), atol=1e-5 * scale)


@pytest.mark.parametrize("order", [0, 1, 2])
def test_p_inf_is_psd(order: int) -> None:
    sde = MaternSDE(variance=0.9, lengthscale=0.3, order=order)
    _F, _L, _H, _Q_c, P_inf = sde.sde_params()
    eigvals = jnp.linalg.eigvalsh(0.5 * (P_inf + P_inf.T))
    assert jnp.all(eigvals > -1e-7)


# --- discretisation ------------------------------------------------------


@pytest.mark.parametrize("order", [0, 1, 2])
def test_discretise_shapes(order: int) -> None:
    sde = MaternSDE(variance=1.0, lengthscale=0.5, order=order)
    dt = jnp.array([0.05, 0.1, 0.2, 0.5])
    A, Q = sde.discretise(dt)
    d = order + 1
    assert A.shape == (4, d, d)
    assert Q.shape == (4, d, d)


@pytest.mark.parametrize("order", [0, 1, 2])
def test_discretise_zero_dt_is_identity_and_zero(order: int) -> None:
    """``A(0) = I`` and ``Q(0) = 0`` (no time evolution)."""
    sde = MaternSDE(variance=1.0, lengthscale=0.7, order=order)
    A, Q = sde.discretise(jnp.array([0.0]))
    d = order + 1
    assert jnp.allclose(A[0], jnp.eye(d), atol=1e-6)
    assert jnp.allclose(Q[0], jnp.zeros((d, d)), atol=1e-6)


@pytest.mark.parametrize("order", [0, 1, 2])
def test_q_k_is_psd(order: int) -> None:
    sde = MaternSDE(variance=1.0, lengthscale=0.5, order=order)
    dt = jnp.array([0.01, 0.05, 0.2, 1.0, 5.0])
    _A, Q = sde.discretise(dt)
    for q in Q:
        eig = jnp.linalg.eigvalsh(0.5 * (q + q.T))
        assert jnp.all(eig > -1e-6)


@pytest.mark.parametrize("order", [0, 1, 2])
def test_discretise_long_dt_relaxes_to_p_inf(order: int) -> None:
    """For ``dt`` much greater than the correlation time, ``Q -> P_inf``."""
    sde = MaternSDE(variance=1.0, lengthscale=0.2, order=order)
    _F, _L, _H, _Q_c, P_inf = sde.sde_params()
    _A, Q = sde.discretise(jnp.array([50.0]))
    assert jnp.allclose(Q[0], P_inf, atol=1e-3)


# --- covariance recovery against the dense Matern kernel ----------------


@pytest.mark.parametrize(
    ("order", "nu"),
    [(0, 0.5), (1, 1.5), (2, 2.5)],
)
def test_sde_autocov_matches_dense_matern(order: int, nu: float) -> None:
    r"""Stationary autocov ``H exp(F tau) P_inf H^T`` matches dense Matern.

    For any stationary SDE kernel the continuous autocovariance is
    :math:`k(\tau) = H \exp(F\tau) P_\infty H^\top` for :math:`\tau \geq 0`.
    This must equal the dense Matern Gram value at lag :math:`\tau`.
    """
    sigma2 = 0.6
    ell = 0.35
    sde = MaternSDE(variance=sigma2, lengthscale=ell, order=order)
    F, _L, H, _Q_c, P_inf = sde.sde_params()

    taus = jnp.linspace(0.0, 2.5, 21)

    def k_sde(tau: jax.Array) -> jax.Array:
        return (H @ jsl.expm(F * tau) @ P_inf @ H.T).squeeze()

    K_sde = jax.vmap(k_sde)(taus)

    X = taus[:, None]
    X0 = jnp.zeros((1, 1))
    K_dense = matern_kernel(
        X, X0, jnp.asarray(sigma2), jnp.asarray(ell), nu=nu
    ).squeeze()

    assert jnp.allclose(K_sde, K_dense, atol=1e-5)


@pytest.mark.parametrize("order", [0, 1, 2])
def test_sde_kernel_is_jit_compatible(order: int) -> None:
    """`sde_params` and `discretise` must compile under jit."""
    sde = MaternSDE(variance=1.0, lengthscale=0.5, order=order)

    @jax.jit
    def go(s: MaternSDE, dt: jax.Array) -> tuple[jax.Array, jax.Array]:
        return s.discretise(dt)

    A, Q = go(sde, jnp.array([0.1, 0.2]))
    d = order + 1
    assert A.shape == (2, d, d) and Q.shape == (2, d, d)


# --- ConstantSDE ----------------------------------------------------------


def test_constant_sde_recovers_variance() -> None:
    sde = ConstantSDE(variance=2.5)
    F, L, H, Q_c, P_inf = sde.sde_params()
    assert sde.state_dim == 1
    assert jnp.allclose(F, 0.0)
    assert jnp.allclose(L, 0.0)
    assert jnp.allclose(Q_c, 0.0)
    assert jnp.allclose((H @ P_inf @ H.T).squeeze(), 2.5)


def test_constant_sde_discretise_is_identity() -> None:
    sde = ConstantSDE(variance=1.0)
    A, Q = sde.discretise(jnp.array([0.0, 0.5, 5.0]))
    assert jnp.allclose(A, jnp.ones((3, 1, 1)))
    assert jnp.allclose(Q, jnp.zeros((3, 1, 1)))


def test_constant_sde_invalid_variance_raises() -> None:
    with pytest.raises(ValueError, match="variance must be positive"):
        ConstantSDE(variance=-1.0)
    with pytest.raises(ValueError, match="variance must be positive"):
        ConstantSDE(variance=0.0)


def test_constant_sde_integer_inputs_coerced_to_float() -> None:
    sde = ConstantSDE(variance=2)
    assert jnp.issubdtype(sde.variance.dtype, jnp.floating)
    _F, _L, _H, _Q_c, P_inf = sde.sde_params()
    assert jnp.issubdtype(P_inf.dtype, jnp.floating)


# --- CosineSDE ------------------------------------------------------------


def test_cosine_sde_autocov_matches_cosine_kernel() -> None:
    """The closed-form rotation transition reproduces ``sigma^2 cos(omega tau)``.

    We use the closed-form ``A_k`` produced by :meth:`CosineSDE.discretise`
    rather than a generic ``expm`` of the generator: float32 ``expm`` of a
    rotation generator accumulates ~1e-3 error for ``omega * tau`` of order
    10, which would mask kernel-correctness issues. The closed-form path
    is exact to machine precision and is what downstream Kalman code uses.
    """
    sigma2 = 1.7
    omega = 2.5
    sde = CosineSDE(variance=sigma2, frequency=omega)
    _F, _L, H, _Q_c, P_inf = sde.sde_params()

    taus = jnp.linspace(0.0, 4.0, 33)
    A, _Q = sde.discretise(taus)
    # K(tau) = H A(tau) P_inf H^T
    K_sde = jnp.einsum("ij,njk,kl,lm->n", H, A, P_inf, H.T)
    K_true = sigma2 * jnp.cos(omega * taus)
    assert jnp.allclose(K_sde, K_true, atol=1e-5)


def test_cosine_sde_discretise_is_pure_rotation() -> None:
    sde = CosineSDE(variance=1.0, frequency=3.0)
    dt = jnp.array([0.0, 0.1, 0.5, 1.0])
    A, Q = sde.discretise(dt)
    # Q should be identically zero (deterministic).
    assert jnp.allclose(Q, jnp.zeros_like(Q))
    # A should be a rotation: A.T @ A = I and det(A) = 1.
    AtA = jnp.einsum("nij,nik->njk", A, A)
    assert jnp.allclose(AtA, jnp.broadcast_to(jnp.eye(2), (4, 2, 2)), atol=1e-6)
    dets = jnp.linalg.det(A)
    assert jnp.allclose(dets, jnp.ones(4), atol=1e-6)


def test_cosine_sde_closed_form_matches_expm() -> None:
    """Closed-form rotation override matches the generic ``expm`` path."""
    sde = CosineSDE(variance=1.0, frequency=1.7)
    F, _L, _H, _Q_c, _P_inf = sde.sde_params()
    dt = jnp.array([0.05, 0.3, 0.9])
    A_closed, _ = sde.discretise(dt)
    A_expm = jax.vmap(lambda t: jsl.expm(F * t))(dt)
    assert jnp.allclose(A_closed, A_expm, atol=1e-5)


def test_cosine_sde_invalid_variance_raises() -> None:
    with pytest.raises(ValueError, match="variance must be positive"):
        CosineSDE(variance=-0.5, frequency=1.0)
    with pytest.raises(ValueError, match="variance must be positive"):
        CosineSDE(variance=0.0, frequency=1.0)


def test_cosine_sde_integer_inputs_coerced_to_float() -> None:
    sde = CosineSDE(variance=1, frequency=2)
    assert jnp.issubdtype(sde.variance.dtype, jnp.floating)
    assert jnp.issubdtype(sde.frequency.dtype, jnp.floating)
    F, _L, _H, _Q_c, P_inf = sde.sde_params()
    assert jnp.issubdtype(F.dtype, jnp.floating)
    assert jnp.issubdtype(P_inf.dtype, jnp.floating)


# --- SumSDE ---------------------------------------------------------------


def test_sum_sde_state_dim_and_shapes() -> None:
    components = (
        MaternSDE(variance=1.0, lengthscale=0.5, order=1),
        ConstantSDE(variance=0.4),
        CosineSDE(variance=0.3, frequency=2.0),
    )
    sde = SumSDE(components)
    assert sde.state_dim == 2 + 1 + 2
    F, _L, H, _Q_c, P_inf = sde.sde_params()
    assert F.shape == (5, 5)
    assert H.shape == (1, 5)
    assert P_inf.shape == (5, 5)
    # Block-diagonal structure: off-block entries of F are zero.
    # Block boundaries are at rows/cols 0..1, 2, 3..4.
    assert jnp.allclose(F[2:3, 0:2], 0.0)
    assert jnp.allclose(F[3:5, 0:3], 0.0)


def test_sum_sde_autocov_equals_sum_of_components() -> None:
    """Autocovariance of ``Sum(k1, k2, ...)`` equals ``sum_i k_i``."""
    sigma2_m, ell = 0.7, 0.4
    c0 = 0.3
    components = (
        MaternSDE(variance=sigma2_m, lengthscale=ell, order=1),
        ConstantSDE(variance=c0),
    )
    sde = SumSDE(components)
    F, _L, H, _Q_c, P_inf = sde.sde_params()

    taus = jnp.linspace(0.0, 2.5, 21)
    K_sde = jax.vmap(lambda t: (H @ jsl.expm(F * t) @ P_inf @ H.T).squeeze())(taus)

    X = taus[:, None]
    X0 = jnp.zeros((1, 1))
    K_truth = matern_kernel(
        X, X0, jnp.asarray(sigma2_m), jnp.asarray(ell), nu=1.5
    ).squeeze() + jnp.asarray(c0)
    assert jnp.allclose(K_sde, K_truth, atol=1e-5)


def test_sum_sde_lyapunov_holds() -> None:
    components = (
        MaternSDE(variance=1.0, lengthscale=0.5, order=2),
        MaternSDE(variance=0.5, lengthscale=1.0, order=1),
    )
    sde = SumSDE(components)
    F, L, _H, Q_c, P_inf = sde.sde_params()
    res = F @ P_inf + P_inf @ F.T + L @ Q_c @ L.T
    scale = jnp.maximum(jnp.abs(F @ P_inf).max(), 1.0)
    assert jnp.allclose(res, jnp.zeros_like(res), atol=1e-5 * scale)


def test_sum_sde_empty_raises() -> None:
    with pytest.raises(ValueError, match="at least one component"):
        SumSDE(())


# --- ProductSDE ----------------------------------------------------------


def test_product_sde_state_dim_and_lyapunov() -> None:
    left = MaternSDE(variance=1.0, lengthscale=0.4, order=1)
    right = CosineSDE(variance=1.0, frequency=2.0)
    prod = ProductSDE(left, right)
    assert prod.state_dim == 2 * 2

    F, L, _H, Q_c, P_inf = prod.sde_params()
    res = F @ P_inf + P_inf @ F.T + L @ Q_c @ L.T
    scale = jnp.maximum(jnp.abs(F @ P_inf).max(), 1.0)
    assert jnp.allclose(res, jnp.zeros_like(res), atol=1e-5 * scale)


def test_product_sde_autocov_matches_product_of_components() -> None:
    """Autocovariance of ``Product(k1, k2)`` equals ``k1 * k2``."""
    sigma2_m, ell = 1.2, 0.5
    omega = 2.0 * jnp.pi  # period 1
    left = MaternSDE(variance=sigma2_m, lengthscale=ell, order=1)
    right = CosineSDE(variance=1.0, frequency=omega)
    prod = ProductSDE(left, right)

    F, _L, H, _Q_c, P_inf = prod.sde_params()

    taus = jnp.linspace(0.0, 2.0, 21)
    K_sde = jax.vmap(lambda t: (H @ jsl.expm(F * t) @ P_inf @ H.T).squeeze())(taus)

    X = taus[:, None]
    X0 = jnp.zeros((1, 1))
    K_left = matern_kernel(
        X, X0, jnp.asarray(sigma2_m), jnp.asarray(ell), nu=1.5
    ).squeeze()
    K_right = jnp.cos(omega * taus)
    K_truth = K_left * K_right
    assert jnp.allclose(K_sde, K_truth, atol=1e-4)


def test_quasi_periodic_is_product() -> None:
    qp = QuasiPeriodicSDE(
        MaternSDE(variance=1.0, lengthscale=2.0, order=1),
        PeriodicSDE(variance=1.0, lengthscale=1.0, period=1.0, n_harmonics=4),
    )
    assert isinstance(qp, ProductSDE)
    # State dim is d_mat * d_per = 2 * (1 + 2*4) = 18.
    assert qp.state_dim == 2 * 9


# --- PeriodicSDE ---------------------------------------------------------


@pytest.mark.parametrize("x", [0.05, 0.25, 1.0, 4.0, 11.0, 30.0])
def test_scaled_bessel_recursion_matches_scipy(x: float) -> None:
    """Sanity-check the Bessel helper against scipy's reference."""
    sp = pytest.importorskip("scipy.special")
    ours = _scaled_bessel_i_seq(jnp.asarray(x), j_max=10)
    truth = jnp.exp(-x) * jnp.array([sp.iv(j, x) for j in range(11)])
    assert jnp.allclose(ours, truth, rtol=1e-4, atol=1e-6)


def test_scaled_bessel_zero_argument_limit() -> None:
    """``exp(-0) I_j(0)`` is ``[1, 0, 0, ...]``; helper must not return NaN."""
    seq = _scaled_bessel_i_seq(jnp.asarray(0.0), j_max=5)
    assert jnp.all(jnp.isfinite(seq))
    expected = jnp.zeros(6).at[0].set(1.0)
    assert jnp.allclose(seq, expected, atol=1e-7)


def test_periodic_sde_handles_infinite_lengthscale() -> None:
    """``lengthscale -> inf`` yields ``x = 1/ell^2 = 0``; ``P_inf`` must be finite."""
    sde = PeriodicSDE(variance=1.0, lengthscale=jnp.inf, period=1.0, n_harmonics=4)
    _F, _L, _H, _Q_c, P_inf = sde.sde_params()
    assert jnp.all(jnp.isfinite(P_inf))


def test_periodic_sde_invalid_n_harmonics_raises() -> None:
    with pytest.raises(ValueError, match="n_harmonics"):
        PeriodicSDE(variance=1.0, lengthscale=1.0, period=1.0, n_harmonics=0)


def test_periodic_sde_invalid_scalar_inputs_raise() -> None:
    with pytest.raises(ValueError, match="variance must be positive"):
        PeriodicSDE(variance=-1.0, lengthscale=1.0, period=1.0)
    with pytest.raises(ValueError, match="lengthscale must be positive"):
        PeriodicSDE(variance=1.0, lengthscale=0.0, period=1.0)
    with pytest.raises(ValueError, match="period must be positive"):
        PeriodicSDE(variance=1.0, lengthscale=1.0, period=-2.0)


def test_periodic_sde_integer_inputs_coerced_to_float() -> None:
    sde = PeriodicSDE(variance=1, lengthscale=1, period=2, n_harmonics=3)
    assert jnp.issubdtype(sde.variance.dtype, jnp.floating)
    assert jnp.issubdtype(sde.lengthscale.dtype, jnp.floating)
    assert jnp.issubdtype(sde.period.dtype, jnp.floating)
    F, _L, _H, _Q_c, P_inf = sde.sde_params()
    assert jnp.issubdtype(F.dtype, jnp.floating)
    assert jnp.issubdtype(P_inf.dtype, jnp.floating)


def test_periodic_sde_no_driving_noise() -> None:
    """``L = 0`` and ``Q_c = 0`` (deterministic harmonic decomposition)."""
    sde = PeriodicSDE(variance=1.0, lengthscale=1.0, period=1.0, n_harmonics=5)
    _F, L, _H, Q_c, _P_inf = sde.sde_params()
    assert jnp.allclose(L, 0.0)
    assert jnp.allclose(Q_c, 0.0)


def test_periodic_sde_lyapunov_holds() -> None:
    sde = PeriodicSDE(variance=1.0, lengthscale=1.0, period=2.0, n_harmonics=6)
    F, L, _H, Q_c, P_inf = sde.sde_params()
    res = F @ P_inf + P_inf @ F.T + L @ Q_c @ L.T
    assert jnp.allclose(res, jnp.zeros_like(res), atol=1e-5)


def test_periodic_sde_closed_form_discretise_is_pure_rotation() -> None:
    """``discretise`` returns block rotations and ``Q_k = 0``.

    For ``j * omega_0 * dt`` of order 5 (e.g. period 1, J=5, dt of a few
    units), float32 ``expm`` accumulates ~1e-3 error and produces a
    spurious nonzero ``Q_k``; the closed-form override is exact.
    """
    sde = PeriodicSDE(variance=1.0, lengthscale=1.0, period=0.5, n_harmonics=5)
    dt = jnp.array([0.0, 0.05, 0.3, 1.0])
    A, Q = sde.discretise(dt)
    d = sde.state_dim
    # Q is identically zero (deterministic dynamics).
    assert jnp.allclose(Q, jnp.zeros_like(Q))
    # A is orthogonal: A^T A = I (rotation blocks plus 1x1 identity).
    AtA = jnp.einsum("nij,nik->njk", A, A)
    assert jnp.allclose(AtA, jnp.broadcast_to(jnp.eye(d), (4, d, d)), atol=1e-5)
    # dt = 0 gives the identity transition.
    assert jnp.allclose(A[0], jnp.eye(d), atol=1e-7)


def test_periodic_sde_autocov_matches_dense_periodic_kernel() -> None:
    """Truncated SDE autocov approximates the MacKay periodic kernel."""
    sigma2 = 1.0
    ell = 1.0
    period = 2.0
    sde = PeriodicSDE(variance=sigma2, lengthscale=ell, period=period, n_harmonics=8)
    F, _L, H, _Q_c, P_inf = sde.sde_params()

    taus = jnp.linspace(0.0, 2.0 * period, 41)
    K_sde = jax.vmap(lambda t: (H @ jsl.expm(F * t) @ P_inf @ H.T).squeeze())(taus)

    X = taus[:, None]
    X0 = jnp.zeros((1, 1))
    K_dense = periodic_kernel(
        X, X0, jnp.asarray(sigma2), jnp.asarray(ell), jnp.asarray(period)
    ).squeeze()
    # 8 harmonics matches MacKay to better than 1e-3 in float32.
    assert jnp.allclose(K_sde, K_dense, atol=1e-3)


# --- jit compatibility for new kernels -----------------------------------


def test_sum_and_product_sde_jit_compatible() -> None:
    from pyrox.gp import SDEKernel

    summed = SumSDE(
        (MaternSDE(variance=1.0, lengthscale=0.5, order=1), ConstantSDE(0.3))
    )
    product = ProductSDE(MaternSDE(order=1), CosineSDE(frequency=2.0))

    @jax.jit
    def discretise(s: SDEKernel, dt: jax.Array) -> tuple[jax.Array, jax.Array]:
        return s.discretise(dt)

    A_s, Q_s = discretise(summed, jnp.array([0.1, 0.2]))
    A_p, Q_p = discretise(product, jnp.array([0.1, 0.2]))
    assert A_s.shape == (2, 3, 3) and Q_s.shape == (2, 3, 3)
    assert A_p.shape == (2, 4, 4) and Q_p.shape == (2, 4, 4)
