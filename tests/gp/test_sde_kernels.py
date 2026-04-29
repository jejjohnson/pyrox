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

from pyrox.gp import MaternSDE
from pyrox.gp._src.kernels import matern_kernel


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
