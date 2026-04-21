"""Tests for `pyrox._basis._spherical` — real spherical harmonics on S^2.

Cross-checked against :func:`scipy.special.sph_harm_y` (complex SHs) under
the convention real ``Y_l^m = sqrt(2) Re[Y_l^m^complex]`` for ``m > 0`` and
``sqrt(2) Im[Y_l^|m|^complex]`` for ``m < 0`` (Condon-Shortley phase
already in the complex SH; no extra ``(-1)^m`` wrap).
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.polynomial.legendre import legval
from scipy.special import sph_harm_y

from pyrox._basis import harmonic_degrees, real_spherical_harmonics


def _random_unit_vectors(n: int, seed: int = 0) -> jnp.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return jnp.asarray(v)


def test_shape_and_degrees():
    xyz = _random_unit_vectors(7)
    Y = real_spherical_harmonics(xyz, l_max=4)
    assert Y.shape == (7, (4 + 1) ** 2)
    degs = harmonic_degrees(4)
    assert len(degs) == (4 + 1) ** 2
    # Ascending in l, with multiplicity 2l+1.
    counts: dict[int, int] = {}
    for d in degs:
        counts[d] = counts.get(d, 0) + 1
    for l, c in counts.items():
        assert c == 2 * l + 1


def test_self_addition_theorem():
    r"""For any unit ``x``, :math:`\sum_m (Y_l^m(x))^2 = (2l+1)/(4\pi)`."""
    xyz = _random_unit_vectors(8, seed=1)
    l_max = 6
    Y = real_spherical_harmonics(xyz, l_max)
    degs = jnp.asarray(harmonic_degrees(l_max))
    for l in range(l_max + 1):
        mask = degs == l
        block = jnp.sum(Y[:, mask] ** 2, axis=1)
        expected = (2 * l + 1) / (4 * math.pi)
        assert jnp.allclose(block, expected, atol=1e-5)


def test_pairwise_addition_theorem():
    r"""Cross-point: :math:`\sum_m Y_l^m(x) Y_l^m(y) = (2l+1)/(4\pi) P_l(x \cdot y)`."""
    xyz = _random_unit_vectors(5, seed=2)
    l_max = 5
    Y = real_spherical_harmonics(xyz, l_max)
    degs = jnp.asarray(harmonic_degrees(l_max))
    for i in range(5):
        for j in range(5):
            cos_g = float(jnp.dot(xyz[i], xyz[j]))
            for l in range(l_max + 1):
                mask = degs == l
                lhs = float(jnp.sum(Y[i, mask] * Y[j, mask]))
                rhs = (2 * l + 1) / (4 * math.pi) * legval(cos_g, [0] * l + [1])
                assert abs(lhs - rhs) < 1e-5, (i, j, l, lhs, rhs)


def _real_y_via_scipy(l: int, m: int, theta: float, phi: float) -> float:
    """Reference real SH built from `scipy.special.sph_harm_y` (complex SH).

    Convention: Condon-Shortley already in scipy's complex SH; real form
    drops to ``sqrt(2) Re`` (m>0) / ``sqrt(2) Im`` (m<0) without an extra
    ``(-1)^m`` wrap.
    """
    if m == 0:
        return float(complex(sph_harm_y(l, 0, theta, phi)).real)
    Yc = complex(sph_harm_y(l, abs(m), theta, phi))
    return math.sqrt(2.0) * (Yc.real if m > 0 else Yc.imag)


def test_matches_scipy_complex_sh():
    """Direct elementwise comparison against scipy's complex SH (all l, m, point)."""
    xyz = _random_unit_vectors(6, seed=42)
    l_max = 5
    Y_ours = np.asarray(real_spherical_harmonics(xyz, l_max))
    xyz_np = np.asarray(xyz)
    z = xyz_np[:, 2]
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    phi = np.arctan2(xyz_np[:, 1], xyz_np[:, 0])

    cols = []
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            cols.append(
                np.array(
                    [
                        _real_y_via_scipy(l, m, float(theta[n]), float(phi[n]))
                        for n in range(xyz_np.shape[0])
                    ]
                )
            )
    Y_scipy = np.stack(cols, axis=-1)
    # Float32 default in JAX — single-precision tolerance.
    np.testing.assert_allclose(Y_ours, Y_scipy, atol=1e-5)


def test_orthonormality_via_quadrature():
    """Sphere quadrature: Gram approximates I_M to within Riemann-sum bias.

    We use a regular product (theta, phi) grid with the standard sin(theta)
    Jacobian. Convergence is only first-order in 1/N — tolerance reflects
    that, the high-precision orthonormality check is the addition-theorem
    test plus :func:`test_matches_scipy_complex_sh`.
    """
    n_theta, n_phi = 200, 400
    theta = jnp.linspace(1e-4, math.pi - 1e-4, n_theta)
    phi = jnp.linspace(0.0, 2 * math.pi, n_phi, endpoint=False)
    tt, pp = jnp.meshgrid(theta, phi, indexing="ij")
    xyz = jnp.stack(
        [jnp.sin(tt) * jnp.cos(pp), jnp.sin(tt) * jnp.sin(pp), jnp.cos(tt)], axis=-1
    ).reshape(-1, 3)
    weights = (jnp.sin(tt) * (math.pi / n_theta) * (2 * math.pi / n_phi)).reshape(-1)

    l_max = 3
    Y = real_spherical_harmonics(xyz, l_max)  # (N, M)
    gram = (Y * weights[:, None]).T @ Y  # (M, M)
    # Off-diagonal must be small; diagonal can be off by ~1% from Riemann bias.
    off_diag_max = float(jnp.max(jnp.abs(gram - jnp.diag(jnp.diag(gram)))))
    diag_err_max = float(jnp.max(jnp.abs(jnp.diag(gram) - 1.0)))
    assert off_diag_max < 1e-3, off_diag_max
    assert diag_err_max < 2e-2, diag_err_max


def test_rejects_bad_l_max():
    with pytest.raises(ValueError, match="l_max"):
        real_spherical_harmonics(jnp.zeros((2, 3)), l_max=-1)


def test_rejects_wrong_input_shape():
    with pytest.raises(ValueError, match="unit_xyz"):
        real_spherical_harmonics(jnp.zeros((2, 4)), l_max=2)


def test_l_max_zero_returns_constant_column():
    xyz = _random_unit_vectors(4, seed=3)
    Y = real_spherical_harmonics(xyz, l_max=0)
    assert Y.shape == (4, 1)
    expected = 1.0 / math.sqrt(4 * math.pi)
    assert jnp.allclose(Y[:, 0], expected, atol=1e-12)
