r"""Inter-domain inducing-feature families for sparse GPs.

An inducing *feature* generalizes an inducing *point*: instead of
:math:`u_m = f(z_m)` for a finite collection of pseudo-inputs, we take

.. math::

    u_m = \langle f, \phi_m \rangle_{\mathcal{H}_k}

for a basis :math:`\{\phi_m\}` of the kernel's RKHS. The payoff: when
:math:`\{\phi_m\}` is an eigenbasis of the (negative) Laplacian on the
input domain *and* the kernel is stationary, :math:`K_{uu}` becomes
diagonal — the bottleneck :math:`M \times M` solve degenerates to an
elementwise divide.

This module ships:

- :class:`FourierInducingFeatures`     — VFF on the bounded box (Hensman
  et al. 2017)
- :class:`SphericalHarmonicInducingFeatures` — VISH on the 2-sphere
  (Dutordoir et al. 2020)
- :class:`LaplacianInducingFeatures`   — Laplacian eigenfeatures on a graph
- :class:`DecoupledInducingFeatures`   — distinct mean / covariance bases
  (Cheng & Boots 2017)

All concretions implement the :class:`InducingFeatures` protocol so that
:class:`pyrox.gp.SparseGPPrior` can accept them in place of a raw ``Z``.

**Diagonal-structure invariant.** ``K_uu`` for the diagonal cases is
constructed via :class:`lineax.DiagonalLinearOperator` and jitter is
folded into the diagonal vector — *never* added as ``jnp.eye``. This
preserves the structural dispatch in :mod:`gaussx.solve` /
:mod:`gaussx.cholesky`, which short-circuits diagonal operators to O(M)
elementwise ops. Test ``test_inducing_features.test_vff_k_uu_is_diagonal``
guards this end-to-end.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import equinox as eqx
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float, Int

from pyrox._basis import (
    fourier_basis,
    fourier_eigenvalues,
    graph_laplacian_eigpairs,
    real_spherical_harmonics,
    spectral_density,
)
from pyrox.gp._context import _kernel_context
from pyrox.gp._kernels import RBF, Matern
from pyrox.gp._protocols import Kernel


@runtime_checkable
class InducingFeatures(Protocol):
    """Protocol for inter-domain inducing features.

    Implementations expose the inducing-prior covariance ``K_uu`` and the
    cross-covariance ``k_ux(X)`` between data points and inducing
    features. Diagonal-friendly concretions return
    :class:`lineax.DiagonalLinearOperator` so the downstream solve dispatches
    to elementwise division.
    """

    @property
    def num_features(self) -> int: ...

    def K_uu(
        self, kernel: Kernel, *, jitter: float = 1e-6
    ) -> lx.AbstractLinearOperator: ...

    def k_ux(self, x: Float[Array, "N D"], kernel: Kernel) -> Float[Array, "N M"]: ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_STATIONARY_KERNELS: tuple[type, ...] = (RBF, Matern)


def _is_stationary(kernel: Kernel) -> bool:
    """Whether ``kernel`` has a registered closed-form spectral density.

    Used as the structural-stationarity check for inducing features that
    derive ``K_uu`` from :func:`pyrox._basis.spectral_density`. Conservative
    by design — kernels that *are* stationary but lack a registered
    spectral density (e.g. ``RationalQuadratic``) currently return ``False``.
    """
    return isinstance(kernel, _STATIONARY_KERNELS)


def _diagonal_with_jitter(
    diag: Float[Array, " M"], jitter: float
) -> lx.DiagonalLinearOperator:
    """Build a ``DiagonalLinearOperator`` with jitter folded into the vector.

    Critical for scalability: adding jitter via ``+ jnp.eye(M)`` would
    densify the operator and silently revert :mod:`gaussx.solve` to its
    O(M^3) fallback. Folding into the diagonal vector keeps dispatch in
    the elementwise-divide short-circuit.
    """
    return lx.DiagonalLinearOperator(diag + jitter)  # ty: ignore[invalid-return-type]


# ---------------------------------------------------------------------------
# Fourier inducing features (VFF / HSGP-equivalent)
# ---------------------------------------------------------------------------


def _to_tuple(value: int | float | tuple, D: int, name: str) -> tuple:
    if isinstance(value, tuple | list):
        out = tuple(value)
        if len(out) != D:
            raise ValueError(
                f"{name} must have length {D} (one per input dim); got {len(out)}."
            )
        return out
    return (value,) * D


class FourierInducingFeatures(eqx.Module):
    r"""VFF — Variational Fourier inducing features on :math:`[-L, L]^D`.

    For a stationary kernel with spectral density :math:`S(\cdot)`, the
    basis :math:`\{\phi_j\}` of Laplacian eigenfunctions on the box gives

    .. math::

        K_{uu} = \mathrm{diag}\!\big(S(\sqrt{\lambda_j})\big),
        \qquad
        K_{ux}(x)_j = S(\sqrt{\lambda_j})\,\phi_j(x).

    With this convention :math:`K_{ux} K_{uu}^{-1} = \phi_j(x)`, so the
    SVGP predictive mean reduces to a basis evaluation. ``K_{uu}`` is
    returned as a :class:`lineax.DiagonalLinearOperator` to preserve the
    O(M) solve dispatch end-to-end.

    Attributes:
        in_features: Input dimension :math:`D`.
        num_basis_per_dim: Per-axis number of 1D eigenfunctions; total
            count is ``prod(num_basis_per_dim)``.
        L: Per-axis box half-width.
    """

    in_features: int = eqx.field(static=True)
    num_basis_per_dim: tuple[int, ...] = eqx.field(static=True)
    L: tuple[float, ...] = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        in_features: int,
        num_basis_per_dim: int | tuple[int, ...],
        L: float | tuple[float, ...],
    ) -> FourierInducingFeatures:
        M_per = _to_tuple(num_basis_per_dim, in_features, "num_basis_per_dim")
        L_per = _to_tuple(L, in_features, "L")
        if any(L_d <= 0 for L_d in L_per):
            raise ValueError(f"L must be all positive; got {L_per}.")
        if any(M_d < 1 for M_d in M_per):
            raise ValueError(f"num_basis_per_dim must be all >= 1; got {M_per}.")
        return cls(
            in_features=in_features,
            num_basis_per_dim=M_per,
            L=tuple(float(L_d) for L_d in L_per),
        )

    @property
    def num_features(self) -> int:
        n = 1
        for m in self.num_basis_per_dim:
            n *= m
        return n

    def _check_stationary(self, kernel: Kernel) -> None:
        if not _is_stationary(kernel):
            raise ValueError(
                f"FourierInducingFeatures requires a stationary kernel with a "
                f"registered spectral density (RBF or Matern); got "
                f"{type(kernel).__name__}."
            )

    def K_uu(
        self, kernel: Kernel, *, jitter: float = 1e-6
    ) -> lx.DiagonalLinearOperator:
        """Diagonal :math:`K_{uu}` — entries ``S(sqrt(lambda_j))`` plus jitter."""
        self._check_stationary(kernel)
        with _kernel_context(kernel):
            lam = fourier_eigenvalues(self.num_basis_per_dim, self.L, self.in_features)
            S = spectral_density(kernel, lam, D=self.in_features)
        return _diagonal_with_jitter(S, jitter)

    def k_ux(self, x: Float[Array, "N D"], kernel: Kernel) -> Float[Array, "N M"]:
        """Cross-covariance entries :math:`S(\\sqrt{\\lambda_j})\\,\\phi_j(x)`."""
        self._check_stationary(kernel)
        if x.ndim != 2 or x.shape[-1] != self.in_features:
            raise ValueError(f"x must be (N, {self.in_features}); got shape {x.shape}.")
        with _kernel_context(kernel):
            Phi, lam = fourier_basis(x, self.num_basis_per_dim, self.L)
            S = spectral_density(kernel, lam, D=self.in_features)
        return Phi * S[None, :]


# ---------------------------------------------------------------------------
# Spherical harmonic inducing features (VISH)
# ---------------------------------------------------------------------------


def funk_hecke_coefficients(
    kernel: Kernel,
    l_max: int,
    *,
    num_quadrature: int = 256,
) -> Float[Array, " l_max_plus_1"]:
    r"""Funk-Hecke coefficients of a zonal kernel on :math:`S^2`.

    For a kernel of the form :math:`k(x, x') = \kappa(x \cdot x')` on the
    unit 2-sphere, the Funk-Hecke theorem gives:

    .. math::

        a_l = 2\pi \int_{-1}^{1} \kappa(t)\,P_l(t)\,dt.

    Returns ``(l_max + 1,)`` coefficients indexed by ``l``. We treat any
    Euclidean kernel as zonal-on-the-sphere via
    :math:`\kappa(t) = k_{\mathrm{euc}}(\hat{n}_0, \hat{n}_t)` for unit
    vectors at angular separation ``arccos(t)``.
    """
    # Gauss-Legendre quadrature on [-1, 1].
    t, w = jnp.asarray(_gauss_legendre_nodes(num_quadrature))
    # Build pairs of unit vectors: x0 = (0, 0, 1), x_t = (sin(arccos t), 0, t).
    sin_t = jnp.sqrt(jnp.maximum(1.0 - t**2, 0.0))
    n0 = jnp.array([0.0, 0.0, 1.0])
    nT = jnp.stack([sin_t, jnp.zeros_like(t), t], axis=-1)  # (Q, 3)
    with _kernel_context(kernel):
        kt = jnp.array(
            [
                float(kernel(n0[None, :], nT[i : i + 1, :])[0, 0])
                for i in range(num_quadrature)
            ]
        )
    # Evaluate P_l(t) for l = 0, ..., l_max via three-term recurrence.
    P_lm1 = jnp.ones_like(t)  # P_0
    P_l = t  # P_1
    coeffs = [2.0 * jnp.pi * jnp.sum(w * kt)]  # a_0 = 2pi * int kt * 1 dt
    if l_max >= 1:
        coeffs.append(2.0 * jnp.pi * jnp.sum(w * kt * P_l))  # a_1
    for ell in range(2, l_max + 1):
        P_lp1 = ((2 * ell - 1) * t * P_l - (ell - 1) * P_lm1) / ell
        coeffs.append(2.0 * jnp.pi * jnp.sum(w * kt * P_lp1))
        P_lm1, P_l = P_l, P_lp1
    return jnp.stack(coeffs, axis=0)


def _gauss_legendre_nodes(n: int) -> tuple[Float[Array, " n"], Float[Array, " n"]]:
    """Numpy Gauss-Legendre nodes/weights on [-1, 1] (used at construction time)."""
    import numpy as np

    nodes, weights = np.polynomial.legendre.leggauss(n)
    return jnp.asarray(nodes), jnp.asarray(weights)


class SphericalHarmonicInducingFeatures(eqx.Module):
    r"""VISH — inducing harmonics on :math:`S^2` (Dutordoir et al. 2020).

    For any zonal kernel :math:`k(x, x') = \kappa(x \cdot x')` on the
    unit 2-sphere, the Funk-Hecke theorem gives a diagonal :math:`K_{uu}`
    whose eigenvalues are the kernel's Funk-Hecke coefficients
    :math:`a_l`. The cross-covariance is :math:`a_l\,Y_{lm}(x)`.

    Funk-Hecke coefficients are computed by Gauss-Legendre quadrature
    (arbitrary kernels supported, no closed form required). For
    kernels that have a closed-form Funk-Hecke series (RBF on S² via
    Bessel functions etc.), the numerical and analytic answers should
    agree to the quadrature tolerance.

    Attributes:
        l_max: Maximum harmonic degree, inclusive.
        num_quadrature: Gauss-Legendre nodes for the Funk-Hecke integral.
    """

    l_max: int = eqx.field(static=True)
    num_quadrature: int = eqx.field(static=True, default=256)

    @classmethod
    def init(
        cls, l_max: int, *, num_quadrature: int = 256
    ) -> SphericalHarmonicInducingFeatures:
        if l_max < 0:
            raise ValueError(f"l_max must be >= 0; got {l_max}.")
        if num_quadrature < 1:
            raise ValueError(f"num_quadrature must be >= 1; got {num_quadrature}.")
        return cls(l_max=l_max, num_quadrature=num_quadrature)

    @property
    def num_features(self) -> int:
        return (self.l_max + 1) ** 2

    def _per_feature_coeffs(self, kernel: Kernel) -> Float[Array, " M"]:
        a = funk_hecke_coefficients(
            kernel, self.l_max, num_quadrature=self.num_quadrature
        )
        # Each l contributes 2l+1 features with the same coefficient.
        return jnp.concatenate(
            [jnp.full((2 * ell + 1,), a[ell]) for ell in range(self.l_max + 1)]
        )

    def K_uu(
        self, kernel: Kernel, *, jitter: float = 1e-6
    ) -> lx.DiagonalLinearOperator:
        """Diagonal :math:`K_{uu}` — Funk-Hecke coefficients per harmonic."""
        diag = self._per_feature_coeffs(kernel)
        return _diagonal_with_jitter(diag, jitter)

    def k_ux(
        self,
        unit_xyz: Float[Array, "N 3"],
        kernel: Kernel,
    ) -> Float[Array, "N M"]:
        r"""Cross-covariance: :math:`a_l\,Y_{lm}(x)`."""
        if unit_xyz.ndim != 2 or unit_xyz.shape[-1] != 3:
            raise ValueError(f"unit_xyz must be (N, 3); got {unit_xyz.shape}.")
        Y = real_spherical_harmonics(unit_xyz, self.l_max)
        a_per_feature = self._per_feature_coeffs(kernel)
        return Y * a_per_feature[None, :]


# ---------------------------------------------------------------------------
# Graph Laplacian inducing features
# ---------------------------------------------------------------------------


class LaplacianInducingFeatures(eqx.Module):
    r"""Inducing features from low-frequency graph Laplacian eigenvectors.

    For a graph with normalized Laplacian :math:`L`, take the smallest
    ``num_basis`` eigenpairs :math:`(\mu_j, v_j)`. Treating the kernel as
    a function of the graph distance — specifically, applying the kernel
    *spectrum* :math:`g(\mu)` to the Laplacian eigenvalues — gives a
    diagonal :math:`K_{uu}`.

    This implementation supports the *heat-kernel* family
    :math:`g(\mu) = \exp(-\mu / (2 \ell^2))` (matching :class:`pyrox.gp.RBF`
    in spectrum) by reusing :func:`pyrox._basis.spectral_density` with the
    eigenvalues as input.

    Attributes:
        eigvals: ``(M,)`` Laplacian eigenvalues.
        eigvecs: ``(V, M)`` Laplacian eigenvectors.
        num_quadrature: Unused (kept for protocol uniformity).

    Note:
        ``X`` is a vector of *node indices* (integer-valued), not
        coordinates. The returned cross-covariance gathers the relevant
        rows of ``eigvecs``.
    """

    eigvals: Float[Array, " M"]
    eigvecs: Float[Array, "V M"]

    @classmethod
    def fit(
        cls,
        adjacency: Float[Array, "V V"],
        num_basis: int,
        *,
        normalized: bool = True,
    ) -> LaplacianInducingFeatures:
        eigvals, eigvecs = graph_laplacian_eigpairs(
            adjacency, num_basis, normalized=normalized
        )
        return cls(eigvals=eigvals, eigvecs=eigvecs)

    @property
    def num_features(self) -> int:
        return int(self.eigvals.shape[0])

    def K_uu(
        self, kernel: Kernel, *, jitter: float = 1e-6
    ) -> lx.DiagonalLinearOperator:
        if not _is_stationary(kernel):
            raise ValueError(
                f"LaplacianInducingFeatures requires a stationary kernel with a "
                f"registered spectral density; got {type(kernel).__name__}."
            )
        with _kernel_context(kernel):
            S = spectral_density(kernel, self.eigvals, D=1)
        return _diagonal_with_jitter(S, jitter)

    def k_ux(
        self, node_indices: Int[Array, " N"], kernel: Kernel
    ) -> Float[Array, "N M"]:
        if node_indices.ndim != 1:
            raise ValueError(
                "node_indices must be a 1D integer array; got shape "
                f"{node_indices.shape}."
            )
        with _kernel_context(kernel):
            S = spectral_density(kernel, self.eigvals, D=1)
        rows = self.eigvecs[node_indices]
        return rows * S[None, :]


# ---------------------------------------------------------------------------
# Decoupled inducing features (Cheng & Boots 2017)
# ---------------------------------------------------------------------------


class DecoupledInducingFeatures(eqx.Module):
    r"""Decoupled mean / covariance inducing-feature bases (Cheng & Boots 2017).

    Two independent inducing-feature sets:

    - ``mean_features``: a large ``alpha``-basis used by the SVGP
      posterior *mean* (cheap — predictive mean cost is linear in the
      mean-basis size).
    - ``cov_features``: a small ``beta``-basis used for the posterior
      *covariance* (the true bottleneck; keep this small).

    The two bases need not share the same family — a common pattern is a
    large Fourier basis for the mean and a small spherical-harmonic
    basis for the covariance, or vice versa. The downstream guide
    consumes both via the standard SVGP machinery.

    Attributes:
        mean_features: Inducing-feature object backing the predictive mean.
        cov_features: Inducing-feature object backing the predictive covariance.

    Note:
        ``DecoupledInducingFeatures`` itself does *not* implement
        :class:`InducingFeatures` (no single ``K_uu`` makes sense for two
        bases). Consumers should access ``.mean_features`` and
        ``.cov_features`` directly.
    """

    mean_features: InducingFeatures
    cov_features: InducingFeatures

    @property
    def num_mean_features(self) -> int:
        return self.mean_features.num_features

    @property
    def num_cov_features(self) -> int:
        return self.cov_features.num_features


__all__ = [
    "DecoupledInducingFeatures",
    "FourierInducingFeatures",
    "InducingFeatures",
    "LaplacianInducingFeatures",
    "SphericalHarmonicInducingFeatures",
    "funk_hecke_coefficients",
]
