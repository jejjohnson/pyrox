r"""Inter-domain inducing-feature families for sparse GPs.

An inducing *feature* generalizes an inducing *point*: instead of
$u_m = f(z_m)$ for a finite collection of pseudo-inputs, we take

$$
u_m = \langle f, \phi_m \rangle_{\mathcal{H}_k}
$$

for a basis $\{\phi_m\}$ of the kernel's RKHS. The payoff: when
$\{\phi_m\}$ is an eigenbasis of the (negative) Laplacian on the
input domain *and* the kernel is stationary, $K_{uu}$ becomes
diagonal — the bottleneck $M \times M$ solve degenerates to an
elementwise divide.

This module ships:

- `FourierInducingFeatures`     — VFF on the bounded box (Hensman
  et al. 2017)
- `SphericalHarmonicInducingFeatures` — VISH on the 2-sphere
  (Dutordoir et al. 2020)
- `LaplacianInducingFeatures`   — Laplacian eigenfeatures on a graph
- `DecoupledInducingFeatures`   — distinct mean / covariance bases
  (Cheng & Boots 2017)

All concretions implement the `InducingFeatures` protocol so that
`pyrox.gp.SparseGPPrior` can accept them in place of a raw ``Z``.

**Diagonal-structure invariant.** ``K_uu`` for the diagonal cases is
constructed via `lineax.DiagonalLinearOperator` and jitter is
folded into the diagonal vector — *never* added as ``jnp.eye``. This
preserves the structural dispatch in `gaussx.solve` /
`gaussx.cholesky`, which short-circuits diagonal operators to O(M)
elementwise ops. Test ``test_inducing_features.test_vff_k_uu_is_diagonal``
guards this end-to-end.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import einx
import equinox as eqx
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float, Int

from pyrox._basis import (
    SlepianCapBasis,
    fourier_basis,
    fourier_eigenvalues,
    graph_laplacian_eigpairs,
    harmonic_degrees,
    real_spherical_harmonics,
    slepian_cap_basis,
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
    `lineax.DiagonalLinearOperator` so the downstream solve dispatches
    to elementwise division.

    **Input shape is family-dependent.** ``k_ux`` takes a batch of data
    points ``X`` in whatever representation the family consumes:

    - `FourierInducingFeatures`: coordinates ``(N, D)``.
    - `SphericalHarmonicInducingFeatures`: unit vectors ``(N, 3)``.
    - `LaplacianInducingFeatures`: integer node indices ``(N,)``.

    Each implementation validates its own expected shape and dtype.
    """

    @property
    def num_features(self) -> int: ...

    def K_uu(
        self, kernel: Kernel, *, jitter: float = 1e-6
    ) -> lx.AbstractLinearOperator: ...

    def k_ux(self, x: Array, kernel: Kernel) -> Float[Array, "N M"]: ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_STATIONARY_KERNELS: tuple[type, ...] = (RBF, Matern)


def _is_stationary(kernel: Kernel) -> bool:
    """Whether ``kernel`` has a registered closed-form spectral density.

    Used as the structural-stationarity check for inducing features that
    derive ``K_uu`` from `pyrox._basis.spectral_density`. Conservative
    by design — kernels that *are* stationary but lack a registered
    spectral density (e.g. ``RationalQuadratic``) currently return ``False``.
    """
    return isinstance(kernel, _STATIONARY_KERNELS)


def _diagonal_with_jitter(
    diag: Float[Array, " M"], jitter: float
) -> lx.DiagonalLinearOperator:
    """Build a ``DiagonalLinearOperator`` with jitter folded into the vector.

    Critical for scalability: adding jitter via ``+ jnp.eye(M)`` would
    densify the operator and silently revert `gaussx.solve` to its
    O(M^3) fallback. Folding into the diagonal vector keeps dispatch in
    the elementwise-divide short-circuit.
    """
    return lx.DiagonalLinearOperator(diag + jitter)


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
    r"""VFF — Variational Fourier inducing features on $[-L, L]^D$.

    For a stationary kernel with spectral density $S(\cdot)$, the
    basis $\{\phi_j\}$ of Laplacian eigenfunctions on the box gives

    $$
    K_{uu} = \mathrm{diag}\!\big(S(\sqrt{\lambda_j})\big),
    \qquad
    K_{ux}(x)_j = S(\sqrt{\lambda_j})\,\phi_j(x).
    $$

    With this convention $K_{ux} K_{uu}^{-1} = \phi_j(x)$, so the
    SVGP predictive mean reduces to a basis evaluation. ``K_{uu}`` is
    returned as a `lineax.DiagonalLinearOperator` to preserve the
    O(M) solve dispatch end-to-end.

    Attributes:
        in_features: Input dimension $D$.
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
        """Diagonal $K_{uu}$ — entries ``S(sqrt(lambda_j))`` plus jitter."""
        self._check_stationary(kernel)
        with _kernel_context(kernel):
            lam = fourier_eigenvalues(self.num_basis_per_dim, self.L, self.in_features)
            S = spectral_density(kernel, lam, D=self.in_features)
        return _diagonal_with_jitter(S, jitter)

    def k_ux(self, x: Float[Array, "N D"], kernel: Kernel) -> Float[Array, "N M"]:
        """Cross-covariance entries $S(\\sqrt{\\lambda_j})\\,\\phi_j(x)$."""
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
    r"""Funk-Hecke coefficients of a zonal kernel on $S^2$.

    For a kernel of the form $k(x, x') = \kappa(x \cdot x')$ on the
    unit 2-sphere, the Funk-Hecke theorem gives:

    $$
    a_l = 2\pi \int_{-1}^{1} \kappa(t)\,P_l(t)\,dt.
    $$

    Returns ``(l_max + 1,)`` coefficients indexed by ``l``. We treat any
    Euclidean kernel as zonal-on-the-sphere via
    $\kappa(t) = k_{\mathrm{euc}}(\hat{n}_0, \hat{n}_t)$ for unit
    vectors at angular separation ``arccos(t)``.
    """
    # Gauss-Legendre quadrature nodes on [-1, 1] (host-side setup constants).
    t, w = _gauss_legendre_nodes(num_quadrature)
    # Build pairs of unit vectors: x0 = (0, 0, 1), x_t = (sin(arccos t), 0, t).
    sin_t = jnp.sqrt(jnp.maximum(1.0 - t**2, 0.0))
    n0 = jnp.array([0.0, 0.0, 1.0])
    nT = jnp.stack([sin_t, jnp.zeros_like(t), t], axis=-1)  # (Q, 3)
    # Single batched kernel call — stays on-device and keeps autodiff edges
    # to any hyperparameters sampled inside ``kernel``. Taking row 0 of the
    # ``(1, Q)`` Gram is O(Q), not O(Q^2).
    with _kernel_context(kernel):
        kt = kernel(n0[None, :], nT)[0]  # (Q,)
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
    r"""VISH — inducing harmonics on $S^2$ (Dutordoir et al. 2020).

    For any zonal kernel $k(x, x') = \kappa(x \cdot x')$ on the
    unit 2-sphere, the Funk-Hecke theorem gives a diagonal $K_{uu}$
    whose eigenvalues are the kernel's Funk-Hecke coefficients
    $a_l$. The cross-covariance is $a_l\,Y_{lm}(x)$.

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
        """Diagonal $K_{uu}$ — Funk-Hecke coefficients per harmonic."""
        diag = self._per_feature_coeffs(kernel)
        return _diagonal_with_jitter(diag, jitter)

    def k_ux(
        self,
        unit_xyz: Float[Array, "N 3"],
        kernel: Kernel,
    ) -> Float[Array, "N M"]:
        r"""Cross-covariance: $a_l\,Y_{lm}(x)$."""
        if unit_xyz.ndim != 2 or unit_xyz.shape[-1] != 3:
            raise ValueError(f"unit_xyz must be (N, 3); got {unit_xyz.shape}.")
        Y = real_spherical_harmonics(unit_xyz, self.l_max)
        a_per_feature = self._per_feature_coeffs(kernel)
        return Y * a_per_feature[None, :]


class SlepianInducingFeatures(eqx.Module):
    r"""Region-localized Slepian inducing features on $S^2$.

    The retained Slepian functions are linear combinations ``G = Y C`` of the
    real spherical-harmonic basis evaluated in the cap-centred frame. For a
    zonal kernel with Funk-Hecke coefficients ``a_l`` this gives dense
    inducing covariance ``K_uu = C.T diag(a_l) C`` and cross-covariance
    ``K_ux = Y(R x) diag(a_l) C``, where ``R`` is the rotation aligning the
    cap centre with the north pole. The basis (a `SlepianCapBasis`)
    is built once at `init` time and stored on the module so that
    ``K_uu``, ``k_ux`` and ``num_features`` are cheap matrix multiplies.
    """

    l_max: int = eqx.field(static=True)
    cap_radius_deg: float = eqx.field(static=True)
    cap_centre_lonlat_deg: tuple[float, float] = eqx.field(static=True)
    eig_threshold: float = eqx.field(static=True)
    n_modes: int | None = eqx.field(static=True)
    num_quadrature: int = eqx.field(static=True)
    basis_num_quadrature: int | None = eqx.field(static=True)
    basis: SlepianCapBasis

    @classmethod
    def init(
        cls,
        *,
        l_max: int,
        cap_radius_deg: float,
        cap_centre_lonlat_deg: tuple[float, float],
        eig_threshold: float = 0.05,
        n_modes: int | None = None,
        num_quadrature: int = 256,
        basis_num_quadrature: int | None = None,
    ) -> SlepianInducingFeatures:
        if l_max < 0:
            raise ValueError(f"l_max must be >= 0; got {l_max}.")
        if cap_radius_deg <= 0.0 or cap_radius_deg > 180.0:
            raise ValueError(
                f"cap_radius_deg must lie in (0, 180]; got {cap_radius_deg}."
            )
        if num_quadrature < 1:
            raise ValueError(f"num_quadrature must be >= 1; got {num_quadrature}.")
        if n_modes is not None and n_modes < 1:
            raise ValueError(f"n_modes must be >= 1; got {n_modes}.")
        if len(cap_centre_lonlat_deg) != 2:
            raise ValueError(
                "cap_centre_lonlat_deg must contain (lon, lat); "
                f"got {cap_centre_lonlat_deg}."
            )
        lon, lat = cap_centre_lonlat_deg
        # Build the basis once at construction so K_uu / k_ux are matrix
        # multiplies; cap geometry is static, so the eigensolve does not
        # rerun per call.
        basis = slepian_cap_basis(
            l_max,
            jnp.deg2rad(cap_radius_deg),
            n_modes=n_modes,
            eig_threshold=eig_threshold,
            lonlat_centre=jnp.deg2rad(jnp.asarray((float(lon), float(lat)))),
            num_quadrature=basis_num_quadrature,
        )
        return cls(
            l_max=l_max,
            cap_radius_deg=float(cap_radius_deg),
            cap_centre_lonlat_deg=(float(lon), float(lat)),
            eig_threshold=float(eig_threshold),
            n_modes=n_modes,
            num_quadrature=num_quadrature,
            basis_num_quadrature=basis_num_quadrature,
            basis=basis,
        )

    @property
    def num_features(self) -> int:
        return self.basis.num_modes

    def _per_feature_coeffs(self, kernel: Kernel) -> Float[Array, " M"]:
        a = funk_hecke_coefficients(
            kernel, self.l_max, num_quadrature=self.num_quadrature
        )
        return a[jnp.asarray(harmonic_degrees(self.l_max))]

    def K_uu(self, kernel: Kernel, *, jitter: float = 1e-6) -> lx.MatrixLinearOperator:
        """Dense Slepian inducing covariance with diagonal jitter."""
        a_per_feature = self._per_feature_coeffs(kernel)
        # Scale each feature row by its Funk-Hecke coefficient, then form the
        # Gram K = Φᵀ diag(a) Φ by contracting the feature axis f.
        weighted_coeffs = einx.multiply(
            "f m, f -> f m", self.basis.coeffs, a_per_feature
        )
        K = einx.dot("f i, f j -> i j", self.basis.coeffs, weighted_coeffs)
        K = K.at[jnp.diag_indices_from(K)].add(jitter)
        return lx.MatrixLinearOperator(K, lx.positive_semidefinite_tag)

    def k_ux(
        self,
        unit_xyz: Float[Array, "N 3"],
        kernel: Kernel,
    ) -> Float[Array, "N K"]:
        """Cross-covariance between unit-sphere inputs and Slepian features.

        Spherical harmonics are evaluated in the cap-centred frame to match
        `SlepianCapBasis.evaluate`; without this rotation, two
        ``SlepianInducingFeatures`` differing only in cap centre would
        produce the same cross-covariance.
        """
        if unit_xyz.ndim != 2 or unit_xyz.shape[-1] != 3:
            raise ValueError(f"unit_xyz must be (N, 3); got {unit_xyz.shape}.")
        centred = self.basis.centred_coordinates(unit_xyz)
        Y = real_spherical_harmonics(centred, self.l_max)
        a_per_feature = self._per_feature_coeffs(kernel)
        # Weight each harmonic by its Funk-Hecke coefficient, then project
        # onto the Slepian modes by contracting the feature axis f.
        Y_weighted = einx.multiply("n f, f -> n f", Y, a_per_feature)
        return einx.dot("n f, f m -> n m", Y_weighted, self.basis.coeffs)


# ---------------------------------------------------------------------------
# Graph Laplacian inducing features
# ---------------------------------------------------------------------------


class LaplacianInducingFeatures(eqx.Module):
    r"""Inducing features from low-frequency graph Laplacian eigenvectors.

    For a graph with normalized Laplacian $L$, take the smallest
    ``num_basis`` eigenpairs $(\mu_j, v_j)$. Treating the kernel as
    a function of the graph distance — specifically, applying the kernel
    *spectrum* $g(\mu)$ to the Laplacian eigenvalues — gives a
    diagonal $K_{uu}$.

    This implementation supports the *heat-kernel* family
    $g(\mu) = \exp(-\mu / (2 \ell^2))$ (matching `pyrox.gp.RBF`
    in spectrum) by reusing `pyrox._basis.spectral_density` with the
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

    def _check_stationary(self, kernel: Kernel) -> None:
        if not _is_stationary(kernel):
            raise ValueError(
                "LaplacianInducingFeatures requires a stationary kernel with a "
                f"registered spectral density; got {type(kernel).__name__}."
            )

    def K_uu(
        self, kernel: Kernel, *, jitter: float = 1e-6
    ) -> lx.DiagonalLinearOperator:
        self._check_stationary(kernel)
        with _kernel_context(kernel):
            S = spectral_density(kernel, self.eigvals, D=1)
        return _diagonal_with_jitter(S, jitter)

    def k_ux(
        self, node_indices: Int[Array, " N"], kernel: Kernel
    ) -> Float[Array, "N M"]:
        self._check_stationary(kernel)
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
        `InducingFeatures` (no single ``K_uu`` makes sense for two
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
    "SlepianInducingFeatures",
    "SphericalHarmonicInducingFeatures",
    "funk_hecke_coefficients",
]
