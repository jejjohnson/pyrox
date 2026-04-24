"""Multi-output Gaussian-process kernels and inducing structures.

These classes keep output mixing explicit instead of hiding it inside a
specialized model. Independent latent kernels stay reusable, while the
multi-output surface exposes the coregionalization matrices and the
Kronecker-style factors that later solvers can exploit.

The ``*_operator`` methods return structure-preserving
:class:`lineax.AbstractLinearOperator` objects built from ``gaussx``
primitives (:class:`gaussx.Kronecker`, :class:`gaussx.SumOperator`,
:class:`gaussx.BlockDiag`) so downstream solvers and log-determinant
strategies can exploit the block / Kronecker structure. Each method also
exposes a dense counterpart (``cross_covariance``, ``K_uu``, ...) for
users who want the materialized matrix.
"""

from __future__ import annotations

import functools

import equinox as eqx
import jax.numpy as jnp
import lineax as lx
from gaussx import BlockDiag, Kronecker, SumOperator, oilmm_back_project, oilmm_project
from jaxtyping import Array, Float

from pyrox.gp._context import _kernel_context, _kernel_contexts
from pyrox.gp._protocols import Kernel


def _validate_mixing(mixing: Float[Array, "P Q"]) -> None:
    if mixing.ndim != 2:
        raise ValueError("mixing must have shape (num_outputs, num_latents).")


def _validate_kernel_count(kernels: tuple[Kernel, ...], num_latents: int) -> None:
    if not kernels:
        raise ValueError("kernels must contain at least one latent kernel.")
    if len(kernels) != num_latents:
        raise ValueError(
            "kernels must contain exactly one kernel per latent process; "
            f"got {len(kernels)} kernels for {num_latents} latent processes."
        )


def _psd_matrix_op(M: Float[Array, "R R"]) -> lx.MatrixLinearOperator:
    """Wrap a square PSD matrix as a tagged lineax operator."""
    return lx.MatrixLinearOperator(M, tags=lx.positive_semidefinite_tag)  # ty: ignore[invalid-return-type]


def _matrix_op(M: Float[Array, "R C"]) -> lx.MatrixLinearOperator:
    """Wrap a (possibly rectangular) matrix as an untagged lineax operator.

    Used for cross-covariance blocks where the ``positive_semidefinite``
    tag would be wrong and trigger a ``lineax`` shape check against
    symmetry.
    """
    return lx.MatrixLinearOperator(M)  # ty: ignore[invalid-return-type]


def _kron_block_op(
    B: Float[Array, "P P"],
    K: Float[Array, "R C"],
    *,
    psd_K: bool,
) -> lx.MatrixLinearOperator:
    """Wrap the per-latent Kronecker factors with appropriate tags.

    ``B`` is always square PSD. The caller must declare whether ``K`` is
    PSD via ``psd_K`` — true only for the train-train Gram (``X1 is X2``).
    A square but non-symmetric cross-covariance (``N1 == N2`` with
    ``X1 != X2``) is *not* PSD, so shape alone is not a safe inference.
    """
    B_op = _psd_matrix_op(B)
    K_op = _psd_matrix_op(K) if psd_K else _matrix_op(K)
    return Kronecker(B_op, K_op)  # ty: ignore[invalid-return-type]


class LMCKernel(eqx.Module):
    """Linear model of coregionalization for vector-valued GPs.

    Each output is a linear combination of latent scalar GPs:
    ``f_p(x) = sum_q W[p, q] g_q(x)``. The cross-output covariance is
    ``Cov[f_p(x), f_{p'}(x')] = sum_q (w_q w_q^T)[p, p'] k_q(x, x')``.
    """

    kernels: tuple[Kernel, ...]
    mixing: Float[Array, "P Q"]

    def __check_init__(self) -> None:
        _validate_mixing(self.mixing)
        _validate_kernel_count(self.kernels, self.mixing.shape[1])

    @property
    def num_outputs(self) -> int:
        """Number of observed output channels ``P``."""
        return self.mixing.shape[0]

    @property
    def num_latents(self) -> int:
        """Number of latent scalar GPs ``Q``."""
        return self.mixing.shape[1]

    def coregionalization_matrix(self, q: int) -> Float[Array, "P P"]:
        """Return the rank-1 coregionalization matrix ``w_q w_q^T``."""
        if not 0 <= q < self.num_latents:
            raise IndexError(f"latent index out of range: {q}")
        column = self.mixing[:, q]
        return jnp.outer(column, column)

    def kronecker_factors(
        self,
        X1: Float[Array, "N1 D"],
        X2: Float[Array, "N2 D"],
    ) -> tuple[tuple[Float[Array, "P P"], Float[Array, "N1 N2"]], ...]:
        """Return ``(B_q, K_q(X1, X2))`` factors for each latent process.

        All latent kernel evaluations share one per-call context per
        unique kernel instance, so reusing the same kernel across latents
        (for hyperparameter tying) registers each sample site exactly once.
        """
        with _kernel_contexts(self.kernels):
            return tuple(
                (self.coregionalization_matrix(q), kernel(X1, X2))
                for q, kernel in enumerate(self.kernels)
            )

    def output_covariance(
        self,
        X1: Float[Array, "N1 D"],
        X2: Float[Array, "N2 D"],
    ) -> Float[Array, "P P N1 N2"]:
        """Return output-pair covariance blocks with shape ``(P, P, N1, N2)``."""
        terms = [
            B_q[:, :, None, None] * K_q[None, None, :, :]
            for B_q, K_q in self.kronecker_factors(X1, X2)
        ]
        return functools.reduce(jnp.add, terms)

    def cross_covariance_operator(
        self,
        X1: Float[Array, "N1 D"],
        X2: Float[Array, "N2 D"],
    ) -> lx.AbstractLinearOperator:
        """Return ``Cov[vec(F(X1)), vec(F(X2))]`` as a sum-of-Kroneckers operator.

        The returned operator preserves the per-latent Kronecker structure
        so structure-aware solvers can avoid materializing the full
        ``(P*N1, P*N2)`` matrix.
        """
        psd_K = X1 is X2
        terms = [
            _kron_block_op(B_q, K_q, psd_K=psd_K)
            for B_q, K_q in self.kronecker_factors(X1, X2)
        ]
        if len(terms) == 1:
            return terms[0]
        return SumOperator(*terms)  # ty: ignore[invalid-return-type]

    def cross_covariance(
        self,
        X1: Float[Array, "N1 D"],
        X2: Float[Array, "N2 D"],
    ) -> Float[Array, "PN1 PN2"]:
        """Return the dense covariance of ``vec(F(X1))`` and ``vec(F(X2))``."""
        return self.cross_covariance_operator(X1, X2).as_matrix()

    def full_covariance_operator(
        self, X: Float[Array, "N D"]
    ) -> lx.AbstractLinearOperator:
        """Return the Gram operator for isotopic multi-output observations."""
        return self.cross_covariance_operator(X, X)

    def full_covariance(self, X: Float[Array, "N D"]) -> Float[Array, "PN PN"]:
        """Return the dense Gram matrix for isotopic multi-output observations."""
        return self.full_covariance_operator(X).as_matrix()

    def diag(self, X: Float[Array, "N D"]) -> Float[Array, "N P"]:
        """Return per-input, per-output marginal variances with shape ``(N, P)``."""
        with _kernel_contexts(self.kernels):
            terms = [
                kernel.diag(X)[:, None] * jnp.square(self.mixing[:, q])[None, :]
                for q, kernel in enumerate(self.kernels)
            ]
        return functools.reduce(jnp.add, terms)


class ICMKernel(eqx.Module):
    """Intrinsic coregionalization model with one shared latent kernel.

    The cross-output covariance is ``kron(B, k(X1, X2))`` with
    ``B = W W^T + diag(kappa)``. When ``kappa is None`` the extra diagonal
    term is omitted.
    """

    kernel: Kernel
    mixing: Float[Array, "P Q"]
    kappa: Float[Array, " P"] | None = None

    def __check_init__(self) -> None:
        _validate_mixing(self.mixing)
        if self.kappa is not None and self.kappa.shape != (self.mixing.shape[0],):
            raise ValueError(
                "kappa must have shape (num_outputs,) when provided; "
                f"got {self.kappa.shape}."
            )

    @property
    def num_outputs(self) -> int:
        """Number of observed output channels ``P``."""
        return self.mixing.shape[0]

    @property
    def num_latents(self) -> int:
        """Number of latent scalar GPs ``Q``."""
        return self.mixing.shape[1]

    def coregionalization_matrix(self) -> Float[Array, "P P"]:
        """Return ``B = W W^T + diag(kappa)``."""
        B = self.mixing @ self.mixing.T
        if self.kappa is None:
            return B
        return B + jnp.diag(self.kappa)

    def kronecker_factors(
        self,
        X1: Float[Array, "N1 D"],
        X2: Float[Array, "N2 D"],
    ) -> tuple[Float[Array, "P P"], Float[Array, "N1 N2"]]:
        """Return the shared ``(B, K(X1, X2))`` Kronecker factors."""
        with _kernel_context(self.kernel):
            return self.coregionalization_matrix(), self.kernel(X1, X2)

    def output_covariance(
        self,
        X1: Float[Array, "N1 D"],
        X2: Float[Array, "N2 D"],
    ) -> Float[Array, "P P N1 N2"]:
        """Return output-pair covariance blocks with shape ``(P, P, N1, N2)``."""
        B, K = self.kronecker_factors(X1, X2)
        return B[:, :, None, None] * K[None, None, :, :]

    def cross_covariance_operator(
        self,
        X1: Float[Array, "N1 D"],
        X2: Float[Array, "N2 D"],
    ) -> Kronecker:
        """Return ``Cov[vec(F(X1)), vec(F(X2))]`` as a ``Kronecker`` operator.

        The ``kron(B, K)`` structure lets downstream solvers apply
        :func:`gaussx.kronecker_mll` and related Kronecker-exact routines
        instead of materializing a ``(P*N1, P*N2)`` matrix.
        """
        B, K = self.kronecker_factors(X1, X2)
        return _kron_block_op(B, K, psd_K=X1 is X2)  # ty: ignore[invalid-return-type]

    def cross_covariance(
        self,
        X1: Float[Array, "N1 D"],
        X2: Float[Array, "N2 D"],
    ) -> Float[Array, "PN1 PN2"]:
        """Return the dense covariance of ``vec(F(X1))`` and ``vec(F(X2))``."""
        return self.cross_covariance_operator(X1, X2).as_matrix()

    def full_covariance_operator(self, X: Float[Array, "N D"]) -> Kronecker:
        """Return the Gram operator for isotopic multi-output observations."""
        return self.cross_covariance_operator(X, X)

    def full_covariance(self, X: Float[Array, "N D"]) -> Float[Array, "PN PN"]:
        """Return the dense Gram matrix for isotopic multi-output observations."""
        return self.full_covariance_operator(X).as_matrix()

    def diag(self, X: Float[Array, "N D"]) -> Float[Array, "N P"]:
        """Return per-input, per-output marginal variances with shape ``(N, P)``."""
        with _kernel_context(self.kernel):
            return (
                self.kernel.diag(X)[:, None]
                * jnp.diag(self.coregionalization_matrix())[None, :]
            )


class OILMMKernel(eqx.Module):
    """Orthogonal instantaneous linear mixing model.

    The latent GP kernels stay independent. Orthogonal mixing makes it
    possible to project observations into latent space and run ``Q`` scalar
    GP problems instead of one monolithic multi-output solve. Observation
    noise lives in a separate :class:`pyrox.gp.Likelihood`; this class
    returns noise-free signal covariance, matching the LMC/ICM convention.

    Pass ``check_orthogonal=True`` to verify ``W^T W ≈ I`` at
    construction — useful as a defensive check when ``W`` comes from
    an external computation that may drift off the Stiefel manifold.
    """

    kernels: tuple[Kernel, ...]
    mixing: Float[Array, "P Q"]
    check_orthogonal: bool = eqx.field(static=True, default=False)

    def __check_init__(self) -> None:
        _validate_mixing(self.mixing)
        _validate_kernel_count(self.kernels, self.mixing.shape[1])
        if self.mixing.shape[1] > self.mixing.shape[0]:
            raise ValueError(
                "orthogonal mixing requires num_latents <= num_outputs to form "
                "a valid semi-orthogonal mixing matrix; "
                f"got {self.mixing.shape[1]} latents and "
                f"{self.mixing.shape[0]} outputs."
            )
        if self.check_orthogonal and not self.is_orthogonal():
            raise ValueError(
                "mixing must satisfy W^T W ≈ I when check_orthogonal=True. "
                "Project via `jnp.linalg.qr(W)[0]` before construction, or "
                "pass check_orthogonal=False to bypass."
            )

    @property
    def num_outputs(self) -> int:
        """Number of observed output channels ``P``."""
        return self.mixing.shape[0]

    @property
    def num_latents(self) -> int:
        """Number of latent scalar GPs ``Q``."""
        return self.mixing.shape[1]

    def is_orthogonal(self, *, atol: float = 1e-6, rtol: float = 1e-6) -> bool:
        """Whether the current mixing matrix satisfies ``W^T W ≈ I``.

        Returns a Python ``bool`` via a host sync; not usable inside
        ``jax.jit`` / ``jax.vmap``.
        """
        gram = self.mixing.T @ self.mixing
        eye = jnp.eye(self.num_latents, dtype=self.mixing.dtype)
        return bool(jnp.allclose(gram, eye, atol=atol, rtol=rtol))

    def project(
        self,
        Y: Float[Array, "N P"],
        noise_var: Float[Array, " P"] | float,
    ) -> tuple[Float[Array, "N Q"], Float[Array, " Q"]]:
        """Project observations to latent space + per-latent noise variances.

        Delegates to :func:`gaussx.oilmm_project`. Returns
        ``(Y_latent, noise_latent)`` with shapes ``(N, Q)`` and ``(Q,)``;
        the per-latent noise is ``noise_latent = (W**2).T @ noise_var``.
        """
        if Y.ndim != 2 or Y.shape[1] != self.num_outputs:
            raise ValueError(
                f"Y must have shape (N, {self.num_outputs}); got {Y.shape}."
            )
        return oilmm_project(Y, self.mixing, noise_var)

    def back_project(
        self,
        f_means: Float[Array, "N Q"],
        f_vars: Float[Array, "N Q"],
    ) -> tuple[Float[Array, "N P"], Float[Array, "N P"]]:
        """Back-project latent GP predictive ``(means, vars)`` to output space.

        Delegates to :func:`gaussx.oilmm_back_project`.
        """
        return oilmm_back_project(f_means, f_vars, self.mixing)

    def independent_gps(self) -> tuple[Kernel, ...]:
        """Return the latent scalar GP kernels used after projection."""
        return self.kernels

    def signal_factors(
        self,
        X1: Float[Array, "N1 D"],
        X2: Float[Array, "N2 D"],
    ) -> tuple[tuple[Float[Array, "P P"], Float[Array, "N1 N2"]], ...]:
        """Return the latent signal factors before any observation noise.

        All latent kernel evaluations share one per-call context per
        unique kernel instance, mirroring :meth:`LMCKernel.kronecker_factors`.
        """
        with _kernel_contexts(self.kernels):
            return tuple(
                (
                    jnp.outer(self.mixing[:, q], self.mixing[:, q]),
                    kernel(X1, X2),
                )
                for q, kernel in enumerate(self.kernels)
            )

    def signal_covariance_operator(
        self,
        X1: Float[Array, "N1 D"],
        X2: Float[Array, "N2 D"],
    ) -> lx.AbstractLinearOperator:
        """Return the noise-free signal covariance as a structured operator."""
        psd_K = X1 is X2
        terms = [
            _kron_block_op(B_q, K_q, psd_K=psd_K)
            for B_q, K_q in self.signal_factors(X1, X2)
        ]
        if len(terms) == 1:
            return terms[0]
        return SumOperator(*terms)  # ty: ignore[invalid-return-type]

    def signal_covariance(
        self,
        X1: Float[Array, "N1 D"],
        X2: Float[Array, "N2 D"],
    ) -> Float[Array, "PN1 PN2"]:
        """Return the dense noise-free signal covariance matrix."""
        return self.signal_covariance_operator(X1, X2).as_matrix()

    # LMC / ICM-parity aliases — OILMM's "signal covariance" is the
    # noise-free vec-cross covariance, same semantics as
    # ``LMCKernel.cross_covariance`` since noise is no longer kernel-side.
    cross_covariance_operator = signal_covariance_operator
    cross_covariance = signal_covariance

    def full_covariance_operator(
        self, X: Float[Array, "N D"]
    ) -> lx.AbstractLinearOperator:
        """Return the noise-free Gram operator for isotopic observations."""
        return self.signal_covariance_operator(X, X)

    def full_covariance(self, X: Float[Array, "N D"]) -> Float[Array, "PN PN"]:
        """Return the dense noise-free Gram matrix."""
        return self.full_covariance_operator(X).as_matrix()

    def diag(self, X: Float[Array, "N D"]) -> Float[Array, "N P"]:
        """Return per-input, per-output marginal signal variances."""
        with _kernel_contexts(self.kernels):
            terms = [
                kernel.diag(X)[:, None] * jnp.square(self.mixing[:, q])[None, :]
                for q, kernel in enumerate(self.kernels)
            ]
        return functools.reduce(jnp.add, terms)


class SharedInducingPoints(eqx.Module):
    """Shared inducing inputs for multi-output latent GP collections."""

    locations: Float[Array, "M D"]

    def __check_init__(self) -> None:
        if self.locations.ndim != 2:
            raise ValueError("locations must have shape (num_inducing, input_dim).")

    @property
    def num_inducing(self) -> int:
        """Number of inducing inputs ``M`` shared by all latent processes."""
        return self.locations.shape[0]

    def latent_covariances(
        self, kernels: tuple[Kernel, ...]
    ) -> tuple[Float[Array, "M M"], ...]:
        """Return one inducing covariance block per latent kernel.

        Shares a per-call context per unique kernel instance so a kernel
        reused across latents registers its sample sites once.
        """
        if not kernels:
            raise ValueError("kernels must be non-empty.")
        with _kernel_contexts(kernels):
            return tuple(kernel(self.locations, self.locations) for kernel in kernels)

    def K_uu_operator(self, kernels: tuple[Kernel, ...]) -> BlockDiag:
        """Return the block-diagonal inducing covariance as a ``BlockDiag``.

        Downstream solvers decompose a ``(Q*M, Q*M)`` solve into ``Q``
        independent ``(M, M)`` solves via the ``block_diagonal_tag``.
        """
        if not kernels:
            raise ValueError("kernels must be non-empty.")
        with _kernel_contexts(kernels):
            blocks = tuple(
                _psd_matrix_op(kernel(self.locations, self.locations))
                for kernel in kernels
            )
        return BlockDiag(*blocks)  # ty: ignore[invalid-return-type]

    def K_uu(self, kernels: tuple[Kernel, ...]) -> Float[Array, "QM QM"]:
        """Materialize the block-diagonal inducing covariance over all latents."""
        return self.K_uu_operator(kernels).as_matrix()

    def cross_covariances(
        self,
        X: Float[Array, "N D"],
        kernels: tuple[Kernel, ...],
    ) -> tuple[Float[Array, "M N"], ...]:
        """Return one ``K(Z, X)`` block per latent kernel."""
        if not kernels:
            raise ValueError("kernels must be non-empty.")
        with _kernel_contexts(kernels):
            return tuple(kernel(self.locations, X) for kernel in kernels)


class MultiOutputInducingVariables(eqx.Module):
    """Shared inducing-point structure for LMC-style sparse workflows.

    ``mixing[p, q]`` is the weight with which latent process ``q`` enters
    output ``p``; the block layout of :meth:`K_uf` matches that convention.
    ``ICMKernel`` with non-zero ``kappa`` cannot be represented here —
    the extra diagonal does not fit the per-latent factorization.
    """

    inducing: SharedInducingPoints
    mixing: Float[Array, "P Q"]

    def __check_init__(self) -> None:
        _validate_mixing(self.mixing)

    @classmethod
    def from_kernel(
        cls,
        kernel: LMCKernel | ICMKernel,
        inducing: SharedInducingPoints,
    ) -> MultiOutputInducingVariables:
        """Construct from a kernel, sharing its mixing matrix.

        Avoids the footgun of maintaining two independent ``mixing``
        copies that can silently disagree between ``K_ff`` and ``K_uf``.
        Only :class:`LMCKernel` and :class:`ICMKernel` are accepted;
        :class:`OILMMKernel` is rejected because the sparse inducing
        workflow does not currently exploit orthogonal projection.

        For :class:`ICMKernel` with non-zero ``kappa``, the extra
        diagonal term is dropped — users should use a dense solve if
        the ``kappa`` contribution matters.
        """
        if isinstance(kernel, (LMCKernel, ICMKernel)):
            return cls(inducing=inducing, mixing=kernel.mixing)
        raise TypeError(
            "from_kernel only accepts LMCKernel or ICMKernel; "
            f"got {type(kernel).__name__}."
        )

    @property
    def num_outputs(self) -> int:
        """Number of observed output channels ``P``."""
        return self.mixing.shape[0]

    @property
    def num_latents(self) -> int:
        """Number of latent scalar GPs ``Q``."""
        return self.mixing.shape[1]

    def K_uu_operator(self, kernels: tuple[Kernel, ...]) -> BlockDiag:
        """Return the block-diagonal inducing covariance as a ``BlockDiag``."""
        _validate_kernel_count(kernels, self.num_latents)
        return self.inducing.K_uu_operator(kernels)

    def K_uu(self, kernels: tuple[Kernel, ...]) -> Float[Array, "QM QM"]:
        """Return the block-diagonal inducing covariance over latent processes."""
        _validate_kernel_count(kernels, self.num_latents)
        return self.inducing.K_uu(kernels)

    def K_uf(
        self,
        X: Float[Array, "N D"],
        kernels: tuple[Kernel, ...],
    ) -> Float[Array, "QM PN"]:
        """Return the inducing-to-output cross-covariance for isotopic outputs.

        The block layout is
        ``K_uf[q*M:(q+1)*M, p*N:(p+1)*N] = mixing[p, q] * k_q(Z, X)``.
        """
        _validate_kernel_count(kernels, self.num_latents)
        latent_blocks = self.inducing.cross_covariances(X, kernels)
        rows = []
        for q, K_zx in enumerate(latent_blocks):
            scaled = self.mixing[:, q][:, None, None] * K_zx[None, :, :]
            row = jnp.transpose(scaled, (1, 0, 2)).reshape(K_zx.shape[0], -1)
            rows.append(row)
        return jnp.concatenate(rows, axis=0)


__all__ = [
    "ICMKernel",
    "LMCKernel",
    "MultiOutputInducingVariables",
    "OILMMKernel",
    "SharedInducingPoints",
]
