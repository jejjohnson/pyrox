"""Multi-output Gaussian-process kernels and inducing structures.

These classes keep output mixing explicit instead of hiding it inside a
specialized model. Independent latent kernels stay reusable, while the
multi-output surface exposes the coregionalization matrices and the
Kronecker-style factors that later solvers can exploit.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

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


def _block_diag(
    blocks: tuple[Float[Array, "R C"], ...],
) -> Float[Array, "R_total C_total"]:
    if not blocks:
        raise ValueError("blocks must be non-empty.")
    total_rows = sum(block.shape[0] for block in blocks)
    total_cols = sum(block.shape[1] for block in blocks)
    out = jnp.zeros((total_rows, total_cols), dtype=jnp.result_type(*blocks))
    row_offset = 0
    col_offset = 0
    for block in blocks:
        n_rows, n_cols = block.shape
        row_slice = slice(row_offset, row_offset + n_rows)
        col_slice = slice(col_offset, col_offset + n_cols)
        out = out.at[row_slice, col_slice].set(block)
        row_offset += n_rows
        col_offset += n_cols
    return out


class LMCKernel(eqx.Module):
    """Linear model of coregionalization for vector-valued GPs.

    Each output is a linear combination of latent scalar GPs:
    ``f_p(x) = sum_q W[p, q] g_q(x)``.
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
        """Return ``(B_q, K_q(X1, X2))`` factors for each latent process."""
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
        blocks = None
        for B_q, K_q in self.kronecker_factors(X1, X2):
            term = B_q[:, :, None, None] * K_q[None, None, :, :]
            blocks = term if blocks is None else blocks + term
        assert blocks is not None
        return blocks

    def cross_covariance(
        self,
        X1: Float[Array, "N1 D"],
        X2: Float[Array, "N2 D"],
    ) -> Float[Array, "PN1 PN2"]:
        """Return the dense covariance of ``vec(F(X1))`` and ``vec(F(X2))``."""
        full = None
        for B_q, K_q in self.kronecker_factors(X1, X2):
            term = jnp.kron(B_q, K_q)
            full = term if full is None else full + term
        assert full is not None
        return full

    def full_covariance(self, X: Float[Array, "N D"]) -> Float[Array, "PN PN"]:
        """Return the dense Gram matrix for isotopic multi-output observations."""
        return self.cross_covariance(X, X)

    def diag(self, X: Float[Array, "N D"]) -> Float[Array, "N P"]:
        """Return per-input, per-output marginal variances with shape ``(N, P)``."""
        diag = None
        for q, kernel in enumerate(self.kernels):
            term = kernel.diag(X)[:, None] * jnp.square(self.mixing[:, q])[None, :]
            diag = term if diag is None else diag + term
        assert diag is not None
        return diag


class ICMKernel(eqx.Module):
    """Intrinsic coregionalization model with one shared latent kernel."""

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
        return self.coregionalization_matrix(), self.kernel(X1, X2)

    def output_covariance(
        self,
        X1: Float[Array, "N1 D"],
        X2: Float[Array, "N2 D"],
    ) -> Float[Array, "P P N1 N2"]:
        """Return output-pair covariance blocks with shape ``(P, P, N1, N2)``."""
        B, K = self.kronecker_factors(X1, X2)
        return B[:, :, None, None] * K[None, None, :, :]

    def cross_covariance(
        self,
        X1: Float[Array, "N1 D"],
        X2: Float[Array, "N2 D"],
    ) -> Float[Array, "PN1 PN2"]:
        """Return the dense covariance of ``vec(F(X1))`` and ``vec(F(X2))``."""
        B, K = self.kronecker_factors(X1, X2)
        return jnp.kron(B, K)

    def full_covariance(self, X: Float[Array, "N D"]) -> Float[Array, "PN PN"]:
        """Return the dense Gram matrix for isotopic multi-output observations."""
        return self.cross_covariance(X, X)

    def diag(self, X: Float[Array, "N D"]) -> Float[Array, "N P"]:
        """Return per-input, per-output marginal variances with shape ``(N, P)``."""
        return (
            self.kernel.diag(X)[:, None]
            * jnp.diag(self.coregionalization_matrix())[None, :]
        )


class OILMMKernel(eqx.Module):
    """Orthogonal instantaneous linear mixing model.

    The latent GP kernels stay independent. Orthogonal mixing makes it
    possible to project observations into latent space and run ``Q`` scalar
    GP problems instead of one monolithic multi-output solve.
    """

    kernels: tuple[Kernel, ...]
    mixing: Float[Array, "P Q"]
    noise_variance: Float[Array, ""]

    def __check_init__(self) -> None:
        _validate_mixing(self.mixing)
        _validate_kernel_count(self.kernels, self.mixing.shape[1])
        if self.mixing.shape[1] > self.mixing.shape[0]:
            raise ValueError(
                "orthogonal mixing requires num_latents <= num_outputs; "
                f"got {self.mixing.shape[1]} latents and "
                f"{self.mixing.shape[0]} outputs."
            )
        if jnp.ndim(self.noise_variance) != 0:
            raise ValueError("noise_variance must be a scalar.")

    @property
    def num_outputs(self) -> int:
        """Number of observed output channels ``P``."""
        return self.mixing.shape[0]

    @property
    def num_latents(self) -> int:
        """Number of latent scalar GPs ``Q``."""
        return self.mixing.shape[1]

    def is_orthogonal(self, *, atol: float = 1e-6, rtol: float = 1e-6) -> bool:
        """Whether the current mixing matrix satisfies ``W^T W ≈ I``."""
        gram = self.mixing.T @ self.mixing
        eye = jnp.eye(self.num_latents, dtype=self.mixing.dtype)
        return bool(jnp.allclose(gram, eye, atol=atol, rtol=rtol))

    def project_observations(self, Y: Float[Array, "N P"]) -> Float[Array, "N Q"]:
        """Project observations into latent space via ``Y @ W``."""
        if Y.ndim != 2 or Y.shape[1] != self.num_outputs:
            raise ValueError(
                f"Y must have shape (N, {self.num_outputs}); got {Y.shape}."
            )
        return Y @ self.mixing

    def independent_gps(self) -> tuple[Kernel, ...]:
        """Return the latent scalar GP kernels used after projection."""
        return self.kernels

    def signal_factors(
        self,
        X1: Float[Array, "N1 D"],
        X2: Float[Array, "N2 D"],
    ) -> tuple[tuple[Float[Array, "P P"], Float[Array, "N1 N2"]], ...]:
        """Return the latent signal factors before the isotropic noise term."""
        return tuple(
            (jnp.outer(self.mixing[:, q], self.mixing[:, q]), kernel(X1, X2))
            for q, kernel in enumerate(self.kernels)
        )

    def signal_covariance(
        self,
        X1: Float[Array, "N1 D"],
        X2: Float[Array, "N2 D"],
    ) -> Float[Array, "PN1 PN2"]:
        """Return the signal covariance without the observation-noise term."""
        full = None
        for B_q, K_q in self.signal_factors(X1, X2):
            term = jnp.kron(B_q, K_q)
            full = term if full is None else full + term
        assert full is not None
        return full

    def full_covariance(self, X: Float[Array, "N D"]) -> Float[Array, "PN PN"]:
        """Return the dense Gram matrix plus isotropic observation noise."""
        signal = self.signal_covariance(X, X)
        noise = self.noise_variance * jnp.eye(signal.shape[0], dtype=signal.dtype)
        return signal + noise

    def diag(self, X: Float[Array, "N D"]) -> Float[Array, "N P"]:
        """Return per-input, per-output marginal variances including noise."""
        diag = None
        for q, kernel in enumerate(self.kernels):
            term = kernel.diag(X)[:, None] * jnp.square(self.mixing[:, q])[None, :]
            diag = term if diag is None else diag + term
        assert diag is not None
        return diag + self.noise_variance


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
        """Return one inducing covariance block per latent kernel."""
        if not kernels:
            raise ValueError("kernels must be non-empty.")
        return tuple(kernel(self.locations, self.locations) for kernel in kernels)

    def K_uu(self, kernels: tuple[Kernel, ...]) -> Float[Array, "QM QM"]:
        """Materialize the block-diagonal inducing covariance over all latents."""
        return _block_diag(self.latent_covariances(kernels))

    def cross_covariances(
        self,
        X: Float[Array, "N D"],
        kernels: tuple[Kernel, ...],
    ) -> tuple[Float[Array, "M N"], ...]:
        """Return one ``K(Z, X)`` block per latent kernel."""
        if not kernels:
            raise ValueError("kernels must be non-empty.")
        return tuple(kernel(self.locations, X) for kernel in kernels)


class MultiOutputInducingVariables(eqx.Module):
    """Shared inducing-point structure for LMC/ICM-style sparse workflows."""

    inducing: SharedInducingPoints
    mixing: Float[Array, "P Q"]

    def __check_init__(self) -> None:
        _validate_mixing(self.mixing)

    @property
    def num_outputs(self) -> int:
        """Number of observed output channels ``P``."""
        return self.mixing.shape[0]

    @property
    def num_latents(self) -> int:
        """Number of latent scalar GPs ``Q``."""
        return self.mixing.shape[1]

    def K_uu(self, kernels: tuple[Kernel, ...]) -> Float[Array, "QM QM"]:
        """Return the block-diagonal inducing covariance over latent processes."""
        _validate_kernel_count(kernels, self.num_latents)
        return self.inducing.K_uu(kernels)

    def K_uf(
        self,
        X: Float[Array, "N D"],
        kernels: tuple[Kernel, ...],
    ) -> Float[Array, "QM PN"]:
        """Return the inducing-to-output cross-covariance for isotopic outputs."""
        _validate_kernel_count(kernels, self.num_latents)
        latent_blocks = self.inducing.cross_covariances(X, kernels)
        rows = []
        for q, K_zx in enumerate(latent_blocks):
            row = jnp.concatenate(
                [self.mixing[p, q] * K_zx for p in range(self.num_outputs)], axis=1
            )
            rows.append(row)
        return jnp.concatenate(rows, axis=0)


__all__ = [
    "ICMKernel",
    "LMCKernel",
    "MultiOutputInducingVariables",
    "OILMMKernel",
    "SharedInducingPoints",
]
