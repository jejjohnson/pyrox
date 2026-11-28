r"""The preconditioner and log-determinant behind `pyrox_gp.matfree_solver` (P4).

With $\hat K = K + \sigma^2 I$ and $\alpha = \hat K^{-1}y$,

$$
\log p(y) = -\tfrac12 y^\top\alpha - \tfrac12\log|\hat K| - \tfrac n2\log 2\pi .
$$

The solve is conjugate gradients preconditioned by a low-rank
approximation of $K$ (gaussx G13); the log-determinant is stochastic
Lanczos quadrature, either on $\hat K$ itself or on $\hat K$
preconditioned by a Nyström approximation (gaussx G17,
`gaussx.NystromLogdet`); gradients flow through both (the preconditioner only
changes the iteration count, so gaussx stops its gradient). The
preconditioner is built from $K$ with the diagonal as its shift, never from
$\hat K$ itself, so the noise is not counted twice (gaussx#345). The system
is read back with `gaussx.split_diagonal`:

- for ``GPPrior(..., matrix_free=True)`` the system is the sum
  $K_{\text{op}} + (\text{jitter} + \sigma^2) I$, and the scalar noise is
  read from it exactly;
- for a sum $K + \mathrm{diag}(d)$ whose diagonal varies by point (site
  variances, heteroscedastic noise), the preconditioner is built by gaussx's
  per-point shift: a preconditioner $P_B$ of
  $B = D^{-1/2} \hat K D^{-1/2}$, $D = \mathrm{diag}(d)$, with unit shift,
  pulled back as $P^{-1} = D^{-1/2} P_B^{-1} D^{-1/2}$ (gaussx#878);
- for any other operator (one dense $\hat K$) the preconditioner is built
  from $\hat K - \mu I$ with a user lower bound $\mu \le \sigma^2$.
"""

from __future__ import annotations

import warnings
from typing import Literal

import equinox as eqx
import gaussx as gx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
from jaxtyping import Array, Float

from pyrox_gp._kernel_operator import KernelOperator


def _scalar_noise(
    operator: lx.AbstractLinearOperator,
) -> tuple[lx.AbstractLinearOperator, Float[Array, ""]] | None:
    """$(K, s)$ for a sum of exactly $K$ and $s I$, by type, or ``None``.

    A Gaussian-noise system keeps the scalar shift it has always had, so its
    preconditioner is bit-identical to the one before per-point shifts
    (pyrox#555). $K$ must hold no diagonal of its own, or the sum is a
    per-point $K' + \\mathrm{diag}(d)$ for `gaussx.split_diagonal`.
    """
    while isinstance(operator, lx.TaggedLinearOperator):
        operator = operator.operator
    if not isinstance(operator, lx.AddLinearOperator):
        return None
    for psd_part, other in (
        (operator.operator1, operator.operator2),
        (operator.operator2, operator.operator1),
    ):
        is_scaled_identity = isinstance(other, lx.MulLinearOperator) and isinstance(
            other.operator, lx.IdentityLinearOperator
        )
        has_diagonal = lx.is_diagonal(psd_part) or (
            gx.split_diagonal(psd_part) is not None
        )
        if is_scaled_identity and not has_diagonal:
            return psd_part, other.scalar
    return None


def _split_system(
    operator: lx.AbstractLinearOperator,
) -> tuple[lx.AbstractLinearOperator, Float[Array, ""] | Float[Array, " N"]] | None:
    r"""$(K, s)$ for $K + s I$, $(K, d)$ for $K + \mathrm{diag}(d)$, else ``None``.

    The split is by type (`gaussx.split_diagonal`), so it is the same under
    ``jit`` with the diagonal traced.
    """
    scalar_split = _scalar_noise(operator)
    if scalar_split is not None:
        # K + s I: the exact scalar, so the Gaussian-noise path is unchanged
        return scalar_split
    split = gx.split_diagonal(operator)
    if split is None:
        return None
    # K + diag(d): a per-point shift, by symmetric scaling in gaussx
    return split.base, split.diagonal


_NOT_A_SUM = (
    "the system is not a sum K + diag(d), so the diagonal cannot be read "
    "from it. Use GPPrior(..., matrix_free=True), or pass the deprecated "
    "preconditioned_cg_solver a shift= (a lower bound on the noise variance)."
)


def _warn_shift_ignored(shift: float | None) -> None:
    if shift is not None:
        warnings.warn(
            f"shift={shift} is ignored: the system is a sum K + diag(d), and "
            "the diagonal is read from it (pyrox#557).",
            UserWarning,
            stacklevel=3,
        )


class _NoiseSplitPreconditioner(gx.AbstractPreconditioner):
    """Low-rank preconditioner of ``K + diag(d)``, rebuilt from the system per solve.

    If the system is the sum ``K_op + s I`` (`GPPrior` with
    ``matrix_free=True``), the low-rank part is built from ``K_op`` with the
    exact ``s``. If it is ``K + diag(d)`` with a diagonal that varies by
    point, it is built from ``K`` with the per-point shift ``d``. Otherwise
    it is built from ``A - shift I`` with the user bound ``shift``.

    It serves the CG solve only. On ``K + s I`` it keeps gaussx's scalar
    Nyström form, whose $P$ is $A$ divided by $\\hat\\gamma_k + s$; CG does
    not see that scale, but a log-determinant that draws probes from $P$
    would, so `pyrox_gp.bbmm_solver` uses gaussx's lazy preconditioners.
    """

    kind: Literal["nystrom", "rpcholesky"] = eqx.field(static=True)
    rank: int = eqx.field(static=True)
    shift: float | None = eqx.field(static=True)
    key: jax.Array

    def _psd_part_and_shift(
        self, operator: lx.AbstractLinearOperator
    ) -> tuple[
        lx.AbstractLinearOperator,
        float | Float[Array, ""] | Float[Array, " N"],
    ]:
        split = _split_system(operator)
        if split is not None:
            _warn_shift_ignored(self.shift)
            return split
        if self.shift is None:
            raise ValueError(f"The pyrox-gp preconditioner: {_NOT_A_SUM}")
        identity = lx.IdentityLinearOperator(operator.in_structure())
        return operator - self.shift * identity, self.shift

    def _built(self, operator: lx.AbstractLinearOperator) -> gx.AbstractPreconditioner:
        """The gaussx preconditioner of this system, built from its kernel part."""
        psd_part, shift = self._psd_part_and_shift(operator)
        psd_part = lx.TaggedLinearOperator(psd_part, lx.positive_semidefinite_tag)
        if self.kind == "nystrom":
            return gx.NystromPreconditioner.from_operator(
                psd_part, self.rank, shift=shift, key=self.key
            )
        return gx.PartialCholeskyPreconditioner.from_operator(
            psd_part, self.rank, shift=shift, pivoting="random", key=self.key
        )

    def as_operator(self, operator=None):
        if operator is None:
            return None
        return self._built(operator).as_operator()


class _NoiseSplitNystromLogdet(gx.AbstractLogdetStrategy):
    """`gaussx.NystromLogdet` of ``K + diag(d)``, ``d`` read from the system per call.

    `gaussx.NystromLogdet` needs the shift contained in the system at
    construction, and a learned noise changes it at every call. If the
    system is the sum ``K_op + s I`` (`GPPrior` with ``matrix_free=True``),
    the shift is the exact scalar of that sum; if it is ``K + diag(d)``, the
    per-point ``d``. Otherwise it is the user bound ``shift`` (a lower bound
    on the noise variance is still valid, since the strategy only needs
    ``A - shift I`` to be PSD; a shift above the noise makes it indefinite
    and the estimate NaN, gaussx#878).
    """

    rank: int = eqx.field(static=True)
    shift: float | None = eqx.field(static=True)
    num_probes: int = eqx.field(static=True)
    lanczos_order: int = eqx.field(static=True)
    key: jax.Array

    def logdet(
        self,
        operator: lx.AbstractLinearOperator,
        *,
        key: jax.Array | None = None,
    ) -> Float[Array, ""]:
        split = _split_system(operator)
        shift: float | Float[Array, ""] | Float[Array, " N"]
        if split is not None:
            _warn_shift_ignored(self.shift)
            _, shift = split
        elif self.shift is not None:
            shift = self.shift
        else:
            raise ValueError(
                f'The Nyström log-determinant (logdet="nystrom"): {_NOT_A_SUM}'
            )
        strategy = gx.NystromLogdet(
            shift=shift,
            rank=self.rank,
            num_probes=self.num_probes,
            lanczos_order=self.lanczos_order,
        )
        # no key from the caller: the solver's own key, so the estimate is a
        # fixed function of the hyperparameters (common random numbers)
        logdet_key = self.key if key is None else key
        return strategy.logdet(operator, key=logdet_key)


def _untagged(operator: lx.AbstractLinearOperator) -> lx.AbstractLinearOperator:
    while isinstance(operator, lx.TaggedLinearOperator):
        operator = operator.operator
    return operator


class _SystemFITCPreconditioner(gx.AbstractPreconditioner):
    r"""`gaussx.FITCPreconditioner` of $A = K + \mathrm{diag}(d)$, built per system.

    $$
    P = U U^\top + \mathrm{diag}(r + d), \qquad
    U = K_{XZ} L_{ZZ}^{-\top}, \qquad
    r = \max(\mathrm{diag}(K) - \mathrm{rowsum}(U \odot U), 0),
    $$

    with the $R$ = ``rank`` inducing inputs $Z$ drawn by ``key`` from the
    training inputs, without replacement. The kernel part $K$ and $d$ are
    read from the system with `gaussx.split_diagonal`, so one object serves
    a Gaussian prior and every Newton step of a non-Gaussian fit. For a
    ``GPPrior(..., matrix_free=True)`` system, $K$ is a `KernelOperator`
    and only $K_{ZZ}$, $K_{XZ}$ and $\mathrm{diag}(K)$ are evaluated,
    $O(N R)$ kernel entries; for a dense prior they are read from its
    matrix. $P$ matches $A$ on the diagonal and is at its scale, so
    `gaussx.BBMMSolver` can draw its probes from it.
    """

    rank: int = eqx.field(static=True)
    key: jax.Array

    def _built(
        self, operator: lx.AbstractLinearOperator | None
    ) -> gx.FITCPreconditioner:
        """The gaussx FITC preconditioner of this system."""
        split = None if operator is None else gx.split_diagonal(operator)
        if split is None:
            raise ValueError(
                'preconditioner="fitc" reads the kernel part K and the diagonal '
                "d from the system, which must be a sum K + diag(d) (a "
                "GPPrior(..., matrix_free=True) system, or the site system of "
                "a non-Gaussian fit)."
            )
        kernel_part = _untagged(split.base)
        size = kernel_part.in_structure().shape[0]
        if self.rank > size:
            raise ValueError(
                f'preconditioner="fitc": rank {self.rank} exceeds the {size} '
                "training inputs it draws its inducing inputs from."
            )
        # Z: R training inputs, drawn without replacement
        index = jr.choice(self.key, size, (self.rank,), replace=False)
        if isinstance(kernel_part, KernelOperator):
            inputs = kernel_part.X
            frozen = kernel_part.kernel

            def kernel(left: Array, right: Array) -> Array:
                # the operator's dtype, as in KernelOperator.mv
                return frozen(left, right).astype(inputs.dtype)

            return gx.FITCPreconditioner.from_kernel(kernel, inputs, inputs[index])
        if not isinstance(kernel_part, lx.MatrixLinearOperator):
            raise ValueError(
                'preconditioner="fitc" builds its inducing blocks from a '
                "KernelOperator (GPPrior(..., matrix_free=True)) or a dense "
                f"kernel matrix, not a {type(kernel_part).__name__}."
            )
        matrix = kernel_part.as_matrix()

        def entries(left: Array, right: Array) -> Array:
            # the "inputs" are row positions, so K(X₁, X₂) is a block of K
            return matrix[left[:, 0][:, None], right[:, 0][None, :]]

        positions = jnp.arange(size)[:, None]
        return gx.FITCPreconditioner.from_kernel(entries, positions, positions[index])

    def as_operator(
        self, operator: lx.AbstractLinearOperator | None = None
    ) -> lx.AbstractLinearOperator:
        return self._built(operator).as_operator(operator)

    def supports_slq(self, operator: lx.AbstractLinearOperator | None = None) -> bool:
        del operator
        return True

    def logdet(
        self, operator: lx.AbstractLinearOperator | None = None
    ) -> Float[Array, ""]:
        return self._built(operator).logdet(operator)

    def sample(
        self,
        key: jax.Array,
        operator: lx.AbstractLinearOperator | None = None,
        *,
        shape: tuple[int, ...] = (),
    ) -> Float[Array, "*batch n"]:
        return self._built(operator).sample(key, operator, shape=shape)

    def approximation(
        self, operator: lx.AbstractLinearOperator | None = None
    ) -> lx.AbstractLinearOperator:
        return self._built(operator).approximation(operator)
