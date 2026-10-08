r"""Matrix-free kernel operator for exact GPs at large ``n`` (P4, pyrox#277).

`GPPrior` with ``matrix_free=True`` builds its system as

$$
\hat K = K_{\text{op}} + (\text{jitter} + \sigma^2)\,I ,
$$

a `lineax.AddLinearOperator` of the kernel operator below and a scaled
identity. ``K`` is never formed: a matvec evaluates ``K`` one row block at
a time,

$$
(K v)_{b} = K(X_b, X)\,v , \qquad b = 1, \dots, \lceil n / B \rceil ,
$$

so the memory is $O(B n)$ and the cost $O(n^2)$ per matvec. Each block is
rematerialised (``jax.checkpoint``) on the backward pass, so reverse-mode
gradients with respect to the kernel hyperparameters also stay at
$O(B n)$ instead of storing ``K``. Vectorising a matvec over several
vectors (probe vectors, sketch columns) turns each block into a
matrix-matrix product.

`split_noise` recovers $K_{\text{op}}$ and the noise scalar from that sum,
so `pyrox_gp.preconditioned_cg_solver` can build its preconditioner from
$K$ with the exact noise as the shift.
"""

from __future__ import annotations

from typing import Any, cast

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from pyrox_gp._protocols import Kernel


_PSD_TAGS = frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag})


def freeze_kernel(kernel: Kernel) -> Kernel:
    """Replace every NumPyro-aware sub-kernel by its plain kernellib kernel.

    A matrix-free matvec evaluates the kernel inside ``jax.lax.map``, where a
    NumPyro site must not be registered. Each `pyrox_gp` kernel with a
    ``frozen()`` method (``RBF``, ``Matern``, ...) is resolved once, here, in
    the caller's kernel context; composite kernels (``k1 + k2``) are frozen
    leaf by leaf. Plain kernellib kernels pass through unchanged.
    """

    def is_pyrox_kernel(node: object) -> bool:
        return hasattr(node, "_get_context") and hasattr(node, "frozen")

    def freeze(node: object) -> object:
        return cast(Any, node).frozen() if is_pyrox_kernel(node) else node

    return jax.tree.map(freeze, kernel, is_leaf=is_pyrox_kernel)


class KernelOperator(lx.AbstractLinearOperator):
    """``K(X, X)`` as a symmetric PSD operator, applied by row blocks.

    Attributes:
        kernel: A kernel without NumPyro state (see `freeze_kernel`); its
            array leaves are differentiable.
        X: Inputs, shape ``(N, D)``.
        block_size: Rows of ``K`` evaluated per step of the matvec.
    """

    kernel: Kernel
    X: Float[Array, "N D"]
    block_size: int = eqx.field(static=True, default=256)

    def mv(self, vector: Float[Array, " N"]) -> Float[Array, " N"]:
        X = self.X

        @jax.checkpoint
        def row(x: Float[Array, " D"]) -> Float[Array, ""]:
            k_row = self.kernel(x[None, :], X)[0]
            return einx.dot("n, n ->", k_row, vector)

        return jax.lax.map(row, X, batch_size=min(self.block_size, X.shape[0]))

    def as_matrix(self) -> Float[Array, "N N"]:
        return self.kernel(self.X, self.X)

    def transpose(self) -> KernelOperator:
        return self

    def in_structure(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((self.X.shape[0],), self.X.dtype)

    def out_structure(self) -> jax.ShapeDtypeStruct:
        return self.in_structure()


lx.is_symmetric.register(KernelOperator)(lambda _operator: True)
lx.is_positive_semidefinite.register(KernelOperator)(lambda _operator: True)
lx.is_negative_semidefinite.register(KernelOperator)(lambda _operator: False)
lx.is_diagonal.register(KernelOperator)(lambda _operator: False)
lx.is_tridiagonal.register(KernelOperator)(lambda _operator: False)
lx.has_unit_diagonal.register(KernelOperator)(lambda _operator: False)
lx.is_lower_triangular.register(KernelOperator)(lambda _operator: False)
lx.is_upper_triangular.register(KernelOperator)(lambda _operator: False)
lx.linearise.register(KernelOperator)(lambda operator: operator)
lx.materialise.register(KernelOperator)(lambda operator: operator)


@lx.diagonal.register(KernelOperator)
def _(operator: KernelOperator) -> Float[Array, " N"]:
    return operator.kernel.diag(operator.X)


def matrix_free_operator(
    kernel: Kernel,
    X: Float[Array, "N D"],
    diagonal: Float[Array, ""],
    *,
    block_size: int = 256,
) -> lx.AbstractLinearOperator:
    r"""$K_{\text{op}} + d\,I$ as a PSD-tagged `lineax.AddLinearOperator`.

    Args:
        kernel: A kernel without NumPyro state (see `freeze_kernel`).
        X: Inputs, shape ``(N, D)``.
        diagonal: The scalar $d$ on the diagonal, e.g. ``jitter + noise_var``.
        block_size: Rows of ``K`` per matvec step.

    Returns:
        ``TaggedLinearOperator(KernelOperator + d I, PSD)``; `split_noise`
        inverts it.
    """
    K_op = KernelOperator(kernel, X, block_size)
    identity = lx.IdentityLinearOperator(K_op.in_structure())
    scaled = identity * jnp.asarray(diagonal, dtype=X.dtype)
    return lx.TaggedLinearOperator(K_op + scaled, _PSD_TAGS)


def split_noise(
    operator: lx.AbstractLinearOperator,
) -> tuple[lx.AbstractLinearOperator, Float[Array, ""]] | None:
    r"""Recover $(K_{\text{op}}, d)$ from $K_{\text{op}} + d\,I$, or ``None``.

    Recognises the sum `matrix_free_operator` builds (tagged or not, with the
    scaled identity on either side). Any other operator, such as one dense
    matrix with the noise on its diagonal, returns ``None``.
    """
    if isinstance(operator, lx.TaggedLinearOperator):
        operator = operator.operator
    if not isinstance(operator, lx.AddLinearOperator):
        return None
    for psd_part, other in (
        (operator.operator1, operator.operator2),
        (operator.operator2, operator.operator1),
    ):
        if isinstance(other, lx.MulLinearOperator) and isinstance(
            other.operator, lx.IdentityLinearOperator
        ):
            return psd_part, other.scalar
    return None
