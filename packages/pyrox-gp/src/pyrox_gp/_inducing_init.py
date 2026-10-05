r"""Choosing inducing inputs from the data (P3).

The gap between the SVGP ELBO and the log marginal likelihood grows with
the Nyström trace error $t = \operatorname{tr}(K_{ff} - Q_{ff})$ of the
inducing set (Burt, Rasmussen & van der Wilk, 2019, 2020). Greedy
conditional-variance selection is pivoted Cholesky on $K_{ff}$;
randomly pivoted Cholesky (kernellib / gaussx) controls $t$ in
expectation without greedy's pull towards outliers.
"""

from __future__ import annotations

from typing import Literal

import kernellib as kl
from jaxtyping import Array, Float, PRNGKeyArray

from pyrox_gp._context import _kernel_context
from pyrox_gp._kernels import Kernel, _ParameterizedKernel


def init_inducing(
    X: Float[Array, "N D"],
    n_inducing: int,
    *,
    kernel: Kernel | None,
    method: Literal["uniform", "rpcholesky", "greedy", "leverage"] = "rpcholesky",
    key: PRNGKeyArray,
) -> Float[Array, "M D"]:
    """Inducing inputs ``(M, D)``: rows of ``X`` picked by `kernellib.select_landmarks`.

    The selection runs once, on ``kernel`` frozen at its current
    hyperparameters (`Kernel.frozen`). If they move a long way during
    training, call it again with the fitted kernel and refit.

    Args:
        X: Training inputs.
        n_inducing: Number of inducing inputs ``M``.
        kernel: The GP's kernel; required by keyword. ``None`` only with
            ``method="uniform"``, which does not look at it.
        method: ``"rpcholesky"`` (default: kernel-aware, no tuning),
            ``"greedy"`` (largest conditional variance; deterministic, but
            drawn to outliers), ``"leverage"`` (approximate ridge leverage)
            or ``"uniform"``.
        key: PRNG key (unused by ``"greedy"``).

    Returns:
        ``X[indices]``, distinct rows of ``X``.

    Raises:
        ValueError: If ``kernel`` is None with a kernel-aware method.

    Examples:
        ```python
        import jax.random as jr
        import pyrox_gp as px

        kernel = px.RBF(init_lengthscale=0.3)
        Z = px.init_inducing(X, 64, kernel=kernel, key=jr.key(0))
        prior = px.SparseGPPrior(kernel, Z=Z)
        # after fitting, if the lengthscale moved a lot:
        # Z = px.init_inducing(X, 64, kernel=fitted_kernel, key=jr.key(1))
        ```
    """
    if kernel is None:
        if method != "uniform":
            raise ValueError(
                f"method={method!r} needs the kernel; "
                'kernel=None only works with "uniform"'
            )
        frozen: kl.AbstractKernel = kl.RBF()  # never evaluated by "uniform"
    else:
        if isinstance(kernel, _ParameterizedKernel):
            with _kernel_context(kernel):
                frozen = kernel.frozen()
        else:  # already a plain kernellib kernel
            frozen = kernel
    idx = kl.select_landmarks(frozen, X, n_inducing, method=method, key=key)
    return X[idx]
