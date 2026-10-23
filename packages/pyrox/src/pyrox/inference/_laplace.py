"""Post-hoc last-layer Laplace approximation with a GGN Hessian.

See [`last_layer_laplace`][pyrox.inference.last_layer_laplace]. The linear
algebra goes through ``gaussx`` on PSD-tagged ``lineax`` operators; both are
optional (``pip install pyrox[laplace]``) and imported lazily, following the
same convention as ``optax`` in the ensemble primitives.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, Float, PyTree


def _require_gaussx() -> tuple[Any, Any]:
    try:
        import gaussx
        import lineax as lx
    except ImportError as e:
        raise ImportError(
            "pyrox.inference.last_layer_laplace requires `gaussx` and `lineax`. "
            "Install via `pip install pyrox[laplace]`."
        ) from e
    return gaussx, lx


def _split(
    params: PyTree, param_filter: Callable[[tuple, Any], bool]
) -> tuple[Float[Array, " P"], Callable[[Array], PyTree], PyTree]:
    """Flatten the selected leaves of ``params``: ``(w, unravel, rest)``."""

    def keep(path: tuple, leaf: Any) -> bool:
        return eqx.is_inexact_array(leaf) and bool(param_filter(path, leaf))

    spec = jax.tree_util.tree_map_with_path(keep, params)
    selected, rest = eqx.partition(params, spec)
    w, unravel = ravel_pytree(selected)
    return w, unravel, rest


def _forward(
    predict_fn: Callable[[PyTree, Array], Array],
    unravel: Callable[[Array], PyTree],
    rest: PyTree,
    x: Array,
) -> Callable[[Array], Array]:
    """``w -> vec(f(x; w))`` with every other parameter frozen."""

    def f(w: Array) -> Array:
        return jnp.ravel(predict_fn(eqx.combine(unravel(w), rest), x))

    return f


class LastLayerLaplace(eqx.Module):
    r"""Gaussian posterior over a selected parameter group of a fitted model.

    Built by [`last_layer_laplace`][pyrox.inference.last_layer_laplace].
    With $w$ the selected weights (the rest frozen at the MAP), $J$ the
    Jacobian $\partial f / \partial w$ on the training inputs and Gaussian
    noise $\sigma^2$, the posterior is $\mathcal{N}(w^*, H^{-1})$ with the
    generalized Gauss-Newton (GGN) precision

    $$
    H = G + \lambda I, \qquad G = \sigma^{-2} J^\top J,
    $$

    where $G$ is kept whole (``structure="full"``) or as its diagonal
    (``structure="diag"``). The posterior registers no NumPyro sites.

    Attributes:
        predict_fn: ``(params, x) -> f``, the same predictive as
            `ensemble_predict` takes.
        params: The MAP parameters the posterior is centred on.
        param_filter: ``(path, leaf) -> bool`` selecting the weights $w$.
        w_map: Flattened MAP weights $w^*$, shape ``(P,)``.
        ggn: $G$, shape ``(P, P)`` (full) or ``(P,)`` (diag).
        log_likelihood: $\log p(y \mid w^*)$ on the training data.
        sigma: Observation noise standard deviation $\sigma$.
        prior_precision: Isotropic prior precision $\lambda$.
        structure: ``"full"`` or ``"diag"``.
        jitter: Added to the diagonal of $H$ before factorizing.
    """

    predict_fn: Callable[[PyTree, Array], Array] = eqx.field(static=True)
    params: PyTree
    param_filter: Callable[[tuple, Any], bool] = eqx.field(static=True)
    w_map: Float[Array, " P"]
    ggn: Float[Array, "P P"] | Float[Array, " P"]
    log_likelihood: Float[Array, ""]
    sigma: Float[Array, ""]
    prior_precision: Float[Array, ""]
    structure: Literal["full", "diag"] = eqx.field(static=True)
    jitter: float = eqx.field(static=True)

    def precision(self) -> Any:
        r"""The posterior precision $H = G + (\lambda + \text{jitter}) I$.

        Returns:
            A PSD-tagged ``lineax`` operator of size ``(P, P)``: a
            ``MatrixLinearOperator`` (full) or a ``DiagonalLinearOperator``
            (diag).
        """
        gaussx, lx = _require_gaussx()
        shift = self.prior_precision + self.jitter
        if self.structure == "diag":
            return lx.DiagonalLinearOperator(self.ggn + shift)
        p = self.w_map.shape[0]
        h = gaussx.symmetrize(self.ggn) + shift * jnp.eye(p, dtype=self.ggn.dtype)
        return lx.MatrixLinearOperator(h, lx.positive_semidefinite_tag)

    def predict(self, x: Array) -> tuple[Array, Array]:
        r"""Linearized predictive mean and variance of $f$ at ``x``.

        $$
        \mathbb{E}[f(x_*)] = f(x_*; w^*), \qquad
        \mathrm{Var}[f(x_*)] = J_* H^{-1} J_*^\top,
        $$

        with $J_* = \partial f(x_*) / \partial w$ at $w^*$. The variance is
        of the latent $f$; add ``sigma**2`` for the predictive of $y$.

        Args:
            x: Inputs, in whatever form ``predict_fn`` takes.

        Returns:
            ``(mean, var)``, each shaped like ``predict_fn(params, x)``.
        """
        gaussx, _ = _require_gaussx()
        _, unravel, rest = _split(self.params, self.param_filter)
        f = _forward(self.predict_fn, unravel, rest, x)
        mean = self.predict_fn(self.params, x)
        jac = jax.jacfwd(f)(self.w_map)  # (M, P)
        h = self.precision()
        if self.structure == "diag":
            # H⁻¹ is diagonal: Var = Σ_p J_*² / h_p.
            hinv_jt = gaussx.solve_columns(h, einx.id("m p -> p m", jac))
            var = einx.sum("[p] m", einx.id("m p -> p m", jac) * hinv_jt)
        else:
            # H = L Lᵀ, so J_* H⁻¹ J_*ᵀ has diagonal Σ_p (L⁻¹ J_*ᵀ)².
            v = gaussx.solve_columns(gaussx.cholesky(h), einx.id("m p -> p m", jac))
            var = einx.sum("[p] m", v**2)
        return mean, jnp.reshape(var, mean.shape)

    def log_marginal_likelihood(
        self, prior_precision: Float[Array, ""] | float | None = None
    ) -> Float[Array, ""]:
        r"""Laplace evidence $\log p(y \mid \lambda)$ at fixed $w^*$.

        $$
        \log p(y \mid \lambda) \approx \log p(y \mid w^*)
            - \tfrac{\lambda}{2} \lVert w^* \rVert^2
            + \tfrac{P}{2} \log \lambda - \tfrac{1}{2} \log\det H.
        $$

        For a model linear in $w$ with $w^*$ the posterior mean under
        $\lambda$, this is the exact Bayesian linear-regression evidence.

        Args:
            prior_precision: $\lambda$ to evaluate at; defaults to the
                posterior's own.

        Returns:
            Scalar log evidence.
        """
        gaussx, _ = _require_gaussx()
        post = self
        if prior_precision is not None:
            post = eqx.tree_at(
                lambda m: m.prior_precision,
                self,
                jnp.asarray(prior_precision, dtype=self.w_map.dtype),
            )
        lam = post.prior_precision
        p = self.w_map.shape[0]
        return (
            self.log_likelihood
            - 0.5 * lam * jnp.sum(self.w_map**2)
            + 0.5 * p * jnp.log(lam)
            - 0.5 * gaussx.logdet(post.precision())
        )

    def optimize_prior_precision(self, *, num_steps: int = 100) -> LastLayerLaplace:
        r"""Return a copy with $\lambda$ maximizing the Laplace evidence.

        With $g_i$ the eigenvalues of $G$ (plus jitter) and $w^*$ fixed,

        $$
        \frac{\partial \log p(y \mid \lambda)}{\partial \log \lambda}
        = \tfrac{1}{2}\Bigl(\sum_i \frac{g_i}{g_i + \lambda}
            - \lambda \lVert w^* \rVert^2\Bigr),
        $$

        whose first term (MacKay's effective number of parameters
        $\gamma$) decreases in $\lambda$ and whose second increases, so the
        evidence is unimodal in $\log \lambda$ and its maximizer is the
        unique root, found by bisection. Since $\gamma \le P$ the root lies
        below $P / \lVert w^* \rVert^2$; the bracket spans $e^{60}$ below
        that.

        Args:
            num_steps: Bisection steps on $\log \lambda$ (each halves the
                bracket of width 60).

        Returns:
            A new `LastLayerLaplace` with the tuned ``prior_precision``.
        """
        gaussx, lx = _require_gaussx()
        if self.structure == "diag":
            g = self.ggn
        else:
            g = gaussx.eigvals(
                lx.MatrixLinearOperator(
                    gaussx.symmetrize(self.ggn), lx.positive_semidefinite_tag
                )
            ).real
        g = jnp.clip(g, 0.0) + self.jitter
        wsq = jnp.maximum(jnp.sum(self.w_map**2), jnp.finfo(self.w_map.dtype).tiny)
        p = self.w_map.shape[0]

        def dlogev(log_lam: Array) -> Array:
            lam = jnp.exp(log_lam)
            return jnp.sum(g / (g + lam)) - lam * wsq

        def step(_: int, bracket: tuple[Array, Array]) -> tuple[Array, Array]:
            lo, hi = bracket
            mid = 0.5 * (lo + hi)
            up = dlogev(mid) > 0
            return jnp.where(up, mid, lo), jnp.where(up, hi, mid)

        hi = jnp.log(p / wsq)
        lo, hi = jax.lax.fori_loop(0, num_steps, step, (hi - 60.0, hi))
        lam = jnp.exp(0.5 * (lo + hi)).astype(self.w_map.dtype)
        return eqx.tree_at(lambda m: m.prior_precision, self, lam)


def last_layer_laplace(
    predict_fn: Callable[[PyTree, Array], Array],
    params: PyTree,
    x: Array,
    y: Array,
    *,
    sigma: Float[Array, ""] | float,
    param_filter: Callable[[tuple, Any], bool],
    prior_precision: Float[Array, ""] | float = 1.0,
    structure: Literal["full", "diag"] = "full",
    jitter: float = 1e-8,
) -> LastLayerLaplace:
    r"""Fit a post-hoc GGN Laplace posterior over a parameter group.

    The fitted model is represented as in `ensemble_map` /
    `ensemble_predict`: a parameter PyTree (one member, no ensemble axis;
    an `equinox.Module` works) and ``predict_fn(params, x) -> f``. The
    weights $w$ selected by ``param_filter`` get a Gaussian posterior; every
    other leaf stays at its MAP value. With $J = \partial f / \partial w$ on
    the training inputs, $R = \sigma^2 I$ and prior $\mathcal{N}(0,
    \lambda^{-1} I)$,

    $$
    \Sigma_w^{-1} = J^\top R^{-1} J + \lambda I, \qquad
    \mathrm{Var}[f(x_*)] = J_* \Sigma_w J_*^\top.
    $$

    For a last layer that is linear in $w$, $J$ is the frozen feature
    matrix and this is exact Bayesian linear regression on those features.
    $J$ is formed densely with forward-mode autodiff, one pass per weight,
    so keep the group small (a last layer, not the network).

    Args:
        predict_fn: ``(params, x) -> f``; any output shape (flattened for
            the Jacobian).
        params: MAP parameters, e.g. one member of `ensemble_map`'s output.
        x: Training inputs.
        y: Training targets, shaped like ``predict_fn(params, x)``.
        sigma: Gaussian observation noise standard deviation.
        param_filter: ``(path, leaf) -> bool`` on ``jax.tree_util`` key
            paths, as `param_group_optimizer`'s ``label_fn``; only inexact
            array leaves are considered.
        prior_precision: Isotropic prior precision $\lambda$.
        structure: ``"full"`` keeps $J^\top J$; ``"diag"`` keeps its
            diagonal (``gaussx.ggn_diagonal``).
        jitter: Added to the diagonal of the precision before factorizing.

    Returns:
        A `LastLayerLaplace` posterior.

    Raises:
        ValueError: If ``structure`` is unknown or ``param_filter`` selects
            no weights.

    Examples:
        >>> post = last_layer_laplace(
        ...     lambda m, x: jax.vmap(m)(x)[:, 0], model, X, y,
        ...     sigma=0.1,
        ...     param_filter=lambda path, _: "layers[-1]" in jax.tree_util.keystr(path),
        ... )
        >>> post = post.optimize_prior_precision()
        >>> mean, var = post.predict(X_new)
    """
    gaussx, _ = _require_gaussx()
    if structure not in ("full", "diag"):
        raise ValueError(f"structure must be 'full' or 'diag', got {structure!r}.")
    w, unravel, rest = _split(params, param_filter)
    if w.shape[0] == 0:
        raise ValueError("param_filter selected no inexact array leaves of params.")
    f = _forward(predict_fn, unravel, rest, x)
    sigma = jnp.asarray(sigma, dtype=w.dtype)
    jac = jax.jacfwd(f)(w)  # (N, P)
    if structure == "diag":
        ggn = gaussx.ggn_diagonal(jac) / sigma**2
    else:
        ggn = einx.dot("n p, n q -> p q", jac, jac) / sigma**2
    log_lik = dist.Normal(f(w), sigma).log_prob(jnp.ravel(y)).sum()
    return LastLayerLaplace(
        predict_fn=predict_fn,
        params=params,
        param_filter=param_filter,
        w_map=w,
        ggn=ggn,
        log_likelihood=log_lik,
        sigma=sigma,
        prior_precision=jnp.asarray(prior_precision, dtype=w.dtype),
        structure=structure,
        jitter=jitter,
    )
