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
    Jacobian $\partial f / \partial w$ at the linearization point $w^*$ on
    the training inputs and Gaussian noise $\sigma^2$, the log-likelihood is
    replaced by its Gauss-Newton quadratic model around $w^*$,

    $$
    \log p(y \mid w) \approx \ell^* + b^\top \Delta
        - \tfrac{1}{2} \Delta^\top G \Delta, \qquad
    \Delta = w - w^*,\;
    G = \sigma^{-2} J^\top J + \epsilon I,\;
    b = \sigma^{-2} J^\top (y - f^*),
    $$

    with $\epsilon$ the ``jitter``, which is exact for a model linear in
    $w$. Under the prior $\mathcal{N}(0, \lambda^{-1} I)$ the posterior
    precision is $H_\lambda = G + \lambda I$ and its mode is
    $w_\lambda = H_\lambda^{-1} (G w^* + b)$. $G$ is kept whole
    (``structure="full"``) or as its diagonal (``structure="diag"``). The
    posterior registers no NumPyro sites.

    Attributes:
        predict_fn: ``(params, x) -> f``, the same predictive as
            `ensemble_predict` takes.
        params: The fitted parameters; their selected weights are $w^*$.
        param_filter: ``(path, leaf) -> bool`` selecting the weights $w$.
        w_map: The linearization point $w^*$, flattened, shape ``(P,)``.
        w_mean: The posterior mean, shape ``(P,)``: $w^*$ as fitted, or
            $w_\lambda$ after `optimize_prior_precision`.
        ggn: $\sigma^{-2} J^\top J$ (without jitter), shape ``(P, P)``
            (full) or ``(P,)`` (diag).
        loglik_grad: $b$, the log-likelihood gradient at $w^*$, ``(P,)``.
        log_likelihood: $\ell^* = \log p(y \mid w^*)$ on the training data.
        sigma: Observation noise standard deviation $\sigma$.
        prior_precision: Isotropic prior precision $\lambda$.
        structure: ``"full"`` or ``"diag"``.
        jitter: $\epsilon$, added to the diagonal of $G$, and so of $H$,
            everywhere it is used.
    """

    predict_fn: Callable[[PyTree, Array], Array] = eqx.field(static=True)
    params: PyTree
    param_filter: Callable[[tuple, Any], bool] = eqx.field(static=True)
    w_map: Float[Array, " P"]
    w_mean: Float[Array, " P"]
    ggn: Float[Array, "P P"] | Float[Array, " P"]
    loglik_grad: Float[Array, " P"]
    log_likelihood: Float[Array, ""]
    sigma: Float[Array, ""]
    prior_precision: Float[Array, ""]
    structure: Literal["full", "diag"] = eqx.field(static=True)
    jitter: float = eqx.field(static=True)

    def _lam(self, prior_precision: Array | float | None) -> Array:
        if prior_precision is None:
            return self.prior_precision
        return jnp.asarray(prior_precision, dtype=self.w_map.dtype)

    def _g_matvec(self, v: Array) -> Array:
        """$G v$, jitter included."""
        if self.structure == "diag":
            return (self.ggn + self.jitter) * v
        return einx.dot("p q, q -> p", self.ggn, v) + self.jitter * v

    def precision(self, prior_precision: Array | float | None = None) -> Any:
        r"""The posterior precision $H_\lambda = G + \lambda I$.

        Args:
            prior_precision: $\lambda$; defaults to the posterior's own.

        Returns:
            A PSD-tagged ``lineax`` operator of size ``(P, P)``: a
            ``MatrixLinearOperator`` (full) or a ``DiagonalLinearOperator``
            (diag).
        """
        gaussx, lx = _require_gaussx()
        shift = self._lam(prior_precision) + self.jitter
        if self.structure == "diag":
            return lx.DiagonalLinearOperator(self.ggn + shift)
        p = self.w_map.shape[0]
        h = gaussx.symmetrize(self.ggn) + shift * jnp.eye(p, dtype=self.ggn.dtype)
        return lx.MatrixLinearOperator(h, lx.positive_semidefinite_tag)

    def mode(self, prior_precision: Array | float | None = None) -> Array:
        r"""$w_\lambda = H_\lambda^{-1} (G w^* + b)$, shape ``(P,)``.

        For a model linear in $w$ this is the exact posterior mean under
        $\lambda$, wherever $w^*$ was fitted.

        Args:
            prior_precision: $\lambda$; defaults to the posterior's own.

        Returns:
            The mode of the quadratic-model posterior.
        """
        gaussx, _ = _require_gaussx()
        rhs = self._g_matvec(self.w_map) + self.loglik_grad
        return gaussx.solve(self.precision(prior_precision), rhs)

    def predict(self, x: Array) -> tuple[Array, Array]:
        r"""Linearized predictive mean and variance of $f$ at ``x``.

        $$
        \mathbb{E}[f(x_*)] = f(x_*; w^*) + J_* (\bar w - w^*), \qquad
        \mathrm{Var}[f(x_*)] = J_* H_\lambda^{-1} J_*^\top,
        $$

        with $J_* = \partial f(x_*) / \partial w$ at $w^*$ and $\bar w$
        the posterior mean ``w_mean``. The variance is of the latent $f$;
        add ``sigma**2`` for the predictive of $y$.

        Args:
            x: Inputs, in whatever form ``predict_fn`` takes.

        Returns:
            ``(mean, var)``, each shaped like ``predict_fn(params, x)``.
        """
        gaussx, _ = _require_gaussx()
        _, unravel, rest = _split(self.params, self.param_filter)
        f = _forward(self.predict_fn, unravel, rest, x)
        f_map = self.predict_fn(self.params, x)
        jac = jax.jacfwd(f)(self.w_map)  # (M, P)
        shift = einx.dot("m p, p -> m", jac, self.w_mean - self.w_map)
        jac_t = einx.id("m p -> p m", jac)
        h = self.precision()
        if self.structure == "diag":
            # H⁻¹ is diagonal: Var = Σ_p J_*² / h_p.
            var = einx.sum("[p] m", jac_t * gaussx.solve_columns(h, jac_t))
        else:
            # H = L Lᵀ, so J_* H⁻¹ J_*ᵀ has diagonal Σ_p (L⁻¹ J_*ᵀ)².
            v = gaussx.solve_columns(gaussx.cholesky(h), jac_t)
            var = einx.sum("[p] m", v**2)
        mean = f_map + jnp.reshape(shift, f_map.shape)
        return mean, jnp.reshape(var, f_map.shape)

    def log_marginal_likelihood(
        self, prior_precision: Array | float | None = None
    ) -> Float[Array, ""]:
        r"""Laplace evidence $\log p(y \mid \lambda)$ of the quadratic model.

        $$
        \log p(y \mid \lambda) \approx \ell^* + b^\top \Delta
            - \tfrac{1}{2} \Delta^\top G \Delta
            - \tfrac{\lambda}{2} \lVert w_\lambda \rVert^2
            + \tfrac{P}{2} \log \lambda
            - \tfrac{1}{2} \log\det H_\lambda,
        \qquad \Delta = w_\lambda - w^*.
        $$

        The mode $w_\lambda$ is re-solved for each $\lambda$, so for a
        model linear in $w$ this is the exact Bayesian linear-regression
        evidence at every $\lambda$ (the jitter counts as part of $G$), not
        only at the $\lambda$ that produced $w^*$.

        Args:
            prior_precision: $\lambda$ to evaluate at; defaults to the
                posterior's own.

        Returns:
            Scalar log evidence.
        """
        gaussx, _ = _require_gaussx()
        lam = self._lam(prior_precision)
        w_lam = self.mode(lam)
        delta = w_lam - self.w_map
        p = self.w_map.shape[0]
        return (
            self.log_likelihood
            + jnp.dot(self.loglik_grad, delta)
            - 0.5 * jnp.dot(delta, self._g_matvec(delta))
            - 0.5 * lam * jnp.sum(w_lam**2)
            + 0.5 * p * jnp.log(lam)
            - 0.5 * gaussx.logdet(self.precision(lam))
        )

    def optimize_prior_precision(
        self, *, num_grid: int = 4097, num_refine: int = 100
    ) -> LastLayerLaplace:
        r"""Return a copy with $\lambda$ maximizing the Laplace evidence.

        In the eigenbasis $G = Q \operatorname{diag}(g) Q^\top$ (the
        identity for ``"diag"``), with $c = Q^\top (G w^* + b)$,
        $d = Q^\top w^*$, $\beta = Q^\top b$ and
        $\omega = c / (g + \lambda)$ the rotated mode, the evidence of
        `log_marginal_likelihood` is the $O(P)$ scalar function

        $$
        \ell^* + \beta^\top (\omega - d)
            - \tfrac{1}{2} \sum_i g_i (\omega_i - d_i)^2
            - \tfrac{\lambda}{2} \lVert \omega \rVert^2
            + \tfrac{P}{2} \log \lambda
            - \tfrac{1}{2} \sum_i \log (g_i + \lambda).
        $$

        It is maximized over $\log \lambda$ on a grid spanning half the
        dtype's exponent range each way (so $\lambda$ and every term stay
        finite, even when $w^* = 0$), then refined by golden-section search
        in the grid cell either side of the best point. For a model linear
        in $w$ this maximizes the exact evidence.

        Args:
            num_grid: Grid points on $\log \lambda$.
            num_refine: Golden-section steps (each shrinks the bracket by
                0.618).

        Returns:
            A new `LastLayerLaplace` with the tuned ``prior_precision`` and
            ``w_mean`` $= w_\lambda$ at it.
        """
        gaussx, lx = _require_gaussx()
        dtype = self.w_map.dtype
        if self.structure == "diag":
            g = self.ggn
            d, beta = self.w_map, self.loglik_grad
        else:
            g, q = gaussx.eig(
                lx.MatrixLinearOperator(
                    gaussx.symmetrize(self.ggn), lx.positive_semidefinite_tag
                )
            )
            d = einx.dot("p k, p -> k", q, self.w_map)
            beta = einx.dot("p k, p -> k", q, self.loglik_grad)
        # Round-off can leave eigenvalues of the PSD G slightly negative.
        g = jnp.clip(g.astype(dtype), 0.0) + self.jitter
        c = g * d + beta
        p = self.w_map.shape[0]

        def evidence(log_lam: Array) -> Array:
            lam = jnp.exp(log_lam)
            omega = c / (g + lam)
            delta = omega - d
            return (
                jnp.dot(beta, delta)
                - 0.5 * jnp.sum(g * delta**2)
                - 0.5 * lam * jnp.sum(omega**2)
                + 0.5 * p * log_lam
                - 0.5 * jnp.sum(jnp.log(g + lam))
            )

        bound = 0.5 * jnp.log(jnp.finfo(dtype).max)
        grid = jnp.linspace(-bound, bound, num_grid, dtype=dtype)
        best = jnp.argmax(jax.vmap(evidence)(grid))
        step = grid[1] - grid[0]
        lo = jnp.maximum(grid[best] - step, -bound)
        hi = jnp.minimum(grid[best] + step, bound)
        ratio = (jnp.sqrt(5.0) - 1.0) / 2.0

        def golden(_: int, bracket: tuple[Array, Array]) -> tuple[Array, Array]:
            a, b = bracket
            x1 = b - ratio * (b - a)
            x2 = a + ratio * (b - a)
            left = evidence(x1) >= evidence(x2)
            return jnp.where(left, a, x1), jnp.where(left, x2, b)

        lo, hi = jax.lax.fori_loop(0, num_refine, golden, (lo, hi))
        lam = jnp.exp(0.5 * (lo + hi)).astype(dtype)
        return eqx.tree_at(
            lambda m: (m.prior_precision, m.w_mean), self, (lam, self.mode(lam))
        )


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
        jitter: Added to the diagonal of the GGN (so of the precision)
            before factorizing, consistently in every use.

    Returns:
        A `LastLayerLaplace` posterior.

    Raises:
        ValueError: If ``structure`` is unknown or ``param_filter`` selects
            no weights.

    Examples:
        >>> import equinox as eqx
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> from pyrox.inference import last_layer_laplace
        >>> model = eqx.nn.MLP(1, 1, 16, 2, activation=jnp.sin, key=jr.key(0))
        >>> X = jnp.linspace(-1.0, 1.0, 50)[:, None]
        >>> y = jnp.sin(3.0 * X[:, 0])
        >>> def predict(m, x):
        ...     return jax.vmap(m)(x)[:, 0]
        >>> # Key paths spell list indices out (".layers[2]"), so build the
        >>> # selector from the index rather than writing "layers[-1]".
        >>> last = f".layers[{len(model.layers) - 1}]"
        >>> def is_last(path, _leaf):
        ...     return jax.tree_util.keystr(path).startswith(last)
        >>> post = last_layer_laplace(
        ...     predict, model, X, y, sigma=0.1, param_filter=is_last
        ... )
        >>> post.w_map.shape  # 16 weights + 1 bias
        (17,)
        >>> post = post.optimize_prior_precision()
        >>> mean, var = post.predict(jnp.linspace(-1.5, 1.5, 7)[:, None])
        >>> mean.shape, var.shape
        ((7,), (7,))
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
    f_map = f(w)
    if structure == "diag":
        ggn = gaussx.ggn_diagonal(jac) / sigma**2
    else:
        # The dense P x P GGN, not gaussx.gauss_newton_precision's Woodbury
        # form: optimize_prior_precision eigendecomposes G once to search
        # lambda in O(P) per step, and storing G (P^2) instead of J (N P)
        # is what lets N, the number of field samples, far exceed P.
        ggn = einx.dot("n p, n q -> p q", jac, jac) / sigma**2
    resid = jnp.ravel(y) - f_map
    loglik_grad = einx.dot("n p, n -> p", jac, resid) / sigma**2
    log_lik = dist.Normal(f_map, sigma).log_prob(jnp.ravel(y)).sum()
    return LastLayerLaplace(
        predict_fn=predict_fn,
        params=params,
        param_filter=param_filter,
        w_map=w,
        w_mean=w,
        ggn=ggn,
        loglik_grad=loglik_grad,
        log_likelihood=log_lik,
        sigma=sigma,
        prior_precision=jnp.asarray(prior_precision, dtype=w.dtype),
        structure=structure,
        jitter=jitter,
    )
