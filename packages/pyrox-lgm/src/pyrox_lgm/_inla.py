r"""The ``inla()`` driver (P8).

Integrated nested Laplace approximations (Rue, Martino & Chopin, 2009) in
three steps:

1. the hyperparameter mode of $\log\tilde\pi(u\mid y) = \log\tilde\pi(y\mid\theta)
   + \log\pi(u)$ (gaussx's Laplace marginal, with exact gradients), by L-BFGS
   in the unconstrained coordinates $u$;
2. an integration design $\{u_k, w_k\}$ around it (`gaussx.theta_design`:
   ``"grid"`` for $m \le 2$, ``"ccd"`` above, ``"eb"`` on request);
3. per design point, the Gaussian approximation of $x\mid y, \theta_k$
   (mode, Takahashi marginal variances kriged onto the hard constraints,
   the low-rank VB mean correction under ``strategy="vb"``), mixed over the
   design with the weights.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from typing import Literal

import equinox as eqx
import gaussx as gx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import ArrayLike

from pyrox_lgm._model import LGM
from pyrox_lgm._result import INLAResult, Summary, mixture_summary


# Components with at most this many nodes get the VB mean correction (with
# the fixed effects), as R-INLA's ``control.vb`` limits it to small effects.
_VB_MAX_NODES = 30


def _vb_subspace(model: LGM) -> np.ndarray:
    idx = []
    for c, (a, b) in zip(model.components, model.slices().values(), strict=False):
        if c.n_nodes <= _VB_MAX_NODES:
            idx.extend(range(a, b))
    n = model.n_latent
    idx.extend(range(n - model.n_fixed, n))
    return np.asarray(idx, dtype=int)


# Module-level compiled pieces. They take the model, the projector and the
# data as arguments (not closures), so a second ``inla()`` on a model of the
# same structure reuses every compilation.
_SOLVER = optax.lbfgs()


def _neg_log_post(model, A, data, max_newton, u):
    return -model.log_posterior_theta(u, data, projector=A, max_newton=max_newton)


@eqx.filter_jit
def _log_post(model, A, data, max_newton, u):
    return model.log_posterior_theta(u, data, projector=A, max_newton=max_newton)


@eqx.filter_jit
def _lbfgs_step(model, A, data, max_newton, u, state):
    def f(v):
        return _neg_log_post(model, A, data, max_newton, v)

    value, grad = optax.value_and_grad_from_state(f)(u, state=state)
    updates, state = _SOLVER.update(grad, state, u, value=value, grad=grad, value_fn=f)
    return optax.apply_updates(u, updates), state, value, grad


@eqx.filter_jit
def _hessian(model, A, data, max_newton, u):
    def lp(v):
        return model.log_posterior_theta(v, data, projector=A, max_newton=max_newton)

    return jax.jacrev(jax.jacrev(lp))(u)


@eqx.filter_jit
def _fit_point(model, A, data, u, n_iter, subspace):
    theta = model.unflatten(u)
    prior = model.latent_prior(theta)
    lik = model.likelihood.build(
        jnp.asarray(data["y"]), model._likelihood_theta(theta), data
    )
    offset = model.offset(data)
    fit = gx.laplace_mode(prior, lik, projector=A, offset=offset, max_iter=n_iter)
    var = fit.factor.diag_inv()
    if isinstance(prior, gx.IntrinsicGMRF):
        V = prior.null_space
        HV = jax.vmap(fit.factor.solve, in_axes=1, out_axes=1)(V)
        S = V.T @ HV
        var = var - jnp.sum(HV * jnp.linalg.solve(S, HV.T).T, axis=1)
    mean = fit.mode
    if subspace:
        mean = gx.vb_mean_correction(
            fit,
            prior,
            lik,
            subspace=jnp.asarray(subspace),
            projector=A,
            offset=offset,
        )
    return mean, var, fit.converged


def _theta_mode(model, A, data, max_newton, u0, *, max_iter, tol, verbose):
    """Minimise the negative log-posterior of ``u`` by optax L-BFGS."""
    u, state = u0, _SOLVER.init(u0)
    for i in range(max_iter):
        u_new, state, value, grad = _lbfgs_step(model, A, data, max_newton, u, state)
        gnorm = float(jnp.max(jnp.abs(grad)))
        if verbose:
            print(
                f"theta-mode iter {i}: -log post {float(value):.6f}, |grad| {gnorm:.2e}"
            )
        if not np.all(np.isfinite(np.asarray(u_new))):
            raise FloatingPointError("theta-mode search produced non-finite values")
        if gnorm < tol:
            return u, i
        u = u_new
    warnings.warn(
        f"theta-mode search did not reach |grad| < {tol} in {max_iter} iterations",
        stacklevel=3,
    )
    return u, max_iter


def inla(
    model: LGM,
    data: Mapping[str, ArrayLike],
    *,
    strategy: Literal["vb", "gaussian"] = "vb",
    integration: Literal["auto", "eb", "grid", "ccd"] = "auto",
    key: jax.Array | None = None,
    max_newton: int = 50,
    theta_init: ArrayLike | None = None,
    max_theta_iter: int = 200,
    theta_tol: float = 1e-5,
    verbose: bool = False,
) -> INLAResult:
    r"""Integrated nested Laplace approximation of a latent Gaussian model.

    Args:
        model: The `LGM`.
        data: ``"y"``, optional ``"offset"`` (and ``"n_trials"`` for a
            binomial model), the node index of each observation per
            component, and the fixed-effect covariates.
        strategy: ``"vb"`` (default, as R-INLA since 22.11) corrects the
            Gaussian approximation's mean by `gaussx.vb_mean_correction` on
            the fixed effects and the components with at most 30 nodes;
            ``"gaussian"`` keeps the Laplace mode.
        integration: The design over $\theta$: ``"auto"`` (``"grid"`` for
            $m \le 2$, ``"ccd"`` above), ``"eb"`` (the mode alone), ``"grid"``
            or ``"ccd"``.
        key: Unused by the deterministic design; reserved for stochastic
            strategies. Accepted so call sites stay stable.
        max_newton: Newton iterations per inner Laplace fit; a design point
            that does not converge is re-run with four times as many, then
            dropped with a warning.
        theta_init: Starting unconstrained $u$ (default zeros: every
            positive hyperparameter at 1, every interval one at its centre).
        max_theta_iter: L-BFGS iterations for the mode.
        theta_tol: Stop when $\max|\nabla_u| <$ ``theta_tol``.
        verbose: Print the mode search.

    Returns:
        An `INLAResult`.

    Examples:
        >>> import jax.numpy as jnp
        >>> import numpy as np
        >>> import pyrox_lgm as lgm
        >>> t = np.arange(30)
        >>> y = np.sin(t / 5.0) + 0.1 * np.cos(7.0 * t)  # smooth + wiggle
        >>> model = lgm.LGM(
        ...     components=(lgm.RW2(30, name="trend"),),
        ...     fixed=lgm.FixedEffects(("intercept",)),
        ...     likelihood=lgm.Gaussian(),
        ... )
        >>> res = lgm.inla(model, {"y": y, "trend": t})
        >>> sorted(res.hyperpar)
        ['lik.prec', 'trend.tau']
        >>> res.random["trend"].mean.shape
        (30,)
    """
    del key
    if strategy not in ("vb", "gaussian"):
        raise ValueError(f"strategy must be 'vb' or 'gaussian', got {strategy!r}")
    A = model.projector(data)  # host work, once
    data = {k: jnp.asarray(v) for k, v in data.items()}
    m = model.n_theta

    def log_post(u):
        return _log_post(model, A, data, max_newton, u)

    # 1. theta-mode, design and marginal likelihood.
    u0 = jnp.zeros(m) if theta_init is None else jnp.asarray(theta_init, dtype=float)
    if m:
        u_star, _ = _theta_mode(
            model,
            A,
            data,
            max_newton,
            u0,
            max_iter=max_theta_iter,
            tol=theta_tol,
            verbose=verbose,
        )
        hessian = _hessian(model, A, data, max_newton, u_star)
        method = None if integration == "auto" else integration
        points, log_w = gx.theta_design(
            log_post, u_star, method=method, hessian=hessian
        )
        neg_h = -hessian
        cov_u = jnp.linalg.inv(neg_h)
        log_ml = _log_marginal_likelihood(log_post, u_star, neg_h, points, log_w)
    else:
        u_star = u0
        points, log_w = u0[None, :], jnp.zeros(1)
        log_ml = log_post(u0)
        cov_u = jnp.zeros((0, 0))

    # 2. per-point Gaussian approximations.
    subspace = tuple(_vb_subspace(model).tolist()) if strategy == "vb" else ()
    means, variances, keep = [], [], []
    for k in range(points.shape[0]):
        mean, var, ok = _fit_point(model, A, data, points[k], max_newton, subspace)
        if not bool(ok):
            mean, var, ok = _fit_point(
                model, A, data, points[k], 4 * max_newton, subspace
            )
        means.append(mean)
        variances.append(var)
        keep.append(bool(ok))
    keep_arr = np.asarray(keep)
    n_dropped = int((~keep_arr).sum())
    if n_dropped == len(keep):
        raise RuntimeError("no design point's inner Newton iteration converged")
    if n_dropped:
        warnings.warn(
            f"dropped {n_dropped} of {len(keep)} design points whose inner Newton "
            "iteration did not converge",
            stacklevel=2,
        )
    idx = jnp.asarray(np.flatnonzero(keep_arr))
    points = points[idx]
    log_w = log_w[idx]
    weights = jnp.exp(log_w - jax.scipy.special.logsumexp(log_w))
    means_arr = jnp.stack(means)[idx]
    vars_arr = jnp.maximum(jnp.stack(variances)[idx], 0.0)

    # 3. mixtures.
    summary = mixture_summary(means_arr, vars_arr, weights)
    random, fixed = {}, {}
    slices = model.slices()
    for c in model.components:
        a, b = slices[c.name]
        random[c.name] = Summary(*(f[a:b] for f in summary))
    if model.fixed is not None:
        for name in model.fixed.names:
            a, _ = slices[name]
            fixed[name] = Summary(*(f[a] for f in summary))
    hyperpar = _hyperpar_summaries(model, u_star, cov_u, points, weights)

    return INLAResult(
        fixed=fixed,
        random=random,
        hyperpar=hyperpar,
        theta_mode=u_star,
        theta_points=points,
        theta_weights=weights,
        latent_means=means_arr,
        latent_variances=vars_arr,
        log_marginal_likelihood=log_ml,
        n_dropped=n_dropped,
        model=model,
        data=data,
        max_newton=max_newton,
    )


def _log_marginal_likelihood(log_post, u_star, neg_h, points, log_w):
    r"""$\log\tilde\pi(y) = \log\int \tilde\pi(u \mid y)\,du$ over the design.

    The Gaussian approximation $\log\tilde\pi(u^\ast\mid y) + \tfrac m2
    \log 2\pi - \tfrac12\log|-\nabla^2|$ is exact for a Gaussian
    $\tilde\pi(u\mid y)$; the design corrects it by the ratio
    $r_k = \tilde\pi(u_k\mid y)/\tilde\pi_G(u_k)$ to that Gaussian. With the
    design weights $w_k \propto \Delta_k\tilde\pi(u_k\mid y)$ (normalised),
    $\int\tilde\pi / \int\tilde\pi_G \approx 1 / \sum_k w_k / r_k$.
    For ``"eb"`` (one point) this is the Gaussian approximation itself.
    """
    m = u_star.shape[0]
    lp_star = log_post(u_star)
    _, logdet = jnp.linalg.slogdet(neg_h)
    log_gauss = lp_star + 0.5 * m * math.log(2.0 * math.pi) - 0.5 * logdet
    lp = jnp.stack([log_post(u) for u in points])  # reuses the compiled log_post
    d = points - u_star
    lp_gauss = lp_star - 0.5 * jnp.einsum("ki,ij,kj->k", d, neg_h, d)
    w = jnp.exp(log_w - jax.scipy.special.logsumexp(log_w))
    return log_gauss - jnp.log(jnp.sum(w * jnp.exp(-(lp - lp_gauss))))


def _hyperpar_summaries(model, u_star, cov_u, points, weights) -> dict[str, Summary]:
    """Hyperparameter summaries on the user scale.

    Means and sds are design-weighted averages of $T(u_k)$; quantiles map the
    Gaussian approximation $u \\sim \\mathcal N(u^\\ast, (-\\nabla^2)^{-1})$
    through each (monotone) bijection, which keeps them exact for that
    approximation.
    """
    spec = model.theta_spec()
    out, i = {}, 0
    z = jnp.asarray([-1.959963984540054, 0.0, 1.959963984540054])
    for key, size, shape in model._sizes():
        transform = spec[key][1]

        def to_user(block, shape=shape, transform=transform):
            return transform(block.reshape(shape) if shape else block[0])

        values = jax.vmap(lambda u, i=i, size=size: to_user(u[i : i + size]))(points)
        w = weights.reshape((-1,) + (1,) * (values.ndim - 1))
        mean = jnp.sum(w * values, axis=0)
        sd = jnp.sqrt(jnp.maximum(jnp.sum(w * values**2, axis=0) - mean**2, 0.0))
        sd_u = jnp.sqrt(jnp.diagonal(cov_u)[i : i + size])
        qs = [to_user(u_star[i : i + size] + zq * sd_u) for zq in z]
        out[key] = Summary(mean, sd, *qs)
        i += size
    return out
