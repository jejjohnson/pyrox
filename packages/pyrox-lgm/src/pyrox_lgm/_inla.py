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
   the low-rank VB mean correction under ``strategy="vb"``, the
   simplified-Laplace skew-normal under ``strategy="sla"``), mixed over the
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

# Gauss-Hermite nodes per dimension for the empirical-Bayes moments.
_EB_ORDER = 20

# R-INLA's integration-grid defaults (control.inla dz and diff.logdens).
_GRID_STEP = 0.75
_GRID_THRESHOLD = 6.0

# Draws of the skewness-corrected hyperparameter posterior (fixed key, so
# the summaries are deterministic).
_HYPER_DRAWS = 100_000


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
    # +inf where the inner Newton fit has not converged: the line search
    # rejects such a candidate, and _theta_mode fails on an accepted one,
    # so no unconverged fit steers the mode.
    fit = model.laplace(u, data, projector=A, max_newton=max_newton)
    value = -(fit.log_marginal + model.log_prior(u))
    return jnp.where(fit.converged, value, jnp.inf)


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
def _fit_point(model, A, data, u, n_iter, subspace, sla=False):
    theta = model.unflatten(u)
    prior = model.latent_prior(theta)
    lik = model.likelihood.build(
        jnp.asarray(data["y"]), model._likelihood_theta(theta), data
    )
    offset = model.offset(data)
    fit = gx.laplace_mode(prior, lik, projector=A, offset=offset, max_iter=n_iter)
    var = fit.factor.diag_inv()
    G = HV = None
    if isinstance(prior, gx.IntrinsicGMRF):
        V = prior.null_space
        HV = jax.vmap(fit.factor.solve, in_axes=1, out_axes=1)(V)
        S = V.T @ HV
        G = jnp.linalg.solve(S, HV.T).T  # Sigma_c = H^-1 - G HV^T
        var = var - jnp.sum(HV * G, axis=1)
    mean = fit.mode
    skew = jnp.zeros_like(var)
    if sla:
        mean, skew = _simplified_laplace(fit, lik, A, offset, var, G, HV)
    if subspace:
        mean = gx.vb_mean_correction(
            fit,
            prior,
            lik,
            subspace=jnp.asarray(subspace),
            projector=A,
            offset=offset,
        )
    eta_mean = A.mv(mean) + (0.0 if offset is None else offset)
    eta_var = _predictor_variance(A, fit.factor.selected_inverse(), G, HV)
    return (
        mean,
        var,
        fit.converged,
        fit.log_marginal + model.log_prior(u),
        eta_mean,
        eta_var,
        skew,
    )


def _simplified_laplace(fit, lik, A, offset, var, G, HV):
    r"""Simplified-Laplace mean shift and skewness of every latent marginal.

    The Laplace marginal $\pi(x_i) \propto \pi(x, y)/\pi_G(x_{-i}\mid x_i)$,
    taken at the Gaussian conditional mean and expanded to third order in
    the standardised $z = (x_i - \mu_i)/\sigma_i$ (Rue, Martino & Chopin,
    2009, sec. 3.2.3), is
    $-\tfrac12 z^2 + \gamma_1 z + \tfrac16\gamma_3 z^3$ with

    $$
    \gamma_1 = \tfrac12\sum_j d_j\,(s_j^2 b_{ij} - b_{ij}^3),\qquad
    \gamma_3 = \sum_j d_j\, b_{ij}^3,
    $$

    $d_j$ the third derivative of $\log p(y_j\mid\eta_j)$ at the mode,
    $b_{ij} = \operatorname{cov}(x_i, \eta_j)/\sigma_i$ and $s_j^2$ the
    variance of $\eta_j$: the cubic term is the likelihood's own, the
    linear one the change of $\log|H_{-i}|$ along the conditional mean.
    To first order that density has mean $\mu_i + \sigma_i(\gamma_1 +
    \gamma_3/2)$, variance $\sigma_i^2$ and skewness $\gamma_3$.

    ``cov(x, eta) = Sigma A^T`` takes one solve per observation (kriged
    onto the hard constraints), so the cost is ``n_obs`` solves.
    """
    eta_hat = A.mv(fit.mode) + (0.0 if offset is None else offset)
    d1 = jax.grad(lik.log_prob)
    d2 = jax.grad(lambda e: jnp.sum(d1(e)))
    d3 = jax.grad(lambda e: jnp.sum(d2(e)))(eta_hat)  # sites are independent
    At = A.as_matrix().T
    C = jax.vmap(fit.factor.solve, in_axes=1, out_axes=1)(At)  # H^-1 A^T
    if G is not None:  # Sigma_c A^T = H^-1 A^T - G (A H^-1 V)^T
        C = C - G @ (At.T @ HV).T
    sd = jnp.sqrt(jnp.maximum(var, 0.0))
    B = C / jnp.where(sd > 0, sd, 1.0)[:, None]
    eta_var = jnp.sum(At * C, axis=0)
    gamma1 = 0.5 * ((B * eta_var[None, :] - B**3) @ d3)
    gamma3 = (B**3) @ d3
    shift = jnp.where(sd > 0, sd * (gamma1 + 0.5 * gamma3), 0.0)
    return fit.mode + shift, jnp.where(sd > 0, gamma3, 0.0)


def _predictor_variance(A, selected, G, HV):
    r"""$\operatorname{var}(\eta_i) = \sum_{a,b} A_{ia}A_{ib}\Sigma_{ab}$ per row.

    Every pair of nodes in a row of $A$ is coupled in $A^\top WA$, so
    $\Sigma_{ab}$ is in the Takahashi selected inverse (pattern of
    $L + L^\top$); the hard constraints subtract $G_a\cdot(H^{-1}V)_b$.
    The pairs and their positions in the selected pattern are host work on
    static patterns.
    """
    rows = np.asarray(A.pattern.rows)
    cols = np.asarray(A.pattern.cols)
    order = np.argsort(rows, kind="stable")
    rows_s, cols_s = rows[order], cols[order]
    m = A.out_size()
    counts = np.bincount(rows_s, minlength=m)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    # All (p, q) entry pairs within each row.
    pair_row = np.repeat(np.arange(m), counts**2)
    local = np.arange(pair_row.size) - np.repeat(
        np.cumsum(counts**2) - counts**2, counts**2
    )
    c_row = counts[pair_row]
    p = starts[pair_row] + local // np.maximum(c_row, 1)
    q = starts[pair_row] + local % np.maximum(c_row, 1)
    a, b = cols_s[p], cols_s[q]
    coef = A.values[jnp.asarray(order[p])] * A.values[jnp.asarray(order[q])]
    sr = np.asarray(selected.pattern.rows)
    sc = np.asarray(selected.pattern.cols)
    n = A.in_size()
    lookup = {}
    for idx, key in enumerate((sr.astype(np.int64) * n + sc).tolist()):
        lookup[key] = idx
    for idx, key in enumerate((sc.astype(np.int64) * n + sr).tolist()):
        lookup.setdefault(key, idx)
    pos = np.fromiter(
        (lookup[k] for k in (a.astype(np.int64) * n + b).tolist()),
        dtype=np.int64,
        count=a.size,
    )
    sigma = selected.values[jnp.asarray(pos)]
    if G is not None:
        sigma = sigma - jnp.sum(G[jnp.asarray(a)] * HV[jnp.asarray(b)], axis=1)
    return jax.ops.segment_sum(coef * sigma, jnp.asarray(pair_row), m)


def _theta_mode(model, A, data, max_newton, u0, *, max_iter, tol, verbose):
    """Minimise the negative log-posterior of ``u`` by optax L-BFGS.

    Returns the mode, the iterations taken, and whether every accepted
    iterate's inner fit converged (``False`` asks `inla` to escalate).

    Raises:
        RuntimeError: If the gradient tolerance is not met in ``max_iter``.
    """
    u, state = u0, _SOLVER.init(u0)
    for i in range(max_iter):
        u_new, state, value, grad = _lbfgs_step(model, A, data, max_newton, u, state)
        if not np.isfinite(float(value)):  # the inner fit at u did not converge
            return u, i, False
        gnorm = float(jnp.max(jnp.abs(grad)))
        if verbose:
            print(
                f"theta-mode iter {i}: -log post {float(value):.6f}, |grad| {gnorm:.2e}"
            )
        if not np.all(np.isfinite(np.asarray(u_new))):
            raise FloatingPointError("theta-mode search produced non-finite values")
        if gnorm < tol:
            return u, i, True
        u = u_new
    raise RuntimeError(
        f"theta-mode search did not reach |grad| < {tol} in {max_iter} "
        "iterations; the Hessian and design would be built off the mode. Raise "
        "max_theta_iter, loosen theta_tol or pass a closer theta_init"
    )


def inla(
    model: LGM,
    data: Mapping[str, ArrayLike],
    *,
    strategy: Literal["vb", "gaussian", "sla"] = "vb",
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
            ``"gaussian"`` keeps the Laplace mode; ``"sla"`` (simplified
            Laplace) shifts every latent marginal's mean and gives it the
            skewness of the third-order Laplace expansion, as a skew-normal
            (`_simplified_laplace`; it costs one solve per observation and
            leaves a Gaussian likelihood's marginals unchanged).
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
        max_theta_iter: L-BFGS iterations for the mode; not reaching
            ``theta_tol`` within them raises.
        theta_tol: Stop when $\max|\nabla_u| <$ ``theta_tol``.
        verbose: Print the mode search.

    Returns:
        An `INLAResult`.

    Raises:
        RuntimeError: If the theta-mode search misses ``theta_tol`` within
            ``max_theta_iter``, or the inner Newton fits along it or at the
            mode do not converge even at four times ``max_newton``.

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

    def run(budget: int) -> tuple[INLAResult | None, bool]:
        return _inla_once(
            model,
            data,
            strategy=strategy,
            integration=integration,
            key=key,
            max_newton=budget,
            theta_init=theta_init,
            max_theta_iter=max_theta_iter,
            theta_tol=theta_tol,
            verbose=verbose,
        )

    result, mode_ok = run(max_newton)
    if mode_ok and result is not None:
        return result
    budget = 4 * max_newton
    warnings.warn(
        f"the inner Newton fit at the theta-mode or along its search did not "
        f"converge in {max_newton} iterations; re-running the whole fit with "
        f"max_newton={budget}",
        stacklevel=2,
    )
    result, mode_ok = run(budget)
    if not mode_ok or result is None:
        raise RuntimeError(
            f"the inner Newton fit at the theta-mode or along its search did not "
            f"converge in {budget} iterations; raise max_newton or check the model"
        )
    return result


def _inla_once(
    model: LGM,
    data: Mapping[str, ArrayLike],
    *,
    strategy: Literal["vb", "gaussian", "sla"] = "vb",
    integration: Literal["auto", "eb", "grid", "ccd"] = "auto",
    key: jax.Array | None = None,
    max_newton: int = 50,
    theta_init: ArrayLike | None = None,
    max_theta_iter: int = 200,
    theta_tol: float = 1e-5,
    verbose: bool = False,
) -> tuple[INLAResult | None, bool]:
    """One fit at a fixed Newton budget; see `inla`."""
    del key
    if strategy not in ("vb", "gaussian", "sla"):
        raise ValueError(
            f"strategy must be 'vb', 'gaussian' or 'sla', got {strategy!r}"
        )
    A = model.projector(data)  # host work, once
    data = {k: jnp.asarray(v) for k, v in data.items()}
    m = model.n_theta

    def log_post(u):
        return _log_post(model, A, data, max_newton, u)

    # 1. theta-mode and design.
    u0 = jnp.zeros(m) if theta_init is None else jnp.asarray(theta_init, dtype=float)
    if u0.shape != (m,):
        raise ValueError(f"theta_init must have shape ({m},), got {u0.shape}")
    if m:
        u_star, _, search_ok = _theta_mode(
            model,
            A,
            data,
            max_newton,
            u0,
            max_iter=max_theta_iter,
            tol=theta_tol,
            verbose=verbose,
        )
        if not search_ok:
            return None, False  # inla() escalates the budget or raises
        hessian = _hessian(model, A, data, max_newton, u_star)
        method = None if integration == "auto" else integration
        # R-INLA's grid: steps of dz = 0.75 out to a log-density drop of 6.
        points, log_w = gx.theta_design(
            log_post,
            u_star,
            method=method,
            hessian=hessian,
            grid_step=_GRID_STEP,
            grid_threshold=_GRID_THRESHOLD,
        )
        neg_h = -hessian
        cov_u = jnp.linalg.inv(neg_h)
        theta_skew = (
            None if points.shape[0] == 1 else _skew_scales(log_post, u_star, neg_h)
        )
    else:
        u_star = u0
        points, log_w = u0[None, :], jnp.zeros(1)
        neg_h = cov_u = jnp.zeros((0, 0))
        theta_skew = None
    # The design's log-weights are log(Delta_k) + lp_k up to a constant, with
    # lp_k from the max_newton fit; keep log(Delta_k) to reweight below.
    lp_design = jnp.stack([log_post(u) for u in points])
    log_delta = log_w - lp_design

    # 2. per-point Gaussian approximations; a point that needs the retry is
    # reweighted with its converged log-posterior, one that never converges
    # is dropped from the mixture and from the marginal likelihood.
    subspace = tuple(_vb_subspace(model).tolist()) if strategy == "vb" else ()
    sla = strategy == "sla"
    means, variances, lps, iters, keep = [], [], [], [], []
    eta_means, eta_vars, skews = [], [], []
    for k in range(points.shape[0]):
        n_iter = max_newton
        mean, var, ok, lp, eta_m, eta_v, skew = _fit_point(
            model, A, data, points[k], n_iter, subspace, sla
        )
        if not bool(ok):
            n_iter = 4 * max_newton
            mean, var, ok, lp, eta_m, eta_v, skew = _fit_point(
                model, A, data, points[k], n_iter, subspace, sla
            )
        means.append(mean)
        skews.append(skew)
        eta_means.append(eta_m)
        eta_vars.append(eta_v)
        variances.append(var)
        lps.append(lp)
        iters.append(n_iter)
        keep.append(bool(ok))
    keep_arr = np.asarray(keep)
    n_dropped = int((~keep_arr).sum())
    if n_dropped == len(keep):
        return None, False  # the mode failed too: inla() escalates or raises
    if n_dropped:
        warnings.warn(
            f"dropped {n_dropped} of {len(keep)} design points whose inner Newton "
            "iteration did not converge",
            stacklevel=2,
        )
    kept = np.flatnonzero(keep_arr)
    idx = jnp.asarray(kept)
    points = points[idx]
    lp_kept = jnp.stack(lps)[idx]
    log_w = log_delta[idx] + lp_kept
    weights = jnp.exp(log_w - jax.scipy.special.logsumexp(log_w))
    means_arr = jnp.stack(means)[idx]
    vars_arr = jnp.maximum(jnp.stack(variances)[idx], 0.0)
    skew_arr = jnp.stack(skews)[idx]
    eta_means_arr = jnp.stack(eta_means)[idx]
    eta_vars_arr = jnp.maximum(jnp.stack(eta_vars)[idx], 0.0)
    newton_iters = tuple(iters[k] for k in kept)
    lp_star = lps[0] if keep[0] else lp_design[0]  # point 0 is the mode
    if m:
        log_ml = _log_marginal_likelihood(
            lp_star, u_star, neg_h, points, lp_kept, log_w
        )
    else:
        log_ml = lp_star

    # 3. mixtures.
    summary = mixture_summary(means_arr, vars_arr, weights, skew_arr if sla else None)
    random, fixed = {}, {}
    slices = model.slices()
    for c in model.components:
        a, b = slices[c.name]
        random[c.name] = Summary(*(f[a:b] for f in summary))
    if model.fixed is not None:
        for name in model.fixed.names:
            a, _ = slices[name]
            fixed[name] = Summary(*(f[a] for f in summary))
    hyperpar = _hyperpar_summaries(
        model, u_star, cov_u, points, weights, theta_skew
    )

    result = INLAResult(
        fixed=fixed,
        random=random,
        hyperpar=hyperpar,
        theta_mode=u_star,
        theta_points=points,
        theta_weights=weights,
        latent_means=means_arr,
        latent_variances=vars_arr,
        latent_skewness=skew_arr,
        predictor_means=eta_means_arr,
        predictor_variances=eta_vars_arr,
        linear_predictor=mixture_summary(eta_means_arr, eta_vars_arr, weights),
        log_marginal_likelihood=log_ml,
        n_dropped=n_dropped,
        model=model,
        data=data,
        newton_iters=newton_iters,
    )
    # The mode search, Hessian and design all ran on max_newton fits: if the
    # mode's own fit needed the retry (or never converged), they are suspect.
    mode_ok = bool(keep[0]) and iters[0] == max_newton
    return result, mode_ok


def _log_marginal_likelihood(lp_star, u_star, neg_h, points, lp, log_w):
    r"""$\log\tilde\pi(y) = \log\int \tilde\pi(u \mid y)\,du$ over the design.

    The Gaussian approximation $\log\tilde\pi(u^\ast\mid y) + \tfrac m2
    \log 2\pi - \tfrac12\log|-\nabla^2|$ is exact for a Gaussian
    $\tilde\pi(u\mid y)$; the design corrects it by the ratio
    $r_k = \tilde\pi(u_k\mid y)/\tilde\pi_G(u_k)$ to that Gaussian. With the
    design weights $w_k \propto \Delta_k\tilde\pi(u_k\mid y)$ (normalised
    over the kept points, ``lp`` their converged log-posteriors),
    $\int\tilde\pi / \int\tilde\pi_G \approx 1 / \sum_k w_k / r_k$. For
    ``"eb"`` (one point) this is the Gaussian approximation itself.
    """
    m = u_star.shape[0]
    _, logdet = jnp.linalg.slogdet(neg_h)
    log_gauss = lp_star + 0.5 * m * math.log(2.0 * math.pi) - 0.5 * logdet
    d = points - u_star
    lp_gauss = lp_star - 0.5 * jnp.einsum("ki,ij,kj->k", d, neg_h, d)
    log_wn = log_w - jax.scipy.special.logsumexp(log_w)
    return log_gauss - jax.scipy.special.logsumexp(log_wn - (lp - lp_gauss))


def _skew_scales(log_post, u_star, neg_h):
    r"""R-INLA's skewness corrections along the Hessian's eigenvectors.

    With $-\nabla^2 = V\Lambda V^\top$ and $u(z) = u^\ast + V\Lambda^{-1/2}z$,
    the posterior is evaluated at $z = \pm\sqrt2\,e_k$; a Gaussian drops by
    exactly one log-unit there, so the half-scales
    $s_k^\pm = 1/\sqrt{-\Delta\log\tilde\pi}$ measure how much heavier
    (or lighter) each side is (Martins et al., 2013, sec. 3.2).
    """
    lam, V = jnp.linalg.eigh(neg_h)
    B = V / jnp.sqrt(lam)[None, :]  # u = u* + B z
    lp0 = log_post(u_star)
    scales = []
    for k in range(u_star.shape[0]):
        for sign in (1.0, -1.0):
            drop = lp0 - log_post(u_star + sign * math.sqrt(2.0) * B[:, k])
            scales.append(1.0 / jnp.sqrt(jnp.clip(drop, 1e-2, 1e2)))
    s = jnp.reshape(jnp.stack(scales), (-1, 2))
    return B, s[:, 0], s[:, 1]


def _hyperpar_summaries(
    model, u_star, cov_u, points, weights, skew=None
) -> dict[str, Summary]:
    """Hyperparameter summaries on the user scale.

    With the skewness corrections of `_skew_scales` (any design beyond the
    mode), the hyperparameter posterior is R-INLA's split-normal in the
    standardised coordinates, independent per eigen-direction with scales
    $s_k^\\pm$; mean, sd and quantiles are those of its push-forward through
    the bijections, by a fixed-key Monte Carlo. Under ``"eb"`` they are
    those of the Gaussian approximation
    $u \\sim \\mathcal N(u^\\ast, (-\\nabla^2)^{-1})$.
    """
    spec = model.theta_spec()
    if skew is not None:
        B, s_pos, s_neg = skew
        k_side, k_mag = jax.random.split(jax.random.key(0))
        shape = (_HYPER_DRAWS, u_star.shape[0])
        positive = jax.random.uniform(k_side, shape) < s_pos / (s_pos + s_neg)
        mag = jnp.abs(jax.random.normal(k_mag, shape))
        z = jnp.where(positive, s_pos * mag, -s_neg * mag)
        draws = u_star + z @ B.T
    out, i = {}, 0
    z = jnp.asarray([-1.959963984540054, 0.0, 1.959963984540054])
    for key, size, shape in model._sizes():
        transform = spec[key][1]

        def to_user(block, shape=shape, transform=transform):
            return transform(block.reshape(shape) if shape else block[0])

        if skew is not None:
            values = jax.vmap(to_user)(draws[:, i : i + size])
            q = jnp.quantile(values, jnp.asarray([0.025, 0.5, 0.975]), axis=0)
            out[key] = Summary(
                jnp.mean(values, axis=0), jnp.std(values, axis=0), q[0], q[1], q[2]
            )
            i += size
            continue
        sd_u = jnp.sqrt(jnp.diagonal(cov_u)[i : i + size])
        qs = [to_user(u_star[i : i + size] + zq * sd_u) for zq in z]
        if points.shape[0] > 1:  # a design without skewness corrections
            values = jax.vmap(lambda u, i=i, size=size: to_user(u[i : i + size]))(
                points
            )
            w = weights.reshape((-1,) + (1,) * (values.ndim - 1))
            mean = jnp.sum(w * values, axis=0)
            var = jnp.sum(w * values**2, axis=0) - mean**2
            sd = jnp.sqrt(jnp.maximum(var, 0.0))
        else:
            # Empirical Bayes: one point carries no spread, so take the mean
            # and sd of the same Gaussian approximation the quantiles come
            # from, pushed through the bijection (a tensor Gauss-Hermite rule
            # on the block's marginal; for exp, E = exp(mu + sigma^2 / 2)).
            nodes, gh = np.polynomial.hermite_e.hermegauss(_EB_ORDER)
            grid = np.stack(np.meshgrid(*[nodes] * size, indexing="ij"), -1)
            wts = np.prod(np.meshgrid(*[gh] * size, indexing="ij"), axis=0)
            wts = jnp.asarray((wts / wts.sum()).reshape(-1))
            L = jnp.linalg.cholesky(cov_u[i : i + size, i : i + size])
            us = u_star[i : i + size] + jnp.asarray(grid.reshape(-1, size)) @ L.T
            values = jax.vmap(to_user)(us)
            w = wts.reshape((-1,) + (1,) * (values.ndim - 1))
            mean = jnp.sum(w * values, axis=0)
            var = jnp.sum(w * values**2, axis=0) - mean**2
            sd = jnp.sqrt(jnp.maximum(var, 0.0))
        out[key] = Summary(mean, sd, *qs)
        i += size
    return out
