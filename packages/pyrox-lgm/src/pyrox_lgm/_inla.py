r"""The ``inla()`` driver (P8).

Integrated nested Laplace approximations (Rue, Martino & Chopin, 2009) in
three steps:

1. the hyperparameter mode of $\log\tilde\pi(u\mid y) = \log\tilde\pi(y\mid\theta)
   + \log\pi(u)$ (gaussx's Laplace marginal, with exact gradients), by L-BFGS
   in the unconstrained coordinates $u$;
2. an integration design $\{u_k, w_k\}$ around it: by default R-INLA's own
   (`_rinla_design`: its fixed grid for $m \le 2$, a CCD above, both
   stretched by the skewness corrections), or `gaussx.theta_design`'s
   ``"grid"``, ``"ccd"`` or ``"eb"`` on request;
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
from typing import Literal, cast

import equinox as eqx
import gaussx as gx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import ArrayLike

from pyrox_lgm._model import LGM
from pyrox_lgm._result import INLAResult, Summary, mixture_summary


# R-INLA's ``control.vb`` node selection for the VB mean correction
# (``f.enable.limit``): each component contributes at most
# min(30, 1024 / (n_f * n_group * n_rep)) nodes per group and replicate.
_VB_LIMIT = 30
_VB_LIMIT_MAX = 1024


def _vb_layout(c):
    """``(n, n_group, n_rep, index)``: a component's nodes as R-INLA's f()
    sees them, ``index(j, g, r)`` the flat node of main node ``j``."""
    from pyrox_lgm._components._combinators import Kronecker, Replicate

    if isinstance(c, Kronecker):
        n, ng = c.main.n_nodes, c.group.n_nodes
        return n, ng, 1, lambda j, g, r: j * ng + g
    if isinstance(c, Replicate):
        n = c.component.n_nodes
        return n, 1, c.n_rep, lambda j, g, r: r * n + j
    return c.n_nodes, 1, 1, lambda j, g, r: j


def _vb_subspace(model: LGM) -> np.ndarray:
    """The nodes R-INLA's VB mean correction moves: the fixed effects, every
    node of a small component, and ``lim`` evenly spaced nodes of a larger
    one (``j * (n // lim) + max(1, (n // lim) // 2)`` for ``j < lim``)."""
    idx = []
    n_f = len(model.components)
    for c, (a, _) in zip(model.components, model.slices().values(), strict=False):
        n, ng, nrep, index = _vb_layout(c)
        lim = min(_VB_LIMIT, _VB_LIMIT_MAX // (n_f * ng * nrep))
        if lim <= 0:
            continue
        if n <= lim:
            js = np.arange(n)
        else:
            step = max(1, n // lim)
            js = (np.arange(lim) * step + max(1, step // 2)) % n
        for r in range(nrep):
            for g in range(ng):
                idx.extend(a + index(j, g, r) for j in js.tolist())
    n = model.n_latent
    idx.extend(range(n - model.n_fixed, n))
    return np.asarray(sorted(set(idx)), dtype=int)


# Gauss-Hermite nodes per dimension for the empirical-Bayes moments.
_EB_ORDER = 20

# R-INLA's integration over theta, as in its default ("experimental") mode
# since 22.11 (GMRFLib ``design.c`` and ``approx-inference.c``). Designs are
# in the standardised z of theta = theta* + V Lambda^{-1/2} z. For m <= 2 a
# fixed grid with fixed weights, for m > 2 a CCD on the unit sphere; either
# way scaled by f = f0 sqrt(m) and, axis by axis, by the skewness corrections.
_RINLA_F0 = 1.1
_RINLA_X1 = np.array([0.0, -3.5, -2.5, -1.75, -1.0, -0.5, 0.5, 1.0, 1.75, 2.5, 3.5])
_RINLA_W1 = np.array(
    [
        *(1.0, 3.187537795, 1.811358205, 1.937929918, 1.431919577, 1.288639321),
        *(1.288639321, 1.431919577, 1.937929918, 1.811358205, 3.187537795),
    ]
)
# m = 2: the 7 x 7 product grid on these axis values minus its four corners,
# weights by (|z_1|, |z_2|) (symmetric in sign and in the two axes).
_RINLA_AXIS2 = (0.0, 0.5, 1.25, 2.25)
_RINLA_W2 = {
    (0.0, 0.0): 1.0,
    (0.0, 0.5): 0.646540918,
    (0.0, 1.25): 1.17894196,
    (0.0, 2.25): 1.93160554,
    (0.5, 0.5): 0.4180151587,
    (0.5, 1.25): 0.762234217,
    (0.5, 2.25): 1.248862019,
    (1.25, 1.25): 1.389904145,
    (1.25, 2.25): 2.277250821,
}
# Skewness corrections from log pi at z = +-sqrt(2) on each axis.
_SKEW_STEP = math.sqrt(2.0)
# R-INLA drops a design point from the latent mixtures when its density is
# below 1/999 of the mode's ("early stop"), and keeps the largest weights
# that make up 0.999 of the rest (``GMRFLib_weight_prob_one``).
_EARLY_STOP_DROP = math.log(999.0)
_WEIGHT_PROB = 0.999
# Range, in mixture sds, of R-INLA's combined latent marginals.
_RINLA_LIMIT = 5.0


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


def _ccd_unit(m: int) -> np.ndarray:
    """CCD on the unit sphere, centre first: axial points and a resolution-V
    fraction of the 2^m factorial (Sanchez & Sanchez, 2005), as R-INLA's."""
    cols: list[int] = []
    forbidden = {0}
    c = 0
    while len(cols) < m:  # no defining word shorter than five
        c += 1
        if c in forbidden:
            continue
        forbidden |= {c} | {c ^ a for a in cols}
        forbidden |= {c ^ a ^ b for i, a in enumerate(cols) for b in cols[i + 1 :]}
        cols.append(c)
    runs = np.arange(1 << max(cols).bit_length())
    parity = np.array([[bin(r & c).count("1") % 2 for c in cols] for r in runs])
    factorial = (1.0 - 2.0 * parity) / math.sqrt(m)
    axial = np.concatenate([np.eye(m), -np.eye(m)])
    return np.concatenate([np.zeros((1, m)), axial, factorial])


def _rinla_design(m: int) -> tuple[np.ndarray, np.ndarray]:
    r"""R-INLA's design in $z$ before the skewness corrections, centre first.

    Returns ``(x, log_delta)``. For $m \le 2$ its fixed grid (11 and 45
    points) with fixed weights, scaled by $f = f_0\sqrt m$; above, the CCD
    on the sphere of radius $f$ with Rue, Martino & Chopin's (2009, sec. 6.5)
    weights $\Delta_0 = 1 - (K - 1)\Delta$,
    $\Delta = [(K - 1)(1 + e^{-f^2/2}(f^2/m - 1))]^{-1}$.
    """
    f = _RINLA_F0 * math.sqrt(m)
    if m == 1:
        return f * _RINLA_X1[:, None], np.log(_RINLA_W1)
    if m == 2:
        axis = sorted({s * a for a in _RINLA_AXIS2 for s in (1.0, -1.0)})
        pts = [(a, b) for a in axis for b in axis if min(abs(a), abs(b)) < 2.0]
        pts.sort(key=lambda p: p != (0.0, 0.0))  # the mode first
        w = [_RINLA_W2[min(abs(a), abs(b)), max(abs(a), abs(b))] for a, b in pts]
        return f * np.asarray(pts), np.log(np.asarray(w))
    x = _ccd_unit(m)
    k = x.shape[0]
    w = 1.0 / ((k - 1) * (1.0 + math.exp(-0.5 * f * f) * (f * f / m - 1.0)))
    log_delta = np.full(k, math.log(w))
    log_delta[0] = math.log(1.0 - (k - 1) * w)
    return f * x, log_delta


def _eigen_axes(neg_h):
    """``(evals, evecs, scale)`` of ``-H``, with ``u = u* + scale @ z``."""
    evals, evecs = np.linalg.eigh(np.asarray(neg_h))
    return evals, evecs, evecs / np.sqrt(evals)


def _skewness_corrections(log_post, u_star, scale, lp_star):
    r"""R-INLA's per-axis stretch of $z$ on each side of the mode.

    Along each eigen-axis of the Hessian, $\log\tilde\pi$ is evaluated at
    $z = \pm\sqrt2$; a Gaussian drops by exactly 1 there, so
    $\sigma_\pm = (\text{drop}_\pm)^{-1/2}$ is the half-scale that a
    split-normal fitted through those points has (Martins et al., 2013,
    sec. 3.2). As R-INLA's default (``hessian.correct.skewness.only``),
    each pair is divided by its geometric mean, so the corrections change
    only the asymmetry and the Hessian keeps the scale. Returns
    ``(sigma_minus, sigma_plus)``, each of shape ``(m,)``.
    """
    m = u_star.shape[0]
    out = np.ones((2, m))
    for j in range(m):
        for side, sign in enumerate((-1.0, 1.0)):
            drop = lp_star - float(log_post(u_star + sign * _SKEW_STEP * scale[:, j]))
            if drop > 0:
                out[side, j] = 1.0 / math.sqrt(drop)
    out /= np.sqrt(out[0] * out[1])
    return out[0], out[1]


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
    grid_step: float = 1.0,
    grid_threshold: float = 2.5,
    verbose: bool = False,
) -> INLAResult:
    r"""Integrated nested Laplace approximation of a latent Gaussian model.

    Args:
        model: The `LGM`.
        data: ``"y"``, optional ``"offset"`` (and ``"n_trials"`` for a
            binomial model), the node index of each observation per
            component, and the fixed-effect covariates.
        strategy: ``"vb"`` (default, as R-INLA since 22.11) corrects the
            Gaussian approximation's mean by `gaussx.vb_mean_correction` in
            the span of R-INLA's nodes (`_vb_subspace`: the fixed effects,
            every node of a component with at most 30, 30 evenly spaced
            nodes of a larger one);
            ``"gaussian"`` keeps the Laplace mode; ``"sla"`` (simplified
            Laplace) shifts every latent marginal's mean and gives it the
            skewness of the third-order Laplace expansion, as a skew-normal
            (`_simplified_laplace`; it costs one solve per observation and
            leaves a Gaussian likelihood's marginals unchanged).
        integration: The design over $\theta$. ``"auto"`` (default) is
            R-INLA's (``int.strategy = "auto"``, `_rinla_design`): a fixed
            11-point grid for one hyperparameter, a 45-point one for two, a
            CCD above, each stretched along the Hessian's eigen-axes by the
            skewness corrections, with R-INLA's weights, early stop and
            pruning; the latent and predictor summaries are then of each
            mixture on its mean $\pm 5$ sds, as R-INLA reports them. This
            reproduces R-INLA 26.8.7 to about 1e-3 of the latent means
            (``tests/test_rinla_fixtures.py``), including its departures
            from the exact integral over $\theta$: the weights leave out the
            stretch's Jacobian, its two-hyperparameter grid scaled by
            $f_0\sqrt2$ integrates a Gaussian's variance as 0.87, and the
            range limit drops the tails that wide design points add.
            ``"grid"`` and ``"ccd"`` are `gaussx.theta_design`'s unstretched
            designs, summarised as plain mixtures (``"grid"`` refined by
            ``grid_step`` and ``grid_threshold`` converges to the exact
            integral); ``"eb"`` is the mode alone. Under every design but
            ``"eb"`` the hyperparameter marginals are R-INLA's
            (`_theta_marginals`), and the log marginal likelihood is the
            Gaussian approximation corrected by the design
            (`_log_marginal_likelihood`), not R-INLA's integrated estimate,
            whose unnormalised weights bias it by a constant per
            dimension.
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
        grid_step: Step in standardised $z$ of ``integration="grid"``.
        grid_threshold: Log-density drop from the mode at which
            ``integration="grid"`` stops.
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
            grid_step=grid_step,
            grid_threshold=grid_threshold,
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
    grid_step: float = 1.0,
    grid_threshold: float = 2.5,
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
        neg_h = -hessian
        cov_u = jnp.linalg.inv(neg_h)
    else:
        u_star = u0
        neg_h = cov_u = jnp.zeros((0, 0))
    lp_mode = log_post(u_star)
    skew_corr = log_jac = evals = evecs = None
    if m and integration == "auto":
        # R-INLA's design: the z-points stretched axis by axis, each side by
        # its skewness correction. Its weights (and so the mixtures) leave
        # out that stretch's Jacobian, as R-INLA's do; the marginal
        # likelihood below keeps it.
        evals, evecs, scale = _eigen_axes(neg_h)
        skew_corr = _skewness_corrections(log_post, u_star, scale, float(lp_mode))
        x, log_delta_np = _rinla_design(m)
        s_minus, s_plus = skew_corr
        stretch = np.where(x > 0, s_plus, np.where(x < 0, s_minus, 1.0))
        points = u_star + jnp.asarray((x * stretch) @ scale.T)
        log_delta = jnp.asarray(log_delta_np)
        log_jac = jnp.asarray(np.sum(np.log(stretch), axis=1))
    elif m:
        method = cast(Literal["eb", "grid", "ccd"], integration)
        points, log_w = gx.theta_design(
            log_post,
            u_star,
            method=method,
            hessian=hessian,
            grid_step=grid_step,
            grid_threshold=grid_threshold,
        )
        if integration != "eb":  # for the hyperparameter marginals
            evals, evecs, scale = _eigen_axes(neg_h)
            skew_corr = _skewness_corrections(log_post, u_star, scale, float(lp_mode))
        # The design's log-weights are log(Delta_k) + lp_k up to a constant,
        # with lp_k from the max_newton fit; keep log(Delta_k) to reweight.
        log_delta = log_w - jnp.stack([log_post(u) for u in points])
    else:
        points, log_delta = u0[None, :], jnp.zeros(1)

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
    lp_all = jnp.stack(lps)
    lp_star = lps[0] if keep[0] else lp_mode  # point 0 is the mode
    if m:
        idx = jnp.asarray(kept)
        log_w_ml = log_delta[idx] + lp_all[idx]
        if log_jac is not None:
            log_w_ml = log_w_ml + log_jac[idx]
        log_ml = _log_marginal_likelihood(
            lp_star, u_star, neg_h, points[idx], lp_all[idx], log_w_ml
        )
    else:
        log_ml = lp_star
    axis_points = None
    if m == 1 and integration == "auto":
        axis_points = (np.asarray(points[kept, 0]), np.asarray(lp_all[kept] - lp_star))
    if integration == "auto" and m:
        # R-INLA's early stop and pruning of the mixtures.
        log_w = np.asarray(log_delta[kept] + lp_all[kept])
        alive = np.asarray(lp_all[kept]) >= float(lp_star) - _EARLY_STOP_DROP
        w = np.where(alive, np.exp(log_w - log_w[alive].max()), 0.0)
        w = w / w.sum()
        order = np.argsort(-w, kind="stable")
        n_keep = int(np.searchsorted(np.cumsum(w[order]), _WEIGHT_PROB)) + 1
        kept = np.sort(kept[order[:n_keep]])
    idx = jnp.asarray(kept)
    points = points[idx]
    log_w = log_delta[idx] + lp_all[idx]
    weights = jnp.exp(log_w - jax.scipy.special.logsumexp(log_w))
    means_arr = jnp.stack(means)[idx]
    vars_arr = jnp.maximum(jnp.stack(variances)[idx], 0.0)
    skew_arr = jnp.stack(skews)[idx]
    eta_means_arr = jnp.stack(eta_means)[idx]
    eta_vars_arr = jnp.maximum(jnp.stack(eta_vars)[idx], 0.0)
    newton_iters = tuple(iters[k] for k in kept)

    # 3. mixtures.
    # R-INLA's summaries are of its combined marginals, which live on the
    # mixture's mean +- 5 sds; under its design, report them the same way.
    limit = _RINLA_LIMIT if integration == "auto" and m else None
    summary = mixture_summary(
        means_arr, vars_arr, weights, skew_arr if sla else None, limit=limit
    )
    random, fixed = {}, {}
    slices = model.slices()
    for c in model.components:
        a, b = slices[c.name]
        random[c.name] = Summary(*(f[a:b] for f in summary))
    if model.fixed is not None:
        for name in model.fixed.names:
            a, _ = slices[name]
            fixed[name] = Summary(*(f[a] for f in summary))
    marginals = None
    if skew_corr is not None:
        marginals = _theta_marginals(
            u_star, cov_u, evecs, evals, skew_corr, axis_points
        )
    hyperpar = _hyperpar_summaries(model, u_star, cov_u, marginals)

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
        linear_predictor=mixture_summary(
            eta_means_arr, eta_vars_arr, weights, limit=limit
        ),
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
    over the kept points, ``lp`` their converged log-posteriors; for a
    stretched design $\Delta_k$ includes the stretch's Jacobian),
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


def _natural_cubic(x, y, t):
    """Natural cubic spline through ``(x, y)`` at ``t``, constant outside."""
    n = x.size
    h = np.diff(x)
    a = np.zeros((n, n))
    r = np.zeros(n)
    a[0, 0] = a[-1, -1] = 1.0
    for i in range(1, n - 1):
        a[i, i - 1 : i + 2] = h[i - 1], 2.0 * (h[i - 1] + h[i]), h[i]
        r[i] = 3.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
    c = np.linalg.solve(a, r)
    b = np.diff(y) / h - h * (2.0 * c[:-1] + c[1:]) / 3.0
    d = np.diff(c) / (3.0 * h)
    t = np.clip(t, x[0], x[-1])
    k = np.clip(np.searchsorted(x, t) - 1, 0, n - 2)
    dt = t - x[k]
    return y[k] + dt * (b[k] + dt * (c[k] + dt * d[k]))


def _theta_marginals(u_star, cov_u, evecs, evals, skew, axis_points=None):
    r"""Each hyperparameter's marginal on a grid of its unconstrained $u_j$.

    As R-INLA: for one hyperparameter, ``axis_points`` ``(u_k, lp_k - lp*)``
    from the design, interpolated as a spline-corrected Gaussian (its
    ``GRIDSUM``); otherwise the split-normal in the Hessian's eigenbasis,
    $\log\tilde\pi(z) = -\tfrac12\sum_i (z_i/\sigma_{i,\pm})^2$ with the
    skewness corrections, along the line $u = u^\ast + \Sigma_{\cdot j}
    (u_j - u^\ast_j)/\Sigma_{jj}$ of the conditional means (its ``CCD``
    interpolator; Martins et al., 2013, sec. 3.2). Returns
    ``(grid, prob)``, both ``(m, G)``, ``prob`` normalised per row.
    """
    u_star, cov_u = np.asarray(u_star), np.asarray(cov_u)
    m = u_star.shape[0]
    sd = np.sqrt(np.diag(cov_u))
    s_minus, s_plus = skew
    g = np.linspace(-1.0, 1.0, 4001) * 10.0 * max(1.0, s_minus.max(), s_plus.max())
    grid, prob = np.empty((m, g.size)), np.empty((m, g.size))
    for j in range(m):
        if axis_points is not None:
            xk = (axis_points[0] - u_star[j]) / sd[j]
            order = np.argsort(xk)
            corr = axis_points[1][order] + 0.5 * xk[order] ** 2
            ld = -0.5 * g**2 + _natural_cubic(xk[order], corr, g)
        else:
            du = np.outer(g * sd[j], cov_u[:, j] / cov_u[j, j])
            z = du @ (evecs * np.sqrt(evals))
            ld = -0.5 * np.sum((z / np.where(z > 0, s_plus, s_minus)) ** 2, axis=1)
        p = np.exp(ld - ld.max())
        grid[j], prob[j] = u_star[j] + sd[j] * g, p / p.sum()
    return grid, prob


def _hyperpar_summaries(model, u_star, cov_u, marginals=None) -> dict[str, Summary]:
    """Hyperparameter summaries on the user scale.

    With ``marginals`` (`_theta_marginals`), the mean, sd and quantiles of
    each hyperparameter's marginal pushed through its (elementwise,
    monotone) bijection. Without (empirical Bayes), those of the Gaussian
    approximation $u \\sim \\mathcal N(u^\\ast, (-\\nabla^2)^{-1})$.
    """
    spec = model.theta_spec()
    out, i = {}, 0
    z = jnp.asarray([-1.959963984540054, 0.0, 1.959963984540054])
    for key, size, shape in model._sizes():
        transform = spec[key][1]

        def to_user(block, shape=shape, transform=transform):
            return transform(block.reshape(shape) if shape else block[0])

        if marginals is not None:
            grid, prob = marginals[0][i : i + size], marginals[1][i : i + size]
            cdf = np.cumsum(prob, axis=1) - 0.5 * prob
            qs = [
                to_user(
                    jnp.asarray(
                        [np.interp(p, c, gr) for c, gr in zip(cdf, grid, strict=True)]
                    )
                )
                for p in (0.025, 0.5, 0.975)
            ]
            means, sds = [], []
            for j in range(size):
                block = np.repeat(
                    np.asarray(u_star[i : i + size])[None], grid.shape[1], 0
                )
                block[:, j] = grid[j]
                v = np.asarray(jax.vmap(to_user)(jnp.asarray(block))).reshape(-1, size)[
                    :, j
                ]
                mean = float(np.sum(prob[j] * v))
                means.append(mean)
                sds.append(math.sqrt(max(float(np.sum(prob[j] * v**2)) - mean**2, 0.0)))
            mean = jnp.asarray(means).reshape(shape)
            sd = jnp.asarray(sds).reshape(shape)
        else:
            # Empirical Bayes: the Gaussian approximation pushed through the
            # bijection (a tensor Gauss-Hermite rule on the block's marginal;
            # for exp, E = exp(mu + sigma^2 / 2)).
            sd_u = jnp.sqrt(jnp.diagonal(cov_u)[i : i + size])
            qs = [to_user(u_star[i : i + size] + zq * sd_u) for zq in z]
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
