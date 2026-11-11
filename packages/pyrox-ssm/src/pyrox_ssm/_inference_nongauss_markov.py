"""Site-based non-Gaussian inference for `MarkovGPPrior`.

Couples the site-based view from `pyrox_gp._inference_nongauss`
(each likelihood factor contributes a diagonal Gaussian site with
naturals $(\\lambda, \\Lambda)$) to the linear-time Kalman /
RTS-smoother backbone in `pyrox_ssm._markov`.

Each iteration runs the filter + smoother once with per-step
pseudo-observation variance $R_n = 1/\\Lambda_n$ and pseudo-target
$\\tilde y_n = \\lambda_n / \\Lambda_n$ to obtain the marginal
posterior $q(f_n) = \\mathcal{N}(m_n, V_n)$ on the training grid;
strategies differ only in how new site naturals are obtained from the
local likelihood given ``(m_n, V_n)``. Cost is $O(I N d^3)$ for
``I`` iterations on ``N`` time points and SDE state dimension ``d`` —
linear in ``N``, where the dense path is $O(I N^3)$.

Predictions reuse the same Kalman trick: the merged grid
``sort(times \\cup t_star)`` with the test points masked produces
self-consistent smoothed marginals at arbitrary test times.

Every ``fit`` takes an optional observation ``mask``. A masked site has
zero gradient and zero curvature, so its naturals are
$(\\lambda_n, \\Lambda_n) = (0, 0)$ and the filter skips its update:
masking step $k$ fits the same sites at the kept steps as dropping it.

Each strategy's iteration is a pure map $s \\mapsto T_\\alpha(s)$ on the site
naturals $s = (\\lambda, \\Lambda)$, with the damping $\\alpha$ inside it,
$T_\\alpha(s) = (1 - \\alpha)\\, s + \\alpha\\, T(s)$, and the loop is
`optimistix.fixed_point` with `optimistix.FixedPointIteration`. A fit therefore
traces under `jax.jit`, and its gradient with respect to the prior's
hyperparameters $\\theta$ comes from the implicit function theorem at the fixed
point $s^\\star$ (`optimistix.ImplicitAdjoint`, the default ``adjoint``),

$$
\\frac{\\partial s^\\star}{\\partial \\theta}
  = \\Big(I - \\frac{\\partial T_\\alpha}{\\partial s}\\Big)^{-1}
    \\frac{\\partial T_\\alpha}{\\partial \\theta}\\Big|_{s^\\star},
$$

which costs one linear solve however many iterations ran; the default solves
it matrix-free with GMRES, from Jacobian-vector products of the step. Damping
does not move the fixed point, so the gradient does not depend on $\\alpha$
either. It is the gradient at a fixed point only when the loop converged: check
``converged``.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import lineax as lx
import numpyro
import optimistix as optx
from gaussx import (
    AbstractIntegrator,
    CallableLikelihood,
    GaussHermiteIntegrator,
    cavity_distribution,
    damped_natural_update,
    ep_tilted_moments,
    newton_update,
)
from jaxtyping import Array, Bool, Float, Int, PyTree
from pyrox.likelihoods import Likelihood

from pyrox_ssm._markov import (
    _build_dt_full,
    _is_stationary,
    _kalman_filter,
    _observation_mask,
    _require_stationary,
    _rts_smoother,
)
from pyrox_ssm._markov_sites import (
    _cavity_site_derivatives,
    _require_gauss_hermite,
    as_gaussx_likelihood,
)


if TYPE_CHECKING:
    from pyrox_ssm._markov import MarkovGPPrior


# --- core smoother helpers ----------------------------------------------


def _prior_marginal_variance(prior: MarkovGPPrior) -> Float[Array, " N"]:
    r"""Prior marginal variance ``Var[f(t_n)]`` per training time.

    Stationary 1-D SDE kernels have a *constant* marginal variance equal
    to $H P_\infty H^\top$ (equivalent to the kernel ``variance``); we
    broadcast it to ``(N,)`` so it can seed the cavity-update loops in PL
    and EP without hard-coding ``1.0`` on rescaled kernels.

    A non-stationary kernel has no such constant — the marginal variance
    of an integrated Wiener process grows with $t$ — so the prior is
    propagated through the Kalman recursion with every observation masked
    out, which reports $H P_k H^\top$ per step and costs one
    $O(N d^3)$ pass. That route would give the same answer for a
    stationary kernel, but the closed form is exact and free, so it is
    kept for the case that has one.
    """
    _F, _L, H, _Qc, P_inf = prior.sde_kernel.sde_params()
    if _is_stationary(prior.sde_kernel):
        P_inf = _require_stationary(prior.sde_kernel, P_inf)
        var0 = (H @ P_inf @ H.T)[0, 0]
        return jnp.broadcast_to(var0, prior.times.shape)

    times = prior.times
    dt_full = _build_dt_full(times)
    A_seq, Q_seq = prior.sde_kernel.discretise_sequence(dt_full)
    zeros = jnp.zeros_like(times)
    # mask = 0 everywhere: no update ever fires, so the filter propagates
    # the prior alone and the residual / R values below are inert. R must
    # still be finite to keep NaN out of the (skipped) update.
    _m_pred, P_pred, _m_filt, _P_filt, _ll = _kalman_filter(
        _F,
        H,
        prior.initial_covariance(),
        A_seq,
        Q_seq,
        zeros,
        zeros,
        jnp.ones_like(times),
    )
    return jax.vmap(lambda P: (H @ P @ H.T)[0, 0])(P_pred)


def _mask_sites(
    nat1: Float[Array, " N"],
    nat2: Float[Array, " N"],
    observed: Bool[Array, " N"],
    precision_floor: float,
) -> tuple[Float[Array, " N"], Float[Array, " N"]]:
    r"""Floor the observed site precisions and zero the masked sites.

    An observed site keeps $(\lambda_n, \max(\Lambda_n, \epsilon))$; a
    masked site becomes $(0, 0)$, the site of a likelihood term that is not
    there.
    """
    zero = jnp.zeros((), dtype=nat2.dtype)
    nat1 = jnp.where(observed, nat1, zero)
    nat2 = jnp.where(observed, jnp.maximum(nat2, precision_floor), zero)
    return nat1, nat2


def _markov_smoothed_posterior(
    prior: MarkovGPPrior,
    nat1: Float[Array, " N"],
    nat2: Float[Array, " N"],
    observed: Bool[Array, " N"] | None = None,
) -> tuple[Float[Array, " N"], Float[Array, " N"], Float[Array, ""]]:
    r"""Smoothed marginals on the training grid given diagonal site naturals.

    Treats the sites as a synthetic Gaussian likelihood with per-step
    variance $1/\Lambda_n$ and target $\lambda_n / \Lambda_n$
    (centred on the prior mean), then runs filter + RTS smoother. Returns
    ``(f_mean, f_var, log_marg)`` over the training times where
    ``log_marg`` is the Kalman log-likelihood of the *pseudo-observations*
    (used as a Laplace-style approximation to the true marginal). A step
    with ``observed`` ``False`` runs the predict only; its zero site is
    replaced by a unit placeholder first, so no ``0 / 0`` is ever formed.
    """
    F, _L, H, _Qc, _P_inf = prior.sde_kernel.sde_params()
    init_cov = prior.initial_covariance()
    times = prior.times
    dt_full = _build_dt_full(times)
    A_seq, Q_seq = prior.sde_kernel.discretise_sequence(dt_full)
    observed = _observation_mask(None, times) if observed is None else observed
    site_prec = jnp.where(observed, nat2, jnp.ones((), dtype=nat2.dtype))
    R_seq = jnp.reciprocal(site_prec)
    pseudo_targets = nat1 / site_prec
    residual = pseudo_targets - prior.mean(times)
    m_pred, P_pred, m_filt, P_filt, log_marg = _kalman_filter(
        F, H, init_cov, A_seq, Q_seq, residual, observed, R_seq
    )
    m_smooth, P_smooth = _rts_smoother(m_pred, P_pred, m_filt, P_filt, A_seq)
    f_mean = (m_smooth @ H.T)[:, 0] + prior.mean(times)
    f_var = jax.vmap(lambda P: (H @ P @ H.T)[0, 0])(P_smooth)
    f_var = jnp.maximum(f_var, 1e-12)
    return f_mean, f_var, log_marg


_Sites = tuple[Float[Array, " N"], Float[Array, " N"]]


class _NonFiniteSafeSolver(lx.AbstractLinearSolver):
    r"""A linear solver for the implicit adjoint that never raises mid-sampler.

    The implicit adjoint solves $(I - \partial T_\alpha / \partial s)^\top v = c$
    at the fixed point. lineax raises whenever a solve fails, which ends a NUTS
    chain at the first point that fails. Two failures are therefore turned into
    values here:

    * Where the forward fit is already non-finite (an SDE discretisation that
      overflows at an extreme lengthscale, say) the solution is zero, so the
      value stays non-finite and NUTS rejects the point as it would any other.
    * An iterative solve that stops at ``max_steps`` or stagnates keeps its last
      iterate, as the forward loop keeps its last sites with ``throw=False``.

    Every other result is reported exactly as ``solver`` reports it.

    Attributes:
        solver: The solver used for finite systems.
    """

    solver: lx.AbstractLinearSolver

    def init(self, operator, options):
        return self.solver.init(operator, options)

    def compute(self, state, vector, options):
        solution, result, stats = self.solver.compute(state, vector, options)
        leaves = jtu.tree_leaves(eqx.filter((state, vector), eqx.is_inexact_array))
        broken = jnp.logical_not(
            jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in leaves]))
        )
        solution = jtu.tree_map(
            lambda x: jnp.where(broken, jnp.zeros_like(x), x), solution
        )
        stopped = (result == lx.RESULTS.max_steps_reached) | (
            result == lx.RESULTS.stagnation
        )
        result = lx.RESULTS.where(broken | stopped, lx.RESULTS.successful, result)
        return solution, result, stats

    def transpose(self, state, options):
        return self.solver.transpose(state, options)

    def conj(self, state, options):
        return self.solver.conj(state, options)

    def assume_full_rank(self):
        return self.solver.assume_full_rank()


# The default adjoint: optimistix's implicit adjoint with a matrix-free solve.
# I - dT/ds is not symmetric, so GMRES, which needs only products with the
# operator (JVPs of the step, and VJPs for its transpose in reverse mode): no
# (2N, 2N) Jacobian is formed, and each product costs one Kalman smoother pass.
_IMPLICIT_ADJOINT = optx.ImplicitAdjoint(
    linear_solver=_NonFiniteSafeSolver(
        lx.GMRES(rtol=1e-10, atol=1e-10, restart=20, max_steps=50)
    )
)


def _solve_sites(
    step: Callable[[_Sites, PyTree], _Sites],
    sites: _Sites,
    args: PyTree,
    *,
    tol: float,
    max_steps: int,
    adjoint: optx.AbstractAdjoint,
) -> tuple[_Sites, Int[Array, ""], Bool[Array, ""]]:
    r"""Iterate ``step`` on the site naturals to a fixed point.

    Stops at the first iteration $k$ whose change in every site natural is
    within a tenth of ``tol`` of the new value, absolutely and relatively,

    $$
    \max_n \frac{|s^{(k)}_n - s^{(k-1)}_n|}
      {\tfrac{1}{10}\,\mathrm{tol}\,(1 + |s^{(k)}_n|)} < 1,
    $$

    over both $\lambda$ and $\Lambda$, or after ``max_steps`` iterations. A
    masked site holds $(0, 0)$ throughout, so it never holds the loop up. The
    tenth keeps the posterior marginals within about ``tol`` of the fixed point,
    where the Python loops this replaced stopped once the posterior mean moved
    by less than ``tol``: a damped map contracts slowly, and the sites settle
    one iteration behind the mean.

    Args:
        step: The damped site update $T_\alpha$, ``step(sites, args) -> sites``.
        sites: The starting naturals $(\lambda, \Lambda)$, each ``(N,)``.
        args: Passed to ``step``; the arrays the gradient flows to.
        tol: The tolerance above.
        max_steps: The most iterations to run.
        adjoint: How gradients pass through the loop.

    Returns:
        The naturals after the last iteration, the number of iterations run, and
        whether the criterion was met (``False`` also when an iterate turned
        non-finite).
    """
    solver = optx.FixedPointIteration(rtol=0.1 * tol, atol=0.1 * tol)
    sol = optx.fixed_point(
        step,
        solver,
        sites,
        args,
        max_steps=max_steps,
        adjoint=adjoint,
        throw=False,
    )
    return sol.value, sol.stats["num_steps"], sol.result == optx.RESULTS.successful


def _site_corrected_log_marginal(
    log_marg: Float[Array, ""],
    q_mean: Float[Array, " N"],
    nat1: Float[Array, " N"],
    nat2: Float[Array, " N"],
    site_lik: CallableLikelihood,
    observed: Bool[Array, " N"],
) -> Float[Array, ""]:
    r"""Turn the pseudo-observation Kalman marginal into an evidence estimate.

    With Gaussian sites $\mathcal{N}(\tilde y_n \mid f_n, \Lambda_n^{-1})$,
    $\tilde y_n = \lambda_n / \Lambda_n$, the Kalman filter returns
    $\log \mathcal{N}(\tilde{\mathbf{y}} \mid \boldsymbol{\mu}, K + \Lambda^{-1})$,
    the marginal of the pseudo-data rather than of $\mathbf{y}$. Swapping
    the pseudo-data fit for the data fit at the posterior mean $m$,

    $$
    \log p(\mathbf{y}) \approx
      \log \mathcal{N}(\tilde{\mathbf{y}} \mid \boldsymbol{\mu}, K + \Lambda^{-1})
      + \sum_n o_n \log p(y_n \mid m_n)
      - \sum_n o_n \log \mathcal{N}(\tilde y_n \mid m_n, \Lambda_n^{-1}),
    $$

    gives the Laplace-style approximation every strategy reports, as the
    dense strategies in pyrox-gp do. Both sums run over the observed sites
    $o_n = 1$ only.
    """
    ll_data = site_lik.log_prob(q_mean)
    site_prec = jnp.where(observed, nat2, jnp.ones((), dtype=nat2.dtype))
    pseudo_targets = nat1 / site_prec
    ll_pseudo = -0.5 * jnp.sum(
        jnp.where(observed, jnp.log(2.0 * jnp.pi / site_prec), 0.0)
    ) - 0.5 * jnp.sum(
        jnp.where(observed, site_prec * (q_mean - pseudo_targets) ** 2, 0.0)
    )
    return log_marg + ll_data - ll_pseudo


# --- result type ---------------------------------------------------------


class NonGaussConditionedMarkovGP(eqx.Module):
    """Markov GP conditioned on a non-Gaussian likelihood via a site-based strategy.

    Equivalent role to `pyrox_ssm.ConditionedMarkovGP` but the
    posterior over training latents is a generic Gaussian
    approximation produced by one of the strategies in this module
    rather than the closed-form Gaussian-likelihood smoother. Predictions
    at arbitrary test times use the standard *site-as-pseudo-observation*
    trick on the merged grid ``sort(times | t_star)`` with the test
    points masked.

    Attributes:
        prior: The originating `MarkovGPPrior`.
        y: Training targets (kept for round-trip / diagnostics).
        site_nat1: Diagonal site naturals $\\lambda \\in \\mathbb{R}^N$.
        site_nat2: Diagonal site precisions $\\Lambda \\in \\mathbb{R}^N$
            (positive).
        q_mean: Posterior mean over training latents.
        q_var: Marginal posterior variance per training point.
        log_marginal_approx: Approximate log marginal likelihood
            $\\log p(y)$: the Kalman marginal of the fitted Gaussian sites with
            the pseudo-data fit swapped for the data fit at the posterior
            mean, the same Laplace-style correction for every strategy.
        n_iter: Iterations used by the strategy, a ``()`` integer array (a
            traced value under `jax.jit`).
        converged: Whether the strategy's ``tol`` was met within ``max_iter``
            iterations, a ``()`` boolean array (a traced value under
            `jax.jit`). Gradients of a fit that did not converge are not those
            of a fixed point.
        mask: Optional ``(N,)`` boolean observation mask the strategy was
            fitted with; masked sites hold zero naturals and `predict`
            leaves them out of the update. ``None`` means every site is
            observed.
    """

    prior: MarkovGPPrior
    y: Float[Array, " N"]
    site_nat1: Float[Array, " N"]
    site_nat2: Float[Array, " N"]
    q_mean: Float[Array, " N"]
    q_var: Float[Array, " N"]
    log_marginal_approx: Float[Array, ""]
    n_iter: Int[Array, ""]
    converged: Bool[Array, ""]
    mask: Bool[Array, " N"] | None = None

    def predict(
        self,
        t_star: Float[Array, " M"],
    ) -> tuple[Float[Array, " M"], Float[Array, " M"]]:
        r"""Predictive marginals ``(mean, var)`` at arbitrary test times.

        Re-runs filter + smoother over the merged grid
        ``sort(times \cup t_star)`` with per-step pseudo-observation
        variances ``1/Λ_n`` on the training points and the test points
        masked out of the update step. Cost is $O((N + M)\,d^3)$.
        """
        F, _L, H, _Qc, _P_inf = self.prior.sde_kernel.sde_params()
        init_cov = self.prior.initial_covariance()
        times = self.prior.times
        t_star = jnp.asarray(t_star)

        N = times.shape[0]
        M = t_star.shape[0]
        merged = jnp.concatenate([times, t_star], axis=0)
        order = jnp.argsort(merged, stable=True)
        merged_sorted = merged[order]

        observed = _observation_mask(self.mask, times)
        is_obs = jnp.concatenate(
            [observed.astype(times.dtype), jnp.zeros(M, dtype=times.dtype)]
        )[order]
        # A masked training site holds Λ = 0; a unit placeholder keeps
        # λ / Λ and 1 / Λ finite, and the filter skips that step anyway.
        site_prec = jnp.where(
            observed, self.site_nat2, jnp.ones((), dtype=self.site_nat2.dtype)
        )
        pseudo_targets = self.site_nat1 / site_prec
        residual_full = jnp.concatenate(
            [
                pseudo_targets - self.prior.mean(times),
                jnp.zeros(M, dtype=self.y.dtype),
            ]
        )[order]
        # ``R_seq`` for masked steps is irrelevant (the update is
        # skipped) but must be finite to avoid NaN propagation through
        # the filter; use 1.0 as a safe placeholder.
        R_train = jnp.reciprocal(site_prec)
        R_full = jnp.concatenate([R_train, jnp.ones(M, dtype=self.y.dtype)])[order]

        dt_full = _build_dt_full(merged_sorted)
        A_seq, Q_seq = self.prior.sde_kernel.discretise_sequence(dt_full)
        m_pred, P_pred, m_filt, P_filt, _ = _kalman_filter(
            F, H, init_cov, A_seq, Q_seq, residual_full, is_obs, R_full
        )
        m_smooth, P_smooth = _rts_smoother(m_pred, P_pred, m_filt, P_filt, A_seq)

        inv_order = jnp.argsort(order, stable=True)
        test_positions = inv_order[N:]
        m_test_state = m_smooth[test_positions]
        P_test_state = P_smooth[test_positions]
        means = (m_test_state @ H.T)[:, 0] + self.prior.mean(t_star)
        vars_ = jax.vmap(lambda P: (H @ P @ H.T)[0, 0])(P_test_state)
        return means, jnp.maximum(vars_, 0.0)


# --- Laplace -------------------------------------------------------------


class LaplaceMarkovInference(eqx.Module):
    r"""Laplace approximation for `MarkovGPPrior`.

    Fixed-point Newton on the smoothed posterior: each iteration runs
    filter + smoother with the current sites to obtain the posterior mean
    $\hat f$, then sets the site naturals from the per-point gradient
    $g_n$ and curvature $h_n$ of $\log p(y \mid f)$ at $\hat f$,

    $$
    T(s) = (g - h \odot \hat f,\; -h), \qquad
    s \leftarrow (1 - \alpha)\, s + \alpha\, T(s),
    $$

    with the site precision $\Lambda = -h$ clipped to ``precision_floor``.
    The loop is `optimistix.fixed_point`, so a fit traces under `jax.jit` and
    differentiates through ``adjoint`` (see the module notes).

    Attributes:
        max_iter: Newton iterations. Default ``20``.
        tol: Convergence tolerance: the loop stops at the first iteration
            where every site natural $\lambda_n$, $\Lambda_n$ changed by less
            than ``0.1 * tol * (1 + |new value|)``, which leaves the posterior
            marginals within about ``tol`` of the fixed point. Default
            ``1e-6``.
        damping: Step-size $\alpha$ in (0, 1] applied to each Newton update.
            Default ``1.0`` (full Newton step). Drop below 1 for
            non-log-concave likelihoods.
        precision_floor: Lower bound on the diagonal precision. Default
            ``1e-6``.
        adjoint: How gradients pass through the loop. Default an
            ``optimistix.ImplicitAdjoint``, the implicit function theorem at the
            fixed point, solved matrix-free by ``lineax.GMRES`` (``rtol`` and
            ``atol`` ``1e-10``) from products with the step's Jacobian, so a
            gradient costs a few Kalman passes and $O(N)$ memory; a fit that is
            already non-finite gets a non-finite gradient rather than an error.
            For small $N$, ``optimistix.ImplicitAdjoint()`` solves the dense
            ``(2N, 2N)`` Jacobian by LU instead, and
            ``optimistix.RecursiveCheckpointAdjoint()`` differentiates through
            the iterations.
    """

    max_iter: int = eqx.field(static=True, default=20)
    tol: float = eqx.field(static=True, default=1e-6)
    damping: float = eqx.field(static=True, default=1.0)
    precision_floor: float = eqx.field(static=True, default=1e-6)
    adjoint: optx.AbstractAdjoint = _IMPLICIT_ADJOINT

    def _newton_sites(
        self,
        f: Float[Array, " N"],
        site_lik: CallableLikelihood,
        observed: Bool[Array, " N"],
    ) -> _Sites:
        r"""The undamped Newton sites $T(s)$ at the posterior mean ``f``."""
        g, h = site_lik.site_derivatives(f)
        nat1, Lam = newton_update(f, g, h, precision_floor=self.precision_floor)
        return _mask_sites(nat1, Lam, observed, self.precision_floor)

    def _step(self, sites: _Sites, args: PyTree) -> _Sites:
        r"""One damped Newton step on the sites, $T_\alpha(s)$."""
        prior, site_lik, observed = args
        nat1, nat2 = sites
        f, _, _ = _markov_smoothed_posterior(prior, nat1, nat2, observed)
        new_nat1, new_nat2 = self._newton_sites(f, site_lik, observed)
        nat1, nat2 = damped_natural_update(
            nat1, nat2, new_nat1, new_nat2, lr=self.damping
        )
        assert isinstance(nat2, jax.Array)  # pyrox sites are diagonal arrays
        return _mask_sites(nat1, nat2, observed, self.precision_floor)

    def fit(
        self,
        prior: MarkovGPPrior,
        likelihood: Likelihood,
        y: Float[Array, " N"],
        *,
        mask: Bool[Array, " N"] | None = None,
    ) -> NonGaussConditionedMarkovGP:
        r"""Fit the sites by Newton steps on the smoothed posterior mean.

        Args:
            prior: The `pyrox_ssm.MarkovGPPrior` over the observation times.
            likelihood: Any scalar-latent `pyrox.likelihoods.Likelihood`.
            y: Observations of shape ``(N,)``; masked entries may be ``NaN``.
            mask: Optional ``(N,)`` boolean mask, ``True`` where observed. A
                masked site contributes $g_n = h_n = 0$, so its naturals are
                $(\lambda_n, \Lambda_n) = (0, 0)$ and the filter skips it.

        Returns:
            The `NonGaussConditionedMarkovGP` with the fitted sites.

        Examples:
            >>> import jax
            >>> import jax.numpy as jnp
            >>> from pyrox_gp import BernoulliLikelihood
            >>> from pyrox_ssm import MaternSDE, MarkovGPPrior
            >>> from pyrox_ssm import LaplaceMarkovInference
            >>> prior = MarkovGPPrior(MaternSDE(1.0, 0.5, order=1), jnp.arange(4.0))
            >>> y = jnp.array([1.0, 0.0, jnp.nan, 1.0])
            >>> mask = jnp.array([True, True, False, True])
            >>> strategy = LaplaceMarkovInference()
            >>> cond = strategy.fit(prior, BernoulliLikelihood(), y, mask=mask)

            The fit traces, so the Laplace evidence jits and differentiates:

            >>> def evidence(ell):
            ...     p = MarkovGPPrior(MaternSDE(1.0, ell, order=1), jnp.arange(4.0))
            ...     fit = strategy.fit(p, BernoulliLikelihood(), y, mask=mask)
            ...     return fit.log_marginal_approx
            >>> grad = jax.jit(jax.grad(evidence))(0.5)
        """
        # Always the masked likelihood, so that mask=None and an all-True mask
        # compile the same loop and agree bit for bit.
        observed = _observation_mask(mask, prior.times)
        site_lik = as_gaussx_likelihood(likelihood, y, mask=observed)

        N = prior.times.shape[0]
        prior_mean = prior.mean(prior.times)
        nat1 = jnp.asarray(prior_mean) * self.precision_floor
        nat2 = jnp.full((N,), self.precision_floor, dtype=prior_mean.dtype)
        sites = _mask_sites(nat1, nat2, observed, self.precision_floor)

        sites, n_iter, converged = _solve_sites(
            self._step,
            sites,
            (prior, site_lik, observed),
            tol=self.tol,
            max_steps=self.max_iter,
            adjoint=self.adjoint,
        )

        # Final site naturals: the undamped Newton sites at the last mean.
        f, _, _ = _markov_smoothed_posterior(prior, *sites, observed)
        nat1, nat2 = self._newton_sites(f, site_lik, observed)
        q_mean, q_var, log_marg = _markov_smoothed_posterior(
            prior, nat1, nat2, observed
        )
        log_marg_corrected = _site_corrected_log_marginal(
            log_marg, q_mean, nat1, nat2, site_lik, observed
        )

        return NonGaussConditionedMarkovGP(
            prior=prior,
            y=y,
            site_nat1=nat1,
            site_nat2=nat2,
            q_mean=q_mean,
            q_var=q_var,
            log_marginal_approx=log_marg_corrected,
            n_iter=n_iter,
            converged=converged,
            mask=None if mask is None else observed,
        )


# --- Gauss-Newton --------------------------------------------------------


class GaussNewtonMarkovInference(eqx.Module):
    r"""Gauss-Newton (Markov) inference: Newton with a strict PSD floor.

    Identical to `LaplaceMarkovInference` for log-concave
    likelihoods (Bernoulli, Poisson). For non-log-concave likelihoods
    (StudentT) the larger ``precision_floor`` keeps the Newton step
    PSD-stable. Same fixed-point loop, damping, tolerance and adjoint as
    `LaplaceMarkovInference`.

    Attributes:
        max_iter: Newton iterations. Default ``20``.
        tol: Convergence tolerance on the site naturals, as in
            `LaplaceMarkovInference`. Default ``1e-6``.
        damping: Step-size in (0, 1]. Default ``1.0``.
        precision_floor: Strict positive floor. Default ``1e-3``.
        adjoint: How gradients pass through the loop. Default an
            ``optimistix.ImplicitAdjoint``, as in `LaplaceMarkovInference`.
    """

    max_iter: int = eqx.field(static=True, default=20)
    tol: float = eqx.field(static=True, default=1e-6)
    damping: float = eqx.field(static=True, default=1.0)
    precision_floor: float = eqx.field(static=True, default=1e-3)
    adjoint: optx.AbstractAdjoint = _IMPLICIT_ADJOINT

    def fit(
        self,
        prior: MarkovGPPrior,
        likelihood: Likelihood,
        y: Float[Array, " N"],
        *,
        mask: Bool[Array, " N"] | None = None,
    ) -> NonGaussConditionedMarkovGP:
        r"""Fit the sites by Gauss-Newton steps, as `LaplaceMarkovInference`.

        Args:
            prior: The `pyrox_ssm.MarkovGPPrior` over the observation times.
            likelihood: Any scalar-latent `pyrox.likelihoods.Likelihood`.
            y: Observations of shape ``(N,)``; masked entries may be ``NaN``.
            mask: Optional ``(N,)`` boolean mask, ``True`` where observed. A
                masked site contributes $g_n = h_n = 0$, so its naturals are
                $(\lambda_n, \Lambda_n) = (0, 0)$ and the filter skips it.

        Returns:
            The `NonGaussConditionedMarkovGP` with the fitted sites.

        Examples:
            >>> import jax.numpy as jnp
            >>> from pyrox_gp import BernoulliLikelihood
            >>> from pyrox_ssm import MaternSDE, MarkovGPPrior
            >>> from pyrox_ssm import GaussNewtonMarkovInference
            >>> prior = MarkovGPPrior(MaternSDE(1.0, 0.5, order=1), jnp.arange(4.0))
            >>> y = jnp.array([1.0, 0.0, jnp.nan, 1.0])
            >>> mask = jnp.array([True, True, False, True])
            >>> strategy = GaussNewtonMarkovInference()
            >>> cond = strategy.fit(prior, BernoulliLikelihood(), y, mask=mask)
        """
        inner = LaplaceMarkovInference(
            max_iter=self.max_iter,
            tol=self.tol,
            damping=self.damping,
            precision_floor=self.precision_floor,
            adjoint=self.adjoint,
        )
        return inner.fit(prior, likelihood, y, mask=mask)


# --- Posterior Linearization --------------------------------------------


def _resolve_integrator(
    integrator: AbstractIntegrator | None, deg: int | None
) -> AbstractIntegrator:
    """The strategy's rule, from the ``integrator`` and deprecated ``deg`` fields.

    Args:
        integrator: The rule passed, or ``None``.
        deg: The deprecated Gauss-Hermite order passed, or ``None``.

    Returns:
        ``integrator``; else ``GaussHermiteIntegrator(order=deg)``; else the
        default ``GaussHermiteIntegrator(order=20)``.

    Raises:
        ValueError: If both are passed.
    """
    if deg is None:
        return GaussHermiteIntegrator(order=20) if integrator is None else integrator
    if integrator is not None:
        raise ValueError(
            "Pass integrator or the deprecated deg, not both; got "
            f"integrator={integrator!r} and deg={deg}."
        )
    warnings.warn(
        "deg is deprecated; use "
        f"integrator=gaussx.GaussHermiteIntegrator(order={deg}).",
        DeprecationWarning,
        stacklevel=4,
    )
    return GaussHermiteIntegrator(order=deg)


def _fit_cavity_sites(
    strategy: PosteriorLinearizationMarkov | ExpectationPropagationMarkov,
    prior: MarkovGPPrior,
    likelihood: Likelihood,
    y: Float[Array, " N"],
    mask: Bool[Array, " N"] | None,
) -> NonGaussConditionedMarkovGP:
    """The fit shared by PL and EP, which differ only in their site update.

    The first iteration forms its cavities from the prior marginals; every
    later one, inside the fixed-point loop, from the smoothed marginals of the
    current sites.

    Args:
        strategy: The PL or EP strategy, for its ``_update`` and settings.
        prior: The `pyrox_ssm.MarkovGPPrior` over the observation times.
        likelihood: Any scalar-latent `pyrox.likelihoods.Likelihood`.
        y: Observations of shape ``(N,)``; masked entries may be ``NaN``.
        mask: Optional ``(N,)`` boolean mask, ``True`` where observed.

    Returns:
        The `NonGaussConditionedMarkovGP` with the fitted sites.
    """
    # The masked likelihood even for mask=None; see LaplaceMarkovInference.fit.
    observed = _observation_mask(mask, prior.times)
    site_lik = as_gaussx_likelihood(likelihood, y, mask=observed)

    N = prior.times.shape[0]
    prior_mean = prior.mean(prior.times)
    nat1 = jnp.zeros(N, dtype=prior_mean.dtype)
    nat2 = jnp.full((N,), strategy.precision_floor, dtype=prior_mean.dtype)
    sites = _mask_sites(nat1, nat2, observed, strategy.precision_floor)
    first = min(strategy.max_iter, 1)
    if first:
        # Seed the first cavities with the prior marginal variance
        # ``H P_inf H^T`` so the first cavity precision ``1/q_var - nat2`` is
        # on the correct scale for kernels with variance != 1.
        prior_var = _prior_marginal_variance(prior).astype(prior_mean.dtype)
        sites = strategy._update(
            sites, jnp.asarray(prior_mean), prior_var, site_lik, observed
        )

    sites, n_steps, converged = _solve_sites(
        strategy._step,
        sites,
        (prior, site_lik, observed),
        tol=strategy.tol,
        max_steps=strategy.max_iter - first,
        adjoint=strategy.adjoint,
    )
    nat1, nat2 = sites
    q_mean, q_var, log_marg = _markov_smoothed_posterior(prior, nat1, nat2, observed)
    log_marg_corrected = _site_corrected_log_marginal(
        log_marg, q_mean, nat1, nat2, site_lik, observed
    )

    return NonGaussConditionedMarkovGP(
        prior=prior,
        y=y,
        site_nat1=nat1,
        site_nat2=nat2,
        q_mean=q_mean,
        q_var=q_var,
        log_marginal_approx=log_marg_corrected,
        n_iter=n_steps + first,
        converged=converged,
        mask=None if mask is None else observed,
    )


class PosteriorLinearizationMarkov(eqx.Module):
    r"""Posterior linearization for `MarkovGPPrior` (Markov IPLF).

    Iterates filter + smoother + cavity-averaged statistical
    linearization. At each iteration, with current sites producing
    smoothed marginals ``(m_n, V_n)``, form the cavity
    ``q_{\\setminus n}(f) = N(m_{c,n}, V_{c,n})``, evaluate the *expected*
    per-point gradient and Hessian under the cavity (via the integrator),
    and update the sites with damping. The loop is `optimistix.fixed_point`,
    so a fit traces under `jax.jit` and differentiates through ``adjoint``.

    Attributes:
        max_iter: Iterations. Default ``30``.
        tol: Convergence tolerance on the site naturals, as in
            `LaplaceMarkovInference`: every $\lambda_n$ and $\Lambda_n$ changed
            by less than ``0.1 * tol * (1 + |new value|)``. Default ``1e-6``.
        damping: Step-size in (0, 1]. Default ``0.5``.
        precision_floor: Floor on the diagonal precision. Default ``1e-6``.
        deg: Deprecated. ``deg=k`` warns and means
            ``integrator=gaussx.GaussHermiteIntegrator(order=k)``; passing
            both raises ``ValueError``. Reads as ``None`` once resolved.
        integrator: Rule for the cavity expectations, any
            `gaussx.AbstractIntegrator`. Default
            ``gaussx.GaussHermiteIntegrator(order=20)``.
        adjoint: How gradients pass through the loop. Default an
            ``optimistix.ImplicitAdjoint``, as in `LaplaceMarkovInference`.
    """

    max_iter: int = eqx.field(static=True, default=30)
    tol: float = eqx.field(static=True, default=1e-6)
    damping: float = eqx.field(static=True, default=0.5)
    precision_floor: float = eqx.field(static=True, default=1e-6)
    deg: int | None = eqx.field(static=True, default=None)
    # Last, so positional arguments keep their meaning. None until
    # __post_init__, which tells "not passed" from a passed rule so that deg and
    # integrator together can raise.
    integrator: AbstractIntegrator = eqx.field(default=None)
    adjoint: optx.AbstractAdjoint = _IMPLICIT_ADJOINT

    def __post_init__(self) -> None:
        self.integrator = _resolve_integrator(self.integrator, self.deg)
        self.deg = None

    def _update(
        self,
        sites: _Sites,
        q_mean: Float[Array, " N"],
        q_var: Float[Array, " N"],
        site_lik: CallableLikelihood,
        observed: Bool[Array, " N"],
    ) -> _Sites:
        """The damped PL sites from the marginals ``(q_mean, q_var)``."""
        nat1, nat2 = sites
        cav_mean, cav_var = cavity_distribution(
            q_mean, q_var, nat1, nat2, precision_floor=self.precision_floor
        )
        assert isinstance(cav_var, jax.Array)  # diagonal path in, diagonal out

        # ḡₙ = 𝔼_{cav}[∂f log p(yₙ | f)], h̄ₙ = 𝔼_{cav}[∂²f log p(yₙ | f)]
        g_avg, h_avg = _cavity_site_derivatives(
            site_lik, cav_mean, cav_var, self.integrator
        )

        new_nat1, new_prec = newton_update(
            cav_mean, g_avg, h_avg, precision_floor=self.precision_floor
        )
        nat1, nat2 = damped_natural_update(
            nat1, nat2, new_nat1, new_prec, lr=self.damping
        )
        assert isinstance(nat2, jax.Array)  # pyrox sites are diagonal arrays
        return _mask_sites(nat1, nat2, observed, self.precision_floor)

    def _step(self, sites: _Sites, args: PyTree) -> _Sites:
        r"""One damped PL sweep, $T_\alpha(s)$, from the smoothed marginals."""
        prior, site_lik, observed = args
        q_mean, q_var, _ = _markov_smoothed_posterior(prior, *sites, observed)
        return self._update(sites, q_mean, q_var, site_lik, observed)

    def fit(
        self,
        prior: MarkovGPPrior,
        likelihood: Likelihood,
        y: Float[Array, " N"],
        *,
        mask: Bool[Array, " N"] | None = None,
    ) -> NonGaussConditionedMarkovGP:
        r"""Fit the sites by posterior linearization.

        Args:
            prior: The `pyrox_ssm.MarkovGPPrior` over the observation times.
            likelihood: Any scalar-latent `pyrox.likelihoods.Likelihood`.
            y: Observations of shape ``(N,)``; masked entries may be ``NaN``.
            mask: Optional ``(N,)`` boolean mask, ``True`` where observed. A
                masked site contributes $g_n = h_n = 0$, so its naturals are
                $(\lambda_n, \Lambda_n) = (0, 0)$ and the filter skips it.

        Returns:
            The `NonGaussConditionedMarkovGP` with the fitted sites.

        Examples:
            >>> import jax.numpy as jnp
            >>> from pyrox_gp import BernoulliLikelihood
            >>> from pyrox_ssm import MaternSDE, MarkovGPPrior
            >>> from pyrox_ssm import PosteriorLinearizationMarkov
            >>> prior = MarkovGPPrior(MaternSDE(1.0, 0.5, order=1), jnp.arange(4.0))
            >>> y = jnp.array([1.0, 0.0, jnp.nan, 1.0])
            >>> mask = jnp.array([True, True, False, True])
            >>> strategy = PosteriorLinearizationMarkov()
            >>> cond = strategy.fit(prior, BernoulliLikelihood(), y, mask=mask)
        """
        return _fit_cavity_sites(self, prior, likelihood, y, mask)


# --- Expectation Propagation --------------------------------------------


class ExpectationPropagationMarkov(eqx.Module):
    r"""Parallel Expectation Propagation for `MarkovGPPrior`.

    Each iteration: run filter + smoother, form cavities at every site,
    match tilted-distribution moments via Gauss-Hermite (with log-space
    stabilisation), and update sites with damping. The loop is
    `optimistix.fixed_point`, so a fit traces under `jax.jit` and
    differentiates through ``adjoint``.

    Attributes:
        max_iter: EP iterations. Default ``40``.
        tol: Convergence tolerance on the site naturals, as in
            `LaplaceMarkovInference`: every $\lambda_n$ and $\Lambda_n$ changed
            by less than ``0.1 * tol * (1 + |new value|)``. Default ``1e-5``.
        damping: Damping in (0, 1]. Default ``0.5``.
        precision_floor: Floor on the diagonal precision. Default ``1e-6``.
        deg: Deprecated. ``deg=k`` warns and means
            ``integrator=gaussx.GaussHermiteIntegrator(order=k)``; passing
            both raises ``ValueError``. Reads as ``None`` once resolved.
        integrator: Rule for the tilted moments. Default
            ``gaussx.GaussHermiteIntegrator(order=20)``. EP reads only its
            ``order``, for `gaussx.ep_tilted_moments`, so any other rule
            raises ``TypeError``.
        adjoint: How gradients pass through the loop. Default an
            ``optimistix.ImplicitAdjoint``, as in `LaplaceMarkovInference`.
    """

    max_iter: int = eqx.field(static=True, default=40)
    tol: float = eqx.field(static=True, default=1e-5)
    damping: float = eqx.field(static=True, default=0.5)
    precision_floor: float = eqx.field(static=True, default=1e-6)
    deg: int | None = eqx.field(static=True, default=None)
    # Last and None until __post_init__; see PosteriorLinearizationMarkov.
    integrator: AbstractIntegrator = eqx.field(default=None)
    adjoint: optx.AbstractAdjoint = _IMPLICIT_ADJOINT

    def __post_init__(self) -> None:
        self.integrator = _resolve_integrator(self.integrator, self.deg)
        self.deg = None

    def __check_init__(self) -> None:
        _require_gauss_hermite(self.integrator)

    def _update(
        self,
        sites: _Sites,
        q_mean: Float[Array, " N"],
        q_var: Float[Array, " N"],
        site_lik: CallableLikelihood,
        observed: Bool[Array, " N"],
    ) -> _Sites:
        """The damped EP sites from the marginals ``(q_mean, q_var)``."""
        nat1, nat2 = sites
        cav_mean, cav_var = cavity_distribution(
            q_mean, q_var, nat1, nat2, precision_floor=self.precision_floor
        )
        assert isinstance(cav_var, jax.Array)  # diagonal path in, diagonal out

        # ``gaussx.ep_tilted_moments`` requires ``log_lik_fn(f)`` with the
        # per-site target baked in; close over the site data ``d_n`` via vmap
        # to recover the (N,) shape contract.
        order = _require_gauss_hermite(self.integrator).order

        def _per_site_tilted(
            m_n: Float[Array, ""],
            v_n: Float[Array, ""],
            d_n: Float[Array, ""],
        ) -> tuple[Float[Array, ""], Float[Array, ""]]:
            return ep_tilted_moments(
                lambda f: site_lik.log_density(f, d_n), m_n, v_n, order=order
            )

        tilted_mean, tilted_var = jax.vmap(_per_site_tilted)(
            cav_mean, cav_var, site_lik.data
        )

        new_prec = jnp.reciprocal(tilted_var) - jnp.reciprocal(cav_var)
        new_prec = jnp.maximum(new_prec, self.precision_floor)
        new_nat1 = tilted_mean / tilted_var - cav_mean / cav_var
        nat1, nat2 = damped_natural_update(
            nat1, nat2, new_nat1, new_prec, lr=self.damping
        )
        assert isinstance(nat2, jax.Array)  # pyrox sites are diagonal arrays
        # A masked site's log density is zero, so its tilted moments are
        # its cavity's and the site it would get is (0, 0) up to rounding;
        # _mask_sites makes that exact.
        return _mask_sites(nat1, nat2, observed, self.precision_floor)

    def _step(self, sites: _Sites, args: PyTree) -> _Sites:
        r"""One damped EP sweep, $T_\alpha(s)$, from the smoothed marginals."""
        prior, site_lik, observed = args
        q_mean, q_var, _ = _markov_smoothed_posterior(prior, *sites, observed)
        return self._update(sites, q_mean, q_var, site_lik, observed)

    def fit(
        self,
        prior: MarkovGPPrior,
        likelihood: Likelihood,
        y: Float[Array, " N"],
        *,
        mask: Bool[Array, " N"] | None = None,
    ) -> NonGaussConditionedMarkovGP:
        r"""Fit the sites by parallel expectation propagation.

        Args:
            prior: The `pyrox_ssm.MarkovGPPrior` over the observation times.
            likelihood: Any scalar-latent `pyrox.likelihoods.Likelihood`.
            y: Observations of shape ``(N,)``; masked entries may be ``NaN``.
            mask: Optional ``(N,)`` boolean mask, ``True`` where observed. A
                masked site contributes $g_n = h_n = 0$, so its naturals are
                $(\lambda_n, \Lambda_n) = (0, 0)$ and the filter skips it.

        Returns:
            The `NonGaussConditionedMarkovGP` with the fitted sites.

        Examples:
            >>> import jax.numpy as jnp
            >>> from pyrox_gp import BernoulliLikelihood
            >>> from pyrox_ssm import MaternSDE, MarkovGPPrior
            >>> from pyrox_ssm import ExpectationPropagationMarkov
            >>> prior = MarkovGPPrior(MaternSDE(1.0, 0.5, order=1), jnp.arange(4.0))
            >>> y = jnp.array([1.0, 0.0, jnp.nan, 1.0])
            >>> mask = jnp.array([True, True, False, True])
            >>> strategy = ExpectationPropagationMarkov()
            >>> cond = strategy.fit(prior, BernoulliLikelihood(), y, mask=mask)
        """
        return _fit_cavity_sites(self, prior, likelihood, y, mask)


# --- NumPyro factor ------------------------------------------------------


def markov_laplace_factor(
    name: str,
    prior: MarkovGPPrior,
    likelihood: Likelihood,
    y: Float[Array, " N"],
    *,
    strategy: LaplaceMarkovInference
    | GaussNewtonMarkovInference
    | PosteriorLinearizationMarkov
    | ExpectationPropagationMarkov,
    mask: Bool[Array, " N"] | None = None,
    reject_unconverged: bool = False,
) -> None:
    r"""Register a non-Gaussian Markov GP's approximate evidence with NumPyro.

    Fits the sites with ``strategy`` and adds its
    ``log_marginal_approx`` as ``numpyro.factor(name, ...)``, the latent path
    marginalised out. With `LaplaceMarkovInference` (or
    `GaussNewtonMarkovInference`) that is the Laplace evidence

    $$
    \log p(y \mid \theta) \approx \log p(y \mid \hat f)
      + \log p(\hat f \mid \theta)
      - \tfrac{1}{2} \log \det (I + W K_\theta),
    $$

    at the mode $\hat f$ with $W = \operatorname{diag}(\Lambda)$, computed by
    the Kalman smoother in $O(N d^3)$. Inside NUTS over $\theta$ this is the
    INLA recipe on a Markov prior. The fit traces and its gradient is implicit
    at the fixed point (see `LaplaceMarkovInference`), so the factor works under
    NUTS and SVI. With `PosteriorLinearizationMarkov` or
    `ExpectationPropagationMarkov` the factor is that strategy's
    ``log_marginal_approx``.

    Args:
        name: The factor site name.
        prior: The `MarkovGPPrior` over the observation times.
        likelihood: Any scalar-latent `pyrox.likelihoods.Likelihood`.
        y: Observations of shape ``(N,)``; masked entries may be ``NaN``.
        strategy: The site-based strategy that fits the sites.
        mask: Optional ``(N,)`` boolean mask, ``True`` where observed.
        reject_unconverged: If ``True``, the factor is $-\infty$ when the fit
            did not meet the strategy's ``tol`` within ``max_iter`` iterations,
            so NUTS rejects that $\theta$ rather than move on an evidence (and
            an implicit gradient) taken away from the fixed point. Default
            ``False``: the factor is the evidence at the last iterate.

    Examples:
        >>> import jax.numpy as jnp
        >>> import numpyro
        >>> import numpyro.distributions as dist
        >>> from pyrox_gp import BernoulliLikelihood
        >>> import pyrox_ssm as pss
        >>> def model(times, y):
        ...     var = numpyro.sample("variance", dist.LogNormal(0.0, 1.0))
        ...     ell = numpyro.sample("lengthscale", dist.LogNormal(0.0, 1.0))
        ...     prior = pss.MarkovGPPrior(pss.MaternSDE(var, ell, order=1), times)
        ...     pss.markov_laplace_factor(
        ...         "y", prior, BernoulliLikelihood(), y,
        ...         strategy=pss.LaplaceMarkovInference(),
        ...         reject_unconverged=True,
        ...     )
    """
    cond = prior.condition_nongauss(likelihood, y, strategy=strategy, mask=mask)
    value = cond.log_marginal_approx
    if reject_unconverged:
        value = jnp.where(cond.converged, value, -jnp.inf)
    numpyro.factor(name, value)
