"""Site-based non-Gaussian inference for :class:`MarkovGPPrior`.

Couples the site-based view from :mod:`pyrox.gp._inference_nongauss`
(each likelihood factor contributes a diagonal Gaussian site with
naturals :math:`(\\lambda, \\Lambda)`) to the linear-time Kalman /
RTS-smoother backbone in :mod:`pyrox.gp._markov`.

Each iteration runs the filter + smoother once with per-step
pseudo-observation variance :math:`R_n = 1/\\Lambda_n` and pseudo-target
:math:`\\tilde y_n = \\lambda_n / \\Lambda_n` to obtain the marginal
posterior :math:`q(f_n) = \\mathcal{N}(m_n, V_n)` on the training grid;
strategies differ only in how new site naturals are obtained from the
local likelihood given ``(m_n, V_n)``. Cost is :math:`O(I N d^3)` for
``I`` iterations on ``N`` time points and SDE state dimension ``d`` —
linear in ``N``, where the dense path is :math:`O(I N^3)`.

Predictions reuse the same Kalman trick: the merged grid
``sort(times \\cup t_star)`` with the test points masked produces
self-consistent smoothed marginals at arbitrary test times.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from pyrox.gp._inference_nongauss import (
    _check_scalar_latent,
    _ep_tilted_moments,
    _log_prob_per_point_factory,
    _per_point_grad_hess,
)
from pyrox.gp._markov import _build_dt_full, _kalman_filter, _rts_smoother
from pyrox.gp._protocols import Likelihood


if TYPE_CHECKING:
    from pyrox.gp._markov import MarkovGPPrior


# --- core smoother helpers ----------------------------------------------


def _markov_smoothed_posterior(
    prior: MarkovGPPrior,
    nat1: Float[Array, " N"],
    nat2: Float[Array, " N"],
) -> tuple[Float[Array, " N"], Float[Array, " N"], Float[Array, ""]]:
    r"""Smoothed marginals on the training grid given diagonal site naturals.

    Treats the sites as a synthetic Gaussian likelihood with per-step
    variance :math:`1/\Lambda_n` and target :math:`\lambda_n / \Lambda_n`
    (centred on the prior mean), then runs filter + RTS smoother. Returns
    ``(f_mean, f_var, log_marg)`` over the training times where
    ``log_marg`` is the Kalman log-likelihood of the *pseudo-observations*
    (used as a Laplace-style approximation to the true marginal).
    """
    F, _L, H, _Qc, P_inf = prior.sde_kernel.sde_params()
    times = prior.times
    dt_full = _build_dt_full(times)
    A_seq, Q_seq = prior.sde_kernel.discretise(dt_full)
    R_seq = jnp.reciprocal(nat2)
    pseudo_targets = nat1 / nat2
    residual = pseudo_targets - prior.mean(times)
    mask = jnp.ones_like(times)
    m_pred, P_pred, m_filt, P_filt, log_marg = _kalman_filter(
        F, H, P_inf, A_seq, Q_seq, residual, mask, R_seq
    )
    m_smooth, P_smooth = _rts_smoother(m_pred, P_pred, m_filt, P_filt, A_seq)
    f_mean = (m_smooth @ H.T)[:, 0] + prior.mean(times)
    f_var = jax.vmap(lambda P: (H @ P @ H.T)[0, 0])(P_smooth)
    f_var = jnp.maximum(f_var, 1e-12)
    return f_mean, f_var, log_marg


# --- result type ---------------------------------------------------------


class NonGaussConditionedMarkovGP(eqx.Module):
    """Markov GP conditioned on a non-Gaussian likelihood via a site-based strategy.

    Equivalent role to :class:`pyrox.gp.ConditionedMarkovGP` but the
    posterior over training latents is a generic Gaussian
    approximation produced by one of the strategies in this module
    rather than the closed-form Gaussian-likelihood smoother. Predictions
    at arbitrary test times use the standard *site-as-pseudo-observation*
    trick on the merged grid ``sort(times | t_star)`` with the test
    points masked.

    Attributes:
        prior: The originating :class:`MarkovGPPrior`.
        y: Training targets (kept for round-trip / diagnostics).
        site_nat1: Diagonal site naturals :math:`\\lambda \\in \\mathbb{R}^N`.
        site_nat2: Diagonal site precisions :math:`\\Lambda \\in \\mathbb{R}^N`
            (positive).
        q_mean: Posterior mean over training latents.
        q_var: Marginal posterior variance per training point.
        log_marginal_approx: Approximate log marginal likelihood (the
            scalar each strategy reports — interpretation is
            strategy-specific).
        n_iter: Iterations used by the strategy.
        converged: Whether convergence tolerance was met.
    """

    prior: MarkovGPPrior
    y: Float[Array, " N"]
    site_nat1: Float[Array, " N"]
    site_nat2: Float[Array, " N"]
    q_mean: Float[Array, " N"]
    q_var: Float[Array, " N"]
    log_marginal_approx: Float[Array, ""]
    n_iter: int = eqx.field(static=True)
    converged: bool = eqx.field(static=True)

    def predict(
        self,
        t_star: Float[Array, " M"],
    ) -> tuple[Float[Array, " M"], Float[Array, " M"]]:
        r"""Predictive marginals ``(mean, var)`` at arbitrary test times.

        Re-runs filter + smoother over the merged grid
        ``sort(times \cup t_star)`` with per-step pseudo-observation
        variances ``1/Λ_n`` on the training points and the test points
        masked out of the update step. Cost is :math:`O((N + M)\,d^3)`.
        """
        F, _L, H, _Qc, P_inf = self.prior.sde_kernel.sde_params()
        times = self.prior.times
        t_star = jnp.asarray(t_star)

        N = times.shape[0]
        M = t_star.shape[0]
        merged = jnp.concatenate([times, t_star], axis=0)
        order = jnp.argsort(merged, stable=True)
        merged_sorted = merged[order]

        is_obs = jnp.concatenate(
            [jnp.ones(N, dtype=times.dtype), jnp.zeros(M, dtype=times.dtype)]
        )[order]
        pseudo_targets = self.site_nat1 / self.site_nat2
        residual_full = jnp.concatenate(
            [
                pseudo_targets - self.prior.mean(times),
                jnp.zeros(M, dtype=self.y.dtype),
            ]
        )[order]
        # ``R_seq`` for masked steps is irrelevant (the update is
        # skipped) but must be finite to avoid NaN propagation through
        # the filter; use 1.0 as a safe placeholder.
        R_train = jnp.reciprocal(self.site_nat2)
        R_full = jnp.concatenate([R_train, jnp.ones(M, dtype=self.y.dtype)])[order]

        dt_full = _build_dt_full(merged_sorted)
        A_seq, Q_seq = self.prior.sde_kernel.discretise(dt_full)
        m_pred, P_pred, m_filt, P_filt, _ = _kalman_filter(
            F, H, P_inf, A_seq, Q_seq, residual_full, is_obs, R_full
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
    r"""Laplace approximation for :class:`MarkovGPPrior`.

    Fixed-point Newton on the smoothed posterior: at each iteration run
    filter + smoother with the current sites to obtain marginals
    ``(m_n, V_n)``, then update site naturals from per-point gradient /
    Hessian of ``log p(y | f)`` evaluated at ``f = m``. Site precision
    :math:`\Lambda = -h` is clipped to ``precision_floor``.

    Args:
        max_iter: Newton iterations. Default ``20``.
        tol: ``inf``-norm convergence on the posterior mean. Default ``1e-6``.
        damping: Step-size in (0, 1] applied to each Newton update.
            Default ``1.0`` (full Newton step). Drop below 1 for
            non-log-concave likelihoods.
        precision_floor: Lower bound on the diagonal precision. Default
            ``1e-6``.
    """

    max_iter: int = eqx.field(static=True, default=20)
    tol: float = eqx.field(static=True, default=1e-6)
    damping: float = eqx.field(static=True, default=1.0)
    precision_floor: float = eqx.field(static=True, default=1e-6)

    def fit(
        self,
        prior: MarkovGPPrior,
        likelihood: Likelihood,
        y: Float[Array, " N"],
    ) -> NonGaussConditionedMarkovGP:
        _check_scalar_latent(likelihood)
        log_prob_per_point = _log_prob_per_point_factory(likelihood)

        N = prior.times.shape[0]
        prior_mean = prior.mean(prior.times)
        nat1 = jnp.asarray(prior_mean) * self.precision_floor
        nat2 = jnp.full((N,), self.precision_floor, dtype=prior_mean.dtype)
        f = jnp.asarray(prior_mean)

        converged = False
        n_iter = 0
        for it in range(self.max_iter):
            g, h = _per_point_grad_hess(log_prob_per_point, f, y)
            Lam = jnp.maximum(-h, self.precision_floor)
            new_nat1 = g + Lam * f
            nat1 = (1.0 - self.damping) * nat1 + self.damping * new_nat1
            nat2 = (1.0 - self.damping) * nat2 + self.damping * Lam
            nat2 = jnp.maximum(nat2, self.precision_floor)
            f_new, _, _ = _markov_smoothed_posterior(prior, nat1, nat2)
            delta = jnp.max(jnp.abs(f_new - f))
            f = f_new
            n_iter = it + 1
            if delta < self.tol:
                converged = True
                break

        # Final site naturals at convergence.
        g, h = _per_point_grad_hess(log_prob_per_point, f, y)
        Lam = jnp.maximum(-h, self.precision_floor)
        nat1 = g + Lam * f
        nat2 = jnp.maximum(Lam, self.precision_floor)
        q_mean, q_var, log_marg = _markov_smoothed_posterior(prior, nat1, nat2)
        # Add the data-fit term ``log p(y | hat f)`` and subtract the
        # pseudo-data fit so the reported scalar is the standard Laplace
        # log-marginal approximation rather than the Kalman pseudo-data
        # likelihood.
        ll_data = log_prob_per_point(q_mean, y).sum()
        pseudo_targets = nat1 / nat2
        ll_pseudo = -0.5 * jnp.sum(jnp.log(2.0 * jnp.pi / nat2)) - 0.5 * jnp.sum(
            nat2 * (q_mean - pseudo_targets) ** 2
        )
        log_marg_corrected = log_marg + ll_data - ll_pseudo

        return NonGaussConditionedMarkovGP(  # ty: ignore[invalid-return-type]
            prior=prior,
            y=y,
            site_nat1=nat1,
            site_nat2=nat2,
            q_mean=q_mean,
            q_var=q_var,
            log_marginal_approx=log_marg_corrected,
            n_iter=n_iter,
            converged=converged,
        )


# --- Gauss-Newton --------------------------------------------------------


class GaussNewtonMarkovInference(eqx.Module):
    r"""Gauss-Newton (Markov) inference: Newton with a strict PSD floor.

    Identical to :class:`LaplaceMarkovInference` for log-concave
    likelihoods (Bernoulli, Poisson). For non-log-concave likelihoods
    (StudentT) the larger ``precision_floor`` keeps the Newton step
    PSD-stable. Same fixed-point loop and damping behaviour as
    :class:`LaplaceMarkovInference`.

    Args:
        max_iter: Newton iterations. Default ``20``.
        tol: ``inf``-norm tolerance on the posterior mean. Default ``1e-6``.
        damping: Step-size in (0, 1]. Default ``1.0``.
        precision_floor: Strict positive floor. Default ``1e-3``.
    """

    max_iter: int = eqx.field(static=True, default=20)
    tol: float = eqx.field(static=True, default=1e-6)
    damping: float = eqx.field(static=True, default=1.0)
    precision_floor: float = eqx.field(static=True, default=1e-3)

    def fit(
        self,
        prior: MarkovGPPrior,
        likelihood: Likelihood,
        y: Float[Array, " N"],
    ) -> NonGaussConditionedMarkovGP:
        inner = LaplaceMarkovInference(
            max_iter=self.max_iter,
            tol=self.tol,
            damping=self.damping,
            precision_floor=self.precision_floor,
        )
        return inner.fit(prior, likelihood, y)  # ty: ignore[unresolved-attribute]


# --- Posterior Linearization --------------------------------------------


class PosteriorLinearizationMarkov(eqx.Module):
    r"""Posterior linearization for :class:`MarkovGPPrior` (Markov IPLF).

    Iterates filter + smoother + cavity-averaged statistical
    linearization. At each iteration, with current sites producing
    smoothed marginals ``(m_n, V_n)``, form the cavity
    ``q_{\\setminus n}(f) = N(m_{c,n}, V_{c,n})``, evaluate the *expected*
    per-point gradient and Hessian under the cavity (via Gauss-Hermite),
    and update the sites.

    Args:
        max_iter: Iterations. Default ``20``.
        tol: ``inf``-norm tolerance on the posterior mean. Default ``1e-6``.
        damping: Step-size in (0, 1]. Default ``0.5``.
        precision_floor: Floor on the diagonal precision. Default ``1e-6``.
        deg: Gauss-Hermite degree for the cavity expectations. Default ``20``.
    """

    max_iter: int = eqx.field(static=True, default=20)
    tol: float = eqx.field(static=True, default=1e-6)
    damping: float = eqx.field(static=True, default=0.5)
    precision_floor: float = eqx.field(static=True, default=1e-6)
    deg: int = eqx.field(static=True, default=20)

    def fit(
        self,
        prior: MarkovGPPrior,
        likelihood: Likelihood,
        y: Float[Array, " N"],
    ) -> NonGaussConditionedMarkovGP:
        from gaussx import gauss_hermite_points

        _check_scalar_latent(likelihood)
        log_prob_per_point = _log_prob_per_point_factory(likelihood)

        def lp(f_n: Float[Array, ""], y_n: Float[Array, ""]) -> Float[Array, ""]:
            return log_prob_per_point(f_n[None], y_n[None])[0]

        grad_fn = jax.vmap(jax.grad(lp))
        hess_fn = jax.vmap(jax.grad(jax.grad(lp)))

        nodes, weights = gauss_hermite_points(self.deg, dim=1)
        x_nodes = nodes[:, 0]
        w_nodes = weights / jnp.sqrt(2.0 * jnp.pi)

        N = prior.times.shape[0]
        prior_mean = prior.mean(prior.times)
        nat1 = jnp.zeros(N, dtype=prior_mean.dtype)
        nat2 = jnp.full((N,), self.precision_floor, dtype=prior_mean.dtype)
        q_mean = jnp.asarray(prior_mean)
        q_var = jnp.full((N,), 1.0, dtype=prior_mean.dtype)

        converged = False
        n_iter = 0
        for it in range(self.max_iter):
            cav_prec = jnp.maximum(jnp.reciprocal(q_var) - nat2, self.precision_floor)
            cav_var = jnp.reciprocal(cav_prec)
            cav_mean = cav_var * (q_mean / q_var - nat1)

            std = jnp.sqrt(cav_var)
            f_grid = cav_mean[None, :] + std[None, :] * x_nodes[:, None]
            g_grid = jax.vmap(lambda f_row: grad_fn(f_row, y))(f_grid)
            h_grid = jax.vmap(lambda f_row: hess_fn(f_row, y))(f_grid)
            g_avg = jnp.sum(w_nodes[:, None] * g_grid, axis=0)
            h_avg = jnp.sum(w_nodes[:, None] * h_grid, axis=0)

            new_prec = jnp.maximum(-h_avg, self.precision_floor)
            new_nat1 = g_avg + new_prec * cav_mean
            nat1 = (1.0 - self.damping) * nat1 + self.damping * new_nat1
            nat2 = (1.0 - self.damping) * nat2 + self.damping * new_prec
            nat2 = jnp.maximum(nat2, self.precision_floor)

            q_mean_new, q_var, _ = _markov_smoothed_posterior(prior, nat1, nat2)
            delta = jnp.max(jnp.abs(q_mean_new - q_mean))
            q_mean = q_mean_new
            n_iter = it + 1
            if delta < self.tol:
                converged = True
                break

        _, _, log_marg = _markov_smoothed_posterior(prior, nat1, nat2)

        return NonGaussConditionedMarkovGP(  # ty: ignore[invalid-return-type]
            prior=prior,
            y=y,
            site_nat1=nat1,
            site_nat2=nat2,
            q_mean=q_mean,
            q_var=q_var,
            log_marginal_approx=log_marg,
            n_iter=n_iter,
            converged=converged,
        )


# --- Expectation Propagation --------------------------------------------


class ExpectationPropagationMarkov(eqx.Module):
    r"""Parallel Expectation Propagation for :class:`MarkovGPPrior`.

    Each iteration: run filter + smoother, form cavities at every site,
    match tilted-distribution moments via Gauss-Hermite (with log-space
    stabilisation), and update sites with damping.

    Args:
        max_iter: EP iterations. Default ``40``.
        tol: ``inf``-norm tolerance on the posterior mean. Default ``1e-5``.
        damping: Damping in (0, 1]. Default ``0.5``.
        precision_floor: Floor on the diagonal precision. Default ``1e-6``.
        deg: Gauss-Hermite degree for the tilted-moment integrals.
            Default ``20``.
    """

    max_iter: int = eqx.field(static=True, default=40)
    tol: float = eqx.field(static=True, default=1e-5)
    damping: float = eqx.field(static=True, default=0.5)
    precision_floor: float = eqx.field(static=True, default=1e-6)
    deg: int = eqx.field(static=True, default=20)

    def fit(
        self,
        prior: MarkovGPPrior,
        likelihood: Likelihood,
        y: Float[Array, " N"],
    ) -> NonGaussConditionedMarkovGP:
        _check_scalar_latent(likelihood)
        log_prob_per_point = _log_prob_per_point_factory(likelihood)

        def lp(f_n: Float[Array, ""], y_n: Float[Array, ""]) -> Float[Array, ""]:
            return log_prob_per_point(f_n[None], y_n[None])[0]

        N = prior.times.shape[0]
        prior_mean = prior.mean(prior.times)
        nat1 = jnp.zeros(N, dtype=prior_mean.dtype)
        nat2 = jnp.full((N,), self.precision_floor, dtype=prior_mean.dtype)
        q_mean = jnp.asarray(prior_mean)
        q_var = jnp.full((N,), 1.0, dtype=prior_mean.dtype)

        converged = False
        n_iter = 0
        for it in range(self.max_iter):
            cav_prec = jnp.maximum(jnp.reciprocal(q_var) - nat2, self.precision_floor)
            cav_var = jnp.reciprocal(cav_prec)
            cav_mean = cav_var * (q_mean / q_var - nat1)

            tilted_mean, tilted_var = _ep_tilted_moments(
                lp, y, cav_mean, cav_var, deg=self.deg
            )

            new_prec = jnp.reciprocal(tilted_var) - cav_prec
            new_prec = jnp.maximum(new_prec, self.precision_floor)
            new_nat1 = tilted_mean / tilted_var - cav_mean / cav_var
            nat1 = (1.0 - self.damping) * nat1 + self.damping * new_nat1
            nat2 = (1.0 - self.damping) * nat2 + self.damping * new_prec
            nat2 = jnp.maximum(nat2, self.precision_floor)

            q_mean_new, q_var, _ = _markov_smoothed_posterior(prior, nat1, nat2)
            delta = jnp.max(jnp.abs(q_mean_new - q_mean))
            q_mean = q_mean_new
            n_iter = it + 1
            if delta < self.tol:
                converged = True
                break

        _, _, log_marg = _markov_smoothed_posterior(prior, nat1, nat2)

        return NonGaussConditionedMarkovGP(  # ty: ignore[invalid-return-type]
            prior=prior,
            y=y,
            site_nat1=nat1,
            site_nat2=nat2,
            q_mean=q_mean,
            q_var=q_var,
            log_marginal_approx=log_marg,
            n_iter=n_iter,
            converged=converged,
        )
