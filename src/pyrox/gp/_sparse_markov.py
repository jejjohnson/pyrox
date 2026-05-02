"""Sparse variational Markov-GP — inducing time points on top of an SDE prior.

:class:`SparseMarkovGPPrior` is the sparse analogue of
:class:`pyrox.gp.MarkovGPPrior`: it wraps an :class:`SDEKernel` and a
sorted inducing time grid ``Z`` of length ``M``, exposing the SVGP
building blocks ``(K_{ZZ}, K_{xZ}, \\mathrm{diag}\\,K_{xx})`` derived
from the SDE autocovariance

.. math::

    k(\\tau) = H\\,\\exp(F\\,|\\tau|)\\,P_\\infty\\,H^\\top.

The class duck-types :class:`pyrox.gp.SparseGPPrior` so it composes
with the existing variational guides (:class:`pyrox.gp.FullRankGuide`,
:class:`pyrox.gp.MeanFieldGuide`, :class:`pyrox.gp.WhitenedGuide`) and
the :func:`pyrox.gp.svgp_elbo` / :func:`pyrox.gp.svgp_factor`
infrastructure.

Cost: this surface is :math:`O(M^3 + N M)` per ELBO evaluation — the
inducing prior is built dense in :math:`M` and the guide owns a dense
:math:`M\\times M` covariance. A truly linear-in-:math:`N` Kalman-aware
ELBO that exploits the Markov factorisation of ``q(u)`` is a follow-up
(BayesNewton / MarkovFlow style); this surface is the SVGP-style
foundation it will build on.

Predictions at arbitrary test times use the same dense SVGP formula via
:meth:`SparseConditionedMarkovGP.predict`. For mean-only predictions
on a long evaluation grid, the merged-grid Kalman trick from
:mod:`pyrox.gp._markov` is asymptotically cheaper, but is not yet
wired in here.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpyro
from gaussx import (
    AbstractIntegrator,
    AbstractSolverStrategy,
    DenseSolver,
    MultivariateNormal,
    gaussian_log_prob,
    sde_autocovariance,
)
from jaxtyping import Array, Float

from pyrox.gp._inference import _ell_numerical
from pyrox.gp._likelihoods import GaussianLikelihood
from pyrox.gp._protocols import Guide, Likelihood, SDEKernel


def _psd_operator(K: Float[Array, "M M"]) -> lx.AbstractLinearOperator:
    """Wrap a Gram matrix as a PSD ``lineax`` operator."""
    return lx.MatrixLinearOperator(K, lx.positive_semidefinite_tag)  # ty: ignore[invalid-return-type]


def _sde_autocov_pairs(
    sde_kernel: SDEKernel,
    t1: Float[Array, " A"],
    t2: Float[Array, " B"],
) -> Float[Array, "A B"]:
    r"""Pairwise autocov ``k(\tau) = H \exp(F|\tau|) P_\infty H^T``.

    ``\tau = |t1 - t2|``. Thin wrapper around
    :func:`gaussx.sde_autocovariance` that builds
    the pairwise-difference grid; gaussx 0.0.11 owns the SDE-kernel
    primitives so the math itself lives there. Same forward-mode
    contract as the pre-0.0.11 pyrox helper.
    """
    diffs = jnp.abs(t1[:, None] - t2[None, :])
    return sde_autocovariance(sde_kernel, diffs)


class SparseMarkovGPPrior(eqx.Module):
    r"""Sparse variational GP prior over an SDE kernel and inducing time grid.

    Equivalent role to :class:`pyrox.gp.SparseGPPrior` but the
    covariance is derived from the state-space autocovariance
    :math:`k(\tau) = H \exp(F\tau) P_\infty H^\top` of an
    :class:`SDEKernel`. Predictions and the SVGP ELBO go through the
    same :meth:`predictive_blocks` contract as the dense
    :class:`SparseGPPrior`, so the existing variational guides
    (:class:`pyrox.gp.FullRankGuide`, :class:`pyrox.gp.MeanFieldGuide`,
    :class:`pyrox.gp.WhitenedGuide`) and :func:`pyrox.gp.svgp_elbo`
    work as-is.

    Attributes:
        sde_kernel: Any :class:`SDEKernel` (Matern, Periodic, Cosine,
            Sum/Product compositions, ...).
        Z: Sorted, strictly increasing inducing times of shape ``(M,)``.
        mean_fn: Optional callable ``times -> (N,)`` global mean.
            Convenience accessor; not folded into the inducing prior.
        solver: Any ``gaussx.AbstractSolverStrategy``. Defaults to
            ``gaussx.DenseSolver()``.
        jitter: Diagonal regularisation added to ``K_{ZZ}`` for numerical
            stability.
    """

    sde_kernel: SDEKernel
    Z: Float[Array, " M"]
    mean_fn: Callable[[Float[Array, " N"]], Float[Array, " N"]] | None = None
    solver: AbstractSolverStrategy | None = None
    jitter: float = eqx.field(static=True, default=1e-6)

    def __init__(
        self,
        sde_kernel: SDEKernel,
        Z: Float[Array, " M"],
        mean_fn: Callable[[Float[Array, " N"]], Float[Array, " N"]] | None = None,
        solver: AbstractSolverStrategy | None = None,
        jitter: float = 1e-6,
    ) -> None:
        Z_arr = jnp.asarray(Z, dtype=jnp.result_type(Z, 0.0))
        if Z_arr.ndim != 1:
            raise ValueError(f"Z must be 1-D, got shape {tuple(Z_arr.shape)!r}")
        if Z_arr.shape[0] >= 2:
            try:
                if not bool(jnp.all(jnp.diff(Z_arr) > 0)):
                    raise ValueError("Z must be strictly increasing")
            except jax.errors.TracerBoolConversionError:
                pass
        self.sde_kernel = sde_kernel
        self.Z = Z_arr
        self.mean_fn = mean_fn
        self.solver = solver
        self.jitter = float(jitter)

    @property
    def num_inducing(self) -> int:
        """Number of inducing time points :math:`M`."""
        return self.Z.shape[0]

    def mean(self, times: Float[Array, " N"]) -> Float[Array, " N"]:
        """Evaluate the mean function at ``times``; zero by default."""
        if self.mean_fn is None:
            return jnp.zeros_like(times)
        return self.mean_fn(times)

    def _stationary_variance(self) -> Float[Array, ""]:
        r"""Stationary marginal variance ``H P_inf H^T`` (= kernel variance)."""
        _F, _L, H, _Qc, P_inf = self.sde_kernel.sde_params()
        return (H @ P_inf @ H.T)[0, 0]

    def inducing_operator(self) -> lx.AbstractLinearOperator:
        r"""Return ``K_{ZZ} + \text{jitter}\,I`` as a PSD ``lineax`` operator."""
        K_zz = _sde_autocov_pairs(self.sde_kernel, self.Z, self.Z)
        K_zz = 0.5 * (K_zz + K_zz.T)
        K_zz = K_zz + self.jitter * jnp.eye(K_zz.shape[0], dtype=K_zz.dtype)
        return _psd_operator(K_zz)

    def cross_covariance(self, times: Float[Array, " N"]) -> Float[Array, "N M"]:
        """``K_{XZ}`` — pairwise SDE autocov between training and inducing times."""
        return _sde_autocov_pairs(self.sde_kernel, jnp.asarray(times), self.Z)

    def kernel_diag(self, times: Float[Array, " N"]) -> Float[Array, " N"]:
        r"""Prior diagonal :math:`\mathrm{diag}\,K(X, X)`.

        Constant for stationary SDE kernels — equal to ``H P_inf H^T``.
        """
        var0 = self._stationary_variance()
        return jnp.broadcast_to(var0, jnp.asarray(times).shape).astype(self.Z.dtype)

    def predictive_blocks(
        self, times: Float[Array, " N"]
    ) -> tuple[
        lx.AbstractLinearOperator,
        Float[Array, "N M"],
        Float[Array, " N"],
    ]:
        r"""Return ``(K_zz_op, K_xz, K_xx_diag)`` for the SVGP predictive math.

        Mirrors :meth:`SparseGPPrior.predictive_blocks`. Delegates to the
        three independent accessors:

        * :meth:`inducing_operator` — ``K_{ZZ} + \\text{jitter}\\,I``
          wrapped as a PSD :mod:`lineax` operator.
        * :meth:`cross_covariance` — pairwise SDE autocov ``K_{XZ}``.
        * :meth:`kernel_diag` — prior marginal variance, constant for
          stationary SDE kernels.

        Each accessor calls :meth:`SDEKernel.sde_params` independently;
        these calls are cheap (parameter unpacking, not a kernel build)
        so no shared-state caching is performed.
        """
        K_zz_op = self.inducing_operator()
        K_xz = self.cross_covariance(times)
        K_xx_diag = self.kernel_diag(times)
        return K_zz_op, K_xz, K_xx_diag

    def _resolved_solver(self) -> AbstractSolverStrategy:
        return DenseSolver() if self.solver is None else self.solver  # ty: ignore[invalid-return-type]

    def log_prob(self, u: Float[Array, " M"]) -> Float[Array, ""]:
        r"""Log-density under the inducing prior.

        :math:`p(u) = \mathcal{N}(0,\, K_{ZZ} + \text{jitter}\,I)`.
        """
        m = jnp.zeros(self.num_inducing, dtype=u.dtype)
        return gaussian_log_prob(
            m, self.inducing_operator(), u, solver=self._resolved_solver()
        )

    def sample(self, key: Array) -> Float[Array, " M"]:
        r"""Draw :math:`u \sim p(u)` from the inducing prior."""
        op = self.inducing_operator()
        loc = jnp.zeros(self.num_inducing, dtype=op.out_structure().dtype)
        mvn = MultivariateNormal(loc, op, solver=self._resolved_solver())
        return mvn.sample(key)


# --- result type for predictions -----------------------------------------


class SparseConditionedMarkovGP(eqx.Module):
    """Sparse Markov GP fitted to a guide.

    Bundles the :class:`SparseMarkovGPPrior` and a fitted
    :class:`pyrox.gp.Guide` so that ``predict(t_star)`` can be called
    against arbitrary test times. The math is the standard SVGP
    predictive

    .. math::

        \\mu_*(t) = K_{*Z} K_{ZZ}^{-1} m_q + \\mu(t),\\qquad
        \\sigma_*^2(t) = k(t, t) - K_{*Z} K_{ZZ}^{-1} K_{Z*}
                        + K_{*Z} K_{ZZ}^{-1} S_q K_{ZZ}^{-1} K_{Z*}.

    Cost is :math:`O(M^3 + |t_*|\\,M)` per call.
    """

    prior: SparseMarkovGPPrior
    guide: Guide

    def predict(
        self, t_star: Float[Array, " M_star"]
    ) -> tuple[Float[Array, " M_star"], Float[Array, " M_star"]]:
        """Predictive ``(mean, var)`` at arbitrary test times."""
        K_zz_op, K_xz, K_xx_diag = self.prior.predictive_blocks(t_star)
        mean, var = self.guide.predict(K_xz, K_zz_op, K_xx_diag)  # ty: ignore[unresolved-attribute]
        return mean + self.prior.mean(t_star), var


# --- ELBO + NumPyro hook --------------------------------------------------


def sparse_markov_elbo(
    prior: SparseMarkovGPPrior,
    guide: Guide,
    likelihood: Likelihood,
    times: Float[Array, " N"],
    y: Float[Array, " N"],
    *,
    integrator: AbstractIntegrator | None = None,
) -> Float[Array, ""]:
    r"""Sparse variational ELBO for :class:`SparseMarkovGPPrior`.

    Mirrors :func:`pyrox.gp.svgp_elbo` for the SDE-derived sparse
    Markov prior. Builds the SVGP predictive blocks
    :math:`(K_{ZZ}, K_{XZ}, \mathrm{diag}\,K_{XX})` from the prior, asks
    the guide for the predictive marginals
    :math:`(\mu_n, \sigma_n^2) = q(f_n)`, and combines them with a closed-form
    Gaussian or quadrature-based expected log-likelihood and the
    inducing KL term:

    .. math::

        \mathcal{L} = \sum_n \mathbb{E}_{q(f_n)}[\log p(y_n \mid f_n)]
                    - \mathrm{KL}[q(u) \\| p(u)].

    Unlike :func:`pyrox.gp.svgp_elbo`, ``times`` stays as a 1-D vector
    of shape ``(N,)`` — the SDE-pair autocov works on raw 1-D times and
    has no need for a feature dimension.

    Args:
        prior: :class:`SparseMarkovGPPrior` over an SDE kernel and
            inducing time grid.
        guide: Variational guide over inducing values.
        likelihood: Observation model.
        times: Training times of shape ``(N,)``.
        y: Observations of shape ``(N,)``.
        integrator: ``gaussx`` integrator for non-conjugate
            likelihoods. ``None`` is fine for
            :class:`pyrox.gp.GaussianLikelihood`.

    Returns:
        Scalar ELBO value (higher is better).

    Raises:
        ValueError: If a non-conjugate likelihood is used without an
            integrator.
    """
    from gaussx import variational_elbo_gaussian

    times_arr = jnp.asarray(times)
    K_zz_op, K_xz, K_xx_diag = prior.predictive_blocks(times_arr)
    f_loc, f_var = guide.predict(K_xz, K_zz_op, K_xx_diag)  # ty: ignore[unresolved-attribute]
    f_loc = f_loc + prior.mean(times_arr)
    kl = guide.kl_divergence(K_zz_op)  # ty: ignore[unresolved-attribute]

    if isinstance(likelihood, GaussianLikelihood):
        return variational_elbo_gaussian(
            y,
            f_loc,
            f_var,
            likelihood.noise_var,  # ty: ignore[invalid-argument-type]
            kl,
        )

    if integrator is None:
        raise ValueError(
            "Non-conjugate likelihoods require an integrator "
            "(e.g. gaussx.GaussHermiteIntegrator). "
            "Pass integrator=GaussHermiteIntegrator(order=20)."
        )
    ell = _ell_numerical(likelihood, y, f_loc, f_var, integrator)
    return ell - kl


def sparse_markov_factor(
    name: str,
    prior: SparseMarkovGPPrior,
    guide: Guide,
    likelihood: Likelihood,
    times: Float[Array, " N"],
    y: Float[Array, " N"],
    *,
    integrator: AbstractIntegrator | None = None,
) -> None:
    """Register :func:`sparse_markov_elbo` as a NumPyro factor site."""
    numpyro.factor(
        name,
        sparse_markov_elbo(prior, guide, likelihood, times, y, integrator=integrator),
    )
