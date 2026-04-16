"""Sparse SVGP variational guide families — concrete :class:`Guide` types.

Three guides over the inducing values ``u`` of a :class:`SparseGPPrior`:

* :class:`FullRankGuide` — :math:`q(u) = \\mathcal{N}(m, L_S L_S^\\top)`,
  expressive but ``O(M^3)`` per ELBO call.
* :class:`MeanFieldGuide` — :math:`q(u) = \\mathcal{N}(m, \\mathrm{diag}(s^2))`,
  cheap and easy to optimize but cannot capture cross-correlations between
  inducing values.
* :class:`WhitenedGuide` — :math:`q(v) = \\mathcal{N}(m_v, L_v L_v^\\top)` in
  whitened coordinates ``v`` such that ``u = L_{ZZ} v``. The KL term is
  against the standard normal and the predictive uses
  :func:`gaussx.whitened_svgp_predict` directly. This is the standard
  parameterization for stable SVGP optimization (Hensman et al., 2015).

All three expose the same building-block interface:

* :meth:`sample(key)` — raw draw from ``q(u)`` (or ``q(v)`` for the
  whitened guide). Does not touch the NumPyro trace.
* :meth:`log_prob(u)` — variational log density at ``u`` (or ``v``).
* :meth:`kl_divergence(prior_cov)` — closed-form KL against the inducing
  prior covariance ``K_zz + jitter I``. The whitened guide ignores its
  argument; the KL is against ``N(0, I)``.
* :meth:`predict(K_xz, K_zz_op, K_xx_diag)` — predictive mean and
  variance at ``X``. ``K_xz`` and ``K_xx_diag`` come from
  :meth:`SparseGPPrior.cross_covariance` and
  :meth:`SparseGPPrior.kernel_diag`; ``K_zz_op`` from
  :meth:`SparseGPPrior.inducing_operator`.

The sparse ELBO entry point that wires these into NumPyro lands in the
Wave 3 inference issue. Until then the user assembles the ELBO manually:
sample ``u`` (or ``v``) via ``numpyro.sample``, predict ``f`` at the
training inputs, integrate the per-point log-likelihood under
``q(f_n)``, and subtract :meth:`kl_divergence`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx
from gaussx import (
    cholesky,
    dist_kl_divergence,
    whitened_svgp_predict,
)
from jaxtyping import Array, Float

from pyrox.gp._protocols import Guide


def _full_cov_operator(scale_tril: Float[Array, "M M"]) -> lx.AbstractLinearOperator:
    """Wrap ``L L^T`` as a PSD lineax operator from a Cholesky factor."""
    cov = scale_tril @ scale_tril.T
    return lx.MatrixLinearOperator(cov, lx.positive_semidefinite_tag)  # ty: ignore[invalid-return-type]


def _diag_cov_operator(scale: Float[Array, " M"]) -> lx.AbstractLinearOperator:
    """Wrap ``diag(s^2)`` as a PSD lineax operator from a vector of scales."""
    cov = jnp.diag(scale**2)
    return lx.MatrixLinearOperator(cov, lx.positive_semidefinite_tag)  # ty: ignore[invalid-return-type]


def _svgp_predict_unwhitened(
    K_xz: Float[Array, "N M"],
    K_zz_op: lx.AbstractLinearOperator,
    K_xx_diag: Float[Array, " N"],
    u_mean: Float[Array, " M"],
    u_cov: Float[Array, "M M"],
) -> tuple[Float[Array, " N"], Float[Array, " N"]]:
    r"""Predictive moments for an *unwhitened* SVGP guide.

    Implements

    .. math::

        \mu_*(x) &= K_{xZ} K_{ZZ}^{-1} m, \\
        \sigma^2_*(x) &= k(x,x)
            - K_{xZ} K_{ZZ}^{-1} K_{Zx}
            + K_{xZ} K_{ZZ}^{-1} S K_{ZZ}^{-1} K_{Zx},

    by reduction to the whitened SVGP predictive
    (:func:`gaussx.whitened_svgp_predict`): if ``u = L_{ZZ} v`` with
    ``v ~ q(v) = N(m_v, S_v)`` then ``m_v = L_{ZZ}^{-1} m`` and
    ``S_v = L_{ZZ}^{-1} S L_{ZZ}^{-T}``. The whitened predictive uses
    only ``L_{v} = chol(S_v)`` and is ``O(M^2 N)`` after the
    ``O(M^3)`` Cholesky. Doing the conversion once keeps the
    unwhitened predictive numerically equivalent to the whitened path.
    """
    L_zz = cholesky(K_zz_op)
    # m_v = L_zz^{-1} m
    m_v = lx.linear_solve(L_zz, u_mean).value
    # L_zz^{-1} S L_zz^{-T} via two solves
    Sv_left = jax.vmap(
        lambda col: lx.linear_solve(L_zz, col).value, in_axes=1, out_axes=1
    )(u_cov)  # L_zz^{-1} S, shape (M, M)
    Sv = jax.vmap(lambda col: lx.linear_solve(L_zz, col).value, in_axes=1, out_axes=1)(
        Sv_left.T
    ).T  # L_zz^{-1} S L_zz^{-T}, shape (M, M)
    # Symmetrize for numerical safety before Cholesky, then add a
    # dtype-aware diagonal jitter that's actually meaningful in the
    # working precision (a fixed 1e-12 vanishes when added to ~unit
    # values in float32). The ``max`` keeps the float64 behavior at the
    # historical 1e-12; only the float32 path is bumped up to ~10 * eps.
    Sv = 0.5 * (Sv + Sv.T)
    jitter = max(1e-12, float(jnp.finfo(Sv.dtype).eps) * 10.0)
    Lv = jnp.linalg.cholesky(Sv + jitter * jnp.eye(Sv.shape[0], dtype=Sv.dtype))
    return whitened_svgp_predict(K_zz_op, K_xz, m_v, Lv, K_xx_diag)


class FullRankGuide(Guide):
    r"""Full-rank Gaussian variational posterior over inducing values ``u``.

    .. math::

        q(u) = \mathcal{N}(m,\, L_S L_S^\top),

    parameterized by the mean ``mean`` of shape ``(M,)`` and the
    *lower-triangular* Cholesky factor ``scale_tril`` of shape ``(M, M)``.

    Attributes:
        mean: Variational mean ``m`` of shape ``(M,)``.
        scale_tril: Lower-triangular Cholesky factor of the variational
            covariance, shape ``(M, M)``. The covariance is
            ``S = scale_tril @ scale_tril.T``.
    """

    mean: Float[Array, " M"]
    scale_tril: Float[Array, "M M"]

    @classmethod
    def init(
        cls,
        num_inducing: int,
        *,
        scale: float = 1.0,
    ) -> FullRankGuide:
        """Construct a guide initialized to ``N(0, scale^2 I)``.

        Uses JAX's default float dtype — set
        ``jax.config.update("jax_enable_x64", True)`` once at the top of
        your script if you want float64 inducing parameters.
        """
        m = jnp.zeros(num_inducing)
        L = scale * jnp.eye(num_inducing)
        return cls(mean=m, scale_tril=L)

    def sample(self, key: Array) -> Float[Array, " M"]:
        r"""Draw ``u = m + L_S \epsilon`` with ``\epsilon ~ N(0, I)``."""
        eps = jax.random.normal(key, self.mean.shape, dtype=self.mean.dtype)
        return self.mean + self.scale_tril @ eps

    def log_prob(self, u: Float[Array, " ..."]) -> Float[Array, ""]:  # ty: ignore[invalid-method-override]
        r"""Variational log density ``\log q(u)``."""
        diff = u - self.mean
        sol = jax.scipy.linalg.solve_triangular(self.scale_tril, diff, lower=True)
        log_det = 2.0 * jnp.sum(jnp.log(jnp.abs(jnp.diag(self.scale_tril))))
        n = self.mean.shape[0]
        return -0.5 * (jnp.sum(sol**2) + log_det + n * jnp.log(2.0 * jnp.pi))

    def kl_divergence(self, prior_cov: lx.AbstractLinearOperator) -> Float[Array, ""]:
        r"""``KL(q(u) || p(u))`` against an inducing prior with zero mean."""
        q_loc = self.mean
        q_cov = _full_cov_operator(self.scale_tril)
        p_loc = jnp.zeros_like(self.mean)
        return dist_kl_divergence(q_loc, q_cov, p_loc, prior_cov)

    def predict(
        self,
        K_xz: Float[Array, "N M"],
        K_zz_op: lx.AbstractLinearOperator,
        K_xx_diag: Float[Array, " N"],
    ) -> tuple[Float[Array, " N"], Float[Array, " N"]]:
        """Predictive ``(mean, variance)`` at points with cross-cov ``K_xz``."""
        u_cov = self.scale_tril @ self.scale_tril.T
        return _svgp_predict_unwhitened(K_xz, K_zz_op, K_xx_diag, self.mean, u_cov)


class MeanFieldGuide(Guide):
    r"""Diagonal Gaussian variational posterior over inducing values ``u``.

    .. math::

        q(u) = \mathcal{N}(m,\, \mathrm{diag}(s^2)),

    parameterized by the mean ``mean`` and the per-coordinate standard
    deviations ``scale``.

    Attributes:
        mean: Variational mean ``m`` of shape ``(M,)``.
        scale: Per-coordinate standard deviations ``s`` of shape ``(M,)``.
            Must be strictly positive.
    """

    mean: Float[Array, " M"]
    scale: Float[Array, " M"]

    @classmethod
    def init(
        cls,
        num_inducing: int,
        *,
        scale: float = 1.0,
    ) -> MeanFieldGuide:
        """Construct a guide initialized to ``N(0, scale^2 I)`` — see
        :meth:`FullRankGuide.init` for the dtype convention."""
        m = jnp.zeros(num_inducing)
        s = jnp.full(num_inducing, scale)
        return cls(mean=m, scale=s)

    def sample(self, key: Array) -> Float[Array, " M"]:
        r"""Draw ``u = m + s \odot \epsilon`` with ``\epsilon ~ N(0, I)``."""
        eps = jax.random.normal(key, self.mean.shape, dtype=self.mean.dtype)
        return self.mean + self.scale * eps

    def log_prob(self, u: Float[Array, " ..."]) -> Float[Array, ""]:  # ty: ignore[invalid-method-override]
        r"""Variational log density ``\log q(u)``, summed across coordinates."""
        z = (u - self.mean) / self.scale
        return -0.5 * jnp.sum(z**2 + jnp.log(2.0 * jnp.pi) + 2.0 * jnp.log(self.scale))

    def kl_divergence(self, prior_cov: lx.AbstractLinearOperator) -> Float[Array, ""]:
        r"""``KL(q(u) || p(u))`` against an inducing prior with zero mean."""
        q_loc = self.mean
        q_cov = _diag_cov_operator(self.scale)
        p_loc = jnp.zeros_like(self.mean)
        return dist_kl_divergence(q_loc, q_cov, p_loc, prior_cov)

    def predict(
        self,
        K_xz: Float[Array, "N M"],
        K_zz_op: lx.AbstractLinearOperator,
        K_xx_diag: Float[Array, " N"],
    ) -> tuple[Float[Array, " N"], Float[Array, " N"]]:
        """Predictive ``(mean, variance)`` at the points whose cross-cov is ``K_xz``."""
        u_cov = jnp.diag(self.scale**2)
        return _svgp_predict_unwhitened(K_xz, K_zz_op, K_xx_diag, self.mean, u_cov)


class WhitenedGuide(Guide):
    r"""Whitened-coordinate Gaussian variational posterior over inducing values.

    Parameterizes ``q(v) = N(m_v, L_v L_v^T)`` in *whitened* coordinates,
    so the inducing values are ``u = L_{ZZ} v`` with
    ``L_{ZZ} = chol(K_{ZZ} + jitter I)``. In whitened coordinates the
    prior is ``p(v) = N(0, I)`` and the KL term has a simple closed form
    that is independent of the kernel — the standard reparameterization
    for numerically stable SVGP optimization (Hensman et al., 2015).

    Attributes:
        mean: Variational mean ``m_v`` in whitened coordinates,
            shape ``(M,)``.
        scale_tril: Lower-triangular Cholesky factor of the whitened
            variational covariance, shape ``(M, M)``. The covariance in
            whitened space is ``L_v @ L_v.T``.
    """

    mean: Float[Array, " M"]
    scale_tril: Float[Array, "M M"]

    @classmethod
    def init(
        cls,
        num_inducing: int,
        *,
        scale: float = 1.0,
    ) -> WhitenedGuide:
        """Construct a guide initialized to ``N(0, scale^2 I)`` in whitened
        space — see :meth:`FullRankGuide.init` for the dtype convention."""
        m = jnp.zeros(num_inducing)
        L = scale * jnp.eye(num_inducing)
        return cls(mean=m, scale_tril=L)

    def sample(self, key: Array) -> Float[Array, " M"]:
        """Draw a single whitened sample ``v = m_v + L_v \\epsilon``."""
        eps = jax.random.normal(key, self.mean.shape, dtype=self.mean.dtype)
        return self.mean + self.scale_tril @ eps

    def log_prob(self, v: Float[Array, " ..."]) -> Float[Array, ""]:  # ty: ignore[invalid-method-override]
        r"""Whitened variational log density ``\log q(v)``."""
        diff = v - self.mean
        sol = jax.scipy.linalg.solve_triangular(self.scale_tril, diff, lower=True)
        log_det = 2.0 * jnp.sum(jnp.log(jnp.abs(jnp.diag(self.scale_tril))))
        n = self.mean.shape[0]
        return -0.5 * (jnp.sum(sol**2) + log_det + n * jnp.log(2.0 * jnp.pi))

    def kl_divergence(
        self,
        prior_cov: lx.AbstractLinearOperator | None = None,
    ) -> Float[Array, ""]:
        r"""``KL(q(v) || N(0, I))`` — kernel-free closed form.

        Computes

        .. math::

            \mathrm{KL}(\mathcal{N}(m_v, L_v L_v^\top) \,\|\,
                       \mathcal{N}(0, I))
            = \tfrac{1}{2} \bigl(
                \|m_v\|^2 + \|L_v\|_F^2 - M
                - 2 \sum_i \log |[L_v]_{ii}|
            \bigr).

        The ``prior_cov`` argument is accepted for signature parity with
        :meth:`FullRankGuide.kl_divergence` and
        :meth:`MeanFieldGuide.kl_divergence`, but is ignored — the
        whitened prior is the standard normal regardless of ``K_{ZZ}``.
        """
        del prior_cov
        m = self.mean
        L = self.scale_tril
        n = m.shape[0]
        quad = jnp.sum(m**2)
        trace = jnp.sum(L**2)  # ||L||_F^2
        log_det = 2.0 * jnp.sum(jnp.log(jnp.abs(jnp.diag(L))))
        return 0.5 * (quad + trace - n - log_det)

    def predict(
        self,
        K_xz: Float[Array, "N M"],
        K_zz_op: lx.AbstractLinearOperator,
        K_xx_diag: Float[Array, " N"],
    ) -> tuple[Float[Array, " N"], Float[Array, " N"]]:
        """Predictive ``(mean, variance)`` via :func:`gaussx.whitened_svgp_predict`."""
        return whitened_svgp_predict(
            K_zz_op, K_xz, self.mean, self.scale_tril, K_xx_diag
        )
