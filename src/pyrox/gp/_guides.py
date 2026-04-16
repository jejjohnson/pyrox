"""Sparse SVGP variational guide families — concrete :class:`Guide` types.

Five guides over the inducing values ``u`` of a :class:`SparseGPPrior`:

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
* :class:`NaturalGuide` — :math:`q(u) = \\mathcal{N}(m, S)` parameterized
  in *natural form* :math:`(\\eta_1, \\eta_2)` with
  :math:`\\eta_1 = S^{-1}m` and :math:`\\eta_2 = -\\tfrac{1}{2} S^{-1}`.
  Exposes a damped :meth:`NaturalGuide.natural_update` for natural-
  gradient and CVI-style workflows that update ``q`` in natural-
  parameter space.
* :class:`DeltaGuide` — point-mass :math:`q(u) = \\delta(u - \\text{loc})`
  for MAP-style workflows. ``log_prob`` is constant and
  :meth:`DeltaGuide.kl_divergence` returns the loc-dependent
  :math:`-\\log p(\\text{loc})` so ``ELL - kl_divergence`` reduces to
  the joint log-density.

All five expose the same building-block interface:

* :meth:`sample(key)` — raw draw from ``q(u)`` (or ``q(v)`` for the
  whitened guide; deterministic for :class:`DeltaGuide`). Does not
  touch the NumPyro trace.
* :meth:`log_prob(u)` — variational log density at ``u`` (or ``v``).
* :meth:`kl_divergence(prior_cov)` — closed-form KL against the inducing
  prior covariance ``K_zz + jitter I``. The whitened guide ignores its
  argument; the KL is against ``N(0, I)``. The delta guide returns the
  loc-dependent ``-log p(loc)``.
* :meth:`predict(K_xz, K_zz_op, K_xx_diag)` — predictive mean and
  variance at ``X``. ``K_xz`` and ``K_xx_diag`` come from
  :meth:`SparseGPPrior.cross_covariance` and
  :meth:`SparseGPPrior.kernel_diag`; ``K_zz_op`` from
  :meth:`SparseGPPrior.inducing_operator`.

All Gaussian linear-algebra is delegated to ``gaussx``: natural-parameter
conversions go through :func:`gaussx.natural_to_mean_cov` /
:func:`gaussx.mean_cov_to_natural`, the damped natural update through
:func:`gaussx.damped_natural_update`, log-densities through
:func:`gaussx.gaussian_log_prob`, KL through
:func:`gaussx.dist_kl_divergence`, and Cholesky through
:func:`gaussx.cholesky` / :func:`gaussx.safe_cholesky`. The same
primitives back the future natural-gradient and CVI inference paths.

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
    damped_natural_update,
    dist_kl_divergence,
    gaussian_log_prob,
    natural_to_mean_cov,
    safe_cholesky,
    whitened_svgp_predict,
)
from jaxtyping import Array, Float

from pyrox.gp._protocols import Guide


def _negsemi_cov_operator(M: Float[Array, "M M"]) -> lx.AbstractLinearOperator:
    """Wrap a symmetric negative-(semi)definite array as a tagged operator."""
    return lx.MatrixLinearOperator(M, lx.negative_semidefinite_tag)  # ty: ignore[invalid-return-type]


def _possemi_cov_operator(M: Float[Array, "M M"]) -> lx.AbstractLinearOperator:
    """Wrap a symmetric positive-(semi)definite array as a tagged operator."""
    return lx.MatrixLinearOperator(M, lx.positive_semidefinite_tag)  # ty: ignore[invalid-return-type]


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
    # Cholesky via gaussx.safe_cholesky — adaptive jitter handles the
    # near-singular S=0 case (DeltaGuide) and the float32 paths where a
    # hard-coded jitter rounds to zero. Symmetrize first for numerics.
    Sv = 0.5 * (Sv + Sv.T)
    Lv = safe_cholesky(_possemi_cov_operator(Sv))
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


class NaturalGuide(Guide):
    r"""Natural-parameter Gaussian variational posterior over inducing values.

    Parameterizes :math:`q(u) = \mathcal{N}(m, S)` in *natural form*
    with parameters

    .. math::

        \eta_1 = S^{-1} m, \qquad
        \eta_2 = -\tfrac{1}{2} S^{-1}.

    The moments are recovered on demand via :attr:`mean` and
    :attr:`covariance`, which delegate to :func:`gaussx.natural_to_mean_cov`.

    The natural form is the parameterization of choice for *natural-
    gradient* and *conjugate-computation variational inference* (CVI)
    workflows: when the true posterior is in the same exponential family
    as the prior, the natural-gradient direction equals the difference
    in natural parameters. :meth:`natural_update` exposes that step
    with a damping factor ``rho`` and delegates to
    :func:`gaussx.damped_natural_update` so the same primitive is shared
    with future natural-gradient EP / VI / Newton workflows.

    Attributes:
        nat1: First natural parameter ``eta_1`` of shape ``(M,)``.
        nat2: Second natural parameter ``eta_2`` of shape ``(M, M)``,
            symmetric negative-definite.
    """

    nat1: Float[Array, " M"]
    nat2: Float[Array, "M M"]

    @classmethod
    def init(
        cls,
        num_inducing: int,
        *,
        scale: float = 1.0,
    ) -> NaturalGuide:
        """Construct a guide initialized to ``N(0, scale^2 I)`` in moment space.

        Mapped to natural form this is ``eta_1 = 0`` and
        ``eta_2 = -1 / (2 scale^2) * I`` — see :meth:`FullRankGuide.init`
        for the dtype convention.
        """
        n = num_inducing
        nat1 = jnp.zeros(n)
        nat2 = (-0.5 / (scale**2)) * jnp.eye(n)
        return cls(nat1=nat1, nat2=nat2)

    def _moments(self) -> tuple[Float[Array, " M"], Float[Array, "M M"]]:
        """Return ``(mean, cov_array)`` via :func:`gaussx.natural_to_mean_cov`."""
        nat2_op = _negsemi_cov_operator(self.nat2)
        m, cov_op = natural_to_mean_cov(self.nat1, nat2_op)
        cov = cov_op.as_matrix()
        cov = 0.5 * (cov + cov.T)
        return m, cov

    @property
    def covariance(self) -> Float[Array, "M M"]:
        """Recover the moment-form covariance ``S = (-2 nat2)^{-1}``."""
        return self._moments()[1]

    @property
    def mean(self) -> Float[Array, " M"]:
        """Recover the moment-form mean ``m = S nat1``."""
        return self._moments()[0]

    def sample(self, key: Array) -> Float[Array, " M"]:
        r"""Draw ``u = m + L \epsilon`` with ``\epsilon ~ N(0, I)``."""
        m, cov = self._moments()
        L = safe_cholesky(_possemi_cov_operator(cov))
        eps = jax.random.normal(key, m.shape, dtype=m.dtype)
        return m + L @ eps

    def log_prob(self, u: Float[Array, " ..."]) -> Float[Array, ""]:  # ty: ignore[invalid-method-override]
        r"""Variational log density ``\log q(u)``.

        Computed via :func:`gaussx.gaussian_log_prob`.
        """
        m, cov = self._moments()
        return gaussian_log_prob(m, _possemi_cov_operator(cov), u)

    def kl_divergence(self, prior_cov: lx.AbstractLinearOperator) -> Float[Array, ""]:
        r"""``KL(q(u) || p(u))`` against an inducing prior with zero mean."""
        m, cov = self._moments()
        p_loc = jnp.zeros_like(m)
        return dist_kl_divergence(m, _possemi_cov_operator(cov), p_loc, prior_cov)

    def predict(
        self,
        K_xz: Float[Array, "N M"],
        K_zz_op: lx.AbstractLinearOperator,
        K_xx_diag: Float[Array, " N"],
    ) -> tuple[Float[Array, " N"], Float[Array, " N"]]:
        """Predictive ``(mean, variance)`` at points with cross-cov ``K_xz``."""
        m, cov = self._moments()
        return _svgp_predict_unwhitened(K_xz, K_zz_op, K_xx_diag, m, cov)

    def natural_update(
        self,
        nat1_hat: Float[Array, " M"],
        nat2_hat: Float[Array, "M M"],
        rho: float | Float[Array, ""] = 1.0,
    ) -> NaturalGuide:
        r"""Damped natural-parameter update via :func:`gaussx.damped_natural_update`.

        Returns a new :class:`NaturalGuide` whose natural parameters are
        the convex combination

        .. math::

            \eta_i \leftarrow (1 - \rho)\,\eta_i + \rho\,\hat{\eta}_i,
            \quad i \in \{1, 2\}.

        The damping factor ``rho`` interpolates between the current
        guide (``rho=0``) and the candidate update
        ``(nat1_hat, nat2_hat)`` (``rho=1``). Convex combinations in
        natural-parameter space preserve membership in the Gaussian
        exponential family — in particular ``rho * nat2_hat + (1 - rho)
        * self.nat2`` stays symmetric negative-definite when both
        endpoints are. CVI-style site updates rely on exactly this
        property.
        """
        new_nat1, new_nat2 = damped_natural_update(
            self.nat1,
            self.nat2,
            nat1_hat,
            nat2_hat,
            lr=rho,  # ty: ignore[invalid-argument-type]
        )
        return NaturalGuide(nat1=new_nat1, nat2=new_nat2)  # ty: ignore[invalid-return-type]


class DeltaGuide(Guide):
    r"""Point-mass (MAP-style) variational posterior over inducing values.

    .. math::

        q(u) = \delta(u - \text{loc}),

    so all the variational mass concentrates on a single point. Pairs
    with the same SVGP ELBO infrastructure as the Gaussian guides but
    reduces it to MAP estimation: when ``ELL - kl_divergence`` is the
    objective, :meth:`kl_divergence` returns the loc-dependent
    :math:`-\log p(\text{loc})` so the objective becomes the joint
    log-density ``log p(y, loc) = ELL(loc) + log p(loc)``, recovering
    standard MAP estimation of the inducing values.

    Attributes:
        loc: The point at which the variational mass concentrates,
            shape ``(M,)``. This is also the MAP estimate of the
            inducing values when used inside a maximization loop.
    """

    loc: Float[Array, " M"]

    @classmethod
    def init(cls, num_inducing: int) -> DeltaGuide:
        """Construct a guide initialized to ``loc = 0`` — see
        :meth:`FullRankGuide.init` for the dtype convention."""
        return cls(loc=jnp.zeros(num_inducing))

    def sample(self, key: Array) -> Float[Array, " M"]:
        """Return ``self.loc`` — the variational draw is deterministic."""
        del key
        return self.loc

    def log_prob(self, u: Float[Array, " ..."]) -> Float[Array, ""]:  # ty: ignore[invalid-method-override]
        r"""Variational log density — constant ``0`` by convention.

        The strict density of a Dirac delta is ``+inf`` at ``loc`` and
        ``-inf`` everywhere else, which carries no useful gradient
        information. Returning ``0`` matches the Pyro / NumPyro
        ``AutoDelta`` convention: the differentiable MAP signal lives
        entirely in :meth:`kl_divergence`.
        """
        del u
        return jnp.zeros((), dtype=self.loc.dtype)

    def kl_divergence(self, prior_cov: lx.AbstractLinearOperator) -> Float[Array, ""]:
        r"""Return the loc-dependent ``-log p(loc)`` for the MAP objective.

        The strict KL of a Dirac delta against a continuous prior is
        ``+inf``, but the only loc-dependent piece is the negative log
        prior density at ``loc``,

        .. math::

            -\log p(\text{loc}) =
                \tfrac{1}{2} \text{loc}^\top K^{-1} \text{loc}
                + \tfrac{1}{2} \log |K|
                + \tfrac{M}{2} \log(2\pi),

        so we return that. With this convention the standard ELBO
        ``ELL - kl_divergence`` reduces to the joint log-density
        ``log p(y, loc)``, which is exactly the MAP objective. Computed
        via :func:`gaussx.gaussian_log_prob` so the same solver / logdet
        primitives back this path as the rest of the GP surface.
        """
        loc = self.loc
        return -gaussian_log_prob(jnp.zeros_like(loc), prior_cov, loc)

    def predict(
        self,
        K_xz: Float[Array, "N M"],
        K_zz_op: lx.AbstractLinearOperator,
        K_xx_diag: Float[Array, " N"],
    ) -> tuple[Float[Array, " N"], Float[Array, " N"]]:
        r"""Predictive ``(mean, variance)`` conditioning on ``u = loc``.

        With ``u = loc`` deterministically, the predictive is the prior
        conditional ``p(f_* | u = loc)``: the mean is
        ``K_{xZ} K_{ZZ}^{-1} loc`` and the variance is the prior
        reduction ``k(x, x) - K_{xZ} K_{ZZ}^{-1} K_{Zx}`` *exactly* —
        no posterior-uncertainty contribution and no Cholesky jitter.

        Implemented by calling :func:`gaussx.whitened_svgp_predict`
        directly with the whitened mean ``L_{ZZ}^{-1} loc`` and a zero
        whitened Cholesky factor. The shared
        :func:`_svgp_predict_unwhitened` helper would route the zero
        variational covariance through :func:`gaussx.safe_cholesky`,
        which injects jitter to make the input PD and would add a tiny
        spurious variance term. Bypassing that path keeps the
        :class:`DeltaGuide` predictive numerically equal to the prior
        conditional.
        """
        m_size = self.loc.shape[0]
        L_zz = cholesky(K_zz_op)
        u_mean_white = lx.linear_solve(L_zz, self.loc).value
        zero_chol = jnp.zeros((m_size, m_size), dtype=self.loc.dtype)
        return whitened_svgp_predict(K_zz_op, K_xz, u_mean_white, zero_chol, K_xx_diag)
