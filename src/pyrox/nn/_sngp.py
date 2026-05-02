"""Spectral-Normalized Gaussian Process (SNGP) output layer.

* :class:`LaplaceRandomFeatureCovariance` — pure-functional container
  for the Laplace-approximation precision matrix used at SNGP test
  time. Updated via the EMA of feature outer products during training.
* :class:`RandomFeatureGaussianProcess` — SNGP output head (Liu et al.,
  2020). RFF feature map :math:`\\phi(x)` plus a linear mean head and
  a Laplace covariance over the linear weights.

This module implements *just the SNGP head* — spectral normalisation
of upstream dense layers is a separate concern (the design doc's
Tier 2 ``spectral_norm`` gap) and the user is responsible for that.
"""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as dist
from jaxtyping import Array, Float, PRNGKeyArray

from pyrox._core.pyrox_module import PyroxModule, pyrox_method


def _glorot_normal(
    key: PRNGKeyArray,
    fan_in: int,
    fan_out: int,
    scale: float = 1.0,
) -> Float[Array, "F_in F_out"]:
    std = scale * math.sqrt(2.0 / (fan_in + fan_out))
    return std * jr.normal(key, (fan_in, fan_out))


class LaplaceRandomFeatureCovariance(eqx.Module):
    r"""Laplace-approximation precision for an SNGP output head.

    Stores the precision matrix :math:`\hat{\Lambda} \in \mathbb{R}^{D \times D}`
    over the linear weights of the output layer. Updated as an
    exponential moving average of feature outer products during
    training:

    .. math::

        \hat{\Lambda}_{t+1} \leftarrow m\,\hat{\Lambda}_t
        + (1 - m)\,\frac{1}{B} \sum_{b=1}^{B} \phi(x_b)\,\phi(x_b)^\top.

    At test time the predictive variance for a feature vector
    :math:`\phi(x_*)` is

    .. math::

        \sigma^2(x_*) = \phi(x_*)^\top \hat{\Sigma}\, \phi(x_*),
        \qquad \hat{\Sigma} = \hat{\Lambda}^{-1},

    computed stably via a Cholesky solve.

    The container is *pure-functional*: :meth:`update` returns a new
    instance with an updated precision rather than mutating ``self``,
    matching how Equinox composes immutable PyTrees with optimisers.
    A small ridge :math:`\lambda I` initialises and stabilises the
    precision; choose :math:`\lambda` small relative to the expected
    feature scale.

    Attributes:
        precision: Current precision matrix :math:`\hat{\Lambda}`.
        momentum: EMA momentum :math:`m \in [0, 1]`. Higher values give
            slower updates; ``0.999`` works well for most settings.
        ridge: Diagonal ridge :math:`\lambda` for numerical stability.
    """

    precision: Float[Array, "D D"]
    momentum: float = eqx.field(default=0.999)
    ridge: float = eqx.field(default=1.0)

    @classmethod
    def init(
        cls,
        num_features: int,
        *,
        momentum: float = 0.999,
        ridge: float = 1.0,
    ) -> LaplaceRandomFeatureCovariance:
        """Construct a fresh covariance container with ``ridge * I`` precision."""
        if num_features <= 0:
            raise ValueError(f"num_features must be > 0; got {num_features}.")
        if not 0.0 <= momentum <= 1.0:
            raise ValueError(f"momentum must lie in [0, 1]; got {momentum}.")
        if ridge <= 0:
            raise ValueError(f"ridge must be > 0; got {ridge}.")
        return cls(
            precision=ridge * jnp.eye(num_features),
            momentum=momentum,
            ridge=ridge,
        )

    def update(self, features: Float[Array, "B D"]) -> LaplaceRandomFeatureCovariance:
        """Return a new container with EMA-updated precision."""
        B = features.shape[0]
        outer = (features.T @ features) / B
        new_precision = self.momentum * self.precision + (1.0 - self.momentum) * outer
        return eqx.tree_at(lambda c: c.precision, self, new_precision)

    def _chol(self) -> Float[Array, "D D"]:
        # Symmetrise to absorb floating-point asymmetry in the EMA.
        sym = 0.5 * (self.precision + self.precision.T)
        return jnp.linalg.cholesky(sym)

    def covariance(self) -> Float[Array, "D D"]:
        """Inverse of the precision matrix (one-shot Cholesky inversion)."""
        L = self._chol()
        D = self.precision.shape[0]
        return jax.scipy.linalg.cho_solve((L, True), jnp.eye(D))

    def variance_at(self, features: Float[Array, "N D"]) -> Float[Array, " N"]:
        r"""Per-row predictive variance :math:`\phi(x_n)^\top \hat{\Sigma}\,\phi(x_n)`.

        Computed via a triangular solve to avoid materialising the full
        :math:`D \times D` covariance:

        .. math::

            y = L^{-1} \phi(x_n)^\top, \qquad
            \sigma^2(x_n) = \lVert y \rVert_2^2
            = \phi(x_n)^\top (L L^\top)^{-1} \phi(x_n).
        """
        L = self._chol()
        y = jax.scipy.linalg.solve_triangular(L, features.T, lower=True)
        return jnp.sum(y * y, axis=0)


class RandomFeatureGaussianProcess(PyroxModule):
    r"""SNGP output layer (Liu et al., 2020).

    A random Fourier feature map followed by a learnable linear head,
    plus a Laplace-approximation covariance over the linear weights.
    The forward pass returns the mean prediction and (optionally) a
    per-input predictive variance summarising distance from the
    training distribution.

    Forward (mean):

    .. math::

        \phi(x) = \sqrt{\tfrac{2}{D}}\,\cos\!\bigl(W\, x / \ell + b\bigr),
        \qquad \mu(x) = \phi(x)\, H + b_H.

    The frequencies :math:`W` and bias :math:`b` of the RFF map are
    *frozen at init* (they implicitly define the kernel approximation);
    the lengthscale :math:`\ell`, the linear head :math:`H, b_H`, and
    the Laplace precision are the trainable / updated quantities.

    Predictive variance — when :math:`\hat{\Lambda}` is the current
    precision matrix:

    .. math::

        \sigma^2(x_*) = \phi(x_*)^\top \hat{\Lambda}^{-1}\, \phi(x_*).

    Training pattern (one minibatch):

    1. ``mean = layer(x)`` registers / reuses the trainable params and
       returns the mean prediction. Compute the loss, take a gradient
       step on the SVI parameter store as usual.
    2. After the gradient step, call
       ``new_layer = layer.update_precision(features)`` where
       ``features`` is the result of :meth:`feature_map` evaluated on
       the same minibatch using the *updated* parameters. This returns
       a new layer with the LRFC's precision EMA-updated.

    At inference, ``mean, var = layer(x, return_cov=True)`` produces
    the mean and the Laplace per-input predictive variance.

    Plate semantics:
        Same as the rest of ``pyrox.nn``'s Bayesian / heteroscedastic
        dense layers — call this layer outside
        ``numpyro.plate("data", ..., subsample_size=...)`` and only
        plate the observation likelihood.

    Attributes:
        in_features: Input dimension :math:`D_\mathrm{in}`.
        num_features: Number of random Fourier features :math:`D`.
        out_features: Output dimension :math:`D_\mathrm{out}`.
        init_lengthscale: Initial lengthscale :math:`\ell`. Optimised
            during training as a positive ``pyrox_param``.
        W_init: Frozen RFF frequencies, shape ``(D_in, D)``, drawn from
            a standard Normal (the RBF spectral density).
        bias_init: Frozen RFF biases, shape ``(D,)``, drawn from
            ``Uniform(0, 2 pi)``.
        output_linear_init: Init for the linear head, shape ``(D, D_out)``.
        covariance: The :class:`LaplaceRandomFeatureCovariance` instance.
        pyrox_name: Explicit scope name for NumPyro site registration.

    References:
        Liu, J. Z., et al. (2020). *Simple and Principled Uncertainty
        Estimation with Deterministic Deep Learning via Distance
        Awareness.* NeurIPS.
    """

    in_features: int = eqx.field(static=True)
    num_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)
    init_lengthscale: float = 1.0
    W_init: Float[Array, "D_in D"] | None = eqx.field(default=None)
    bias_init: Float[Array, " D"] | None = eqx.field(default=None)
    output_linear_init: Float[Array, "D D_out"] | None = eqx.field(default=None)
    covariance: LaplaceRandomFeatureCovariance | None = eqx.field(default=None)
    pyrox_name: str | None = None

    @classmethod
    def init(
        cls,
        key: PRNGKeyArray,
        in_features: int,
        num_features: int,
        out_features: int,
        *,
        init_lengthscale: float = 1.0,
        momentum: float = 0.999,
        ridge: float = 1.0,
        head_scale: float = 0.01,
        pyrox_name: str | None = None,
    ) -> RandomFeatureGaussianProcess:
        """Construct an SNGP head with frozen RFF freqs and an empty precision."""
        if in_features <= 0 or num_features <= 0 or out_features <= 0:
            raise ValueError(
                "in_features, num_features, out_features must all be > 0; "
                f"got {in_features=}, {num_features=}, {out_features=}."
            )
        if init_lengthscale <= 0:
            raise ValueError(f"init_lengthscale must be > 0; got {init_lengthscale}.")
        kw, kb, kh = jr.split(key, 3)
        W_init = jr.normal(kw, (in_features, num_features))
        bias_init = jr.uniform(kb, (num_features,), minval=0.0, maxval=2 * jnp.pi)
        output_linear_init = _glorot_normal(
            kh, num_features, out_features, scale=head_scale
        )
        cov = LaplaceRandomFeatureCovariance.init(
            num_features, momentum=momentum, ridge=ridge
        )
        return cls(
            in_features=in_features,
            num_features=num_features,
            out_features=out_features,
            init_lengthscale=init_lengthscale,
            W_init=W_init,
            bias_init=bias_init,
            output_linear_init=output_linear_init,
            covariance=cov,
            pyrox_name=pyrox_name,
        )

    def __post_init__(self) -> None:
        for name, attr, expected in (
            ("W_init", self.W_init, (self.in_features, self.num_features)),
            ("bias_init", self.bias_init, (self.num_features,)),
            (
                "output_linear_init",
                self.output_linear_init,
                (self.num_features, self.out_features),
            ),
        ):
            if attr is None:
                raise ValueError(
                    f"RandomFeatureGaussianProcess requires {name}. Use "
                    "RandomFeatureGaussianProcess.init(key, ...) to construct."
                )
            if attr.shape != expected:
                raise ValueError(f"{name} shape {attr.shape} != expected {expected}.")
        if self.covariance is None:
            raise ValueError(
                "RandomFeatureGaussianProcess requires a covariance container. "
                "Use RandomFeatureGaussianProcess.init(key, ...) to construct."
            )

    @pyrox_method
    def feature_map(self, x: Float[Array, "*batch D_in"]) -> Float[Array, "*batch D"]:
        r"""Random Fourier feature map: :math:`\phi(x) = \sqrt{2/D}\,\cos(Wx/\ell + b)`.

        Frequencies and bias are frozen ``pyrox_param`` sites — they
        are passed through SVI's param store with their init value but
        rarely move appreciably; the lengthscale is the active
        bandwidth control and is constrained positive.
        """
        W = self.pyrox_param("W", self.W_init)
        b = self.pyrox_param("bias", self.bias_init)
        ls = self.pyrox_param(
            "lengthscale",
            jnp.asarray(self.init_lengthscale),
            constraint=dist.constraints.positive,
        )
        z = x @ W / ls + b
        return jnp.sqrt(2.0 / self.num_features) * jnp.cos(z)

    @pyrox_method
    def __call__(
        self,
        x: Float[Array, "*batch D_in"],
        *,
        return_cov: bool = False,
    ) -> (
        Float[Array, "*batch D_out"]
        | tuple[Float[Array, "*batch D_out"], Float[Array, " *batch"]]
    ):
        features = self.feature_map(x)
        H = self.pyrox_param("output_linear", self.output_linear_init)
        b_out = self.pyrox_param(
            "output_bias", jnp.zeros(self.out_features, dtype=x.dtype)
        )
        mean = features @ H + b_out
        if return_cov:
            # variance_at expects a 2-D (N, D) features matrix; flatten any
            # leading batch dims, compute the per-row variance, then restore.
            flat_features = features.reshape(-1, self.num_features)
            # `__post_init__` guarantees `self.covariance is not None`.
            assert self.covariance is not None
            var = self.covariance.variance_at(flat_features).reshape(
                features.shape[:-1]
            )
            return mean, var
        return mean

    def update_precision(
        self, features: Float[Array, "*batch D"]
    ) -> RandomFeatureGaussianProcess:
        """Return a new layer with an EMA-updated Laplace precision.

        Pure-functional: ``self`` is unchanged. Pass features computed
        on the current minibatch (e.g. via :meth:`feature_map`) — the
        update folds the empirical second moment into the EMA. Call
        this once per training batch *after* the gradient step.
        """
        flat = features.reshape(-1, self.num_features)
        # `__post_init__` guarantees `self.covariance is not None`.
        assert self.covariance is not None
        new_cov = self.covariance.update(flat)
        return eqx.tree_at(lambda layer: layer.covariance, self, new_cov)
