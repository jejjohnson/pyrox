"""Spectral-Normalized Gaussian Process (SNGP) output layer.

* `RandomFeatureGaussianProcess` — SNGP output head (Liu et al.,
  2020). RFF feature map $\\phi(x)$ plus a linear mean head and
  a Laplace covariance over the linear weights.

The Laplace-approximation covariance container
(``LaplaceRandomFeatureCovariance``) lives in ``geonnax`` and is
re-exported from `pyrox.nn._geonnax` for backwards-compatible
imports.

This module implements *just the SNGP head* — spectral normalisation
of upstream dense layers is a separate concern (the design doc's
Tier 2 ``spectral_norm`` gap) and the user is responsible for that.
"""

from __future__ import annotations

import einx
import equinox as eqx
import geonnax
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jaxtyping import Array, Float, PRNGKeyArray

from pyrox._core.pyrox_module import PyroxModule, pyrox_method
from pyrox.nn._batching import vmap_over_flat_batch


class RandomFeatureGaussianProcess(PyroxModule):
    r"""SNGP output layer (Liu et al., 2020).

    A random Fourier feature map followed by a learnable linear head,
    plus a Laplace-approximation covariance over the linear weights.
    The forward pass returns the mean prediction and (optionally) a
    per-input predictive variance summarising distance from the
    training distribution.

    Forward (mean):

    $$
    \phi(x) = \sqrt{\tfrac{2}{D}}\,\cos\!\bigl(W\, x / \ell + b\bigr),
    \qquad \mu(x) = \phi(x)\, H + b_H.
    $$

    The frequencies $W$ and bias $b$ of the RFF map are
    *frozen* (they implicitly define the kernel approximation): they
    are registered as ``pyrox_param`` sites for substitution and
    checkpointing, then guarded with `jax.lax.stop_gradient`
    inside `feature_map` so SGD-style optimisers leave them
    untouched. The lengthscale $\ell$, the linear head
    $H, b_H$, and the Laplace precision are the trainable /
    updated quantities.

    Predictive variance — when $\hat{\Lambda}$ is the current
    precision matrix:

    $$
    \sigma^2(x_*) = \phi(x_*)^\top \hat{\Lambda}^{-1}\, \phi(x_*).
    $$

    Training pattern (one minibatch):

    1. ``mean = layer(x)`` registers / reuses the trainable params and
       returns the mean prediction. Compute the loss, take a gradient
       step on the SVI parameter store as usual.
    2. After the gradient step, call
       ``new_layer = layer.update_precision(features)`` where
       ``features`` is the result of `feature_map` evaluated on
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
        in_features: Input dimension $D_\mathrm{in}$.
        num_features: Number of random Fourier features $D$.
        out_features: Output dimension $D_\mathrm{out}$.
        init_lengthscale: Initial lengthscale $\ell$. Optimised
            during training as a positive ``pyrox_param``.
        W_init: Frozen RFF frequencies, shape ``(D_in, D)``, drawn from
            a standard Normal (the RBF spectral density).
        bias_init: Frozen RFF biases, shape ``(D,)``, drawn from
            ``Uniform(0, 2 pi)``.
        output_linear_init: Init for the linear head, shape ``(D, D_out)``.
        covariance: The `LaplaceRandomFeatureCovariance` instance.
        pyrox_name: Explicit scope name for NumPyro site registration.

    References:
        Liu, J. Z., et al. (2020). *Simple and Principled Uncertainty
        Estimation with Deterministic Deep Learning via Distance
        Awareness.* NeurIPS.
    """

    core: geonnax.RandomFeatureGaussianProcess
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
        # geonnax validates positive dims and init_lengthscale > 0;
        # momentum / ridge constraints are validated inside the LRFC init.
        core = geonnax.RandomFeatureGaussianProcess.init(
            in_features=in_features,
            num_features=num_features,
            out_features=out_features,
            key=key,
            init_lengthscale=init_lengthscale,
            momentum=momentum,
            ridge=ridge,
            head_scale=head_scale,
        )
        return cls(core=core, pyrox_name=pyrox_name)

    # Read-only property accessors retain the pre-refactor attribute names so
    # external callers (tests, user code reading static dims) keep working.
    @property
    def in_features(self) -> int:
        return self.core.in_features

    @property
    def num_features(self) -> int:
        return self.core.num_features

    @property
    def out_features(self) -> int:
        return self.core.out_features

    @property
    def init_lengthscale(self) -> float:
        return float(self.core.lengthscale)

    @property
    def W_init(self) -> Float[Array, "D_in D"]:
        return self.core.W

    @property
    def bias_init(self) -> Float[Array, " D"]:
        return self.core.bias

    @property
    def output_linear_init(self) -> Float[Array, "D D_out"]:
        return self.core.output_linear

    @property
    def covariance(self) -> geonnax.LaplaceRandomFeatureCovariance:
        return self.core.covariance

    def _swap_feature_core(self) -> geonnax.RandomFeatureGaussianProcess:
        """Register the RFF-map params and swap them into the core.

        Only the frequency/bias/lengthscale arrays are needed for the
        feature map; the linear-head params are registered inside
        ``__call__`` so disabled branches (e.g. pure feature-map usage
        via `feature_map`) don't materialise unused sites.
        """
        W = jax.lax.stop_gradient(self.pyrox_param("W", self.core.W))
        b = jax.lax.stop_gradient(self.pyrox_param("bias", self.core.bias))
        ls = self.pyrox_param(
            "lengthscale",
            self.core.lengthscale,
            constraint=dist.constraints.positive,
        )
        return eqx.tree_at(
            lambda c: (c.W, c.bias, c.lengthscale),
            self.core,
            (W, b, ls),
        )

    @pyrox_method
    def feature_map(self, x: Float[Array, "*batch D_in"]) -> Float[Array, "*batch D"]:
        r"""Random Fourier feature map: $\phi(x) = \sqrt{2/D}\,\cos(Wx/\ell + b)$.

        Frequencies and bias are registered as ``pyrox_param`` sites for
        substitution / checkpointing, but `jax.lax.stop_gradient`
        is applied so SVI's gradient-based optimisers leave them
        frozen at their init values. The lengthscale is the active
        bandwidth control and is constrained positive.
        """
        new_core = self._swap_feature_core()
        # geonnax feature_map is single-example `(D_in,) -> (D,)`.
        return vmap_over_flat_batch(new_core.feature_map, x)

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
        # Register the RFF + linear-head params and swap them into the core.
        new_core = self._swap_feature_core()
        H = self.pyrox_param("output_linear", self.core.output_linear)
        b_out = self.pyrox_param(
            "output_bias", jnp.zeros(self.out_features, dtype=x.dtype)
        )
        new_core = eqx.tree_at(
            lambda c: (c.output_linear, c.output_bias),
            new_core,
            (H, b_out),
        )

        # geonnax `__call__` is single-example `(D_in,)`. `return_cov=False`
        # keeps the output a plain array; `return_cov=True` returns
        # (mean, var) per example and the helper restores both leaves.
        if return_cov:
            mean, var = vmap_over_flat_batch(
                lambda xi: new_core(xi, return_cov=True), x
            )
            return mean, var

        return vmap_over_flat_batch(new_core, x)

    def update_precision(
        self, features: Float[Array, "*batch D"]
    ) -> RandomFeatureGaussianProcess:
        """Return a new layer with an EMA-updated Laplace precision.

        Pure-functional: ``self`` is unchanged. Pass features computed
        on the current minibatch (e.g. via `feature_map`) — the
        update folds the empirical second moment into the EMA. Call
        this once per training batch *after* the gradient step.
        """
        # Flatten any leading batch dims down to the single batch axis the
        # geonnax `update_precision` expects.
        flat = einx.id("b... d -> (b...) d", features)
        new_core = self.core.update_precision(flat)
        return eqx.tree_at(lambda layer: layer.core, self, new_core)
