"""Heteroscedastic output layers (Collier et al., 2021).

* `MCSoftmaxDenseFA` — multi-class output with input-dependent
  low-rank-plus-diagonal logit noise, MC-averaged softmax probabilities.
* `MCSigmoidDenseFA` — same noise model, sigmoid output for
  multi-label / binary classification.

Both share the same heteroscedastic logit-noise model — given an
input-dependent low-rank factor $V(x) \\in \\mathbb{R}^{C \\times r}$
and diagonal $\\sigma(x) \\in \\mathbb{R}^C$,

$$
\\eta(x) = W_\\mu x + b_\\mu + \\epsilon, \\qquad
\\Sigma(x) = V(x) V(x)^\\top + \\mathrm{diag}\\!\\bigl(\\sigma^2(x)\\bigr),
\\;\\; \\epsilon \\sim \\mathcal{N}(0, \\Sigma(x)).
$$

Predictions average a small number of Monte Carlo softmax / sigmoid
samples.
"""

from __future__ import annotations

from typing import Self, cast

import equinox as eqx
import geonnax
import jax
import jax.numpy as jnp
import numpyro
from jax import Array as JaxArray
from jaxtyping import Array, Float, PRNGKeyArray
from pyrox._core.pyrox_module import PyroxModule, pyrox_method


class _HeteroscedasticBase(PyroxModule):
    """Shared state and init for the FA-noise heteroscedastic layers."""

    core: geonnax.HeteroscedasticHead
    pyrox_name: str | None = eqx.field(static=True, default=None)

    @classmethod
    def init(
        cls,
        key: PRNGKeyArray,
        in_features: int,
        num_classes: int,
        rank: int,
        *,
        num_mc_samples: int = 10,
        diag_init_bias: float = -3.0,
        scale_init_factor: float = 0.1,
        pyrox_name: str | None = None,
    ) -> Self:
        """Construct the layer with Glorot-init linear factors."""
        # geonnax already validates positive dims and num_mc_samples; let
        # those errors propagate from the core init.
        core_cls = cls._core_cls()
        core = core_cls.init(
            in_features=in_features,
            num_classes=num_classes,
            rank=rank,
            key=key,
            num_mc_samples=num_mc_samples,
            diag_init_bias=diag_init_bias,
            scale_init_factor=scale_init_factor,
        )
        return cls(core=core, pyrox_name=pyrox_name)

    @classmethod
    def _core_cls(cls) -> type[geonnax.HeteroscedasticHead]:
        raise NotImplementedError

    # Convenience read-only attribute access — keeps the pre-refactor
    # call sites (`layer.in_features`, etc.) working without storing the
    # static dims as duplicate fields on the wrapper.
    @property
    def in_features(self) -> int:
        return self.core.in_features

    @property
    def num_classes(self) -> int:
        return self.core.num_classes

    @property
    def rank(self) -> int:
        return self.core.rank

    @property
    def num_mc_samples(self) -> int:
        return self.core.num_mc_samples

    @property
    def diag_init_bias(self) -> float:
        return self.core.diag_init_bias

    def _swap_core(self) -> geonnax.HeteroscedasticHead:
        """Register weight arrays as ``pyrox_param`` sites and swap into the core.

        Each of the six arrays (``W_loc``/``b_loc``/``W_scale``/
        ``b_scale``/``W_diag``/``b_diag``) is registered once per
        ``model()`` call using the core's stored value as the init,
        then folded back into a new core via `equinox.tree_at`.
        """
        W_loc = self.pyrox_param("W_loc", self.core.W_loc)
        b_loc = self.pyrox_param("b_loc", self.core.b_loc)
        W_scale = self.pyrox_param("W_scale", self.core.W_scale)
        b_scale = self.pyrox_param("b_scale", self.core.b_scale)
        W_diag = self.pyrox_param("W_diag", self.core.W_diag)
        b_diag = self.pyrox_param("b_diag", self.core.b_diag)
        return eqx.tree_at(
            lambda c: (c.W_loc, c.b_loc, c.W_scale, c.b_scale, c.W_diag, c.b_diag),
            self.core,
            (W_loc, b_loc, W_scale, b_scale, W_diag, b_diag),
        )


def _batched_logits(
    new_core: geonnax.HeteroscedasticHead, x: Float[Array, "N D_in"]
) -> Float[Array, "S N C"]:
    """Run ``geonnax.hetero_noisy_logits`` over a batch of inputs.

    The geonnax helper is single-example and consumes a PRNG ``key``.
    To keep the wrapper batched-friendly we draw one fresh key per
    batch element from ``numpyro.prng_key()`` and ``vmap`` over the
    input/key pair. ``hetero_noisy_logits`` returns ``(S, C)`` per
    example, so the vmapped output is ``(N, S, C)`` which we move to
    the ``(S, N, C)`` layout that the existing softmax/sigmoid averaging
    expects.
    """
    N = x.shape[0]
    key = cast(JaxArray, numpyro.prng_key())
    keys = jax.random.split(key, N)
    per_example = jax.vmap(
        lambda xi, ki: geonnax.hetero_noisy_logits(new_core, xi, key=ki)
    )(x, keys)
    # (N, S, C) -> (S, N, C)
    return jnp.swapaxes(per_example, 0, 1)


class MCSoftmaxDenseFA(_HeteroscedasticBase):
    r"""Heteroscedastic multi-class output layer (FA noise + softmax).

    Implements Collier et al. (2021): the logit covariance is
    input-dependent low-rank-plus-diagonal,

    $$
    \eta(x) = W_\mu x + b_\mu + \epsilon, \qquad
    \Sigma(x) = V(x) V(x)^\top + \operatorname{diag}\!\bigl(\sigma^2(x)\bigr),
    \;\; \epsilon \sim \mathcal{N}(0, \Sigma(x)),
    $$

    where $V(x) = \mathrm{reshape}(W_V x + b_V, [C, r])$ and
    $\sigma(x) = \exp(W_\sigma x + b_\sigma)$. Output is the
    Monte Carlo average of softmaxed perturbed logits

    $$
    \hat{p}(y = k \mid x) \approx
    \frac{1}{S}\sum_{s=1}^{S}
    \mathrm{softmax}_k\!\bigl(\eta(x) + \epsilon_s\bigr).
    $$

    All linear factors are deterministic ``pyrox_param`` sites — the
    layer is heteroscedastic but not Bayesian over its weights. Use it
    as a drop-in head for classification when label noise is known to
    be input-dependent (label disagreement, fine-grained categories).

    Plate semantics:
        Same as other ``pyrox_nn`` Bayesian dense layers — call
        outside ``numpyro.plate("data", ..., subsample_size=...)`` so
        the parameter sites are unscaled. The MC noise is drawn from
        ``numpyro.prng_key()``.

    Attributes:
        in_features: Input dimension $D_\mathrm{in}$.
        num_classes: Number of classes $C$.
        rank: Rank $r$ of the low-rank factor $V(x)$.
        num_mc_samples: Number of MC softmax samples $S$ per
            forward call.
        diag_init_bias: Initial value for the diagonal-scale bias
            ``b_diag`` (a small negative number keeps initial noise
            small).
        pyrox_name: Explicit scope name for NumPyro site registration.

    Examples:
        >>> import jax.random as jr
        >>> import jax.numpy as jnp
        >>> from numpyro import handlers
        >>> layer = MCSoftmaxDenseFA.init(
        ...     jr.PRNGKey(0), in_features=4, num_classes=3, rank=2,
        ... )
        >>> x = jnp.ones((5, 4))
        >>> with handlers.seed(rng_seed=0):
        ...     probs = layer(x)
        >>> probs.shape
        (5, 3)
        >>> bool(jnp.allclose(probs.sum(axis=-1), 1.0))
        True

    References:
        Collier, M., Mustafa, B., Kokiopoulou, E., Jenatton, R., &
        Berent, J. (2021). *Correlated Input-Dependent Label Noise in
        Large-Scale Image Classification.* CVPR.
    """

    @classmethod
    def _core_cls(cls) -> type[geonnax.HeteroscedasticHead]:
        return geonnax.MCSoftmaxDenseFA

    @pyrox_method
    def __call__(self, x: Float[Array, "N D_in"]) -> Float[Array, "N C"]:
        new_core = self._swap_core()
        logits = _batched_logits(new_core, x)
        return jnp.mean(jax.nn.softmax(logits, axis=-1), axis=0)


class MCSigmoidDenseFA(_HeteroscedasticBase):
    r"""Heteroscedastic multi-label output layer (FA noise + sigmoid).

    Identical low-rank-plus-diagonal logit-noise model as
    `MCSoftmaxDenseFA`, but the per-class outputs are
    independent Bernoullis — final probabilities are the MC average of
    *element-wise* sigmoids, not a softmax. Use this for multi-label
    classification or independent binary heads.

    $$
    \hat{p}(y_k = 1 \mid x) \approx
    \frac{1}{S}\sum_{s=1}^{S}
    \sigma\!\bigl(\eta(x) + \epsilon_s\bigr)_k.
    $$

    See `MCSoftmaxDenseFA` for the noise model, plate semantics,
    init API, and references.
    """

    @classmethod
    def _core_cls(cls) -> type[geonnax.HeteroscedasticHead]:
        return geonnax.MCSigmoidDenseFA

    @pyrox_method
    def __call__(self, x: Float[Array, "N D_in"]) -> Float[Array, "N C"]:
        new_core = self._swap_core()
        logits = _batched_logits(new_core, x)
        return jnp.mean(jax.nn.sigmoid(logits), axis=0)
