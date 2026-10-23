"""Heteroscedastic output layers (Collier et al., 2021).

* `MCSoftmaxDenseFA` — multi-class output with input-dependent
  low-rank-plus-diagonal logit noise, MC-averaged softmax probabilities.
* `MCSigmoidDenseFA` — same noise model, sigmoid output for
  multi-label / binary classification.
* `GaussianRegressionHead` — Bayesian regression head returning a mean
  and an input-dependent log standard deviation.

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
import jax.random as jr
import numpyro
import numpyro.distributions as dist
from jax import Array as JaxArray
from jaxtyping import Array, Float, PRNGKeyArray
from pyrox._core.pyrox_module import PyroxModule, pyrox_method

from pyrox_nn._batching import vmap_over_flat_batch


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


class GaussianRegressionHead(PyroxModule):
    r"""Bayesian heteroscedastic regression head (mean plus log-scale).

    Wraps `geonnax.GaussianRegressionHead` and samples its four weight
    arrays. For a feature vector $h \in \mathbb{R}^{D_\mathrm{in}}$,

    $$
    \mu = W_\mu^\top h + b_\mu, \qquad
    \log\sigma = \operatorname{clip}\bigl(W_s^\top h + b_s,\ -7,\ 3\bigr),
    \qquad y \sim \mathcal{N}\bigl(\mu, \sigma^2\bigr),
    $$

    with independent Gaussian priors centred on the homoscedastic head,

    $$
    W_\mu, W_s \sim \mathcal{N}(0, s^2), \qquad
    b_\mu \sim \mathcal{N}(0, s^2), \qquad
    b_s \sim \mathcal{N}(\log\sigma_\mathrm{init}, s^2),
    $$

    where $s$ is ``prior_std``. The wrapped core holds these prior
    locations (zero weights, $b_s = \log\sigma_\mathrm{init}$), so
    ``init_to_median`` (or substituting the core's arrays) starts inference
    from a head that predicts $\mu = 0$ and $\sigma = \sigma_\mathrm{init}$
    everywhere; the log-scale weights then move only as the data ask.

    The layer returns $(\mu, \log\sigma)$; the caller writes the
    likelihood. With a known per-observation error $\sigma_i$, use
    $\sigma_i^2 + \sigma(h)^2$ as the variance.

    Sites (scope ``pyrox_name``, default ``"GaussianRegressionHead"``):
    ``<scope>.W_loc`` $(D_\mathrm{in}, K)$, ``<scope>.b_loc`` $(K,)$,
    ``<scope>.W_log_scale`` $(D_\mathrm{in}, K)$ and
    ``<scope>.b_log_scale`` $(K,)$, each sampled once per model call.

    Plate semantics:
        Call outside ``numpyro.plate("data", ..., subsample_size=...)`` so
        the weight sites are not scaled by the subsample factor; put the
        observation site inside the plate.

    Attributes:
        core: The deterministic geonnax head holding the prior locations.
        prior_std: Prior standard deviation $s$ of every weight.
        pyrox_name: Explicit scope name for NumPyro site registration.

    Examples:
        >>> import jax.numpy as jnp
        >>> from numpyro import handlers
        >>> head = GaussianRegressionHead.init(in_features=8, out_features=2)
        >>> with handlers.seed(rng_seed=0):
        ...     mu, log_sigma = head(jnp.ones((5, 8)))
        >>> mu.shape, log_sigma.shape
        ((5, 2), (5, 2))

    References:
        Nix, D. A., & Weigend, A. S. (1994). *Estimating the mean and
        variance of the target probability distribution.* ICNN.
    """

    core: geonnax.GaussianRegressionHead
    prior_std: float = eqx.field(static=True, default=1.0)
    pyrox_name: str | None = eqx.field(static=True, default=None)

    @classmethod
    def init(
        cls,
        in_features: int,
        out_features: int = 1,
        *,
        prior_std: float = 1.0,
        sigma_init: float = 1.0,
        pyrox_name: str | None = None,
    ) -> Self:
        r"""Build the head with priors centred on a homoscedastic core.

        Args:
            in_features: Feature dimension $D_\mathrm{in}$; must be > 0.
            out_features: Number of targets $K$; must be > 0.
            prior_std: Prior standard deviation of every weight; must be > 0.
            sigma_init: Prior location of $\sigma$, i.e.
                $b_s \sim \mathcal{N}(\log\sigma_\mathrm{init}, s^2)$;
                $\log\sigma_\mathrm{init}$ must lie in $[-7, 3]$.
            pyrox_name: Explicit scope name for the sites.

        Returns:
            The head.

        Raises:
            ValueError: If ``prior_std`` is not positive, or (from the
                geonnax core) a dimension or ``sigma_init`` is invalid.
        """
        if not prior_std > 0:
            raise ValueError(f"prior_std must be > 0; got {prior_std}.")
        # The weights are sampled, so the core only carries the prior
        # locations: scale_init_factor=0 zeroes W_log_scale, and W_loc is
        # zeroed below, which makes the key irrelevant.
        core = geonnax.GaussianRegressionHead.init(
            in_features,
            out_features,
            key=jr.PRNGKey(0),
            sigma_init=sigma_init,
            scale_init_factor=0.0,
        )
        core = eqx.tree_at(lambda c: c.W_loc, core, jnp.zeros_like(core.W_loc))
        return cls(core=core, prior_std=prior_std, pyrox_name=pyrox_name)

    @property
    def in_features(self) -> int:
        return self.core.in_features

    @property
    def out_features(self) -> int:
        return self.core.out_features

    def _prior(self, loc: Array) -> dist.Distribution:
        return dist.Normal(loc, self.prior_std).to_event(loc.ndim)

    @pyrox_method
    def __call__(
        self, h: Float[Array, "*batch D_in"]
    ) -> tuple[Float[Array, "*batch K"], Float[Array, "*batch K"]]:
        r"""Sample the weights once and map features to $(\mu, \log\sigma)$.

        Args:
            h: Features of shape ``(*batch, D_in)``; ``(D_in,)`` for one.

        Returns:
            ``(mu, log_sigma)``, each of shape ``(*batch, K)``, with
            ``log_sigma`` clamped to $[-7, 3]$.
        """
        c = self.core
        names = ("W_loc", "b_loc", "W_log_scale", "b_log_scale")
        sampled = tuple(
            self.pyrox_sample(name, self._prior(getattr(c, name))) for name in names
        )
        core = eqx.tree_at(
            lambda m: (m.W_loc, m.b_loc, m.W_log_scale, m.b_log_scale), c, sampled
        )
        return vmap_over_flat_batch(core, h)
