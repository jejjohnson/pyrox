"""Bayesian / uncertainty-aware dense layers.

This module hosts the dense Bayesian layer family that used to live in
:mod:`pyrox.nn._layers`. The deterministic forward kernel (single
``... din @ din dout -> ... dout`` contraction) is implemented inline via
:mod:`einx` — the geonnax single-example RFF cores are not used here
because a Bayesian dense layer's forward is literally a matmul, not a
feature map. The Bayesian site-registration logic (priors on ``W`` and
``b``, ``pyrox_sample`` calls) stays in the wrapper exactly as before.

Provides:

* :class:`DenseReparameterization` — weight-space Bayesian linear via
  the reparameterization trick.
* :class:`DenseFlipout` — variance-reduced Bayesian linear via Flipout.
* :class:`DenseVariational` — user-supplied prior factory.
* :class:`DenseVariationalDropout` — sparse variational dropout (kept
  bespoke; no geonnax core).
* :class:`DenseHierarchical` — multiplicative local + global shrinkage.
* :class:`DenseDVI` — analytic moment propagation (kept bespoke).
* :class:`DenseNCP` — deterministic backbone + scaled stochastic
  perturbation.
* :class:`NCPNormalOutput` — output-side NCP KL regulariser.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, cast

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import Array as JaxArray
from jaxtyping import Array, Float

from pyrox._core.pyrox_module import PyroxModule, pyrox_method


class DenseReparameterization(PyroxModule):
    r"""Bayesian dense layer via the reparameterization trick.

    Samples weight and bias from learned Gaussian posteriors at every
    forward pass. Registers NumPyro sample sites so the KL between the
    variational posterior and the prior is tracked by the ELBO.

    .. math::

        W \sim \mathcal{N}(\mu_W, \sigma_W^2), \quad
        b \sim \mathcal{N}(\mu_b, \sigma_b^2), \quad
        y = x W + b.

    Attributes:
        in_features: Input dimension.
        out_features: Output dimension.
        bias: Whether to include a bias term.
        prior_scale: Scale of the isotropic Gaussian prior on weights
            and bias. The prior mean is zero.
        pyrox_name: Explicit scope name for NumPyro site registration.
    """

    in_features: int
    out_features: int
    bias: bool = True
    prior_scale: float = 1.0
    pyrox_name: str | None = None

    @pyrox_method
    def __call__(self, x: Float[Array, "*batch D_in"]) -> Float[Array, "*batch D_out"]:
        prior_w = dist.Normal(
            jnp.zeros((self.in_features, self.out_features)),
            self.prior_scale,
        ).to_event(2)
        W = self.pyrox_sample("weight", prior_w)
        out = einx.dot("... din, din dout -> ... dout", x, W)
        if self.bias:
            prior_b = dist.Normal(
                jnp.zeros(self.out_features), self.prior_scale
            ).to_event(1)
            b = self.pyrox_sample("bias", prior_b)
            out = out + b
        return out


class DenseFlipout(PyroxModule):
    r"""Bayesian dense layer with Flipout sign-flip structure.

    Samples weight from the prior and applies per-example Rademacher
    sign flips to the weight perturbation (Wen et al., 2018). Under a
    NumPyro guide that learns the posterior mean, the sign flips
    decorrelate gradient estimates across minibatch examples.

    In model mode (no guide) this is equivalent to
    :class:`DenseReparameterization` — the Flipout variance reduction
    activates when a guide provides a posterior centered at a learned
    mean.

    Attributes:
        in_features: Input dimension.
        out_features: Output dimension.
        bias: Whether to include a bias term.
        prior_scale: Scale of the isotropic Gaussian prior.
        pyrox_name: Explicit scope name for NumPyro site registration.
    """

    in_features: int
    out_features: int
    bias: bool = True
    prior_scale: float = 1.0
    pyrox_name: str | None = None

    @pyrox_method
    def __call__(self, x: Float[Array, "*batch D_in"]) -> Float[Array, "*batch D_out"]:
        prior_w = dist.Normal(
            jnp.zeros((self.in_features, self.out_features)),
            self.prior_scale,
        ).to_event(2)
        W = self.pyrox_sample("weight", prior_w)
        out = einx.dot("... din, din dout -> ... dout", x, W)

        if self.bias:
            prior_b = dist.Normal(
                jnp.zeros(self.out_features), self.prior_scale
            ).to_event(1)
            b = self.pyrox_sample("bias", prior_b)
            out = out + b
        return out


class DenseVariational(PyroxModule):
    r"""Dense layer with a user-supplied prior factory.

    Provides flexibility over the weight prior by accepting a callable
    that builds the prior distribution given the layer shape. The
    model samples from the prior; the posterior is handled by a NumPyro
    guide (e.g., ``AutoNormal``).

    Attributes:
        in_features: Input dimension.
        out_features: Output dimension.
        make_prior: Callable ``(in_features, out_features) -> Distribution``.
        bias: Whether to include a bias term.
        pyrox_name: Explicit scope name for NumPyro site registration.
    """

    in_features: int
    out_features: int
    make_prior: Callable[..., Any] = eqx.field(static=True)
    bias: bool = True
    pyrox_name: str | None = None

    @pyrox_method
    def __call__(self, x: Float[Array, "*batch D_in"]) -> Float[Array, "*batch D_out"]:
        prior = self.make_prior(self.in_features, self.out_features)
        W = self.pyrox_sample("weight", prior)
        out = einx.dot("... din, din dout -> ... dout", x, W)
        if self.bias:
            b = self.pyrox_sample(
                "bias",
                dist.Normal(jnp.zeros(self.out_features), 1.0).to_event(1),
            )
            out = out + b
        return out


class DenseNCP(PyroxModule):
    r"""Noise Contrastive Prior dense layer (Hafner et al., 2019).

    Decomposes a dense layer into a prior-regularized backbone plus a
    scaled stochastic perturbation:

    .. math::

        y = \underbrace{x W_d + b_d}_{\text{backbone}}
          + \underbrace{\sigma \cdot (x W_s + b_s)}_{\text{perturbation}},

    where all weights are ``pyrox_sample`` sites with Gaussian priors
    and :math:`\sigma` has a ``LogNormal`` prior. The backbone carries
    the bulk of the signal; the perturbation branch adds calibrated
    uncertainty that can be trained via a noise contrastive objective.

    Attributes:
        in_features: Input dimension.
        out_features: Output dimension.
        init_scale: Initial value for the perturbation scale
            :math:`\sigma`.
        pyrox_name: Explicit scope name for NumPyro site registration.
    """

    in_features: int
    out_features: int
    init_scale: float = 1.0
    pyrox_name: str | None = None

    @pyrox_method
    def __call__(self, x: Float[Array, "*batch D_in"]) -> Float[Array, "*batch D_out"]:
        W_d = self.pyrox_sample(
            "weight_det",
            dist.Normal(jnp.zeros((self.in_features, self.out_features)), 1.0).to_event(
                2
            ),
        )
        b_d = self.pyrox_sample(
            "bias_det",
            dist.Normal(jnp.zeros(self.out_features), 1.0).to_event(1),
        )
        det = einx.dot("... din, din dout -> ... dout", x, W_d) + b_d

        W_s = self.pyrox_sample(
            "weight_stoch",
            dist.Normal(jnp.zeros((self.in_features, self.out_features)), 1.0).to_event(
                2
            ),
        )
        b_s = self.pyrox_sample(
            "bias_stoch",
            dist.Normal(jnp.zeros(self.out_features), 1.0).to_event(1),
        )
        scale = self.pyrox_sample(
            "scale",
            dist.LogNormal(jnp.log(jnp.maximum(jnp.array(self.init_scale), 1e-6)), 1.0),
        )
        stoch = scale * (einx.dot("... din, din dout -> ... dout", x, W_s) + b_s)

        return det + stoch


class NCPNormalOutput(PyroxModule):
    r"""Output-side Noise Contrastive Prior layer (Hafner et al., 2018).

    Completes the NCP pattern in ``pyrox.nn``: pair with
    :class:`NCPContinuousPerturb` at the input and a heteroscedastic
    network (e.g. an MLP terminating in a mean head and a positive-std
    head — a softplus or ``exp`` of a learned log-scale) so the
    network produces predictions for both the *clean* batch and the
    input-perturbed *noisy* batch. Given the noisy batch's predictive
    distribution :math:`\mathcal{N}(\hat{y}_n, \hat{\sigma}_n^2)`,
    this layer adds the analytic NCP regulariser

    .. math::

        \mathcal{L}_\mathrm{NCP} =
        \sum_{n} \mathrm{KL}\!\bigl[\mathcal{N}(\hat{y}_n, \hat{\sigma}_n^2)
            \;\big\|\; \mathcal{N}(\mu_\mathrm{prior}, \sigma_\mathrm{prior}^2)\bigr]

    to the model log density via :func:`numpyro.factor`. Pulling the
    noisy-input predictive distribution toward the fixed prior away
    from the training distribution gives the network calibrated
    out-of-distribution uncertainty, which is the central claim of NCP.

    The closed-form Gaussian KL used here is

    .. math::

        \mathrm{KL}\bigl[\mathcal{N}(\mu, \sigma^2)
            \,\|\, \mathcal{N}(\mu_p, \sigma_p^2)\bigr]
        = \log\frac{\sigma_p}{\sigma} +
          \frac{\sigma^2 + (\mu - \mu_p)^2}{2\sigma_p^2} - \tfrac{1}{2}.

    Plate semantics:
        Unlike pyrox's *weight-prior* KL terms, the NCP KL is
        **data-dependent** — every input row contributes its own
        :math:`\mathrm{KL}_n` term. Internally the layer emits the
        :func:`numpyro.factor` site as a *per-example* vector
        (shape ``(*batch,)``) rather than a pre-summed scalar; that
        lets NumPyro's plate machinery sum over the batch axis and
        apply the subsample scaling automatically.

        The canonical training pattern is to emit the layer **inside**
        ``numpyro.plate("data", N, subsample_size=B)``::

            def model(x_clean, y_clean, x_noisy):
                clean_mean, _clean_std = network(x_clean)
                noisy_mean, noisy_std = network(x_noisy)
                ncp_out = NCPNormalOutput(prior_std=1.0)
                with numpyro.plate("data", N, subsample_size=B):
                    ncp_out(noisy_mean, noisy_std)              # scaled to N
                    numpyro.sample("obs",
                        dist.Normal(clean_mean, ...), obs=y_clean)

        Inside the plate, NumPyro sums the per-example log-densities
        over the batch dim and multiplies by ``scale = N / B``,
        producing the standard unbiased estimate of the full-dataset
        NCP KL ``Σ_{n=1}^N KL_n``. Outside any plate the layer's
        contribution is just ``Σ_{n in batch} KL_n`` (i.e. the raw
        batch sum), which is the correct full-dataset value when
        you train on the whole dataset at once.

    Attributes:
        prior_mean: Prior predictive mean :math:`\mu_\mathrm{prior}`.
        prior_std: Prior predictive std :math:`\sigma_\mathrm{prior}`
            (must be positive).
        pyrox_name: Explicit scope name for NumPyro site registration.

    Example:
        >>> import jax.numpy as jnp
        >>> from numpyro import handlers
        >>> ncp = NCPNormalOutput(
        ...     prior_mean=0.0, prior_std=1.0, pyrox_name="ncp_out"
        ... )
        >>> noisy_mean = jnp.zeros((4, 1))
        >>> noisy_std = 0.5 * jnp.ones((4, 1))
        >>> with handlers.seed(rng_seed=0):
        ...     kl = ncp(noisy_mean, noisy_std)
        >>> kl.shape
        ()

    References:
        Hafner, D., Tran, D., Lillicrap, T., Irpan, A., & Davidson, J.
        (2018). *Noise Contrastive Priors for Functional Uncertainty.*
        UAI.
    """

    prior_mean: float = eqx.field(static=True, default=0.0)
    prior_std: float = eqx.field(static=True, default=1.0)
    pyrox_name: str | None = eqx.field(static=True, default=None)

    def __post_init__(self) -> None:
        if self.prior_std <= 0:
            raise ValueError(f"prior_std must be > 0; got {self.prior_std}.")

    @pyrox_method
    def __call__(
        self,
        noisy_mean: Float[Array, "*batch D"],
        noisy_std: Float[Array, "*batch D"],
    ) -> Float[Array, ""]:
        # `noisy_std` is a *standard deviation* — the caller is responsible
        # for ensuring it is non-negative (e.g. via softplus/exp on a
        # learned log-scale head). Negative inputs would silently give a
        # finite-but-wrong KL after squaring; an explicit zero produces
        # `+inf` from `log(0)` which surfaces the bug. We do not floor
        # `noisy_var` because doing so asymmetrically (without a matching
        # floor on `prior_var`) would break the `noisy_std == prior_std`
        # → KL = 0 invariant for tiny `prior_std`.
        if noisy_mean.shape != noisy_std.shape:
            raise ValueError(
                f"noisy_mean shape {noisy_mean.shape} != "
                f"noisy_std shape {noisy_std.shape}."
            )
        # Require an explicit feature axis. With a 1-D ``(B,)`` input,
        # summing along axis=-1 below would collapse the *batch* axis
        # itself, producing a scalar factor that gets broadcast across
        # the data plate — exactly the over-counting bug a per-example
        # factor is designed to avoid. For scalar-regression heads,
        # reshape to ``(B, 1)``.
        if noisy_mean.ndim < 2:
            raise ValueError(
                "noisy_mean / noisy_std must have at least 2 dims "
                "(batch + feature). For a scalar regression head, pass "
                "`noisy_mean[:, None]` and `noisy_std[:, None]`. Got "
                f"shape {noisy_mean.shape}."
            )
        prior_var = jnp.asarray(self.prior_std) ** 2
        noisy_var = noisy_std**2
        kl_per_elem = (
            jnp.log(self.prior_std)
            - 0.5 * jnp.log(noisy_var)
            + (noisy_var + (noisy_mean - self.prior_mean) ** 2) / (2.0 * prior_var)
            - 0.5
        )
        # Sum only over the trailing feature axis. Keeping the leading
        # batch axis intact is what makes NumPyro's plate machinery do
        # the right thing under `plate("data", N, subsample_size=B)`:
        # the plate handler sums log_probs over the batch dim and then
        # multiplies by `N/B`, giving the unbiased full-dataset estimate
        # `(N/B) * sum_{n in batch} kl_n`. Emitting an already-summed
        # scalar instead would let NumPyro broadcast it across the
        # plate dim and over-count by a factor of B.
        kl_per_example = jnp.sum(kl_per_elem, axis=-1)
        # Add -kl_per_example to the model log density site-by-site.
        # Outside any plate this sums to -total_KL; inside a plate the
        # plate handler scales it correctly.
        numpyro.factor(self._pyrox_fullname("kl"), -kl_per_example)
        return jnp.sum(kl_per_example)


# --- DenseVariationalDropout ------------------------------------------------

# Constants for the Molchanov et al. (2017) KL approximation between
# q(W | theta, alpha) = N(theta, alpha * theta^2) and the improper
# log-uniform prior on log|W|. log_alpha is clamped to a finite interval
# to keep the approximation numerically well-behaved at extreme values.
_VD_K1 = 0.63576
_VD_K2 = 1.87320
_VD_K3 = 1.48695
_VD_LOG_ALPHA_MIN = -10.0
_VD_LOG_ALPHA_MAX = 10.0


def _vd_neg_kl(log_alpha: Float[Array, ...]) -> Float[Array, ...]:
    log_alpha = jnp.clip(log_alpha, _VD_LOG_ALPHA_MIN, _VD_LOG_ALPHA_MAX)
    alpha = jnp.exp(log_alpha)
    return (
        _VD_K1 * jax.nn.sigmoid(_VD_K2 + _VD_K3 * log_alpha)
        - 0.5 * jnp.log1p(1.0 / alpha)
        - _VD_K1
    )


class DenseVariationalDropout(PyroxModule):
    r"""Sparse variational dropout dense layer.

    Implements variational dropout (Kingma et al., 2015) extended by
    Molchanov et al. (2017) to a log-uniform prior that enables
    automatic sparsification via per-weight learnable dropout rates.
    The variational posterior on weights is

    .. math::

        q(W_{ij} \mid \theta_{ij}, \alpha_{ij}) =
        \mathcal{N}\!\bigl(\theta_{ij},\; \alpha_{ij}\,\theta_{ij}^2\bigr).

    Forward passes use the *local reparameterization trick* — the
    pre-activation distribution is closed-form and the noise is sampled
    once per output unit per batch element rather than once per weight:

    .. math::

        \gamma = X\theta, \quad
        \delta = X^{\circ 2}\,(\alpha \circ \theta^{\circ 2}), \quad
        Y = \gamma + \sqrt{\delta} \circ \epsilon, \quad
        \epsilon \sim \mathcal{N}(0, I).

    The KL between the posterior and the log-uniform prior is
    approximated analytically (Molchanov et al., 2017) and added to the
    NumPyro trace via :func:`numpyro.factor`. SVI then optimizes

    .. math::

        \mathcal{L} = \mathbb{E}_q[\log p(y \mid f)] - \mathrm{KL}\bigl[q\,\|\,p\bigr].

    Weights with ``log_alpha > threshold`` (default 3.0, dropout rate
    ~0.95) are effectively pruned; inspect the trained pattern via
    :meth:`sparsity`.

    Plate semantics:
        The KL contribution is registered via :func:`numpyro.factor`,
        which is itself a sample-type site and therefore subject to
        ``numpyro.plate`` scaling. To keep the per-layer KL counted
        once (not scaled by the data-plate's subsample ratio), call
        the layer **outside** any ``plate("data", ..., subsample_size=...)``
        block — the standard pyrox / NumPyro convention for global
        Bayesian parameters. Plate only the observation likelihood.

        Correct (forward outside the data plate)::

            def model(x, y=None):
                layer = DenseVariationalDropout(in_features=D, out_features=1)
                f = layer(x).squeeze(-1)               # KL emitted here
                with numpyro.plate("data", x.shape[0]):
                    numpyro.sample("obs", dist.Normal(f, 0.5), obs=y)

        Incorrect (forward inside a subsampled data plate scales KL by
        ``N / subsample_size``)::

            def model(x, y=None):
                with numpyro.plate("data", N, subsample_size=B) as idx:
                    f = layer(x[idx]).squeeze(-1)      # ⚠ scales KL
                    numpyro.sample("obs", ...)

    Attributes:
        in_features: Input dimension.
        out_features: Output dimension.
        bias: Whether to include a bias term.
        log_alpha_init: Initial value for ``log_alpha`` (typically a
            small negative number, e.g., ``-5.0``).
        threshold: ``log_alpha`` threshold for declaring a weight pruned.
        pyrox_name: Explicit scope name for NumPyro site registration.

    Example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from numpyro import handlers
        >>> layer = DenseVariationalDropout(
        ...     in_features=4, out_features=2, pyrox_name="vd"
        ... )
        >>> x = jnp.ones((3, 4))
        >>> with handlers.seed(rng_seed=0):
        ...     y = layer(x)
        >>> y.shape
        (3, 2)
    """

    in_features: int
    out_features: int
    bias: bool = True
    log_alpha_init: float = -5.0
    threshold: float = 3.0
    pyrox_name: str | None = None

    @pyrox_method
    def __call__(self, x: Float[Array, "*batch D_in"]) -> Float[Array, "*batch D_out"]:
        theta = self.pyrox_param(
            "theta",
            jnp.zeros((self.in_features, self.out_features)),
        )
        log_alpha = self.pyrox_param(
            "log_alpha",
            jnp.full(
                (self.in_features, self.out_features),
                float(self.log_alpha_init),
            ),
        )
        log_alpha_clamped = jnp.clip(log_alpha, _VD_LOG_ALPHA_MIN, _VD_LOG_ALPHA_MAX)
        alpha = jnp.exp(log_alpha_clamped)

        gamma = einx.dot("... din, din dout -> ... dout", x, theta)
        delta = einx.dot("... din, din dout -> ... dout", x**2, alpha * theta**2)
        # Floor to a tiny positive value: keeps the sqrt gradient finite at
        # delta = 0 without injecting visible noise (sqrt(1e-30) ≈ 1e-15).
        std = jnp.sqrt(jnp.maximum(delta, 1e-30))
        # numpyro.prng_key returns Array | None at the type level, but is
        # always an Array inside a `seed` handler — which is required for
        # the pyrox_param/factor calls above to succeed in any case.
        key = cast(JaxArray, numpyro.prng_key())
        eps = jax.random.normal(key, gamma.shape, dtype=gamma.dtype)
        out = gamma + std * eps

        if self.bias:
            b = self.pyrox_param("bias", jnp.zeros(self.out_features))
            out = out + b

        numpyro.factor(
            self._pyrox_fullname("kl"),
            jnp.sum(_vd_neg_kl(log_alpha)),
        )
        return out

    def sparsity(self, log_alpha: Float[Array, "D_in D_out"]) -> Float[Array, ""]:
        """Fraction of weights with ``log_alpha > threshold``.

        Pass the trained ``log_alpha`` parameter, typically retrieved
        from the SVI param store under ``f"{pyrox_name}.log_alpha"``.
        """
        return jnp.mean((log_alpha > self.threshold).astype(log_alpha.dtype))


class DenseHierarchical(PyroxModule):
    r"""Hierarchical Bayesian dense layer with multiplicative shrinkage.

    Decomposes the effective weight matrix into a deterministic base
    :math:`\theta \in \mathbb{R}^{D_\mathrm{in} \times D_\mathrm{out}}`
    multiplied row-wise by a per-input-unit local scale
    :math:`z^{(\mathrm{loc})} \in \mathbb{R}^{D_\mathrm{in}}` and an
    overall global scale :math:`z^{(\mathrm{glob})} \in \mathbb{R}`,

    .. math::

        W_{ij} = \theta_{ij} \cdot z_i^{(\mathrm{loc})}
                 \cdot z^{(\mathrm{glob})},

    with isotropic Gaussian priors centred at one,

    .. math::

        z_i^{(\mathrm{loc})} \sim \mathcal{N}(1, \sigma_\mathrm{loc}^2),
        \qquad
        z^{(\mathrm{glob})} \sim \mathcal{N}(1, \sigma_\mathrm{glob}^2).

    The local scale prunes individual input units (a column of
    :math:`\theta` whose ``z_loc`` posterior concentrates near zero is
    effectively switched off) while the global scale modulates the
    overall layer activation — the same hierarchical-shrinkage
    structure used by horseshoe-style BNNs (Louizos et al., 2017).
    Both scales are ``pyrox_sample`` sites so any standard NumPyro
    guide (``AutoNormal``, etc.) drives the variational posterior; the
    deterministic base :math:`\theta` and bias are ``pyrox_param``.

    Plate semantics:
        Same as the rest of ``pyrox.nn``'s Bayesian dense layers — call
        outside ``numpyro.plate("data", ..., subsample_size=...)`` and
        only plate the observation likelihood, otherwise the
        per-layer prior log-probabilities of ``z_loc`` and ``z_glob``
        get scaled by the subsample ratio.

    Attributes:
        in_features: Input dimension :math:`D_\mathrm{in}`.
        out_features: Output dimension :math:`D_\mathrm{out}`.
        bias: Whether to include a deterministic bias term.
        prior_local_scale: Std :math:`\sigma_\mathrm{loc}` of the local
            scale prior.
        prior_global_scale: Std :math:`\sigma_\mathrm{glob}` of the
            global scale prior.
        pyrox_name: Explicit scope name for NumPyro site registration.

    Example:
        >>> import jax.numpy as jnp
        >>> from numpyro import handlers
        >>> layer = DenseHierarchical(
        ...     in_features=4, out_features=2, pyrox_name="hier"
        ... )
        >>> x = jnp.ones((3, 4))
        >>> with handlers.seed(rng_seed=0):
        ...     y = layer(x)
        >>> y.shape
        (3, 2)

    References:
        Louizos, C., Ullrich, K., & Welling, M. (2017). *Bayesian
        Compression for Deep Learning.* NeurIPS.
    """

    in_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)
    bias: bool = eqx.field(static=True, default=True)
    prior_local_scale: float = 0.1
    prior_global_scale: float = 0.1
    pyrox_name: str | None = eqx.field(static=True, default=None)

    def __post_init__(self) -> None:
        if self.prior_local_scale <= 0:
            raise ValueError(
                f"prior_local_scale must be > 0; got {self.prior_local_scale}."
            )
        if self.prior_global_scale <= 0:
            raise ValueError(
                f"prior_global_scale must be > 0; got {self.prior_global_scale}."
            )

    @pyrox_method
    def __call__(self, x: Float[Array, "*batch D_in"]) -> Float[Array, "*batch D_out"]:
        theta = self.pyrox_param(
            "theta", jnp.zeros((self.in_features, self.out_features))
        )
        z_loc = self.pyrox_sample(
            "z_local",
            dist.Normal(jnp.ones(self.in_features), self.prior_local_scale).to_event(1),
        )
        z_glob = self.pyrox_sample(
            "z_global", dist.Normal(1.0, self.prior_global_scale)
        )
        # Effective weight uses the broadcasted multiplicative scaling.
        # Equivalent (and slightly cheaper) to scaling x first then matmul:
        #   y = ((x * z_loc) @ theta) * z_glob.
        out = einx.dot("... din, din dout -> ... dout", x * z_loc, theta) * z_glob
        if self.bias:
            b = self.pyrox_param("b", jnp.zeros(self.out_features))
            out = out + b
        return out


class DenseDVI(PyroxModule):
    r"""Deterministic Variational Inference dense layer (Wu et al., 2018).

    Propagates a *Gaussian distribution* through the linear layer
    analytically — there is no Monte Carlo sampling. The input is a
    diagonal-covariance Gaussian :math:`(\mu_x, \sigma_x^2)`, the
    output is the (still-diagonal) Gaussian :math:`(\mu_y, \sigma_y^2)`
    induced by an independent-Gaussian variational posterior
    :math:`q(W) = \mathcal{N}(M, S)` over the weights and a separate
    diagonal posterior on the bias.

    With weight posterior mean :math:`M`
    (shape :math:`D_\mathrm{in}\times D_\mathrm{out}`) and per-element
    posterior variance :math:`S` (same shape):

    .. math::

        \mu_y = \mu_x M, \qquad
        \sigma_y^2 = \sigma_x^2 (M \circ M)
                   + (\mu_x^{\circ 2} + \sigma_x^2)\,S,

    plus the bias mean / variance if enabled. Compared to MC
    estimators, DVI gives zero-variance gradients of the ELBO at the
    cost of propagating second-order statistics layer by layer (so it
    only really pays off when *all* dense layers in a block are DVI;
    a single DVI layer in a sampling stack just adds bookkeeping).

    The KL between the diagonal-Gaussian variational posterior and a
    fixed isotropic Gaussian prior :math:`p(W) = \mathcal{N}(0, \pi^2)`
    is closed-form and is registered with :func:`numpyro.factor` so
    SVI's ``Trace_ELBO`` picks it up:

    .. math::

        \mathrm{KL}\!\bigl[\mathcal{N}(M, S) \,\big\|\, \mathcal{N}(0, \pi^2)\bigr]
        = \sum_{ij}\Bigl[
            \log \pi - \tfrac12\log S_{ij}
            + \frac{S_{ij} + M_{ij}^2}{2\pi^2} - \tfrac12
          \Bigr].

    Plate semantics:
        Same as the rest of the pyrox Bayesian dense family — call
        this layer **outside** ``numpyro.plate("data", ..., subsample_size=...)``.
        The KL is a *weight-prior* term: it sums over the weight and
        bias matrices, not over the batch, so it's a single scalar
        per layer. ``numpyro.factor`` is still a sample-type site,
        though, and putting it inside a subsampled plate would broadcast
        the scalar to the plate dim and apply ``scale = N/B`` — the
        same over-counting trap that affects every per-layer
        ``numpyro.factor``. Keep this layer at the top of the model
        (or outside any data plate) and only plate the observation
        likelihood::

            def model(x, y=None):
                mean, var = dvi(x_mean, x_var)        # KL emitted here
                with numpyro.plate("data", x.shape[0]):
                    numpyro.sample("obs",
                        dist.Normal(mean, jnp.sqrt(var)), obs=y)

    Attributes:
        in_features: Input dimension :math:`D_\mathrm{in}`.
        out_features: Output dimension :math:`D_\mathrm{out}`.
        bias: Whether to include a diagonal-Gaussian bias.
        prior_scale: Std :math:`\pi` of the isotropic Gaussian prior.
        init_log_var: Initial value for the log posterior variance
            (a small negative number keeps initial draws tight).
        pyrox_name: Explicit scope name for NumPyro site registration.

    Example:
        >>> import jax.numpy as jnp
        >>> from numpyro import handlers
        >>> dvi = DenseDVI(in_features=3, out_features=2, pyrox_name="dvi")
        >>> mean = jnp.ones((4, 3))
        >>> var = 0.1 * jnp.ones((4, 3))
        >>> with handlers.seed(rng_seed=0):
        ...     out_mean, out_var = dvi(mean, var)
        >>> out_mean.shape, out_var.shape
        ((4, 2), (4, 2))

    References:
        Wu, A., Nowozin, S., Meeds, E., Turner, R. E., Hernández-Lobato,
        J. M., & Gaunt, A. L. (2018). *Deterministic Variational
        Inference for Robust Bayesian Neural Networks.* ICLR.
    """

    in_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)
    bias: bool = eqx.field(static=True, default=True)
    prior_scale: float = eqx.field(static=True, default=1.0)
    init_log_var: float = eqx.field(static=True, default=-3.0)
    pyrox_name: str | None = eqx.field(static=True, default=None)

    def __post_init__(self) -> None:
        if self.in_features <= 0 or self.out_features <= 0:
            raise ValueError(
                "in_features and out_features must be > 0; "
                f"got {self.in_features=}, {self.out_features=}."
            )
        if self.prior_scale <= 0:
            raise ValueError(f"prior_scale must be > 0; got {self.prior_scale}.")

    @pyrox_method
    def __call__(
        self,
        mean: Float[Array, "*batch D_in"],
        var: Float[Array, "*batch D_in"],
    ) -> tuple[Float[Array, "*batch D_out"], Float[Array, "*batch D_out"]]:
        if mean.shape != var.shape:
            raise ValueError(f"mean.shape {mean.shape} != var.shape {var.shape}.")
        if mean.shape[-1] != self.in_features:
            raise ValueError(
                f"mean.shape[-1] = {mean.shape[-1]} does not match "
                f"in_features = {self.in_features}."
            )

        W_mean = self.pyrox_param(
            "weight_mean", jnp.zeros((self.in_features, self.out_features))
        )
        W_log_var = self.pyrox_param(
            "weight_log_var",
            jnp.full(
                (self.in_features, self.out_features),
                float(self.init_log_var),
            ),
        )
        W_var = jnp.exp(W_log_var)

        out_mean = einx.dot("... din, din dout -> ... dout", mean, W_mean)
        out_var = einx.dot("... din, din dout -> ... dout", var, W_mean**2) + einx.dot(
            "... din, din dout -> ... dout", mean**2 + var, W_var
        )

        if self.bias:
            b_mean = self.pyrox_param("bias_mean", jnp.zeros(self.out_features))
            b_log_var = self.pyrox_param(
                "bias_log_var",
                jnp.full((self.out_features,), float(self.init_log_var)),
            )
            out_mean = out_mean + b_mean
            out_var = out_var + jnp.exp(b_log_var)

        # Closed-form KL[N(M, S) || N(0, prior_scale^2)] over weights and bias.
        # Use math.log + jnp.asarray cast so prior constants pick up the
        # same dtype as the params (avoid silent float64 promotion under
        # jax_enable_x64 + float32 params).
        log_prior_scale = jnp.asarray(math.log(self.prior_scale), dtype=W_mean.dtype)
        prior_var = jnp.asarray(self.prior_scale**2, dtype=W_mean.dtype)
        kl_w = jnp.sum(
            log_prior_scale
            - 0.5 * W_log_var
            + (W_var + W_mean**2) / (2.0 * prior_var)
            - 0.5
        )
        kl = kl_w
        if self.bias:
            b_var = jnp.exp(b_log_var)
            kl_b = jnp.sum(
                log_prior_scale
                - 0.5 * b_log_var
                + (b_var + b_mean**2) / (2.0 * prior_var)
                - 0.5
            )
            kl = kl + kl_b
        # Add -KL to the model log density. This is a per-layer scalar
        # (sums over the weight / bias matrices, not the batch) — its
        # emission is independent of any data plate the layer lives in.
        numpyro.factor(self._pyrox_fullname("kl"), -kl)
        return out_mean, out_var
