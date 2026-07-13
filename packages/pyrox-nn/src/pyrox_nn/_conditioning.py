"""Bayesian conditioning primitives for ``pyrox_nn``.

The deterministic conditioners (``ConcatConditioner``, ``AffineModulation``,
``HyperLinear``) and the ``ConditionedINR`` / ``HyperSIREN`` composites now
live in `geonnax` and are re-exported via `pyrox_nn._geonnax`.
This module keeps the Bayesian variants whose forward passes register
NumPyro sample sites — those cannot live in ``geonnax`` because they
depend on the pyrox ``PyroxModule`` / ``pyrox_sample`` machinery.

Bayesian variants (`BayesianConcatConditioner`,
`BayesianAffineModulation`, `BayesianHyperLinear`) put
Normal priors on the **generator** weights only — never on ``h``, ``z``,
or the inner network — so prior cost scales with the generator size, not
the target size. This is the architectural advantage of doing Bayesian
amortised inference via hypernetworks (NIF, MetaSDF) rather than directly
over the target weights.

Each Bayesian wrapper holds a frozen `geonnax` core, samples the
generator's ``(W, b)`` once per ``model()`` call, swaps the sampled
arrays into the core's ``eqx.nn.Linear`` via `eqx.tree_at`, and
then ``jax.vmap`` s the core forward across the batch axis. The
`HyperFourierFeatures` and `ConditionedRFFNet` follow the
same pattern, using `geonnax.rff_forward` as the per-example
feature kernel.
"""

from __future__ import annotations

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from geonnax import (
    AffineModulation as _GxAffineModulation,
    ConcatConditioner as _GxConcatConditioner,
    HyperLinear as _GxHyperLinear,
)
from geonnax.conditioning import (  # type: ignore[attr-defined]
    _GAMMA_ACTIVATIONS,
    GammaActivation,
)
from jaxtyping import Array, Float
from pyrox._core.pyrox_module import PyroxModule, pyrox_method

from pyrox_nn._batching import vmap_over_flat_batch


# ---------------------------------------------------------------------------
# Bayesian variants — priors live on the generator only
# ---------------------------------------------------------------------------


class BayesianConcatConditioner(PyroxModule):
    """`geonnax.ConcatConditioner` with Normal priors on the projection.

    Registers two NumPyro sample sites — ``{scope}.proj_W`` and
    ``{scope}.proj_b`` — under ``Normal(0, prior_std)``. Total of two
    sites per forward call; nothing is sampled from the inner ``h`` or
    the context ``z``.

    Holds a frozen `geonnax.ConcatConditioner` core whose
    ``proj`` weights are swapped with the sampled arrays each call.

    Attributes:
        core: Frozen `geonnax.ConcatConditioner` carrying the
            single-example forward.
        num_features: Output channels.
        cond_dim: Context dimension.
        prior_std: Scale of the Normal priors.
        pyrox_name: Optional explicit scope name for NumPyro.
    """

    core: _GxConcatConditioner
    num_features: int = eqx.field(static=True)
    cond_dim: int = eqx.field(static=True)
    prior_std: float = eqx.field(static=True, default=1.0)
    pyrox_name: str | None = eqx.field(static=True, default=None)

    @classmethod
    def init(
        cls,
        num_features: int,
        cond_dim: int,
        *,
        prior_std: float = 1.0,
        pyrox_name: str | None = None,
    ) -> BayesianConcatConditioner:
        """Build a `BayesianConcatConditioner`.

        Args:
            num_features: Output channels.
            cond_dim: Context dimension.
            prior_std: Scale of the Normal priors.
            pyrox_name: Optional explicit scope name.
        """
        if num_features <= 0 or cond_dim <= 0:
            raise ValueError(
                "num_features and cond_dim must be positive; got "
                f"num_features={num_features}, cond_dim={cond_dim}."
            )
        if prior_std <= 0:
            raise ValueError(f"prior_std must be > 0; got {prior_std}.")
        # Initialise the core with a dummy key — the proj weights are
        # overwritten by sampled values on every forward.
        core = _GxConcatConditioner.init(
            num_features=num_features,
            cond_dim=cond_dim,
            key=jax.random.key(0),
        )
        return cls(
            core=core,
            num_features=num_features,
            cond_dim=cond_dim,
            prior_std=prior_std,
            pyrox_name=pyrox_name,
        )

    @pyrox_method
    def __call__(
        self,
        h: Float[Array, "*batch C"],
        z: Float[Array, "*batch K"] | Float[Array, " K"],
    ) -> Float[Array, "*batch C"]:
        if h.shape[-1] != self.num_features:
            raise ValueError(
                f"h.shape[-1]={h.shape[-1]} does not match "
                f"num_features={self.num_features}."
            )
        if z.shape[-1] != self.cond_dim:
            raise ValueError(
                f"z.shape[-1]={z.shape[-1]} does not match cond_dim={self.cond_dim}."
            )
        in_dim = self.num_features + self.cond_dim
        # eqx.nn.Linear stores weight as (out, in); match that shape.
        W = self.pyrox_sample(
            "proj_W",
            dist.Normal(0.0, self.prior_std)
            .expand([self.num_features, in_dim])
            .to_event(2),
        )
        b = self.pyrox_sample(
            "proj_b",
            dist.Normal(0.0, self.prior_std).expand([self.num_features]).to_event(1),
        )
        new_proj = eqx.tree_at(lambda m: (m.weight, m.bias), self.core.proj, (W, b))
        new_core = eqx.tree_at(lambda m: m.proj, self.core, new_proj)

        # Broadcast and batch via vmap; geonnax core expects single example.
        squeeze_h = h.ndim == 1
        if squeeze_h:
            h = h[None, :]
        if z.ndim == 1:
            z = jnp.broadcast_to(z, (h.shape[0], z.shape[-1]))
        out = jax.vmap(new_core, in_axes=(0, 0))(h, z)
        return out[0] if squeeze_h else out


class BayesianAffineModulation(PyroxModule):
    """`geonnax.AffineModulation` with Normal priors on the FiLM generator.

    Registers two sites — ``{scope}.gen_W`` and ``{scope}.gen_b`` —
    under ``Normal(0, prior_std)``. The ``γ`` activation is fixed by
    construction (default ``"one_plus_tanh"``) so the prior over the raw
    generator output induces a well-defined prior over ``γ``, ``β``.

    Holds a frozen `geonnax.AffineModulation` core whose
    ``generator`` weights are swapped with the sampled arrays each call.

    Attributes:
        core: Frozen `geonnax.AffineModulation` carrying the
            single-example forward.
        num_features: Output channels.
        cond_dim: Context dimension.
        gamma_activation: Parameterisation of ``γ``.
        prior_std: Scale of the Normal priors.
        pyrox_name: Optional explicit scope name.
    """

    core: _GxAffineModulation
    num_features: int = eqx.field(static=True)
    cond_dim: int = eqx.field(static=True)
    gamma_activation: GammaActivation = eqx.field(static=True, default="one_plus_tanh")
    prior_std: float = eqx.field(static=True, default=1.0)
    pyrox_name: str | None = eqx.field(static=True, default=None)

    @classmethod
    def init(
        cls,
        num_features: int,
        cond_dim: int,
        *,
        gamma_activation: GammaActivation = "one_plus_tanh",
        prior_std: float = 1.0,
        pyrox_name: str | None = None,
    ) -> BayesianAffineModulation:
        """Build a `BayesianAffineModulation`."""
        if num_features <= 0 or cond_dim <= 0:
            raise ValueError(
                "num_features and cond_dim must be positive; got "
                f"num_features={num_features}, cond_dim={cond_dim}."
            )
        if gamma_activation not in _GAMMA_ACTIVATIONS:
            raise ValueError(
                f"gamma_activation must be one of {_GAMMA_ACTIVATIONS}; "
                f"got {gamma_activation!r}."
            )
        if prior_std <= 0:
            raise ValueError(f"prior_std must be > 0; got {prior_std}.")
        core = _GxAffineModulation.init(
            num_features=num_features,
            cond_dim=cond_dim,
            key=jax.random.key(0),
            gamma_activation=gamma_activation,
        )
        return cls(
            core=core,
            num_features=num_features,
            cond_dim=cond_dim,
            gamma_activation=gamma_activation,
            prior_std=prior_std,
            pyrox_name=pyrox_name,
        )

    @pyrox_method
    def __call__(
        self,
        h: Float[Array, "*batch C"],
        z: Float[Array, "*batch K"] | Float[Array, " K"],
    ) -> Float[Array, "*batch C"]:
        if h.shape[-1] != self.num_features:
            raise ValueError(
                f"h.shape[-1]={h.shape[-1]} does not match "
                f"num_features={self.num_features}."
            )
        if z.shape[-1] != self.cond_dim:
            raise ValueError(
                f"z.shape[-1]={z.shape[-1]} does not match cond_dim={self.cond_dim}."
            )
        out_dim = 2 * self.num_features
        # eqx.nn.Linear stores weight as (out, in); match that shape.
        W = self.pyrox_sample(
            "gen_W",
            dist.Normal(0.0, self.prior_std)
            .expand([out_dim, self.cond_dim])
            .to_event(2),
        )
        b = self.pyrox_sample(
            "gen_b",
            dist.Normal(0.0, self.prior_std).expand([out_dim]).to_event(1),
        )
        new_gen = eqx.tree_at(lambda m: (m.weight, m.bias), self.core.generator, (W, b))
        new_core = eqx.tree_at(lambda m: m.generator, self.core, new_gen)

        squeeze_h = h.ndim == 1
        if squeeze_h:
            h = h[None, :]
        if z.ndim == 1:
            z = jnp.broadcast_to(z, (h.shape[0], z.shape[-1]))
        out = jax.vmap(new_core, in_axes=(0, 0))(h, z)
        return out[0] if squeeze_h else out


class BayesianHyperLinear(PyroxModule):
    """`geonnax.HyperLinear` with Normal priors on the generator only.

    Two sites: ``{scope}.gen_W`` and ``{scope}.gen_b``. The target
    weights ``(W_target, b_target)`` are *generated* — not sampled — so
    Bayesian inference cost scales with the generator size
    ``cond_dim * (target_out * target_in + target_out)``, not with the
    target-network size. This is the architectural advantage of doing
    Bayesian amortised inference via hypernetworks.

    Attributes:
        core: Frozen `geonnax.HyperLinear` carrying the
            single-example forward.
        target_in: Inner ``Linear``'s input dim ``C_in``.
        target_out: Inner ``Linear``'s output dim ``C_out``.
        cond_dim: Context dimension ``K``.
        num_features: Alias for ``target_out``.
        prior_std: Scale of the Normal priors on the generator.
        pyrox_name: Optional explicit scope name.
    """

    core: _GxHyperLinear
    num_features: int = eqx.field(static=True)
    cond_dim: int = eqx.field(static=True)
    target_in: int = eqx.field(static=True)
    target_out: int = eqx.field(static=True)
    prior_std: float = eqx.field(static=True, default=1.0)
    pyrox_name: str | None = eqx.field(static=True, default=None)

    @classmethod
    def init(
        cls,
        target_in: int,
        target_out: int,
        cond_dim: int,
        *,
        prior_std: float = 1.0,
        pyrox_name: str | None = None,
    ) -> BayesianHyperLinear:
        """Build a `BayesianHyperLinear`."""
        if target_in <= 0 or target_out <= 0 or cond_dim <= 0:
            raise ValueError(
                "target_in, target_out, and cond_dim must all be positive; got "
                f"target_in={target_in}, target_out={target_out}, "
                f"cond_dim={cond_dim}."
            )
        if prior_std <= 0:
            raise ValueError(f"prior_std must be > 0; got {prior_std}.")
        core = _GxHyperLinear.init(
            target_in=target_in,
            target_out=target_out,
            cond_dim=cond_dim,
            key=jax.random.key(0),
        )
        return cls(
            core=core,
            num_features=target_out,
            cond_dim=cond_dim,
            target_in=target_in,
            target_out=target_out,
            prior_std=prior_std,
            pyrox_name=pyrox_name,
        )

    @pyrox_method
    def __call__(
        self,
        x: Float[Array, "*batch C_in"],
        z: Float[Array, "*batch K"] | Float[Array, " K"],
    ) -> Float[Array, "*batch C_out"]:
        if x.shape[-1] != self.target_in:
            raise ValueError(
                f"x.shape[-1]={x.shape[-1]} does not match target_in={self.target_in}."
            )
        if z.shape[-1] != self.cond_dim:
            raise ValueError(
                f"z.shape[-1]={z.shape[-1]} does not match cond_dim={self.cond_dim}."
            )
        flat_size = self.target_out * self.target_in + self.target_out
        # eqx.nn.Linear stores weight as (out, in); match that shape.
        W = self.pyrox_sample(
            "gen_W",
            dist.Normal(0.0, self.prior_std)
            .expand([flat_size, self.cond_dim])
            .to_event(2),
        )
        b = self.pyrox_sample(
            "gen_b",
            dist.Normal(0.0, self.prior_std).expand([flat_size]).to_event(1),
        )
        new_gen = eqx.tree_at(lambda m: (m.weight, m.bias), self.core.generator, (W, b))
        new_core = eqx.tree_at(lambda m: m.generator, self.core, new_gen)

        squeeze_x = x.ndim == 1
        if squeeze_x:
            x = x[None, :]
        if z.ndim == 1:
            z = jnp.broadcast_to(z, (x.shape[0], z.shape[-1]))
        out = jax.vmap(new_core, in_axes=(0, 0))(x, z)
        return out[0] if squeeze_x else out


# ---------------------------------------------------------------------------
# Hyper-Fourier features: a parameter net produces (W, b, log_lengthscale)
# ---------------------------------------------------------------------------


class HyperFourierFeatures(PyroxModule):
    r"""Random Fourier features with ``(W, b, log_lengthscale)`` from a parameter net.

    The deterministic counterpart `pyrox_nn.RBFFourierFeatures`
    *samples* its frequencies and lengthscale from priors. This layer
    instead amortises them over a context vector ``z`` via a user-supplied
    ``parameter_net``:

    $$
    \begin{aligned}
    (W(z), b(z), \log\ell(z)) &= \text{unflatten}(\text{parameter\_net}(z)) \\
    \phi(x; z) &= \sqrt{1/n_{\text{features}}}\;
        \bigl[\cos(W(z)^\top x / \ell(z) + b(z)),\;
              \sin(W(z)^\top x / \ell(z) + b(z))\bigr]
    \end{aligned}
    $$

    Two execution modes are supported:

    * **Shared mode** (``z.ndim == 1``): the parameter net runs once
      and the generated features are reused across all rows of ``x``
      — same efficiency trick as `HyperLinear`'s shared path.
    * **Per-sample mode** (``z.ndim == 2``): a distinct
      ``(W, b, log_lengthscale)`` is generated per row of ``z`` via
      ``jax.vmap`` and applied with ``einx.dot``. This is
      substantially more expensive in compute and memory because the
      Fourier parameters are no longer shared across rows of ``x``,
      but it is required when each ``x`` row needs its own context.

    The flat output of ``parameter_net(z)`` must have size
    ``in_features * n_features + n_features + 1`` (frequencies, phases,
    log-lengthscale). ``init`` does **not** invoke ``parameter_net`` —
    a misshapen output surfaces only on the first call.

    Attributes:
        parameter_net: Callable ``(K,) -> (P,)`` producing the flat
            feature parameters from the context. Typically a small MLP
            or any `PyroxModule`.
        in_features: Coordinate dimension (``D_in``).
        n_features: Number of frequency pairs; output dim is
            ``2 * n_features``.
        cond_dim: Context dimension expected by ``parameter_net``.
        pyrox_name: Optional explicit scope name.

    Examples:
        >>> import jax.random as jr, jax.numpy as jnp
        >>> import equinox as eqx
        >>> key = jr.key(0)
        >>> # Parameter net: (cond_dim=2,) -> (1*16 + 16 + 1 = 33,)
        >>> pnet = eqx.nn.MLP(in_size=2, out_size=33, width_size=32, depth=2, key=key)
        >>> hff = HyperFourierFeatures.init(
        ...     parameter_net=pnet, in_features=1, n_features=16, cond_dim=2,
        ... )
        >>> y = hff(jnp.ones((5, 1)), jnp.ones((2,)))
        >>> y.shape
        (5, 32)
    """

    parameter_net: PyroxModule | eqx.Module
    in_features: int = eqx.field(static=True)
    n_features: int = eqx.field(static=True)
    cond_dim: int = eqx.field(static=True)
    pyrox_name: str | None = eqx.field(static=True, default=None)

    @classmethod
    def init(
        cls,
        *,
        parameter_net: PyroxModule | eqx.Module,
        in_features: int,
        n_features: int,
        cond_dim: int,
        pyrox_name: str | None = None,
    ) -> HyperFourierFeatures:
        """Build `HyperFourierFeatures`.

        ``parameter_net`` is **not** invoked at construction time, so
        Bayesian / numpyro-aware parameter nets that rely on
        ``pyrox_sample`` work without needing a seed handler at init.
        The expected output size is ``in_features * n_features +
        n_features + 1``; a mismatch surfaces as a shape error on the
        first ``__call__``.
        """
        if in_features <= 0 or n_features <= 0 or cond_dim <= 0:
            raise ValueError(
                "in_features, n_features, and cond_dim must all be positive; got "
                f"in_features={in_features}, n_features={n_features}, "
                f"cond_dim={cond_dim}."
            )
        return cls(
            parameter_net=parameter_net,
            in_features=in_features,
            n_features=n_features,
            cond_dim=cond_dim,
            pyrox_name=pyrox_name,
        )

    def _unpack(self, z: Array) -> tuple[Array, Array, Array]:
        """Split ``parameter_net(z)`` into ``(W, b, log_l)``.

        ``W`` is returned with shape ``(in_features, n_features)`` to
        match the layout expected by `geonnax.rff_forward`.
        """
        flat = self.parameter_net(z)  # ty: ignore[call-non-callable]
        w_size = self.in_features * self.n_features
        flat_W = flat[:w_size]
        b = flat[w_size : w_size + self.n_features]
        log_l = flat[-1]
        W = einx.id("(i j) -> i j", flat_W, i=self.in_features)
        return W, b, log_l

    def _single_example(
        self,
        x: Float[Array, " D_in"],
        W: Float[Array, "D_in n"],
        b: Float[Array, " n"],
        log_l: Float[Array, ""],
    ) -> Float[Array, " D_rff"]:
        """Per-example RFF with phase ``b``; mirrors `geonnax.rff_forward`."""
        proj = einx.dot("d, d n -> n", x, W) * jnp.exp(-log_l) + b  # (n,)
        scale = jnp.sqrt(1.0 / self.n_features)
        return scale * jnp.concatenate([jnp.cos(proj), jnp.sin(proj)], axis=-1)

    @pyrox_method
    def __call__(
        self,
        x: Float[Array, "*batch D_in"],
        z: Float[Array, "*batch K"] | Float[Array, " K"],
    ) -> Float[Array, "*batch D_rff"]:
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"x.shape[-1]={x.shape[-1]} does not match "
                f"in_features={self.in_features}."
            )
        if z.shape[-1] != self.cond_dim:
            raise ValueError(
                f"z.shape[-1]={z.shape[-1]} does not match cond_dim={self.cond_dim}."
            )
        # Single-example forward over arbitrary leading batch dims of x
        # (and z, when conditioning is per-sample).
        if z.ndim == 1:
            W, b, log_l = self._unpack(z)
            return vmap_over_flat_batch(
                lambda xi: self._single_example(xi, W, b, log_l), x
            )

        batch_shape = x.shape[:-1]
        x_flat = einx.id("b... d -> (b...) d", x)
        z_flat = einx.id("b... k -> (b...) k", z)
        W_all, b_all, log_l_all = jax.vmap(self._unpack)(z_flat)
        out_flat = jax.vmap(self._single_example)(x_flat, W_all, b_all, log_l_all)
        return einx.id("(b...) k -> b... k", out_flat, b=batch_shape)


class ConditionedRFFNet(PyroxModule):
    """Conditional analogue of `pyrox_nn.RandomKitchenSinks`.

    Composes a `HyperFourierFeatures` feature map with a learnable
    linear readout. The full forward is

    $$
    y(x; z) = \\phi(x; z)\\, \\beta + b_{\\text{out}}
    $$

    where $\\phi(x; z)$ is the ``HyperFourierFeatures`` output and
    ``(beta, b_out)`` are the readout's deterministic weights. For the
    Bayesian variant, wrap ``readout`` in a ``DenseReparameterization`` and
    move the priors there — this composite stays minimal.

    Attributes:
        feat: A `HyperFourierFeatures` instance.
        readout: ``eqx.nn.Linear`` mapping ``2 * n_features -> out_features``.
        pyrox_name: Optional explicit scope name.

    Examples:
        >>> import jax.random as jr, jax.numpy as jnp
        >>> import equinox as eqx
        >>> key = jr.key(0)
        >>> pnet = eqx.nn.MLP(
        ...     in_size=4, out_size=1 * 32 + 32 + 1, width_size=32, depth=2, key=key,
        ... )
        >>> feat = HyperFourierFeatures.init(
        ...     parameter_net=pnet, in_features=1, n_features=32, cond_dim=4,
        ... )
        >>> net = ConditionedRFFNet.init(feat=feat, out_features=1, key=key)
        >>> y = net(jnp.zeros((10, 1)), jnp.zeros((10, 4)))
        >>> y.shape
        (10, 1)
    """

    feat: HyperFourierFeatures
    readout: eqx.nn.Linear
    pyrox_name: str | None = eqx.field(static=True, default=None)

    @classmethod
    def init(
        cls,
        *,
        feat: HyperFourierFeatures,
        out_features: int,
        key: Array,
        pyrox_name: str | None = None,
    ) -> ConditionedRFFNet:
        """Build `ConditionedRFFNet` with a default linear readout."""
        if out_features <= 0:
            raise ValueError(f"out_features must be > 0; got {out_features}.")
        readout = eqx.nn.Linear(2 * feat.n_features, out_features, key=key)
        return cls(feat=feat, readout=readout, pyrox_name=pyrox_name)

    @pyrox_method
    def __call__(
        self,
        x: Float[Array, "*batch D_in"],
        z: Float[Array, "*batch K"] | Float[Array, " K"],
    ) -> Float[Array, "*batch D_out"]:
        phi = self.feat(x, z)
        # Single-example linear readout over arbitrary leading batch dims.
        return vmap_over_flat_batch(self.readout, phi)


__all__ = [
    "BayesianAffineModulation",
    "BayesianConcatConditioner",
    "BayesianHyperLinear",
    "ConditionedRFFNet",
    "HyperFourierFeatures",
]
