"""Unified conditioning primitives for ``pyrox.nn``.

A conditioner is a layer ``c(h, z) -> y`` that transforms an inner
activation ``h`` based on a context / latent code ``z``. Three concrete
conditioners cover the literature:

* :class:`ConcatConditioner` — ``y = Linear([h ‖ z])``. Cheapest baseline,
  parameter-heavy in ``cond_dim``. Same trick as
  ``gauss_flows._src.nn.diffeq_nets.DiffeqMLP``.
* :class:`AffineModulation` (also exported as :data:`FiLM`) — feature-wise
  affine ``y = γ(z) ⊙ h + β(z)`` with a single ``eqx.nn.Linear`` generator.
  Adapted from ``gauss_flows._src.transforms.bijections.conditioner.Conditioner``;
  generalised here with selectable ``γ`` activation. The
  ``gamma_activation="exp"`` mode exposes :meth:`AffineModulation.log_det`
  so flowjax-style code can use it as a bijection wrapper.
* :class:`HyperLinear` — full hypernetwork, ``(W, b) = g(z); y = W x + b``.
  Generator is a single ``eqx.nn.Linear`` of width
  ``target_out * target_in + target_out``.

Bayesian variants (:class:`BayesianConcatConditioner`,
:class:`BayesianAffineModulation`, :class:`BayesianHyperLinear`) put
Normal priors on the **generator** weights only — never on ``h``, ``z``,
or the inner network — so prior cost scales with the generator size, not
the target size. This is the architectural advantage of doing Bayesian
amortised inference via hypernetworks (NIF, MetaSDF) rather than directly
over the target weights.

The composite :class:`ConditionedINR` wraps any inner network exposing a
``layers`` sequence (e.g. :class:`pyrox.nn.SIREN`,
:class:`pyrox.nn.BayesianNeuralField`) with a per-layer conditioner. The
:func:`HyperSIREN` constructor sugar builds the NIF ShapeNet/ParameterNet
composite (Pan, Brunton, Kutz — JMLR 2023) by special-casing per-layer
init-scale calibration so the generated weights match Sitzmann's
variance-preservation property.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal

import einops
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as dist
from jaxtyping import Array, Float, PRNGKeyArray

from pyrox._core.pyrox_module import PyroxModule, pyrox_method
from pyrox.nn._layers import SIREN, SirenDense, _siren_W_limit


GammaActivation = Literal["one_plus_tanh", "exp", "softplus", "identity"]
ConditionedMode = Literal["input", "feature"]


_GAMMA_ACTIVATIONS: tuple[str, ...] = ("one_plus_tanh", "exp", "softplus", "identity")


def _apply_gamma(raw: Array, kind: str) -> Array:
    if kind == "one_plus_tanh":
        return 1.0 + jnp.tanh(raw)
    if kind == "exp":
        return jnp.exp(raw)
    if kind == "softplus":
        return jax.nn.softplus(raw)
    if kind == "identity":
        return raw
    raise ValueError(
        f"gamma_activation must be one of {_GAMMA_ACTIVATIONS}; got {kind!r}."
    )


def _broadcast_z(z: Array, n_rows: int) -> Array:
    """Broadcast a single context ``(K,)`` to ``(N, K)``; pass-through otherwise."""
    if z.ndim == 1:
        return einops.repeat(z, "k -> n k", n=n_rows)
    return z


def _atleast_2d_pair(h: Array, z: Array) -> tuple[Array, Array, bool]:
    """Promote ``h``, ``z`` to 2D for the inner kernel; record whether to squeeze."""
    squeeze = h.ndim == 1
    if squeeze:
        h = einops.rearrange(h, "c -> 1 c")
    z = _broadcast_z(z, h.shape[0])
    return h, z, squeeze


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class AbstractConditioner(PyroxModule):
    """Duck-typed protocol for ``(h, z) -> y`` conditioning layers.

    Concrete subclasses share the contract ``__call__(h, z) -> Array``
    where ``h.shape[-1] == num_features`` and ``z.shape[-1] == cond_dim``.
    There is no ``abstractmethod`` enforcement — subclasses simply
    implement ``__call__`` decorated with :func:`pyrox_method`.

    Attributes:
        num_features: Output channel count, matching ``h.shape[-1]``.
        cond_dim: Latent / context dimension, matching ``z.shape[-1]``.
    """

    num_features: int = eqx.field(static=True)
    cond_dim: int = eqx.field(static=True)

    def __call__(self, h: Array, z: Array) -> Array:  # pragma: no cover
        # Declared so type checkers know subclasses are callable; concrete
        # subclasses override this with ``@pyrox_method``.
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Concat conditioner: y = Linear([h ‖ z])
# ---------------------------------------------------------------------------


class ConcatConditioner(AbstractConditioner):
    """Concatenate ``h`` and ``z`` then apply a single ``Linear``.

    Cheapest, most expressive in principle, but parameter count grows
    linearly with ``cond_dim``: ``(num_features + cond_dim) * num_features
    + num_features`` (the bias). No init ceremony required — uses
    ``eqx.nn.Linear`` defaults.

    Attributes:
        proj: Linear projection ``R^{C+K} -> R^{C}``.
        num_features: Output channels ``C``.
        cond_dim: Context dimension ``K``.
        pyrox_name: Optional explicit scope name for NumPyro.

    Example:
        >>> import jax.random as jr, jax.numpy as jnp
        >>> layer = ConcatConditioner.init(num_features=8, cond_dim=4, key=jr.key(0))
        >>> y = layer(jnp.ones((5, 8)), jnp.ones((5, 4)))
        >>> y.shape
        (5, 8)
    """

    proj: eqx.nn.Linear
    pyrox_name: str | None = eqx.field(static=True, default=None)

    @classmethod
    def init(
        cls,
        num_features: int,
        cond_dim: int,
        *,
        key: PRNGKeyArray,
        pyrox_name: str | None = None,
    ) -> ConcatConditioner:
        """Build a :class:`ConcatConditioner` with default ``eqx.nn.Linear`` init.

        Args:
            num_features: Output channel count.
            cond_dim: Context dimension.
            key: PRNG key for the projection's init.
            pyrox_name: Optional explicit scope name.

        Returns:
            Initialised :class:`ConcatConditioner`.

        Raises:
            ValueError: If ``num_features`` or ``cond_dim`` is non-positive.
        """
        if num_features <= 0 or cond_dim <= 0:
            raise ValueError(
                "num_features and cond_dim must be positive; got "
                f"num_features={num_features}, cond_dim={cond_dim}."
            )
        proj = eqx.nn.Linear(num_features + cond_dim, num_features, key=key)
        return cls(
            num_features=num_features,
            cond_dim=cond_dim,
            proj=proj,
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
        h2d, z2d, squeeze = _atleast_2d_pair(h, z)
        # Concatenate on the feature axis, then per-row Linear via vmap.
        cat = jnp.concatenate([h2d, z2d], axis=-1)
        out = jax.vmap(self.proj)(cat)
        return out[0] if squeeze else out


# ---------------------------------------------------------------------------
# Affine modulation (FiLM): y = γ(z) ⊙ h + β(z)
# ---------------------------------------------------------------------------


class AffineModulation(AbstractConditioner):
    r"""Feature-wise Linear Modulation (FiLM): ``y = γ(z) ⊙ h + β(z)``.

    A single ``eqx.nn.Linear`` of output size ``2 * num_features``
    produces the concatenated ``(raw_β, raw_γ)`` from the context vector.
    The two halves are split on the feature axis via
    :func:`einops.rearrange` (no raw ``jnp.split``), then ``γ`` is passed
    through the chosen activation:

    * ``"one_plus_tanh"`` (default): ``γ = 1 + tanh(raw_γ)`` — identity at
      init when the generator's bias is zero. The choice that gives FiLM
      its "does nothing until trained" property.
    * ``"exp"``: ``γ = exp(raw_γ)`` — strictly positive, required for
      bijection use. In this mode :meth:`log_det` returns
      ``sum(raw_γ, axis=-1)``, the closed-form log-Jacobian of an
      element-wise scale.
    * ``"softplus"``: ``γ = softplus(raw_γ)`` — strictly positive, slower
      to leave the prior than ``exp``.
    * ``"identity"``: ``γ = raw_γ`` — no shape guarantee, rarely useful.

    Attributes:
        generator: Linear ``R^K -> R^{2C}``.
        num_features: Output channels ``C``.
        cond_dim: Context dimension ``K``.
        gamma_activation: Parameterisation of ``γ`` (see above).
        pyrox_name: Optional explicit scope name for NumPyro.

    Example:
        >>> import jax.random as jr, jax.numpy as jnp
        >>> film = AffineModulation.init(num_features=8, cond_dim=4, key=jr.key(0))
        >>> y = film(jnp.ones((5, 8)), jnp.ones((5, 4)))
        >>> y.shape
        (5, 8)
    """

    generator: eqx.nn.Linear
    gamma_activation: GammaActivation = eqx.field(static=True)
    pyrox_name: str | None = eqx.field(static=True, default=None)

    @classmethod
    def init(
        cls,
        num_features: int,
        cond_dim: int,
        *,
        key: PRNGKeyArray,
        gamma_activation: GammaActivation = "one_plus_tanh",
        pyrox_name: str | None = None,
    ) -> AffineModulation:
        """Build :class:`AffineModulation` with the default 2-output Linear generator.

        Args:
            num_features: Output channel count.
            cond_dim: Context dimension.
            key: PRNG key for the generator's init.
            gamma_activation: Parameterisation of ``γ``; see the class docstring.
            pyrox_name: Optional explicit scope name.

        Returns:
            Initialised :class:`AffineModulation`.

        Raises:
            ValueError: If ``num_features`` or ``cond_dim`` is non-positive,
                or if ``gamma_activation`` is not a recognised value.
        """
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
        generator = eqx.nn.Linear(cond_dim, 2 * num_features, key=key)
        # Bias-only zero-init: with bias=0 and "one_plus_tanh", γ=1 and β=0
        # → the layer is identity at init (Perez et al. 2018 default).
        generator = eqx.tree_at(
            lambda m: m.bias,
            generator,
            jnp.zeros_like(generator.bias),  # ty: ignore[unresolved-attribute]
        )
        return cls(
            num_features=num_features,
            cond_dim=cond_dim,
            generator=generator,
            gamma_activation=gamma_activation,
            pyrox_name=pyrox_name,
        )

    def _gamma_beta(self, z: Array) -> tuple[Array, Array, Array]:
        """Compute ``(raw_γ, γ, β)`` from a context array of shape ``(N, K)``."""
        raw = jax.vmap(self.generator)(z)  # (N, 2C)
        # Split on the feature axis: first half = β, second half = raw_γ.
        # einops.rearrange keeps the (two, c) split explicit and avoids jnp.split.
        split = einops.rearrange(raw, "n (two c) -> two n c", two=2)
        beta, raw_gamma = split[0], split[1]
        gamma = _apply_gamma(raw_gamma, self.gamma_activation)
        return raw_gamma, gamma, beta

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
        h2d, z2d, squeeze = _atleast_2d_pair(h, z)
        _raw_gamma, gamma, beta = self._gamma_beta(z2d)
        out = gamma * h2d + beta
        return out[0] if squeeze else out

    def log_det(
        self,
        z: Float[Array, "*batch K"] | Float[Array, " K"],
    ) -> Float[Array, " *batch"]:
        """Sum of ``log γ`` across the feature axis.

        Only valid when ``gamma_activation="exp"`` — that's the only
        parameterisation for which ``log γ = raw_γ`` exactly. For other
        modes this raises :class:`NotImplementedError`; callers that need
        a generic Jacobian must compute it manually.

        Args:
            z: Context array of shape ``(N, K)`` or ``(K,)``.

        Returns:
            Log-determinant of the diagonal scaling, shape ``(N,)`` (or
            scalar for 1-D ``z``).

        Raises:
            NotImplementedError: If ``gamma_activation != "exp"``.
        """
        if self.gamma_activation != "exp":
            raise NotImplementedError(
                "log_det is only defined for gamma_activation='exp'; got "
                f"{self.gamma_activation!r}. Use exp parameterisation for "
                "bijection wrappers."
            )
        squeeze = z.ndim == 1
        z2d = einops.rearrange(z, "k -> 1 k") if squeeze else z
        raw_gamma, _gamma, _beta = self._gamma_beta(z2d)
        ldj = einops.reduce(raw_gamma, "n c -> n", "sum")
        return ldj[0] if squeeze else ldj


#: Backwards-compatible alias for :class:`AffineModulation`.
#:
#: Issue #86 specs the layer as ``FiLM``; both names point at the same class.
FiLM = AffineModulation


# ---------------------------------------------------------------------------
# Hypernetwork: (W, b) = g(z); y = W x + b
# ---------------------------------------------------------------------------


class HyperLinear(AbstractConditioner):
    """Generate a target ``Linear``'s ``(W, b)`` from ``z``, then apply.

    A single ``eqx.nn.Linear`` of output size ``target_out * target_in +
    target_out`` produces the flat parameter vector for an ad-hoc linear
    layer; ``W`` and ``b`` are split out via :func:`einops.rearrange`.
    The forward dispatches on ``z.ndim``:

    * ``z.shape == (K,)`` — *shared* path: one ``(W, b)`` generated and
      reused across every row of ``x``. Cheap (one small affine + one
      matmul).
    * ``z.shape == (N, K)`` — *per-sample* path: ``(W, b)`` generated for
      each row, applied via ``einops.einsum``. Costs ``N * C * C_in``
      flops per call.

    The generator weight scale is multiplied by ``init_scale`` so the
    generated ``W`` magnitude starts small and the composite is near-zero
    at init. Default ``init_scale=0.1`` matches NIF (Pan et al. 2023).

    Attributes:
        generator: Linear ``R^K -> R^{C_out * C_in + C_out}``.
        target_in: Inner ``Linear``'s input dim ``C_in``.
        target_out: Inner ``Linear``'s output dim ``C_out`` (= num_features).
        cond_dim: Context dimension ``K``.
        num_features: Alias for ``target_out`` (satisfies the protocol).
        pyrox_name: Optional explicit scope name for NumPyro.

    Example:
        >>> import jax.random as jr, jax.numpy as jnp
        >>> hyper = HyperLinear.init(
        ...     target_in=4, target_out=8, cond_dim=3, key=jr.key(0)
        ... )
        >>> y_shared = hyper(jnp.ones((6, 4)), jnp.ones((3,)))
        >>> y_persample = hyper(jnp.ones((6, 4)), jnp.ones((6, 3)))
        >>> (y_shared.shape, y_persample.shape)
        ((6, 8), (6, 8))
    """

    generator: eqx.nn.Linear
    target_in: int = eqx.field(static=True)
    target_out: int = eqx.field(static=True)
    pyrox_name: str | None = eqx.field(static=True, default=None)

    @classmethod
    def init(
        cls,
        target_in: int,
        target_out: int,
        cond_dim: int,
        *,
        key: PRNGKeyArray,
        init_scale: float = 0.1,
        pyrox_name: str | None = None,
    ) -> HyperLinear:
        """Build a :class:`HyperLinear` with a small-magnitude generator init.

        Args:
            target_in: Input dimension of the generated ``Linear``.
            target_out: Output dimension of the generated ``Linear``.
            cond_dim: Context dimension.
            key: PRNG key for generator init.
            init_scale: Multiplicative factor on the generator weights so
                the generated ``W`` stays small at init. Default ``0.1``.
            pyrox_name: Optional explicit scope name.

        Returns:
            Initialised :class:`HyperLinear`.

        Raises:
            ValueError: If any of ``target_in``, ``target_out``,
                ``cond_dim``, or ``init_scale`` is non-positive.
        """
        if target_in <= 0 or target_out <= 0 or cond_dim <= 0:
            raise ValueError(
                "target_in, target_out, and cond_dim must all be positive; got "
                f"target_in={target_in}, target_out={target_out}, "
                f"cond_dim={cond_dim}."
            )
        if init_scale <= 0:
            raise ValueError(f"init_scale must be > 0; got {init_scale}.")
        flat_size = target_out * target_in + target_out
        gen = eqx.nn.Linear(cond_dim, flat_size, key=key)
        # Scale weights and zero the bias so the generated (W, b) are small at init.
        gen = eqx.tree_at(
            lambda m: m.weight,
            gen,
            gen.weight * init_scale,  # ty: ignore[unresolved-attribute]
        )
        gen = eqx.tree_at(
            lambda m: m.bias,
            gen,
            jnp.zeros_like(gen.bias),
        )
        return cls(
            num_features=target_out,
            cond_dim=cond_dim,
            generator=gen,
            target_in=target_in,
            target_out=target_out,
            pyrox_name=pyrox_name,
        )

    def _split_params(self, flat: Array) -> tuple[Array, Array]:
        """Split flat ``(out * in + out,)`` into ``W: (out, in)`` and ``b: (out,)``."""
        w_size = self.target_out * self.target_in
        flat_W, flat_b = flat[:w_size], flat[w_size:]
        W = einops.rearrange(
            flat_W, "(c c_in) -> c c_in", c=self.target_out, c_in=self.target_in
        )
        return W, flat_b

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
        squeeze = x.ndim == 1
        if squeeze:
            x = einops.rearrange(x, "c -> 1 c")

        if z.ndim == 1:
            # Shared (W, b): generate once, reuse across all rows.
            flat = self.generator(z)
            W, b = self._split_params(flat)
            out = einops.einsum(W, x, "c c_in, n c_in -> n c") + b
        else:
            # Per-sample (W, b): generate per row of z, einsum per row.
            flats = jax.vmap(self.generator)(z)  # (N, out*in + out)
            w_size = self.target_out * self.target_in
            flat_W = flats[:, :w_size]
            flat_b = flats[:, w_size:]
            W = einops.rearrange(
                flat_W,
                "n (c c_in) -> n c c_in",
                c=self.target_out,
                c_in=self.target_in,
            )
            out = einops.einsum(W, x, "n c c_in, n c_in -> n c") + flat_b

        return out[0] if squeeze else out


# ---------------------------------------------------------------------------
# Bayesian variants — priors live on the generator only
# ---------------------------------------------------------------------------


class BayesianConcatConditioner(AbstractConditioner):
    """:class:`ConcatConditioner` with Normal priors on the projection.

    Registers two NumPyro sample sites — ``{scope}.proj_W`` and
    ``{scope}.proj_b`` — under ``Normal(0, prior_std)``. Total of two
    sites per forward call; nothing is sampled from the inner ``h`` or
    the context ``z``.

    Attributes:
        num_features: Output channels.
        cond_dim: Context dimension.
        prior_std: Scale of the Normal priors.
        pyrox_name: Optional explicit scope name for NumPyro.
    """

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
        """Build a :class:`BayesianConcatConditioner`.

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
        return cls(
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
        in_dim = self.num_features + self.cond_dim
        W = self.pyrox_sample(
            "proj_W",
            dist.Normal(0.0, self.prior_std)
            .expand([in_dim, self.num_features])
            .to_event(2),
        )
        b = self.pyrox_sample(
            "proj_b",
            dist.Normal(0.0, self.prior_std).expand([self.num_features]).to_event(1),
        )
        h2d, z2d, squeeze = _atleast_2d_pair(h, z)
        cat = jnp.concatenate([h2d, z2d], axis=-1)
        out = cat @ W + b
        return out[0] if squeeze else out


class BayesianAffineModulation(AbstractConditioner):
    """:class:`AffineModulation` with Normal priors on the FiLM generator.

    Registers two sites — ``{scope}.gen_W`` and ``{scope}.gen_b`` —
    under ``Normal(0, prior_std)``. The ``γ`` activation is fixed by
    construction (default ``"one_plus_tanh"``) so the prior over the raw
    generator output induces a well-defined prior over ``γ``, ``β``.

    Attributes:
        num_features: Output channels.
        cond_dim: Context dimension.
        gamma_activation: Parameterisation of ``γ``.
        prior_std: Scale of the Normal priors.
        pyrox_name: Optional explicit scope name.
    """

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
        """Build a :class:`BayesianAffineModulation`."""
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
        return cls(
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
        out_dim = 2 * self.num_features
        W = self.pyrox_sample(
            "gen_W",
            dist.Normal(0.0, self.prior_std)
            .expand([self.cond_dim, out_dim])
            .to_event(2),
        )
        b = self.pyrox_sample(
            "gen_b",
            dist.Normal(0.0, self.prior_std).expand([out_dim]).to_event(1),
        )
        h2d, z2d, squeeze = _atleast_2d_pair(h, z)
        raw = z2d @ W + b  # (N, 2C)
        split = einops.rearrange(raw, "n (two c) -> two n c", two=2)
        beta, raw_gamma = split[0], split[1]
        gamma = _apply_gamma(raw_gamma, self.gamma_activation)
        out = gamma * h2d + beta
        return out[0] if squeeze else out


class BayesianHyperLinear(AbstractConditioner):
    """:class:`HyperLinear` with Normal priors on the generator only.

    Two sites: ``{scope}.gen_W`` and ``{scope}.gen_b``. The target
    weights ``(W_target, b_target)`` are *generated* — not sampled — so
    Bayesian inference cost scales with the generator size
    ``cond_dim * (target_out * target_in + target_out)``, not with the
    target-network size. This is the architectural advantage of doing
    Bayesian amortised inference via hypernetworks.

    Attributes:
        target_in: Inner ``Linear``'s input dim ``C_in``.
        target_out: Inner ``Linear``'s output dim ``C_out``.
        cond_dim: Context dimension ``K``.
        num_features: Alias for ``target_out``.
        prior_std: Scale of the Normal priors on the generator.
        pyrox_name: Optional explicit scope name.
    """

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
        """Build a :class:`BayesianHyperLinear`."""
        if target_in <= 0 or target_out <= 0 or cond_dim <= 0:
            raise ValueError(
                "target_in, target_out, and cond_dim must all be positive; got "
                f"target_in={target_in}, target_out={target_out}, "
                f"cond_dim={cond_dim}."
            )
        if prior_std <= 0:
            raise ValueError(f"prior_std must be > 0; got {prior_std}.")
        return cls(
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
        flat_size = self.target_out * self.target_in + self.target_out
        W_gen = self.pyrox_sample(
            "gen_W",
            dist.Normal(0.0, self.prior_std)
            .expand([self.cond_dim, flat_size])
            .to_event(2),
        )
        b_gen = self.pyrox_sample(
            "gen_b",
            dist.Normal(0.0, self.prior_std).expand([flat_size]).to_event(1),
        )
        squeeze = x.ndim == 1
        if squeeze:
            x = einops.rearrange(x, "c -> 1 c")
        w_size = self.target_out * self.target_in

        if z.ndim == 1:
            flat = z @ W_gen + b_gen  # (flat_size,)
            flat_W, flat_b = flat[:w_size], flat[w_size:]
            W = einops.rearrange(
                flat_W, "(c c_in) -> c c_in", c=self.target_out, c_in=self.target_in
            )
            out = einops.einsum(W, x, "c c_in, n c_in -> n c") + flat_b
        else:
            flats = z @ W_gen + b_gen  # (N, flat_size)
            flat_W = flats[:, :w_size]
            flat_b = flats[:, w_size:]
            W = einops.rearrange(
                flat_W,
                "n (c c_in) -> n c c_in",
                c=self.target_out,
                c_in=self.target_in,
            )
            out = einops.einsum(W, x, "n c c_in, n c_in -> n c") + flat_b

        return out[0] if squeeze else out


# ---------------------------------------------------------------------------
# Composite: any inner-network + per-layer conditioning
# ---------------------------------------------------------------------------


class ConditionedINR(PyroxModule):
    """Wrap an inner network's per-layer activations with conditioners.

    Given an ``inner`` network exposing a ``layers`` sequence (true for
    :class:`pyrox.nn.SIREN` and any module that holds a list of callables
    named ``layers``), :class:`ConditionedINR` runs the inner forward and
    inserts a conditioner after each non-readout layer:

    .. code:: text

        z_0 = layer_0(x)
        z_0 = cond_0(z_0, c)
        z_1 = layer_1(z_0)
        z_1 = cond_1(z_1, c)
        ...
        y   = layer_{L-1}(z_{L-2})        # readout, not conditioned

    The ``mode="input"`` shortcut concatenates ``c`` to the input
    *before* running ``inner`` — useful for inner networks that don't
    expose a ``layers`` sequence (e.g. plain ``eqx.nn.MLP`` instances).

    Conditioners must be ``AbstractConditioner`` instances whose
    ``num_features`` matches the corresponding ``inner`` layer's output
    width. The composite forward registers the union of the inner
    network's sample sites and the per-layer conditioners' sites — no
    site clashes because each conditioner gets a distinct ``pyrox_name``.

    Attributes:
        inner: Inner network with a ``layers`` attribute (for
            ``mode="feature"``) or any callable (for ``mode="input"``).
        conditioners: Per-layer conditioner list. Length equals
            ``len(inner.layers) - 1`` for ``"feature"`` (no readout
            conditioning) or 1 for ``"input"`` (a single
            ``ConcatConditioner``-style head).
        cond_dim: Context dimension shared by all conditioners.
        mode: ``"feature"`` for per-layer modulation;
            ``"input"`` for input-side concatenation.
        pyrox_name: Optional explicit scope for NumPyro.

    Example:
        >>> import jax.random as jr, jax.numpy as jnp
        >>> from pyrox.nn import SIREN
        >>> key = jr.key(0)
        >>> inner = SIREN.init(2, 32, 1, depth=4, key=key)
        >>> wrapped = ConditionedINR.init(
        ...     inner, conditioner_cls=AffineModulation, cond_dim=4, key=key
        ... )
        >>> y = wrapped(jnp.zeros((10, 2)), jnp.zeros((10, 4)))
        >>> y.shape
        (10, 1)
    """

    inner: PyroxModule | eqx.Module
    conditioners: list[AbstractConditioner]
    cond_dim: int = eqx.field(static=True)
    mode: ConditionedMode = eqx.field(static=True)
    pyrox_name: str | None = eqx.field(static=True, default=None)

    @classmethod
    def init(
        cls,
        inner: PyroxModule | eqx.Module,
        *,
        conditioner_cls: type[AbstractConditioner],
        cond_dim: int,
        key: PRNGKeyArray,
        mode: ConditionedMode = "feature",
        pyrox_name: str | None = None,
        **conditioner_kwargs: float,
    ) -> ConditionedINR:
        """Build a :class:`ConditionedINR` around ``inner``.

        Args:
            inner: Inner network. Must have ``layers: Sequence`` for
                ``mode="feature"``; any callable works for ``mode="input"``.
            conditioner_cls: One of :class:`ConcatConditioner`,
                :class:`AffineModulation`, :class:`HyperLinear`, or any of
                the Bayesian variants.
            cond_dim: Context dimension passed to each conditioner.
            key: PRNG key, split internally for each conditioner.
            mode: ``"feature"`` (per-layer modulation, default) or
                ``"input"`` (single input-side concatenation).
            pyrox_name: Optional explicit scope name.
            **conditioner_kwargs: Extra kwargs forwarded to each
                ``conditioner_cls.init``.

        Returns:
            Initialised :class:`ConditionedINR`.

        Raises:
            ValueError: If ``mode == "feature"`` and ``inner`` lacks a
                ``layers`` attribute, or if any layer is missing the
                ``out_features`` shape needed to size the conditioners.
        """
        if mode not in ("feature", "input"):
            raise ValueError(f"mode must be 'feature' or 'input'; got {mode!r}.")

        if mode == "input":
            keys = jr.split(key, 1)
            head = _build_conditioner(
                conditioner_cls,
                num_features=_inner_in_features(inner),
                cond_dim=cond_dim,
                key=keys[0],
                pyrox_name=(f"{pyrox_name}.cond_input" if pyrox_name else None),
                **conditioner_kwargs,
            )
            return cls(
                inner=inner,
                conditioners=[head],
                cond_dim=cond_dim,
                mode="input",
                pyrox_name=pyrox_name,
            )

        layers = getattr(inner, "layers", None)
        if layers is None:
            raise ValueError(
                "mode='feature' requires `inner` to expose a `layers` sequence "
                "(e.g. pyrox.nn.SIREN). For inners without `layers`, use mode='input'."
            )
        if len(layers) < 2:
            raise ValueError(
                "mode='feature' needs at least two inner layers; got "
                f"{len(layers)}. Use mode='input' for shallow inners."
            )

        # One conditioner per non-readout layer (skip the last).
        n_cond = len(layers) - 1
        keys = jr.split(key, n_cond)
        conditioners: list[AbstractConditioner] = []
        for i, k in enumerate(keys):
            num_features = _layer_out_features(layers[i], i)
            conditioners.append(
                _build_conditioner(
                    conditioner_cls,
                    num_features=num_features,
                    cond_dim=cond_dim,
                    key=k,
                    pyrox_name=(
                        f"{pyrox_name}.cond_{i}" if pyrox_name else f"cond_{i}"
                    ),
                    **conditioner_kwargs,
                )
            )
        return cls(
            inner=inner,
            conditioners=conditioners,
            cond_dim=cond_dim,
            mode="feature",
            pyrox_name=pyrox_name,
        )

    @pyrox_method
    def __call__(
        self,
        x: Float[Array, "*batch D_in"],
        z: Float[Array, "*batch K"] | Float[Array, " K"],
    ) -> Float[Array, "*batch D_out"]:
        if z.shape[-1] != self.cond_dim:
            raise ValueError(
                f"z.shape[-1]={z.shape[-1]} does not match cond_dim={self.cond_dim}."
            )
        if self.mode == "input":
            head = self.conditioners[0]
            x = head(x, z)
            return self.inner(x)  # ty: ignore[call-non-callable]

        # mode == "feature": run each non-readout inner layer, then condition.
        layers = self.inner.layers  # ty: ignore[unresolved-attribute]
        h = x
        for i, layer in enumerate(layers[:-1]):
            h = layer(h)
            h = self.conditioners[i](h, z)
        return layers[-1](h)


def _build_conditioner(
    cls: type[AbstractConditioner],
    *,
    num_features: int,
    cond_dim: int,
    key: PRNGKeyArray,
    pyrox_name: str | None,
    **kwargs: float,
) -> AbstractConditioner:
    """Construct a conditioner, threading the key only for variants that need one."""
    if cls is HyperLinear:
        # HyperLinear needs (target_in, target_out, cond_dim).
        return HyperLinear.init(
            target_in=num_features,
            target_out=num_features,
            cond_dim=cond_dim,
            key=key,
            pyrox_name=pyrox_name,
            **kwargs,
        )
    if cls is BayesianHyperLinear:
        return BayesianHyperLinear.init(
            target_in=num_features,
            target_out=num_features,
            cond_dim=cond_dim,
            pyrox_name=pyrox_name,
            **kwargs,
        )
    if cls in (BayesianConcatConditioner, BayesianAffineModulation):
        # Bayesian variants don't need a PRNG key (weights come from prior).
        return cls.init(  # ty: ignore[unresolved-attribute]
            num_features=num_features,
            cond_dim=cond_dim,
            pyrox_name=pyrox_name,
            **kwargs,
        )
    return cls.init(  # ty: ignore[unresolved-attribute]
        num_features=num_features,
        cond_dim=cond_dim,
        key=key,
        pyrox_name=pyrox_name,
        **kwargs,
    )


def _layer_out_features(layer: object, index: int) -> int:
    """Best-effort extraction of a layer's output width."""
    for attr in ("out_features", "output_dim", "hidden_features"):
        v = getattr(layer, attr, None)
        if isinstance(v, int) and v > 0:
            return v
    raise ValueError(
        f"Could not infer out_features for layer #{index} of type "
        f"{type(layer).__name__}. Expose an int `out_features` attribute."
    )


def _inner_in_features(inner: object) -> int:
    for attr in ("in_features", "input_dim"):
        v = getattr(inner, attr, None)
        if isinstance(v, int) and v > 0:
            return v
    raise ValueError(
        "Could not infer in_features for the inner network. Expose an int "
        "`in_features` attribute or use mode='feature'."
    )


# ---------------------------------------------------------------------------
# Constructor sugar: NIF-style HyperSIREN
# ---------------------------------------------------------------------------


class _GeneratedSiren(eqx.Module):
    """Internal wrapper around a SIREN whose layers consume generated weights."""

    parameter_net: eqx.Module
    siren: SIREN
    hyper_layers: list[HyperLinear]
    in_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)

    def __call__(self, x: Array, mu: Array) -> Array:
        z = self.parameter_net(mu)  # ty: ignore[call-non-callable]
        squeeze = x.ndim == 1
        if squeeze:
            x = einops.rearrange(x, "c -> 1 c")
        h = x
        for siren_layer, hyper in zip(
            self.siren.layers, self.hyper_layers, strict=True
        ):
            pre = hyper(h, z)
            if siren_layer.layer_type == "last":
                h = pre
            else:
                h = jnp.sin(siren_layer.omega * pre)
        return h[0] if squeeze else h


def HyperSIREN(
    in_features: int,
    hidden_features: int,
    out_features: int,
    *,
    depth: int,
    cond_dim: int,
    parameter_net: eqx.Module,
    key: PRNGKeyArray,
    first_omega: float = 30.0,
    hidden_omega: float = 30.0,
    c: float = 6.0,
    init_scale: float = 0.1,
) -> _GeneratedSiren:
    """NIF-style ShapeNet/ParameterNet composite (Pan, Brunton, Kutz — JMLR 2023).

    Builds a SIREN shape-net of the requested topology, then constructs a
    parallel list of :class:`HyperLinear` generators — one per SIREN layer
    — whose ``init_scale`` is calibrated per Sitzmann regime so the
    *expected magnitude* of each generated ``W`` matches the half-width
    of Sitzmann's :func:`pyrox.nn._layers._siren_W_limit` at init.
    Without this calibration the ShapeNet's pre-activation variance is
    wrong and training is unstable.

    The user-supplied ``parameter_net`` runs once on ``mu`` per forward
    call to produce the latent ``z``; ``z`` then drives every per-layer
    :class:`HyperLinear`. ``parameter_net`` must be callable with signature
    ``(P,) -> (cond_dim,)``.

    Args:
        in_features: Coordinate dimension of the SIREN.
        hidden_features: Hidden width.
        out_features: Output dimension.
        depth: SIREN depth (must be ≥ 2).
        cond_dim: Latent dimension produced by ``parameter_net``.
        parameter_net: User-supplied callable ``(P,) -> (cond_dim,)``.
        key: PRNG key, split internally for the SIREN init and the hyper
            generators.
        first_omega: First-layer ``omega``.
        hidden_omega: Hidden-layer ``omega``.
        c: SIREN Theorem-1 constant.
        init_scale: Multiplicative factor applied on top of the per-regime
            calibration; default ``0.1`` matches NIF.

    Returns:
        A composite that takes ``(x, mu)`` and runs the NIF forward.

    Raises:
        ValueError: If ``depth < 2`` or any positive-only argument is
            non-positive.
    """
    if depth < 2:
        raise ValueError(f"depth must be >= 2 (first + last); got depth={depth}.")
    for name, v in {
        "in_features": in_features,
        "hidden_features": hidden_features,
        "out_features": out_features,
        "cond_dim": cond_dim,
        "first_omega": first_omega,
        "hidden_omega": hidden_omega,
        "c": c,
        "init_scale": init_scale,
    }.items():
        if v <= 0:
            raise ValueError(f"{name} must be > 0; got {v}.")

    siren_key, *hyper_keys = jr.split(key, 1 + depth)
    siren = SIREN.init(
        in_features,
        hidden_features,
        out_features,
        depth=depth,
        key=siren_key,
        first_omega=first_omega,
        hidden_omega=hidden_omega,
        c=c,
    )
    hyper_layers: list[HyperLinear] = []
    for i, layer in enumerate(siren.layers):
        layer_in = layer.in_features
        layer_out = layer.out_features
        # Calibrate so generated W magnitude ≈ Sitzmann's per-regime half-width.
        regime_limit = _siren_W_limit(layer.layer_type, layer_in, layer.omega, c)
        per_layer_scale = init_scale * regime_limit / math.sqrt(max(cond_dim, 1))
        hyper_layers.append(
            HyperLinear.init(
                target_in=layer_in,
                target_out=layer_out,
                cond_dim=cond_dim,
                key=hyper_keys[i],
                init_scale=per_layer_scale,
                pyrox_name=f"hyper_{i}",
            )
        )
    return _GeneratedSiren(  # ty: ignore[invalid-return-type]
        parameter_net=parameter_net,
        siren=siren,
        hyper_layers=hyper_layers,
        in_features=in_features,
        out_features=out_features,
        depth=depth,
    )


__all__ = [
    "AbstractConditioner",
    "AffineModulation",
    "BayesianAffineModulation",
    "BayesianConcatConditioner",
    "BayesianHyperLinear",
    "ConcatConditioner",
    "ConditionedINR",
    "ConditionedRFFNet",
    "FiLM",
    "HyperFourierFeatures",
    "HyperLinear",
    "HyperSIREN",
]


# Silence "imported but unused" — kept for users patching internals.
_unused: tuple[Callable[..., object], ...] = (SirenDense,)


# ---------------------------------------------------------------------------
# Hyper-Fourier features: a parameter net produces (W, b, log_lengthscale)
# ---------------------------------------------------------------------------


class HyperFourierFeatures(PyroxModule):
    r"""Random Fourier features with ``(W, b, log_lengthscale)`` from a parameter net.

    The deterministic counterpart :class:`pyrox.nn.RBFFourierFeatures`
    *samples* its frequencies and lengthscale from priors. This layer
    instead amortises them over a context vector ``z`` via a user-supplied
    ``parameter_net``:

    .. math::

        (W(z), b(z), \log\ell(z)) &= \text{unflatten}(\text{parameter\_net}(z)) \\
        \phi(x; z) &= \sqrt{1/n_{\text{features}}}\;
            \bigl[\cos(W(z)^\top x / \ell(z) + b(z)),\;
                  \sin(W(z)^\top x / \ell(z) + b(z))\bigr]

    The parameter net runs once per call; the generated features are
    reused across all rows of ``x`` — same efficiency trick as
    :class:`HyperLinear`'s shared path.

    The flat output of ``parameter_net(z)`` must have size
    ``in_features * n_features + n_features + 1`` (frequencies, phases,
    log-lengthscale). The layer validates this at construction time by
    invoking ``parameter_net`` on a dummy ``z``.

    Attributes:
        parameter_net: Callable ``(K,) -> (P,)`` producing the flat
            feature parameters from the context. Typically a small MLP
            or any :class:`PyroxModule`.
        in_features: Coordinate dimension (``D_in``).
        n_features: Number of frequency pairs; output dim is
            ``2 * n_features``.
        cond_dim: Context dimension expected by ``parameter_net``.
        pyrox_name: Optional explicit scope name.

    Example:
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
        """Build :class:`HyperFourierFeatures` and validate parameter_net output."""
        if in_features <= 0 or n_features <= 0 or cond_dim <= 0:
            raise ValueError(
                "in_features, n_features, and cond_dim must all be positive; got "
                f"in_features={in_features}, n_features={n_features}, "
                f"cond_dim={cond_dim}."
            )
        expected = in_features * n_features + n_features + 1
        try:
            probe = parameter_net(jnp.zeros((cond_dim,)))  # ty: ignore[call-non-callable]
        except Exception as exc:  # pragma: no cover — re-raised below for clarity
            raise ValueError(
                f"parameter_net failed when called with a dummy z of shape "
                f"({cond_dim},). Make sure parameter_net accepts a 1-D array."
            ) from exc
        if probe.shape != (expected,):
            raise ValueError(
                f"parameter_net(z) must return shape ({expected},) — splits "
                f"into W:({in_features},{n_features}), b:({n_features},), "
                f"log_l:(). Got shape {probe.shape}."
            )
        return cls(
            parameter_net=parameter_net,
            in_features=in_features,
            n_features=n_features,
            cond_dim=cond_dim,
            pyrox_name=pyrox_name,
        )

    def _unpack(self, z: Array) -> tuple[Array, Array, Array]:
        """Split ``parameter_net(z)`` into ``(W, b, log_l)``."""
        flat = self.parameter_net(z)  # ty: ignore[call-non-callable]
        w_size = self.in_features * self.n_features
        flat_W = flat[:w_size]
        b = flat[w_size : w_size + self.n_features]
        log_l = flat[-1]
        W = einops.rearrange(
            flat_W,
            "(d n) -> d n",
            d=self.in_features,
            n=self.n_features,
        )
        return W, b, log_l

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
        squeeze = x.ndim == 1
        if squeeze:
            x = einops.rearrange(x, "d -> 1 d")

        if z.ndim == 1:
            W, b, log_l = self._unpack(z)
            # (N, D) @ (D, n) / scalar  -> (N, n)
            inv_l = jnp.exp(-log_l)
            proj = (x @ W) * inv_l + b
        else:
            # Per-sample features. Vectorise the unpack across the N axis.
            W_all, b_all, log_l_all = jax.vmap(self._unpack)(z)
            inv_l = jnp.exp(-log_l_all)  # (N,)
            # (N, D) and (N, D, n) -> (N, n)
            proj = einops.einsum(W_all, x, "n d k, n d -> n k") * inv_l[:, None]
            proj = proj + b_all

        scale = jnp.sqrt(1.0 / self.n_features)
        out = scale * jnp.concatenate([jnp.cos(proj), jnp.sin(proj)], axis=-1)
        return out[0] if squeeze else out


class ConditionedRFFNet(PyroxModule):
    """Conditional analogue of :class:`pyrox.nn.RandomKitchenSinks`.

    Composes a :class:`HyperFourierFeatures` feature map with a learnable
    linear readout. The full forward is

    .. math::

        y(x; z) = \\phi(x; z)\\, \\beta + b_{\\text{out}}

    where :math:`\\phi(x; z)` is the ``HyperFourierFeatures`` output and
    ``(beta, b_out)`` are the readout's deterministic weights. For the
    Bayesian variant, wrap ``readout`` in a ``DenseReparameterization`` and
    move the priors there — this composite stays minimal.

    Attributes:
        feat: A :class:`HyperFourierFeatures` instance.
        readout: ``eqx.nn.Linear`` mapping ``2 * n_features -> out_features``.
        pyrox_name: Optional explicit scope name.

    Example:
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
        key: PRNGKeyArray,
        pyrox_name: str | None = None,
    ) -> ConditionedRFFNet:
        """Build :class:`ConditionedRFFNet` with a default linear readout."""
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
        squeeze = phi.ndim == 1
        if squeeze:
            phi = einops.rearrange(phi, "d -> 1 d")
        out = jax.vmap(self.readout)(phi)
        return out[0] if squeeze else out
