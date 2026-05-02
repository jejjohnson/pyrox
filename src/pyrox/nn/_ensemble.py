"""Ensemble-style layers for ``pyrox.nn``.

* :class:`DenseRank1` — rank-1 ensemble dense layer (Wen et al., 2020;
  Dusenberry et al., 2020). A shared full-rank kernel :math:`W` plus
  per-ensemble-member rank-1 multiplicative perturbations :math:`r_i`,
  :math:`s_i`. Available in two modes via the ``bayesian`` flag:
  deterministic BatchEnsemble (point-estimate :math:`r, s`) and rank-1
  BNN (Gaussian priors on :math:`r, s` centered at per-member inits).
* :class:`LayerNormEnsemble` — per-ensemble-member LayerNorm. Required
  drop-in replacement for ``LayerNorm`` inside BatchEnsemble / Rank1
  architectures.
* :class:`MultiHeadAttentionBE` — multi-head attention with
  BatchEnsemble per-member rank-1 perturbations on each of the four
  Q / K / V / O projections. Output gains a leading ensemble axis.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as dist
from jaxtyping import Array, Float, PRNGKeyArray

from pyrox._core.pyrox_module import PyroxModule, pyrox_method


def _glorot_uniform(
    key: PRNGKeyArray, in_features: int, out_features: int
) -> Float[Array, "D_in D_out"]:
    """Glorot-uniform init for a shared dense kernel."""
    lim = math.sqrt(6.0 / (in_features + out_features))
    return jr.uniform(
        key,
        (in_features, out_features),
        minval=-lim,
        maxval=lim,
    )


def _rs_init(
    key: PRNGKeyArray,
    ensemble_size: int,
    feature_dim: int,
    init_scale: float,
) -> Float[Array, "M D"]:
    """Per-member rank-1-vector init: ``1 + init_scale * N(0, I)``.

    Centered on 1.0 so the rank-1 perturbation is the identity in
    expectation. ``init_scale`` controls per-member diversity.
    """
    return 1.0 + init_scale * jr.normal(key, (ensemble_size, feature_dim))


class DenseRank1(PyroxModule):
    r"""Rank-1 ensemble dense layer.

    Implements the BatchEnsemble (Wen et al., 2020) / rank-1 BNN
    (Dusenberry et al., 2020) parameterization: a single shared kernel
    :math:`W \in \mathbb{R}^{D_\mathrm{in} \times D_\mathrm{out}}` and
    per-member rank-1 multiplicative perturbations
    :math:`s_i \in \mathbb{R}^{D_\mathrm{in}}`,
    :math:`r_i \in \mathbb{R}^{D_\mathrm{out}}` for
    :math:`i = 1, \ldots, M`. The per-member effective weight is

    .. math::

        W_i = (s_i \otimes r_i) \circ W,

    and the efficient forward pass avoids materialising :math:`W_i`:

    .. math::

        y_i = \bigl((x \circ s_i)\, W\bigr) \circ r_i + b_i.

    Two modes via the ``bayesian`` flag:

    * ``bayesian=False`` (default) — BatchEnsemble. :math:`r, s, W, b`
      are all deterministic ``pyrox_param`` sites and per-member
      diversity comes purely from the random initialisation of
      :math:`r_i, s_i`. Use this for ensemble training under a single
      shared SGD trajectory.
    * ``bayesian=True`` — rank-1 BNN. :math:`r, s` are
      ``pyrox_sample`` sites with Normal priors centered at the
      per-member init values; :math:`W, b` remain deterministic. Plug
      into NumPyro's SVI machinery (an ``AutoNormal`` guide on
      ``r, s`` recovers Dusenberry et al., 2020).

    Plate semantics:
        Identical to other pyrox Bayesian dense layers — call this
        layer **outside** ``numpyro.plate("data", ..., subsample_size=...)``
        and only plate the observation likelihood. The model log
        density picks up :math:`\log p(r_i)` and :math:`\log p(s_i)`
        once per layer (not once per example) under the canonical
        pattern.

    Attributes:
        in_features: Input dimension :math:`D_\mathrm{in}`.
        out_features: Output dimension :math:`D_\mathrm{out}`.
        ensemble_size: Number of ensemble members :math:`M`.
        bias: Whether to include a per-member bias.
        bayesian: If ``True``, place Normal priors on :math:`r, s`.
        prior_scale: Std of the Bayesian priors on :math:`r, s`. Only
            used when ``bayesian=True``.
        W_init: Shared kernel init, shape ``(D_in, D_out)``.
        r_init: Per-member output-side init, shape ``(M, D_out)``.
        s_init: Per-member input-side init, shape ``(M, D_in)``.
        pyrox_name: Explicit scope name for NumPyro site registration.

    Example:
        >>> import jax.random as jr
        >>> import jax.numpy as jnp
        >>> from numpyro import handlers
        >>> layer = DenseRank1.init(
        ...     jr.PRNGKey(0),
        ...     in_features=4,
        ...     out_features=2,
        ...     ensemble_size=3,
        ... )
        >>> x = jnp.ones((5, 4))
        >>> with handlers.seed(rng_seed=0):
        ...     y = layer(x)
        >>> y.shape
        (3, 5, 2)

    References:
        Wen, Y., Tran, D., & Ba, J. (2020). *BatchEnsemble: An
        Alternative Approach to Efficient Ensemble and Lifelong
        Learning.* ICLR.

        Dusenberry, M. W., et al. (2020). *Efficient and Scalable
        Bayesian Neural Nets with Rank-1 Factors.* ICML.
    """

    in_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)
    ensemble_size: int = eqx.field(static=True)
    bias: bool = eqx.field(static=True, default=True)
    bayesian: bool = eqx.field(static=True, default=False)
    prior_scale: float = 0.5
    W_init: Float[Array, "D_in D_out"] | None = eqx.field(default=None)
    r_init: Float[Array, "M D_out"] | None = eqx.field(default=None)
    s_init: Float[Array, "M D_in"] | None = eqx.field(default=None)
    pyrox_name: str | None = None

    @classmethod
    def init(
        cls,
        key: PRNGKeyArray,
        in_features: int,
        out_features: int,
        ensemble_size: int,
        *,
        bias: bool = True,
        bayesian: bool = False,
        init_scale: float = 0.5,
        prior_scale: float = 0.5,
        pyrox_name: str | None = None,
    ) -> DenseRank1:
        """Construct a layer with random per-member init vectors."""
        if in_features <= 0 or out_features <= 0 or ensemble_size <= 0:
            raise ValueError(
                "in_features, out_features, ensemble_size must all be > 0; "
                f"got {in_features=}, {out_features=}, {ensemble_size=}."
            )
        if init_scale < 0:
            raise ValueError(f"init_scale must be >= 0; got {init_scale}.")
        # `prior_scale` is only consulted in Bayesian mode — don't reject
        # configs that pass through a sentinel default in deterministic mode.
        if bayesian and prior_scale <= 0:
            raise ValueError(
                f"prior_scale must be > 0 when bayesian=True; got {prior_scale}."
            )
        kw, kr, ks = jr.split(key, 3)
        W_init = _glorot_uniform(kw, in_features, out_features)
        r_init = _rs_init(kr, ensemble_size, out_features, init_scale)
        s_init = _rs_init(ks, ensemble_size, in_features, init_scale)
        return cls(
            in_features=in_features,
            out_features=out_features,
            ensemble_size=ensemble_size,
            bias=bias,
            bayesian=bayesian,
            prior_scale=prior_scale,
            W_init=W_init,
            r_init=r_init,
            s_init=s_init,
            pyrox_name=pyrox_name,
        )

    def __post_init__(self) -> None:
        # Defensive shape checks: the dataclass init lets a user pass
        # raw arrays bypassing init(), so validate here once at module
        # construction time (cheap; not in the hot path).
        if self.W_init is None or self.r_init is None or self.s_init is None:
            raise ValueError(
                "DenseRank1 requires W_init, r_init, s_init. Use "
                "DenseRank1.init(key, ...) to construct from a PRNG key."
            )
        expected_W = (self.in_features, self.out_features)
        expected_r = (self.ensemble_size, self.out_features)
        expected_s = (self.ensemble_size, self.in_features)
        if self.W_init.shape != expected_W:
            raise ValueError(
                f"W_init shape {self.W_init.shape} != expected {expected_W}."
            )
        if self.r_init.shape != expected_r:
            raise ValueError(
                f"r_init shape {self.r_init.shape} != expected {expected_r}."
            )
        if self.s_init.shape != expected_s:
            raise ValueError(
                f"s_init shape {self.s_init.shape} != expected {expected_s}."
            )

    @pyrox_method
    def __call__(
        self, x: Float[Array, "*batch D_in"]
    ) -> Float[Array, "M *batch D_out"]:
        W = self.pyrox_param("W", self.W_init)

        if self.bayesian:
            r = self.pyrox_sample(
                "r",
                dist.Normal(self.r_init, self.prior_scale).to_event(2),
            )
            s = self.pyrox_sample(
                "s",
                dist.Normal(self.s_init, self.prior_scale).to_event(2),
            )
        else:
            r = self.pyrox_param("r", self.r_init)
            s = self.pyrox_param("s", self.s_init)

        # y_i = ((x ∘ s_i) @ W) ∘ r_i + b_i. einsum's `...` lets us keep
        # arbitrary leading batch dims while broadcasting s_i, r_i over
        # them. Avoids materialising the per-member effective kernel
        # W_i = (s_i ⊗ r_i) ∘ W.
        x_scaled = jnp.einsum("...d,md->m...d", x, s)
        h = jnp.einsum("m...d,do->m...o", x_scaled, W)
        # Broadcast r_i over the *batch dims by expanding to (M, *(1,)*B, D_out)
        # where B is the number of batch dims (h.ndim - 2 to drop M and D_out).
        batch_ndim = h.ndim - 2
        r_b = r.reshape(
            (self.ensemble_size,) + (1,) * batch_ndim + (self.out_features,)
        )
        out = h * r_b

        if self.bias:
            b = self.pyrox_param(
                "b",
                jnp.zeros((self.ensemble_size, self.out_features)),
            )
            b_b = b.reshape(
                (self.ensemble_size,) + (1,) * batch_ndim + (self.out_features,)
            )
            out = out + b_b
        return out


class LayerNormEnsemble(PyroxModule):
    r"""Per-ensemble-member LayerNorm.

    Drop-in replacement for ``LayerNorm`` inside BatchEnsemble / Rank1
    architectures. Computes the standard LayerNorm normalisation over
    the trailing feature dimension and applies a *per-member* affine
    transform — each ensemble member :math:`i \in \{1, \ldots, M\}`
    gets its own learnable scale :math:`\gamma_i \in \mathbb{R}^D`
    and bias :math:`\beta_i \in \mathbb{R}^D`:

    .. math::

        \hat{x}_i = \frac{x_i - \mu(x_i)}{\sqrt{\sigma^2(x_i) + \epsilon}},
        \qquad
        y_i = \gamma_i \odot \hat{x}_i + \beta_i,

    where :math:`\mu` and :math:`\sigma^2` are the empirical mean and
    variance over the trailing feature axis (computed independently
    for each member-batch slice). Without per-member scale/bias,
    sharing a single LayerNorm across the ensemble would couple all
    members and erase the diversity introduced by :class:`DenseRank1`
    or any other BatchEnsemble layer upstream.

    Input is expected to carry a leading ensemble axis of size
    ``ensemble_size`` and a trailing feature axis of size
    ``feature_dim``. Any number of intermediate batch / time axes
    are supported and pass through unchanged.

    Attributes:
        ensemble_size: Number of ensemble members :math:`M`.
        feature_dim: Trailing feature dimension :math:`D` over which
            the normalisation is computed.
        eps: Small positive constant added to the variance for
            numerical stability.
        pyrox_name: Explicit scope name for NumPyro site registration.

    Example:
        >>> import jax.numpy as jnp
        >>> from numpyro import handlers
        >>> ln = LayerNormEnsemble(
        ...     ensemble_size=3, feature_dim=4, pyrox_name="ln"
        ... )
        >>> x = jnp.ones((3, 5, 4))  # (M, batch, D)
        >>> with handlers.seed(rng_seed=0):
        ...     y = ln(x)
        >>> y.shape
        (3, 5, 4)
    """

    ensemble_size: int = eqx.field(static=True)
    feature_dim: int = eqx.field(static=True)
    eps: float = eqx.field(static=True, default=1e-5)
    pyrox_name: str | None = eqx.field(static=True, default=None)

    def __post_init__(self) -> None:
        if self.ensemble_size <= 0:
            raise ValueError(f"ensemble_size must be > 0; got {self.ensemble_size}.")
        if self.feature_dim <= 0:
            raise ValueError(f"feature_dim must be > 0; got {self.feature_dim}.")
        if self.eps <= 0:
            raise ValueError(f"eps must be > 0; got {self.eps}.")

    @pyrox_method
    def __call__(self, x: Float[Array, "M *batch D"]) -> Float[Array, "M *batch D"]:
        if x.ndim < 2:
            raise ValueError(
                f"x must have at least 2 dims (M and D); got shape {x.shape}."
            )
        if x.shape[0] != self.ensemble_size:
            raise ValueError(
                f"x.shape[0] = {x.shape[0]} does not match "
                f"ensemble_size = {self.ensemble_size}."
            )
        if x.shape[-1] != self.feature_dim:
            raise ValueError(
                f"x.shape[-1] = {x.shape[-1]} does not match "
                f"feature_dim = {self.feature_dim}."
            )

        scales = self.pyrox_param(
            "scales", jnp.ones((self.ensemble_size, self.feature_dim))
        )
        biases = self.pyrox_param(
            "biases", jnp.zeros((self.ensemble_size, self.feature_dim))
        )

        # Per-slice mean/var over the trailing feature axis.
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        x_hat = (x - mean) * jax.lax.rsqrt(var + self.eps)

        # Broadcast (M, D) scale/bias over the *batch axes between them.
        batch_ndim = x.ndim - 2
        broadcast_shape = (
            (self.ensemble_size,) + (1,) * batch_ndim + (self.feature_dim,)
        )
        return scales.reshape(broadcast_shape) * x_hat + biases.reshape(broadcast_shape)


class _Rank1ProjInit(NamedTuple):
    """Per-projection BatchEnsemble inits used by :class:`MultiHeadAttentionBE`.

    Bundles the four arrays needed for one rank-1 projection
    (:math:`W` shared, :math:`r, s, b` per-member) into a single
    PyTree leaf so the parent module can carry one such NamedTuple
    per Q/K/V/O projection.
    """

    W: Float[Array, "D_in D_out"]
    r: Float[Array, "M D_out"]
    s: Float[Array, "M D_in"]
    b: Float[Array, "M D_out"]


def _init_rank1_proj(
    key: PRNGKeyArray,
    in_features: int,
    out_features: int,
    ensemble_size: int,
    init_scale: float,
) -> _Rank1ProjInit:
    """Glorot-init shared kernel + per-member rank-1 vectors + zero bias."""
    kw, kr, ks = jr.split(key, 3)
    return _Rank1ProjInit(
        W=_glorot_uniform(kw, in_features, out_features),
        r=_rs_init(kr, ensemble_size, out_features, init_scale),
        s=_rs_init(ks, ensemble_size, in_features, init_scale),
        b=jnp.zeros((ensemble_size, out_features)),
    )


def _apply_rank1_proj(
    x: Float[Array, ...],
    proj: _Rank1ProjInit,
    ensemble_size: int,
    bias: bool,
    *,
    has_ensemble: bool,
) -> Float[Array, ...]:
    r"""Apply ``y_i = ((x ⊙ s_i) @ W) ⊙ r_i + b_i`` per ensemble member.

    Two modes selected by the *explicit* ``has_ensemble`` flag rather
    than a shape heuristic — heuristics on ``x.shape[0] == ensemble_size``
    silently mis-classify inputs whose sequence length happens to equal
    the ensemble size (e.g. self-attention with ``T = M``):

    * ``has_ensemble=False``: ``x`` has shape ``(*batch, D_in)`` and the
      output gains a leading ``M`` axis: ``(M, *batch, D_out)``. Use
      this for the Q/K/V projections, whose inputs are un-ensembled.
    * ``has_ensemble=True``: ``x`` already carries an ``M`` leading
      axis (``(M, *batch, D_in)``); the per-member projection flows
      through unchanged. Use this for the O projection after attention
      has already added the ensemble axis.
    """
    if has_ensemble:
        if x.ndim < 2 or x.shape[0] != ensemble_size:
            raise ValueError(
                f"Expected x.shape[0] == ensemble_size ({ensemble_size}) when "
                f"has_ensemble=True; got x.shape = {x.shape}."
            )
        x_scaled = x * proj.s.reshape((ensemble_size,) + (1,) * (x.ndim - 2) + (-1,))
    else:
        x_scaled = jnp.einsum("...d,md->m...d", x, proj.s)
    h = jnp.einsum("m...d,do->m...o", x_scaled, proj.W)
    out = h * proj.r.reshape((ensemble_size,) + (1,) * (h.ndim - 2) + (-1,))
    if bias:
        out = out + proj.b.reshape((ensemble_size,) + (1,) * (out.ndim - 2) + (-1,))
    return out


class MultiHeadAttentionBE(PyroxModule):
    r"""Multi-head attention with BatchEnsemble rank-1 projections.

    Standard scaled-dot-product multi-head attention where each of the
    four linear projections — query, key, value, and output — uses a
    BatchEnsemble parameterisation: a shared full-rank kernel plus
    per-ensemble-member rank-1 multiplicative perturbations. So for
    member :math:`i \in \{1, \ldots, M\}` and projection
    :math:`P \in \{Q, K, V, O\}`,

    .. math::

        W_i^{(P)} = (s_i^{(P)} \otimes r_i^{(P)}) \circ W^{(P)},

    and the attention itself is the usual

    .. math::

        \mathrm{Attn}(Q, K, V) = \mathrm{softmax}\!
            \Bigl(\frac{Q K^\top}{\sqrt{d_k}}\Bigr) V.

    The forward consumes un-ensembled inputs (``query``, ``key``,
    ``value`` of shape ``(T, D)`` / ``(S, D)``), adds the ensemble
    axis when projecting to ``Q``, ``K``, ``V``, runs per-member
    attention in parallel, and returns the per-member output of
    shape ``(M, T, D)``. Equivalent to running ``M`` independent
    attention heads with rank-1 weight perturbations and stacking
    their outputs.

    Plate semantics:
        Same convention as :class:`DenseRank1` and the rest of the
        ``pyrox.nn`` ensemble / Bayesian dense family — call this
        layer **outside** ``numpyro.plate("data", ..., subsample_size=...)``
        and only plate the observation likelihood. All four projections
        register their parameters as ``pyrox_param`` sites; nothing
        about the layer is data-dependent so plate-scaling does not
        come into play unless the user puts the call inside a
        subsampled plate.

    Attributes:
        embed_dim: Total feature dimension :math:`D` of query / key /
            value (must be divisible by ``num_heads``).
        num_heads: Number of attention heads :math:`H`. Each head sees
            ``embed_dim // num_heads`` features.
        ensemble_size: Number of ensemble members :math:`M`.
        bias: Whether each of the four projections includes a
            per-member bias. When ``False``, no bias param sites are
            registered for any of Q / K / V / O.
        q_init / k_init / v_init / o_init: Per-projection
            BatchEnsemble init arrays. Build via :meth:`init`.
        pyrox_name: Explicit scope name for NumPyro site registration.

    Example:
        >>> import jax.random as jr
        >>> import jax.numpy as jnp
        >>> from numpyro import handlers
        >>> mha = MultiHeadAttentionBE.init(
        ...     jr.PRNGKey(0),
        ...     embed_dim=8, num_heads=2, ensemble_size=3,
        ... )
        >>> x = jnp.ones((5, 8))
        >>> with handlers.seed(rng_seed=0):
        ...     y = mha(x, x, x)         # self-attention
        >>> y.shape
        (3, 5, 8)
    """

    embed_dim: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    ensemble_size: int = eqx.field(static=True)
    bias: bool = eqx.field(static=True, default=True)
    pyrox_name: str | None = eqx.field(static=True, default=None)
    q_init: _Rank1ProjInit | None = eqx.field(default=None)
    k_init: _Rank1ProjInit | None = eqx.field(default=None)
    v_init: _Rank1ProjInit | None = eqx.field(default=None)
    o_init: _Rank1ProjInit | None = eqx.field(default=None)

    @classmethod
    def init(
        cls,
        key: PRNGKeyArray,
        embed_dim: int,
        num_heads: int,
        ensemble_size: int,
        *,
        bias: bool = True,
        init_scale: float = 0.5,
        pyrox_name: str | None = None,
    ) -> MultiHeadAttentionBE:
        """Construct an MHA-BE layer with random Q/K/V/O projection inits."""
        if embed_dim <= 0 or num_heads <= 0 or ensemble_size <= 0:
            raise ValueError(
                "embed_dim, num_heads, ensemble_size must all be > 0; "
                f"got {embed_dim=}, {num_heads=}, {ensemble_size=}."
            )
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
            )
        if init_scale < 0:
            raise ValueError(f"init_scale must be >= 0; got {init_scale}.")
        kq, kk, kv, ko = jr.split(key, 4)
        return cls(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ensemble_size=ensemble_size,
            bias=bias,
            pyrox_name=pyrox_name,
            q_init=_init_rank1_proj(
                kq, embed_dim, embed_dim, ensemble_size, init_scale
            ),
            k_init=_init_rank1_proj(
                kk, embed_dim, embed_dim, ensemble_size, init_scale
            ),
            v_init=_init_rank1_proj(
                kv, embed_dim, embed_dim, ensemble_size, init_scale
            ),
            o_init=_init_rank1_proj(
                ko, embed_dim, embed_dim, ensemble_size, init_scale
            ),
        )

    def __post_init__(self) -> None:
        # Validate positive dims first so the divisibility check below can't
        # ZeroDivisionError on `num_heads = 0` (and so manual-construction
        # users bypassing `.init` get the same clear error message as that
        # classmethod's callers).
        if self.embed_dim <= 0 or self.num_heads <= 0 or self.ensemble_size <= 0:
            raise ValueError(
                "embed_dim, num_heads, ensemble_size must all be > 0; "
                f"got {self.embed_dim=}, {self.num_heads=}, "
                f"{self.ensemble_size=}."
            )
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by "
                f"num_heads ({self.num_heads})."
            )
        # Defensive shape checks: the dataclass init lets a user pass raw
        # _Rank1ProjInit tuples bypassing .init(), so validate per-array
        # shapes once at construction time. Mirrors DenseRank1.__post_init__.
        D, M = self.embed_dim, self.ensemble_size
        expected = {"W": (D, D), "r": (M, D), "s": (M, D), "b": (M, D)}
        for proj_name, attr in (
            ("q_init", self.q_init),
            ("k_init", self.k_init),
            ("v_init", self.v_init),
            ("o_init", self.o_init),
        ):
            if attr is None:
                raise ValueError(
                    f"{type(self).__name__} requires {proj_name}. Use "
                    f"{type(self).__name__}.init(key, ...) to construct."
                )
            for arr_name, want in expected.items():
                got = getattr(attr, arr_name).shape
                if got != want:
                    raise ValueError(
                        f"{proj_name}.{arr_name} shape {got} != expected {want}."
                    )

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads

    def _register_proj(self, name: str, init: _Rank1ProjInit) -> _Rank1ProjInit:
        """Register a projection's arrays as ``pyrox_param`` sites.

        Skips the ``b`` site when ``self.bias`` is ``False`` so disabled
        biases don't leak unused params into the SVI parameter store.
        """
        b = (
            self.pyrox_param(f"{name}_b", init.b)
            if self.bias
            else jnp.zeros_like(init.b)
        )
        return _Rank1ProjInit(
            W=self.pyrox_param(f"{name}_W", init.W),
            r=self.pyrox_param(f"{name}_r", init.r),
            s=self.pyrox_param(f"{name}_s", init.s),
            b=b,
        )

    @pyrox_method
    def __call__(
        self,
        query: Float[Array, "T D"],
        key: Float[Array, "S D"],
        value: Float[Array, "S D"],
    ) -> Float[Array, "M T D"]:
        if query.ndim != 2 or query.shape[-1] != self.embed_dim:
            raise ValueError(
                f"query must be (T, embed_dim={self.embed_dim}); got {query.shape}."
            )
        if key.ndim != 2 or key.shape[-1] != self.embed_dim:
            raise ValueError(
                f"key must be (S, embed_dim={self.embed_dim}); got {key.shape}."
            )
        if value.shape != key.shape:
            raise ValueError(
                f"key and value must share shape; got {key.shape} vs {value.shape}."
            )

        # Register the four projections.
        assert self.q_init is not None  # __post_init__ guarantees
        assert self.k_init is not None
        assert self.v_init is not None
        assert self.o_init is not None
        q_proj = self._register_proj("q", self.q_init)
        k_proj = self._register_proj("k", self.k_init)
        v_proj = self._register_proj("v", self.v_init)
        o_proj = self._register_proj("o", self.o_init)

        M = self.ensemble_size
        H = self.num_heads
        d = self.head_dim
        T = query.shape[0]
        S = key.shape[0]

        # Project inputs (no ensemble axis on input → M added on output).
        Q = _apply_rank1_proj(
            query, q_proj, M, self.bias, has_ensemble=False
        )  # (M, T, D)
        K = _apply_rank1_proj(
            key, k_proj, M, self.bias, has_ensemble=False
        )  # (M, S, D)
        V = _apply_rank1_proj(
            value, v_proj, M, self.bias, has_ensemble=False
        )  # (M, S, D)

        # Per-head reshape: (M, *, D) → (M, num_heads, *, head_dim).
        Q = Q.reshape(M, T, H, d).transpose(0, 2, 1, 3)  # (M, H, T, d)
        K = K.reshape(M, S, H, d).transpose(0, 2, 1, 3)  # (M, H, S, d)
        V = V.reshape(M, S, H, d).transpose(0, 2, 1, 3)  # (M, H, S, d)

        scores = jnp.einsum("mhtd,mhsd->mhts", Q, K) / math.sqrt(d)
        weights = jax.nn.softmax(scores, axis=-1)
        attn = jnp.einsum("mhts,mhsd->mhtd", weights, V)  # (M, H, T, d)

        # Concatenate heads back to (M, T, embed_dim).
        attn = attn.transpose(0, 2, 1, 3).reshape(M, T, self.embed_dim)

        # Output projection: input already has the M axis.
        return _apply_rank1_proj(attn, o_proj, M, self.bias, has_ensemble=True)
