"""Ensemble-style layers for ``pyrox.nn``.

* `DenseRank1` — rank-1 ensemble dense layer (Wen et al., 2020;
  Dusenberry et al., 2020). A shared full-rank kernel $W$ plus
  per-ensemble-member rank-1 multiplicative perturbations $r_i$,
  $s_i$. Available in two modes via the ``bayesian`` flag:
  deterministic BatchEnsemble (point-estimate $r, s$) and rank-1
  BNN (Gaussian priors on $r, s$ centered at per-member inits).
* `LayerNormEnsemble` — per-ensemble-member LayerNorm. Required
  drop-in replacement for ``LayerNorm`` inside BatchEnsemble / Rank1
  architectures.
* `MultiHeadAttentionBE` — multi-head attention with
  BatchEnsemble per-member rank-1 perturbations on each of the four
  Q / K / V / O projections. Output gains a leading ensemble axis.
"""

from __future__ import annotations

import einx
import equinox as eqx
import geonnax
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from geonnax import Rank1ProjInit
from jaxtyping import Array, Float, PRNGKeyArray

from pyrox._core.pyrox_module import PyroxModule, pyrox_method
from pyrox.nn._batching import vmap_over_flat_batch


# Pre-refactor private alias retained for tests / external callers that imported
# the NamedTuple by its old underscored name.
_Rank1ProjInit = Rank1ProjInit


def _vmap_collapsed(core_fn, x: Float[Array, ...]) -> Float[Array, ...]:
    """Apply a single-example ``core_fn`` over arbitrary leading batch dims.

    Delegates the flatten / vmap / restore dance to
    `vmap_over_flat_batch`, then moves the per-example ensemble
    axis ``M`` to the front because the geonnax cores are
    single-example BatchEnsemble layers.
    """
    out = vmap_over_flat_batch(core_fn, x)  # (*batch, M, D_out)
    return einx.id("b... m k -> m b... k", out)


class DenseRank1(PyroxModule):
    r"""Rank-1 ensemble dense layer.

    Implements the BatchEnsemble (Wen et al., 2020) / rank-1 BNN
    (Dusenberry et al., 2020) parameterization: a single shared kernel
    $W \in \mathbb{R}^{D_\mathrm{in} \times D_\mathrm{out}}$ and
    per-member rank-1 multiplicative perturbations
    $s_i \in \mathbb{R}^{D_\mathrm{in}}$,
    $r_i \in \mathbb{R}^{D_\mathrm{out}}$ for
    $i = 1, \ldots, M$. The per-member effective weight is

    $$
    W_i = (s_i \otimes r_i) \circ W,
    $$

    and the efficient forward pass avoids materialising $W_i$:

    $$
    y_i = \bigl((x \circ s_i)\, W\bigr) \circ r_i + b_i.
    $$

    Two modes via the ``bayesian`` flag:

    * ``bayesian=False`` (default) — BatchEnsemble. $r, s, W, b$
      are all deterministic ``pyrox_param`` sites and per-member
      diversity comes purely from the random initialisation of
      $r_i, s_i$. Use this for ensemble training under a single
      shared SGD trajectory.
    * ``bayesian=True`` — rank-1 BNN. $r, s$ are
      ``pyrox_sample`` sites with Normal priors centered at the
      per-member init values; $W, b$ remain deterministic. Plug
      into NumPyro's SVI machinery (an ``AutoNormal`` guide on
      ``r, s`` recovers Dusenberry et al., 2020).

    Plate semantics:
        Identical to other pyrox Bayesian dense layers — call this
        layer **outside** ``numpyro.plate("data", ..., subsample_size=...)``
        and only plate the observation likelihood. The model log
        density picks up $\log p(r_i)$ and $\log p(s_i)$
        once per layer (not once per example) under the canonical
        pattern.

    Attributes:
        in_features: Input dimension $D_\mathrm{in}$.
        out_features: Output dimension $D_\mathrm{out}$.
        ensemble_size: Number of ensemble members $M$.
        bias: Whether to include a per-member bias.
        bayesian: If ``True``, place Normal priors on $r, s$.
        prior_scale: Std of the Bayesian priors on $r, s$. Only
            used when ``bayesian=True``.
        W_init: Shared kernel init, shape ``(D_in, D_out)``.
        r_init: Per-member output-side init, shape ``(M, D_out)``.
        s_init: Per-member input-side init, shape ``(M, D_in)``.
        pyrox_name: Explicit scope name for NumPyro site registration.

    Examples:
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

    core: geonnax.DenseRank1
    bayesian: bool = eqx.field(static=True, default=False)
    prior_scale: float = 0.5
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
        # `prior_scale` is only consulted in Bayesian mode — don't reject
        # configs that pass through a sentinel default in deterministic mode.
        if bayesian and prior_scale <= 0:
            raise ValueError(
                f"prior_scale must be > 0 when bayesian=True; got {prior_scale}."
            )
        # geonnax handles positive-dim / init_scale validation.
        core = geonnax.DenseRank1.init(
            key,
            in_features=in_features,
            out_features=out_features,
            ensemble_size=ensemble_size,
            bias=bias,
            init_scale=init_scale,
        )
        return cls(
            core=core,
            bayesian=bayesian,
            prior_scale=prior_scale,
            pyrox_name=pyrox_name,
        )

    # Convenience read-only attribute access.
    @property
    def in_features(self) -> int:
        return self.core.in_features

    @property
    def out_features(self) -> int:
        return self.core.out_features

    @property
    def ensemble_size(self) -> int:
        return self.core.ensemble_size

    @property
    def bias(self) -> bool:
        return self.core.bias

    @pyrox_method
    def __call__(
        self, x: Float[Array, "*batch D_in"]
    ) -> Float[Array, "M *batch D_out"]:
        W = self.pyrox_param("W", self.core.W)

        if self.bayesian:
            r = self.pyrox_sample(
                "r",
                dist.Normal(self.core.r, self.prior_scale).to_event(2),
            )
            s = self.pyrox_sample(
                "s",
                dist.Normal(self.core.s, self.prior_scale).to_event(2),
            )
        else:
            r = self.pyrox_param("r", self.core.r)
            s = self.pyrox_param("s", self.core.s)

        b = self.pyrox_param("b", self.core.b) if self.bias else self.core.b

        new_core = eqx.tree_at(
            lambda c: (c.W, c.r, c.s, c.b),
            self.core,
            (W, r, s, b),
        )
        return _vmap_collapsed(new_core, x)


class LayerNormEnsemble(PyroxModule):
    r"""Per-ensemble-member LayerNorm.

    Drop-in replacement for ``LayerNorm`` inside BatchEnsemble / Rank1
    architectures. Computes the standard LayerNorm normalisation over
    the trailing feature dimension and applies a *per-member* affine
    transform — each ensemble member $i \in \{1, \ldots, M\}$
    gets its own learnable scale $\gamma_i \in \mathbb{R}^D$
    and bias $\beta_i \in \mathbb{R}^D$:

    $$
    \hat{x}_i = \frac{x_i - \mu(x_i)}{\sqrt{\sigma^2(x_i) + \epsilon}},
    \qquad
    y_i = \gamma_i \odot \hat{x}_i + \beta_i,
    $$

    where $\mu$ and $\sigma^2$ are the empirical mean and
    variance over the trailing feature axis (computed independently
    for each member-batch slice). Without per-member scale/bias,
    sharing a single LayerNorm across the ensemble would couple all
    members and erase the diversity introduced by `DenseRank1`
    or any other BatchEnsemble layer upstream.

    Input is expected to carry a leading ensemble axis of size
    ``ensemble_size`` and a trailing feature axis of size
    ``feature_dim``. Any number of intermediate batch / time axes
    are supported and pass through unchanged.

    Attributes:
        ensemble_size: Number of ensemble members $M$.
        feature_dim: Trailing feature dimension $D$ over which
            the normalisation is computed.
        eps: Small positive constant added to the variance for
            numerical stability.
        pyrox_name: Explicit scope name for NumPyro site registration.

    Examples:
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

    core: geonnax.LayerNormEnsemble
    pyrox_name: str | None = eqx.field(static=True, default=None)

    def __init__(
        self,
        ensemble_size: int,
        feature_dim: int,
        eps: float = 1e-5,
        pyrox_name: str | None = None,
        *,
        core: geonnax.LayerNormEnsemble | None = None,
    ) -> None:
        # Keep the prior call signature (positional dims, kw eps / pyrox_name)
        # so existing callers don't need to know about the geonnax core
        # construction.
        if core is None:
            # geonnax validates positive ensemble_size / feature_dim / eps.
            core = geonnax.LayerNormEnsemble.init(
                ensemble_size=ensemble_size,
                feature_dim=feature_dim,
                eps=eps,
            )
        self.core = core
        self.pyrox_name = pyrox_name

    @property
    def ensemble_size(self) -> int:
        return self.core.ensemble_size

    @property
    def feature_dim(self) -> int:
        return self.core.feature_dim

    @property
    def eps(self) -> float:
        return self.core.eps

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

        scales = self.pyrox_param("scales", self.core.scales)
        biases = self.pyrox_param("biases", self.core.biases)

        new_core = eqx.tree_at(
            lambda c: (c.scales, c.biases),
            self.core,
            (scales, biases),
        )

        # Core takes (M, D) → (M, D); collapse arbitrary intermediate batch
        # dims to a single leading axis, vmap over it, then restore. The
        # ensemble axis stays at position 0 because the core is per-member.
        batch_shape = x.shape[1:-1]
        flat = einx.id("m b... d -> m (b...) d", x)  # (M, B, D)
        # vmap over the B axis (position 1 on both input and output).
        out = jax.vmap(new_core, in_axes=1, out_axes=1)(flat)  # (M, B, D)
        return einx.id("m (b...) d -> m b... d", out, b=batch_shape)


class MultiHeadAttentionBE(PyroxModule):
    r"""Multi-head attention with BatchEnsemble rank-1 projections.

    Standard scaled-dot-product multi-head attention where each of the
    four linear projections — query, key, value, and output — uses a
    BatchEnsemble parameterisation: a shared full-rank kernel plus
    per-ensemble-member rank-1 multiplicative perturbations. So for
    member $i \in \{1, \ldots, M\}$ and projection
    $P \in \{Q, K, V, O\}$,

    $$
    W_i^{(P)} = (s_i^{(P)} \otimes r_i^{(P)}) \circ W^{(P)},
    $$

    and the attention itself is the usual

    $$
    \mathrm{Attn}(Q, K, V) = \mathrm{softmax}\!
        \Bigl(\frac{Q K^\top}{\sqrt{d_k}}\Bigr) V.
    $$

    The forward consumes un-ensembled inputs (``query``, ``key``,
    ``value`` of shape ``(T, D)`` / ``(S, D)``), adds the ensemble
    axis when projecting to ``Q``, ``K``, ``V``, runs per-member
    attention in parallel, and returns the per-member output of
    shape ``(M, T, D)``. Equivalent to running ``M`` independent
    attention heads with rank-1 weight perturbations and stacking
    their outputs.

    Plate semantics:
        Same convention as `DenseRank1` and the rest of the
        ``pyrox.nn`` ensemble / Bayesian dense family — call this
        layer **outside** ``numpyro.plate("data", ..., subsample_size=...)``
        and only plate the observation likelihood. All four projections
        register their parameters as ``pyrox_param`` sites; nothing
        about the layer is data-dependent so plate-scaling does not
        come into play unless the user puts the call inside a
        subsampled plate.

    Attributes:
        embed_dim: Total feature dimension $D$ of query / key /
            value (must be divisible by ``num_heads``).
        num_heads: Number of attention heads $H$. Each head sees
            ``embed_dim // num_heads`` features.
        ensemble_size: Number of ensemble members $M$.
        bias: Whether each of the four projections includes a
            per-member bias. When ``False``, no bias param sites are
            registered for any of Q / K / V / O.
        q_init / k_init / v_init / o_init: Per-projection
            BatchEnsemble init arrays. Build via `init`.
        pyrox_name: Explicit scope name for NumPyro site registration.

    Examples:
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

    core: geonnax.MultiHeadAttentionBE
    pyrox_name: str | None = eqx.field(static=True, default=None)

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
        # geonnax validates positive dims, divisibility, init_scale.
        core = geonnax.MultiHeadAttentionBE.init(
            key,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ensemble_size=ensemble_size,
            bias=bias,
            init_scale=init_scale,
        )
        return cls(core=core, pyrox_name=pyrox_name)

    @property
    def embed_dim(self) -> int:
        return self.core.embed_dim

    @property
    def num_heads(self) -> int:
        return self.core.num_heads

    @property
    def ensemble_size(self) -> int:
        return self.core.ensemble_size

    @property
    def bias(self) -> bool:
        return self.core.bias

    @property
    def head_dim(self) -> int:
        return self.core.head_dim

    # Maintain the pre-refactor attribute names for any external accessors.
    @property
    def q_init(self) -> Rank1ProjInit:
        return self.core.q_proj

    @property
    def k_init(self) -> Rank1ProjInit:
        return self.core.k_proj

    @property
    def v_init(self) -> Rank1ProjInit:
        return self.core.v_proj

    @property
    def o_init(self) -> Rank1ProjInit:
        return self.core.o_proj

    def _register_proj(self, name: str, init: Rank1ProjInit) -> Rank1ProjInit:
        """Register a projection's arrays as ``pyrox_param`` sites.

        Skips the ``b`` site when ``self.bias`` is ``False`` so disabled
        biases don't leak unused params into the SVI parameter store.
        """
        b = (
            self.pyrox_param(f"{name}_b", init.b)
            if self.bias
            else jnp.zeros_like(init.b)
        )
        return Rank1ProjInit(
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
        # Register the four projections.
        q_proj = self._register_proj("q", self.core.q_proj)
        k_proj = self._register_proj("k", self.core.k_proj)
        v_proj = self._register_proj("v", self.core.v_proj)
        o_proj = self._register_proj("o", self.core.o_proj)

        # Swap all four projections into the core; geonnax's __call__
        # validates query/key/value shapes and runs the per-member
        # attention. The core forward matches the pyrox call signature
        # (un-ensembled `(T, D)` / `(S, D)` in → `(M, T, D)` out), so no
        # vmap is needed here — the ensemble axis is intrinsic to the
        # core, not a data batch.
        new_core = eqx.tree_at(
            lambda c: (c.q_proj, c.k_proj, c.v_proj, c.o_proj),
            self.core,
            (q_proj, k_proj, v_proj, o_proj),
        )
        return new_core(query, key, value)
