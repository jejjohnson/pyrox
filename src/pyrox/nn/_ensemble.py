"""Ensemble-style dense layers for ``pyrox.nn``.

* :class:`DenseRank1` — rank-1 ensemble dense layer (Wen et al., 2020;
  Dusenberry et al., 2020). A shared full-rank kernel :math:`W` plus
  per-ensemble-member rank-1 multiplicative perturbations :math:`r_i`,
  :math:`s_i`. Available in two modes via the ``bayesian`` flag:
  deterministic BatchEnsemble (point-estimate :math:`r, s`) and rank-1
  BNN (Gaussian priors on :math:`r, s` centered at per-member inits).
"""

from __future__ import annotations

import math

import equinox as eqx
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
