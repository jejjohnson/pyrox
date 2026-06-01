"""Deep Variational SSGP — Bayesian wrapper around the geonnax core.

The deterministic ``DeepVSSGPCore`` lives in :mod:`geonnax`; this module
hosts only the pyrox-specific :class:`DeepVSSGP` wrapper that swaps the
core's per-layer ``W_freqs`` / ``W_projs`` / ``lengthscales`` for
``pyrox_sample`` sites with appropriate priors.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from geonnax import DeepVSSGPCore
from jaxtyping import Array, Float

from pyrox._core.pyrox_module import PyroxModule, pyrox_method


class DeepVSSGP(PyroxModule):
    r"""Deep Random Feature Expansion for Variational SSGP (Cutajar et al. 2017).

    A stack of :math:`L` variational SSGP layers, each with random
    spectral frequencies :math:`\Omega_l` and random projection weights
    :math:`W_l`:

    .. math::

        F_0 &= X, \\
        F_{l+1} &= \Phi_l(F_l;\, \Omega_l, \ell_l)\, W_l,
            \quad l = 0, \ldots, L-1, \\
        \Phi_l(F;\, \Omega_l, \ell_l) &=
            \sqrt{1/M}\,
            [\cos(F\,\Omega_l/\ell_l), \sin(F\,\Omega_l/\ell_l)].

    Each layer registers three sample sites:

    - ``layer_{l}.W_freq`` — RFF frequencies, prior :math:`\mathcal{N}(0, 1)`
      (RBF spectral density in lengthscale-1 units).
    - ``layer_{l}.lengthscale`` — kernel lengthscale, prior
      :math:`\mathrm{LogNormal}(\log \ell_{\mathrm{init}}, 1)`.
    - ``layer_{l}.W_proj`` — projection weights, prior
      :math:`\mathcal{N}(0, \sigma_W^2)`.

    Under SVI an
    :class:`~numpyro.infer.autoguide.AutoNormal` learns mean-field
    Gaussian posteriors over all :math:`3L` sites — one MC sample per
    forward pass gives the doubly-stochastic reparameterised ELBO of
    Cutajar et al. (2017).

    At ``depth=1`` this reduces to a single VSSGP layer mapping
    ``in_features -> out_features`` via the RFF basis (same model class
    as :class:`VariationalFourierFeatures` followed by a
    :class:`DenseReparameterization` head). Stacking adds
    non-stationarity at the cost of a non-Gaussian aggregate likelihood
    — the layer-wise marginalisation that makes single-layer SSGP
    closed-form is no longer available, hence the variational
    treatment.

    Attributes:
        in_features: Input dimension :math:`D_{\mathrm{in}}`.
        hidden_features: Inter-layer dimension :math:`D_h` (constant
            across hidden layers).
        out_features: Output dimension :math:`D_{\mathrm{out}}`.
        n_features: Per-layer Fourier-feature pair count :math:`M` (so
            each layer's hidden state is :math:`2M`-dim before
            projection).
        depth: Total number of stacked SSGP layers :math:`L`. Must
            be :math:`\ge 1`.
        init_lengthscale: Prior location for each layer's lengthscale.
        prior_std: Standard deviation of the per-layer projection
            prior :math:`\mathcal{N}(0, \sigma_W^2)`.
        pyrox_name: Explicit scope name for NumPyro site registration.

    Example:
        >>> import jax.random as jr, jax.numpy as jnp
        >>> from numpyro import handlers
        >>> net = DeepVSSGP.init(in_features=2, hidden_features=4,
        ...                       out_features=1, depth=3, n_features=16)
        >>> with handlers.seed(rng_seed=0):
        ...     y = net(jnp.zeros((8, 2)))
        >>> y.shape
        (8, 1)
    """

    core: DeepVSSGPCore
    init_lengthscale: float = 1.0
    prior_std: float = 1.0
    pyrox_name: str | None = None

    # Structural accessors so wrapper exposes the same field surface as before.
    @property
    def in_features(self) -> int:
        return self.core.in_features

    @property
    def hidden_features(self) -> int:
        return self.core.hidden_features

    @property
    def out_features(self) -> int:
        return self.core.out_features

    @property
    def n_features(self) -> int:
        return self.core.n_features

    @property
    def depth(self) -> int:
        return self.core.depth

    @classmethod
    def init(
        cls,
        in_features: int,
        hidden_features: int,
        out_features: int,
        *,
        depth: int,
        n_features: int = 64,
        lengthscale: float = 1.0,
        prior_std: float = 1.0,
        pyrox_name: str | None = None,
    ) -> DeepVSSGP:
        """Construct a :class:`DeepVSSGP`.

        Args:
            in_features: Input dimension. Must be :math:`\\ge 1`.
            hidden_features: Hidden dimension. Must be :math:`\\ge 1`.
            out_features: Output dimension. Must be :math:`\\ge 1`.
            depth: Total stacked SSGP layers (including readout).
                Must be :math:`\\ge 1`.
            n_features: Per-layer Fourier-feature pair count. Must be
                :math:`\\ge 1`.
            lengthscale: Prior location for each layer's lengthscale.
                Must be :math:`> 0`.
            prior_std: Per-layer projection prior standard deviation.
                Must be :math:`> 0`.
            pyrox_name: Optional explicit scope name for NumPyro site
                registration.

        Returns:
            Initialised :class:`DeepVSSGP`.

        Raises:
            ValueError: If ``depth``, any feature dimension, or
                ``n_features`` is :math:`< 1`, or if ``lengthscale`` /
                ``prior_std`` is :math:`\\le 0`.
        """
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}.")
        if n_features < 1:
            raise ValueError(f"n_features must be >= 1, got {n_features}.")
        if lengthscale <= 0:
            raise ValueError(f"lengthscale must be > 0, got {lengthscale}.")
        if prior_std <= 0:
            raise ValueError(f"prior_std must be > 0, got {prior_std}.")
        for name, dim in (
            ("in_features", in_features),
            ("hidden_features", hidden_features),
            ("out_features", out_features),
        ):
            if dim < 1:
                raise ValueError(f"{name} must be >= 1, got {dim}.")
        # The Bayesian wrapper samples every per-layer array on each
        # forward call, so the core's stored arrays are only used as a
        # structural skeleton.  We build it with a dummy PRNG key — its
        # weight values are irrelevant because eqx.tree_at swaps them
        # for sampled values below.
        core = DeepVSSGPCore.init(
            in_features,
            hidden_features,
            out_features,
            depth=depth,
            key=jax.random.PRNGKey(0),
            n_features=n_features,
            lengthscale=lengthscale,
            prior_std=prior_std,
        )
        return cls(
            core=core,
            init_lengthscale=lengthscale,
            prior_std=prior_std,
            pyrox_name=pyrox_name,
        )

    @pyrox_method
    def __call__(self, x: Float[Array, "*batch D_in"]) -> Float[Array, "*batch D_out"]:
        sampled_W_freqs: list[Float[Array, "d_in n_features"]] = []
        sampled_W_projs: list[Float[Array, "d_rff d_out"]] = []
        sampled_lengthscales: list[Float[Array, ""]] = []
        for layer_idx in range(self.core.depth):
            in_dim = (
                self.core.in_features if layer_idx == 0 else self.core.hidden_features
            )
            out_dim = (
                self.core.out_features
                if layer_idx == self.core.depth - 1
                else self.core.hidden_features
            )
            W_freq = self.pyrox_sample(
                f"layer_{layer_idx}.W_freq",
                dist.Normal(0.0, 1.0)
                .expand([in_dim, self.core.n_features])
                .to_event(2),
            )
            ls = self.pyrox_sample(
                f"layer_{layer_idx}.lengthscale",
                dist.LogNormal(jnp.log(jnp.asarray(self.init_lengthscale)), 1.0),
            )
            W_proj = self.pyrox_sample(
                f"layer_{layer_idx}.W_proj",
                dist.Normal(0.0, self.prior_std)
                .expand([2 * self.core.n_features, out_dim])
                .to_event(2),
            )
            sampled_W_freqs.append(W_freq)
            sampled_W_projs.append(W_proj)
            sampled_lengthscales.append(ls)

        sampled_core = eqx.tree_at(
            lambda c: (c.W_freqs, c.W_projs, c.lengthscales),
            self.core,
            (
                sampled_W_freqs,
                sampled_W_projs,
                jnp.stack(sampled_lengthscales),
            ),
        )
        return jax.vmap(sampled_core)(x)


__all__ = ["DeepVSSGP"]
