"""Bayesian SIREN — sinusoidal network with regime-scaled Normal priors.

The deterministic ``SIREN`` / ``SirenDense`` primitives and the
``SirenLayerSpec`` / ``build_siren_specs`` / ``siren_W_limit`` helpers
all live in :mod:`geonnax`; this module hosts only the pyrox-specific
Bayesian wrapper that swaps each layer's ``W`` / ``b`` for
``pyrox_sample`` sites.
"""

from __future__ import annotations

import math

import einx
import equinox as eqx
import jax.numpy as jnp
import numpyro.distributions as dist
from geonnax import SirenLayerSpec, build_siren_specs, siren_W_limit
from jaxtyping import Array, Float

from pyrox._core.pyrox_module import PyroxModule, pyrox_method


def _require_positive(**values: float) -> None:
    """Raise ``ValueError`` if any keyword value is non-positive."""
    for name, v in values.items():
        if v <= 0:
            raise ValueError(f"{name} must be > 0, got {v}.")


class BayesianSIREN(PyroxModule):
    r"""SIREN with regime-scaled Normal priors on all layer weights.

    Replaces the deterministic weight matrices of :class:`SIREN` with NumPyro
    sample sites.  For layer :math:`i` with Sitzmann Theorem 1 half-width
    :math:`a_i` (the uniform bound used by :class:`SirenDense`):

    .. math::

        W_i \sim \mathcal{N}\!\left(0,\, \sigma_0 \cdot \frac{a_i}{\sqrt{3}}\right),
        \qquad
        b_i \sim \mathcal{N}\!\left(0,\,
            \sigma_0 \cdot \frac{1}{\sqrt{3 \, d_i}}\right),

    where :math:`\sigma_0` is ``prior_std`` and :math:`d_i` is the input
    dimension of layer :math:`i`.  The :math:`a_i / \sqrt{3}` factor makes
    :math:`\operatorname{Var}(W_i)` equal to the variance of Sitzmann's
    :math:`\mathcal{U}(-a_i, a_i)` init exactly, so the Bayesian prior
    preserves the activation variance prescribed by Theorem 1 — avoiding
    the saturated-sine pathology that a flat :math:`\mathcal{N}(0, 1)`
    prior would cause.

    Registered sites: ``{scope}.layer_0.W``, ``{scope}.layer_0.b``, …,
    ``{scope}.layer_{depth-1}.W``, ``{scope}.layer_{depth-1}.b``
    — exactly ``2 · depth`` sites per forward call.

    Attributes:
        specs: Tuple of per-layer specs (static).  Holds each layer's
            ``layer_type``, ``in_features``, ``out_features``, ``omega``,
            and ``c`` — i.e. everything needed to scale the priors.
        in_features: Input dimension.
        hidden_features: Hidden dimension.
        out_features: Output dimension.
        depth: Total layers including readout.  Must be ≥ 2.
        first_omega: Frequency multiplier for the first layer.
        hidden_omega: Frequency multiplier for hidden layers.
        prior_std: Scale factor for the regime-scaled Normal prior (default 1.0).
        pyrox_name: Explicit scope name for NumPyro site registration.

    Example:
        >>> import jax.random as jr, jax.numpy as jnp
        >>> from numpyro import handlers
        >>> net = BayesianSIREN.init(2, 32, 1, depth=3)
        >>> with handlers.seed(rng_seed=0):
        ...     y = net(jnp.zeros((4, 2)))
        >>> y.shape
        (4, 1)
    """

    specs: tuple[SirenLayerSpec, ...] = eqx.field(static=True)
    in_features: int = eqx.field(static=True)
    hidden_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    first_omega: float = eqx.field(static=True)
    hidden_omega: float = eqx.field(static=True)
    prior_std: float = eqx.field(static=True, default=1.0)
    pyrox_name: str | None = eqx.field(static=True, default=None)

    @classmethod
    def init(
        cls,
        in_features: int,
        hidden_features: int,
        out_features: int,
        *,
        depth: int,
        first_omega: float = 30.0,
        hidden_omega: float = 30.0,
        c: float = 6.0,
        prior_std: float = 1.0,
        pyrox_name: str | None = None,
    ) -> BayesianSIREN:
        """Construct a :class:`BayesianSIREN`.

        All weights come from the prior, so no PRNG key is needed at
        construction time — the key enters when sampling inside a
        ``numpyro`` handler (``handlers.seed``, SVI, etc.).

        Args:
            in_features: Input dimension.
            hidden_features: Hidden dimension.
            out_features: Output dimension.
            depth: Total layers including readout.  Must be ≥ 2.
            first_omega: Frequency for the first layer.
            hidden_omega: Frequency for hidden layers.
            c: Theorem-1 constant.
            prior_std: Scale factor for the Normal priors (default 1.0, must be > 0).
            pyrox_name: Optional explicit scope name for NumPyro.

        Returns:
            Initialised :class:`BayesianSIREN`.

        Raises:
            ValueError: If ``depth < 2``, or any of the feature dimensions,
                omegas, ``c``, or ``prior_std`` is non-positive.
        """
        if depth < 2:
            raise ValueError(f"depth must be >= 2 (first + last); got depth={depth}")
        _require_positive(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            first_omega=first_omega,
            hidden_omega=hidden_omega,
            c=c,
            prior_std=prior_std,
        )
        specs = build_siren_specs(
            in_features,
            hidden_features,
            out_features,
            depth,
            first_omega,
            hidden_omega,
            c,
        )
        return cls(
            specs=specs,
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            depth=depth,
            first_omega=first_omega,
            hidden_omega=hidden_omega,
            prior_std=prior_std,
            pyrox_name=pyrox_name,
        )

    @pyrox_method
    def __call__(self, x: Float[Array, "*batch D_in"]) -> Float[Array, "*batch D_out"]:
        """Sample weights from regime-scaled priors and run the forward pass.

        Registers ``layer_{i}.W`` and ``layer_{i}.b`` NumPyro sample sites
        for each layer ``i`` in ``[0, depth)``.

        Args:
            x: Input tensor of shape ``(*batch, in_features)``.

        Returns:
            Output tensor of shape ``(*batch, out_features)``.
        """
        # Normal stddev = (Uniform half-width) / √3 so Var(W) matches
        # Sitzmann's U(-a, a) init exactly.
        inv_sqrt3 = 1.0 / math.sqrt(3.0)
        z = x
        for i, spec in enumerate(self.specs):
            a = siren_W_limit(spec.layer_type, spec.in_features, spec.omega, spec.c)
            w_scale = self.prior_std * a * inv_sqrt3
            b_scale = self.prior_std * inv_sqrt3 / math.sqrt(spec.in_features)
            W = self.pyrox_sample(
                f"layer_{i}.W",
                dist.Normal(0.0, w_scale)
                .expand([spec.in_features, spec.out_features])
                .to_event(2),
            )
            b = self.pyrox_sample(
                f"layer_{i}.b",
                dist.Normal(0.0, b_scale).expand([spec.out_features]).to_event(1),
            )
            pre = einx.dot("... i, i o -> ... o", z, W) + b
            z = pre if spec.layer_type == "last" else jnp.sin(spec.omega * pre)
        return z


__all__ = ["BayesianSIREN"]
