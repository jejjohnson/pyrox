"""Bayesian Multiplicative Filter Networks.

The deterministic ``FourierNet`` / ``GaborNet`` / ``mfn_forward``
primitives live in `geonnax`; this module hosts the pyrox-specific
Bayesian wrappers ``BayesianFourierNet`` / ``BayesianGaborNet`` that
sample each filter and readout linear from a NumPyro prior.

Composition is used over inheritance: the geonnax cores are plain
``eqx.Module`` subclasses, while the pyrox variants need the
``PyroxModule`` sample-site machinery, and mixing the two via MRO is
brittle. Instead each wrapper holds the deterministic core as a field
and rebuilds it with sampled weights via `eqx.tree_at` before
running the single-example forward under `jax.vmap`.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from geonnax import FourierFilter, FourierNet, GaborFilter, GaborNet
from jaxtyping import Array, Float, PRNGKeyArray
from pyrox._core.pyrox_module import PyroxModule, pyrox_method


def _require_positive(**values: float) -> None:
    """Raise ``ValueError`` if any keyword value is non-positive."""
    for name, v in values.items():
        if v <= 0:
            raise ValueError(f"{name} must be > 0, got {v}.")


def _sample_filter_omega_phi(
    module: PyroxModule,
    i: int,
    f: FourierFilter | GaborFilter,
    prior_std: float,
) -> tuple[Array, Array]:
    """Sample the shared MFN filter parameters ``(Omega, phi)``.

    Both filter families share a Normal prior on the frequency matrix
    ``Omega`` and a Uniform prior on the phase ``phi``.
    """
    Omega = module.pyrox_sample(
        f"filter_{i}.Omega",
        dist.Normal(0.0, prior_std).expand([f.out_features, f.in_features]).to_event(2),
    )
    phi = module.pyrox_sample(
        f"filter_{i}.phi",
        dist.Uniform(-jnp.pi, jnp.pi).expand([f.out_features]).to_event(1),
    )
    return Omega, phi


def _sample_normal_linears(
    module: PyroxModule,
    linears: list[eqx.nn.Linear],
    prior_std: float,
) -> list[eqx.nn.Linear]:
    """Sample ``(W, b)`` for each MFN readout linear from a Normal prior."""
    sampled: list[eqx.nn.Linear] = []
    for i, lin in enumerate(linears):
        W = module.pyrox_sample(
            f"linear_{i}.W",
            dist.Normal(0.0, prior_std)
            .expand([lin.out_features, lin.in_features])
            .to_event(2),
        )
        b_vec = module.pyrox_sample(
            f"linear_{i}.b",
            dist.Normal(0.0, prior_std).expand([lin.out_features]).to_event(1),
        )
        sampled.append(eqx.tree_at(lambda ll: (ll.weight, ll.bias), lin, (W, b_vec)))
    return sampled


class BayesianFourierNet(PyroxModule):
    r"""FourierNet with Bayesian priors on all filter and linear weights.

    A thin subclass of `FourierNet` that overrides ``__call__`` to
    register NumPyro sample sites for every parameter:

    - Per filter *i*: ``filter_{i}.Omega`` and ``filter_{i}.phi``.
    - Per linear *i*: ``linear_{i}.W`` and ``linear_{i}.b``.

    Total number of sites: $4L$ where $L$ is ``depth``.

    Priors:

    - $\Omega_i \sim \mathcal{N}(0, \sigma^2)$ (matrix).
    - $\varphi_i \sim \mathrm{Uniform}(-\pi, \pi)$.
    - $W_i \sim \mathcal{N}(0, \sigma^2)$ (matrix).
    - $b_i \sim \mathcal{N}(0, \sigma^2)$ (vector).

    Attributes:
        prior_std: Prior standard deviation $\\sigma$ for Gaussian
            sites (default 1.0).  Phase sites always use
            $\mathrm{Uniform}(-\pi, \pi)$.
    """

    core: FourierNet
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
        key: PRNGKeyArray,
        freq_scale: float = 256.0,
        prior_std: float = 1.0,
        pyrox_name: str | None = None,
    ) -> BayesianFourierNet:
        """Construct a `BayesianFourierNet`.

        Args mirror `FourierNet.init`, plus:

        Args:
            prior_std: Prior standard deviation for Gaussian sites
                (default 1.0).

        Raises:
            ValueError: If ``prior_std`` is non-positive or any
                `FourierNet.init` validation fails.
        """
        _require_positive(prior_std=prior_std)
        core = FourierNet.init(
            in_features,
            hidden_features,
            out_features,
            depth=depth,
            key=key,
            freq_scale=freq_scale,
        )
        return cls(core=core, prior_std=prior_std, pyrox_name=pyrox_name)

    # Convenience accessors so downstream code that reads structural fields
    # off the wrapper (in_features/depth/etc.) keeps working transparently.
    @property
    def filters(self) -> list[FourierFilter]:
        return self.core.filters

    @property
    def linears(self) -> list[eqx.nn.Linear]:
        return self.core.linears

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
    def depth(self) -> int:
        return self.core.depth

    @pyrox_method
    def __call__(self, x: Float[Array, "N D"]) -> Float[Array, "N O"]:
        """Forward pass with sampled parameters.

        Args:
            x: Input of shape ``(N, D)`` or ``(D,)`` (single point).

        Returns:
            Output of shape ``(N, O)`` or ``(O,)`` if input was 1-D.
        """
        squeeze = x.ndim == 1
        x2d = jnp.atleast_2d(x)

        sampled_filters: list[FourierFilter] = []
        for i, f in enumerate(self.core.filters):
            Omega, phi = _sample_filter_omega_phi(self, i, f, self.prior_std)
            sampled_filters.append(
                eqx.tree_at(lambda ff: (ff.Omega, ff.phi), f, (Omega, phi))
            )

        sampled_linears = _sample_normal_linears(
            self, self.core.linears, self.prior_std
        )

        sampled_core = eqx.tree_at(
            lambda c: (c.filters, c.linears),
            self.core,
            (sampled_filters, sampled_linears),
        )
        out = jax.vmap(sampled_core)(x2d)
        return out[0] if squeeze else out


class BayesianGaborNet(PyroxModule):
    r"""GaborNet with Bayesian priors on all filter and linear weights.

    A thin subclass of `GaborNet` that overrides ``__call__`` to
    register NumPyro sample sites for every parameter:

    - Per filter *i*: ``filter_{i}.Omega``, ``filter_{i}.phi``,
      ``filter_{i}.mu``, and ``filter_{i}.log_gamma``.
    - Per linear *i*: ``linear_{i}.W`` and ``linear_{i}.b``.

    Total number of sites: $6L$ where $L$ is ``depth``.

    Priors:

    - $\Omega_i \sim \mathcal{N}(0, \sigma^2)$ (matrix).
    - $\varphi_i \sim \mathrm{Uniform}(-\pi, \pi)$.
    - $\mu_i \sim \mathrm{Uniform}(\texttt{domain\_low},\texttt{domain\_high})$.
    - $\log\gamma_i \sim \mathcal{N}(0, \sigma^2)$ (log-space).
    - $W_i \sim \mathcal{N}(0, \sigma^2)$ (matrix).
    - $b_i \sim \mathcal{N}(0, \sigma^2)$ (vector).

    Attributes:
        prior_std: Prior standard deviation $\\sigma$ for Gaussian
            and log-gamma sites (default 1.0).
    """

    core: GaborNet
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
        key: PRNGKeyArray,
        domain: tuple[float, float] = (-1.0, 1.0),
        gamma_alpha: float = 6.0,
        gamma_beta: float = 1.0,
        prior_std: float = 1.0,
        pyrox_name: str | None = None,
    ) -> BayesianGaborNet:
        """Construct a `BayesianGaborNet`.

        Args mirror `GaborNet.init`, plus:

        Args:
            prior_std: Prior standard deviation for Gaussian and
                log-gamma sites (default 1.0).

        Raises:
            ValueError: If ``prior_std`` is non-positive or any
                `GaborNet.init` validation fails.
        """
        _require_positive(prior_std=prior_std)
        core = GaborNet.init(
            in_features,
            hidden_features,
            out_features,
            depth=depth,
            key=key,
            domain=domain,
            gamma_alpha=gamma_alpha,
            gamma_beta=gamma_beta,
        )
        return cls(core=core, prior_std=prior_std, pyrox_name=pyrox_name)

    # Structural accessors mirror the previous direct-inheritance API.
    @property
    def filters(self) -> list[GaborFilter]:
        return self.core.filters

    @property
    def linears(self) -> list[eqx.nn.Linear]:
        return self.core.linears

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
    def depth(self) -> int:
        return self.core.depth

    @property
    def domain(self) -> tuple[float, float]:
        return self.core.domain

    @property
    def gamma_alpha(self) -> float:
        return self.core.gamma_alpha

    @property
    def gamma_beta(self) -> float:
        return self.core.gamma_beta

    @pyrox_method
    def __call__(self, x: Float[Array, "N D"]) -> Float[Array, "N O"]:
        """Forward pass with sampled parameters.

        Args:
            x: Input of shape ``(N, D)`` or ``(D,)`` (single point).

        Returns:
            Output of shape ``(N, O)`` or ``(O,)`` if input was 1-D.
        """
        squeeze = x.ndim == 1
        x2d = jnp.atleast_2d(x)
        low, high = self.core.domain

        sampled_filters: list[GaborFilter] = []
        for i, f in enumerate(self.core.filters):
            Omega, phi = _sample_filter_omega_phi(self, i, f, self.prior_std)
            mu = self.pyrox_sample(
                f"filter_{i}.mu",
                dist.Uniform(low, high)
                .expand([f.out_features, f.in_features])
                .to_event(2),
            )
            log_gamma = self.pyrox_sample(
                f"filter_{i}.log_gamma",
                dist.Normal(0.0, self.prior_std).expand([f.out_features]).to_event(1),
            )
            sampled_filters.append(
                eqx.tree_at(
                    lambda ff: (ff.Omega, ff.phi, ff.mu, ff.log_gamma),
                    f,
                    (Omega, phi, mu, log_gamma),
                )
            )

        sampled_linears = _sample_normal_linears(
            self, self.core.linears, self.prior_std
        )

        sampled_core = eqx.tree_at(
            lambda c: (c.filters, c.linears),
            self.core,
            (sampled_filters, sampled_linears),
        )
        out = jax.vmap(sampled_core)(x2d)
        return out[0] if squeeze else out


__all__ = ["BayesianFourierNet", "BayesianGaborNet"]
