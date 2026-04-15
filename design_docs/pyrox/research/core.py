"""pyrox._core reference implementation.

Minimal, self-contained implementation of all core abstractions and
key layer families. Executable specification of the design doc.

Covers:
- Layer 0: PyroxModule, PyroxParam, PyroxSample, Parameterized, _Context
- pyrox.nn layers: DenseVariational, MCDropout, RandomFourierFeatures
- Parameterized GP kernel example: RBFKernel

NOT production code — for design exploration and testing only.

Formerly: pyrox-nn/research/base.py
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
from jaxtyping import Array, Float, PRNGKeyArray

# ─── Layer 0: Core Abstractions (pyrox._core) ───────────────────────


class _Context:
    """Per-call cache preventing duplicate sample sites."""

    def __init__(self):
        self._cache: dict[str, Any] = {}
        self._depth: int = 0

    def __enter__(self):
        self._depth += 1
        return self

    def __exit__(self, *args):
        self._depth -= 1
        if self._depth == 0:
            self._cache.clear()

    def get(self, name: str) -> Any | None:
        return self._cache.get(name)

    def set(self, name: str, value: Any) -> None:
        if self._depth > 0:
            self._cache[name] = value


class PyroxModule(eqx.Module):
    """Base class bridging Equinox modules with NumPyro tracing.

    Provides pyrox_param() and pyrox_sample() methods that register
    NumPyro sites during model tracing.
    """

    # Class-level context registry keyed by id(instance)
    _contexts: dict[int, _Context] = {}

    def _get_context(self) -> _Context:
        key = id(self)
        if key not in PyroxModule._contexts:
            PyroxModule._contexts[key] = _Context()
        return PyroxModule._contexts[key]

    def _pyro_fullname(self, name: str) -> str:
        return name  # simplified — no hierarchy

    def pyrox_param(
        self,
        name: str,
        init_value: Array,
        *,
        constraint: Any = dist.constraints.real,
    ) -> Array:
        """Register/retrieve a deterministic learnable parameter."""
        ctx = self._get_context()
        fullname = self._pyro_fullname(name)
        cached = ctx.get(fullname)
        if cached is not None:
            return cached
        value = numpyro.param(fullname, init_value, constraint=constraint)
        ctx.set(fullname, value)
        return value

    def pyrox_sample(
        self,
        name: str,
        prior: dist.Distribution | Callable,
    ) -> Array:
        """Register/retrieve a random variable."""
        ctx = self._get_context()
        fullname = self._pyro_fullname(name)
        cached = ctx.get(fullname)
        if cached is not None:
            return cached
        if callable(prior) and not isinstance(prior, dist.Distribution):
            prior = prior(self)
        value = numpyro.sample(fullname, prior)
        ctx.set(fullname, value)
        return value


class Parameterized(PyroxModule):
    """GP-style base class with register_param, set_prior, autoguide.

    Supports model/guide mode switching for SVI workflows.
    """

    # Class-level registry keyed by id(instance)
    _registry: dict[int, dict] = {}

    def _get_registry(self) -> dict:
        key = id(self)
        if key not in Parameterized._registry:
            Parameterized._registry[key] = {
                "params": {},
                "priors": {},
                "guides": {},
                "mode": "model",
            }
        return Parameterized._registry[key]

    def register_param(
        self,
        name: str,
        init_value: Array,
        constraint: Any = dist.constraints.real,
    ) -> None:
        reg = self._get_registry()
        reg["params"][name] = {"init_value": init_value, "constraint": constraint}

    def set_prior(self, name: str, distribution: dist.Distribution) -> None:
        reg = self._get_registry()
        reg["priors"][name] = distribution

    def autoguide(self, name: str, guide_type: str = "normal") -> None:
        reg = self._get_registry()
        reg["guides"][name] = guide_type

    def set_mode(self, mode: str) -> None:
        reg = self._get_registry()
        reg["mode"] = mode

    def get_param(self, name: str) -> Array:
        reg = self._get_registry()
        param_info = reg["params"][name]
        prior = reg["priors"].get(name)

        if prior is not None and reg["mode"] == "model":
            return self.pyrox_sample(name, prior)
        return self.pyrox_param(
            name, param_info["init_value"], constraint=param_info["constraint"]
        )

    def load_pyro_samples(self) -> None:
        reg = self._get_registry()
        for name in reg["params"]:
            self.get_param(name)


# ─── pyrox.nn: Probabilistic Layers ─────────────────────────────────


class DenseVariational(PyroxModule):
    """Bayesian dense layer with Normal prior on weights.

    Registers weight and bias as NumPyro sample sites with
    N(0, prior_std) priors.
    """

    output_dim: int
    prior_std: float
    use_bias: bool

    def __init__(self, output_dim: int, prior_std: float = 1.0, use_bias: bool = True, *, key: PRNGKeyArray):
        self.output_dim = output_dim
        self.prior_std = prior_std
        self.use_bias = use_bias

    def __call__(self, x: Float[Array, "N D"]) -> Float[Array, "N H"]:
        input_dim = x.shape[-1]
        W = self.pyrox_sample(
            "weight",
            dist.Normal(0, self.prior_std)
            .expand([input_dim, self.output_dim])
            .to_event(2),
        )
        out = x @ W
        if self.use_bias:
            b = self.pyrox_sample(
                "bias",
                dist.Normal(0, self.prior_std).expand([self.output_dim]).to_event(1),
            )
            out = out + b
        return out


class MCDropout(eqx.Module):
    """Monte Carlo dropout — always stochastic.

    Unlike standard dropout, MCDropout remains active at inference
    time to provide uncertainty estimates via multiple forward passes.
    """

    keep_prob: float

    def __init__(self, keep_prob: float = 0.5):
        self.keep_prob = keep_prob

    def __call__(self, x: Array, *, key: PRNGKeyArray) -> Array:
        mask = jr.bernoulli(key, self.keep_prob, x.shape)
        return jnp.where(mask, x / self.keep_prob, 0.0)


class RandomFourierFeatures(eqx.Module):
    """RBF kernel approximation via random Fourier features.

    phi(x) = sqrt(2/D) [cos(Wx + b)]
    where W ~ N(0, 1/l^2), b ~ Uniform[0, 2*pi].

    Output dim = 2 * n_features (cos + sin components).
    """

    omega: Float[Array, "D d"]  # random frequencies
    bias: Float[Array, " D"]  # random phases
    n_features: int
    lengthscale: float

    def __init__(
        self,
        n_features: int,
        input_dim: int,
        lengthscale: float = 1.0,
        *,
        key: PRNGKeyArray,
    ):
        k1, k2 = jr.split(key)
        self.n_features = n_features
        self.lengthscale = lengthscale
        self.omega = jr.normal(k1, (n_features, input_dim)) / lengthscale
        self.bias = jr.uniform(k2, (n_features,), minval=0, maxval=2 * jnp.pi)

    def __call__(self, x: Float[Array, "N d"]) -> Float[Array, "N 2D"]:
        proj = x @ self.omega.T + self.bias  # (N, D)
        scale = jnp.sqrt(2.0 / self.n_features)
        return scale * jnp.concatenate([jnp.cos(proj), jnp.sin(proj)], axis=-1)


# ─── pyrox.gp: Parameterized GP Kernels ─────────────────────────────


class RBFKernel(Parameterized):
    """Parameterized RBF kernel with learnable variance and lengthscale."""

    def setup(self):
        self.register_param(
            "variance", jnp.array(1.0), constraint=dist.constraints.positive
        )
        self.register_param(
            "lengthscale", jnp.array(1.0), constraint=dist.constraints.positive
        )
        self.set_prior("variance", dist.LogNormal(0.0, 1.0))
        self.set_prior("lengthscale", dist.LogNormal(0.0, 1.0))

    def __call__(
        self, X1: Float[Array, "N d"], X2: Float[Array, "M d"]
    ) -> Float[Array, "N M"]:
        v = self.get_param("variance")
        l = self.get_param("lengthscale")
        sq_dist = jnp.sum((X1[:, None] - X2[None, :]) ** 2 / l**2, axis=-1)
        return v * jnp.exp(-0.5 * sq_dist)
