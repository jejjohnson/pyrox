"""Parameterized base class for GP-style modules.

Provides per-instance registries for constrained parameters, attached
priors, guide-type metadata, and a ``model`` / ``guide`` mode switch.
Both GP kernels and NN layers that want a structured priors+guides
workflow subclass `Parameterized`.

Pattern C usage:

    class RBFKernel(Parameterized):
        def setup(self):
            self.register_param(
                "variance", jnp.array(1.0),
                constraint=dist.constraints.positive,
            )
            self.set_prior("variance", dist.LogNormal(0.0, 1.0))

        @pyrox_method
        def __call__(self, X1, X2):
            v = self.get_param("variance")
            ...
"""

from __future__ import annotations

import contextlib
import weakref
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

import jax.numpy as jnp
import numpyro.distributions as dist

from .pyrox_module import PyroxModule
from .utils import _biject_to, _is_real_support


GuideType = Literal["delta", "normal"]
Mode = Literal["model", "guide"]
_VALID_GUIDES: frozenset[str] = frozenset({"delta", "normal"})


@dataclass
class _Entry:
    init_value: Any
    constraint: Any = None
    prior: Any | None = None
    guide_type: GuideType = "delta"


@dataclass
class _State:
    params: dict[str, _Entry] = field(default_factory=dict)
    mode: Mode = "model"


class Parameterized(PyroxModule):
    """Shared base for modules with priors, constraints, and mode switching.

    Subclasses typically declare parameters inside `setup`, which is
    invoked automatically after ``__init__`` completes. Use
    `register_param` to declare a parameter, `set_prior` to
    attach a prior, `autoguide` to pick a guide type, and
    `set_mode` to switch between sampling from the prior and
    sampling from the guide.

    Per-instance state (params, priors, guides, mode) lives in a
    class-level registry keyed by ``id(self)``. Cleanup happens via
    `weakref.finalize` when the instance is collected; call
    `_teardown` for explicit cleanup.
    """

    _registry: ClassVar[dict[int, _State]] = {}

    def __post_init__(self) -> None:
        setup = getattr(self, "setup", None)
        if callable(setup):
            setup()

    def _state(self) -> _State:
        key = id(self)
        state = Parameterized._registry.get(key)
        if state is None:
            state = _State()
            Parameterized._registry[key] = state
            with contextlib.suppress(TypeError):
                weakref.finalize(self, Parameterized._registry.pop, key, None)
        return state

    def _entry(self, name: str) -> _Entry:
        entry = self._state().params.get(name)
        if entry is None:
            raise KeyError(
                f"parameter {name!r} not registered; call register_param first"
            )
        return entry

    def register_param(
        self,
        name: str,
        init_value: Any,
        constraint: Any = None,
    ) -> None:
        self._state().params[name] = _Entry(
            init_value=init_value, constraint=constraint
        )

    def set_prior(self, name: str, prior: Any) -> None:
        self._entry(name).prior = prior

    def autoguide(self, name: str, guide_type: GuideType) -> None:
        if guide_type not in _VALID_GUIDES:
            raise ValueError(
                f"guide_type must be one of {sorted(_VALID_GUIDES)!r}, "
                f"got {guide_type!r}"
            )
        self._entry(name).guide_type = guide_type

    def set_mode(self, mode: Mode) -> None:
        if mode not in ("model", "guide"):
            raise ValueError(f"mode must be 'model' or 'guide', got {mode!r}")
        self._state().mode = mode

    def get_param(self, name: str) -> Any:
        entry = self._entry(name)
        state = self._state()
        if state.mode == "model" and entry.prior is not None:
            return self.pyrox_sample(name, entry.prior)
        if state.mode == "guide" and entry.prior is not None:
            return self._guide_param(name, entry)
        return self.pyrox_param(name, entry.init_value, constraint=entry.constraint)

    def load_pyro_samples(self) -> None:
        for name in list(self._state().params):
            self.get_param(name)

    def _teardown(self) -> None:
        Parameterized._registry.pop(id(self), None)
        super()._teardown()

    def _guide_param(self, name: str, entry: _Entry) -> Any:
        guide = entry.guide_type
        if guide == "delta":
            return self._guide_delta(name, entry)
        if guide == "normal":
            return self._guide_normal(name, entry)
        raise NotImplementedError(
            f"guide_type {guide!r} is not yet supported at the "
            "get_param level; materialize via a dedicated guide layer."
        )

    def _guide_delta(self, name: str, entry: _Entry) -> Any:
        """Point-estimate (MAP) guide — a constrained param replayed as a Delta.

        Mirrors `numpyro.infer.autoguide.AutoDelta`: the value is a
        `numpyro.param` in the constraint's support, wrapped in a
        `numpyro.distributions.Delta` **sample** site. NumPyro's ``replay``
        handler conditions the model only on guide *sample* sites, so a bare
        param under the latent's name is invisible to SVI and raises
        ``RuntimeError: Site ... must be sampled in trace``. The Delta's
        ``event_dim`` spans the full parameter shape so its (zero) log-density
        reduces to a scalar, matching the MAP objective.
        """
        loc = self.pyrox_param(
            f"{name}_loc", entry.init_value, constraint=entry.constraint
        )
        loc = jnp.asarray(loc)
        return self.pyrox_sample(name, dist.Delta(loc).to_event(jnp.ndim(loc)))

    def _guide_normal(self, name: str, entry: _Entry) -> Any:
        """Mean-field normal guide in unconstrained space.

        When ``entry.constraint`` is non-trivial, the latent site is a
        ``TransformedDistribution`` wrapping ``Normal(loc, scale)`` with
        the constraint's bijection, so guide draws always land in the
        prior's support. ``loc`` is initialized by the inverse transform
        of ``init_value`` so guide and prior agree at step zero.
        """
        init = jnp.asarray(entry.init_value)
        if _is_real_support(entry.constraint):
            loc = self.pyrox_param(f"{name}_loc", init)
            scale = self.pyrox_param(
                f"{name}_scale",
                jnp.ones_like(init) * 0.1,
                constraint=dist.constraints.positive,
            )
            return self.pyrox_sample(name, dist.Normal(loc, scale))
        transform = _biject_to(entry.constraint)
        # loc AND scale live in unconstrained space. For shape-changing
        # transforms (e.g. StickBreaking for a simplex: K -> K-1) the
        # unconstrained shape differs from ``init``; sizing scale from
        # ``init`` would fail to broadcast against ``transform.inv(init)``.
        unconstrained = transform.inv(init)
        loc = self.pyrox_param(f"{name}_loc", unconstrained)
        scale = self.pyrox_param(
            f"{name}_scale",
            jnp.full_like(unconstrained, 0.1),
            constraint=dist.constraints.positive,
        )
        # Promote the base to the transform's domain event rank so the
        # TransformedDistribution's event structure matches the constrained
        # support (a no-op for element-wise transforms like Exp/Sigmoid).
        base = dist.Normal(loc, scale).to_event(transform.domain.event_dim)
        return self.pyrox_sample(name, dist.TransformedDistribution(base, transform))
