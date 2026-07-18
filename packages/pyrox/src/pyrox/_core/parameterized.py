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
                f"parameter {name!r} not registered; call register_param "
                "first. If this module previously worked and was then "
                "rebuilt by a functional pytree update (eqx.tree_at, "
                "eqx.apply_updates, jit/filter_jit with the module as an "
                "argument, flatten/unflatten, or checkpoint load), note "
                "that reconstruction skips __init__/setup(), so the "
                "rebuilt copy has an empty registry — keep the original "
                "instance for probabilistic calls, or reconstruct via "
                "__init__ so setup() re-registers its parameters."
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
            self._reserve_aux(name, ("_loc",))
            return self._guide_delta(name, entry)
        if guide == "normal":
            self._reserve_aux(name, ("_loc", "_scale"))
            return self._guide_normal(name, entry)
        raise NotImplementedError(
            f"guide_type {guide!r} is not yet supported at the "
            "get_param level; materialize via a dedicated guide layer."
        )

    def _reserve_aux(self, name: str, suffixes: tuple[str, ...]) -> None:
        """Fail loudly if a guide's auxiliary site name collides with a
        user-registered parameter.

        The ``normal`` / ``delta`` guides materialize backing sites named
        ``{name}_loc`` (and ``{name}_scale``). If the user also registered a
        parameter with that exact name, the two would share a fully-qualified
        site name and silently clobber each other via the per-call cache (or
        trip NumPyro's duplicate-site check). Reject it at guide time with an
        actionable message rather than let it corrupt inference.
        """
        params = self._state().params
        for suffix in suffixes:
            aux = f"{name}{suffix}"
            if aux in params:
                raise ValueError(
                    f"the guide for parameter {name!r} needs an auxiliary "
                    f"site named {aux!r}, but a parameter with that name is "
                    "already registered; rename one of them to avoid the "
                    "collision."
                )

    def _guide_delta(self, name: str, entry: _Entry) -> Any:
        """Point-estimate (MAP) guide — a constrained param replayed as a Delta.

        Mirrors `numpyro.infer.autoguide.AutoDelta`: the value is a
        `numpyro.param` in the **prior's** support, wrapped in a
        `numpyro.distributions.Delta` **sample** site. NumPyro's ``replay``
        handler conditions the model only on guide *sample* sites, so a bare
        param under the latent's name is invisible to SVI and raises
        ``RuntimeError: Site ... must be sampled in trace``.

        Support and event rank are taken from the resolved prior — matching
        the model site — rather than from the registered ``constraint``:

        * The point estimate must live in the model prior's support, or
          ``replay`` evaluates the prior's log-density outside its domain and
          SVI diverges (a positive prior on an unconstrained-registered param
          would otherwise let ``{name}_loc`` slide to <= 0).
        * The Delta's ``event_dim`` is the prior's, so batched priors keep
          their plate dimensions instead of collapsing the whole value into a
          single event (which would mis-expand under ``numpyro.plate`` /
          block subsampling of the backing param).
        """
        prior = entry.prior
        if callable(prior) and not isinstance(prior, dist.Distribution):
            prior = prior(self)
        if isinstance(prior, dist.Distribution):
            support = prior.support
            event_dim = prior.event_dim
        else:  # non-distribution prior (deterministic value): fall back.
            support = entry.constraint
            event_dim = 0
        loc = self.pyrox_param(
            f"{name}_loc",
            entry.init_value,
            constraint=support,
            event_dim=event_dim,
        )
        loc = jnp.asarray(loc)
        return self.pyrox_sample(name, dist.Delta(loc, event_dim=event_dim))

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
