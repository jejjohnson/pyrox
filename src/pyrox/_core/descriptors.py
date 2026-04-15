"""Declarative wrappers for NumPyro param and sample sites."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple


class PyroxParam(NamedTuple):
    """Lightweight metadata container for a parameter site.

    Bundles init value, constraint, and optional event dimension as a
    single descriptor. This type is a plain value object — higher-level
    APIs that consume it (for example a future declarative registration
    helper) live elsewhere; :meth:`PyroxModule.pyrox_param` takes the
    fields individually as keyword arguments.

    Attributes:
        init_value: Initial value, lazy callable, or ``None`` to look up
            an existing param site by name.
        constraint: NumPyro constraint on the parameter domain; ``None``
            means unconstrained real.
        event_dim: Number of rightmost event dimensions, or ``None``.
    """

    init_value: Any = None
    constraint: Any = None
    event_dim: int | None = None


@dataclass(frozen=True)
class PyroxSample:
    """Lightweight metadata container for a random sample site.

    Wraps the prior — either a :class:`numpyro.distributions.Distribution`
    or a callable ``(self) -> Distribution`` for dependent priors that
    reference other sampled values on the same module. Like
    :class:`PyroxParam`, this is a plain value object; call
    :meth:`PyroxModule.pyrox_sample` with the underlying prior directly.
    """

    prior: Any | Callable[[Any], Any]
