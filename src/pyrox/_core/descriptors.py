"""Declarative wrappers for NumPyro param and sample sites."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple


class PyroxParam(NamedTuple):
    """Declarative wrapper for a deterministic parameter site.

    Carries the init value, constraint, and optional event dimension for
    registration with :meth:`PyroxModule.pyrox_param`. Usable inline inside
    ``__call__`` or as a class-variable default.

    Attributes:
        init_value: Initial value, lazy callable, or ``None`` to look up an
            existing param site by name.
        constraint: NumPyro constraint on the parameter domain; ``None``
            means unconstrained real.
        event_dim: Number of rightmost event dimensions, or ``None``.
    """

    init_value: Any = None
    constraint: Any = None
    event_dim: int | None = None


@dataclass(frozen=True)
class PyroxSample:
    """Declarative wrapper for a random sample site.

    The ``prior`` is either a :class:`numpyro.distributions.Distribution` or
    a callable ``(self) -> Distribution`` for dependent priors where the
    prior references other sampled values on the same module.
    """

    prior: Any | Callable[[Any], Any]
