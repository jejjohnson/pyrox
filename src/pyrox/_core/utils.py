"""Core utilities shared across the Equinox/NumPyro bridge."""

from __future__ import annotations

from typing import Any

import numpyro.distributions as dist


def _biject_to(constraint: Any) -> Any:
    """Return the bijective transform from unconstrained to constrained space.

    Thin wrapper over :func:`numpyro.distributions.biject_to`. Kept here so
    the rest of ``pyrox._core`` can depend on a single local import point.
    """
    return dist.biject_to(constraint)


def _is_real_support(support: Any) -> bool:
    """Return ``True`` if ``support`` is the unconstrained real line.

    Used to short-circuit constraint handling when no bijection is required.
    """
    return support is None or support is dist.constraints.real
