"""``f(...)``: R-INLA-style construction of a component bound to a data column (P9).

Sugar over the component constructors, not a formula parser: the
component's name *is* the data column holding each observation's node
index, which is how an `LGM` binds the two.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import kernellib as kl
import numpyro.distributions as dist

from pyrox_lgm._components import (
    AR1,
    BYM2,
    CAR,
    IID,
    RW1,
    RW2,
    SPDE,
    AbstractComponent,
    Besag,
    Generic,
    Leroux,
)


# model name -> (constructor, how it is sized, its hyperparameter keywords)
_MODELS: dict[str, tuple[Callable[..., AbstractComponent], str, dict[str, str]]] = {
    "iid": (IID, "n", {"tau": "tau_prior"}),
    "rw1": (RW1, "n", {"tau": "tau_prior"}),
    "rw2": (RW2, "n", {"tau": "tau_prior"}),
    "ar1": (AR1, "n", {"tau": "tau_prior", "rho": "rho_prior"}),
    "besag": (Besag, "graph", {"tau": "tau_prior"}),
    "bym2": (BYM2, "graph", {"tau": "tau_prior", "phi": "phi_prior"}),
    "car": (CAR, "graph", {"tau": "tau_prior", "rho": "rho_prior"}),
    "leroux": (Leroux, "graph", {"tau": "tau_prior", "rho": "rho_prior"}),
    "spde": (SPDE, "none", {"range_sigma": "prior"}),
    "generic": (Generic, "none", {"tau": "tau_prior"}),
}


def f(
    column: str,
    model: str,
    *,
    n: int | None = None,
    graph: kl.AbstractGraph | None = None,
    hyper: Mapping[str, dist.Distribution] | None = None,
    **kwargs: Any,
) -> AbstractComponent:
    """Build a latent component named after the data column it indexes.

    Args:
        column: The data column with each observation's node index; becomes
            the component's ``name``.
        model: ``"iid"``, ``"rw1"``, ``"rw2"``, ``"ar1"`` (sized by ``n``),
            ``"besag"``, ``"bym2"``, ``"car"``, ``"leroux"`` (on ``graph``),
            ``"spde"`` (``mesh=`` / ``grid=`` in ``kwargs``) or ``"generic"``
            (the structure as the first of ``kwargs``' ``structure=``).
        n: Number of nodes, for the temporal and unstructured models.
        graph: A kernellib graph, for the areal models.
        hyper: Hyperparameter priors by name (``{"tau": ..., "phi": ...}``;
            ``{"range_sigma": ...}`` for an SPDE).
        **kwargs: Passed through to the constructor (``scale_model=``,
            ``cyclic=``, ``alpha=``, ...).

    Returns:
        The component.

    Raises:
        ValueError: For an unknown model, a missing size or graph, or an
            unknown hyperparameter.

    Examples:
        >>> import kernellib as kl
        >>> import pyrox_lgm as lgm
        >>> g = kl.grid_graph((4, 4))
        >>> tau = lgm.PCPrecision(1.0, 0.01)
        >>> region = lgm.f("region", "bym2", graph=g, hyper={"tau": tau})
        >>> region.name, type(region).__name__
        ('region', 'BYM2')
        >>> lgm.f("week", "rw2", n=52, scale_model=True).n_index
        52
    """
    if model not in _MODELS:
        raise ValueError(f"unknown model {model!r}; choose from {sorted(_MODELS)}")
    build, sized_by, hyper_kw = _MODELS[model]
    args: list[Any] = []
    if sized_by == "n":
        if n is None:
            raise ValueError(f"model {model!r} needs n=")
        args.append(n)
    elif sized_by == "graph":
        if graph is None:
            raise ValueError(f"model {model!r} needs graph=")
        args.append(graph)
    elif model == "generic":
        if "structure" not in kwargs:
            raise ValueError("model 'generic' needs structure=")
        args.append(kwargs.pop("structure"))
    for key, prior in (hyper or {}).items():
        if key not in hyper_kw:
            raise ValueError(
                f"model {model!r} has no hyperparameter {key!r}; "
                f"it has {sorted(hyper_kw)}"
            )
        kwargs[hyper_kw[key]] = prior
    return build(*args, name=column, **kwargs)
