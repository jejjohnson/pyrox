"""NumPyro faces of an LGM: sampled components and collapsed factors.

`sample_component` is what `AbstractComponent.sample` runs: it lets any
component sit inside an ordinary NumPyro model and be sampled by NUTS, with or
without ``inla()``. `lgm_factor` and `lgm_laplace_factor` integrate the whole
latent field out instead, so that NUTS only sees the hyperparameters.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import gaussx as gx
import jax.numpy as jnp
import numpyro
from jaxtyping import Array, ArrayLike, Float, Int
from pyrox.likelihoods import AbstractObservation
from pyrox_gauss import GMRFDistribution


if TYPE_CHECKING:
    from pyrox_lgm._components._base import AbstractComponent
    from pyrox_lgm._model import LGMPrior


def sample_component(
    component: AbstractComponent,
    index: Int[ArrayLike, " n_obs"] | None = None,
    *,
    soft_constraint_scale: float = 1e-2,
) -> Array:
    r"""Sample a component's hyperparameters and field inside a NumPyro model.

    Each hyperparameter ``k`` is the site ``f"{name}_{k}"`` with its prior
    from `theta_spec`; the field is the site ``name``, drawn from
    `prior(theta, constraint="soft")`: a tight Gaussian
    $V^\top x \sim \mathcal N(0, s^2 I)$ on the null space of an intrinsic
    field, since NUTS cannot sample a hard constraint.

    The default $s = 10^{-2}$ is looser than gaussx's $10^{-3}$: the
    constraint is a narrow direction NUTS's diagonal mass matrix cannot
    adapt to, so its width sets the step size. On a 12 x 12 BYM2 Poisson
    model, $s = 10^{-3}$ hits the maximum tree depth (1023 leapfrog steps per
    iteration), $10^{-2}$ takes 255 and $10^{-1}$ 30, with the same posterior.
    For a connected field on $n$ nodes, $s = 10^{-2}$ pins the mean to a
    standard deviation of $10^{-2}/\sqrt n$, about Stan's BYM2 soft
    constraint (0.001 on the mean) at $n = 100$.

    Args:
        component: The component.
        index: Node of each observation; ``None`` returns every addressable
            node (padding nodes dropped), through `AbstractComponent.field_at`.
        soft_constraint_scale: $s$ above.

    Returns:
        The field at ``index``.

    Examples:
        >>> import jax
        >>> from numpyro import handlers
        >>> import pyrox_lgm as lgm
        >>> rw1 = lgm.RW1(10, name="trend")
        >>> tr = handlers.trace(handlers.seed(rw1.sample, 0)).get_trace()
        >>> list(tr)
        ['trend_tau', 'trend']
        >>> tr["trend"]["value"].shape
        (10,)
    """
    theta = {
        k: jnp.asarray(numpyro.sample(f"{component.name}_{k}", prior))
        for k, (prior, _) in component.theta_spec().items()
    }
    gmrf = component.prior(
        theta, constraint="soft", soft_constraint_scale=soft_constraint_scale
    )
    x = jnp.asarray(numpyro.sample(component.name, GMRFDistribution(gmrf)))
    return component.field_at(x, index)


def lgm_factor(
    name: str,
    prior: LGMPrior,
    theta: Mapping[str, ArrayLike],
    data: Mapping[str, ArrayLike],
    noise_var: Float[ArrayLike, ""],
    *,
    projector: gx.SparseOperator | None = None,
) -> None:
    r"""Register an LGM's exact log marginal under Gaussian noise with NumPyro.

    Adds ``numpyro.factor(name, prior.log_marginal(theta, data, noise_var))``,

    $$
    \log p(y \mid \theta)
      = \log \mathcal N(y \mid o,\; A Q(\theta)^{-1} A^\top + \sigma^2 I),
    $$

    the whole latent vector integrated out (`LGMPrior.log_marginal` says what
    $Q^{-1}$ is for an intrinsic component). Registers the one site ``name``:
    the hyperparameters in ``theta`` are the model's own sites, under any
    prior, and the components' hyperpriors are not used.

    Args:
        name: The factor site name.
        prior: The `LGMPrior`.
        theta: Hyperparameter values keyed as in `LGMPrior.theta_spec`.
        data: The observations ``"y"`` and where they are.
        noise_var: Observation noise variance $\sigma^2$.
        projector: ``prior.projector(data)``, built once outside the model;
            required under NUTS for an `SPDE` at locations.

    Examples:
        >>> import numpy as np
        >>> import numpyro
        >>> import numpyro.distributions as dist
        >>> import pyrox_lgm as lgm
        >>> prior = lgm.LGMPrior((lgm.RW1(6, name="t"),))
        >>> data = {"y": np.sin(np.arange(6.0)), "t": np.arange(6)}
        >>> def model():
        ...     theta = {"t.tau": numpyro.sample("t.tau", dist.Gamma(2.0, 1.0))}
        ...     noise_var = numpyro.sample("noise_var", dist.HalfNormal(1.0))
        ...     lgm.lgm_factor("y", prior, theta, data, noise_var)
        >>> seeded = numpyro.handlers.seed(model, 0)
        >>> sorted(numpyro.handlers.trace(seeded).get_trace())
        ['noise_var', 't.tau', 'y']
    """
    value = prior.log_marginal(theta, data, noise_var, projector=projector)
    numpyro.factor(name, value)


def lgm_laplace_factor(
    name: str,
    prior: LGMPrior,
    theta: Mapping[str, ArrayLike],
    likelihood: AbstractObservation,
    data: Mapping[str, ArrayLike],
    *,
    projector: gx.SparseOperator | None = None,
    max_newton: int = 50,
    reject_unconverged: bool = False,
) -> None:
    r"""Register an LGM's Laplace log marginal with NumPyro, for any likelihood.

    Adds the evidence of `LGM.log_marginal` as ``numpyro.factor(name, ...)``,

    $$
    \log p(y \mid \theta) \approx \log p(y \mid \hat x, \theta)
      + \log p(\hat x \mid \theta) - \log p_G(\hat x \mid y, \theta),
    $$

    the whole latent vector integrated out at its mode $\hat x$. Inside NUTS
    over $\theta$, and over any parameter outside the latent Gaussian model
    that ``data`` depends on, this is MCMC with INLA (Gómez-Rubio & Rue,
    2018). The gradient is implicit at the mode, so the factor works under
    NUTS and SVI. Registers the one site ``name``.

    Args:
        name: The factor site name.
        prior: The `LGMPrior`.
        theta: Hyperparameter values keyed as in `LGM.theta_spec`: the
            components' and the likelihood's (``"lik.size"``).
        likelihood: The observation model.
        data: The observations ``"y"`` and where they are.
        projector: ``prior.projector(data)``, built once outside the model
            when no covariate depends on a sampled parameter; required under
            NUTS for an `SPDE` at locations.
        max_newton: Newton iterations of the inner fit.
        reject_unconverged: If ``True``, the factor is $-\infty$ when the
            inner fit did not converge, so NUTS rejects that $\theta$ rather
            than move on an evidence (and an implicit gradient) taken away
            from the mode. Default ``False``: the factor is the evidence at
            the last iterate.

    Examples:
        >>> import numpy as np
        >>> import numpyro
        >>> import numpyro.distributions as dist
        >>> import pyrox_lgm as lgm
        >>> prior = lgm.LGMPrior((lgm.IID(3, name="g"),))
        >>> data = {"y": np.array([1, 0, 2, 4]), "g": np.array([0, 1, 1, 2])}
        >>> def model():
        ...     theta = {"g.tau": numpyro.sample("g.tau", lgm.PCPrecision(1.0, 0.01))}
        ...     lgm.lgm_laplace_factor("y", prior, theta, lgm.Poisson(), data)
        >>> seeded = numpyro.handlers.seed(model, 0)
        >>> sorted(numpyro.handlers.trace(seeded).get_trace())
        ['g.tau', 'y']
    """
    prior._check_observation(likelihood)
    laplace = prior._laplace_under(theta, likelihood, data, projector, max_newton)
    fit = laplace.checked().laplace
    value = fit.log_marginal
    if reject_unconverged:
        value = jnp.where(fit.converged, value, -jnp.inf)
    numpyro.factor(name, value)
