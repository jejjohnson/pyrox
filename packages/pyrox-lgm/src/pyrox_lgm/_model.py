r"""The latent Gaussian model, LGM, and its prior, LGMPrior (P8).

$$
\eta = \sum_j A_j x_j + X\beta + o,\qquad
x_j \mid \theta \sim \text{GMRF}_j(\theta),\quad
\beta \sim \mathcal N(0, \lambda^{-1} I),\qquad
y \mid \eta, \theta \sim \prod_i p(y_i \mid \eta_i, \theta).
$$

The latent vector is $(x_1, \dots, x_k, \beta)$: the components in order,
the fixed effects last (they add dense rows to $Q + A^\top WA$; last keeps the
fill of the sparse Cholesky small, as R-INLA orders them). Its prior is one
GMRF whose precision is the block-diagonal of the components' precisions,
assembled sparse on a pattern that does not depend on $\theta$, and whose
log-density is the *sum of the components'* exact log-densities, each with
its own $\theta$-dependent normaliser.
"""

from __future__ import annotations

import abc
import math
from collections.abc import Mapping

import equinox as eqx
import gaussx as gx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import numpyro.distributions as dist
from argminax.solve import LaplaceModeResult, laplace_mode
from jaxtyping import Array, ArrayLike, Float, PRNGKeyArray
from numpyro.distributions.transforms import Transform

from pyrox_lgm._assembly import block_diagonal, hstack
from pyrox_lgm._components._base import AbstractComponent, prior_shape, probe_theta
from pyrox_lgm._likelihood import AbstractObservation


_LOG_2PI = math.log(2.0 * math.pi)


class FixedEffects(eqx.Module):
    r"""Fixed effects $\beta \sim \mathcal N(0, \lambda^{-1} I)$, columns of $X$.

    Column ``"intercept"`` is all ones unless ``data`` has its own; every
    other name is read from ``data[name]``.

    Attributes:
        names: Fixed-effect names, in the order they enter the latent vector.
        prior_precision: $\lambda$ (R-INLA's default is 0.001).
    """

    names: tuple[str, ...] = eqx.field(static=True)
    prior_precision: float = eqx.field(static=True, default=1e-3)

    def design(self, data: Mapping[str, ArrayLike], n_obs: int) -> Array:
        """The ``(n_obs, p)`` design matrix.

        Covariates may be traced JAX values (a covariate that depends on a
        parameter sampled outside the LGM, as in an MCMC-INLA hybrid): the
        design is dense, so its pattern does not depend on them.
        """
        cols = []
        for name in self.names:
            if name in data:
                col = jnp.asarray(data[name], dtype=float)
            elif name == "intercept":
                col = jnp.ones(n_obs)
            else:
                raise KeyError(f"fixed effect {name!r} is not in data")
            if col.shape != (n_obs,):
                raise ValueError(f"{name!r} has shape {col.shape}, expected ({n_obs},)")
            cols.append(col)
        return jnp.stack(cols, axis=1)


class _Latent:
    """Shared log-density of the assembled latent prior."""

    parts: tuple
    slices: tuple[tuple[int, int], ...]
    consts: tuple
    n_fixed: int
    fixed_precision: float

    def _sum_log_prob(self, x):
        total = jnp.zeros((), dtype=x.dtype)
        for part, (a, b), const in zip(
            self.parts, self.slices, self.consts, strict=True
        ):
            total = total + part.log_prob(x[a:b]) + const
        if self.n_fixed:
            beta = x[-self.n_fixed :]
            lam = self.fixed_precision
            total = total + 0.5 * self.n_fixed * (jnp.log(lam) - _LOG_2PI)
            total = total - 0.5 * lam * jnp.sum(beta**2)
        return total


class LatentGMRF(gx.GaussianMRF, _Latent):
    """Proper assembled latent prior (no intrinsic component)."""

    parts: tuple
    consts: tuple
    slices: tuple[tuple[int, int], ...] = eqx.field(static=True)
    n_fixed: int = eqx.field(static=True)
    fixed_precision: float = eqx.field(static=True)

    def __init__(self, parts, slices, consts, n_fixed, fixed_precision, precision):
        self.parts, self.slices, self.consts = (
            tuple(parts),
            tuple(slices),
            tuple(consts),
        )
        self.n_fixed, self.fixed_precision = int(n_fixed), float(fixed_precision)
        super().__init__(jnp.zeros(precision.in_size()), precision)

    def log_prob(self, value):
        return self._sum_log_prob(value)


class LatentIntrinsicGMRF(gx.IntrinsicGMRF, _Latent):
    """Assembled latent prior with hard constraints from intrinsic components."""

    parts: tuple
    consts: tuple
    slices: tuple[tuple[int, int], ...] = eqx.field(static=True)
    n_fixed: int = eqx.field(static=True)
    fixed_precision: float = eqx.field(static=True)

    def __init__(
        self, parts, slices, consts, n_fixed, fixed_precision, structure, null_space
    ):
        self.parts, self.slices, self.consts = (
            tuple(parts),
            tuple(slices),
            tuple(consts),
        )
        self.n_fixed, self.fixed_precision = int(n_fixed), float(fixed_precision)
        super().__init__(
            jnp.zeros(structure.in_size()),
            1.0,
            structure,
            null_space,
            constraint="hard",
        )

    def log_prob(self, value):
        return self._sum_log_prob(value)


# A Gaussian likelihood's Newton step lands on the mode; the second only sees
# that it has, and the rest are room for the round-off of a stiff system.
_GAUSSIAN_NEWTON = 10
_RESERVED = frozenset({"y", "offset", "n_trials"})


def observation_theta(
    likelihood: AbstractObservation, theta: Mapping[str, ArrayLike]
) -> dict[str, Array]:
    """``likelihood``'s own hyperparameters, from the model's ``"name.key"`` ones."""
    name = likelihood.name
    return {k: jnp.asarray(theta[f"{name}.{k}"]) for k in likelihood.theta_spec()}


class _Hyperparameters(eqx.Module):
    """The unconstrained vector $u$ of whatever `theta_spec` declares."""

    @abc.abstractmethod
    def theta_spec(self) -> dict[str, tuple[dist.Distribution, Transform]]:
        """Every hyperparameter: ``f"{name}.{k}"`` to (prior, bijection)."""

    def _sizes(self) -> list[tuple[str, int, tuple[int, ...]]]:
        out = []
        for key, (prior, _) in self.theta_spec().items():
            shape = prior_shape(prior)
            out.append((key, int(np.prod(shape)) if shape else 1, shape))
        return out

    @property
    def n_theta(self) -> int:
        """Length $m$ of the unconstrained vector $u$."""
        return sum(size for _, size, _ in self._sizes())

    def unflatten(self, u: Float[Array, " m"]) -> dict[str, Array]:
        """Constrained $\\theta$ by key, from the unconstrained vector."""
        spec = self.theta_spec()
        theta, i = {}, 0
        for key, size, shape in self._sizes():
            block = u[i : i + size].reshape(shape) if shape else u[i]
            theta[key] = spec[key][1](block)
            i += size
        return theta

    def log_prior(self, u: Float[Array, " m"]) -> Array:
        """$\\log\\pi(u) = \\sum_k \\log\\pi_k(T_k(u_k)) + \\log|T_k'(u_k)|$."""
        spec = self.theta_spec()
        total, i = jnp.zeros(()), 0
        for key, size, shape in self._sizes():
            block = u[i : i + size].reshape(shape) if shape else u[i]
            prior, transform = spec[key]
            value = transform(block)
            total = total + jnp.sum(prior.log_prob(value))
            total = total + jnp.sum(transform.log_abs_det_jacobian(block, value))
            i += size
        return total


class LGMPrior(_Hyperparameters):
    r"""The latent field of an `LGM`: its components and fixed effects.

    The prior of the latent vector $x = (x_1, \dots, x_k, \beta)$ at
    hyperparameters $\theta$, without a likelihood:

    $$
    x_j \mid \theta \sim \text{GMRF}_j(\theta),\qquad
    \beta \sim \mathcal N(0, \lambda^{-1} I),\qquad
    \eta = \sum_j A_j x_j + X\beta + o.
    $$

    ``theta`` is a mapping of constrained values keyed as in `theta_spec`
    (``"t.tau"``); keys the prior does not know are ignored, so the values of
    a whole `LGM` can be passed. ``data`` is an `LGM`'s: the node index (or
    locations) of each observation per component, the fixed-effect
    covariates, an optional ``"offset"`` and, where observations are scored,
    ``"y"``.

    Args:
        components: Latent components, with distinct names.
        fixed: Fixed effects, or ``None``.

    Examples:
        >>> import jax
        >>> import numpy as np
        >>> import pyrox_lgm as lgm
        >>> prior = lgm.LGMPrior((lgm.AR1(5, name="t"),), lgm.FixedEffects(("z",)))
        >>> list(prior.theta_spec())
        ['t.tau', 't.rho']
        >>> draws = prior.sample(jax.random.key(0), {"t.tau": 4.0, "t.rho": 0.8}, 3)
        >>> draws.shape  # 5 nodes + the slope of z
        (3, 6)

        The likelihood joins in an `LGM`:

        >>> model = lgm.LGM(prior, likelihood=lgm.Poisson())
        >>> model.components is prior.components
        True
    """

    components: tuple[AbstractComponent, ...]
    fixed: FixedEffects | None
    consts: tuple[Float[Array, ""], ...]

    def __init__(
        self,
        components: tuple[AbstractComponent, ...],
        fixed: FixedEffects | None = None,
    ) -> None:
        names = [c.name for c in components]
        if fixed is not None:
            names += list(fixed.names)
        if len(set(names)) != len(names):
            raise ValueError(
                f"component and fixed-effect names must be distinct, got {names}"
            )
        # Each effect reads its data column by name, so it cannot share one
        # with the response or the observation model's inputs.
        reserved = sorted(set(names) & _RESERVED)
        if reserved:
            raise ValueError(
                f"{reserved} name observation data; rename the effect "
                f"(reserved: {sorted(_RESERVED)})"
            )
        self.components = tuple(components)
        self.fixed = fixed
        self.consts = tuple(self._normaliser(c) for c in self.components)

    # -- hyperparameters --------------------------------------------------

    def theta_spec(self) -> dict[str, tuple[dist.Distribution, Transform]]:
        """The components' hyperparameters: ``f"{name}.{k}"`` to (prior, bijection)."""
        spec = {}
        for c in self.components:
            spec |= {f"{c.name}.{k}": v for k, v in c.theta_spec().items()}
        return spec

    def _component_theta(
        self, theta: Mapping[str, ArrayLike], comp: AbstractComponent
    ) -> dict[str, Array]:
        return {k: jnp.asarray(theta[f"{comp.name}.{k}"]) for k in comp.theta_spec()}

    def _check_observation(self, likelihood: AbstractObservation) -> None:
        # Hyperparameters are keyed f"{name}.{k}": the observation model's
        # namespace must not be a component's.
        if likelihood.name in {c.name for c in self.components}:
            raise ValueError(
                f"the observation model's name {likelihood.name!r} is also "
                "a component's; rename one"
            )

    # -- latent field -----------------------------------------------------

    @property
    def n_fixed(self) -> int:
        return 0 if self.fixed is None else len(self.fixed.names)

    def slices(self) -> dict[str, tuple[int, int]]:
        """``name -> (start, stop)`` in the latent vector; fixed effects last."""
        out, start = {}, 0
        for c in self.components:
            out[c.name] = (start, start + c.n_nodes)
            start += c.n_nodes
        if self.fixed is not None:
            for j, name in enumerate(self.fixed.names):
                out[name] = (start + j, start + j + 1)
        return out

    @property
    def n_latent(self) -> int:
        return sum(c.n_nodes for c in self.components) + self.n_fixed

    def _normaliser(self, comp: AbstractComponent) -> Float[Array, ""]:
        """The theta-free constant an intrinsic component's log_prob omits.

        Computed once, so it must not depend on theta: probed at two points,
        a structure whose pseudo-determinant moves is rejected.
        """
        spec = comp.theta_spec()
        first, second = (
            self._intrinsic_constant(comp.prior(probe_theta(spec, o)))
            for o in (0.0, 0.5)
        )
        if first is None or second is None:
            return jnp.asarray(0.0)
        if not np.isclose(float(first), float(second), rtol=1e-8, atol=1e-8):
            raise ValueError(
                f"component {comp.name!r} returns an IntrinsicGMRF whose structure "
                "depends on its hyperparameters beyond precision_scale; return it "
                "with include_normalizer=True so its normaliser follows theta"
            )
        return first

    @staticmethod
    def _intrinsic_constant(gmrf) -> Float[Array, ""] | None:
        if (
            not isinstance(gmrf, gx.IntrinsicGMRF)
            or gmrf.include_normalizer
            or getattr(gmrf, "normalized", False)
        ):
            return None
        V = gmrf.null_space
        rank = gmrf.structure.in_size() - V.shape[1]
        log_pdet = gx.pseudo_logdet(gmrf.structure, null_space=V)
        return 0.5 * log_pdet - 0.5 * rank * _LOG_2PI

    def latent_prior(
        self, theta: Mapping[str, ArrayLike]
    ) -> LatentGMRF | LatentIntrinsicGMRF:
        """The assembled prior of $(x_1, \\dots, x_k, \\beta)$ at ``theta``."""
        parts, blocks, nulls, slices, start = [], [], [], [], 0
        for c in self.components:
            theta_c = self._component_theta(theta, c)
            gmrf = c.prior(theta_c, constraint="hard")
            op, V = c.assembly_precision(theta_c, gmrf), c.constraint_basis(gmrf)
            parts.append(gmrf)
            blocks.append(op)
            nulls.append((start, V))
            slices.append((start, start + c.n_nodes))
            start += c.n_nodes
        lam = 0.0
        if self.fixed is not None:
            lam = self.fixed.prior_precision
            blocks.append(lx.DiagonalLinearOperator(jnp.full(self.n_fixed, lam)))
        Q = block_diagonal(blocks)
        n = Q.in_size()
        constrained = [(s, V) for s, V in nulls if V is not None]
        if not constrained:
            return LatentGMRF(parts, slices, self.consts, self.n_fixed, lam, Q)
        c_total = sum(V.shape[1] for _, V in constrained)
        null = jnp.zeros((n, c_total))
        col = 0
        for s, V in constrained:
            null = null.at[s : s + V.shape[0], col : col + V.shape[1]].set(V)
            col += V.shape[1]
        return LatentIntrinsicGMRF(
            parts, slices, self.consts, self.n_fixed, lam, Q, null
        )

    def sample(
        self, key: PRNGKeyArray, theta: Mapping[str, ArrayLike], n_samples: int = 1
    ) -> Float[Array, "S n"]:
        r"""Draws of the latent vector $x = (x_1, \dots, x_k, \beta)$ from the prior.

        One draw per component and one of the fixed effects. An intrinsic
        component is improper along its null space, where it is drawn as
        zero: an `RW2` draw sums to zero and has no linear trend, which an
        `LGM` leaves free.

        A component is drawn by its own `gaussx.IntrinsicGMRF`, whose sampler
        regularises the structure by about $10^{-8}$ of its diagonal. That is
        more than the smallest eigenvalues of a long `RW2`, so draws of one
        with more than about 100 nodes are less variable than the prior
        that is conditioned on.

        Args:
            key: PRNG key.
            theta: Hyperparameter values keyed as in `theta_spec`.
            n_samples: Number of independent draws $S$.

        Returns:
            The draws, shaped ``(S, n_latent)``, in the order of `slices`.
        """
        keys = jax.random.split(key, len(self.components) + 1)
        draws = [
            c.prior(self._component_theta(theta, c), constraint="hard").sample(
                k, (n_samples,)
            )
            for c, k in zip(self.components, keys[:-1], strict=True)
        ]
        if self.fixed is not None:
            z = jax.random.normal(keys[-1], (n_samples, self.n_fixed))
            draws.append(z / math.sqrt(self.fixed.prior_precision))
        return jnp.concatenate(draws, axis=1) if draws else jnp.zeros((n_samples, 0))

    # -- observations ------------------------------------------------------

    def _n_rows(self, data: Mapping[str, ArrayLike]) -> int:
        """How many rows ``data`` has: the length of ``"y"`` or of any column."""
        names = ["y", *(c.name for c in self.components)]
        names += [] if self.fixed is None else self.fixed.names
        for name in (*names, "offset"):
            if name in data and np.ndim(data[name]):
                return np.shape(data[name])[0]
        raise KeyError(f"data has none of the columns {names}, so no number of rows")

    def projector(self, data: Mapping[str, ArrayLike]) -> gx.SparseOperator:
        """``[A_1 | ... | A_k | X]`` for ``data``, built on the host.

        ``data`` needs no ``"y"``: its rows are counted from any column.
        """
        m = self._n_rows(data)
        blocks: list[lx.AbstractLinearOperator] = []
        for c in self.components:
            if c.name not in data:
                raise KeyError(f"data has no index for component {c.name!r}")
            idx = np.asarray(data[c.name])
            if idx.ndim == 2:  # locations (any dtype), not node indices
                project = getattr(c, "project_points", None)
                if project is None:
                    raise ValueError(f"{c.name!r} takes node indices, not points")
                blocks.append(project(idx.astype(float)))
            else:
                blocks.append(c.projector(idx))
        if self.fixed is not None:
            X = self.fixed.design(data, m)
            r, cidx = np.divmod(np.arange(X.size), X.shape[1])
            blocks.append(
                gx.SparseOperator.from_coo(
                    r, cidx, jnp.asarray(X.ravel()), (m, X.shape[1])
                )
            )
        return hstack(blocks, m)

    def offset(self, data: Mapping[str, ArrayLike]) -> Array | None:
        return None if "offset" not in data else jnp.asarray(data["offset"])

    # -- evidence ----------------------------------------------------------

    def _laplace(
        self,
        theta: Mapping[str, ArrayLike],
        likelihood: gx.AbstractLikelihood,
        data: Mapping[str, ArrayLike],
        projector: gx.SparseOperator | None,
        max_newton: int,
    ) -> LaplaceModeResult:
        """The Gaussian approximation of $x \\mid y, \\theta$ at its mode."""
        return laplace_mode(
            self.latent_prior(theta),
            likelihood,
            projector=self.projector(data) if projector is None else projector,
            offset=self.offset(data),
            max_iter=max_newton,
        )

    def _laplace_under(
        self,
        theta: Mapping[str, ArrayLike],
        likelihood: AbstractObservation,
        data: Mapping[str, ArrayLike],
        projector: gx.SparseOperator | None,
        max_newton: int,
    ) -> LaplaceModeResult:
        """`_laplace` under an observation model, its hyperparameters in ``theta``."""
        own = observation_theta(likelihood, theta)
        built = likelihood.build(jnp.asarray(data["y"]), own, data)
        return self._laplace(theta, built, data, projector, max_newton)

    def _laplace_gaussian(
        self,
        theta: Mapping[str, ArrayLike],
        data: Mapping[str, ArrayLike],
        noise_var: Float[ArrayLike, ""],
        projector: gx.SparseOperator | None,
    ) -> LaplaceModeResult:
        """`_laplace` under Gaussian noise: exact, so one Newton step and its check."""
        y, noise = jnp.asarray(data["y"]), jnp.asarray(noise_var)
        # gaussx annotates a Python float and takes a (traced) scalar too.
        built = gx.GaussianLikelihood(y, noise)  # ty: ignore[invalid-argument-type]
        return self._laplace(theta, built, data, projector, _GAUSSIAN_NEWTON)

    def log_marginal(
        self,
        theta: Mapping[str, ArrayLike],
        data: Mapping[str, ArrayLike],
        noise_var: Float[ArrayLike, ""],
        *,
        projector: gx.SparseOperator | None = None,
    ) -> Float[Array, ""]:
        r"""Exact log marginal likelihood under Gaussian noise.

        $$
        \log p(y \mid \theta)
          = \log \mathcal N(y \mid o,\; A Q(\theta)^{-1} A^\top + \sigma^2 I),
        $$

        the latent vector integrated out in precision form, at the cost of a
        sparse Cholesky of $Q(\theta) + \sigma^{-2} A^\top A$, and without the
        hyperparameters' priors. An intrinsic component contributes the
        density of its constrained field, as in `LGM.log_posterior_theta`.
        For any other likelihood, `LGM.log_marginal` is the Laplace
        approximation.

        Args:
            theta: Hyperparameter values keyed as in `theta_spec`.
            data: The observations ``"y"`` and where they are.
            noise_var: Observation noise variance $\sigma^2$.
            projector: ``self.projector(data)``, to keep its host work out of
                a traced loop.

        Returns:
            The scalar log marginal likelihood.

        Examples:
            >>> import jax
            >>> import numpy as np
            >>> import pyrox_lgm as lgm
            >>> prior = lgm.LGMPrior((lgm.RW1(6, name="t"),))
            >>> data = {"y": np.sin(np.arange(6.0)), "t": np.arange(6)}
            >>> prior.log_marginal({"t.tau": 2.0}, data, 0.1).shape
            ()

            Differentiable in the noise and the hyperparameters:

            >>> grad = jax.grad(lambda v: prior.log_marginal({"t.tau": 2.0}, data, v))
            >>> bool(np.isfinite(grad(0.1)))
            True
        """
        return self._laplace_gaussian(theta, data, noise_var, projector).log_marginal


class LGM(_Hyperparameters):
    r"""A latent Gaussian model: components, fixed effects and an observation model.

    $\theta$ collects every component's and the likelihood's hyperparameters
    under ``f"{name}.{k}"``. `inla()` works on the *unconstrained* vector
    $u$, $\theta = T(u)$ with each prior's bijection, whose log-prior
    includes the Jacobian; `log_posterior_theta` is $\log\tilde\pi(u\mid y)$
    in those coordinates.

    ``data`` holds ``"y"``, an optional ``"offset"``, the node index of each
    observation for every component (``data[component.name]``; an
    `SPDE` also takes locations, ``(n, 2)`` points on a mesh or ``(n, d)`` on
    a grid) and the fixed-effect covariates.

    The latent field is an `LGMPrior`, ``model.prior``, and the model forwards
    ``components``, ``fixed``, ``slices``, ``n_latent``, ``n_fixed``,
    ``latent_prior``, ``projector`` and ``offset`` to it.

    Args:
        components: Latent components, with distinct names, or an `LGMPrior`.
        fixed: Fixed effects, or ``None``; an `LGMPrior` has its own.
        likelihood: Observation model (`Gaussian` by default).

    Examples:
        >>> import jax.numpy as jnp
        >>> import numpy as np
        >>> import pyrox_lgm as lgm
        >>> model = lgm.LGM(
        ...     components=(lgm.RW1(5, name="t"),),
        ...     fixed=lgm.FixedEffects(("intercept",)),
        ...     likelihood=lgm.Poisson(),
        ... )
        >>> list(model.theta_spec())
        ['t.tau']
        >>> data = {"y": np.array([1, 0, 2, 3, 1]), "t": np.arange(5)}
        >>> model.projector(data).as_matrix().shape  # 5 nodes + intercept
        (5, 6)

        The same latent field under another likelihood:

        >>> lgm.LGM(model.prior, likelihood=lgm.Gaussian()).n_theta
        2
    """

    prior: LGMPrior
    likelihood: AbstractObservation

    def __init__(
        self,
        components: tuple[AbstractComponent, ...] | LGMPrior,
        fixed: FixedEffects | None = None,
        likelihood: AbstractObservation | None = None,
    ) -> None:
        from pyrox_lgm._likelihood import Gaussian

        if isinstance(components, LGMPrior):
            if fixed is not None:
                raise ValueError("an LGMPrior has its own fixed effects; drop fixed=")
            self.prior = components
        else:
            self.prior = LGMPrior(components, fixed)
        self.likelihood = Gaussian() if likelihood is None else likelihood
        self.prior._check_observation(self.likelihood)

    @property
    def components(self) -> tuple[AbstractComponent, ...]:
        """The latent components, ``prior.components``."""
        return self.prior.components

    @property
    def fixed(self) -> FixedEffects | None:
        """The fixed effects, ``prior.fixed``."""
        return self.prior.fixed

    @property
    def consts(self) -> tuple[Float[Array, ""], ...]:
        """The components' theta-free normalisers, ``prior.consts``."""
        return self.prior.consts

    # -- hyperparameters --------------------------------------------------

    def theta_spec(self) -> dict[str, tuple[dist.Distribution, Transform]]:
        """Every hyperparameter: ``f"{name}.{k}"`` to (prior, bijection)."""
        lik = self.likelihood
        own = {f"{lik.name}.{k}": v for k, v in lik.theta_spec().items()}
        return self.prior.theta_spec() | own

    def _likelihood_theta(self, theta: Mapping[str, ArrayLike]) -> dict[str, Array]:
        return observation_theta(self.likelihood, theta)

    # -- latent field (the prior's) ----------------------------------------

    @property
    def n_fixed(self) -> int:
        """Number of fixed effects."""
        return self.prior.n_fixed

    def slices(self) -> dict[str, tuple[int, int]]:
        """``name -> (start, stop)`` in the latent vector; fixed effects last."""
        return self.prior.slices()

    @property
    def n_latent(self) -> int:
        """Size of the latent vector."""
        return self.prior.n_latent

    def latent_prior(
        self, theta: Mapping[str, ArrayLike]
    ) -> LatentGMRF | LatentIntrinsicGMRF:
        """The assembled prior of $(x_1, \\dots, x_k, \\beta)$ at ``theta``."""
        return self.prior.latent_prior(theta)

    # -- observations ------------------------------------------------------

    def projector(self, data: Mapping[str, ArrayLike]) -> gx.SparseOperator:
        """``[A_1 | ... | A_k | X]`` for ``data``, built on the host."""
        return self.prior.projector(data)

    def offset(self, data: Mapping[str, ArrayLike]) -> Array | None:
        """``data["offset"]`` as an array, or ``None``."""
        return self.prior.offset(data)

    def log_marginal(
        self,
        theta: Mapping[str, ArrayLike],
        data: Mapping[str, ArrayLike],
        *,
        projector: gx.SparseOperator | None = None,
        max_newton: int = 50,
    ) -> Float[Array, ""]:
        r"""The Laplace log marginal likelihood at constrained hyperparameters.

        $$
        \log p(y \mid \theta) \approx \log p(y \mid \hat x, \theta)
          + \log p(\hat x \mid \theta)
          - \log p_G(\hat x \mid y, \theta),
        $$

        at the mode $\hat x$ of the latent vector, with $p_G$ the Gaussian
        approximation there: exact for a `Gaussian` observation model. It
        holds no hyperprior, so a model can put its own on $\theta$.
        Gradients are implicit at the mode (argminax), so the Newton loop is
        not unrolled.

        Args:
            theta: Constrained values of every hyperparameter, keyed as in
                `theta_spec`.
            data: The observations ``"y"`` and where they are.
            projector: ``self.projector(data)``, to keep its host work out of
                a traced loop.
            max_newton: Newton iterations of the inner fit.

        Returns:
            The scalar log marginal likelihood.

        Examples:
            >>> import numpy as np
            >>> import pyrox_lgm as lgm
            >>> model = lgm.LGM((lgm.RW1(5, name="t"),), likelihood=lgm.Poisson())
            >>> data = {"y": np.array([1, 0, 2, 3, 1]), "t": np.arange(5)}
            >>> model.log_marginal({"t.tau": 2.0}, data).shape
            ()
        """
        fit = self.prior._laplace_under(
            theta, self.likelihood, data, projector, max_newton
        )
        return fit.log_marginal

    def log_posterior_theta(
        self,
        u: Float[Array, " m"],
        data: Mapping[str, ArrayLike],
        *,
        projector: gx.SparseOperator | None = None,
        max_newton: int = 50,
    ) -> Array:
        """The unnormalised log-posterior of the unconstrained hyperparameters.

        $\\log\\tilde\\pi(u \\mid y) = \\log\\tilde\\pi(y\\mid\\theta)
        + \\log\\pi(u)$. Exact gradients flow through argminax's
        implicit-differentiated Laplace mode. Pass a prebuilt ``projector`` to
        keep host work out of a traced loop.
        """
        log_marginal = self.log_marginal(
            self.unflatten(u), data, projector=projector, max_newton=max_newton
        )
        return log_marginal + self.log_prior(u)

    def laplace(
        self,
        u: Float[Array, " m"],
        data: Mapping[str, ArrayLike],
        *,
        projector: gx.SparseOperator | None = None,
        max_newton: int = 50,
    ) -> LaplaceModeResult:
        """The Laplace approximation of $x \\mid y, \\theta = T(u)$, by
        `argminax.solve.laplace_mode`."""
        return self.prior._laplace_under(
            self.unflatten(u), self.likelihood, data, projector, max_newton
        )
