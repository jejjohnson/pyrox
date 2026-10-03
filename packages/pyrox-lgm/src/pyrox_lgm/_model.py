r"""The latent Gaussian model, LGM (P8).

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

import math
from collections.abc import Mapping

import equinox as eqx
import gaussx as gx
import jax.numpy as jnp
import lineax as lx
import numpy as np
import numpyro.distributions as dist
from jaxtyping import Array, ArrayLike, Float
from numpyro.distributions.transforms import Transform

from pyrox_lgm._assembly import block_diagonal, hstack
from pyrox_lgm._components._base import AbstractComponent
from pyrox_lgm._likelihood import AbstractObservation


_LOG_2PI = math.log(2.0 * math.pi)


class FixedEffects(eqx.Module):
    r"""Fixed effects $\beta \sim \mathcal N(0, \lambda^{-1} I)$, columns of $X$.

    Column ``"intercept"`` is all ones unless ``data`` has its own; every
    other name is read from ``data[name]``.

    Args:
        names: Fixed-effect names, in the order they enter the latent vector.
        prior_precision: $\lambda$ (R-INLA's default is 0.001).
    """

    names: tuple[str, ...] = eqx.field(static=True)
    prior_precision: float = eqx.field(static=True, default=1e-3)

    def design(self, data: Mapping[str, ArrayLike], n_obs: int) -> np.ndarray:
        """The ``(n_obs, p)`` design matrix (host)."""
        cols = []
        for name in self.names:
            if name in data:
                col = np.asarray(data[name], dtype=float)
            elif name == "intercept":
                col = np.ones(n_obs)
            else:
                raise KeyError(f"fixed effect {name!r} is not in data")
            if col.shape != (n_obs,):
                raise ValueError(f"{name!r} has shape {col.shape}, expected ({n_obs},)")
            cols.append(col)
        return np.stack(cols, axis=1)


def _block(gmrf) -> lx.AbstractLinearOperator:
    """The precision operator of a component GMRF."""
    if isinstance(gmrf, gx.IntrinsicGMRF):
        from pyrox_lgm._components._base import scale_operator

        return scale_operator(gmrf.structure, gmrf.precision_scale)
    return gmrf.precision


class _Latent:
    """Shared log-density of the assembled latent prior."""

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

    pytree_data_fields = ("parts", "consts")
    pytree_aux_fields = ("slices", "n_fixed", "fixed_precision")

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

    pytree_data_fields = ("parts", "consts")
    pytree_aux_fields = ("slices", "n_fixed", "fixed_precision")

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


class LGM(eqx.Module):
    r"""A latent Gaussian model: components, fixed effects and an observation model.

    $\theta$ collects every component's and the likelihood's hyperparameters
    under ``f"{name}.{k}"``. `inla()` works on the *unconstrained* vector
    $u$, $\theta = T(u)$ with each prior's bijection, whose log-prior
    includes the Jacobian; `log_posterior_theta` is $\log\tilde\pi(u\mid y)$
    in those coordinates.

    ``data`` holds ``"y"``, an optional ``"offset"``, the node index of each
    observation for every component (``data[component.name]``; an
    `SPDE` on a mesh also takes ``(n, 2)`` points) and the fixed-effect
    covariates.

    Args:
        components: Latent components, with distinct names.
        fixed: Fixed effects, or ``None``.
        likelihood: Observation model.

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
    """

    components: tuple[AbstractComponent, ...]
    fixed: FixedEffects | None
    likelihood: AbstractObservation
    consts: tuple[Float[Array, ""], ...]

    def __init__(
        self,
        components: tuple[AbstractComponent, ...],
        fixed: FixedEffects | None = None,
        likelihood: AbstractObservation | None = None,
    ) -> None:
        from pyrox_lgm._likelihood import Gaussian

        names = [c.name for c in components]
        if len(set(names)) != len(names):
            raise ValueError(f"component names must be distinct, got {names}")
        self.components = tuple(components)
        self.fixed = fixed
        self.likelihood = Gaussian() if likelihood is None else likelihood
        self.consts = tuple(self._normaliser(c) for c in self.components)

    # -- hyperparameters --------------------------------------------------

    def theta_spec(self) -> dict[str, tuple[dist.Distribution, Transform]]:
        """Every hyperparameter: ``f"{name}.{k}"`` to (prior, bijection)."""
        spec = {}
        for c in self.components:
            spec |= {f"{c.name}.{k}": v for k, v in c.theta_spec().items()}
        lik = self.likelihood
        spec |= {f"{lik.name}.{k}": v for k, v in lik.theta_spec().items()}
        return spec

    def _sizes(self) -> list[tuple[str, int, tuple[int, ...]]]:
        out = []
        for key, (prior, _) in self.theta_spec().items():
            shape = tuple(prior.event_shape)
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
            total = total + prior.log_prob(value)
            total = total + jnp.sum(transform.log_abs_det_jacobian(block, value))
            i += size
        return total

    def _component_theta(self, theta: Mapping[str, Array], comp) -> dict[str, Array]:
        return {k: theta[f"{comp.name}.{k}"] for k in comp.theta_spec()}

    def _likelihood_theta(self, theta: Mapping[str, Array]) -> dict[str, Array]:
        lik = self.likelihood
        return {k: theta[f"{lik.name}.{k}"] for k in lik.theta_spec()}

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
        """The theta-free constant an intrinsic component's log_prob omits."""
        probe = {
            k: transform(jnp.zeros(prior.event_shape))
            for k, (prior, transform) in comp.theta_spec().items()
        }
        gmrf = comp.prior(probe)
        if (
            not isinstance(gmrf, gx.IntrinsicGMRF)
            or gmrf.include_normalizer
            or getattr(gmrf, "normalized", False)
        ):
            return jnp.asarray(0.0)
        V = gmrf.null_space
        rank = gmrf.structure.in_size() - V.shape[1]
        log_pdet = gx.pseudo_logdet(gmrf.structure, null_space=V)
        return 0.5 * log_pdet - 0.5 * rank * _LOG_2PI

    def latent_prior(
        self, theta: Mapping[str, Array]
    ) -> LatentGMRF | LatentIntrinsicGMRF:
        """The assembled prior of $(x_1, \\dots, x_k, \\beta)$ at ``theta``."""
        parts, blocks, nulls, slices, start = [], [], [], [], 0
        for c in self.components:
            gmrf = c.prior(self._component_theta(theta, c), constraint="hard")
            op, V = _block(gmrf), c.constraint_basis(gmrf)
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

    # -- observations ------------------------------------------------------

    def projector(self, data: Mapping[str, ArrayLike]) -> gx.SparseOperator:
        """``[A_1 | ... | A_k | X]`` for ``data``, built on the host."""
        y = np.asarray(data["y"])
        m = y.shape[0]
        blocks: list[lx.AbstractLinearOperator] = []
        for c in self.components:
            if c.name not in data:
                raise KeyError(f"data has no index for component {c.name!r}")
            idx = np.asarray(data[c.name])
            if idx.ndim == 2 and np.issubdtype(idx.dtype, np.floating):
                project = getattr(c, "project_points", None)
                if project is None:
                    raise ValueError(f"{c.name!r} takes node indices, not points")
                blocks.append(project(idx))
            else:
                blocks.append(c.projector(idx))
        if self.fixed is not None:
            X = self.fixed.design(data, m)
            r, cidx = np.divmod(np.arange(X.size), X.shape[1])
            blocks.append(
                gx.SparseOperator.from_coo(r, cidx, jnp.asarray(X.ravel()), X.shape)
            )
        return hstack(blocks, m)

    def offset(self, data: Mapping[str, ArrayLike]) -> Array | None:
        return None if "offset" not in data else jnp.asarray(data["offset"])

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
        + \\log\\pi(u)$. Exact gradients flow through gaussx's
        implicit-differentiated Laplace mode. Pass a prebuilt ``projector`` to
        keep host work out of a traced loop.
        """
        fit = self.laplace(u, data, projector=projector, max_newton=max_newton)
        return fit.log_marginal + self.log_prior(u)

    def laplace(
        self,
        u: Float[Array, " m"],
        data: Mapping[str, ArrayLike],
        *,
        projector: gx.SparseOperator | None = None,
        max_newton: int = 50,
    ) -> gx.LaplaceResult:
        """The Laplace approximation of $x \\mid y, \\theta = T(u)$."""
        theta = self.unflatten(u)
        A = self.projector(data) if projector is None else projector
        lik = self.likelihood.build(
            jnp.asarray(data["y"]), self._likelihood_theta(theta), data
        )
        return gx.laplace_mode(
            self.latent_prior(theta),
            lik,
            projector=A,
            offset=self.offset(data),
            max_iter=max_newton,
        )
