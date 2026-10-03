"""Base class for latent components (P7).

A latent component is one additive piece of the linear predictor
$\\eta = \\sum_j A_j x_j$ of a latent Gaussian model: a Gaussian Markov random
field $x_j$ over its own nodes (time points, areas, mesh vertices), its
hyperparameters $\\theta_j$ with their priors, and the projector $A_j$ from
nodes to observations. The same object serves ``inla()`` (through `prior`,
`theta_spec` and `projector`) and any NumPyro model (through `sample`).
"""

from __future__ import annotations

import abc
from typing import Literal

import equinox as eqx
import gaussx as gx
import jax.numpy as jnp
import lineax as lx
import numpy as np
import numpyro.distributions as dist
from jaxtyping import Array, ArrayLike, Float, Int
from numpyro.distributions.transforms import Transform


Constraint = Literal["hard", "soft", "none"]


class AbstractComponent(eqx.Module):
    """A latent field with hyperpriors and a projector to observations.

    Subclasses define `n_nodes`, `theta_spec` and `prior`. The default
    `projector` selects nodes by index; the default `sample` is the NumPyro
    face (see `pyrox_lgm._numpyro.sample_component`).

    Attributes:
        name: Site-name prefix, unique within a model: the field is the
            NumPyro site ``name`` and hyperparameter ``k`` the site
            ``f"{name}_{k}"``.
    """

    name: eqx.AbstractVar[str]

    @property
    @abc.abstractmethod
    def n_nodes(self) -> int:
        """Size of the latent field (including any padding node)."""

    @property
    def n_index(self) -> int:
        """Number of addressable nodes: what an observation index refers to.

        Equal to `n_nodes` except where the field carries padding nodes
        (an odd-length `RW2`), which observations never touch.
        """
        return self.n_nodes

    @abc.abstractmethod
    def theta_spec(self) -> dict[str, tuple[dist.Distribution, Transform]]:
        """Hyperparameters: name to (prior, bijection from the reals).

        The transform maps an unconstrained real to the parameter's support,
        which is how ``inla()`` explores $\\theta$.
        """

    @abc.abstractmethod
    def prior(
        self,
        theta: dict[str, Array],
        *,
        constraint: Constraint = "hard",
        soft_constraint_scale: float = 1e-3,
    ) -> gx.GaussianMRF | gx.IntrinsicGMRF:
        r"""The GMRF of the field given its hyperparameters.

        Args:
            theta: Hyperparameter values keyed as in `theta_spec`.
            constraint: For intrinsic fields, how the null space is handled
                (``"hard"`` for ``inla()``, ``"soft"`` for NUTS); ignored by
                proper fields.
            soft_constraint_scale: Standard deviation $s$ of the soft
                constraint $V^\top x \sim \mathcal N(0, s^2 I)$.
        """

    def assembly_precision(
        self, theta: dict[str, Array], gmrf: gx.GaussianMRF | gx.IntrinsicGMRF
    ) -> lx.AbstractLinearOperator:
        """The precision an `LGM` assembles into its sparse block-diagonal.

        ``gmrf`` is ``prior(theta)``; its precision by default (``scale *
        structure`` for an intrinsic field). A component whose prior uses an
        operator that cannot be assembled entry-wise (a grid SPDE's spectral
        function) returns an equivalent sparse one here.
        """
        if isinstance(gmrf, gx.IntrinsicGMRF):
            return scale_operator(gmrf.structure, gmrf.precision_scale)
        return gmrf.precision

    def constraint_basis(
        self, gmrf: gx.GaussianMRF | gx.IntrinsicGMRF
    ) -> Float[Array, "n c"] | None:
        """Orthonormal basis of the hard constraints ``inla()`` imposes.

        The intrinsic field's null space by default (sum-to-zero per
        connected component, say). A component may constrain fewer
        directions and leave the rest to the data, as R-INLA's ``rw2``
        constrains only the sum and keeps the linear trend; the density's
        rank is unchanged.
        """
        return gmrf.null_space if isinstance(gmrf, gx.IntrinsicGMRF) else None

    def projector(self, index: Int[ArrayLike, " n_obs"]) -> gx.SparseOperator:
        """``(n_obs, n_nodes)`` selector with a one where observation i sits.

        Args:
            index: Node of each observation, integers in ``[0, n_index)``.
        """
        index = np.asarray(index)
        if index.ndim != 1:
            raise ValueError(f"index must be 1-D, got shape {index.shape}")
        if index.size and (index.min() < 0 or index.max() >= self.n_index):
            raise ValueError(
                f"index out of range for {self.n_index} nodes: "
                f"[{index.min()}, {index.max()}]"
            )
        m = index.shape[0]
        return gx.SparseOperator.from_coo(
            np.arange(m), index, jnp.ones(m), (m, self.n_nodes)
        )

    def sample(
        self,
        index: Int[ArrayLike, " n_obs"] | None = None,
        *,
        soft_constraint_scale: float = 1e-2,
    ) -> Array:
        """NumPyro face: sample $\\theta$ and the field, return it at ``index``.

        Draws each hyperparameter from its prior (site ``f"{name}_{k}"``),
        then the field (site ``name``) with a *soft* constraint for intrinsic
        fields, so it works under NUTS. Returns the field at ``index``, or all
        `n_index` addressable nodes when ``index`` is ``None``. See
        `pyrox_lgm._numpyro.sample_component` for ``soft_constraint_scale``.
        """
        from pyrox_lgm._numpyro import sample_component

        return sample_component(
            self, index, soft_constraint_scale=soft_constraint_scale
        )


def default_transform(prior: dist.Distribution) -> Transform:
    """The bijection from the reals onto ``prior``'s support."""
    return dist.biject_to(prior.support)


def scale_operator(
    operator: lx.AbstractLinearOperator, scale: Float[ArrayLike, ""]
) -> lx.AbstractLinearOperator:
    """``scale * operator`` keeping the operator's structure (and its solvers).

    ``scale * SparseOperator`` would otherwise become a generic lineax
    ``MulLinearOperator`` and lose the sparse-Cholesky dispatch.
    """
    if isinstance(operator, gx.SparseOperator):
        return eqx.tree_at(lambda o: o.values, operator, scale * operator.values)
    if isinstance(operator, lx.DiagonalLinearOperator):
        return lx.DiagonalLinearOperator(scale * operator.diagonal)
    return scale * operator
