r"""Areal (graph) components: Besag (ICAR), BYM2, proper CAR and Leroux (P7).

All take a kernellib graph (`kernellib.Graph` from edges or polygons'
contiguity, or a `kernellib.GridGraph`) and keep its structure: a
`gaussx.SparseOperator` precision on a general graph (sparse Cholesky), a
`gaussx.KroneckerSum` on a face-connected grid where the model allows it.

| Component | Precision | Hyperparameters |
|---|---|---|
| `Besag` | $\tau R$, intrinsic | $\tau$ |
| `BYM2` | `gaussx.BYM2GMRF` on $(b, u_\ast)$ | $\tau$, $\phi$ (`PCBYM2Phi`) |
| `CAR` | $\tau(D - \rho W)$ | $\tau$, $\rho \in (0, 1)$ |
| `Leroux` | $\tau(\rho R + (1-\rho) I)$ | $\tau$, $\rho \in (0, 1)$ |

$R = D - W$ is the graph Laplacian; `Besag` and `BYM2` use the per-component
BYM2 scaling of kernellib's `structure_matrix`. The intrinsic fields carry
one sum-to-zero constraint per connected component (`graph_null_space`), so
an isolated node is pinned to zero (R-INLA instead gives it unit variance);
drop or merge isolated areas first.
"""

from __future__ import annotations

from typing import Literal

import equinox as eqx
import gaussx as gx
import jax.numpy as jnp
import kernellib as kl
import lineax as lx
import numpy as np
import numpyro.distributions as dist
from jaxtyping import Array, Float
from numpyro.distributions.transforms import Transform

from pyrox_lgm._components._base import (
    AbstractComponent,
    Constraint,
    default_transform,
)
from pyrox_lgm._priors import PCBYM2Phi, PCPrecision, structure_spectrum
from pyrox_lgm._priors._pc import _dense_eigvals


# Above this size the log-determinant spectrum is not cached; gaussx then
# takes log|Q| from the sparse Cholesky factor at each theta.
_DENSE_SPECTRUM_MAX = 10_000


def _tau_prior(prior: dist.Distribution | None) -> dist.Distribution:
    return PCPrecision(1.0, 0.01) if prior is None else prior


def _rho_prior(prior: dist.Distribution | None) -> dist.Distribution:
    return dist.Uniform(0.0, 1.0) if prior is None else prior


def _sparse_laplacian(graph: kl.AbstractGraph) -> gx.SparseOperator:
    """The Laplacian as a `SparseOperator`, also for a `GridGraph`."""
    if isinstance(graph, kl.GridGraph):
        graph = kl.Graph(graph.topology, graph.weights)
    L = graph.laplacian_operator()
    if not isinstance(L, gx.SparseOperator):
        raise TypeError(f"expected a SparseOperator Laplacian, got {type(L)}")
    return L


def _diag_mask(op: gx.SparseOperator) -> Array:
    return jnp.asarray(np.asarray(op.pattern.rows) == np.asarray(op.pattern.cols))


def _affine_sparse(op: gx.SparseOperator, a, b) -> gx.SparseOperator:
    """``a * op + b * I`` on the stored pattern (which holds the diagonal)."""
    values = a * op.values + jnp.where(_diag_mask(op), b, 0.0)
    return eqx.tree_at(lambda o: o.values, op, values)


def _affine_kronecker_sum(op, a, b):
    """``a * op + b * I`` for a Kronecker sum, staying a Kronecker sum.

    ``a (A ⊕ B) + b I = (a A + b I) ⊕ a B``, recursively on ``A``.
    """
    if isinstance(op, gx.KroneckerSum):
        return gx.KroneckerSum(
            _affine_kronecker_sum(op.A, a, b), _affine_kronecker_sum(op.B, a, 0.0)
        )
    if isinstance(op, gx.SparseOperator):
        return _affine_sparse(op, a, b)
    return lx.MatrixLinearOperator(
        a * op.as_matrix() + b * jnp.eye(op.in_size()), lx.positive_semidefinite_tag
    )


def _check_no_isolated(graph: kl.AbstractGraph, what: str) -> None:
    if np.any(np.asarray(graph.degree()) <= 0):
        raise ValueError(
            f"{what} needs every node to have a neighbour (positive degree); "
            "the graph has isolated nodes"
        )


class Besag(AbstractComponent):
    r"""Intrinsic CAR (ICAR / Besag): precision $\tau R$, $R = D - W$.

    One sum-to-zero constraint per connected component. With
    ``scale_model=True`` (default) $R$ is BYM2-scaled per component, so
    $\tau$ is comparable across graphs (R-INLA's ``scale.model``).

    Args:
        graph: A kernellib graph.
        name: Site-name prefix.
        tau_prior: Prior on $\tau$; default ``PCPrecision(1, 0.01)``.
        scale_model: Scale the structure per connected component.

    Examples:
        >>> import jax, jax.numpy as jnp
        >>> import kernellib as kl
        >>> import pyrox_lgm as lgm
        >>> g = kl.graph_from_edges([0, 1, 3], [1, 2, 4], 5)  # two components
        >>> icar = lgm.Besag(g)
        >>> x = icar.prior({"tau": jnp.asarray(1.0)}).sample(jax.random.key(0))
        >>> [round(float(s), 6) + 0.0 for s in (x[:3].sum(), x[3:].sum())]
        [0.0, 0.0]
    """

    graph: kl.AbstractGraph
    structure: lx.AbstractLinearOperator
    null_space: Float[Array, "n c"]
    name: str = eqx.field(static=True)
    tau_prior: dist.Distribution

    def __init__(
        self,
        graph: kl.AbstractGraph,
        *,
        name: str = "besag",
        tau_prior: dist.Distribution | None = None,
        scale_model: bool = True,
    ) -> None:
        self.graph = graph
        self.structure = kl.structure_matrix(graph, scaled=scale_model)
        self.null_space = kl.graph_null_space(graph)
        self.name = name
        self.tau_prior = _tau_prior(tau_prior)

    @property
    def n_nodes(self) -> int:
        return self.structure.in_size()

    def theta_spec(self) -> dict[str, tuple[dist.Distribution, Transform]]:
        return {"tau": (self.tau_prior, default_transform(self.tau_prior))}

    def prior(
        self, theta: dict[str, Array], *, constraint: Constraint = "hard"
    ) -> gx.IntrinsicGMRF:
        return gx.IntrinsicGMRF(
            jnp.zeros(self.n_nodes),
            theta["tau"],
            self.structure,
            self.null_space,
            constraint=constraint,
        )


class BYM2(AbstractComponent):
    r"""BYM2 (Riebler et al., 2016), a mixture of unstructured and ICAR effects.

    $b = \tau^{-1/2}(\sqrt{1-\phi}\,v + \sqrt\phi\,u_\ast)$, with $v$
    unstructured and $u_\ast$ a scaled ICAR, so $\tau$ is the total
    precision of $b$ and $\phi$ the fraction of its variance that is spatially
    structured. The field is the pair $(b, u_\ast)$ (``n_nodes = 2n``),
    distributed as `gaussx.BYM2GMRF` with its exact constrained density;
    observations see $b$ only (``n_index = n``). The PC prior on $\phi$
    needs the spectrum of $R_\ast^{+}$, computed once here
    (`structure_spectrum`), as is $\log|R_\ast|_+$ for the normaliser.

    Args:
        graph: A kernellib graph.
        name: Site-name prefix.
        tau_prior: Prior on $\tau$; default ``PCPrecision(1, 0.01)``.
        phi_prior: Prior on $\phi$; default ``PCBYM2Phi(0.5, 2/3)`` on this
            graph's spectrum.
        spectrum_method: How to compute that spectrum (see
            `structure_spectrum`).

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> import pyrox_lgm as lgm
        >>> bym2 = lgm.BYM2(kl.grid_graph((4, 4)))
        >>> bym2.n_nodes, bym2.n_index
        (32, 16)
        >>> gmrf = bym2.prior({"tau": jnp.asarray(2.0), "phi": jnp.asarray(0.5)})
        >>> var = gmrf.marginal_variances()
        >>> var_b, var_u = var[:16], var[16:]  # var(b) = (1 - phi + phi var(u*)) / tau
        >>> bool(jnp.allclose(var_b, (0.5 + 0.5 * var_u) / 2.0))
        True
    """

    graph: kl.AbstractGraph
    structure: lx.AbstractLinearOperator
    null_space: Float[Array, "n c"]
    log_pdet: Float[Array, ""]
    name: str = eqx.field(static=True)
    tau_prior: dist.Distribution
    phi_prior: dist.Distribution

    def __init__(
        self,
        graph: kl.AbstractGraph,
        *,
        name: str = "bym2",
        tau_prior: dist.Distribution | None = None,
        phi_prior: dist.Distribution | None = None,
        spectrum_method: Literal["auto", "dense", "lanczos"] = "auto",
    ) -> None:
        self.graph = graph
        scaled = kl.structure_matrix(graph, scaled=True)
        # gaussx's BYM2 precision is assembled sparse, so a grid's Kronecker
        # sum is used only for its (exact, cheap) spectrum.
        self.structure = (
            kl.structure_matrix(kl.Graph(graph.topology, graph.weights), scaled=True)
            if isinstance(graph, kl.GridGraph)
            else scaled
        )
        self.null_space = kl.graph_null_space(graph)
        self.log_pdet = gx.pseudo_logdet(self.structure, structure="laplacian")
        self.name = name
        self.tau_prior = _tau_prior(tau_prior)
        if phi_prior is None:
            spectrum = structure_spectrum(
                scaled, self.null_space, method=spectrum_method
            )
            phi_prior = PCBYM2Phi(0.5, 2.0 / 3.0, structure_spectrum=spectrum)
        self.phi_prior = phi_prior

    @property
    def n_nodes(self) -> int:
        return 2 * self.structure.in_size()

    @property
    def n_index(self) -> int:
        return self.structure.in_size()

    def theta_spec(self) -> dict[str, tuple[dist.Distribution, Transform]]:
        return {
            "tau": (self.tau_prior, default_transform(self.tau_prior)),
            "phi": (self.phi_prior, default_transform(self.phi_prior)),
        }

    def prior(
        self, theta: dict[str, Array], *, constraint: Constraint = "hard"
    ) -> gx.BYM2GMRF:
        return gx.BYM2GMRF(
            self.structure,
            theta["tau"],
            theta["phi"],
            self.null_space,
            constraint=constraint,
            include_normalizer=True,
            log_pdet=self.log_pdet,
        )


class _ProperAreal(AbstractComponent):
    """Shared code of the proper areal fields (CAR, Leroux)."""

    graph: eqx.AbstractVar[kl.AbstractGraph]
    name: eqx.AbstractVar[str]
    tau_prior: eqx.AbstractVar[dist.Distribution]
    rho_prior: eqx.AbstractVar[dist.Distribution]
    spectrum: eqx.AbstractVar[Float[Array, " n"] | None]

    def theta_spec(self) -> dict[str, tuple[dist.Distribution, Transform]]:
        return {
            "tau": (self.tau_prior, default_transform(self.tau_prior)),
            "rho": (self.rho_prior, default_transform(self.rho_prior)),
        }


class CAR(_ProperAreal):
    r"""Proper CAR: precision $\tau(D - \rho W)$, $\rho \in [0, 1)$.

    $D$ is the diagonal of degrees and $W$ the weighted adjacency; $\rho \to
    1$ approaches the ICAR. Every node needs a neighbour. The
    log-determinant comes from the spectrum $\mu_i$ of
    $D^{-1/2} W D^{-1/2}$ (one dense ``eigh`` at construction, for
    ``n ≤ 10⁴``): $\log|Q| = n\log\tau + \sum_i \log d_i + \sum_i \log(1 - \rho\mu_i)$;
    larger graphs use gaussx's sparse Cholesky at each $\theta$.

    Args:
        graph: A kernellib graph without isolated nodes.
        name: Site-name prefix.
        tau_prior: Prior on $\tau$; default ``PCPrecision(1, 0.01)``.
        rho_prior: Prior on $\rho$; default ``Uniform(0, 1)``.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> import pyrox_lgm as lgm
        >>> car = lgm.CAR(kl.graph_from_edges([0, 1], [1, 2], 3))
        >>> Q = car.prior({"tau": jnp.asarray(1.0), "rho": jnp.asarray(0.5)})
        >>> Q.precision.as_matrix().tolist()
        [[1.0, -0.5, 0.0], [-0.5, 2.0, -0.5], [0.0, -0.5, 1.0]]
    """

    graph: kl.AbstractGraph
    laplacian: gx.SparseOperator
    log_degree: Float[Array, ""]
    spectrum: Float[Array, " n"] | None
    name: str = eqx.field(static=True)
    tau_prior: dist.Distribution
    rho_prior: dist.Distribution

    def __init__(
        self,
        graph: kl.AbstractGraph,
        *,
        name: str = "car",
        tau_prior: dist.Distribution | None = None,
        rho_prior: dist.Distribution | None = None,
    ) -> None:
        _check_no_isolated(graph, "CAR")
        self.graph = graph
        self.laplacian = _sparse_laplacian(graph)
        d = jnp.asarray(graph.degree())
        self.log_degree = jnp.sum(jnp.log(d))
        n = d.shape[0]
        if n <= _DENSE_SPECTRUM_MAX:
            L = self.laplacian.as_matrix()
            s = 1.0 / jnp.sqrt(d)
            # D^-1/2 W D^-1/2 = I - D^-1/2 L D^-1/2.
            self.spectrum = 1.0 - jnp.linalg.eigvalsh(s[:, None] * L * s[None, :])
        else:
            self.spectrum = None
        self.name = name
        self.tau_prior = _tau_prior(tau_prior)
        self.rho_prior = _rho_prior(rho_prior)

    @property
    def n_nodes(self) -> int:
        return self.laplacian.in_size()

    def prior(
        self, theta: dict[str, Array], *, constraint: Constraint = "hard"
    ) -> gx.GaussianMRF:
        del constraint  # proper
        tau, rho = theta["tau"], theta["rho"]
        # tau (D - rho W) = tau ((1 - rho) D + rho L): degrees on the
        # diagonal, rho times the Laplacian's off-diagonal.
        Q = eqx.tree_at(
            lambda o: o.values,
            self.laplacian,
            tau
            * jnp.where(
                _diag_mask(self.laplacian),
                self.laplacian.values,
                rho * self.laplacian.values,
            ),
        )
        log_det = None
        if self.spectrum is not None:
            n = self.n_nodes
            log_det = (
                n * jnp.log(tau)
                + self.log_degree
                + jnp.sum(jnp.log1p(-rho * self.spectrum))
            )
        return gx.GaussianMRF(jnp.zeros(self.n_nodes), Q, log_det_precision=log_det)


class Leroux(_ProperAreal):
    r"""Leroux CAR: precision $\tau(\rho R + (1 - \rho) I)$, $\rho \in [0, 1]$.

    Interpolates between independent effects ($\rho = 0$) and the ICAR
    ($\rho \to 1$). On a face-connected `kernellib.GridGraph` the precision
    stays a `gaussx.KroneckerSum` (exact eigen-structured solves and
    log-determinant); elsewhere it is a `gaussx.SparseOperator`, with
    $\log|Q| = n\log\tau + \sum_i \log(\rho\lambda_i + 1 - \rho)$ from the
    Laplacian spectrum $\lambda_i$ cached at construction (``n ≤ 10⁴``).

    Args:
        graph: A kernellib graph.
        name: Site-name prefix.
        tau_prior: Prior on $\tau$; default ``PCPrecision(1, 0.01)``.
        rho_prior: Prior on $\rho$; default ``Uniform(0, 1)``.
        scale_model: Use the BYM2-scaled $R$ (per connected component).

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> import pyrox_lgm as lgm
        >>> lr = lgm.Leroux(kl.grid_graph((3, 3)))
        >>> gmrf = lr.prior({"tau": jnp.asarray(1.0), "rho": jnp.asarray(0.0)})
        >>> bool(jnp.allclose(gmrf.precision.as_matrix(), jnp.eye(9)))  # IID
        True
    """

    graph: kl.AbstractGraph
    structure: gx.KroneckerSum | gx.SparseOperator
    spectrum: Float[Array, " n"] | None
    name: str = eqx.field(static=True)
    tau_prior: dist.Distribution
    rho_prior: dist.Distribution

    def __init__(
        self,
        graph: kl.AbstractGraph,
        *,
        name: str = "leroux",
        tau_prior: dist.Distribution | None = None,
        rho_prior: dist.Distribution | None = None,
        scale_model: bool = False,
    ) -> None:
        self.graph = graph
        R = kl.structure_matrix(graph, scaled=scale_model)
        if not isinstance(R, gx.KroneckerSum | gx.SparseOperator):
            raise TypeError(f"unsupported structure type {type(R)}")
        self.structure = R
        n = R.in_size()
        if isinstance(R, gx.KroneckerSum) or n <= _DENSE_SPECTRUM_MAX:
            self.spectrum = _dense_eigvals(R)
        else:
            self.spectrum = None
        self.name = name
        self.tau_prior = _tau_prior(tau_prior)
        self.rho_prior = _rho_prior(rho_prior)

    @property
    def n_nodes(self) -> int:
        return self.structure.in_size()

    def prior(
        self, theta: dict[str, Array], *, constraint: Constraint = "hard"
    ) -> gx.GaussianMRF:
        del constraint  # proper
        tau, rho = theta["tau"], theta["rho"]
        if isinstance(self.structure, gx.KroneckerSum):
            Q = _affine_kronecker_sum(self.structure, tau * rho, tau * (1.0 - rho))
        else:
            Q = _affine_sparse(self.structure, tau * rho, tau * (1.0 - rho))
        log_det = None
        if self.spectrum is not None:
            log_det = self.n_nodes * jnp.log(tau) + jnp.sum(
                jnp.log(rho * self.spectrum + 1.0 - rho)
            )
        return gx.GaussianMRF(jnp.zeros(self.n_nodes), Q, log_det_precision=log_det)
