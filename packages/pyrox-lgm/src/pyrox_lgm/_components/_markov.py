r"""Markov GP component: a gaussx SDE kernel on sorted times as a GMRF (P7).

An SDE kernel of state dimension $d$ observed at $N$ sorted times is a
Gauss-Markov chain over the states $x_t \in \mathbb R^d$ (the function and
its derivatives). Its precision is block tridiagonal,

$$\Lambda = -2\,\Theta(A, Q, \Sigma_0),$$

with $A_k, Q_k$ the kernel's transitions and model-error covariances over the
$N - 1$ steps, $\Sigma_0$ the initial state covariance, and
$\Theta = -\tfrac12 \Lambda$ the precision part of what
`gaussx.ssm_to_naturals` returns (pinned to `AR1` and R-INLA's ``ar1`` in
``tests/test_markov_precision_prototype.py``).
"""

from __future__ import annotations

import equinox as eqx
import gaussx as gx
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from jaxtyping import Array, ArrayLike, Float, Int
from numpyro.distributions.transforms import Transform

from pyrox_lgm._components._base import (
    AbstractComponent,
    Constraint,
    default_transform,
)
from pyrox_lgm._priors import PCMatern, PCPrecision


def _chain_precision(
    sde: gx.SDEKernel, times: Float[Array, " N"], initial_covariance: Array
) -> gx.BlockTriDiag:
    """$\\Lambda$ of the discretised ``sde`` on ``times``, ``(N d) x (N d)``."""
    A, Q = sde.discretise_sequence(jnp.diff(times))
    d = sde.state_dim
    _, neg_half_precision = gx.ssm_to_naturals(
        A, Q, jnp.zeros(d, dtype=times.dtype), initial_covariance
    )
    return -2.0 * neg_half_precision


class MarkovGP(AbstractComponent):
    r"""A temporal Gaussian process with a gaussx SDE kernel, in precision form.

    The latent field is the kernel's state $x_t \in \mathbb R^d$ at each of
    the $N$ sorted ``times`` (node $t d + c$ holds component $c$ of $x_t$),
    with the block-tridiagonal precision of its discretisation (module
    docstring). Observations read the function value $f(t) = H x_t$, so
    `n_index` is $N$ while `n_nodes` is $N d$, and an observation index is
    a time index. ``inla()``'s summaries of the field (``result.random``) are
    over all $N d$ nodes; ``[::d]`` picks $f$.

    Two kernel families:

    - `gaussx.MaternSDE` (orders 0, 1, 2: Matérn-1/2, 3/2, 5/2), a proper
      stationary field. The hyperparameter ``"range_sigma"`` is the pair
      $[\rho, \sigma]$, the practical range $\rho = 2\ell$ ($\ell$ the
      kernel's lengthscale; correlation 0.14 at distance $\rho$ for each
      order) and the marginal sd $\sigma$, under the joint `PCMatern` prior in
      dimension 1, as `SPDE` uses in two. Matérn-1/2 on a regular grid is
      `AR1` with $\rho_{AR} = e^{-2\Delta/\rho}$ and $\tau = 1/\sigma^2$, for
      positive correlation only.
    - `gaussx.IntegratedWienerSDE` (order $p$), an intrinsic field: the
      top derivative $f^{(p)}$ is white noise of spectral density $1/\tau$,
      with hyperparameter ``"tau"``. Its null space is the $p + 1$ polynomial
      paths of degree at most $p$ (no initial distribution: the kernel's
      ``P_0`` is not used). Inside ``inla()`` only $\sum_t f(t) = 0$ is
      imposed, as `RW2` does, and the trends are left to the data; the
      NumPyro face soft-constrains the whole null space, so add polynomial
      fixed effects there if the data have a trend.

    Only the kernel's family and order are read from ``sde``; its variance,
    lengthscale or diffusion come from $\theta$.

    Conditioning: a smooth kernel's precision grows ill-conditioned as the
    spacing shrinks relative to $\ell$. Measured for Matérn-5/2 on 25 regular
    times, $\mathrm{cond}(\Lambda)$ is $5 \times 10^2$ at $\Delta = 0.77\ell$,
    $1.5 \times 10^8$ at $0.077\ell$ and $5.6 \times 10^{13}$ at
    $0.0077\ell$, where the implied covariance of $f$ is off the kernel by
    $2 \times 10^{-15}$, $1.5 \times 10^{-9}$ and $2.6 \times 10^{-4}$
    (relative to $\sigma^2$). Prefer a lower order, or coarser times, when
    the spacing is far below the lengthscale.

    Args:
        sde: A `gaussx.MaternSDE` or `gaussx.IntegratedWienerSDE`.
        times: Strictly increasing times, at least 2.
        name: Site-name prefix.
        prior: Joint prior on ``[range, sigma]`` (Matérn); default
            ``PCMatern(span / 5, 0.5, 1.0, 0.01, d=1)``, ``span`` the
            length of ``times``, as `SPDE`'s default.
        tau_prior: Prior on $\tau$ (integrated Wiener); default
            ``PCPrecision(1, 0.01)``.

    Raises:
        TypeError: If ``sde`` is another kernel.
        ValueError: If ``times`` is not strictly increasing, has fewer than
            2 entries, or a prior is given for the other family.

    Examples:
        >>> import gaussx as gx
        >>> import jax.numpy as jnp
        >>> import pyrox_lgm as lgm
        >>> times = jnp.linspace(0.0, 10.0, 20)
        >>> trend = lgm.MarkovGP(gx.MaternSDE(1.0, 1.0, order=1), times, name="trend")
        >>> trend.n_nodes, trend.n_index
        (40, 20)
        >>> gmrf = trend.prior({"range_sigma": jnp.array([3.0, 2.0])})
        >>> [round(float(v), 4) for v in gmrf.marginal_variances()[::2][:3]]
        [4.0, 4.0, 4.0]

        An integrated Wiener process is intrinsic, with the polynomials of
        degree at most its order as null space:

        >>> spline = lgm.MarkovGP(gx.IntegratedWienerSDE(1.0, order=1), times)
        >>> spline.null_space.shape
        (40, 2)
    """

    sde: gx.SDEKernel
    times: Float[Array, " N"]
    observation_row: Float[Array, " d"]
    name: str = eqx.field(static=True)
    order: int = eqx.field(static=True)
    prior_range_sigma: dist.Distribution | None
    tau_prior: dist.Distribution | None
    structure: gx.BlockTriDiag | None
    null_space: Float[Array, "Nd c"] | None

    def __init__(
        self,
        sde: gx.SDEKernel,
        times: Float[ArrayLike, " N"],
        *,
        name: str = "markov",
        prior: dist.Distribution | None = None,
        tau_prior: dist.Distribution | None = None,
    ) -> None:
        t = jnp.asarray(times, dtype=float)
        t_host = np.asarray(t)
        if t_host.ndim != 1 or t_host.size < 2:
            raise ValueError(
                f"times must be 1-D with at least 2 entries, got {t.shape}"
            )
        if np.any(np.diff(t_host) <= 0):
            raise ValueError("times must be strictly increasing")
        self.sde = sde
        self.times = t
        self.name = name
        self.observation_row = jnp.asarray(sde.sde_params().H, dtype=t.dtype)[0]
        if isinstance(sde, gx.MaternSDE):
            self.order = sde.order
            if tau_prior is not None:
                raise ValueError("tau_prior is for an IntegratedWienerSDE; pass prior=")
            if prior is None:
                span = float(t_host[-1] - t_host[0])
                prior = PCMatern(span / 5.0, 0.5, 1.0, 0.01, d=1)
            self.prior_range_sigma = prior
            self.tau_prior = None
            self.structure = None
            self.null_space = None
        elif isinstance(sde, gx.IntegratedWienerSDE):
            self.order = sde.order
            if prior is not None:
                raise ValueError("prior= is for a MaternSDE; pass tau_prior=")
            self.prior_range_sigma = None
            self.tau_prior = PCPrecision(1.0, 0.01) if tau_prior is None else tau_prior
            unit = gx.IntegratedWienerSDE(
                jnp.asarray(1.0, dtype=t.dtype), order=sde.order
            )
            d = unit.state_dim
            eye = jnp.eye(d, dtype=t.dtype)
            # Any P_0 enters only the first diagonal block, as P_0^{-1}:
            # build with P_0 = I and take it out, a diffuse start.
            full = _chain_precision(unit, t, eye)
            self.structure = gx.BlockTriDiag(
                full.diagonal.at[0].add(-eye), full.sub_diagonal
            )
            # Null space: the noise-free paths x_{k+1} = A_k x_k from each
            # unit initial state; column 0 is the constant f (all
            # derivatives zero), the only direction inla() constrains.
            A, _ = unit.discretise_sequence(jnp.diff(t))
            states = [eye]
            for A_k in A:
                states.append(A_k @ states[-1])
            basis = jnp.stack(states).reshape(-1, d)
            self.null_space, _ = jnp.linalg.qr(basis)
        else:
            raise TypeError(
                "MarkovGP takes a gaussx.MaternSDE or gaussx.IntegratedWienerSDE, "
                f"got {type(sde).__name__}"
            )

    @property
    def state_dim(self) -> int:
        """State dimension $d$ of the kernel."""
        return self.sde.state_dim

    @property
    def n_nodes(self) -> int:
        return self.times.shape[0] * self.state_dim

    @property
    def n_index(self) -> int:
        return self.times.shape[0]

    def theta_spec(self) -> dict[str, tuple[dist.Distribution, Transform]]:
        if self.prior_range_sigma is not None:
            p = self.prior_range_sigma
            return {"range_sigma": (p, default_transform(p))}
        assert self.tau_prior is not None
        return {"tau": (self.tau_prior, default_transform(self.tau_prior))}

    def prior(
        self,
        theta: dict[str, Array],
        *,
        constraint: Constraint = "hard",
        soft_constraint_scale: float = 1e-3,
    ) -> gx.GaussianMRF | gx.IntrinsicGMRF:
        loc = jnp.zeros(self.n_nodes, dtype=self.times.dtype)
        if self.structure is None:
            range_, sigma = theta["range_sigma"][0], theta["range_sigma"][1]
            sde = gx.MaternSDE(sigma**2, range_ / 2.0, order=self.order)
            return gx.GaussianMRF(
                loc, _chain_precision(sde, self.times, sde.initial_covariance())
            )
        assert self.null_space is not None
        return gx.IntrinsicGMRF(
            loc,
            theta["tau"],
            self.structure,
            self.null_space,
            constraint=constraint,
            soft_constraint_scale=soft_constraint_scale,
        )

    def constraint_basis(self, gmrf):
        if self.null_space is None:
            return None
        return self.null_space[:, :1]

    def projector(self, index: Int[ArrayLike, " n_obs"]) -> gx.SparseOperator:
        """``(n_obs, N d)``: row $j$ reads $f = H x$ at time ``index[j]``.

        Args:
            index: Time index of each observation, integers in ``[0, N)``.
        """
        index = np.asarray(index)
        if index.ndim != 1:
            raise ValueError(f"index must be 1-D, got shape {index.shape}")
        if index.size and (index.min() < 0 or index.max() >= self.n_index):
            raise ValueError(
                f"index out of range for {self.n_index} times: "
                f"[{index.min()}, {index.max()}]"
            )
        d = self.state_dim
        H = np.asarray(self.observation_row)
        (c,) = np.nonzero(H)
        m = index.shape[0]
        rows = np.repeat(np.arange(m), c.size)
        cols = (index[:, None] * d + c[None, :]).ravel()
        vals = jnp.tile(jnp.asarray(H[c]), m)
        return gx.SparseOperator.from_coo(rows, cols, vals, (m, self.n_nodes))

    def field_at(
        self, x: Float[Array, " Nd"], index: Int[ArrayLike, " n_obs"] | None = None
    ) -> Array:
        f = x.reshape(self.n_index, self.state_dim) @ self.observation_row
        return f if index is None else f[jnp.asarray(index)]
