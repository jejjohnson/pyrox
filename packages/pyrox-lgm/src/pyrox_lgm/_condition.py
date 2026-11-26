r"""An LGM conditioned at fixed hyperparameters.

`LGMPrior.condition` and `LGMPrior.condition_nongauss` return the Gaussian
$x \mid y, \theta \approx \mathcal N(\hat x, \Sigma)$ of the latent vector at
its mode, with

$$
\Sigma = H^{-1} - W (V^\top W)^{-1} W^\top,\qquad
H = Q(\theta) + A^\top \Lambda A,\qquad W = H^{-1} V,
$$

$\Lambda$ the likelihood's negative second derivatives and $V$ the basis of
the hard constraints (no correction without any). $H$ need only be positive
definite on $\{V^\top x = 0\}$: where it is singular off it, gaussx's factor is
that of $H + V B V^\top$ for a positive diagonal $B$, which changes nothing on
that subspace.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

import einx
import equinox as eqx
import gaussx as gx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
from argminax.solve import LaplaceModeResult
from jaxtyping import Array, ArrayLike, Bool, Float, Int, PRNGKeyArray


if TYPE_CHECKING:
    from pyrox_lgm._model import LGMPrior


# Predictor rows per batch of solves: bounds the memory of `predict`'s
# variances at this many latent vectors, whatever the number of rows.
_ROWS_PER_BATCH = 256


class _ConditionedLatent(eqx.Module):
    """The Gaussian of the latent vector at the mode, shared by both results."""

    prior: LGMPrior
    theta: Mapping[str, Array]
    fit: LaplaceModeResult
    constraints: Float[Array, "n c"] | None
    null_space: Float[Array, "n k"] | None

    @property
    def posterior(self) -> gx.GaussianMRF | gx.ConstrainedGMRF:
        """The latent vector's posterior as a gaussx field, hard constraints kriged."""
        field = gx.GaussianMRF(self.fit.mode, self.fit.hessian)
        if self.constraints is None:
            return field
        # Vᵀ x = 0
        A_c = self.constraints.T
        # H may be singular along the components' null directions
        return field.condition_on_constraints(
            A_c, jnp.zeros(A_c.shape[0]), null_space=self.null_space
        )

    def predict_latent(self) -> tuple[Float[Array, " n"], Float[Array, " n"]]:
        """Posterior mean and variance of every latent node.

        Returns:
            ``(mean, var)`` over the latent vector $(x_1, \\dots, x_k, \\beta)$,
            each shaped ``(n_latent,)``; ``prior.slices()`` says which
            entries are which component's.
        """
        return self.fit.mode, self.posterior.marginal_variances()

    def _predict(
        self, new_data: Mapping[str, ArrayLike], projector: gx.SparseOperator | None
    ) -> tuple[Float[Array, " M"], Float[Array, " M"]]:
        A = self.prior.projector(new_data) if projector is None else projector
        mean = A.mv(self.fit.mode)
        offset = self.prior.offset(new_data)
        if offset is not None:
            mean = mean + offset
        return mean, self._predictor_variance(A)

    def _predictor_variance(self, A: gx.SparseOperator) -> Float[Array, " M"]:
        r"""$a_i^\top \Sigma a_i$ for every row of ``A``, one solve per row.

        A new row may couple nodes that no observation couples, so the
        Takahashi selected inverse does not hold its covariances.
        """
        solve = self.fit.factor.solve
        n = A.in_size()
        cols, vals = row_entries(A)
        kriging = None
        if self.constraints is not None:
            kriging = constraint_kriging(solve, self.constraints)

        def variance(row):
            idx, val = row
            # a_iᵀ H⁻¹ a_i
            solved = solve(jnp.zeros(n, val.dtype).at[idx].add(val))
            var = einx.dot("k, k ->", val, solved[idx])
            if kriging is not None:
                W, G = kriging
                # a_iᵀ W S⁻¹ Wᵀ a_i
                left = einx.dot("k, k c -> c", val, W[idx])
                right = einx.dot("k, k c -> c", val, G[idx])
                var = var - einx.dot("c, c ->", left, right)
            return jnp.maximum(var, 0.0)

        return jax.lax.map(variance, (cols, vals), batch_size=_ROWS_PER_BATCH)

    def sample(
        self,
        key: PRNGKeyArray,
        n_samples: int = 1,
        new_data: Mapping[str, ArrayLike] | None = None,
        *,
        projector: gx.SparseOperator | None = None,
    ) -> Float[Array, "S n"] | Float[Array, "S M"]:
        r"""Joint posterior draws of the latent vector, or of $\eta$ at ``new_data``.

        Each draw is a whole field, so a quantity of several nodes (a
        maximum, a total over a region, a difference) gets its joint
        uncertainty, which the marginals of `predict` cannot give.

        Args:
            key: PRNG key.
            n_samples: Number of joint draws $S$.
            new_data: Where to read the linear predictor $\eta = A x + o$;
                ``None`` returns the latent vector itself.
            projector: ``prior.projector(new_data)``, as in `predict`.

        Returns:
            The draws, shaped ``(S, n_latent)``, or ``(S, M)`` with
            ``new_data`` of ``M`` rows.
        """
        draws = self.posterior.sample(key, (n_samples,))
        if new_data is None:
            return draws
        A = self.prior.projector(new_data) if projector is None else projector
        eta = jax.vmap(A.mv)(draws)
        offset = self.prior.offset(new_data)
        return eta if offset is None else eta + offset


class ConditionedLGM(_ConditionedLatent):
    r"""An `LGMPrior` conditioned on Gaussian observations, at fixed $\theta$.

    What `LGMPrior.condition` returns: the exact Gaussian of the latent
    vector. ``log_marginal`` is the evidence of the same fit.

    Attributes:
        prior: The `LGMPrior`.
        theta: The hyperparameter values conditioned at.
        fit: argminax's fit: the mode $\hat x$, the precision $H$ and its
            factor.
        constraints: Basis $V$ of the hard constraints, or ``None``.
        null_space: Null spaces of the intrinsic components, or ``None``.
        noise_var: The observation noise variance $\sigma^2$.

    Examples:
        >>> import jax
        >>> import numpy as np
        >>> import pyrox_lgm as lgm
        >>> prior = lgm.LGMPrior((lgm.RW1(6, name="t"),), lgm.FixedEffects(("z",)))
        >>> data = {"y": np.sin(np.arange(4.0)), "t": np.arange(4), "z": np.ones(4)}
        >>> posterior = prior.condition({"t.tau": 2.0}, data, 0.1)

        The linear predictor at nodes that were not observed:

        >>> new = {"t": np.array([4, 5]), "z": np.zeros(2)}
        >>> mean, var = posterior.predict(new)
        >>> mean.shape, var.shape
        ((2,), (2,))
        >>> posterior.sample(jax.random.key(0), 10, new).shape
        (10, 2)
    """

    noise_var: Float[Array, ""]

    @property
    def log_marginal(self) -> Float[Array, ""]:
        r"""$\log p(y \mid \theta)$, as `LGMPrior.log_marginal` gives it."""
        return self.fit.log_marginal

    def predict(
        self,
        new_data: Mapping[str, ArrayLike],
        *,
        projector: gx.SparseOperator | None = None,
        include_noise: bool = False,
    ) -> tuple[Float[Array, " M"], Float[Array, " M"]]:
        r"""Mean and variance of the linear predictor $\eta = A x + o$ at ``new_data``.

        Args:
            new_data: The component indices (or locations) and covariates of
                the ``M`` new rows, and an optional ``"offset"``; no ``"y"``.
            projector: ``prior.projector(new_data)``; required under ``jit``
                when ``new_data`` is traced, or has an `SPDE` at locations.
            include_noise: Add the noise variance $\sigma^2$, giving the
                variance of a new observation. The mean is the same.

        Returns:
            ``(mean, var)``, each shaped ``(M,)``.
        """
        mean, var = self._predict(new_data, projector)
        return mean, var + self.noise_var if include_noise else var


class ConditionedNonGaussLGM(_ConditionedLatent):
    r"""An `LGMPrior` conditioned under any likelihood, at fixed $\theta$.

    What `LGMPrior.condition_nongauss` returns: the Laplace approximation,
    the Gaussian at the mode of the latent vector. It is the approximation
    `inla` makes at each of its design points with ``strategy="gaussian"``.

    Attributes:
        prior: The `LGMPrior`.
        theta: The hyperparameter values conditioned at.
        fit: argminax's fit: the mode $\hat x$, the precision $H$ and its
            factor.
        constraints: Basis $V$ of the hard constraints, or ``None``.
        null_space: Null spaces of the intrinsic components, or ``None``.

    Examples:
        >>> import numpy as np
        >>> import pyrox_lgm as lgm
        >>> prior = lgm.LGMPrior((lgm.IID(3, name="g"),))
        >>> data = {"y": np.array([1, 0, 2, 4]), "g": np.array([0, 1, 1, 2])}
        >>> posterior = prior.condition_nongauss({"g.tau": 1.0}, lgm.Poisson(), data)
        >>> bool(posterior.converged)
        True
        >>> mean, var = posterior.predict({"g": np.arange(3)})
        >>> mean.shape
        (3,)
    """

    @property
    def log_marginal_approx(self) -> Float[Array, ""]:
        r"""The Laplace approximation of $\log p(y \mid \theta)$."""
        return self.fit.log_marginal

    @property
    def converged(self) -> Bool[Array, ""]:
        """Whether the Newton loop to the mode converged."""
        return self.fit.converged

    def predict(
        self,
        new_data: Mapping[str, ArrayLike],
        *,
        projector: gx.SparseOperator | None = None,
    ) -> tuple[Float[Array, " M"], Float[Array, " M"]]:
        r"""Mean and variance of the linear predictor $\eta = A x + o$ at ``new_data``.

        The predictor, not the response: pass draws of it (`sample`) through
        the likelihood for a predictive distribution of new observations.

        Args:
            new_data: The component indices (or locations) and covariates of
                the ``M`` new rows, and an optional ``"offset"``; no ``"y"``.
            projector: ``prior.projector(new_data)``; required under ``jit``
                when ``new_data`` is traced, or has an `SPDE` at locations.

        Returns:
            ``(mean, var)``, each shaped ``(M,)``.
        """
        return self._predict(new_data, projector)


def row_entries(A: gx.SparseOperator) -> tuple[Int[Array, "M k"], Float[Array, "M k"]]:
    """Columns and values of each row of ``A``, padded with zeros to one length.

    The layout is host work on the static pattern; the values stay traced.
    """
    rows, cols = np.asarray(A.pattern.rows), np.asarray(A.pattern.cols)
    order = np.argsort(rows, kind="stable")
    counts = np.bincount(rows, minlength=A.out_size())
    width = max(int(counts.max(initial=0)), 1)
    slot = np.arange(rows.size) - np.repeat(np.cumsum(counts) - counts, counts)
    idx = np.zeros((A.out_size(), width), dtype=int)
    take = np.full((A.out_size(), width), rows.size)  # the appended zero
    idx[rows[order], slot] = cols[order]
    take[rows[order], slot] = order
    vals = jnp.append(A.values, jnp.zeros((), A.values.dtype))
    return jnp.asarray(idx), vals[jnp.asarray(take)]


def constraint_kriging(
    solve: Callable[[Float[Array, " n"]], Float[Array, " n"]],
    V: Float[Array, "n c"],
) -> tuple[Float[Array, "n c"], Float[Array, "n c"]]:
    r"""The terms that krige $H^{-1}$ onto the hard constraints $V^\top x = 0$.

    With $W = H^{-1} V$ and $G = W (V^\top W)^{-1}$, the constrained covariance
    is $\Sigma = H^{-1} - G W^\top$, so $\Sigma_{ab} = (H^{-1})_{ab} - G_a \cdot W_b$.

    Args:
        solve: $v \mapsto H^{-1} v$, the factor of the posterior precision.
        V: Basis of the hard constraints.

    Returns:
        ``(W, G)``, each shaped like ``V``.
    """
    # W = H⁻¹ V
    W = jax.vmap(solve, in_axes=1, out_axes=1)(V)
    # S = Vᵀ W
    S = einx.dot("n a, n b -> a b", V, W)
    S_op = lx.MatrixLinearOperator(S, lx.positive_semidefinite_tag)
    # G = W S⁻¹
    return W, gx.solve_columns(S_op, W.T).T
