r"""Normalizing Kalman Filter prior on top of the gaussx state-space surface.

The Normalizing Kalman Filter (de Bézenac et al., 2020) composes a
linear-Gaussian state-space base with an invertible per-timestep
observation warp $G$:

$$
x_t = A x_{t-1} + q_t, \qquad y_t = G(H x_t + r_t).
$$

Because the warp acts on **observations** rather than on the latent state,
the change-of-variables term

$$
\log p(y_{1:T}) = \log p_{\mathrm{LGSSM}}\!\left(G^{-1}(y_{1:T})\right)
  + \sum_t \log\left|\det \frac{\partial G^{-1}}{\partial y_t}\right|
$$

is independent of $x_t$. The Kalman recursion therefore stays exact, the
marginal likelihood stays closed-form, and ``numpyro.factor`` over the
collapsed marginal is valid — exactly as `pyrox_gp.markov_gp_factor`
is for the unwarped scalar case. In particular, **none of
`pyrox_gp._inference_nongauss_markov` is needed here**: a warped
observation model is *not* a non-Gaussian likelihood in the sense that
module handles (no CVI, no posterior linearisation, no Gauss-Newton), and
no changes to that module accompany this one.

The base `gaussx.LGSSM` / `gaussx.MaskedLGSSM` is a hard
``pyrox-gp`` dependency, so the **unwarped** multivariate state-space
model works with no flow dependencies at all. Passing a ``warp`` requires
the ``flows`` extra (``pip install 'pyrox-gp[flows]'``), which supplies
``gauss_flows.normalizing_kalman_filter`` — including its masked
change-of-variables handling and its rejection of channel-mixing warps
over a masked base.
"""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType

import einx
import equinox as eqx
import gaussx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import numpyro
from flowjax.bijections import AbstractBijection
from gaussx import LGSSM, MaskedLGSSM
from jaxtyping import Array, Bool, Float


def _require_flows() -> ModuleType:
    """Import ``gauss_flows``, with an install hint when it is absent."""
    try:
        import gauss_flows  # ty: ignore[unresolved-import]
    except ImportError as exc:
        raise ImportError(
            "A non-None warp on NormalizingKalmanPrior requires the flows "
            "extra: pip install 'pyrox-gp[flows]'"
        ) from exc
    return gauss_flows


def _dense(
    operand: Float[Array, ...] | lx.AbstractLinearOperator,
) -> Float[Array, ...]:
    """Materialise a lineax operator; pass dense arrays through."""
    if isinstance(operand, lx.AbstractLinearOperator):
        return operand.as_matrix()
    return jnp.asarray(operand)


class NormalizingKalmanPrior(eqx.Module):
    r"""Normalizing Kalman Filter prior over a multivariate time grid.

    Wraps a `gaussx.LGSSM` (or `gaussx.MaskedLGSSM`) base and an
    optional per-timestep observation warp into the ``pyrox_gp`` model
    surface, so the state-space parameters and the warp can carry NumPyro
    priors and be fitted with any ``pyrox.inference`` driver
    (`pyrox.inference.EnsembleMAP`, `pyrox.inference.EnsembleVI`, SVI,
    MCMC).

    Because the warp acts on observations rather than on the latent
    state, the log-determinant term is independent of $x_t$ and the
    Kalman recursion stays exact — the marginal likelihood is
    closed-form and no non-Gaussian inference strategy is required.
    With ``warp=None`` this reduces *exactly* to the base LGSSM marginal
    likelihood, with no flow dependencies at runtime.

    The state-space parameters $(A, H, Q, R)$ are rotationally
    non-identifiable in the same way a latent-factor mixing matrix is,
    so ensembling over seeds (`pyrox.inference.EnsembleMAP`) is the
    recommended fitting pattern rather than a convenience.

    Attributes:
        base: `gaussx.LGSSM` or `gaussx.MaskedLGSSM` with event shape
            ``(T, M)``. Available without the ``flows`` extra. A
            `gaussx.MaskedLGSSM` base contributes its ``obs_mask``
            whenever no call-time ``mask`` is given.
        warp: Optional bijection with event shape ``(M,)``, applied
            independently at each step (requires the ``flows`` extra).
            ``None`` means the identity, in which case this is exactly
            the base LGSSM. Conditional warps (non-``None``
            ``cond_shape``) are rejected. A channel-mixing warp is
            usable with `log_marginal` on an unmasked base, but nothing
            else: ``gauss_flows`` refuses it over a masked base (masking
            and a non-diagonal Jacobian do not commute), and `predict`
            refuses it because its per-channel quadrature would be wrong
            (see there). Put cross-channel structure in ``H`` and ``R``
            instead.
        mean_fn: Optional callable ``times -> (T, M)`` evaluated on the
            integer grid ``0, ..., T-1`` of the base. The mean acts in
            **observation space** — it is subtracted from ``y`` before
            the inverse warp and filtering, and added back at predict
            time — so the model is $y_t = m(t) + G(H x_t + r_t)$ and
            conjugacy is untouched.

    Examples:
        >>> import jax.numpy as jnp, jax.random as jr
        >>> from gaussx import LGSSM
        >>> from pyrox_gp import NormalizingKalmanPrior
        >>> T, M = 12, 2
        >>> base = LGSSM(0.9 * jnp.eye(M), jnp.eye(M), 0.1 * jnp.eye(M),
        ...              0.2 * jnp.eye(M), jnp.zeros(M), jnp.eye(M), n_steps=T)
        >>> prior = NormalizingKalmanPrior(base)   # unwarped: no flow deps
        >>> y = base.sample(jr.key(0))
        >>> prior.log_marginal(y).shape
        ()
        >>> mean, var = prior.predict(y, n_ahead=3)
        >>> mean.shape
        (15, 2)
    """

    base: LGSSM
    warp: AbstractBijection | None = None
    mean_fn: Callable[[Float[Array, " T"]], Float[Array, "T M"]] | None = None

    def __init__(
        self,
        base: LGSSM,
        warp: AbstractBijection | None = None,
        mean_fn: Callable[[Float[Array, " T"]], Float[Array, "T M"]] | None = None,
    ) -> None:
        if warp is not None:
            _require_flows()
            if warp.cond_shape is not None:
                raise ValueError(
                    "Conditional warps are not supported: the collapsed "
                    "marginal assumes one fixed observation warp per channel. "
                    f"Got cond_shape={warp.cond_shape!r}."
                )
            n_channels = base.event_shape[1]
            if len(warp.shape) != 1 or warp.shape[0] != n_channels:
                raise ValueError(
                    f"warp must have event shape ({n_channels},) to match the "
                    f"base's channel count; got {warp.shape!r}. Lift scalar "
                    "bijections over the channel axis first, e.g. "
                    "Vmap(RationalQuadraticSpline(...), in_axes=None, "
                    f"axis_size={n_channels})."
                )
        self.base = base
        self.warp = warp
        self.mean_fn = mean_fn

    @property
    def n_steps(self) -> int:
        """Sequence length ``T`` of the base."""
        return self.base.event_shape[0]

    @property
    def n_channels(self) -> int:
        """Observation dimension ``M`` of the base."""
        return self.base.event_shape[1]

    def _mean_grid(self, n_steps: int, dtype: np.dtype) -> Float[Array, "T M"]:
        """Mean values on the integer grid ``0, ..., n_steps - 1``."""
        if self.mean_fn is None:
            return jnp.zeros((n_steps, self.n_channels), dtype=dtype)
        return jnp.asarray(self.mean_fn(jnp.arange(n_steps, dtype=dtype)))

    def _effective_mask(
        self, mask: Bool[Array, "T M"] | None
    ) -> Bool[Array, "T M"] | None:
        """Call-time mask if given, else a `MaskedLGSSM` base's own mask."""
        if mask is not None:
            return jnp.asarray(mask, dtype=bool)
        if isinstance(self.base, MaskedLGSSM):
            return self.base.obs_mask
        return None

    def _check_shapes(
        self,
        y: Float[Array, "T M"],
        mask: Bool[Array, "T M"] | None,
    ) -> None:
        """Reject ``y`` / ``mask`` that do not match the base event shape.

        ``y - mean_grid`` and the filter's own broadcasting would other-
        wise accept a ``(T, 1)`` series against an ``(T, M)`` base,
        silently replicating the one observed channel across all ``M``
        and returning a finite log-likelihood for data the caller never
        supplied. Shapes are static under ``jit``, so this costs nothing
        at trace time.
        """
        expected = (self.n_steps, self.n_channels)
        if y.shape != expected:
            raise ValueError(
                f"y has shape {y.shape}, but the base event shape is "
                f"{expected}. Observations must match the base exactly — "
                "a mismatched channel or time axis would broadcast rather "
                "than raise."
            )
        if mask is not None and mask.shape != expected:
            raise ValueError(
                f"mask has shape {mask.shape}, but the base event shape "
                f"is {expected}. The mask marks observed entries of y, so "
                "it must have the same shape."
            )

    def _require_elementwise_warp(self, flows: ModuleType) -> None:
        """Raise unless ``gauss_flows`` classifies the warp as elementwise.

        Asked through the public surface rather than a private helper:
        `gauss_flows.normalizing_kalman_filter` documents that it
        refuses a channel-mixing warp over a **conditional** (mask-
        consuming) base, because masking and a non-diagonal Jacobian do
        not commute. Building that form is construction-only and cheap,
        so it doubles as the classifier — and it stays correct if
        ``gauss_flows`` refines what counts as elementwise. Shapes were
        already validated in ``__init__``, so a ``ValueError`` out of
        this constructor is that rejection.
        """
        try:
            self._nkf(flows, masked=True)
        except ValueError as exc:
            raise ValueError(
                "predict requires an elementwise warp: it pushes the "
                "per-channel marginal moments through the warp with a "
                "scalar Gauss-Hermite rule, which is only the right "
                "integral when each output channel depends on its own "
                "input channel alone. This warp mixes channels (or "
                "cannot be shown not to), so those moments would be "
                "silently wrong — log_marginal is unaffected and stays "
                "exact. Put cross-channel structure in H and R instead."
            ) from exc

    def _nkf(self, flows: ModuleType, *, masked: bool):
        """Build the ``gauss_flows`` NKF density for the current warp.

        Unmasked: wrap the base directly. Masked: rebuild the base's
        parameters into a `gaussx.LGSSMFactory` so the mask arrives as
        a flowjax ``condition`` — that is what routes the log-det through
        ``gauss_flows``' masked change-of-variables (observed channels
        only) and triggers its rejection of channel-mixing warps.
        """
        if not masked:
            return flows.normalizing_kalman_filter(
                flows.NumpyroBase(dist=self.base), self.warp
            )
        factory = gaussx.LGSSMFactory(
            self.base.A,
            self.base.H,
            self.base.Q,
            self.base.R,
            self.base.m0,
            self.base.P0,
            self.n_steps,
        )
        conditional_base = flows.NumpyroBase(
            dist_factory=factory,
            event_shape=tuple(self.base.event_shape),
            cond_shape=tuple(self.base.event_shape),
        )
        return flows.normalizing_kalman_filter(conditional_base, self.warp)

    def log_marginal(
        self,
        y: Float[Array, "T M"],
        mask: Bool[Array, "T M"] | None = None,
    ) -> Float[Array, ""]:
        r"""Exact log marginal likelihood, warp included.

        Computes $\log p(y) = \log p_{\mathrm{LGSSM}}(G^{-1}(y - m)) +
        \sum_t \log|\det \partial G^{-1}/\partial y_t|$ — a Kalman
        forward pass on the inverse-warped observations plus the summed
        log-determinant. The log-det does not depend on the latent path,
        so the result is exact, not a bound.

        Args:
            y: Observations, shape ``(T, M)``. Masked entries are never
                read and may be ``NaN``.
            mask: Optional observation mask, shape ``(T, M)``; ``True``
                marks an observed entry. Overrides a
                `gaussx.MaskedLGSSM` base's own mask when both are
                present. Unobserved channels are marginalised exactly;
                with a warp this requires an elementwise warp (see
                the class docstring).

        Returns:
            Scalar $\log p(y_{\mathrm{obs}} \mid \theta)$.

        Raises:
            ValueError: If ``y`` or ``mask`` does not match the base
                event shape.
        """
        y = jnp.asarray(y)
        mask_eff = self._effective_mask(mask)
        self._check_shapes(y, mask_eff)
        residual = y - self._mean_grid(y.shape[0], y.dtype)
        if self.warp is None:
            return gaussx.kalman_filter(
                self.base.A,
                self.base.H,
                self.base.Q,
                self.base.R,
                residual,
                self.base.m0,
                self.base.P0,
                mask=mask_eff,
            ).log_likelihood
        flows = _require_flows()
        nkf = self._nkf(flows, masked=mask_eff is not None)
        if mask_eff is None:
            return nkf.log_prob(residual)
        return nkf.log_prob(residual, condition=mask_eff)

    def predict(
        self,
        y: Float[Array, "T M"],
        mask: Bool[Array, "T M"] | None = None,
        *,
        n_ahead: int = 0,
        order: int = 32,
    ) -> tuple[Float[Array, "Tp M"], Float[Array, "Tp M"]]:
        r"""Predictive moments in **observation** space.

        Two stages. RTS smoothing gives Gaussian moments of the warped
        observation $z_t = H x_t + r_t \sim N(\mu_t, \sigma_t^2)$
        (per channel, noise included) on the training grid, extended
        ``n_ahead`` steps by open-loop propagation through $(A, Q)$.
        Those moments are then pushed through the warp by Gauss-Hermite
        quadrature:

        $$
        \mathbb{E}[y_t] = \int G(z)\, N(z; \mu_t, \sigma_t^2)\, dz,
        \qquad
        \mathrm{Var}[y_t] = \int G(z)^2 N(z; \mu_t, \sigma_t^2)\, dz
          - \mathbb{E}[y_t]^2 .
        $$

        The returned mean is $\mathbb{E}[G(z)]$, **not**
        $G(\mathbb{E}[z])$ — the latter is the pushforward median for a
        monotone warp and is badly biased for a skewed one.

        The quadrature is per-channel over the *marginal* warped-space
        moments, so it is the right integral only when each output
        channel depends on its own input channel alone. **This method
        therefore requires an elementwise warp** and raises otherwise —
        a channel-mixing warp's outputs depend on the full joint
        Gaussian, cross-channel covariances included, and per-channel
        moments would be silently wrong rather than merely imprecise.
        `log_marginal` carries no such restriction: it stays exact for
        any unmasked warp.

        Gauss-Hermite converges spectrally only for analytic
        integrands. A piecewise rational-quadratic spline warp is not
        analytic at its knots, so its quadrature error plateaus around
        ``~3e-3`` and can get *worse* with increasing order — the
        default ``order=32`` is the sweet spot for splines, and raising
        the order is not a convergence diagnostic.

        Args:
            y: Observations, shape ``(T, M)``. Masked entries are never
                read and may be ``NaN``.
            mask: Optional observation mask, shape ``(T, M)``; ``True``
                marks an observed entry. Same semantics as in
                `log_marginal`.
            n_ahead: Number of forecast steps appended after the
                training grid.
            order: Gauss-Hermite order for the warped pushforward.
                Ignored when ``warp is None``.

        Returns:
            Tuple ``(mean, var)`` of observation-space marginal moments,
            each of shape ``(T + n_ahead, M)``. Variances are clamped at
            zero, so ``sqrt`` on them is always safe.

        Raises:
            ValueError: If ``y`` or ``mask`` does not match the base
                event shape, or if the warp is not elementwise.
        """
        y = jnp.asarray(y)
        n_steps, n_channels = self.n_steps, self.n_channels
        mask_eff = self._effective_mask(mask)
        self._check_shapes(y, mask_eff)
        mean_grid = self._mean_grid(n_steps + n_ahead, y.dtype)
        residual = y - mean_grid[:n_steps]

        if self.warp is None:
            z = residual
        else:
            flows = _require_flows()
            # The scalar per-channel quadrature below is only the right
            # integral for an elementwise warp, so predict requires one
            # even when log_marginal would not.
            self._require_elementwise_warp(flows)
            if mask_eff is not None:
                # Unobserved slots hold junk (often NaN); substitute an
                # in-support reference before the inverse warp. The
                # filter never reads those entries afterwards.
                reference = self.warp.transform(jnp.zeros(n_channels, dtype=y.dtype))
                residual = jnp.where(mask_eff, residual, reference)
            z = jax.vmap(self.warp.inverse)(residual)  # (T, M)

        state = gaussx.kalman_filter(
            self.base.A,
            self.base.H,
            self.base.Q,
            self.base.R,
            z,
            self.base.m0,
            self.base.P0,
            mask=mask_eff,
        )
        m_smooth, P_smooth = gaussx.rts_smoother(state, self.base.A, self.base.Q)

        if n_ahead > 0:
            A_dense = _dense(self.base.A)
            Q_dense = _dense(self.base.Q)

            def step(carry, _):
                m, P = carry
                m_next = A_dense @ m
                P_next = A_dense @ P @ A_dense.T + Q_dense
                return (m_next, P_next), (m_next, P_next)

            last = (state.filtered_means[-1], state.filtered_covs[-1])
            _, (m_ahead, P_ahead) = jax.lax.scan(step, last, None, length=n_ahead)
            m_smooth = jnp.concatenate([m_smooth, m_ahead], axis=0)
            P_smooth = jnp.concatenate([P_smooth, P_ahead], axis=0)

        H_dense = _dense(self.base.H)
        R_diag = jnp.diagonal(_dense(self.base.R))
        # z-space marginals: mean H m_t, variance diag(H P_t H^T + R).
        mz = m_smooth @ H_dense.T  # (T', M)
        # einx.dot is typed as possibly returning a tuple (multi-output
        # patterns); narrow back to a single array for the typechecker.
        # Clamped at zero: the quadratic form is PSD in exact arithmetic,
        # but a rounding-negative entry would become NaN under the sqrt
        # taken for the quadrature nodes below.
        vz = jnp.maximum(
            jnp.asarray(einx.dot("m n, t n k, m k -> t m", H_dense, P_smooth, H_dense))
            + R_diag,
            0.0,
        )

        if self.warp is None:
            return mz + mean_grid, vz

        nodes, weights = np.polynomial.hermite_e.hermegauss(order)
        nodes = jnp.asarray(nodes, dtype=y.dtype)
        weights = jnp.asarray(weights, dtype=y.dtype) / np.sqrt(2.0 * np.pi)
        # fs: (order, T', M) — per-node evaluation points in the warped space.
        fs = mz[None] + jnp.sqrt(vz)[None] * nodes[:, None, None]
        g = jax.vmap(jax.vmap(self.warp.transform))(fs)
        m1 = jnp.asarray(einx.dot("s, s t m -> t m", weights, g))
        m2 = jnp.asarray(einx.dot("s, s t m -> t m", weights, g**2))
        # E[G^2] - E[G]^2 cancels catastrophically for a concentrated
        # predictive: the two moments agree to working precision and the
        # difference can land just below zero, which callers would turn
        # into NaN on the first sqrt.
        return m1 + mean_grid, jnp.maximum(m2 - m1**2, 0.0)


def normalizing_kalman_factor(
    name: str,
    prior: NormalizingKalmanPrior,
    y: Float[Array, "T M"],
    mask: Bool[Array, "T M"] | None = None,
) -> None:
    """Register the NKF marginal log-likelihood with NumPyro.

    Computes the exact collapsed marginal via `NormalizingKalmanPrior.log_marginal`
    and adds it as ``numpyro.factor(name, ...)``. Mirrors
    `pyrox_gp.markov_gp_factor` — the latent state path is marginalised
    analytically, so the model only carries sample sites for the
    state-space (and warp) hyperparameters.

    Args:
        name: NumPyro factor site name.
        prior: The `NormalizingKalmanPrior`.
        y: Observations, shape ``(T, M)``.
        mask: Optional observation mask, shape ``(T, M)``.
    """
    numpyro.factor(name, prior.log_marginal(y, mask))
