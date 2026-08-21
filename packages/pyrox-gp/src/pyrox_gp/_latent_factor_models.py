"""Model surface for collapsed latent-factor regression.

Mirrors `pyrox_gp.OILMMGPPrior` / `pyrox_gp.ConditionedOILMMGP`, but the
mixing matrix is marginalized rather than held as a field — see
`pyrox_gp._latent_factor`.

End-to-end usage:

```python
from functools import partial

import jax, optax
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta
from numpyro.infer.initialization import init_to_median, init_to_sample
from pyrox_gp import RBF, LatentFactorGPPrior, lfr_model

Q = 6
kernels = tuple(RBF(pyrox_name=f"RBF_q{q}") for q in range(Q))
prior = LatentFactorGPPrior(kernels=kernels, X=X)

# Sample Z from its GP prior to break the rotational symmetry; a median
# (zero) init starts exactly on a saddle. Hyperparameters init
# deterministically.
# NumPyro treats a bare callable as a strategy *factory*, so a custom
# init function must be wrapped in functools.partial.
def init_latents_by_sampling(site):
    return init_to_sample(site) if site["name"] == "Z_T" else init_to_median(site)

guide = AutoDelta(lfr_model, init_loc_fn=partial(init_latents_by_sampling))

# Z is N x Q free parameters on a well-conditioned objective; kernel
# hyperparameters live on a log scale. Give the latents a 10x larger step.
from pyrox.inference import param_group_optimizer

optimizer = param_group_optimizer(
    {"latents": optax.adam(1e-2), "globals": optax.adam(1e-3)},
    lambda path, _: "latents" if "Z_T" in str(path) else "globals",
)
svi = SVI(lfr_model, guide, optimizer, loss=Trace_ELBO())
result = svi.run(jax.random.PRNGKey(0), 5000, X, Y, prior)

Z_map = result.params["Z_T_auto_loc"].T
noise_var = result.params["noise_auto_loc"] ** 2

# Condition and predict *under the fitted parameters*. The kernels resolve
# their hyperparameters when they are evaluated, so outside a substitution
# context they fall back to their initial values and the predictions come
# from different kernels than the fit.
import numpyro


def _predict():
    cond = prior.condition(Y, Z_map, noise_var)
    return cond.predict(X_test), cond.predict(X_test, include_noise=True)


# AutoDelta stores a prior'd kernel hyperparameter as `<site>_auto_loc`,
# not under its site name, so substituting result.params alone would miss
# it. guide.median maps those back onto the site names; merge the two so
# both prior'd and deterministic (`pyrox_param`) kernel parameters replay.
fitted = {**result.params, **guide.median(result.params)}
(mean, var), (mean_obs, var_obs) = numpyro.handlers.substitute(
    _predict, fitted
)()                                                  # each (T, P)
```

A single global learning rate (plain ``optax.adam``) also works, but is a
compromise: it either crawls on ``Z`` or destabilizes the kernel
hyperparameters. See
[`param_group_optimizer`][pyrox.inference.param_group_optimizer].

``Y`` is assumed centered — there is no mean function; center it
externally before fitting.
"""

from __future__ import annotations

from typing import cast

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from flowjax.bijections import AbstractBijection
from jaxtyping import Array, Float
from numpyro import distributions as dist

from pyrox_gp._context import _kernel_contexts
from pyrox_gp._latent_factor import (
    collapsed_lfr_log_prob,
    decoder_posterior,
    lfr_predictive_moments,
    warped_decoder_posterior,
    warped_lfr_log_prob,
)
from pyrox_gp._models import ConditionedGP, GPPrior
from pyrox_gp._multi_output import _validate_kernel_scopes_unique
from pyrox_gp._protocols import Kernel


class LatentFactorGPPrior(eqx.Module):
    """GP latent factors with an analytically marginalized linear decoder.

    Unlike `pyrox_gp.OILMMKernel`, the mixing matrix is not a field of
    this module. It is a random variable with a fixed $\\mathcal{N}(0,1)$
    prior, integrated out in
    [`collapsed_lfr_log_prob`][pyrox_gp.collapsed_lfr_log_prob] and recovered
    in closed form by
    [`decoder_posterior`][pyrox_gp.decoder_posterior]. There is therefore no
    orthogonality requirement and no ``Q <= P`` constraint, and the output
    dimension ``P`` is not fixed at construction — it is read from ``Y`` at
    `condition` time.

    Caveats to keep in mind:

    - **Only the span of ``Z`` is identified.** The $\\mathcal{N}(0, I)$
      prior on the decoder is rotation-invariant, so any orthogonal
      $Z \\to ZA$, $W \\to A^\\top W$ leaves the objective unchanged.
      Individual latent factors carry no physical meaning without an extra
      rotation criterion.
    - **Scale is fixed by the decoder prior.** Do not add a free kernel
      amplitude per latent on top — $Z \\to cZ$, $W \\to W/c$ would be
      degenerate. Keep kernel ``variance`` fixed at ``1.0`` for latent
      kernels, or document the degeneracy in your model.
    - **`predict_latents` variance understates uncertainty.** It treats the
      MAP ``Z`` as noiseless observations of the latent processes, so it
      captures input-space extrapolation but not uncertainty in the point
      estimate itself.
    - **``z_*`` and ``W`` are assumed independent** in `predict`. They are
      not — both depend on the training data through the fitted ``Z``.
      Reasonable when ``P`` is large; understates variance for small ``P``.
    - **This is a low-data model.** ``Z`` is ``N x Q`` free parameters, does
      not amortize, does not minibatch over ``N``, and carries ``O(Q N^3)``
      through the latent priors. The reference experiments top out at
      ``N = 800``.

    Warp-specific caveats (ignore when ``warp is None``):

    - **``Y`` must sit inside the warp's support.**
      ``gauss_flows.RQSplineMarginal`` is linear outside
      ``[-interval, interval]``; scale ``Y`` sensibly or the warp
      degenerates to affine.
    - **Identifiability gets worse.** A per-channel affine warp is
      degenerate with $\\sigma$ and with the columns of $W$. Identity
      init plus a distinct learning-rate group
      ([`param_group_optimizer`][pyrox.inference.param_group_optimizer])
      is the mitigation.
    - **The warp is fitted to the marginals of ``Y``, not the residuals.**
      A warp that Gaussianizes raw channel marginals is not necessarily
      the one that Gaussianizes the noise — a genuine modelling
      approximation, not a bug.
    - **Invertibility is mandatory.** The change of variables requires a
      bijection, unlike warped-*likelihood* GPs where a bound holds for
      any monotone link.

    Attributes:
        kernels: One `pyrox_gp.Kernel` per latent process; ``len`` is ``Q``.
            Distinct `pyrox.PyroxModule` kernels must carry distinct
            ``pyrox_name`` values or their NumPyro sites collide.
        X: Training inputs of shape ``(N, D)``.
        warp: Optional bijection with event shape ``(P,)`` — one marginal
            transform per output channel. The *warped* observations
            $G^{-1}(Y)$ are modelled as the linear factor model, which
            handles skewed / heavy-tailed / positive channels without
            breaking the analytic decoder marginalization (the log-det
            Jacobian is free of $Z$, $W$, and $\\sigma$). ``None`` keeps
            the plain Gaussian factor model. Prefer
            ``gauss_flows.RQSplineMarginal`` — closed-form in both
            directions and the exact identity at initialization, so a
            warped fit starts precisely at the unwarped fit. Conditional
            (input-dependent) warps are rejected.
        latent_noise: Model nugget added to each latent GP covariance. A
            modelling choice that controls how tightly the latent GP
            interpolates the MAP factors — distinct from ``jitter``.
        jitter: Numerical diagonal regularization for Cholesky stability.
    """

    kernels: tuple[Kernel, ...]
    X: Float[Array, "N D"]
    warp: AbstractBijection | None = None
    latent_noise: float = 1e-3
    jitter: float = 1e-6

    def __check_init__(self) -> None:
        if not self.kernels:
            raise ValueError(
                "kernels must contain at least one latent kernel; an empty "
                "tuple gives Q = 0 and fails inside latent_cholesky."
            )
        _validate_kernel_scopes_unique(self.kernels)
        if self.warp is not None and self.warp.cond_shape is not None:
            raise ValueError(
                "Conditional warps are not supported: the log-det Jacobian "
                "would depend on the inputs, and the per-channel marginal "
                "interpretation no longer holds."
            )

    @property
    def num_latents(self) -> int:
        """Number of latent scalar GPs ``Q``."""
        return len(self.kernels)

    def latent_priors(self) -> tuple[GPPrior, ...]:
        """Return one scalar `pyrox_gp.GPPrior` per latent process."""
        return tuple(
            GPPrior(kernel=k, X=self.X, jitter=self.jitter) for k in self.kernels
        )

    def latent_cholesky(self) -> Float[Array, "Q N N"]:
        """Batched Cholesky factors of the ``Q`` latent GP covariances."""
        n = self.X.shape[0]
        eye = jnp.eye(n, dtype=self.X.dtype)
        with _kernel_contexts(self.kernels):
            covs = jnp.stack(
                [
                    k(self.X, self.X) + (self.latent_noise + self.jitter) * eye
                    for k in self.kernels
                ]
            )
        return jnp.linalg.cholesky(covs)

    def condition(
        self,
        Y: Float[Array, "N P"],
        Z: Float[Array, "N Q"],
        noise_var: Float[Array, ""],
    ) -> ConditionedLatentFactorGP:
        """Recover the decoder posterior for MAP latents ``Z``.

        Returns a `pyrox_gp.ConditionedLatentFactorGP` holding the
        closed-form matrix-normal decoder posterior alongside the latents,
        with each latent GP conditioned once so repeated `predict` calls
        do not repeat the ``O(Q N^3)`` training solve.

        !!! warning "Call this under the same context you fitted in"
            The kernels resolve their hyperparameters when this method
            evaluates them. Outside a NumPyro substitution context they
            resolve to their *initial* values, so predictions would come
            from different kernels than the fit. Wrap the call in
            ``numpyro.handlers.substitute(fn, result.params)`` — see the
            module docstring for the end-to-end pattern.
        """
        if Y.shape[0] != self.X.shape[0]:
            raise ValueError(
                f"Y must have {self.X.shape[0]} rows to match X; got {Y.shape[0]}."
            )
        if Z.shape != (self.X.shape[0], self.num_latents):
            raise ValueError(
                f"Z must have shape {(self.X.shape[0], self.num_latents)}; "
                f"got {Z.shape}."
            )
        if self.warp is None:
            mu_W, Sigma_W = decoder_posterior(Y, Z, noise_var)
        else:
            mu_W, Sigma_W = warped_decoder_posterior(Y, Z, noise_var, self.warp)
        # Condition each latent GP once, here, rather than on every predict
        # call: the training solve is O(N^3) per factor and does not depend
        # on the test inputs. Doing it here also captures the kernel
        # hyperparameters under whatever context the caller conditions in.
        with _kernel_contexts(self.kernels):
            latents = tuple(
                prior.condition(Z[:, q], noise_var=jnp.asarray(self.latent_noise))
                for q, prior in enumerate(self.latent_priors())
            )
        return ConditionedLatentFactorGP(
            prior=self,
            Z=Z,
            mu_W=mu_W,
            Sigma_W=Sigma_W,
            noise_var=noise_var,
            latents=latents,
        )


class ConditionedLatentFactorGP(eqx.Module):
    """Latent-factor posterior — MAP latents plus the decoder posterior.

    Attributes:
        prior: The `pyrox_gp.LatentFactorGPPrior` this was conditioned from.
        Z: MAP latent factor values at the training inputs, ``(N, Q)``.
        mu_W: Decoder posterior mean, ``(Q, P)``.
        Sigma_W: Decoder posterior row covariance, ``(Q, Q)``.
        noise_var: Scalar isotropic observation noise variance.
        latents: One conditioned scalar GP per latent process, built once
            by `LatentFactorGPPrior.condition` so the ``O(Q N^3)``
            training solve is not repeated on every prediction.
    """

    prior: LatentFactorGPPrior
    Z: Float[Array, "N Q"]
    mu_W: Float[Array, "Q P"]
    Sigma_W: Float[Array, "Q Q"]
    noise_var: Float[Array, ""]
    latents: tuple[ConditionedGP, ...]

    def predict_latents(
        self, X_new: Float[Array, "T D"]
    ) -> tuple[Float[Array, "T Q"], Float[Array, "T Q"]]:
        """Per-latent GP conditional means and marginal variances at ``X_new``.

        The MAP latents are treated as observations of the latent processes
        with ``latent_noise`` as their noise, so the returned variance
        reflects input-space extrapolation only — not uncertainty in the
        point estimate ``Z`` itself.
        """
        means, variances = [], []
        with _kernel_contexts(self.prior.kernels):
            for cond in self.latents:
                m, v = cond.predict(X_new)
                means.append(m)
                variances.append(v)
        return jnp.stack(means, -1), jnp.stack(variances, -1)

    def predict(
        self,
        X_new: Float[Array, "T D"],
        *,
        include_noise: bool = False,
        quad_order: int = 32,
    ) -> tuple[Float[Array, "T P"], Float[Array, "T P"]]:
        """Posterior predictive mean and variance over all ``P`` outputs.

        Composes the latent GP conditional with the decoder posterior via
        [`lfr_predictive_moments`][pyrox_gp.lfr_predictive_moments]. The
        variance decomposes into a decoder term, a latent term, and an
        interaction term; the first and third are output-independent.

        With a warp, the warped-space moments are pushed through $G$
        (``transform``) by Gauss-Hermite quadrature. The returned mean is
        $\\mathbb{E}[G(f)]$, **not** $G(\\mathbb{E}[f])$ — the latter is the
        pushforward median for a monotone warp and is badly biased for a
        skewed one.

        !!! warning "The warped predictive is an approximation"
            $f = z_*^\\top W$ is a *product* of two independent Gaussians,
            which is not itself Gaussian;
            [`lfr_predictive_moments`][pyrox_gp.lfr_predictive_moments]
            gives its exact first two moments, and the quadrature below
            builds Gaussian nodes from them. The observation-space moments
            are therefore moment-matched, not exact, and the error grows
            with the product of the latent and decoder variances relative
            to the mean. The unwarped path (``warp=None``) is unaffected —
            it returns the exact moments directly.

        Args:
            X_new: Test inputs, shape ``(T, D)``.
            include_noise: Add the observation noise variance. With a
                warp, the noise lives in the warped space, so it is added
                before the pushforward.
            quad_order: Gauss-Hermite order for the warped pushforward.
                Ignored when ``warp is None``. A moderate order is fine —
                the quadrature appears only here, never in training.
        """
        z_mean, z_var = self.predict_latents(X_new)
        mean, var = lfr_predictive_moments(
            z_mean,
            z_var,
            self.mu_W,
            self.Sigma_W,
            self.noise_var if include_noise else None,
        )
        warp = self.prior.warp
        if warp is None:
            return mean, var
        nodes, weights = np.polynomial.hermite_e.hermegauss(quad_order)
        nodes = jnp.asarray(nodes)
        weights = jnp.asarray(weights) / np.sqrt(2.0 * np.pi)
        # fs: (order, T, P) — per-node evaluation points in the warped space.
        fs = mean[None] + jnp.sqrt(var)[None] * nodes[:, None, None]
        g = jax.vmap(jax.vmap(warp.transform))(fs)
        m1 = einx.dot("s, s t p -> t p", weights, g)
        # Accumulate the *centered* second moment. E[G^2] - E[G]^2 cancels
        # catastrophically when the transformed values carry a large offset
        # relative to their spread (a mean of 1e4 with variance 1 loses the
        # variance entirely in float32), and can even come out negative.
        centered = g - m1[None]
        var = einx.dot("s, s t p -> t p", weights, centered**2)
        return m1, var


def lfr_factor(
    Y: Float[Array, "N P"],
    Z: Float[Array, "N Q"],
    noise_var: Float[Array, ""],
    *,
    warp: AbstractBijection | None = None,
    beta: float | None = None,
    name: str = "collapsed_lfr",
) -> None:
    """Register the collapsed latent-factor log-likelihood with NumPyro.

    The likelihood term grows as $O(NP)$ while the GP prior on $Z$ grows
    as $O(NQ)$, so for $P \\gg Q$ an untempered MAP drives $Z$ to
    interpolate noise. The likelihood is therefore scaled by an inverse
    temperature ``beta``; priors are left at unit weight.

    Args:
        Y: Observations, ``(N, P)``.
        Z: Latent factor values, ``(N, Q)``.
        noise_var: Scalar isotropic observation noise variance (in the
            warped space when ``warp`` is given).
        warp: Optional bijection with event shape ``(P,)``; when given,
            registers [`warped_lfr_log_prob`][pyrox_gp.warped_lfr_log_prob]
            instead of the plain collapsed likelihood.
        beta: Inverse temperature on the likelihood. ``None`` selects
            ``Q / P``, which keeps the likelihood and the latent GP prior
            balanced as the output dimension grows. Tune by held-out
            likelihood when it matters.
        name: NumPyro factor site name.
    """
    if beta is None:
        beta = Z.shape[1] / Y.shape[1]
    if warp is None:
        log_prob = collapsed_lfr_log_prob(Y, Z, noise_var)
    else:
        log_prob = warped_lfr_log_prob(Y, Z, noise_var, warp)
    with numpyro.handlers.scale(scale=beta):
        numpyro.factor(name, log_prob)


def lfr_model(
    X: Float[Array, "N D"],
    Y: Float[Array, "N P"],
    prior: LatentFactorGPPrior,
    *,
    beta: float | None = None,
    noise_prior_scale: float = 0.5,
) -> None:
    """MAP-over-latents collapsed latent-factor regression model.

    Pair with ``numpyro.infer.autoguide.AutoDelta`` — the latents are a
    point estimate, not a marginalized quantity. Latents are stored
    transposed, as ``(Q, N)``, so the ``Q`` independent GP priors form a
    batch dimension of one batched multivariate normal.

    Args:
        X: Training inputs, ``(N, D)``. Must match ``prior.X`` — the latent
            covariance is built from the prior, so ``X`` contributes only
            shape and dtype and a mismatch is rejected rather than fitted
            against the wrong locations.
        Y: Centered observations, ``(N, P)``.
        prior: The `pyrox_gp.LatentFactorGPPrior` to fit.
        beta: Inverse temperature forwarded to
            [`lfr_factor`][pyrox_gp.lfr_factor]; ``None`` means ``Q / P``.
        noise_prior_scale: Scale of the half-normal prior on the noise
            standard deviation.

    With a warp on the prior, the warp's array leaves are registered as a
    single pytree-valued ``numpyro.param`` site (``"warp_params"``) so all
    four blocks — latents, noise, kernel hyperparameters, and the warp —
    fit jointly under SVI. After fitting, rebuild the warp for prediction
    with ``eqx.combine(result.params["warp_params"],
    eqx.partition(prior.warp, eqx.is_inexact_array)[1])`` and pass it via
    ``eqx.tree_at`` (or reconstruct the prior) before calling `condition`.
    """
    if X.shape != prior.X.shape:
        raise ValueError(
            f"X must be the prior's training inputs, shape {prior.X.shape}; "
            f"got {X.shape}. The latent covariance is built from prior.X, so "
            "a different X would silently pair Y with the wrong locations."
        )
    n = X.shape[0]
    q = prior.num_latents
    Z_T = jnp.asarray(
        numpyro.sample(
            "Z_T",
            dist.MultivariateNormal(
                loc=jnp.zeros((q, n), dtype=X.dtype),
                scale_tril=prior.latent_cholesky(),
            ).to_event(1),
        )
    )
    noise = jnp.asarray(numpyro.sample("noise", dist.HalfNormal(noise_prior_scale)))
    if prior.warp is None:
        warp = None
    else:
        # Register the warp's array leaves as one pytree-valued numpyro.param
        # (the same mechanism numpyro.contrib.module uses for flax/haiku
        # params), so SVI fits the warp jointly with Z, noise, and the
        # kernel hyperparameters.
        params, static = eqx.partition(prior.warp, eqx.is_inexact_array)
        params = numpyro.param("warp_params", params)
        warp = cast(AbstractBijection, eqx.combine(params, static))
    lfr_factor(Y, Z_T.T, noise**2, warp=warp, beta=beta)


def latent_total_correlation(Z: Float[Array, "N Q"]) -> Float[Array, ""]:
    """Total correlation of the fitted latent factors.

    The model places an independent GP prior on each of the $Q$ latent
    factors. A large value here means that assumption is violated in the
    fit — usually because $Q$ is larger than the data supports, or because
    the optimizer landed in a badly rotated gauge (only the span of $Z$ is
    identified; see `pyrox_gp.LatentFactorGPPrior`).

    Diagnostic only — this does not enter the objective. Requires the
    ``flows`` optional dependency (``pip install 'pyrox-gp[flows]'``).

    Args:
        Z: MAP latent factor values, shape ``(N, Q)``.

    Returns:
        Scalar total correlation, zero for exactly independent factors.
    """
    from gauss_flows import (  # ty: ignore[unresolved-import]
        gaussian_total_correlation,
    )

    # jnp.cov collapses to a scalar for a single factor; the total-
    # correlation routine needs a (Q, Q) matrix (and returns zero there).
    return gaussian_total_correlation(jnp.atleast_2d(jnp.cov(Z.T)))
