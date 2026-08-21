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

import equinox as eqx
import jax.numpy as jnp
import numpyro
from flowjax.bijections import AbstractBijection
from jaxtyping import Array, Float
from numpyro import distributions as dist

from pyrox_gp._context import _kernel_contexts
from pyrox_gp._latent_factor import (
    collapsed_lfr_log_prob,
    decoder_posterior,
    lfr_predictive_moments,
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

    Attributes:
        kernels: One `pyrox_gp.Kernel` per latent process; ``len`` is ``Q``.
            Distinct `pyrox.PyroxModule` kernels must carry distinct
            ``pyrox_name`` values or their NumPyro sites collide.
        X: Training inputs of shape ``(N, D)``.
        warp: Reserved for the warped-observation extension; must be ``None``
            here.
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
        if self.warp is not None:
            raise NotImplementedError(
                "Warped observations are not implemented in this issue; see the "
                "warped latent-factor regression issue. Pass warp=None."
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
        mu_W, Sigma_W = decoder_posterior(Y, Z, noise_var)
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
    ) -> tuple[Float[Array, "T P"], Float[Array, "T P"]]:
        """Posterior predictive mean and variance over all ``P`` outputs.

        Composes the latent GP conditional with the decoder posterior via
        [`lfr_predictive_moments`][pyrox_gp.lfr_predictive_moments]. The
        variance decomposes into a decoder term, a latent term, and an
        interaction term; the first and third are output-independent.
        """
        z_mean, z_var = self.predict_latents(X_new)
        return lfr_predictive_moments(
            z_mean,
            z_var,
            self.mu_W,
            self.Sigma_W,
            self.noise_var if include_noise else None,
        )


def lfr_factor(
    Y: Float[Array, "N P"],
    Z: Float[Array, "N Q"],
    noise_var: Float[Array, ""],
    *,
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
        noise_var: Scalar isotropic observation noise variance.
        beta: Inverse temperature on the likelihood. ``None`` selects
            ``Q / P``, which keeps the likelihood and the latent GP prior
            balanced as the output dimension grows. Tune by held-out
            likelihood when it matters.
        name: NumPyro factor site name.
    """
    if beta is None:
        beta = Z.shape[1] / Y.shape[1]
    with numpyro.handlers.scale(scale=beta):
        numpyro.factor(name, collapsed_lfr_log_prob(Y, Z, noise_var))


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
    lfr_factor(Y, Z_T.T, noise**2, beta=beta)


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
