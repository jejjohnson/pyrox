r"""Per-site likelihood numerics for the Markov track, on gaussx.

The Markov modules (`pyrox_ssm._markov`, `_sparse_markov`,
`_inference_nongauss_markov`, `flows._markov_flow`) import their per-site
numerics from here, so pyrox-ssm depends only on the pyrox core and gaussx.

`as_gaussx_likelihood` views a likelihood as a `gaussx.CallableLikelihood`
(one log density per site),

$$
\ell(f_i, d_i) = \log p(y_i \mid f_i, X_i), \qquad
\log p(y \mid f) = \sum_i \ell(f_i, d_i),
$$

so the per-site derivatives and expectations come from gaussx
(``site_derivatives``, `gaussx.expected_log_likelihood`). An observation
model (`pyrox.likelihoods.AbstractObservation`, such as pyrox-gp's
`BernoulliLikelihood`) gives $\ell$ through the one-site gaussx likelihood it
builds; any other `pyrox.likelihoods.Likelihood` through its ``log_prob``
on one point.
`_check_scalar_latent` and `_require_gauss_hermite` are copies of the checks
in `pyrox_gp._inference_nongauss`, and `_pushforward_moments` a copy of the
one in `pyrox_gp._warped` (no gaussx function computes it yet).
"""

from __future__ import annotations

from collections.abc import Callable

import einx
import equinox as eqx
import gaussx
import jax
import jax.numpy as jnp
import lineax as lx
from gaussx import AbstractIntegrator, CallableLikelihood, GaussHermiteIntegrator
from jaxtyping import Array, Bool, Float, PyTree
from pyrox.likelihoods import (
    AbstractObservation,
    Likelihood,
    _observation_site_log_prob,
    likelihood_accepts_inputs,
)


def _check_scalar_latent(lik: Likelihood) -> None:
    latent_dim = getattr(lik, "latent_dim", 1)
    if latent_dim != 1:
        msg = (
            f"{type(lik).__name__} declares latent_dim={latent_dim}; the "
            "scalar-latent advanced inference strategies (Laplace, GN, EP, "
            "PL, QN) do not yet support multi-latent likelihoods. Use SVGP "
            "or wait for the multi-latent inference follow-up."
        )
        raise ValueError(msg)


def _require_gauss_hermite(integrator: AbstractIntegrator) -> GaussHermiteIntegrator:
    """Reject a rule EP cannot use: tilted moments are Gauss-Hermite only.

    Args:
        integrator: The rule passed to an EP strategy.

    Returns:
        The same rule, narrowed.

    Raises:
        TypeError: If ``integrator`` is not a `gaussx.GaussHermiteIntegrator`.
    """
    if not isinstance(integrator, GaussHermiteIntegrator):
        raise TypeError(
            "Expectation propagation takes its tilted moments from "
            "gaussx.ep_tilted_moments, a Gauss-Hermite rule, so it needs a "
            f"gaussx.GaussHermiteIntegrator; got {type(integrator).__name__}."
        )
    return integrator


class _SiteLogDensity(eqx.Module):
    r"""One site's $\ell(f_i, d_i)$ from a pyrox likelihood summed over points.

    An `equinox.Module` rather than a closure, so the likelihood's parameters
    stay leaves of the `gaussx.CallableLikelihood` pytree.

    Attributes:
        likelihood: The pyrox likelihood.
        with_inputs: Whether each site's data carries its inputs $X_i$.
    """

    likelihood: Likelihood
    with_inputs: bool = eqx.field(static=True)

    def __call__(self, f_i: Float[Array, ""], site: PyTree) -> Float[Array, ""]:
        if self.with_inputs:
            y_i, x_i = site
            # ℓ(fᵢ, (yᵢ, Xᵢ)) = log p(yᵢ ∣ fᵢ, Xᵢ)
            return self.likelihood.log_prob(f_i[None], y_i[None], x_i)
        # ℓ(fᵢ, yᵢ) = log p(yᵢ ∣ fᵢ)
        return self.likelihood.log_prob(f_i[None], site[None])


class _ObservationSiteLogDensity(eqx.Module):
    r"""One site's $\ell(f_i, d_i)$ from the gaussx likelihood an observation builds.

    Site $i$'s data is $(y_i, \mathrm{data}_i)$, where $\mathrm{data}_i$ is
    its slice of the ``data`` mapping (``{"X": X}`` or empty), evaluated by
    `pyrox.likelihoods._observation_site_log_prob`, the adapter pyrox-gp's
    inference shares.

    Attributes:
        observation: The observation model.
    """

    observation: AbstractObservation

    def __call__(self, f_i: Float[Array, ""], site: PyTree) -> Float[Array, ""]:
        y_i, data_i = site
        # ℓ(fᵢ, (yᵢ, dᵢ)) = build(yᵢ, ∅, dᵢ).log_prob(fᵢ)
        return _observation_site_log_prob(self.observation, f_i, y_i, data_i)


class _MaskedSiteLogDensity(eqx.Module):
    r"""A site log density that is zero at masked sites, $o_i\,\ell(f_i, d_i)$.

    Each site's data is ``(d_i, o_i)``; a masked site ($o_i$ ``False``)
    returns zero, so its gradient and curvature in $f_i$ are zero too.

    Attributes:
        site_log_density: The unmasked per-site log density.
    """

    site_log_density: _SiteLogDensity | _ObservationSiteLogDensity

    def __call__(self, f_i: Float[Array, ""], site: PyTree) -> Float[Array, ""]:
        data, observed = site
        # oᵢ ℓ(fᵢ, dᵢ)
        value = self.site_log_density(f_i, data)
        return jnp.where(observed, value, jnp.zeros((), dtype=value.dtype))


def as_gaussx_likelihood(
    lik: Likelihood,
    y: Float[Array, " N"],
    X: Float[Array, "N D"] | None = None,
    *,
    mask: Bool[Array, " N"] | None = None,
) -> CallableLikelihood:
    r"""Per-site view of a pyrox likelihood, as a `gaussx.CallableLikelihood`.

    An observation model (`pyrox.likelihoods.AbstractObservation`) gives
    site $i$'s term through the gaussx likelihood it builds on that site
    alone, with the inputs in ``data["X"]``,

    $$
    \ell(f_i, (y_i, X_i)) = \texttt{lik.build}(y_{i:i+1}, \varnothing,
    \{X: X_{i:i+1}\}).\texttt{log\_prob}(f_{i:i+1}).
    $$

    Any other `pyrox.likelihoods.Likelihood` gives it through ``log_prob``,
    which is summed over points, on that point alone,

    $$
    \ell(f_i, (y_i, X_i)) = \texttt{lik.log\_prob}(f_{i:i+1}, y_{i:i+1}, X_i);
    $$

    ``X`` is passed only to a likelihood whose ``log_prob`` takes it
    (`pyrox.likelihoods.likelihood_accepts_inputs`); otherwise each site's
    data is $y_i$ alone. With a mask $o$, site $i$'s term is
    $o_i\,\ell(f_i, d_i)$, so a masked site has $g_i = h_i = 0$.

    Args:
        lik: A scalar-latent observation model or pyrox likelihood.
        y: The observations, one per site. Masked entries may be ``NaN``:
            they are replaced by the first observed value before ``lik``
            sees them (zero can lie outside a likelihood's support), so no
            ``NaN`` reaches a derivative.
        X: The inputs, one row per site, for an input-dependent likelihood.
        mask: Optional boolean mask, ``True`` where a site is observed.

    Returns:
        The likelihood as a sum of per-site log densities, with
        ``site_derivatives`` and `gaussx.expected_log_likelihood`.

    Raises:
        ValueError: If ``lik`` is multi-latent (``latent_dim > 1``): its
            Hessian in $f$ is not diagonal across one site's latents.

    Examples:
        A Gaussian likelihood has $g_i = (y_i - f_i) / \sigma^2$ and
        $h_i = -1 / \sigma^2$ at every site:

        >>> import jax.numpy as jnp
        >>> from pyrox.likelihoods import GaussianLikelihood
        >>> y = jnp.array([1.0, -1.0])
        >>> site_lik = as_gaussx_likelihood(GaussianLikelihood(noise_var=0.5), y)
        >>> g, h = site_lik.site_derivatives(jnp.zeros(2))
        >>> jnp.round(g, 4).tolist(), jnp.round(h, 4).tolist()
        ([2.0, -2.0], [-2.0, -2.0])
    """
    _check_scalar_latent(lik)
    site_log_density: _SiteLogDensity | _ObservationSiteLogDensity
    if isinstance(lik, AbstractObservation):
        inputs = {} if X is None else {"X": X}
        site_log_density = _ObservationSiteLogDensity(lik)

        def site_data(y: Float[Array, " N"]) -> PyTree:
            return (y, inputs)

    else:
        with_inputs = X is not None and likelihood_accepts_inputs(lik)
        site_log_density = _SiteLogDensity(lik, with_inputs=with_inputs)

        def site_data(y: Float[Array, " N"]) -> PyTree:
            return (y, X) if with_inputs else y

    if mask is None:
        return CallableLikelihood(site_log_density, site_data(y))
    observed = jnp.asarray(mask).astype(bool)
    # The first observed value is in the likelihood's support; a masked entry
    # only needs ℓ to be finite there, since oᵢ = 0 discards it.
    y = jnp.where(observed, y, y[jnp.argmax(observed)])
    return CallableLikelihood(
        _MaskedSiteLogDensity(site_log_density), (site_data(y), observed)
    )


def _cavity_site_derivatives(
    site_lik: CallableLikelihood,
    mean: Float[Array, " N"],
    var: Float[Array, " N"],
    integrator: AbstractIntegrator,
) -> tuple[Float[Array, " N"], Float[Array, " N"]]:
    r"""Expected per-site gradient and curvature under independent Gaussians.

    With $\mathcal{E}(\mu) = \sum_i \mathbb{E}[\ell(f_i, d_i)]$ for
    $f_i \sim N(\mu_i, v_i)$, the Bonnet and Price identities give

    $$
    \bar g_i = \mathbb{E}[\partial_f \ell(f_i, d_i)]
             = \partial_{\mu_i} \mathcal{E}, \qquad
    \bar h_i = \mathbb{E}[\partial_f^2 \ell(f_i, d_i)]
             = \partial_{\mu_i}^2 \mathcal{E}.
    $$

    $\mathcal{E}$ is `gaussx.expected_log_likelihood` on the diagonal state.
    A rule whose nodes are $\mu + \sqrt{v}\,\xi_k$ (Gauss-Hermite, unscented,
    a fixed-key Monte Carlo) makes the two identities exact for the rule, not
    only in the limit: differentiating the weighted sum in $\mu$
    differentiates $\ell$ at each node. $\mathcal{E}$ is a sum of one term per
    site, so its Hessian is diagonal and one forward pass of the gradient
    along $\mathbf{1}$ reads it off.

    Args:
        site_lik: The per-site likelihood (`as_gaussx_likelihood`).
        mean: The site means $\mu$.
        var: The site variances $v$.
        integrator: The rule for each one-dimensional site expectation.

    Returns:
        Tuple ``(g_bar, h_bar)``, one entry per site.
    """

    def expected(site_mean: Float[Array, " N"]) -> Float[Array, ""]:
        # q(f) = 𝒩(μ, diag(v))
        state = gaussx.GaussianState(site_mean, lx.DiagonalLinearOperator(var))
        # 𝓔(μ) = ∑ᵢ 𝔼[ℓ(fᵢ, dᵢ)]
        return gaussx.expected_log_likelihood(site_lik, state, integrator)

    # ḡ = ∇_μ 𝓔, h̄ = diag ∇²_μ 𝓔 (diagonal: 𝓔 is a sum over sites)
    return jax.jvp(jax.grad(expected), (mean,), (jnp.ones_like(mean),))


def _pushforward_moments(
    transform: Callable[[Float[Array, " *batch"]], Float[Array, " *batch"]],
    mean: Float[Array, " *batch"],
    var: Float[Array, " *batch"],
    *,
    order: int,
) -> tuple[Float[Array, " *batch"], Float[Array, " *batch"]]:
    r"""Mean and centred variance of $G(f)$, with $f \sim N(\mu, \sigma^2)$ elementwise.

    With probabilists' Gauss-Hermite nodes $\xi_k$ and normalised weights
    $\tilde w_k = w_k / \sqrt{2\pi}$,

    $$
    m_1 = \sum_k \tilde w_k\, G(\mu + \sigma \xi_k), \qquad
    v = \sum_k \tilde w_k\, (G(\mu + \sigma \xi_k) - m_1)^2 .
    $$

    A copy of `pyrox_gp._warped._pushforward_moments`; a candidate for gaussx.

    Args:
        transform: The warp $G$, applied to one batch-shaped array per node.
        mean: The means $\mu$, in any batch shape.
        var: The variances $\sigma^2$, in the same shape.
        order: Number of nodes $Q$, in ``[2, 256]``.

    Returns:
        Tuple ``(m1, v)`` in the shape and dtype of ``mean``.

    Raises:
        ValueError: If ``order`` is outside ``[2, 256]``.
    """
    if not 2 <= order <= 256:
        # order == 1 places a single node at the mean, so the centred second
        # moment is zero and every bit of latent uncertainty would be silently
        # discarded. The rule overflows to NaN by order 512.
        raise ValueError(
            f"order must be in [2, 256]; got {order}. A single quadrature "
            "node cannot represent any spread."
        )
    points, raw_weights = gaussx.gauss_hermite_points(order, 1, dtype=mean.dtype)
    # w̃ₖ = wₖ / √(2π), so ∑ₖ w̃ₖ = 1
    weights = raw_weights / jnp.sqrt(jnp.asarray(2.0 * jnp.pi, dtype=mean.dtype))
    # σ = √v_f
    std = jnp.sqrt(var)
    # σ ξₖ
    spread = einx.multiply("q, ... -> q ...", points[:, 0], std)
    # fₖ = μ + σ ξₖ
    latent = einx.add("q ..., ... -> q ...", spread, mean)
    # gₖ = G(fₖ)
    warped = jax.vmap(transform)(latent)
    # m₁ = ∑ₖ w̃ₖ gₖ
    first_moment = jnp.asarray(einx.dot("q, q ... -> ...", weights, warped))
    # gₖ − m₁. Centred, never E[G²] − m₁², which cancels catastrophically once
    # G's output carries a large offset relative to its spread: an Affine warp
    # centred near 1e4 loses the whole variance in float32, and can go negative.
    centred = einx.subtract("q ..., ... -> q ...", warped, first_moment)
    # v = ∑ₖ w̃ₖ (gₖ − m₁)²
    variance = jnp.asarray(einx.dot("q, q ... -> ...", weights, centred**2))
    return first_moment, variance
