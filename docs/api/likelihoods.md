# Likelihoods API

The `pyrox.likelihoods` module holds the protocols that every observation model and variational guide in the pyrox packages subclasses, so a package built on the core can take a pyrox likelihood without depending on pyrox-gp. It carries no numerics: expectations and per-site derivatives of a likelihood are computed by gaussx in the packages that need them. `pyrox_gp` re-exports the protocols, so `pyrox_gp.Likelihood is pyrox.likelihoods.Likelihood`.

## Protocols

::: pyrox.likelihoods.Likelihood

::: pyrox.likelihoods.Guide

## Observation models

`AbstractObservation` is the base of an observation model that declares its own hyperparameters and builds a gaussx likelihood holding the data; `latent_dim` (1 by default) is the number of latents per site. `pyrox_lgm` re-exports it, so `pyrox_lgm.AbstractObservation is pyrox.likelihoods.AbstractObservation`.

pyrox-gp's likelihoods are observation models too: each one's `build(y, theta, data)` returns the gaussx likelihood below, with the inputs in `data["X"]`, and `log_prob(f, y, X)` evaluates it. `pyrox_gp.GaussianLikelihood` subclasses `pyrox.likelihoods.GaussianLikelihood`, so the closed-form paths that dispatch on the core class take it.

| pyrox name | gaussx likelihood it builds |
|---|---|
| `pyrox_gp.GaussianLikelihood` | `gaussx.GaussianLikelihood` |
| `pyrox_gp.BernoulliLikelihood` | `gaussx.BernoulliLikelihood` |
| `pyrox_gp.PoissonLikelihood` | `gaussx.PoissonLikelihood` |
| `pyrox_gp.StudentTLikelihood` | `gaussx.StudentTLikelihood` |
| `pyrox_gp.SoftmaxLikelihood` | `gaussx.SoftmaxLikelihood` |
| `pyrox_gp.HeteroscedasticGaussianLikelihood` | `gaussx.HeteroscedasticGaussianLikelihood` |

::: pyrox.likelihoods.AbstractObservation

## Bring your own likelihood

`DistLikelihood` turns a NumPyro distribution into an observation model, for a likelihood gaussx has no class for (a zero-inflated Poisson, a Beta, a censored Gaussian). Pass `make_dist(eta_i, theta, data_i)`, which returns site $i$'s distribution of $y_i$, and the hyperparameters it reads as `theta`:

```python
import jax.numpy as jnp
import numpyro.distributions as dist
from pyrox.likelihoods import DistLikelihood

zip_lik = DistLikelihood(
    lambda eta, th, d: dist.ZeroInflatedPoisson(gate=th["pi"], rate=jnp.exp(eta)),
    theta={"pi": 0.2},
)
```

Its log density is the distribution's `log_prob`, site by site. The core imports no gaussx, so `build(y, theta, data)` does not return a gaussx likelihood as the other observation models do: it returns the per-site log density and data that `gaussx.CallableLikelihood` takes. pyrox-gp's inference (Laplace, Gauss-Newton, posterior linearisation, EP, the SVGP expected log likelihood, CVI) and pyrox-ssm's Markov strategies wrap that pair as `gaussx.CallableLikelihood(*zip_lik.build(y, {}, data))`, so they take its site derivatives and expectations from gaussx. The inputs reach `make_dist` as `data_i["X"]`.

`theta` holds arrays that are leaves of the module, so they receive gradients and can be NumPyro sample values: inside a model, `DistLikelihood(make_dist, theta={"pi": numpyro.sample("pi", dist.Beta(2.0, 4.0))})` passed to `pyrox_ssm.markov_laplace_factor` puts the gate under NUTS. `make_dist` itself is a static field, so a parameter it closes over never trains. `make_dist` must return one site's distribution for one site's inputs, and `log_prob(f, y, X)` calls it on every site at once, so it should broadcast over a leading site axis as NumPyro distributions do.

`pyrox_gp.DistLikelihood` is this class. The old one-argument link, `DistLikelihood(lambda f: dist.Bernoulli(logits=f))`, still works and ignores `theta` and the inputs; `DistLikelihood(dist_fn=...)` warns and points to `make_dist`.

::: pyrox.likelihoods.DistLikelihood

## Gaussian observations

::: pyrox.likelihoods.GaussianLikelihood

## Helpers

::: pyrox.likelihoods.likelihood_accepts_inputs
